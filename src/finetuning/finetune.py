# code adapted from Alpaca Stanford

import copy
import logging
import json
import io
import torch
import numpy as np

import subprocess

from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence

import transformers
from transformers import Trainer
from torch.utils.data import Dataset
from peft import get_peft_model, LoraConfig, TaskType
from peft import PeftModel
import time

from tqdm import tqdm

import os

from utils import load_model, get_base_model_path

from src.folder_handler import FolderHandler


PROMPT_DICT = {
    "prompt_input": (
        "Below is an instruction that describes a task, paired with an input that provides further context. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:"
    ),
    "prompt_no_input": (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Response:"
    ),
}

decoding_layers = {
    "Qwen": "Qwen2DecoderLayer",
    "Llama": "LlamaDecoderLayer",
    "Olmo": "Olmo2DecoderLayer",
}

IGNORE_INDEX = -100


@dataclass
class FinetuningArguments:
    model_id: int = field(default=None, metadata={"help": "Model ID"})
    dataset_size: int = field(default=1, metadata={"help": "Number of samples to use from the dataset for training"})
    pii_rate: float = field(default=0.1, metadata={"help": "Rate of PII (Personally Identifiable Information) in the dataset, controls privacy level"})
    kg: str = field(default="no-kg", metadata={"help": "Knowledge graph configuration - 'no-kg' for no knowledge graph, or specify KG type"})
    ds_file: Optional[str] = field(default=None, metadata={"help": "Path to DeepSpeed configuration file for distributed training"})
    dataset_path: Optional[str] = field(
        default=None, metadata={"help": "Path to the training dataset file"}
    )
    model_name_or_path: Optional[str] = field(
        default=None, metadata={"help": "Path to the base model or HuggingFace model identifier"}
    )
    output_dir: str = field(default="ft_small", metadata={"help": "Directory to save the fine-tuned model and training outputs"})
    model_max_length: int = field(default=8192, metadata={"help": "Maximum sequence length for tokenization and model input"})
    lora: bool = field(default=False, metadata={"help": "Enable LoRA (Low-Rank Adaptation) for parameter-efficient fine-tuning (not recommended)"})
    ppl: bool = field(default=False, metadata={"help": "Compute perplexity on evaluation set instead of training"})
    n_epochs: int = field(default=1, metadata={"help": "Number of training epochs"})
    learning_rate: float = field(default=2e-5, metadata={"help": "Optimizer learning rate (paper default 2e-5)"})
    gradient_accumulation_steps: int = field(default=8, metadata={"help": "Gradient accumulation steps (paper default 8)"})
    save_total_limit: int = field(default=20, metadata={"help": "Maximum number of model checkpoints to keep during training"})

def _tokenize_fn(strings: Sequence[str], tokenizer: transformers.PreTrainedTokenizer) -> Dict:
    """Tokenize a list of strings."""
    tokenized_list = [
        tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        )
        for text in strings
    ]
    input_ids = labels = [tokenized.input_ids[0] for tokenized in tokenized_list]
    input_ids_lens = labels_lens = [
        tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item() for tokenized in tokenized_list
    ]    
    return dict(
        input_ids=input_ids,
        labels=labels,
        input_ids_lens=input_ids_lens,
        labels_lens=labels_lens,
    )


def preprocess(
    sources: Sequence[str],
    targets: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
) -> Dict:
    """Preprocess the data by tokenizing."""
    examples = [s + t for s, t in zip(sources, targets)]
    examples_tokenized, sources_tokenized = [
        _tokenize_fn(strings, tokenizer) for strings in (examples, sources)
    ]
    input_ids = examples_tokenized["input_ids"]
    labels = copy.deepcopy(input_ids)
    for label, source_len in zip(labels, sources_tokenized["input_ids_lens"]):
        label[:source_len] = IGNORE_INDEX
    return dict(input_ids=input_ids, labels=labels)


def _make_r_io_base(f, mode: str):
    if not isinstance(f, io.IOBase):
        f = open(f, mode=mode)
    return f


def jload(f, mode="r"):
    """Load a .json file into a dictionary."""
    f = _make_r_io_base(f, mode)
    jdict = json.load(f)
    f.close()
    return jdict


class SupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(
        self,
        data_path: str,
        tokenizer: transformers.PreTrainedTokenizer,
        cache_path: str = None,
        device=None,
        chunk_size: int = None,
    ):
        super(SupervisedDataset, self).__init__()

        if cache_path is None:
            cache_path = data_path.replace(
                ".json", f"{tokenizer.name_or_path.split('/')[-1]}_cache.pt"
            )

        # Try loading the cached dataset
        if os.path.exists(cache_path):  # False: # model dependent #os.path.exists(cache_path):
            logging.warning(f"Loading cached dataset from {cache_path}...")
            cached_data = torch.load(cache_path)
            self.input_ids = [x.to(device) for x in tqdm(cached_data["input_ids"])]
            self.labels = [x.to(device) for x in tqdm(cached_data["labels"])]
        else:
            logging.warning("Loading raw data and processing...")

            logging.warning("Loading data...")
            list_data_dict = jload(data_path)

            logging.warning("Formatting inputs...")
            prompt_input, prompt_no_input = (
                PROMPT_DICT["prompt_input"],
                PROMPT_DICT["prompt_no_input"],
            )
            sources = [
                (
                    prompt_input.format_map(example)
                    if example.get("input", "") != ""
                    else prompt_no_input.format_map(example)
                )
                for example in list_data_dict
            ]
            targets = [f"{example['output']}{tokenizer.eos_token}" for example in list_data_dict]

            logging.warning("Tokenizing inputs... This may take some time...")
            data_dict = preprocess(sources, targets, tokenizer)

            self.input_ids = data_dict["input_ids"]
            self.labels = data_dict["labels"]

            logging.warning(f"Saving processed dataset to {cache_path} for future use...")
            torch.save({"input_ids": self.input_ids, "labels": self.labels}, cache_path)

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        return dict(input_ids=self.input_ids[i], labels=self.labels[i])


@dataclass
class DataCollatorForSupervisedDataset(object):
    """
    Collate examples for supervised fine-tuning.

    This class is responsible for batching individual training examples together.
    It performs the following steps:
    1. Takes a sequence of instances (dicts with input_ids and labels)
    2. Extracts and pads the input_ids and labels to the same length within a batch
    3. Creates an attention mask to ignore padding tokens

    Args:
        tokenizer: The tokenizer used to get pad token ID for padding

    Returns:
        A dictionary containing:
        - input_ids: Padded input token IDs
        - labels: Padded label IDs (with IGNORE_INDEX for padding)
        - attention_mask: Binary mask of 1s for real tokens and 0s for padding
    """

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        # Extract input_ids and labels from each instance and combine into lists
        input_ids, labels = tuple(
            [instance[key] for instance in instances] for key in ("input_ids", "labels")
        )

        # Pad sequences to the same length
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(
            labels, batch_first=True, padding_value=IGNORE_INDEX
        )

        # Create attention mask (1 for real tokens, 0 for padding)
        attention_mask = input_ids.ne(self.tokenizer.pad_token_id)

        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=attention_mask,
        )


def get_path_dataset(args, split):
    data_path = args.dataset_path.replace("[SPLIT]", split)
    # Rewrite 'train'->split (to derive the val path from a train path) only in
    # the FILENAME, not in parent directories: an absolute path whose parent
    # contains 'train' (e.g. /scratch/training/...) would otherwise be corrupted.
    head, tail = os.path.split(data_path)
    data_path = os.path.join(head, tail.replace("train", split))
    data_path = (
        data_path
        .replace("[SIZE]", str(args.dataset_size))
        .replace("[PII_RATE]", str(args.pii_rate))
        .replace("[KG]", str(args.kg))
    )
    print("Reading dataset path", data_path)
    return data_path


def make_supervised_data_module(
    tokenizer: transformers.PreTrainedTokenizer, data_args, device
) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    dataset_class = SupervisedDataset
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)

    train_dataset = dataset_class(
        tokenizer=tokenizer, data_path=get_path_dataset(data_args, "train")
    )
    eval_dataset = dataset_class(tokenizer=tokenizer, data_path=get_path_dataset(data_args, "val"))

    return dict(train_dataset=train_dataset, eval_dataset=eval_dataset, data_collator=data_collator)


def finetune():
    start_time = time.time()
    print("Time start", start_time)
    parser = transformers.HfArgumentParser((FinetuningArguments,))

    (ft_args,) = parser.parse_args_into_dataclasses()
    ft_args.infer = False
    print(ft_args)

    if ft_args.model_id is not None:
        # Index mode: pull base model / dataset / output / epochs from index/models.csv.
        folder_handler = FolderHandler()
        model = folder_handler.query_model_unique(kwargs_filter={"model_id": ft_args.model_id}, property=None)
        print(model)
        ft_args.model_name_or_path = model['src_model_path']  # base model
        ft_args.output_dir = model['model_path']
        ft_args.n_epochs = model['n_epochs']
        ft_args.dataset_size = model['dataset_size']
        ft_args.pii_rate = model['pii_rate']
        ft_args.kg = "no-kg"
        ft_args.dataset_path = model['dataset_path']
    else:
        # Direct mode (no index, no config): everything comes from the CLI flags
        # --model_name_or_path / --dataset_path / --output_dir / --n_epochs.
        if not (ft_args.model_name_or_path and ft_args.dataset_path):
            raise ValueError(
                "Provide --model_id (index mode) OR --model_name_or_path and --dataset_path "
                "(direct mode).")

    model, tokenizer = load_model(ft_args)

    data_module = make_supervised_data_module(
        tokenizer=tokenizer, data_args=ft_args, device=model.device
    )

    dc_layer = None
    for key, item in decoding_layers.items():
        if key in ft_args.model_name_or_path:
            dc_layer = item

    if ft_args.lora and not ft_args.ppl:
        print("LORA mode")

        # https://github.com/huggingface/peft/issues/137
        model.enable_input_require_grads()
        lora_config = LoraConfig(
            r=16,  # 256
            lora_alpha=32,  # 512
            target_modules=[
                "q_proj",
                "v_proj",
                "o_proj",
                "k_proj",
            ],  # Using a regex to target all modules
            lora_dropout=0.05,
            bias="lora_only",  # Updated to match user's request
            task_type=TaskType.CAUSAL_LM,
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
    else:
        print("Full Finetuning")

    print("DC Layer", dc_layer)

    fdsp_config = {
        "fsdp_auto_wrap_policy": "TRANSFORMER_BASED_WRAP",
        "fsdp_transformer_layer_cls_to_wrap": dc_layer,
        "fsdp_state_dict_type": "SHARDED_STATE_DICT",
    }

    dp_config = None
    if ft_args.ds_file is not None:
        dp_config = {
            "deepspeed": ft_args.ds_file,
        }
    else:
        dp_config = {
            "fsdp_config": fdsp_config,
        }

    os.environ["WANDB_PROJECT"] = "LLM-MIMIC-Notes"  # name your W&B project

    tr_args = transformers.TrainingArguments(
        output_dir=ft_args.output_dir,
        bf16=True,
        num_train_epochs=ft_args.n_epochs,
        per_device_eval_batch_size=1,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=ft_args.gradient_accumulation_steps,
        eval_strategy="epoch",
        save_strategy="epoch",
        # save_strategy="steps",
        # save_steps=20,
        save_total_limit=ft_args.save_total_limit,
        learning_rate=ft_args.learning_rate,
        weight_decay=0,
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        logging_steps=1,
        report_to="wandb",
        gradient_checkpointing=True,
        prediction_loss_only=True,
        tf32=True,
        **dp_config,
    )
    trainer = Trainer(model=model, tokenizer=tokenizer, args=tr_args, **data_module)

    if not ft_args.ppl:
        trainer.train() #(resume_from_checkpoint=True)
        # trainer.train(resume_from_checkpoint=True)
        trainer.save_state()
        trainer.save_model(output_dir=ft_args.output_dir)

        if ft_args.ds_file is not None:
            command = [
                "python",
                f"{ft_args.output_dir}/zero_to_fp32.py",
                ft_args.output_dir,
                ft_args.output_dir,
            ]

            # Run the command to convert the model to fp32
            try:
                print("Running", command)
                result = subprocess.run(command, check=True, capture_output=True, text=True)
                print("Output:\n", result.stdout)
            except subprocess.CalledProcessError as e:
                print("Error:\n", e.stderr)
    else:
        results = trainer.evaluate()

        loss_value = torch.tensor(results["eval_loss"])
        perplexity_value = np.exp(np.array([results["eval_loss"]]))

        print(f"Loss: {loss_value}")
        print(f"Perplexity: {perplexity_value}")

        with open(f"{ft_args.output_dir}/results.txt", "w") as f:
            f.write(f"Loss: {loss_value}\n")
            f.write(f"Perplexity: {perplexity_value[0]}\n")

    end_time = time.time()
    print("Time end", end_time)
    with open(f"{ft_args.output_dir}/time_{ft_args.ppl}.txt", "w") as f:
        f.write(f"{start_time}\n")
        f.write(f"{end_time}\n")



# inspired by Stanford Alpaca
# https://github.com/QwenLM/Qwen/blob/main/tokenization_note.md
if __name__ == "__main__":
    finetune()