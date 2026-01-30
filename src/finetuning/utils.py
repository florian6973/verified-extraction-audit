import transformers
from transformers import BitsAndBytesConfig
from peft import PeftModel
import json
import os

from typing import Dict, Optional, Sequence

import torch

DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "<s>"
DEFAULT_UNK_TOKEN = "<unk>"


def smart_tokenizer_and_embedding_resize(
    special_tokens_dict: Dict,
    tokenizer: transformers.PreTrainedTokenizer,
    model: transformers.PreTrainedModel,
):
    """Resize tokenizer and embedding.

    CRUCIAL to avoid out of memory errors.

    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg


def get_base_model_path(directory):
    """
    Reads adapter_config.json from the given directory and extracts the value of 'base_model_name_or_path'.

    Args:
        directory (str): The directory containing the adapter_config.json file.

    Returns:
        str: The extracted base model path, or an error message if the file is not found or is invalid.
    """
    json_file_path = os.path.join(directory, "adapter_config.json")

    try:
        with open(json_file_path, "r") as file:
            config_data = json.load(file)
            return config_data.get("base_model_name_or_path", "Key not found")
    except FileNotFoundError:
        return f"Error: {json_file_path} not found."
    except json.JSONDecodeError:
        return f"Error: Failed to parse JSON in {json_file_path}."


def get_base_tokenizer_path(directory):
    """
    Reads adapter_config.json from the given directory and extracts the value of 'base_model_name_or_path'.

    Args:
        directory (str): The directory containing the adapter_config.json file.

    Returns:
        str: The extracted base model path, or an error message if the file is not found or is invalid.
    """
    json_file_path = os.path.join(directory, "config.json")

    try:
        with open(json_file_path, "r") as file:
            config_data = json.load(file)
            return config_data.get("_name_or_path", "Key not found")
    except FileNotFoundError:
        return f"Error: {json_file_path} not found."
    except json.JSONDecodeError:
        return f"Error: Failed to parse JSON in {json_file_path}."


def load_model(args, device="auto"):
    if args.lora and (args.infer or args.ppl):
        base_model_path = get_base_model_path(args.model_name_or_path)
    else:
        base_model_path = args.model_name_or_path

    # qf = BitsAndBytesConfig(
    #     # load_in_8bit=True
    #             load_in_4bit=True,
    #             bnb_4bit_compute_dtype=torch.float16,
    #             bnb_4bit_use_double_quant=True,
    #             bnb_4bit_quant_type='nf4'
    #         )

    if not args.model_name_or_path.startswith("scratch-"):
        print(base_model_path)
        model = transformers.AutoModelForCausalLM.from_pretrained(
            base_model_path,
            cache_dir=base_model_path,
            quantization_config=None,  # qf,#qf if (args.infer or args.lora) else None
            device_map=device if args.infer else None,
            # load_in_4bit=(args.infer or args.lora),
            # bnb_4bit_compute_dtype=torch.float16 if (args.infer or args.lora) else None
        )
        print(model)
        print(base_model_path)

        tokenizer = transformers.AutoTokenizer.from_pretrained(
            base_model_path,
            cache_dir=base_model_path,
            model_max_length=args.model_max_length,
            padding_side="right",
            use_fast=False,
        )

        if not tokenizer:
            tok_path = get_base_tokenizer_path(base_model_path)
            tokenizer = transformers.AutoTokenizer.from_pretrained(
                tok_path,
                cache_dir=tok_path,
                model_max_length=args.model_max_length,
                padding_side="right",
                use_fast=False,
            )
    
    special_tokens_dict = dict()
    if tokenizer.pad_token is None:
        special_tokens_dict["pad_token"] = DEFAULT_PAD_TOKEN
    if tokenizer.eos_token is None:
        special_tokens_dict["eos_token"] = DEFAULT_EOS_TOKEN
    if tokenizer.bos_token is None:
        special_tokens_dict["bos_token"] = DEFAULT_BOS_TOKEN
    if tokenizer.unk_token is None:
        special_tokens_dict["unk_token"] = DEFAULT_UNK_TOKEN

    smart_tokenizer_and_embedding_resize(
        special_tokens_dict=special_tokens_dict,
        tokenizer=tokenizer,
        model=model,
    )

    if args.lora and (args.infer or args.ppl):
        model = PeftModel.from_pretrained(model, args.model_name_or_path)

    return model, tokenizer
