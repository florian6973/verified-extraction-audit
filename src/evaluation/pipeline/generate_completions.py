# generate_ll_batched.py

import os
import gc
import random
import numpy as np
import torch
import pandas as pd
from tqdm import tqdm

import pyarrow as pa
import pyarrow.parquet as pq

from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig

from src.folder_handler import FolderHandler


# ======================
# Reproducibility
# ======================

from src._repo import REPO_ROOT
SEED = 42
torch.manual_seed(SEED)
random.seed(SEED)
np.random.seed(SEED)


# ======================
# Parquet writer
# ======================

class ParquetWriter:
    def __init__(self, path, schema):
        self.path = path
        self.schema = schema
        self.writer = None

    def write(self, records):
        table = pa.Table.from_pylist(records, schema=self.schema)
        if self.writer is None:
            os.makedirs(os.path.dirname(self.path), exist_ok=True)
            self.writer = pq.ParquetWriter(self.path, self.schema)
        self.writer.write_table(table)

    def close(self):
        if self.writer is not None:
            self.writer.close()


# ======================
# Batched generation with KV cache
# ======================

@torch.no_grad()
def generate_batch(
    prompt,
    tok,
    model,
    batch_size,
    max_new_tokens=20,
    tau=1.0,
):
    device = model.device

    encoded = tok(prompt.rstrip(), return_tensors="pt")
    input_ids = encoded["input_ids"].to(device)
    input_ids = input_ids.repeat(batch_size, 1)  # [B, T]

    out = model(input_ids, use_cache=True)
    past = out.past_key_values

    eos_token_id = tok.eos_token_id

    all_tokens = []
    ll = torch.zeros(batch_size, device=device)
    finished = torch.zeros(batch_size, dtype=torch.bool, device=device)

    for _ in range(max_new_tokens):
        logits = out.logits[:, -1, :]  # [B, V]

        if tau != 1.0:
            logits = logits / tau

        probs = torch.softmax(logits, dim=-1)
        next_tokens = torch.multinomial(probs, 1).squeeze(-1)

        token_ll = torch.log(probs.gather(-1, next_tokens.unsqueeze(-1)).squeeze(-1))
        ll += token_ll * (~finished)

        all_tokens.append(next_tokens)

        if eos_token_id is not None:
            finished |= next_tokens.eq(eos_token_id)

        if finished.all():
            break

        out = model(
            next_tokens.unsqueeze(-1),
            past_key_values=past,
            use_cache=True,
        )
        past = out.past_key_values

    token_ids = torch.stack(all_tokens, dim=1)  # [B, L]

    decoded = tok.batch_decode(token_ids, skip_special_tokens=True)

    n_tokens = (
        token_ids.ne(eos_token_id).sum(dim=1).cpu().tolist()
        if eos_token_id is not None
        else [token_ids.shape[1]] * batch_size
    )

    return {
        "values": decoded,
        "ll": ll.cpu().tolist(),
        "n_tokens": n_tokens,
        "tokens": token_ids.cpu().tolist(),
    }


# ======================
# Model preparation
# ======================

def prepare_models():
    fh = FolderHandler()
    df = fh.load_models()

    df = df[df["injection_strategy"] == "manual"]
    df = df[df["model_id"] > 27]
    # df = df[(df["model_id"] == 71)]
    # df = df[(df["model_id"] == 73)]
    # df = df[(df["model_id"] == 72)]
    # df = df[(df["model_id"] == 78)]
    # df = df[(df["model_id"] == 75)]
    df = df[(df["model_id"] == 76)]

    return df[["dataset_size", "pii_rate", "n_epochs", "model_path", "model_size"]]


def load_model(model_path, base=False):
    tok = AutoTokenizer.from_pretrained(model_path)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    if base == "scratch":
        config = AutoConfig.from_pretrained(model_path)
        model = AutoModelForCausalLM.from_config(config)
    elif torch.cuda.is_available():
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map="auto",
        )
    else:
        # CPU fallback (e.g. local smoke test): fp32, no device_map.
        model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float32)

    model.eval()
    return tok, model


# Shared Arrow schema for the generation parquet.
GEN_SCHEMA = pa.schema([
    ("dataset_size", pa.int64()),
    ("model_size", pa.string()),
    ("pii_rate", pa.float32()),
    ("n_epochs", pa.int64()),
    ("pii_type", pa.string()),
    ("prompt", pa.string()),
    ("value", pa.string()),
    ("ll", pa.float32()),
    ("n_tokens", pa.int32()),
    ("tokens", pa.list_(pa.int32())),
])


def generate_single(model_path, output_path, k, prompt="Name: ", pii_type="name",
                    batch_size=64, max_new_tokens=20, meta=None, base=False):
    """Generate ``k`` completions of ``prompt`` from one model to ``output_path``.

    The dataset-agnostic entry used by the smoke test / new datasets: it drives
    the same ``generate_batch`` sampler as the index-based path, but takes an
    explicit model path instead of resolving a hardcoded ``model_id``.
    """
    meta = meta or {}
    tok, model = load_model(model_path, base=base)
    writer = ParquetWriter(output_path, GEN_SCHEMA)
    buffer = []
    remaining = k
    pbar = tqdm(total=k, desc=f"{pii_type} | {prompt}")
    while remaining > 0:
        b = min(batch_size, remaining)
        out = generate_batch(prompt=prompt, tok=tok, model=model,
                             batch_size=b, max_new_tokens=max_new_tokens)
        for i in range(b):
            buffer.append({
                "dataset_size": int(meta.get("dataset_size", 0)),
                "model_size": str(meta.get("model_size", "")),
                "pii_rate": float(meta.get("pii_rate", 0.0)),
                "n_epochs": int(meta.get("n_epochs", 0)),
                "pii_type": pii_type,
                "prompt": prompt,
                "value": out["values"][i],
                "ll": out["ll"][i],
                "n_tokens": out["n_tokens"][i],
                "tokens": out["tokens"][i],
            })
        if len(buffer) >= 5000:
            writer.write(buffer)
            buffer.clear()
        remaining -= b
        pbar.update(b)
    pbar.close()
    if buffer:
        writer.write(buffer)
    writer.close()
    del tok, model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print("Finished:", output_path)


# ======================
# Main generation loop
# ======================

def generate_ll(
    k,
    base=False,
    batch_size=64,
    max_new_tokens=20,
):

    prompts = {
        # "name": ["Name: ", "Patient: "],
        "name": ["Name: "],
    }

    if base:
        df_models = pd.DataFrame([{
            "dataset_size": 1,
            "pii_rate": 0.0,
            "model_size": "1B",
            "n_epochs": 0,
            "model_path": REPO_ROOT + "/models/base/Llama_3.2-1B",
        }])
    else:
        df_models = prepare_models()

    schema = pa.schema([
        ("dataset_size", pa.int64()),
        ("model_size", pa.string()),
        ("pii_rate", pa.float32()),
        ("n_epochs", pa.int64()),
        ("pii_type", pa.string()),
        ("prompt", pa.string()),
        ("value", pa.string()),
        ("ll", pa.float32()),
        ("n_tokens", pa.int32()),
        ("tokens", pa.list_(pa.int32())),
    ])

    for _, row in df_models.iterrows():
        row_ds = f'{row["dataset_size"]}_{row["model_size"]}_{row["pii_rate"]}_{row["n_epochs"]}'
        output_path = os.path.join(
            REPO_ROOT, "outputs", "pii_leakage",
            "experimental-recall-all-large",
            f"generation_{base}_all_{row_ds}_{k}.parquet"
        )

        if os.path.exists(output_path):
            print(f"Skipping existing file: {output_path}")
            continue
        if os.path.exists(os.path.dirname(output_path)):
            print(f"Creating directory: {os.path.dirname(output_path)}")
            os.makedirs(os.path.dirname(output_path), exist_ok=True)

        print("Loading model:", row["model_path"])
        tok, model = load_model(row["model_path"], base=base)

        writer = ParquetWriter(output_path, schema)
        buffer = []
        FLUSH_EVERY = 5_000

        for pii_type, prompt_list in prompts.items():
            for prompt in prompt_list:
                remaining = k

                pbar = tqdm(total=k, desc=f"{pii_type} | {prompt}")

                while remaining > 0:
                    b = min(batch_size, remaining)

                    out = generate_batch(
                        prompt=prompt,
                        tok=tok,
                        model=model,
                        batch_size=b,
                        max_new_tokens=max_new_tokens,
                    )

                    for i in range(b):
                        buffer.append({
                            "dataset_size": int(row["dataset_size"]),
                            "model_size": row["model_size"],
                            "pii_rate": float(row["pii_rate"]),
                            "n_epochs": int(row["n_epochs"]),
                            "pii_type": pii_type,
                            "prompt": prompt,
                            "value": out["values"][i],
                            "ll": out["ll"][i],
                            "n_tokens": out["n_tokens"][i],
                            "tokens": out["tokens"][i],
                        })

                    if len(buffer) >= FLUSH_EVERY:
                        writer.write(buffer)
                        buffer.clear()

                    remaining -= b
                    pbar.update(b)

                pbar.close()

        if buffer:
            writer.write(buffer)

        writer.close()

        del tok, model
        gc.collect()
        torch.cuda.empty_cache()

        print("Finished:", output_path)


# ======================
# Entrypoint
# ======================

def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="Generate attacker-query completions from a fine-tuned model.")
    parser.add_argument("--model-path", default=None,
                        help="Generate for this single model (bypasses the hardcoded index). "
                             "If omitted, runs the authors' index-based loop.")
    parser.add_argument("--output", default=None, help="Output parquet path")
    parser.add_argument("--k", type=int, default=10000, help="Number of completions to sample")
    parser.add_argument("--prompt", default="Name: ", help="Attacker query prompt")
    parser.add_argument("--pii-type", default="name")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-new-tokens", type=int, default=20)
    parser.add_argument("--base", action="store_true", help="Treat model as the base/reference model")
    parser.add_argument("--dataset-size", type=int, default=1)
    parser.add_argument("--model-size", default="")
    parser.add_argument("--pii-rate", type=float, default=1.0)
    parser.add_argument("--n-epochs", type=int, default=1)
    args = parser.parse_args()

    if args.model_path:
        out = args.output or os.path.join(
            REPO_ROOT, "outputs", "pii_leakage", "experimental-recall-all-large",
            f"generation_{bool(args.base)}_all_{args.dataset_size}_{args.model_size}_"
            f"{args.pii_rate}_{args.n_epochs}_{args.k}.parquet")
        meta = dict(dataset_size=args.dataset_size, model_size=args.model_size,
                    pii_rate=args.pii_rate, n_epochs=args.n_epochs)
        generate_single(args.model_path, out, args.k, prompt=args.prompt, pii_type=args.pii_type,
                        batch_size=args.batch_size, max_new_tokens=args.max_new_tokens,
                        meta=meta, base=args.base)
    else:
        # Authors' index-based path (unchanged): resolves a hardcoded model_id.
        for k in [1000, 10_000, 100_000, 1_000_000, 10_000_000]:
            generate_ll(k=k, batch_size=64, max_new_tokens=20)


if __name__ == "__main__":
    main()

# CUDA_VISIBLE_DEVICES=0 python -m src.evaluation.pipeline.generate_completions \
#   --model-path outputs/mydata/finetuned --k 10000 --output outputs/mydata/completions.parquet