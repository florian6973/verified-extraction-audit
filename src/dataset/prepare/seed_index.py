"""Seed a minimal ``index/`` (datasets.csv + models.csv) for one dataset+model.

Training is driven by the ``index/`` registry (see ``src/folder_handler.py``):
``finetune.py --model_id N`` looks up model ``N`` joined to its dataset to find
the base model, output path, epochs, and SFT dataset. This writes a single-row
registry pointing at an already-prepared SFT dataset and a base model, so
``finetune.py`` runs without hand-editing CSVs.

Writes into ``--index-folder`` (point ``INDEX_FOLDER`` there) so it never
clobbers the shipped placeholder ``index/`` at the repo root.

Example
-------
    python -m src.dataset.prepare.seed_index \
        --index-folder outputs/smoke/index \
        --sft-train outputs/smoke/sft/train_100_1.0_no-kg.json \
        --personas-train outputs/smoke/processed/splits_personas_v8/train.parquet \
        --base-model models/base/Llama_tiny \
        --model-out outputs/smoke/models/finetuned --n-epochs 1
"""

import argparse
import os

import pandas as pd


def seed_index(index_folder, sft_train, personas_train, base_model, model_out,
               n_epochs, model_name, model_size, dataset_size, pii_rate):
    os.makedirs(index_folder, exist_ok=True)
    abspath = os.path.abspath

    datasets = pd.DataFrame([{
        "dataset_id": 0,
        "dataset_size": dataset_size,
        "pii_rate": pii_rate,
        "kg": "no-kg",
        "injection_strategy": "manual",
        "name_strategy": "real",
        "sampling_strategy": "uniform",
        "dataset_path": abspath(sft_train),
        "status": "done",
        "persona_path": abspath(personas_train) if personas_train else "",
        "person_path_name": os.path.basename(personas_train) if personas_train else "",
    }])
    models = pd.DataFrame([{
        "model_id": 0,
        "model_name": model_name,
        "type": "instruct",
        "model_size": model_size,
        "dataset_id": 0,
        "n_epochs": n_epochs,
        "model_path": abspath(model_out),
        "src_model_path": abspath(base_model),
        "status": "training",
    }])
    generated = pd.DataFrame(columns=[
        "generated_notes_id", "model_id", "n_notes", "temperature", "top_p",
        "min_p", "top_k", "repetition_penalty", "generated_notes_path", "status",
    ])

    datasets.to_csv(os.path.join(index_folder, "datasets.csv"), index=False)
    models.to_csv(os.path.join(index_folder, "models.csv"), index=False)
    generated.to_csv(os.path.join(index_folder, "generated_notes.csv"), index=False)
    print(f"Seeded index at {index_folder}: dataset 0 -> {abspath(sft_train)}; "
          f"model 0 -> src {abspath(base_model)} out {abspath(model_out)}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index-folder", required=True)
    parser.add_argument("--sft-train", required=True, help="Path to the train SFT JSON")
    parser.add_argument("--personas-train", default=None, help="Path to the train personas parquet")
    parser.add_argument("--base-model", required=True, help="Path to the base model dir")
    parser.add_argument("--model-out", required=True, help="Where the fine-tuned model will be written")
    parser.add_argument("--n-epochs", type=int, default=1)
    parser.add_argument("--model-name", default="Llama_tiny")
    parser.add_argument("--model-size", default="tiny")
    parser.add_argument("--dataset-size", type=int, default=1)
    parser.add_argument("--pii-rate", type=float, default=1.0)
    args = parser.parse_args()
    seed_index(args.index_folder, args.sft_train, args.personas_train, args.base_model,
               args.model_out, args.n_epochs, args.model_name, args.model_size,
               args.dataset_size, args.pii_rate)


if __name__ == "__main__":
    main()
