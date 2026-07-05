# Verified Extraction Audit

Code for the verified extraction audit pipeline: **data preprocessing** (PII insertion into clinical notes), **model training** (supervised fine-tuning), and **evaluation** (extraction risk metrics and plotting).

This repository is an anonymized, self-contained export of the components used for the paper. It does not include MIMIC-IV data, base model weights, or job submission scripts; you provide those locally.

---

## Overview

End-to-end flow:

1. **MIMIC notes** → download and build train/val/test splits (script not in this repo; see below).
2. **Data preprocessing** → generate synthetic personas, inject PII into notes (LLM or manual), sample to target PII rate → SFT JSON.
3. **Training** → fine-tune a language model on the SFT data (index-based or direct paths).
4. **Evaluation & plotting** → compute extraction-risk metrics (e.g. log-likelihood of PII) and plot results.

---

## Setup

- **Python:** 3.8+
- **Install:** From the repo root (e.g. `verified-extraction-audit/`):

  ```bash
  cd verified-extraction-audit
  pip install -e .
  ```

  Or run scripts with `PYTHONPATH` set to the repo root:

  ```bash
  export PYTHONPATH=/path/to/verified-extraction-audit
  ```

- **Paths:** Default paths are relative to the repo root. Override with env vars when needed:
  - `INDEX_FOLDER` — index directory (default: `./index`).
  - `PII_INSERTION_OUTPUTS` — base directory for PII insertion outputs.
  - `OUTPUT_DIR` — base directory for evaluation outputs (e.g. denominators, plots).
  - `GEMINI_USAGE_LOG` — path for Gemini API usage log (if using Gemini).
  - `GOOGLE_CLOUD_PROJECT`, `GOOGLE_CLOUD_LOCATION` — for Gemini/Vertex (no defaults in repo).

---

## Step 1: MIMIC notes and splits

MIMIC-IV Clinical Notes require [PhysioNet access](https://physionet.org/content/mimic-iv-note/2.2/).

**What you need to do:**

1. Download from [MIMIC-IV Note](https://physionet.org/content/mimic-iv-note/2.2/note):
   - `discharge.csv` (and any other note tables you use).
2. Optionally from [MIMIC-IV](https://physionet.org/content/mimiciv/3.1/): `admissions.csv`, `patients.csv`.
3. Place them under `data/raw/` (or your chosen `data_dir`).
4. **Build splits** using the included script (Hydra config: `src/configs/dataset/mimic.yaml`; default `data_dir: data/raw`, `output_dir: data/processed`):

   ```bash
   python -m src.dataset.splits.mimic
   ```

   Override paths via Hydra, e.g.:

   ```bash
   python -m src.dataset.splits.mimic data_dir=data/raw output_dir=data/processed
   ```

   This writes parquet files under `output_dir/splits/` (e.g. `train_1`, `val_1`, `train_10`, `val_10`, `train`, `val`, `test`). You still need to **filter/mask** note text so that PII slots are `___` (e.g. `data/processed/splits_filtered_v8/`) and build **personas** (Step 1).

5. **Optional:** compute split statistics for papers:

   ```bash
   python -m src.dataset.splits.stats_to_latex --splits_dir data/processed/splits --output outputs/splits/stats_table.tex
   ```

The rest of the pipeline expects:
- **Splits:** Parquet files per split with at least `text`, `subject_id`, and note identifiers.
- **Filtered notes:** Same structure with `___` where PII will be injected (e.g. `data/processed/splits_filtered_v8/`).
- **Personas:** One parquet per split (e.g. `data/processed/splits_personas_v8/`), produced in Step 1.

---

## Step 2: Data preprocessing (PII insertion)

From the repo root.

1. **Generate synthetic personas** (e.g. patient/physician names and other PII):

   The `FakePersonas` class in `src/dataset/pii_insertion/fake_persona.py` is used to generate the synthetic PII. 

2. **PII injection** (fill `___` with synthetic PII using an LLM or manual mapping):

   - **LLM-based** (e.g. Gemini):

     ```bash
     # Set Google Cloud / Gemini env vars first
     python -m src.dataset.pii_insertion.pii_injection \
       --api gemini --model gemini-2.5-flash-preview-05-20 \
       --files val_1 train_1 --num-workers 4
     ```

     Outputs go under `outputs/pii_insertion/direct/<model>_v8/<split>/` (tags, JSON, etc.). Paths assume personas at `data/processed/splits_personas_v8/<split>.parquet` and texts at `data/processed/splits_filtered_v8/<split>.parquet`; adjust in the script if your layout differs.

     This step is necessary to determine the semantic tags of each blank.

   - **Manual insertion:** use `manual_insertion.py` with your paths (see script and `src/dataset/pii_insertion/README.md`).

3. **Sampling** (target PII rate and build SFT JSON):

   ```bash
   python -m src.dataset.pii_insertion.sampling \
     --splits_raw_path data/processed/splits_filtered_v8 \
     --splits_base_path outputs/pii_insertion/direct \
     --output_path data/processed/splits_sft_with_index \
     --model gemini-2.5-flash-preview-05-20_v8 \
     --proportion_pii 0.05
   ```

   Or use `sampling_manual.py` for manual-insertion data. This produces JSON files such as `data/processed/splits_sft_with_index/train_1_0.05_no-kg.json` (path pattern may vary). Those are the **SFT datasets** used for training.

4. **Validation (optional):**  
   - `persona_check.py` — check persona vs. note alignment and duplicates.  

---

## Step 3: Training

- **Base model:** Download a Hugging Face model into e.g. `models/base/`:

  ```bash
  huggingface-cli download meta-llama/Llama-3.2-1B-Instruct --local-dir models/base/Llama_3.2-1B
  ```

- **Index:** Training in this repo is driven by `index/` (see `EXPORT_PLAN.md`). You need:
  - `index/datasets.csv` — one row per dataset (columns at least: `dataset_id`, `dataset_size`, `pii_rate`, `kg`, `injection_strategy`, `name_strategy`, `sampling_strategy`, `dataset_path`, `status`, `persona_path`, `person_path_name`).
  - `index/models.csv` — one row per (base model, dataset, checkpoint) you want to train (columns at least: `model_id`, `model_name`, `type`, `model_size`, `dataset_id`, `n_epochs`, `model_path`, `src_model_path`, `status`).

  Point `dataset_path` to your SFT JSON (e.g. `data/processed/splits_sft_with_index/train_1_0.05_no-kg.json`) and `model_path` / `src_model_path` to your output dir and base model.

- **Job config (reference):** A sample finetuning job config is in `src/configs/jobs/submit_finetuning_job_1b_large.yaml` (base model, dataset size, epochs, PII rate, etc.). Set `base_path` and `home_path` to your repo root and home; use it with your own job launcher if you have one.

- **Run training** (single GPU, using `model_id` from the index):

  ```bash
  DS_SKIP_CUDA_CHECK=1 torchrun --nproc_per_node=1 src/finetuning/finetune.py --model_id <model_id>
  ```

  All other arguments (dataset path, base model, output dir, epochs, etc.) are taken from the index row for that `model_id`. For multi-GPU or DeepSpeed, use the same entry point with your launcher and config (not shipped here).

- **Post-processing checkpoints (optional):**  
  `src/finetuning/postproc_models.py` can add extra checkpoint rows to `index/models.csv` (e.g. different epochs). Run from repo root; it reads `INDEX_FOLDER` or `index/models.csv` by default. Edit the script’s checkpoint names/paths to match your run.

---

## Step 4: Evaluation and plotting

- **Compute risk (log-likelihood of PII, train/val tables):**

  Uses Hydra; config lives under `src/configs/evaluation/log_likelihood/eval.yaml`. From repo root:

  ```bash
  python -m src.evaluation.pipeline.compute_risk
  ```

  Default config uses `dataset_size: 10`, `model_size: 8B`, `output_dir: ./outputs/pipeline`. Override via Hydra, e.g.:

  ```bash
  python -m src.evaluation.pipeline.compute_risk dataset_size=1 model_size=1B output_dir=./outputs/pipeline_test
  ```

  This script:
  - Uses `index/` (and `src.evaluation.exploration.denominators`) to resolve datasets and persona paths.
  - Writes CSVs under `output_dir` (e.g. `ll_train_*.csv`, `ll_val_true_*.csv`, `ll_all_*.csv`, `ll_all_output_*_batch.csv`).

  For a **batch** variant (e.g. larger batch size):

  ```bash
  python -m src.evaluation.pipeline.compute_risk_batch
  ```

  Set `base_model_1B` / `base_model_8B` in `eval.yaml` (or env `BASE_MODEL_1B` / `BASE_MODEL_8B`) if you are not using index for base models.

- **Plotting (e.g. relative leakage risk):**

  After you have the evaluation CSVs (e.g. `ll_all_output_False_1B_10_batch.csv` and `ll_all_output_True_1B_10_batch.csv` under your pipeline output dir), run:

  ```bash
  python -m src.evaluation.pipeline.plot_relative_leakage_risk \
    --output outputs/plots/relative_emission_probability_change_leakage_risk.png \
    --top_names_output outputs/plots/top_100_names_by_factor.csv \
    --dataset 10,100
  ```

  The script expects finetuned- and base-model CSVs under a fixed `base_dir` inside the script (see `plot_relative_leakage_risk.py`); you may need to set `base_dir` or `OUTPUT_DIR` there to match your `output_dir` from the evaluation step. Other options: `--include_val`, `--percentile`, `--prompt`, etc.

  Other plotting/analysis scripts under `src/evaluation/pipeline/` and `experimental/` can be run similarly once the corresponding CSVs exist.

---

## First end-to-end test (minimal)

Goal: run the pipeline once with minimal data (no real MIMIC download required for the code to *run*; you still need valid paths and optionally tiny synthetic data).

1. **Environment**
   ```bash
   cd verified-extraction-audit
   pip install -e .
   export PYTHONPATH=.   # if not using pip install -e .
   ```

2. **Index**
   - Ensure `index/datasets.csv` and `index/models.csv` exist (placeholders are in the repo). For a real minimal run, add one dataset row whose `dataset_path` points to an SFT JSON (from Step 1), and one model row whose `dataset_id` matches, `src_model_path` points to a base model, and `model_path` points to where you want the checkpoint.

3. **MIMIC / splits**
   - Download MIMIC-IV `discharge.csv` to `data/raw/`, then run `python -m src.dataset.splits.mimic` to build `data/processed/splits/`. Then create filtered notes (with `___` placeholders) and personas for at least one split (e.g. `train_1`, `val_1`), and one SFT JSON (e.g. `train_1_0.05_no-kg.json`) referenced in `index/datasets.csv`. Alternatively create minimal parquet/JSON by hand to match that layout.

4. **Preprocessing**
   - Run `fake_persona` (or your wrapper) → personas parquet.
   - Run `pii_injection` (e.g. `--files val_1 train_1`) or manual insertion.
   - Run `sampling` or `sampling_manual` → SFT JSON.

5. **Training**
   - Point `index/models.csv` to your base model and that SFT dataset; set `model_id` to that row.
   - Run: `DS_SKIP_CUDA_CHECK=1 torchrun --nproc_per_node=1 src/finetuning/finetune.py --model_id <id>`.

6. **Evaluation**
   - Run: `python -m src.evaluation.pipeline.compute_risk` (override `output_dir` and config as needed). If you don’t have generation outputs yet, this still builds the train/val risk tables from the index and model paths.
   - Run: `python -m src.evaluation.pipeline.plot_relative_leakage_risk ...` with the CSVs produced in `output_dir`.

7. **Optional**
   - Run `compute_risk_batch` instead of `compute_risk` for different batch behavior.
   - Use `INDEX_FOLDER`, `PII_INSERTION_OUTPUTS`, `OUTPUT_DIR`, and base-model env vars to avoid hardcoded paths.

This gives you a first end-to-end pass: **MIMIC (or minimal splits) → preprocessing → training → evaluation → plotting**. Add generation (Step 3) when you have the external generation script and desired output format.

---

## Layout (summary)

- `src/dataset/pii_insertion/` — PII insertion and sampling (fake personas, injection, sampling, validation).
- `src/dataset/splits/` — Build MIMIC splits (`mimic.py`) and optional stats table (`stats_to_latex.py`); config in `src/configs/dataset/mimic.yaml`.
- `src/finetuning/` — Training (`finetune.py`), utils, post-processing of model index.
- `src/configs/jobs/` — Sample finetuning job config (e.g. `submit_finetuning_job_1b_large.yaml`).
- `src/evaluation/pipeline/` — Risk computation (`compute_risk.py`, `compute_risk_batch.py`) and plotting (`plot_relative_leakage_risk.py`, etc.); `experimental/` and `experimental/mia/` for extra analyses.
- `src/evaluation/exploration/` — Helpers (e.g. `denominators.py`) used by the pipeline.
- `src/configs/evaluation/log_likelihood/` — Hydra config for evaluation.
- `src/folder_handler.py` — Index loading (datasets, models, generated notes).
- `src/llm/` — Minimal LLM backend for PII injection (Gemini + vLLM stubs).
- `index/` — Placeholder CSVs; replace with your datasets and models for real runs.

---

## Citation

Please cite the paper below when this repository contributes to your method,
experiments, results, or implementation.

```bibtex
@inproceedings{pollet2026privacy,
  title     = {Privacy Audits for Clinical Large Language Models},
  author    = {Pollet, Florent and Nikitin, Kirill and Wang, Tong and
               Gupta, Rahul and Elhadad, No\'emie and Gursoy, Gamze},
  booktitle = {Machine Learning for Healthcare Conference (MLHC)},
  year      = {2026}
}
```

## License

The code in this repository is licensed under the Apache License 2.0.
See [LICENSE](LICENSE).

This repository does not distribute MIMIC-IV data, trained artifacts, or any
other restricted data. Users must obtain any required data access independently
and comply with the applicable data-use agreements.