# Verified Extraction Audit

Code for the verified extraction audit pipeline: **data preparation**, **model training** (supervised fine-tuning), and **evaluation** (verified extraction-risk metrics).

---

## Two scenarios

The audit answers *"how much does a fine-tuned model leak the direct identifiers in its training data?"* — in one of two settings.

- **Scenario 1 — synthetic or "perfectly" de-identified data (as in the paper).** Your notes have their direct identifiers removed (`___`). You **inject** synthetic identifiers into the blanks at a chosen rate, fine-tune, and measure how much the model memorizes and leaks. No labeled file to prepare — injection records the members for you.
- **Scenario 2 — a real, imperfectly de-identified corpus you want to audit.** The identifiers are already in the notes; there is no injection. You provide a small **labeled** file (a handful of identifiers marked member / non-member, from a manual review or a de-identification tool) and estimate the actual leakage.

---

## Overview

Only how the training data and labeled set are produced (Steps 1–2) differs between the two scenarios — **Steps 3–5 are identical**:

1. **Prepare data** → get your notes into a `(subject_id, note)` Parquet and `ingest` them into internal splits. *(Scenario 1: the notes carry `___` blanks and `ingest` also builds the synthetic personas the injector fills.)*
2. **Inject** *(scenario 1 only)* → fill the blanks with synthetic direct identifiers (offline or LLM), sampled to the target rate → SFT dataset + labeled set. *Scenario 2 skips this: the identifiers are already in the notes and you bring your own labeled set.*
3. **Train** → fine-tune a language model on the SFT data.
4. **Generate** → sample attacker-query *completions* from the fine-tuned model.
5. **Audit** → train the verification classifier and report the analytical + experimental extraction curves.

### Examples

```bash
# scenario 1, fully synthetic — builds a tiny model; no MIMIC, API keys, or download:
bash src/jobs/smoke/run_smoke.sh              # or: sbatch src/jobs/smoke/smoke_test.slurm
SCENARIO=2 bash src/jobs/smoke/run_smoke.sh   # scenario 2 (identifiers already in the notes)

# MIMIC-IV subset (default 1% of subjects) as one SLURM job:
sbatch --export=ALL,DISCHARGE=data/raw/discharge.csv,API_BASE=http://<host>:<port>/v1,MODEL=<served-model> \
       src/jobs/mimic/mimic_test.slurm        # knobs: FRAC, DI_RATE, CLASSIFIER, BASE_MODEL, N_EPOCHS, K … (run_mimic.sh)
```

Each writes a leakage report to `<work>/audit/audit_report.json` (e.g. `outputs/smoke/scenario1/…`). Training uses bf16 → request an **Ampere-or-newer GPU** (A100/H100/L4/RTX 30+); adjust the `SBATCH` / `CONDA_ENV` / `CUDA_MODULE` lines in the `.slurm` files for your cluster.

---

## Setup

- **Python:** 3.8+
- **Install:** From the repo root:

  ```bash
  cd verified-extraction-audit
  pip install -e .          # or: export PYTHONPATH=/path/to/verified-extraction-audit
  ```

- **Configuration is minimal.** The whole pipeline is driven by command-line flags and a single env var:
  - `DATA_ROOT` — where the pipeline writes its intermediate per-split files (default: `data/processed`).

- **Reproducibility.** Every step is seeded (`--seed`, default 42) and deterministic — splits, personas, which blanks get injected, generation, the verifier, training. The one exception is LLM classification (`inject --classifier llm`), which depends on the model; use `--classifier label` for a bit-for-bit reproducible injection.

---

## Input format

The pipeline consumes **at most two small Parquet files** — nothing MIMIC-specific:

1. **Dataset** — one row per note, columns `subject_id` and `note`; removed direct-identifier spans marked `___`:

   ```text
   subject_id,note
   1,"Patient ___ is a 48yo with chest pain"
   2,"John Doe has visited the ED on May 2, 2026"
   3,"Mr. ___ with blurred vision met the doctor Jane Smith"
   ```

2. **Labeled data** *(scenario 2 only)* — a handful of direct identifiers labeled member/non-member (`1` = member of the fine-tuning set, `0` = non-member), used to train the verification classifier:

   ```text
   entry,label
   "John Doe",1
   "Jane Doe",0
   ```

`subject_id` keeps all of a subject's notes on one side of the train/val split (use a unique id per note if there is no natural grouping). Which of the two files you need depends on your [scenario](#two-scenarios): scenario 1 needs only the notes (injection builds the labeled set); scenario 2 needs both. A ready-made synthetic example is in [`examples/synthetic/`](examples/synthetic/) (`notes.parquet`, `labeled.parquet`, plus `.csv` previews).

---

## Step 1 — Prepare your data

Get your notes into the internal per-split layout under `$DATA_ROOT` (default `data/processed`) that the rest of the pipeline reads — notes and synthetic personas matched **by `note_id`**, so nothing depends on row order. Produce it **either** way below; both converge on the same layout, so Steps 2–5 are unchanged.

### Option A — your own dataset (dataset-agnostic)

From a `(subject_id, note)` Parquet (see [Input format](#input-format)):

```bash
export DATA_ROOT=data/processed          # where the internal splits live (default; used by Steps 1–3)

# optional: generate a synthetic (subject_id, note) file to try things out
python -m src.dataset.prepare.make_synthetic --out data/notes.parquet --n-subjects 60

python -m src.dataset.prepare.ingest --input data/notes.parquet --name mydata \
  --out-root $DATA_ROOT --version 8 --val-frac 0.5
```

`ingest` splits on `subject_id` and generates the synthetic personas — no demographics required.

### Option B — MIMIC-IV (the paper's setup)

MIMIC-IV Clinical Notes require [PhysioNet access](https://physionet.org/content/mimic-iv-note/2.2/); download `discharge.csv` to `data/raw/`. The notes already contain `___` where identifiers were removed, so nothing is masked.

**Subsample → ingest.** Keep a subject-level fraction and feed it straight through Option A's `ingest` — no Hydra, no `index/`, no `admissions.csv`/`patients.csv`. This is exactly what the one-shot MIMIC job runs:

```bash
python -m src.dataset.prepare.mimic_subset --discharge data/raw/discharge.csv \
  --out data/mimic_1pct.parquet --frac 0.01          # --frac 1.0 for the full corpus
export DATA_ROOT=data/processed_mimic1
python -m src.dataset.prepare.ingest --input data/mimic_1pct.parquet --name mimic1 --out-root $DATA_ROOT
```

Then continue with Steps 2–5, or run the whole pipeline (subsample → ingest → inject → train → generate → audit) as one SLURM job — see [Examples](#examples).

*For the paper's exact build — the canonical `train/val/test` splits and demographically-matched personas — and for adapting a **different** corpus, see [Reproducing the paper](#reproducing-the-paper).*

---

## Step 2 — Inject direct identifiers

`inject` does the whole job in one step: for each note it **classifies** every `___` blank into a direct-identifier category, **fills** it with the matching field of the note's persona (matched **by `note_id`**, never by row order), and **samples** at the direct-identifier rate `--di-rate` — writing the SFT dataset plus the scenario-2 labeled set. Classification comes from either the note label or a single LLM call:

- **Offline / deterministic** — no LLM. `--classifier label` reads the note label (`Name:`, `MRN:`, …) before each blank; `--classifier first` fills only the first blank (handy for MIMIC, whose notes lead with `Name:`). Used by the smoke test:

  ```bash
  python -m src.dataset.prepare.inject --splits-root $DATA_ROOT --version 8 \
    --classifier label --di-type name --di-rate 0.05 \
    --output-sft $DATA_ROOT/sft --emit-labeled data/labeled.parquet
  ```

- **LLM** — one call per note asks the model to classify every blank at once; the persona then fills them. Use any **OpenAI-compatible server** (your local vLLM / llama.cpp / …) with `--api openai --api-base http://<host>:<port>/v1`, or `--api gemini` (set the Google Cloud env vars), `--api vllm` (default `localhost:12346`), or `--api mock` (offline dry run):

  ```bash
  python -m src.dataset.prepare.inject --splits-root $DATA_ROOT --version 8 \
    --classifier llm --api openai --api-base http://localhost:8000/v1 --model my-served-model \
    --di-type name --di-rate 0.05 --output-sft $DATA_ROOT/sft --emit-labeled data/labeled.parquet
  ```

**Direct-identifier rate.** `--di-rate` is the fraction of direct-identifier blanks that keep an identifier (`0.05` = 5%; the rest stay `___`) — the paper's "DI rate", and what replaces the old `pii_rate`/index column. It is baked into the SFT filename `<split>_<rate>.json`, so you can build several rates side by side and train/audit each independently:

  ```bash
  for R in 0.01 0.05 0.1 1.0; do
    python -m src.dataset.prepare.inject --splits-root $DATA_ROOT --classifier label \
      --di-type name --di-rate $R --output-sft $DATA_ROOT/sft --emit-labeled data/labeled_$R.parquet
  done   # then train + audit each train_$R.json independently (Steps 3 & 5)
  ```

This writes `$DATA_ROOT/sft/{train,val}_0.05.json` (the **SFT datasets** for training) and `members_<split>.csv`. Every classifier fills from the same personas by `note_id`, so results are independent of parquet row order or `os.listdir`.

---

## Step 3 — Train

Download a base model and fine-tune directly on the SFT data:

```bash
huggingface-cli download meta-llama/Llama-3.2-1B-Instruct --local-dir models/base/Llama_3.2-1B

python src/finetuning/finetune.py \
  --model_name_or_path models/base/Llama_3.2-1B \
  --dataset_path $DATA_ROOT/sft/train_0.05.json \
  --output_dir outputs/mydata/finetuned --n_epochs 3    # logs to W&B; run `wandb login` first
```

`finetune.py` derives the val SFT path from `--dataset_path` (`train`→`val`); launch it **by file path** (not `-m`). For multi-GPU / DeepSpeed use the same entry point with your launcher.

**Choosing the model & hyperparameters:**

- **Base model** — `--model_name_or_path` takes any Hugging Face causal-LM directory, so to change from Llama just download another model (Qwen, OLMo, a bigger Llama, …) and point at it. Use the **same** base for the audit's `--base-model` (the non-fine-tuned reference). For **multi-GPU FSDP**, the model directory name must contain a family key (`Llama`, `Qwen`, `Olmo`) so `finetune.py`'s `decoding_layers` picks the right decoder layer — add your `<Family>DecoderLayer` there for other architectures (single-GPU training doesn't need it).
- **Training hyperparameters are not in any config file.** The core ones — learning rate `2e-5`, per-device batch size `1`, gradient accumulation `8`, cosine schedule, `bf16` — are set directly in `finetune.py`'s `TrainingArguments`; edit them there. The CLI exposes `--n_epochs`, `--output_dir`, `--model_max_length`, `--lora`, `--save_total_limit`.

---

## Step 4 — Generate completions

The attacker queries the fine-tuned model with a direct-identifier prompt and samples **completions** (short continuations, not full notes):

```bash
python -m src.evaluation.pipeline.generate_completions \
  --model-path outputs/mydata/finetuned --output outputs/mydata/completions.parquet \
  --prompt "Name: " --k 100000 --max-new-tokens 20
```

These are the empirical attacker queries used to validate the analytical curves in Step 5. (Optional for an analytical-only audit.)

---

## Step 5 — Audit / estimate leakage

Train the verification classifier (5-fold cross-fit) and derive the extraction curves at the **extracted-stream FPR ≤ 5%** operating threshold:

```bash
python -m src.evaluation.audit.from_labels \
  --labeled data/labeled.parquet \
  --base-model models/base/Llama_3.2-1B \
  --finetuned-model outputs/mydata/finetuned \
  --di-type name --budgets 1e5 1e6 \
  --generations outputs/mydata/completions.parquet \
  --output-dir outputs/mydata/audit
```

`outputs/mydata/audit/audit_report.json` reports the verifier AUC and the closed-form **analytical** curves — recall, extracted-stream FPR/TPR vs attacker query budget *Q*, at the operating threshold τ. With `--generations` (Step 4) it adds the measured **experimental** extraction: the full confusion matrix (TP/FP, recall, TPR, FPR, PPV) over the generated candidates with 95% bootstrap CIs, so you can check the analytical curves against experiment at each *Q*. A completion counts as extracting a member only when the name is clean at its start (the paper's position filter, on by default; `--no-position-match` disables it), and `--filter-names` drops non-name junk from the false-positive pool.

---

## Other direct identifiers (MRNs, addresses, …)

Extending to a new direct identifier requires only three things, all captured per type in [`src/dataset/prepare/di_types.py`](src/dataset/prepare/di_types.py): the **query prompt(s)** (e.g. `"MRN: "` instead of `"Name: "`), the **generation length** (an address needs more tokens than a name), and the **parsing method** (extract digits for an MRN, two words for a name). `name`, `attending`, `mrn`, `address`, `phone`, and `email` are built in; pass `--di-type mrn` to the injection (Step 2) and audit (Step 5) steps, or add a `DirectIdentifierType` entry to extend the registry. Nothing else in the pipeline changes.

---

## Reproducing the paper

The paper's runs use an `index/` registry and Hydra configs instead of the flag-driven path above. They're optional — the new-dataset and MIMIC-subset paths never touch them — and kept here for exact reproduction.

- **Canonical splits + demographic personas.** Build the paper's fixed `train/val/test` (+ `train_1`/`val_1`/… subset) splits and demographically-matched personas (from `admissions.csv`/`patients.csv` in `data/raw/`), instead of the generic synthetic personas `ingest` makes — Steps 2–5 are otherwise identical:

  ```bash
  python -m src.dataset.splits.mimic                 # Hydra config src/configs/dataset/mimic.yaml
  python src/dataset/pii_insertion/fake_persona.py   # -> splits_filtered_v* + splits_personas_v* (validate with persona_check.py)
  ```

  *(Optional stats table: `src.dataset.splits.stats_to_latex`. To adapt a different corpus, replace `src/dataset/splits/mimic.py` with your own splits builder emitting `text`/`subject_id`/`note_id`.)*

- **Index-based training.** `seed_index.py` writes the `index/` registry; `finetune.py --model_id N` then reads the base model / dataset / epochs from it (env `INDEX_FOLDER`, default `./index`). The Hydra YAML under `src/configs/jobs/` drives only the SLURM launcher `src/jobs/finetuning/submit_finetuning_job.py` — orchestration (`n_gpu`, backend, which index dataset), not the training hyperparameters.

- **Index-based evaluation.** Per-name log-likelihood tables via Hydra (`src/configs/evaluation/log_likelihood/eval.yaml`), MIMIC/index-coupled (env `OUTPUT_DIR`); for a new dataset prefer the self-contained `from_labels` in [Step 5](#step-5--audit--estimate-leakage):

  ```bash
  python -m src.evaluation.pipeline.compute_risk dataset_size=1 model_size=1B output_dir=./outputs/pipeline
  python -m src.evaluation.pipeline.plot_relative_leakage_risk --dataset 10,100 --output outputs/plots/relative_leakage.png
  ```

- **Gemini injection.** `--api gemini` (Step 2) needs `GOOGLE_CLOUD_PROJECT` / `GOOGLE_CLOUD_LOCATION`.

The paper's figure/table/bootstrap analysis scripts live in [`src/evaluation/pipeline/paper/`](src/evaluation/pipeline/paper/) (see its README).

---

## Layout (summary)

- `src/dataset/prepare/` — the data-prep pipeline: `ingest` (minimal Parquet → internal splits + personas), `inject` (classify + fill + sample, offline or LLM, matched by `note_id`), `make_synthetic` (scenario 1 & 2), the `di_types` registry, and `seed_index` (optional — only for the index-driven training path).
- `src/dataset/pii_insertion/` — synthetic-persona utilities: `fake_persona` (persona generation, incl. the MIMIC build), `build_name_filter_list`, `persona_check`.
- `src/dataset/splits/` — MIMIC splits (`mimic.py`) and optional stats table; config in `src/configs/dataset/mimic.yaml`.
- `src/finetuning/` — training (`finetune.py`), utils, checkpoint post-processing.
- `src/evaluation/pipeline/` — attacker-query generation (`generate_completions`), closed-form curves (`theory_curves`, `attack_curves`), the paper's risk eval (`compute_risk*`, `plot_relative_leakage_risk`), and the verifier pieces the audit reuses under `experimental/` + `experimental/mia/`.
- `src/evaluation/pipeline/paper/` — legacy paper-reproduction analysis (figures, tables, bootstrap CIs); **not used by the live audit** (see its `README.md`).
- `src/evaluation/audit/` — the generic audit (`from_labels`, both scenarios) that composes the pipeline pieces.
- `src/jobs/smoke/` — the self-contained SLURM smoke test (`smoke_test.slurm`, `run_smoke.sh`, `build_tiny_model`).
- `src/jobs/mimic/` — the one-shot MIMIC-subset SLURM job (`mimic_test.slurm`, `run_mimic.sh`).
- `src/llm/` — LLM backends for injection: any OpenAI-compatible server (`openai`), plus `gemini`, `vllm`, and an offline `mock`.
- `src/folder_handler.py`, `index/` — the dataset/model index (placeholder CSVs; `seed_index` writes real ones).
- `examples/synthetic/` — a ready-made synthetic dataset.

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
