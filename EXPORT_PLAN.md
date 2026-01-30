# Anonymized Repo Export Plan: verified-extraction-audit

This document lists what to copy, what to add for dependencies, and which paths/imports to change so the anonymized repo runs standalone.

---

## 1. Target layout (recommended)

Keep the `src/` layout so existing imports work with minimal edits:

```
verified-extraction-audit/
├── README.md                    # You will write
├── pyproject.toml               # Anonymized, reduced deps
├── .gitignore                   # Copy from main repo
├── index/                       # Placeholder or sample CSVs (see below)
├── src/
│   ├── folder_handler.py        # Shared; paths parameterized
│   ├── llm/                     # Minimal LLM backend (for pii_insertion)
│   │   ├── __init__.py
│   │   ├── gemini/
│   │   │   └── utils.py
│   │   └── vllm/
│   │       └── utils.py
│   ├── dataset/
│   │   └── pii_insertion/       # Full folder (all .py, README, templates/)
│   ├── finetuning/
│   │   ├── finetune.py
│   │   ├── postproc_models.py
│   │   ├── utils.py
│   │   └── config_ds.json       # If used by finetune (optional)
│   ├── evaluation/
│   │   ├── exploration/
│   │   │   └── denominators.py
│   │   ├── pipeline/            # Full folder including experimental/, experimental/mia/
│   │   └── (no other evaluation subdirs)
│   └── configs/
        JOBS too submit_finetuning_job.yaml for finetuning
│       └── evaluation/
│           └── log_likelihood/
│               └── eval.yaml
```

Run from repo root so that `src` is on `PYTHONPATH` (e.g. `pip install -e .` or `export PYTHONPATH=...`).

---

## 2. What to copy (file list)

### 2.1 Data preprocessing (PII insertion)

| Source | Destination |
|--------|-------------|
| `src/dataset/pii_insertion/*.py` | `src/dataset/pii_insertion/` |
| `src/dataset/pii_insertion/README.md` | same |
| `src/dataset/pii_insertion/templates/evaluate_pii.html` | same |

**Files:**  
`build_name_filter_list.py`, `check_incomplete_completions.py`, `count_caregivers.py`, `evaluate_pii.py`, `fake_persona.py`, `manual_insertion.py`, `persona_check.py`, `pii_injection.py`, `sampling_manual_old.py`, `sampling_manual.py`, `sampling.py`, `zip_data.py`

**Dependency:**  
`pii_injection.py` uses `from src.llm import call_llm` → you need the minimal `src/llm` tree below.

---

### 2.2 Model training

| Source | Destination |
|--------|-------------|
| `src/finetuning/finetune.py` | `src/finetuning/finetune.py` |
| `src/finetuning/postproc_models.py` | `src/finetuning/postproc_models.py` |
| `src/finetuning/utils.py` | `src/finetuning/utils.py` |
| `src/folder_handler.py` | `src/folder_handler.py` |

Optional: `src/finetuning/config_ds.json` if your finetuning flow reads it. YES

**Dependencies:**  
- `finetune.py`: `from utils import load_model, get_base_model_path` (local), `from src.folder_handler import FolderHandler`.
- `postproc_models.py`: only `pandas` (no internal deps).

---

### 2.3 Evaluation (pipeline)

| Source | Destination |
|--------|-------------|
| `src/evaluation/pipeline/*.py` (all top-level .py) | `src/evaluation/pipeline/` |
| `src/evaluation/pipeline/*.sh` | same |
| `src/evaluation/pipeline/README.md` | same |
| `src/evaluation/pipeline/experimental/` (all files and subdirs) | `src/evaluation/pipeline/experimental/` |
| `src/evaluation/exploration/denominators.py` | `src/evaluation/exploration/denominators.py` |
| `src/configs/evaluation/log_likelihood/eval.yaml` | `src/configs/evaluation/log_likelihood/eval.yaml` |

Include `experimental/mia/` (e.g. `name_filter.py`, `config_loader.py`, `config_helper.py`, etc.) and any YAML configs under `experimental/` that your paper’s scripts use.

**Dependencies:**  
- `compute_risk.py`: Hydra `config_path="../../configs/evaluation/log_likelihood"`, `config_name="eval"`; `select_datasets` / `read_dataset` from `src.evaluation.exploration.denominators`; `FolderHandler`.
- `compute_risk_batch.py`: same config + `FolderHandler`; `from names_dataset import NameDataset, NameWrapper` (only used in commented `prepare_last_name` — see path changes).
- Many scripts in `experimental/`: `config_loader`, `config_helper` (which use `FolderHandler`), and `experimental/mia/name_filter`.

---

### 2.4 Shared / LLM backend (for PII insertion only)

| Source | Destination |
|--------|-------------|
| `src/llm/__init__.py` | `src/llm/__init__.py` |
| `src/llm/gemini/utils.py` | `src/llm/gemini/utils.py` |
| `src/llm/vllm/utils.py` | `src/llm/vllm/utils.py` |

Add empty `src/llm/gemini/__init__.py` and `src/llm/vllm/__init__.py` if missing.

---

## 3. Paths and config to change (anonymization + portability)

### 3.1 `src/folder_handler.py`

- **Line 5:** `folder_default = "/gpfs/commons/groups/gursoy_lab/fpollet/Git/clinical-exposure-metric/index"`  
  → Use repo-relative default, e.g. `os.path.join(os.path.dirname(__file__), "..", "..", "index")` or read from env `INDEX_FOLDER` (default `./index`).
- **Lines 88–91, 105–106, 141–142, 181–228:** Hardcoded `/gpfs/commons/...` paths in `build_index_based_on_existing_folders()` and `update_persona_path()`.  
  → Either remove this method from the export or replace with relative paths / config; for the paper you can leave the method but document that it is optional/legacy and that `index_folder` is the single source of paths.

### 3.2 `src/evaluation/exploration/denominators.py`

- **Line 10:** `datasets_index_path = '/gpfs/commons/.../index/datasets.csv'`  
  → Use same convention as `FolderHandler` (e.g. `index/datasets.csv` relative to repo root or from env).
- **Lines 36–37:** `path_tags` and `path_value` with `/gpfs/commons/.../outputs/pii_insertion/...`  
  → Parameterize via config or function arguments (e.g. a base path for “PII insertion outputs”).
- **Line 98:** `results.to_csv('.../denominators.csv', ...)`  
  → Use a path under `output_dir` or a configurable output path.

### 3.3 `src/finetuning/postproc_models.py`

- **Line 3:** `df_path = '/gpfs/commons/.../index/models.csv'`  
  → Same as index: e.g. `index/models.csv` relative to repo root or from env/config.

### 3.4 `src/evaluation/pipeline/compute_risk.py`

- Hydra decorator: `config_path="../../configs/evaluation/log_likelihood"` is correct relative to `src/evaluation/pipeline/`; no change if you keep the same `src/` layout.
- No other hardcoded paths in the snippet; relies on `cfg.output_dir` and denominators/FolderHandler.

### 3.5 `src/evaluation/pipeline/compute_risk_batch.py`

- **Line 16:** `from names_dataset import NameDataset, NameWrapper` — only used in commented `prepare_last_name`.  
  → Either remove this import (and add a note in README that `prepare_last_name` needs `names_dataset` if re-enabled) or keep and add `names-dataset` to `pyproject.toml`.
- **Lines 264–266, 271–274:** Hardcoded model paths for `base == 'scratch'` and `base` (e.g. `Llama_3.2-1B`, `Llama_3.1-8B`).  
  → Replace with values from Hydra config (e.g. `cfg.base_model_1B`, `cfg.base_model_8B`) or from `FolderHandler` / a small config file.

### 3.6 `src/configs/evaluation/log_likelihood/eval.yaml`

- Replace absolute `output_dir` and any `/gpfs/commons/...` paths with relative or default paths, e.g.  
  `output_dir: ./outputs/pipeline`  
  and document that users can override via Hydra.

### 3.7 `src/llm/gemini/utils.py`

- **Lines 11–14:** `GOOGLE_CLOUD_PROJECT`, `GOOGLE_CLOUD_LOCATION`, `GOOGLE_GENAI_USE_VERTEXAI` — keep as env vars; do not hardcode project/lab names in the anonymized repo.
- **Line 34:** Usage log path `/gpfs/commons/.../gemini_usage.txt`  
  → Use env var (e.g. `GEMINI_USAGE_LOG`) or a path under `./outputs/` or disable in the export.

### 3.8 `src/evaluation/pipeline/experimental/config_helper.py`

- `get_base_model(config)` uses `config['models']['base_model_1B']` and `base_model_8B`.  
  → Ensure at least one sample YAML under `experimental/` (or referenced by your scripts) defines these keys with placeholder paths so the code runs; document in README.

---

## 4. Imports to adjust (if you flatten or rename packages)

If you keep the layout above, no import renames are needed. If you move code:

- All `from src.folder_handler import FolderHandler` stay as-is.
- All `from src.evaluation.exploration.denominators import select_datasets, read_dataset` stay as-is.
- All `from src.evaluation.pipeline.experimental.config_loader` / `config_helper` and `experimental.mia.name_filter` stay as-is.
- `pii_injection.py`: keep `from src.llm import call_llm`.
- `finetune.py`: keep `from utils import ...` and `from src.folder_handler import FolderHandler`.

So the only edits are path/config changes above, not package renames.

---

## 5. Index and config placeholders

- **index/:**  
  Add minimal `index/datasets.csv` and `index/models.csv` with the columns that `FolderHandler` and denominators expect (see `folder_handler.py` and `denominators.select_datasets`). You can use 1–2 placeholder rows so that `load_datasets()` / `load_models()` and `select_datasets()` don’t fail when someone runs the pipeline.
- **eval.yaml:**  
  As above: relative `output_dir`, and optional `base_model_1B` / `base_model_8B` (or equivalent) if you use them in the evaluation scripts.
- **experimental:**  
  If any script expects `config.yaml` in `experimental/`, add a minimal `config.yaml` or point `DEFAULT_CONFIG_PATH` in `config_loader.py` to an existing YAML under `configs/`.

---

## 6. Dependencies (pyproject.toml)

From the original `pyproject.toml`, keep only what the three parts need:

- **PII insertion:** pandas, tqdm, loguru, natsort, Faker, langcodes, json_repair, flask, termcolor; optional: google-genai (Gemini), requests (vLLM).
- **Finetuning:** torch, transformers, peft, bitsandbytes, accelerate, deepspeed, wandb, tqdm.
- **Evaluation:** pandas, numpy, tqdm, torch, transformers, hydra-core, omegaconf, scipy, scikit-learn, pyarrow; optional: names-dataset if you keep `prepare_last_name` and the import in `compute_risk_batch.py`.

Remove or don’t list: presidio, spacy, pyap, stanza, vllm (unless you want to run vLLM in the anonymized repo), and any other unused libs.  
Anonymize `authors`, `urls`, and project name/description in `pyproject.toml`.

---

## 7. Checklist before you finalize

- [x] Create `verified-extraction-audit` and copy all files per sections 2.1–2.4.
- [x] Apply path changes in section 3 (folder_handler, denominators, postproc_models, compute_risk_batch, eval.yaml, gemini utils).
- [x] Add `index/` with minimal CSVs and, if needed, a sample `config.yaml` for experimental scripts.
- [x] Add/trim `pyproject.toml` and ensure `pip install -e .` works.
- [ ] Write README: how to run data preprocessing, finetuning, and evaluation; required env vars (e.g. index folder, Gemini usage log, Google Cloud env vars); optional `prepare_last_name` / names_dataset.
- [ ] Run: data preprocessing (one script), finetuning (one job), evaluation (compute_risk or compute_risk_batch) and fix any remaining path/import errors.

**Run from repo root:** Use `PYTHONPATH=.` or `pip install -e .` so that `src` resolves to this repo (e.g. `cd verified-extraction-audit && PYTHONPATH=. python -m src.evaluation.pipeline.compute_risk`).

---

## 8. Summary table: paths to change

| File | What to change |
|------|----------------|
| `src/folder_handler.py` | `folder_default` → relative or env; optional: strip or relativize paths in `build_index_*` / `update_persona_path` |
| `src/evaluation/exploration/denominators.py` | `datasets_index_path`, `path_tags`, `path_value`, output CSV path → configurable/relative |
| `src/finetuning/postproc_models.py` | `df_path` → e.g. `index/models.csv` or env |
| `src/evaluation/pipeline/compute_risk_batch.py` | Hardcoded base model paths → config; optionally drop `names_dataset` import |
| `src/configs/evaluation/log_likelihood/eval.yaml` | `output_dir` and any absolute paths → relative/defaults |
| `src/llm/gemini/utils.py` | Usage log path → env or `./outputs`; keep Google env vars (no lab names in repo) |

After that, you can focus on the README and a quick end-to-end run to confirm the anonymized repo runs as intended.
