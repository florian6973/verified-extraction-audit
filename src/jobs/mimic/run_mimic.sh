#!/usr/bin/env bash
# Run the full audit on a MIMIC-IV subset (default 1% of subjects).
#
#   subsample discharge.csv -> ingest -> inject (LLM classify) -> train
#   -> generate completions -> audit
#
# MIMIC discharge notes already contain ___, so no admissions/patients tables and
# no full fake_persona build are needed. Direct-mode training (no index/Hydra).
#
# Configure via environment variables (defaults below). The LLM classifier talks
# to any OpenAI-compatible server (your local vLLM/llama.cpp/…) via API_BASE.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="${REPO:-$(cd "$SCRIPT_DIR/../../.." && pwd)}"
cd "$REPO"

PYTHON="${PYTHON:-python}"
WORK="${WORK:-$REPO/outputs/mimic}"

# --- data ---
DISCHARGE="${DISCHARGE:-data/raw/discharge.csv}"
FRAC="${FRAC:-0.01}"                 # fraction of SUBJECTS to keep
VAL_FRAC="${VAL_FRAC:-0.2}"          # held-out non-member subjects
DI_TYPE="${DI_TYPE:-name}"
DI_RATE="${DI_RATE:-0.01}"           # direct-identifier rate (Table-4 rows)

# --- blank classifier ---
CLASSIFIER="${CLASSIFIER:-llm}"      # llm (one call/note) | label (offline heuristic) | first (offline, DI in 1st blank)
# --- LLM classifier (OpenAI-compatible; e.g. your local vLLM) -- only used when CLASSIFIER=llm ---
API="${API:-openai}"                 # openai | vllm | gemini | mock
API_BASE="${API_BASE:-http://localhost:8000/v1}"
MODEL="${MODEL:-local}"
# API_KEY optional -> exported OPENAI_API_KEY or --api-key
NO_THINK="${NO_THINK:-}"             # set to 1 for reasoning models (disable thinking)
LLM_MAX_TOKENS="${LLM_MAX_TOKENS:-}" # raise if a reasoning model needs more room

# --- training / eval ---
BASE_MODEL="${BASE_MODEL:-models/base/Llama_3.2-1B}"
N_EPOCHS="${N_EPOCHS:-3}"
K="${K:-100000}"                     # attacker-query completions
BUDGETS="${BUDGETS:-1e5 1e6}"
AUDIT_ONLY="${AUDIT_ONLY:-}"         # set to 1 to skip steps 1-5 and re-audit existing $WORK artifacts

export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
export DATA_ROOT="$WORK/processed"
# finetune.py logs to W&B: run `wandb login` first or set WANDB_MODE=offline.
# Do NOT set WANDB_DISABLED=true.
mkdir -p "$WORK"

if [ -n "$AUDIT_ONLY" ]; then
  echo "==== [audit-only] skipping steps 1-5; re-auditing existing artifacts in $WORK ===="
else

echo "==== [1/6] subsample MIMIC discharge notes (${FRAC}) ===="
$PYTHON -m src.dataset.prepare.mimic_subset \
    --discharge "$DISCHARGE" --out "$WORK/mimic_subset.parquet" --frac "$FRAC"

echo "==== [2/6] ingest -> splits + synthetic personas ===="
$PYTHON -m src.dataset.prepare.ingest \
    --input "$WORK/mimic_subset.parquet" --name mimic \
    --out-root "$DATA_ROOT" --version 8 --val-frac "$VAL_FRAC"

echo "==== [3/6] inject direct identifiers (classifier=$CLASSIFIER -> fill by note_id) ===="
$PYTHON -m src.dataset.prepare.inject \
    --splits-root "$DATA_ROOT" --version 8 --classifier "$CLASSIFIER" \
    --api "$API" --api-base "$API_BASE" --model "$MODEL" ${API_KEY:+--api-key "$API_KEY"} \
    ${NO_THINK:+--llm-no-think} ${LLM_MAX_TOKENS:+--llm-max-tokens "$LLM_MAX_TOKENS"} \
    --di-type "$DI_TYPE" --di-rate "$DI_RATE" \
    --output-sft "$DATA_ROOT/sft" --emit-labeled "$WORK/labeled.parquet"
SFT_TRAIN="$(ls "$DATA_ROOT"/sft/train_*.json | head -1)"
echo "SFT train: $SFT_TRAIN"

echo "==== [4/6] train (direct; no index) ===="
$PYTHON "$REPO/src/finetuning/finetune.py" \
    --model_name_or_path "$BASE_MODEL" \
    --dataset_path "$SFT_TRAIN" \
    --output_dir "$WORK/finetuned" --n_epochs "$N_EPOCHS"

echo "==== [5/6] generate completions ===="
$PYTHON -m src.evaluation.pipeline.generate_completions \
    --model-path "$WORK/finetuned" --output "$WORK/completions.parquet" \
    --prompt "Name: " --k "$K" --max-new-tokens 20

fi   # end steps 1-5 (skipped when AUDIT_ONLY=1)

echo "==== [6/6] audit (verifier + extracted-stream FPR + theory & experimental) ===="
$PYTHON -m src.evaluation.audit.from_labels \
    --labeled "$WORK/labeled.parquet" \
    --base-model "$BASE_MODEL" --finetuned-model "$WORK/finetuned" \
    --di-type "$DI_TYPE" --budgets $BUDGETS \
    --generations "$WORK/completions.parquet" --output-dir "$WORK/audit"

echo "==== MIMIC RUN COMPLETE ===="
echo "Audit report: $WORK/audit/audit_report.json"
