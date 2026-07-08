#!/usr/bin/env bash
# End-to-end smoke test of the audit pipeline on a fully synthetic dataset.
#
# Runs every stage on data it generates itself — no MIMIC, no Gemini/vLLM, no
# base-model download (a tiny Llama is built locally). Only training, generation
# and log-likelihood scoring use the GPU; everything else is CPU.
#
#   SCENARIO=1 (default): synthetic notes with ___ blanks -> ingest -> inject
#                         (deterministic) -> SFT + labeled set.
#   SCENARIO=2          : notes with identifiers already embedded + a labeled set
#                         (as a user would bring to audit a real corpus) -> SFT.
#
# Shared: build tiny model -> train (direct) -> generate completions -> audit.
# Override any knob via environment variables (see defaults below).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="${REPO:-$(cd "$SCRIPT_DIR/../../.." && pwd)}"
cd "$REPO"

PYTHON="${PYTHON:-python}"
SCENARIO="${SCENARIO:-1}"
WORK="${WORK:-$REPO/outputs/smoke/scenario$SCENARIO}"
N_SUBJECTS="${N_SUBJECTS:-60}"
# The tiny model is randomly initialized, so to *demonstrate* leakage it has to
# overfit the ~50 injected names. With grad_accum=1 + a higher lr, one epoch is
# ~50 optimizer steps, so ~30 epochs (~1.5k steps) memorizes them in seconds.
# (The paper defaults — lr 2e-5, grad_accum 8 — are for real pretrained models.)
N_EPOCHS="${N_EPOCHS:-30}"
LR="${LR:-1e-3}"
GRAD_ACCUM="${GRAD_ACCUM:-1}"
N_SAMPLES="${N_SAMPLES:-1000}"          # number of attacker-query completions
BUDGETS="${BUDGETS:-100 1000}"          # query budgets Q for the extraction curves
TOKENIZER="${TOKENIZER:-gpt2}"
DI_TYPE="${DI_TYPE:-name}"

export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
# finetune.py logs to Weights & Biases (report_to="wandb"); run `wandb login` first,
# or set WANDB_MODE=offline. Do NOT set WANDB_DISABLED=true (WandbCallback would raise).

mkdir -p "$WORK/data" "$WORK/models/base"
LABELED="$WORK/data/labeled.parquet"

echo "==== [scenario $SCENARIO] prepare data ===="
if [ "$SCENARIO" = "1" ]; then
    $PYTHON -m src.dataset.prepare.make_synthetic \
        --out "$WORK/data/notes.parquet" --n-subjects "$N_SUBJECTS" --seed 42
    $PYTHON -m src.dataset.prepare.ingest \
        --input "$WORK/data/notes.parquet" --name synthetic \
        --out-root "$WORK/processed" --version 8 --val-frac 0.5 --seed 42
    # classify each blank (offline label heuristic) + fill from persona (by note_id) + sample
    $PYTHON -m src.dataset.prepare.inject \
        --splits-root "$WORK/processed" --version 8 --classifier label \
        --di-type "$DI_TYPE" --di-rate "${DI_RATE:-1.0}" \
        --output-sft "$WORK/sft" --emit-labeled "$LABELED"
    SFT_TRAIN="$(ls "$WORK"/sft/train_*.json | head -1)"
else
    # notes already contain identifiers -> writes notes.parquet, labeled.parquet, sft/{train,val}.json
    $PYTHON -m src.dataset.prepare.make_synthetic --scenario 2 \
        --out "$WORK/data/notes.parquet" --n-subjects "$N_SUBJECTS" --seed 42
    SFT_TRAIN="$WORK/data/sft/train.json"
fi
echo "SFT train: $SFT_TRAIN | labeled: $LABELED"

echo "==== build tiny base model ===="
$PYTHON -m src.jobs.smoke.build_tiny_model \
    --out "$WORK/models/base/Llama_tiny" --tokenizer "$TOKENIZER"

echo "==== train (finetune.py, direct — no index) ===="
# Launch by file path (not -m) so `from utils import ...` resolves; PYTHONPATH gives `from src...`.
# finetune derives the val SFT path from --dataset_path (train -> val).
$PYTHON "$REPO/src/finetuning/finetune.py" \
    --model_name_or_path "$WORK/models/base/Llama_tiny" \
    --dataset_path "$SFT_TRAIN" \
    --output_dir "$WORK/models/finetuned" --n_epochs "$N_EPOCHS" \
    --learning_rate "$LR" --gradient_accumulation_steps "$GRAD_ACCUM"

echo "==== generate completions (attacker queries) ===="
$PYTHON -m src.evaluation.pipeline.generate_completions \
    --model-path "$WORK/models/finetuned" --output "$WORK/completions.parquet" \
    --k "$N_SAMPLES" --prompt "Name: " --pii-type "$DI_TYPE" \
    --batch-size 16 --max-new-tokens 20 \
    --model-size tiny --dataset-size 1 --pii-rate 1.0 --n-epochs "$N_EPOCHS"

echo "==== audit: verifier + extracted-stream FPR + theory & experimental curves ===="
$PYTHON -m src.evaluation.audit.from_labels \
    --labeled "$LABELED" \
    --base-model "$WORK/models/base/Llama_tiny" \
    --finetuned-model "$WORK/models/finetuned" \
    --di-type "$DI_TYPE" --budgets $BUDGETS \
    --generations "$WORK/completions.parquet" \
    --output-dir "$WORK/audit"

echo "==== SMOKE TEST (scenario $SCENARIO) COMPLETE ===="
echo "Artifacts under: $WORK"
echo "Audit report:    $WORK/audit/audit_report.json"
