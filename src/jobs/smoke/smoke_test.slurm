#!/bin/bash
# SLURM smoke test: run the full audit pipeline on a synthetic dataset on one GPU.
#
# Submit from the repo root:
#     sbatch src/jobs/smoke/smoke_test.slurm
#
# Everything is self-contained (synthetic data + tiny model), so the job needs
# no MIMIC data, no API keys, and no base-model download. Adjust the SBATCH
# directives (partition, account, time) to your cluster, and the CONDA_ENV /
# CUDA_MODULE variables below to your environment.
#
# NOTE: finetune.py trains in bf16, which requires an Ampere-or-newer GPU
# (A100/H100/L4/RTX30+). On clusters with mixed hardware, pin the architecture,
# e.g. add:  #SBATCH --constraint=a100   (older V100/T4/P100 will error on bf16).
#
#SBATCH --job-name=audit-smoke
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --partition=gpu
# Logs go to the submission directory (which always exists); SLURM opens these
# BEFORE the script body runs, so they must not point at a not-yet-created dir.
#SBATCH --output=audit-smoke_%j.out
#SBATCH --error=audit-smoke_%j.err

set -euo pipefail

# --- environment (edit to match your cluster) ---
CONDA_ENV="${CONDA_ENV:-clinical-exposure}"
CUDA_MODULE="${CUDA_MODULE:-cuda/12.4}"

# Under SLURM the batch script runs from a spool copy, so BASH_SOURCE/$0 are NOT
# the repo path — use SLURM_SUBMIT_DIR (the dir you ran `sbatch` from = repo root).
# Submit from the repo root: `cd <repo> && sbatch src/jobs/smoke/smoke_test.slurm`.
REPO="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/../../.." && pwd)}"
export REPO
cd "$REPO"

# Best-effort module + conda activation (skip cleanly if unavailable).
if command -v module >/dev/null 2>&1; then module load "$CUDA_MODULE" || true; fi
if command -v conda  >/dev/null 2>&1; then
    # shellcheck disable=SC1091
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate "$CONDA_ENV"
fi

export DS_SKIP_CUDA_CHECK=1
export TOKENIZERS_PARALLELISM=false

echo "Repo:   $REPO"
echo "Python: $(command -v python)"
nvidia-smi || true

# All pipeline logic lives in run_smoke.sh so it can also be run interactively
# (e.g. on a login node with a GPU) without SLURM.
bash "$REPO/src/jobs/smoke/run_smoke.sh"
