#!/bin/bash
# SLURM: run the audit on a MIMIC-IV subset (default 1% of subjects) on one GPU.
#
#     cd <repo-root>
#     sbatch --export=ALL,DISCHARGE=data/raw/discharge.csv,API_BASE=http://<host>:<port>/v1,MODEL=<served-model> \
#            src/jobs/mimic/mimic_test.slurm
#
# All knobs are environment variables (see src/jobs/mimic/run_mimic.sh): FRAC,
# DI_RATE, BASE_MODEL, N_EPOCHS, K, API/API_BASE/MODEL/API_KEY, WORK, VAL_FRAC.
# Requires PhysioNet MIMIC-IV access (discharge.csv) and an OpenAI-compatible LLM
# server reachable from the compute node (your local vLLM etc.).
#
#SBATCH --job-name=audit-mimic
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --partition=gpu
# Training uses bf16 -> Ampere+ GPU (A100/H100/L4/RTX30+); pin with --constraint if mixed.
#SBATCH --output=audit-mimic_%j.out
#SBATCH --error=audit-mimic_%j.err

set -euo pipefail

CONDA_ENV="${CONDA_ENV:-clinical-exposure}"
CUDA_MODULE="${CUDA_MODULE:-cuda/12.4.0}"

REPO="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/../../.." && pwd)"
cd "$REPO"

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

bash "$REPO/src/jobs/mimic/run_mimic.sh"
