#!/bin/bash
#SBATCH --job-name=exp_76
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=1-00:00:00
#SBATCH --output=/gpfs/commons/groups/gursoy_lab/fpollet/Git/clinical-exposure-metric/src/jobs/logs/experimental_recall_%j.out
#SBATCH --error=/gpfs/commons/groups/gursoy_lab/fpollet/Git/clinical-exposure-metric/src/jobs/logs/experimental_recall_%j.err
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=100G
#SBATCH --exclude=ne1dg6-001

source /gpfs/commons/home/fpollet/.bashrc
conda activate clinical-exposure
module load cuda/12.4.0
cd /gpfs/commons/groups/gursoy_lab/fpollet/Git/clinical-exposure-metric/

# Set environment variable to skip CUDA check
export DS_SKIP_CUDA_CHECK=1
export CUDA_VISIBLE_DEVICES=0
export PYTHONUNBUFFERED=1

# Print some information about the job
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Start time: $(date)"
echo "Working directory: $(pwd)"

# Run the compute_risk.py script
# The script uses Hydra, so config can be overridden via command line if needed
python src/evaluation/pipeline/experimental_recall_2.py

echo "End time: $(date)"





