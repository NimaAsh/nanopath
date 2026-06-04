#!/usr/bin/env bash
# Alliance/Trillium single-GPU launcher.
# Defaults to configs/cc-main.yaml; pass configs/cc-smoke.yaml for a smoke run.
#SBATCH --job-name=main_1gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-gpu=16
#SBATCH --time=04:00:00
#SBATCH --account=def-ssfels
#SBATCH --mail-user=nima.ashjaee@ubc.ca
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --output=/scratch/%u/nanopath-%j.out
#SBATCH --error=/scratch/%u/nanopath-%j.err
set -euo pipefail

REPO_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
CONFIG_FILE="${1:-${REPO_DIR}/configs/cc-main.yaml}"
[[ $# -gt 0 ]] && shift

cd "${REPO_DIR}"
module load python/3.12
source "${REPO_DIR}/.venv/bin/activate"

BASE="/scratch/$(id -un)/nanopath"
export HF_HOME="${BASE}/cache/hf"
export TORCH_HOME="${BASE}/cache/torch"
export WANDB_CACHE_DIR="${BASE}/wandb/cache"
export WANDB_CONFIG_DIR="${BASE}/wandb/config"
export WANDB_DATA_DIR="${BASE}/wandb/data"
export WANDB_MODE=offline
mkdir -p "${HF_HOME}" "${TORCH_HOME}" "${WANDB_CACHE_DIR}" "${WANDB_CONFIG_DIR}" "${WANDB_DATA_DIR}"

echo "CONFIG_FILE=${CONFIG_FILE}"
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-local}"
echo "SLURMD_NODENAME=${SLURMD_NODENAME:-local}"
echo "SLURM_CPUS_PER_GPU=${SLURM_CPUS_PER_GPU:-unset}"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv

python train.py "${CONFIG_FILE}" "$@"
