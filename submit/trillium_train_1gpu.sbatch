#!/usr/bin/env bash
# Single-GPU Trillium launcher. Trillium rejects --gres and --mem, and compute
# nodes treat /home as read-only, so Slurm logs go under /scratch/$USER.
# Prepare data on the login node first; compute jobs run W&B offline.
#SBATCH --job-name=main_1gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-gpu=16
#SBATCH --time=01:00:00
#SBATCH --account=def-ssfels
#SBATCH --mail-user=nima.ashjaee@ubc.ca
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --output=/scratch/%u/nanopath/slurm/%j.out
#SBATCH --error=/scratch/%u/nanopath/slurm/%j.err
#SBATCH --signal=B:USR1@900
set -euo pipefail

REPO_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
CONFIG_FILE="${1:-${REPO_DIR}/configs/cc-main.yaml}"
[[ $# -gt 0 ]] && shift

cd "${REPO_DIR}"
module load python/3.12
VENV_DIR="${REPO_DIR}/.venv"
if [[ ! -f "${VENV_DIR}/bin/activate" ]]; then
  VENV_DIR="${HOME}/nanopath/.venv"
fi
source "${VENV_DIR}/bin/activate"
BASE="/scratch/$(id -un)/nanopath"
export HF_HOME="${BASE}/cache/hf"
export TORCH_HOME="${BASE}/cache/torch"
export WANDB_CACHE_DIR="${BASE}/wandb/cache"
export WANDB_CONFIG_DIR="${BASE}/wandb/config"
export WANDB_DATA_DIR="${BASE}/wandb/data"
export WANDB_MODE=offline
mkdir -p "${BASE}/wandb/cache" "${BASE}/wandb/config" "${BASE}/wandb/data"

echo "CONFIG_FILE=${CONFIG_FILE}"
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-local}"
echo "SLURMD_NODENAME=${SLURMD_NODENAME:-local}"
echo "SLURM_CPUS_PER_GPU=${SLURM_CPUS_PER_GPU:-unset}"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv || true

python train.py "${CONFIG_FILE}" "$@" &
PY_PID=$!
trap 'kill -USR1 "$PY_PID" 2>/dev/null || true' USR1
while true; do
  wait "$PY_PID" && break
  status=$?
  kill -0 "$PY_PID" 2>/dev/null || exit "$status"
done
