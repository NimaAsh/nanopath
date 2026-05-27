#!/usr/bin/env bash
#SBATCH --job-name=main_1gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --account=def-ssfels
#SBATCH --mail-user=nima.ashjaee@ubc.ca
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --output=slurm/%j.out
#SBATCH --error=slurm/%j.err
#SBATCH --signal=B:USR1@900
set -euo pipefail

# Single-GPU Compute Canada launcher. Pass --account=... to sbatch if your
# cluster account is not the default. Compute nodes are offline on Rorqual, so
# prepare.py download=True must run on the login node before this job starts.
# Optional first arg overrides the YAML config (defaults to cc-main.yaml). Any
# extra args after that are forwarded to train.py as overrides, e.g.
#   sbatch --account=def-... submit/cc_train_1gpu.sbatch configs/cc-main.yaml output_dir=/scratch/$USER/nanopath/runs/main/myrun
# Logs land in repo-local slurm/<jobid>.{out,err}.
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
BASE="/scratch/${USER}/nanopath"
export HF_HOME="${BASE}/cache/hf"
export TORCH_HOME="${BASE}/cache/torch"
export WANDB_MODE=offline

echo "CONFIG_FILE=${CONFIG_FILE}"
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-local}"
echo "SLURMD_NODENAME=${SLURMD_NODENAME:-local}"
echo "SLURM_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK:-unset}"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv || true

# SLURM sends USR1 to this batch shell; forward it to the python child so
# train.py saves, probes, and exits cleanly before wall-clock expiry.
python train.py "${CONFIG_FILE}" "$@" &
PY_PID=$!
trap 'kill -USR1 "$PY_PID" 2>/dev/null || true' USR1
while true; do
  wait "$PY_PID" && break
  status=$?
  kill -0 "$PY_PID" 2>/dev/null || exit "$status"
done
