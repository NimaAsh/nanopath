#!/usr/bin/env bash
# Alliance single-GPU launcher for Compute Canada clusters.
# Run it directly from a login node:
#   ./submit/cc_train_1gpu.sbatch
#   ./submit/cc_train_1gpu.sbatch configs/cc-main-jepa-mask10.yaml output_dir=/scratch/$USER/nanopath/main-leopard/myrun
# It self-submits with cluster-specific GPU flags, then runs train.py.
#SBATCH --job-name=nanopath
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --account=def-ssfels
#SBATCH --output=/scratch/%u/nanopath-%j.out
#SBATCH --error=/scratch/%u/nanopath-%j.err
set -euo pipefail

SCRIPT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/$(basename "${BASH_SOURCE[0]}")"
REPO_DIR="${SLURM_SUBMIT_DIR:-$(dirname "$(dirname "${SCRIPT}")")}"

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
    case "${CC_CLUSTER:-$(hostname -f)}" in
        *rorqual*)         FLAGS=(--gres=gpu:1      --cpus-per-task=16 --mem=128G --time=04:00:00) ;;
        *trillium*|*trig*) FLAGS=(--gpus-per-node=1 --cpus-per-gpu=16            --time=04:00:00) ;;
        *nibi*)            FLAGS=(--gres=gpu:h100:1 --cpus-per-task=16 --mem=128G --time=04:00:00) ;;
        *) echo "Unknown cluster CC_CLUSTER='${CC_CLUSTER:-}' / $(hostname -f); add a case branch in ${SCRIPT}." >&2; exit 1 ;;
    esac
    exec sbatch --export=ALL,NANOPATH_LAUNCHED=1 "${FLAGS[@]}" "${SCRIPT}" "$@"
fi

if [[ -z "${NANOPATH_LAUNCHED:-}" ]]; then
    echo "Run this directly, not via sbatch: ./submit/cc_train_1gpu.sbatch [config]" >&2
    exit 1
fi

CONFIG_FILE="${1:-${REPO_DIR}/configs/cc-main-jepa-mask10.yaml}"
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

echo "CC_CLUSTER=${CC_CLUSTER:-unknown} CONFIG_FILE=${CONFIG_FILE} SLURM_JOB_ID=${SLURM_JOB_ID} NODE=${SLURMD_NODENAME:-?}"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv

python train.py "${CONFIG_FILE}" "$@"
