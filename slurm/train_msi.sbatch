#!/bin/bash -l
#SBATCH --job-name=tps_train
#SBATCH --partition=msigpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --tmp=16G
#SBATCH --array=0-2
#SBATCH --output=%x_%A_%a.out
#SBATCH --error=%x_%A_%a.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=lind1775@umn.edu

# Single-GPU training of the time-conditioned FNO, one array task per forcing
# representation:
#   0 -> summary          (4 broadcast flux statistics)
#   1 -> temporal_global  (one global embedding of the flux history)
#   2 -> temporal_local   (per-lateral-row embeddings of the flux history)
#
# There is no DDP path in this codebase and none is wanted here; the three
# tasks are fully independent single-GPU jobs.
#
# Usage (submit from the repository root):
#   sbatch slurm/train_msi.sbatch
#   sbatch --array=0 slurm/train_msi.sbatch          # summary only
#   sbatch slurm/train_msi.sbatch --set training.learning_rate=2.0e-4
#
# Trailing arguments are forwarded verbatim to `fno-tps train`.
#
# Environment overrides:
#   PROJECT_DIR       repository checkout   (default: $SLURM_SUBMIT_DIR)
#   DATA_DIR          dataset directory     (default: $PROJECT_DIR/data/production-36x48)
#   CONFIG            study config          (default: conf/nonlinear-production-36x48.yaml)
#   PERSISTENT_RUNS   result destination    (default: $HOME/tps_runs)
#   TPS_ENV_PREFIX    conda environment     (default: $HOME/.conda/envs/tps)
#   EPOCHS            override config epochs

set -euo pipefail

REPRESENTATIONS=(summary temporal_global temporal_local)
TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
if (( TASK_ID < 0 || TASK_ID >= ${#REPRESENTATIONS[@]} )); then
    echo "ERROR: array index $TASK_ID has no representation mapping." >&2
    exit 1
fi
REPRESENTATION="${REPRESENTATIONS[$TASK_ID]}"

# ──── Paths ────
PROJECT_DIR="${PROJECT_DIR:-${SLURM_SUBMIT_DIR:-}}"
if [[ -z "$PROJECT_DIR" || ! -f "$PROJECT_DIR/pyproject.toml" ]]; then
    echo "ERROR: PROJECT_DIR must point to the repository checkout." >&2
    exit 1
fi
DATA_DIR="${DATA_DIR:-$PROJECT_DIR/data/production-36x48}"
CONFIG="${CONFIG:-conf/nonlinear-production-36x48.yaml}"
PERSISTENT_RUNS="${PERSISTENT_RUNS:-$HOME/tps_runs}"
TPS_ENV_PREFIX="${TPS_ENV_PREFIX:-$HOME/.conda/envs/tps}"

cd "$PROJECT_DIR"

if [[ ! -f "$DATA_DIR/manifest.json" ]]; then
    echo "ERROR: no manifest.json under DATA_DIR=$DATA_DIR." >&2
    exit 1
fi
if [[ ! -f "$CONFIG" ]]; then
    echo "ERROR: config not found: $CONFIG" >&2
    exit 1
fi

# `fno-tps train` aborts unless manifest.study_config_sha256 equals the hash of
# the config passed here, so conf/ must be transferred byte-identically with
# the dataset. Failing early with a readable message beats a stack trace.

# ──── Environment ────
module purge
module load miniforge
eval "$(conda shell.bash hook)"
if [[ ! -x "$TPS_ENV_PREFIX/bin/python" ]]; then
    echo "ERROR: MSI environment not found at $TPS_ENV_PREFIX." >&2
    echo "Run: PROJECT_DIR=$PROJECT_DIR bash slurm/setup_env_msi.sh" >&2
    exit 1
fi
source activate "$TPS_ENV_PREFIX"

# ──── Node-local scratch ────
# trajectories.npy is memory-mapped and read in random (case, time) order, so
# it is staged off shared storage onto the node-local disk.
LOCAL_ROOT="/tmp/tps_${SLURM_ARRAY_JOB_ID:-$SLURM_JOB_ID}_${TASK_ID}"
LOCAL_DATA="$LOCAL_ROOT/data"
LOCAL_RUN="$LOCAL_ROOT/run"
mkdir -p "$LOCAL_DATA" "$LOCAL_RUN" "$PERSISTENT_RUNS"
trap 'rm -rf "$LOCAL_ROOT"' EXIT

cp -a "$DATA_DIR"/. "$LOCAL_DATA"/
if [[ ! -f "$LOCAL_DATA/SHA256SUMS.txt" ]]; then
    echo "ERROR: dataset has no SHA256SUMS.txt integrity manifest." >&2
    exit 1
fi
(
    cd "$LOCAL_DATA"
    sha256sum -c SHA256SUMS.txt
)

# ──── Logging ────
echo "Job ${SLURM_ARRAY_JOB_ID:-$SLURM_JOB_ID} task $TASK_ID started on $(hostname) at $(date)"
echo "REPRESENTATION=$REPRESENTATION"
echo "PROJECT_DIR=$PROJECT_DIR"
echo "CONFIG=$CONFIG"
echo "DATA_DIR=$DATA_DIR -> $LOCAL_DATA"
echo "PERSISTENT_RUNS=$PERSISTENT_RUNS"
echo "TPS_ENV_PREFIX=$TPS_ENV_PREFIX"
nvidia-smi
python -c "import torch; assert torch.cuda.is_available(), 'CUDA is unavailable in the training job'; print('torch', torch.__version__, 'CUDA', torch.version.cuda, 'GPU', torch.cuda.get_device_name(0))"

# ──── Run ────
# One worker per allocated core, minus the main process. Input tensors are
# built in NumPy per sample, so this is what keeps the A100 fed.
NUM_WORKERS=$(( ${SLURM_CPUS_PER_TASK:-1} - 1 ))
(( NUM_WORKERS < 0 )) && NUM_WORKERS=0

EPOCH_ARGS=()
if [[ -n "${EPOCHS:-}" ]]; then
    EPOCH_ARGS=(--epochs "$EPOCHS")
fi

srun fno-tps train \
    --config "$CONFIG" \
    --data "$LOCAL_DATA" \
    --run-dir "$LOCAL_RUN" \
    --representation "$REPRESENTATION" \
    --device cuda \
    --num-workers "$NUM_WORKERS" \
    "${EPOCH_ARGS[@]}" \
    "$@"

# ──── Copy results back to persistent storage ────
DESTINATION="$PERSISTENT_RUNS/${SLURM_ARRAY_JOB_ID:-$SLURM_JOB_ID}_${REPRESENTATION}"
mkdir -p "$DESTINATION"
cp -a "$LOCAL_RUN"/. "$DESTINATION"/
echo "Results copied to $DESTINATION"

echo "Task $TASK_ID finished at $(date)"
