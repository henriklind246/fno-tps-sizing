#!/bin/bash -l
#SBATCH --job-name=tps_datagen
#SBATCH --partition=msismall
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=7G
#SBATCH --time=12:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=lind1775@umn.edu

# Generates the mesh-refined production dataset (36x48) with `fno-tps generate`.
#
# This is a CPU job. `generate_dataset` runs one serial loop over the 972
# sampled cases and the linear solves go through SuperLU (`splu`), which is
# single-threaded, so no GPU is requested and extra cores buy nothing. The two
# allocated cores are the work plus headroom; BLAS threading is pinned to 1
# below so a shared msismall node is not oversubscribed.
#
# Usage (submit from the repository root):
#   sbatch slurm/generate_data_msi.sbatch
#   CONFIG=conf/nonlinear-pilot.yaml sbatch slurm/generate_data_msi.sbatch
#
# Trailing arguments are forwarded verbatim to `fno-tps generate`. Note that
# `--set` overrides change the effective config hash recorded in the manifest,
# and training rejects any config whose hash does not match, so any `--set`
# used here must be repeated identically on the training job.
#
# Environment overrides:
#   PROJECT_DIR   repository checkout   (default: $SLURM_SUBMIT_DIR)
#   CONFIG        study config          (default: conf/nonlinear-production-36x48.yaml)
#   OUTPUT_DIR    dataset destination   (default: $PROJECT_DIR/data/production-36x48)
#   TPS_ENV_PREFIX conda environment    (default: $HOME/.conda/envs/tps)
#
# Runtime and sizing basis. A five-case production-physics check measured
# roughly 16-25 s/case at 36x48 and 45-87 s/case at 72x96 on a laptop core.
# Actual MSI time depends on the assigned CPU:
#
#   mesh     checked s/case   972-case projection   trajectories.npy
#   36x48       16-25 s             4.3-6.8 h            1.42 GB
#   72x96       45-87 s            12.2-23.5 h            5.70 GB
#
# The 12 h request gives the selected 36x48 run substantial margin. This is
# deliberate:
# `generate_dataset` writes nothing until every case has finished
# (data.py:330-349), so a walltime kill discards the entire run.
#
# To promote this to 72x96 later, copy the config, change the four mesh
# numbers, and submit with CONFIG=, at least 12G memory, and a longer walltime.

set -euo pipefail

# ──── Paths ────
PROJECT_DIR="${PROJECT_DIR:-${SLURM_SUBMIT_DIR:-}}"
if [[ -z "$PROJECT_DIR" || ! -f "$PROJECT_DIR/pyproject.toml" ]]; then
    echo "ERROR: PROJECT_DIR must point to the repository checkout." >&2
    exit 1
fi
CONFIG="${CONFIG:-conf/nonlinear-production-36x48.yaml}"
OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT_DIR/data/production-36x48}"
TPS_ENV_PREFIX="${TPS_ENV_PREFIX:-$HOME/.conda/envs/tps}"

cd "$PROJECT_DIR"

if [[ ! -f "$CONFIG" ]]; then
    echo "ERROR: config not found: $CONFIG" >&2
    exit 1
fi

# Generate into a job-specific sibling and promote it only after the solver
# and checksum verification succeed. A killed job therefore cannot look like
# a complete production dataset.
STAGING_DIR="${OUTPUT_DIR}.incomplete-${SLURM_JOB_ID:-manual}"
if [[ -e "$OUTPUT_DIR" ]]; then
    echo "ERROR: OUTPUT_DIR already exists: $OUTPUT_DIR" >&2
    echo "Move it aside or select a new OUTPUT_DIR." >&2
    exit 1
fi
if [[ -e "$STAGING_DIR" ]]; then
    echo "ERROR: staging directory already exists: $STAGING_DIR" >&2
    exit 1
fi

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

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

# ──── Logging ────
echo "Job $SLURM_JOB_ID started on $(hostname) at $(date)"
echo "PROJECT_DIR=$PROJECT_DIR"
echo "CONFIG=$CONFIG"
echo "OUTPUT_DIR=$OUTPUT_DIR"
echo "STAGING_DIR=$STAGING_DIR"
echo "TPS_ENV_PREFIX=$TPS_ENV_PREFIX"
echo "Extra generator args: $*"

# ──── Run ────
# There is no per-case progress output, so this log stays quiet until the run
# completes; `tail -f` will show nothing in the meantime. Use `squeue` to
# confirm the job is still alive.
mkdir -p "$(dirname "$OUTPUT_DIR")"
srun fno-tps generate \
    --config "$CONFIG" \
    --output "$STAGING_DIR" \
    "$@"

(
    cd "$STAGING_DIR"
    sha256sum \
        bond_conductivity.npy \
        cases.jsonl \
        dx.npy \
        manifest.json \
        max_hot_face_temperature.npy \
        splits.npz \
        times.npy \
        trajectories.npy \
        trajectory_diagnostics.jsonl \
        x_centers.npy \
        y_centers.npy \
        > SHA256SUMS.txt
    sha256sum -c SHA256SUMS.txt
)
mv "$STAGING_DIR" "$OUTPUT_DIR"

echo "Job $SLURM_JOB_ID finished at $(date)"
ls -lh "$OUTPUT_DIR"

# The training job verifies manifest.study_config_sha256 against the hash of
# the config it is given, so the same CONFIG must be passed to both:
#   CONFIG=$CONFIG DATA_DIR=$OUTPUT_DIR sbatch slurm/train_msi.sbatch
echo
echo "Next: CONFIG=$CONFIG DATA_DIR=$OUTPUT_DIR sbatch slurm/train_msi.sbatch"
