#!/bin/bash -l
# One-time MSI environment setup for fno-tps-sizing.
# Usage: bash slurm/setup_env_msi.sh
#
# Creates a self-contained conda environment with the official CUDA 12.6
# PyTorch 2.7.0 wheel. PyTorch stopped publishing official conda binaries
# starting with 2.6, so installing pytorch==2.7.0 from the pytorch conda
# channel does not work.

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$HOME/fno-tps-sizing}"
TPS_ENV_PREFIX="${TPS_ENV_PREFIX:-$HOME/.conda/envs/tps}"

echo "=== Setting up MSI environment for fno-tps-sizing ==="
echo "PROJECT_DIR=$PROJECT_DIR"
echo "TPS_ENV_PREFIX=$TPS_ENV_PREFIX"

if [[ ! -f "$PROJECT_DIR/pyproject.toml" ]]; then
    echo "ERROR: $PROJECT_DIR does not look like the repository checkout." >&2
    exit 1
fi

module purge
module load miniforge
eval "$(conda shell.bash hook)"

# MSI recommends --copy so the environment does not retain links into the
# centrally managed miniforge module.
conda create --copy --prefix "$TPS_ENV_PREFIX" python=3.11 pip -y
source activate "$TPS_ENV_PREFIX"

# Official PyTorch 2.7.0 Linux wheel with its CUDA runtime. A100 nodes only
# need a compatible NVIDIA driver; a separately loaded CUDA module is not
# required for this project because it builds no custom CUDA extensions.
python -m pip install \
    torch==2.7.0 \
    --index-url https://download.pytorch.org/whl/cu126
python -m pip install \
    numpy==2.2.6 \
    scipy==1.15.3 \
    PyYAML==6.0.2

cd "$PROJECT_DIR"
python -m pip install -e . --no-deps

mkdir -p "$HOME/tps_runs"

echo "=== Environment ready. Activate with: source activate $TPS_ENV_PREFIX ==="
# cuda_available is normally false on a login node; train_msi.sbatch performs
# the actual GPU assertion after Slurm assigns an A100.
python -c "import torch; print('torch', torch.__version__, 'built_cuda', torch.version.cuda, 'cuda_available_here', torch.cuda.is_available())"
python -c "from fno_tps.cli import build_parser; build_parser(); print('fno_tps import OK')"
