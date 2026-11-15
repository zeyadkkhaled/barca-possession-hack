#!/bin/bash
#SBATCH --job-name=m1_setup_env
#SBATCH --time=01:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=5
#SBATCH --mem=16G
# Creates ~/envs/m1 (Python 3.12, PyTorch CUDA 12.6 build, PathCRF requirements). Skipped if ~/envs/m1 already works.
source "$HOME/projects/m1-possession/code/jobs/common.sh"
# shellcheck disable=SC1090
source "$CONDA_SH"
trap '"$BOOT_PY" -m m1pc.cycle ledger setup_env "${SLURM_JOB_ID:-local}" "$SECONDS" 1 || true' EXIT
echo "=== setup_env | job ${SLURM_JOB_ID:-local} | $(date '+%Y-%m-%d %H:%M:%S') ==="
CHECK="import torch, numpy, pandas, scipy, sklearn, tqdm; assert torch.cuda.is_available(); print('env OK:', torch.__version__, torch.cuda.get_device_name(0))"
if [ -x "$ENV/bin/python" ] && "$ENV/bin/python" -c "$CHECK"; then
    echo "~/envs/m1 already works: skipped"
    exit 0
fi
set -x
conda create -y -p "$ENV" python=3.12
"$ENV/bin/python" -m pip install --no-cache-dir --upgrade pip
"$ENV/bin/python" -m pip install --no-cache-dir torch==2.7.1 --index-url https://download.pytorch.org/whl/cu126
"$ENV/bin/python" -m pip install --no-cache-dir -r "$CODE/pathcrf/requirements.txt" torch_geometric
set +x
"$ENV/bin/python" -c "$CHECK"
"$ENV/bin/python" -c "import torch; print('torch.cuda.is_available() =', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0))"
