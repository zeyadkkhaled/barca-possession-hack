#!/bin/bash
# CPU job. Reuses ~/envs/m1 (M1's env: Python 3.12, PyTorch CUDA 12.6, PathCRF requirements) and adds what step 2b needs
# (lightgbm for the ball fill and C3-T, scipy for the Hungarian re-linking). Creates the env only if it is missing.
source "$HOME/projects/barca-step2/code/jobs/common.sh"
# shellcheck disable=SC1090
source "$CONDA_SH"
trap '"$BOOT_PY" -m pathcrf_adapter.cycle ledger setup_env "${SLURM_JOB_ID:-local}" "$SECONDS" 0 || true' EXIT
echo "=== setup_env | job ${SLURM_JOB_ID:-local} | $(date '+%Y-%m-%d %H:%M:%S') ==="
CHECK="import torch, numpy, scipy, lightgbm, torch_geometric; print('env OK: torch', torch.__version__, '| lightgbm', lightgbm.__version__, '| scipy', scipy.__version__)"
if [ ! -x "$ENV/bin/python" ]; then
    set -x
    conda create -y -p "$ENV" python=3.12
    "$ENV/bin/python" -m pip install --no-cache-dir --upgrade pip
    "$ENV/bin/python" -m pip install --no-cache-dir torch==2.7.1 --index-url https://download.pytorch.org/whl/cu126
    "$ENV/bin/python" -m pip install --no-cache-dir -r "$CODE/vendor/pathcrf/requirements.txt" torch_geometric
    set +x
fi
if ! "$ENV/bin/python" -c "$CHECK"; then
    "$ENV/bin/python" -m pip install --no-cache-dir "lightgbm==4.7.0" scipy
fi
"$ENV/bin/python" -c "$CHECK"
