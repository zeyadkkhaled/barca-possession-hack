#!/bin/bash
# Sourced by every job script. Layout: ~/projects/m1-possession/{code,data,runs,final}, ~/envs/m1, ~/logs, ~/scratch.
# Uses only bash builtins, conda and python (no df / head / other uncommon tools).
set -eo pipefail
ROOT="$HOME/projects/m1-possession"
CODE="$ROOT/code"
DATA="$ROOT/data"
RUNS="$ROOT/runs"
FINAL="$ROOT/final"
ENV="$HOME/envs/m1"
CONDA_SH="${M1_CONDA_SH:-/nfs/slurm/conda/etc/profile.d/conda.sh}"
BOOT_PY="${M1_BOOT_PY:-/nfs/slurm/conda/bin/python}"     # conda's base Python: runs the stdlib-only bookkeeping before ~/envs/m1 exists
mkdir -p "$RUNS" "$FINAL" "$HOME/logs" "$HOME/scratch"
export PYTHONPATH="$CODE"
export PYTHONUNBUFFERED=1
export PYTHONIOENCODING=utf-8
export TMPDIR="$HOME/scratch"
export CONDA_PKGS_DIRS="$HOME/scratch/conda_pkgs"
export PIP_CACHE_DIR="$HOME/scratch/pip_cache"

activate_env() {
    # shellcheck disable=SC1090
    source "$CONDA_SH"
    set +u
    conda activate "$ENV"
}

PY() { python "$@"; }

# GPU-hour ledger: every job adds its wall time when it ends (also on failure).
# Call `ledger_on STEP GPU(0/1)` right after activating the environment.
ledger_on() {
    _STEP="$1"; _GPU="$2"
    trap 'python -m m1pc.cycle ledger "$_STEP" "${SLURM_JOB_ID:-local}" "$SECONDS" "$_GPU" || true' EXIT
}

banner() {
    echo "=== $1 | job ${SLURM_JOB_ID:-local} on ${SLURMD_NODENAME:-$HOSTNAME} | $(date '+%Y-%m-%d %H:%M:%S') ==="
    python -c "import torch; print('torch', torch.__version__, '| cuda', torch.cuda.is_available(), '|', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu')" || true
}
