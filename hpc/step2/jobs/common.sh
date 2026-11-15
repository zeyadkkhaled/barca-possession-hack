#!/bin/bash
# Sourced by every step-2b job. Layout: ~/projects/barca-step2/{code,built,runs,final}; the M1 PFF package stays in
# ~/projects/m1-possession/data (read only); env ~/envs/m1; logs ~/logs; temporary files ~/scratch.
set -eo pipefail
ROOT="${B2_ROOT:-$HOME/projects/barca-step2}"
CODE="$ROOT/code"
BUILT="$ROOT/built"
RUNS="$ROOT/runs"
FINAL="$ROOT/final"
PKG="${B2_PACKAGE:-$HOME/projects/m1-possession/data}"
ENV="${B2_ENV:-$HOME/envs/m1}"
CONDA_SH="${B2_CONDA_SH:-/nfs/slurm/conda/etc/profile.d/conda.sh}"
BOOT_PY="${B2_BOOT_PY:-/nfs/slurm/conda/bin/python}"     # conda base Python: stdlib-only bookkeeping before the env exists
mkdir -p "$RUNS" "$FINAL" "$BUILT" "$HOME/logs" "$HOME/scratch"
export B2_ROOT="$ROOT" B2_PACKAGE="$PKG"
export PYTHONPATH="$CODE/src"
export PYTHONUNBUFFERED=1
export PYTHONIOENCODING=utf-8
export PYTHONUTF8=1
export TMPDIR="$HOME/scratch"
export CONDA_PKGS_DIRS="$HOME/scratch/conda_pkgs"
export PIP_CACHE_DIR="$HOME/scratch/pip_cache"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"

activate_env() {
    # shellcheck disable=SC1090
    source "$CONDA_SH"
    set +u
    conda activate "$ENV"
}

# Hours ledger: every job adds its wall time when it ends (also on failure). GPU jobs count against the 7 h guard.
ledger_on() {
    _STEP="$1"; _GPU="$2"
    trap 'python -m pathcrf_adapter.cycle ledger "$_STEP" "${SLURM_JOB_ID:-local}" "$SECONDS" "$_GPU" || true' EXIT
}

banner() {
    echo "=== $1 | job ${SLURM_JOB_ID:-local} on ${SLURMD_NODENAME:-$HOSTNAME} | $(date '+%Y-%m-%d %H:%M:%S') ==="
    python -c "import torch; print('torch', torch.__version__, '| cuda', torch.cuda.is_available(), '|', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu')" || true
}

guard_or_exit() {
    # usage: guard_or_exit STEP CAP_KEY  -> exits the job cleanly (exit 0) when the 7 h GPU guard says SKIP
    if [ "$(python -m pathcrf_adapter.cycle guard "$1" "$(python -m pathcrf_adapter.cycle cap "$2")")" = "SKIP" ]; then
        echo "GPU-hour guard: $1 skipped (runs/guard_stop.json)"; exit 0
    fi
}

fold0_held() {
    python -c "import json, sys; print(' '.join(map(str, json.load(open(sys.argv[1]))['folds'][0])))" "$BUILT/folds.json"
}
