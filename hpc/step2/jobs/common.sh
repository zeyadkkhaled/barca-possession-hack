#!/bin/bash
# Sourced by every step-2b job. Layout:
#   ~/projects/barca-step2/{code,built,built_eval,eval_data,runs,final}   this cycle
#   ~/projects/m1-possession/{data,runs,final}                             M1, READ ONLY (data package, M1 checkpoints)
#   ~/envs/m1 (env), ~/logs (b2_<step>_<jobid>.log / .err), ~/scratch (temporary)
# Every job: skips itself if runs/steps/<step>.done exists; marks running / done / failed / skipped; adds its wall time
# to runs/gpu_ledger.json (every job holds a MIG slice and is charged, CPU work included).
set -eo pipefail
ROOT="${B2_ROOT:-$HOME/projects/barca-step2}"
CODE="$ROOT/code"
BUILT="$ROOT/built"
BUILT_EVAL="$ROOT/built_eval"
EVAL_PKG="${B2_EVAL_PACKAGE:-$ROOT/eval_data}"
RUNS="$ROOT/runs"
FINAL="$ROOT/final"
PKG="${B2_PACKAGE:-$HOME/projects/m1-possession/data}"
ENV="${B2_ENV:-$HOME/envs/m1}"
CONDA_SH="${B2_CONDA_SH:-/nfs/slurm/conda/etc/profile.d/conda.sh}"
BOOT_PY="${B2_BOOT_PY:-/nfs/slurm/conda/bin/python}"     # conda base Python: the stdlib-only bookkeeping (cycle.py)
mkdir -p "$RUNS/steps" "$FINAL" "$BUILT" "$HOME/logs" "$HOME/scratch"
export B2_ROOT="$ROOT"
export PYTHONPATH="$CODE/src"
export PYTHONUNBUFFERED=1 PYTHONIOENCODING=utf-8 PYTHONUTF8=1
export TMPDIR="$HOME/scratch"
export PIP_CACHE_DIR="$HOME/scratch/pip_cache"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
CY() { "$BOOT_PY" -m pathcrf_adapter.cycle "$@"; }

activate_env() {
    # shellcheck disable=SC1090
    source "$CONDA_SH"
    set +u
    conda activate "$ENV"
}

_finish() {
    local rc=$?
    if [ "${_STEP_ALREADY:-0}" = "1" ]; then exit 0; fi
    if [ "$rc" != "0" ]; then CY mark "$_STEP" failed "${SLURM_JOB_ID:-local}" || true
    elif [ "${_STEP_SKIPPED:-0}" = "1" ]; then CY mark "$_STEP" skipped "${SLURM_JOB_ID:-local}" || true
    else CY mark "$_STEP" done "${SLURM_JOB_ID:-local}" || true; fi
    CY ledger "$_STEP" "${SLURM_JOB_ID:-local}" "$SECONDS" || true
    echo "=== $_STEP finished (exit $rc) after $((SECONDS / 60)) min ==="
    exit "$rc"
}

# usage: step_begin STEP  -> exits 0 at once if the step is already done; otherwise marks it running, starts the ledger,
# checks the 10 GPU-hour guard (a SKIP ends the job cleanly, marked "skipped") and activates the env.
step_begin() {
    _STEP="$1"
    if [ "$_STEP" != "summary" ] && [ "$(CY is-done "$_STEP")" = "yes" ]; then
        echo "$_STEP already done (runs/steps/$_STEP.done): nothing to do"; _STEP_ALREADY=1; exit 0
    fi
    trap _finish EXIT
    CY init > /dev/null                                     # idempotent: run folders exist even for --only / a manual run
    CY mark "$_STEP" running "${SLURM_JOB_ID:-local}"
    echo "=== $_STEP | job ${SLURM_JOB_ID:-local} on ${SLURMD_NODENAME:-$HOSTNAME} | $(date '+%Y-%m-%d %H:%M:%S') | cap $(CY cap "$_STEP") GPU h ==="
    if [ "$(CY guard "$_STEP")" = "SKIP" ]; then
        echo "GPU-hour guard: $_STEP skipped (runs/guard_stop.json)"; _STEP_SKIPPED=1; exit 0
    fi
    activate_env
    python -c "import torch; print('torch', torch.__version__, '| cuda', torch.cuda.is_available(), '|', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu')" || true
    cd "$CODE"
}

fold0_held() {
    python -c "import json, sys; print(' '.join(map(str, json.load(open(sys.argv[1]))['folds'][0])))" "$BUILT/folds.json"
}

# write a short NOTES.md into a folder: notes DIR "text"
notes() {
    mkdir -p "$1"
    printf '# %s\n\n%s\n\n- job %s, %s, wall time so far %s min\n' "$(basename "$1")" "$2" "${SLURM_JOB_ID:-local}" \
        "$(date '+%Y-%m-%d %H:%M')" "$((SECONDS / 60))" >> "$1/NOTES.md"
}
