#!/bin/bash
# Light, for the login node (Python only; no head / df): job states (squeue), step markers, last 3 epochs per training
# run, fold-0 scores so far, GPU hours used vs the 10 h guard, myquota, and the last 20 lines of the newest log.
#   bash ~/projects/barca-step2/code/status_step2.sh
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export B2_ROOT="${B2_ROOT:-$HOME/projects/barca-step2}" PYTHONPATH="$HERE/src" PYTHONUTF8=1
"${B2_BOOT_PY:-/nfs/slurm/conda/bin/python}" -m pathcrf_adapter.cycle status
