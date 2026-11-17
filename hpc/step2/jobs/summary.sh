#!/bin/bash
# f. Always runs at the end (afterany): runs/STEP2_SUMMARY.md with every score, the steps' states, GPU hours per job and
# the chosen variant.
source "$HOME/projects/barca-step2/code/jobs/common.sh"
step_begin summary
python -m pathcrf_adapter.cycle summary
