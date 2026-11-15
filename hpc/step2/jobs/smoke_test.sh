#!/bin/bash
# GPU job. ~200 fine-tuning steps of fold 0 (ball input): time per step, GPU memory, windows skipped; writes
# runs/smoke_report.json and runs/recommended_max_epochs.txt (epochs that fit the fold-0 cap).
source "$HOME/projects/barca-step2/code/jobs/common.sh"
activate_env
ledger_on smoke_test 1
banner smoke_test
cd "$CODE"
guard_or_exit smoke_test smoke
SMOKE="$(python -m pathcrf_adapter.cycle run-dir smoke)"
python -m pathcrf_adapter.train --data "$BUILT" --run-dir "$SMOKE" --fold 0 --max-steps 200 --gpu-hours 0.25
python -m pathcrf_adapter.cycle smoke-report "$SMOKE"
rm -f "$SMOKE"/checkpoints/*.pt
