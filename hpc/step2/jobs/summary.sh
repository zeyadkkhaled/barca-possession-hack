#!/bin/bash
# CPU job. runs/STEP2_SUMMARY.md (built-data stats, smoke test, fold-0 scores, decision, hours per step, files to
# download) and runs/space_report.txt.
source "$HOME/projects/barca-step2/code/jobs/common.sh"
activate_env
ledger_on summary 0
cd "$CODE"
python -m pathcrf_adapter.cycle decide > /dev/null || true
python -m pathcrf_adapter.cycle summary
python -m pathcrf_adapter.cycle space > "$RUNS/space_report.txt"
