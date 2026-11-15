#!/bin/bash
# CPU job. C3-T trained on folds 1-4 of the built features, predictions for fold 0 (score.py swaps its team probability
# for PathCRF's). Skipped if the decoder file exists.
source "$HOME/projects/barca-step2/code/jobs/common.sh"
activate_env
ledger_on c3t_fold0 0
banner c3t_fold0
cd "$CODE"
OUT="$(python -m pathcrf_adapter.cycle run-dir c3t)"
if [ -f "$OUT/decoder.json" ]; then echo "$OUT/decoder.json exists: skipped"; exit 0; fi
python -m pathcrf_adapter.c3t_fold --data "$BUILT" --fold 0 --out "$OUT" --threads "${SLURM_CPUS_PER_TASK:-8}"
