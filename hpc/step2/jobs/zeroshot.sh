#!/bin/bash
# GPU job. PathCRF's published model 140 as is (no ball input, no fine-tuning) on fold 0's 6 held-out matches, scored
# against C3-T -> runs/score_zeroshot.{json,md}.
source "$HOME/projects/barca-step2/code/jobs/common.sh"
activate_env
ledger_on zeroshot 1
banner zeroshot
cd "$CODE"
guard_or_exit zeroshot zeroshot
ZS="$(python -m pathcrf_adapter.cycle run-dir zeroshot)"
# shellcheck disable=SC2046
python -m pathcrf_adapter.predict --data "$BUILT" --zero-shot --out "$ZS/preds" --matches $(fold0_held)
python -m pathcrf_adapter.score --data "$BUILT" --c3t "$(python -m pathcrf_adapter.cycle run-dir c3t)" --preds "$ZS/preds" \
    --out "$RUNS/score_zeroshot.json" --title "zero-shot"
