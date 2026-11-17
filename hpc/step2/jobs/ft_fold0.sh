#!/bin/bash
# c. usage: ft_fold0.sh ball|noball
# Fine-tunes from M1's fold-0 best.pt on fold 0's training matches (degraded + relinked; inner validation: the first
# match of each other fold): dead windows kept, stride 3 s, <= 6 epochs, patience 2, training stops at 2.2 GPU h (job cap
# 2.5). A checkpoint every epoch, auto-resume from last.pt, logs/epochs.jsonl with validation spellF1 / macroF1 / loss /
# minutes after every epoch; the best epoch = highest validation spellF1. Then predicts and scores fold 0
# -> runs/score_ft_<variant>.{json,md}, and the score goes into the run's NOTES.md.
source "$HOME/projects/barca-step2/code/jobs/common.sh"
V="${1:?usage: ft_fold0.sh ball|noball}"
step_begin "ft_$V"
FL=""; [ "$V" = "noball" ] && FL="--no-ball"
RUN="$(CY run-dir "ft_$V")"
C3T="$(CY run-dir c3t)"
# shellcheck disable=SC2086
python -m pathcrf_adapter.train --data "$BUILT" --run-dir "$RUN" --fold 0 --init-from "$(CY m1-ckpt fold0)" \
    --keep-dead-windows --stride 3 --max-epochs 6 --patience 2 --gpu-hours "$(CY train-hours "ft_$V")" --workers 4 \
    --c3t "$C3T" $FL
python -m pathcrf_adapter.predict --data "$BUILT" --run-dir "$RUN"
python -m pathcrf_adapter.score --data "$BUILT" --c3t "$C3T" --preds "$RUN/preds" --out "$RUNS/score_ft_$V.json" \
    --title "fine-tuned fold 0, $V" --notes "$RUN"
