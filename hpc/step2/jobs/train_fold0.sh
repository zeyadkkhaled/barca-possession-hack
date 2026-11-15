#!/bin/bash
# GPU job. usage: train_fold0.sh KEY [train flags]   (KEY = ball or noball)
# Fine-tunes PathCRF from its published weights on folds 1-4 (inner validation: one match per fold, early stopping),
# auto-resumes from last.pt, stops cleanly at the GPU cap; then predicts fold 0 and scores it -> runs/score_KEY.{json,md}.
source "$HOME/projects/barca-step2/code/jobs/common.sh"
KEY="${1:?usage: train_fold0.sh KEY [flags]}"; shift
activate_env
ledger_on "train_$KEY" 1
banner "train_fold0 $KEY $*"
cd "$CODE"
guard_or_exit "train_$KEY" "$KEY"
RUN="$(python -m pathcrf_adapter.cycle run-dir "$KEY")"
MAXEP="$(python -m pathcrf_adapter.cycle max-epochs)"
TRAIN_H="$(python -c "import sys; print(max(0.2, float(sys.argv[1]) - 0.4))" "$(python -m pathcrf_adapter.cycle cap "$KEY")")"
echo "run: $RUN  max epochs: $MAXEP  training cap: $TRAIN_H h  flags: $*"
python -m pathcrf_adapter.train --data "$BUILT" --run-dir "$RUN" --fold 0 --max-epochs "$MAXEP" --gpu-hours "$TRAIN_H" "$@"
python -m pathcrf_adapter.predict --data "$BUILT" --run-dir "$RUN"
python -m pathcrf_adapter.score --data "$BUILT" --c3t "$(python -m pathcrf_adapter.cycle run-dir c3t)" --preds "$RUN/preds" \
    --out "$RUNS/score_$KEY.json" --title "fine-tuned $KEY"
