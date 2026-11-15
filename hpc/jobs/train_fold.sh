#!/bin/bash
#SBATCH --job-name=m1_train_fold
#SBATCH --time=08:30:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=5
#SBATCH --mem=32G
# usage: train_fold.sh FOLD [extra train flags]
# Full training of one fold (fast config + flags). Cap 8 GPU hours (stops cleanly, keeps best.pt), a checkpoint every epoch,
# auto-resume from last.pt, early stopping, validation score after every epoch, NOTES.md. Max epochs: 10, or the smoke test's
# recommendation if 10 epochs would not fit 8 h.
source "$HOME/projects/m1-possession/code/jobs/common.sh"
FOLD="${1:-0}"; shift || true
activate_env
ledger_on "train_fold$FOLD" 1
banner "train_fold $FOLD"
cd "$CODE"
if [ "$(python -m m1pc.cycle guard "train_fold$FOLD" 8)" = "SKIP" ]; then
    echo "GPU-hour guard: training skipped (see runs/guard_stop.json)"; exit 0
fi
if [ "$FOLD" = "0" ]; then RUN="$(python -m m1pc.cycle run-dir main)"; else RUN="$RUNS/$(date +%F)_pathcrf_fast_fold$FOLD"; fi
MAXEP="$(python -m m1pc.cycle max-epochs)"
echo "run: $RUN  max epochs: $MAXEP  flags: $*"
python -m m1pc.train --data "$DATA" --run-dir "$RUN" --fold "$FOLD" --max-epochs "$MAXEP" --gpu-hours 7.8 "$@"
