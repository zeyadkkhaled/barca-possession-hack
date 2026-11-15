#!/bin/bash
#SBATCH --job-name=m1_decide_train_all
#SBATCH --time=08:45:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=5
#SBATCH --mem=32G
# Reads runs/results_fold0.json and runs/ablation.json. If PathCRF alone or combined beats C3 on fold 0 (frame accuracy >= +0.3
# points OR spell-F1 >= +0.5 points, no metric worse by more than 1 point): trains ONE model on all 30 learning matches (fast
# config + every HARMFUL cut switched back; epochs = the main run's best epoch), cap 8 GPU h, and promotes it to final/.
# Else writes "PathCRF did not beat C3" (runs/decision.json; the summary repeats it).
source "$HOME/projects/m1-possession/code/jobs/common.sh"
activate_env
ledger_on decide_and_train_all 1
banner decide_and_train_all
cd "$CODE"
RESULT="$(python -m m1pc.cycle decide)"
echo "$RESULT"
case "$RESULT" in
    *TRAIN_ALL*) ;;
    *) echo "no training"; exit 0 ;;
esac
if [ "$(python -m m1pc.cycle guard train_all 8)" = "SKIP" ]; then
    echo "GPU-hour guard: all-matches training skipped (see runs/guard_stop.json)"; exit 0
fi
RUN="$(python -m m1pc.cycle run-dir all)"
EPOCHS="$(python -m m1pc.cycle decision max_epochs)"
read -r -a FLAGS <<< "$(python -m m1pc.cycle decision flags)"
echo "run: $RUN  epochs: $EPOCHS  flags: ${FLAGS[*]}"
python -m m1pc.train --data "$DATA" --run-dir "$RUN" --all --max-epochs "$EPOCHS" --gpu-hours 7.8 "${FLAGS[@]}"
python -m m1pc.cycle promote "$RUN"
