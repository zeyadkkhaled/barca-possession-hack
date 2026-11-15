#!/bin/bash
#SBATCH --job-name=m1_ablation
#SBATCH --time=03:15:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=5
#SBATCH --mem=32G
# usage: ablation_check.sh ref|a1|a2|a3|a4   (run_cycle.sh submits them one after another)
#   ref  fast config            3 epochs, cap 2 GPU h
#   a1   --from-scratch         3 epochs, cap 2 GPU h
#   a2   --keep-dead-windows    3 epochs, cap 2 GPU h
#   a3   --stride 1             3 epochs, cap 3 GPU h (compared at the epochs reached if it hits the cap)
#   a4   the main model with --fp32 at inference only (no training); then writes runs/ablation.json + ablation.md
source "$HOME/projects/m1-possession/code/jobs/common.sh"
KEY="${1:?usage: ablation_check.sh ref|a1|a2|a3|a4}"
activate_env
ledger_on "ablation_$KEY" 1
banner "ablation $KEY"
cd "$CODE"
case "$KEY" in
    ref) FLAGS=(); CAP=2 ;;
    a1)  FLAGS=(--from-scratch); CAP=2 ;;
    a2)  FLAGS=(--keep-dead-windows); CAP=2 ;;
    a3)  FLAGS=(--stride 1); CAP=3 ;;
    a4)  FLAGS=(); CAP=1 ;;
    *)   echo "unknown ablation $KEY"; exit 1 ;;
esac
if [ "$KEY" = "a4" ]; then
    MAIN="$(python -m m1pc.cycle run-dir main)"
    if [ -f "$MAIN/checkpoints/best.pt" ]; then
        python -m m1pc.predict --data "$DATA" --run-dir "$MAIN" --fp32 --out-subdir preds_fp32
        python -m m1pc.cycle score "$MAIN" --preds preds_fp32 --out score_fp32
    else
        echo "main run has no best.pt: A4 skipped"
    fi
    python -m m1pc.cycle ablation
    exit 0
fi
if [ "$(python -m m1pc.cycle guard "ablation_$KEY" "$CAP")" = "SKIP" ]; then
    echo "GPU-hour guard: ablation $KEY skipped (see runs/guard_stop.json)"; exit 0
fi
RUN="$(python -m m1pc.cycle run-dir "$KEY")"
GH="$(python -c "print($CAP - 0.25)")"
python -m m1pc.train --data "$DATA" --run-dir "$RUN" --fold 0 --max-epochs 3 --patience 99 --gpu-hours "$GH" --ablation "${FLAGS[@]}"
python -m m1pc.predict --data "$DATA" --run-dir "$RUN" --checkpoint last.pt
python -m m1pc.cycle score "$RUN"
