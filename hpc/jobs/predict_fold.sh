#!/bin/bash
#SBATCH --job-name=m1_predict_fold
#SBATCH --time=01:30:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=5
#SBATCH --mem=32G
# usage: predict_fold.sh RUN_DIR [extra predict flags, e.g. --fp32 --out-subdir preds_fp32]
# Per-frame outputs for the run's held-out matches only -> RUN_DIR/preds/<match>.npz (format: m1pc/predict.py).
source "$HOME/projects/m1-possession/code/jobs/common.sh"
RUN="${1:?usage: predict_fold.sh RUN_DIR}"; shift
activate_env
ledger_on predict_fold 1
banner "predict_fold $RUN"
cd "$CODE"
if [ ! -f "$RUN/checkpoints/best.pt" ]; then
    echo "no best.pt in $RUN (training skipped or failed): nothing to predict"; exit 0
fi
python -m m1pc.predict --data "$DATA" --run-dir "$RUN" "$@"
