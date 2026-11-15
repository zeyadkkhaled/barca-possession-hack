#!/bin/bash
#SBATCH --job-name=m1_smoke_test
#SBATCH --time=00:30:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=5
#SBATCH --mem=32G
# ~200 training steps of fold 0 with the fast config; prints time per step, GPU memory, % of windows skipped and the estimated
# time per epoch / per fold. If fold 0 would take more than 8 h, writes runs/recommended_max_epochs.txt (read by train_fold.sh).
source "$HOME/projects/m1-possession/code/jobs/common.sh"
activate_env
ledger_on smoke_test 1
banner smoke_test
cd "$CODE"
SMOKE="$HOME/scratch/smoke_${SLURM_JOB_ID:-local}"
python -m m1pc.train --data "$DATA" --run-dir "$SMOKE" --fold 0 --max-steps 200 --gpu-hours 0.4
python -m m1pc.cycle smoke-report "$SMOKE"
rm -f "$SMOKE"/checkpoints/*.pt
