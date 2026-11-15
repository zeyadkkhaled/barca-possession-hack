#!/bin/bash
#SBATCH --job-name=m1_score_fold0
#SBATCH --time=00:45:00
#SBATCH --cpus-per-task=5
#SBATCH --mem=16G
# Scores the frozen C3, PathCRF alone and C3 + PathCRF (travelling-vs-loose from C3 and from PathCRF's head, and PathCRF naming
# only the player) on fold 0's held-out matches. Writes runs/results_fold0.json + results_fold0.md. CPU only.
source "$HOME/projects/m1-possession/code/jobs/common.sh"
activate_env
ledger_on score_fold0 0
banner score_fold0
cd "$CODE"
python -m m1pc.cycle score-fold0
