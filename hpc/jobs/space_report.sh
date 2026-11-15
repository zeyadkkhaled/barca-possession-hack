#!/bin/bash
#SBATCH --job-name=m1_space_report
#SBATCH --time=00:10:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
# Size per folder under ~, largest first (pure Python). Also fine to run directly: bash space_report.sh
source "$HOME/projects/m1-possession/code/jobs/common.sh"
cd "$CODE"
"$BOOT_PY" -m m1pc.cycle space
