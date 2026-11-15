#!/bin/bash
#SBATCH --job-name=m1_summary
#SBATCH --time=00:15:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
# Writes runs/CYCLE_SUMMARY.md (results, ablation verdicts, decision, GPU hours per step, files to download) and
# runs/space_report.txt. CPU only.
source "$HOME/projects/m1-possession/code/jobs/common.sh"
activate_env
ledger_on summary 0
cd "$CODE"
python -m m1pc.cycle summary
python -m m1pc.cycle space > "$RUNS/space_report.txt"
python -c "import sys; sys.stdout.write(open(sys.argv[1]).read())" "$RUNS/space_report.txt"
