#!/bin/bash
# ONE command for the whole cycle. Submits every step with sbatch, strictly one after another
# (--dependency=afterok:<previous job>, --kill-on-invalid-dep=yes: if a job fails, Slurm cancels all later ones).
#
#   bash run_cycle.sh                       # full cycle
#   bash run_cycle.sh --from score_fold0    # restart from a step (earlier results and checkpoints are reused; training resumes)
#   bash run_cycle.sh --dry-run             # print the job chain, submit nothing
#
# Steps: setup_env smoke_test train_fold predict_fold score_fold0 abl_ref abl_a1 abl_a2 abl_a3 abl_a4 decide_and_train_all summary
# GPU-hour guard: before every training step, if the GPU hours used so far + that step's cap would exceed 30 h, the step (and
# every later training step) is skipped and the reason is written to runs/guard_stop.json and CYCLE_SUMMARY.md.
set -eo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$HOME/projects/m1-possession"
BOOT_PY="${M1_BOOT_PY:-/nfs/slurm/conda/bin/python}"
SBATCH="${M1_SBATCH:-sbatch}"
FROM=""; DRY=0
while [ $# -gt 0 ]; do
    case "$1" in
        --from) FROM="$2"; shift 2 ;;
        --dry-run) DRY=1; shift ;;
        -h|--help) echo "usage: bash run_cycle.sh [--from STEP] [--dry-run]"; exit 0 ;;
        *) echo "unknown argument $1"; exit 1 ;;
    esac
done
STEPS=(setup_env smoke_test train_fold predict_fold score_fold0 abl_ref abl_a1 abl_a2 abl_a3 abl_a4 decide_and_train_all summary)
declare -A CMD TIME GPU
CMD[setup_env]="env/setup_env.sh";                 TIME[setup_env]="01:00:00";            GPU[setup_env]=1
CMD[smoke_test]="jobs/smoke_test.sh";               TIME[smoke_test]="00:30:00";           GPU[smoke_test]=1
CMD[train_fold]="jobs/train_fold.sh 0";             TIME[train_fold]="08:30:00";           GPU[train_fold]=1
CMD[predict_fold]="jobs/predict_fold.sh __MAIN__";  TIME[predict_fold]="01:30:00";         GPU[predict_fold]=1
CMD[score_fold0]="jobs/score_fold0.sh";             TIME[score_fold0]="00:45:00";          GPU[score_fold0]=0
CMD[abl_ref]="jobs/ablation_check.sh ref";          TIME[abl_ref]="02:15:00";              GPU[abl_ref]=1
CMD[abl_a1]="jobs/ablation_check.sh a1";            TIME[abl_a1]="02:15:00";               GPU[abl_a1]=1
CMD[abl_a2]="jobs/ablation_check.sh a2";            TIME[abl_a2]="02:15:00";               GPU[abl_a2]=1
CMD[abl_a3]="jobs/ablation_check.sh a3";            TIME[abl_a3]="03:15:00";               GPU[abl_a3]=1
CMD[abl_a4]="jobs/ablation_check.sh a4";            TIME[abl_a4]="01:30:00";               GPU[abl_a4]=1
CMD[decide_and_train_all]="jobs/decide_and_train_all.sh"; TIME[decide_and_train_all]="08:45:00"; GPU[decide_and_train_all]=1
CMD[summary]="jobs/summary.sh";                     TIME[summary]="00:15:00";              GPU[summary]=0

if [ -n "$FROM" ] && [ -z "${CMD[$FROM]+x}" ]; then echo "unknown step '$FROM'; steps: ${STEPS[*]}"; exit 1; fi
mkdir -p "$HOME/logs" "$ROOT/runs" "$HOME/scratch"
cd "$HERE"
export PYTHONPATH="$HERE"
if [ "$DRY" = "1" ] && [ -n "${M1_DRY_DATE:-}" ]; then INIT=(init --date "$M1_DRY_DATE"); else INIT=(init); fi
"$BOOT_PY" -m m1pc.cycle "${INIT[@]}"
MAIN="$("$BOOT_PY" -m m1pc.cycle run-dir main)"
started=0; prev=""; n=0
for s in "${STEPS[@]}"; do
    if [ "$started" = "0" ]; then
        if [ -z "$FROM" ] || [ "$s" = "$FROM" ]; then started=1; else continue; fi
    fi
    n=$((n + 1))
    read -r -a C <<< "${CMD[$s]//__MAIN__/$MAIN}"
    OPTS=(--parsable --kill-on-invalid-dep=yes --job-name="m1_$s" --time="${TIME[$s]}" --cpus-per-task=5 --mem=32G
          --output="$HOME/logs/%x_%j.log" --error="$HOME/logs/%x_%j.err")
    if [ "${GPU[$s]}" = "1" ]; then OPTS+=(--gres=gpu:1); fi
    if [ -n "$prev" ]; then OPTS+=(--dependency="afterok:$prev"); fi
    if [ "$DRY" = "1" ]; then
        jid="DRY$n"
        echo "[$n] $s -> job $jid:  $SBATCH ${OPTS[*]} ${C[*]}"
    else
        jid="$($SBATCH "${OPTS[@]}" "${C[@]}")"
        jid="${jid%%;*}"
        echo "[$n] $s -> job $jid"
    fi
    prev="$jid"
done
if [ "$started" = "0" ]; then echo "nothing submitted"; exit 1; fi
echo "submitted $n jobs. Progress: squeue -u \$USER ; logs: ~/logs/m1_<step>_<jobid>.log ; results: ~/projects/m1-possession/runs/"
