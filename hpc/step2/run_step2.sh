#!/bin/bash
# ONE command for step 2b. Submits every step with sbatch, strictly one after another (--dependency=afterok:<previous>,
# --kill-on-invalid-dep=yes: if a job fails, Slurm cancels all later ones; restart with --from STEP).
#
#   bash run_step2.sh                     # whole chain
#   bash run_step2.sh --from train_ball   # restart from a step (finished steps are skipped by their own checks; training resumes)
#   bash run_step2.sh --dry-run           # print the job chain and the sbatch options, submit nothing, write nothing
#
# GPU budget: guard at 7 GPU hours in total (runs/gpu_ledger.json); per-step caps in pathcrf_adapter/cycle.py CAPS.
set -eo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${B2_ROOT:-$HOME/projects/barca-step2}"
BOOT_PY="${B2_BOOT_PY:-/nfs/slurm/conda/bin/python}"
SBATCH="${B2_SBATCH:-sbatch}"
FROM=""; DRY=0
while [ $# -gt 0 ]; do
    case "$1" in
        --from) FROM="$2"; shift 2 ;;
        --dry-run) DRY=1; shift ;;
        -h|--help) echo "usage: bash run_step2.sh [--from STEP] [--dry-run]"; exit 0 ;;
        *) echo "unknown argument $1"; exit 1 ;;
    esac
done
STEPS=(setup_env build_data c3t_fold0 smoke_test zeroshot train_ball train_noball train_all summary)
declare -A CMD TIME GPU CPU MEM
CMD[setup_env]="env/setup_env.sh";            TIME[setup_env]="01:00:00";    GPU[setup_env]=0;    CPU[setup_env]=2; MEM[setup_env]=8G
CMD[build_data]="jobs/build_data.sh";         TIME[build_data]="04:00:00";   GPU[build_data]=0;   CPU[build_data]=8; MEM[build_data]=64G
CMD[c3t_fold0]="jobs/c3t_fold0.sh";           TIME[c3t_fold0]="02:00:00";    GPU[c3t_fold0]=0;    CPU[c3t_fold0]=8; MEM[c3t_fold0]=32G
CMD[smoke_test]="jobs/smoke_test.sh";         TIME[smoke_test]="00:40:00";   GPU[smoke_test]=1;   CPU[smoke_test]=5; MEM[smoke_test]=32G
CMD[zeroshot]="jobs/zeroshot.sh";             TIME[zeroshot]="00:40:00";     GPU[zeroshot]=1;     CPU[zeroshot]=5; MEM[zeroshot]=32G
CMD[train_ball]="jobs/train_fold0.sh ball";   TIME[train_ball]="03:00:00";   GPU[train_ball]=1;   CPU[train_ball]=5; MEM[train_ball]=32G
CMD[train_noball]="jobs/train_fold0.sh noball --no-ball"; TIME[train_noball]="02:40:00"; GPU[train_noball]=1; CPU[train_noball]=5; MEM[train_noball]=32G
CMD[train_all]="jobs/train_all.sh";           TIME[train_all]="02:20:00";    GPU[train_all]=1;    CPU[train_all]=5; MEM[train_all]=32G
CMD[summary]="jobs/summary.sh";               TIME[summary]="00:15:00";      GPU[summary]=0;      CPU[summary]=2; MEM[summary]=4G

if [ -n "$FROM" ] && [ -z "${CMD[$FROM]+x}" ]; then echo "unknown step '$FROM'; steps: ${STEPS[*]}"; exit 1; fi
cd "$HERE"
if [ "$DRY" = "0" ]; then
    mkdir -p "$HOME/logs" "$ROOT/runs" "$HOME/scratch"
    B2_ROOT="$ROOT" PYTHONPATH="$HERE/src" "$BOOT_PY" -m pathcrf_adapter.cycle init
fi
started=0; prev=""; n=0
for s in "${STEPS[@]}"; do
    if [ "$started" = "0" ]; then
        if [ -z "$FROM" ] || [ "$s" = "$FROM" ]; then started=1; else continue; fi
    fi
    n=$((n + 1))
    read -r -a C <<< "${CMD[$s]}"
    OPTS=(--parsable --kill-on-invalid-dep=yes --job-name="b2_$s" --time="${TIME[$s]}" --cpus-per-task="${CPU[$s]}" --mem="${MEM[$s]}"
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
if [ "$DRY" = "1" ]; then echo "dry run: $n jobs listed, nothing submitted"; exit 0; fi
echo "submitted $n jobs. Progress: squeue -u \$USER ; logs: ~/logs/b2_<step>_<jobid>.log ; results: $ROOT/runs/"
