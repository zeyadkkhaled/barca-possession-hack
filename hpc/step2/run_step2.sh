#!/bin/bash
# ONE command for step 2b. Submits every step with sbatch (each job: 1 MIG slice, --gres=gpu:1, 5 CPUs, 32 GB):
#
#   build -> smoke -> { zeroshot, ft_ball, ft_noball }   (the three run in parallel after the smoke test)
#   { ft_ball, ft_noball } -afterany-> final -> predict_eval -afterany-> summary
#
# Dependencies: afterok everywhere except final (afterany on both fold-0 jobs) and summary (afterany), with
# --kill-on-invalid-dep=yes, so a failed job cancels what needs it. final also refuses to start unless build and smoke are
# done (step markers), so a failed smoke test stops every long training.
#
#   bash run_step2.sh                      # the whole chain
#   bash run_step2.sh --from ft_noball     # resubmit from a step; finished steps (runs/steps/*.done) exit at once,
#                                          # training resumes from last.pt
#   bash run_step2.sh --only zeroshot      # one step, no dependencies
#   bash run_step2.sh --dry-run            # print the jobs, dependencies and caps; submit and write nothing
# GPU guard: 10 GPU hours in total (runs/gpu_ledger.json + reservations of running jobs); caps in pathcrf_adapter/cycle.py.
set -eo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${B2_ROOT:-$HOME/projects/barca-step2}"
BOOT_PY="${B2_BOOT_PY:-/nfs/slurm/conda/bin/python}"
SBATCH="${B2_SBATCH:-sbatch}"
FROM=""; ONLY=""; DRY=0
while [ $# -gt 0 ]; do
    case "$1" in
        --from) FROM="$2"; shift 2 ;;
        --only) ONLY="$2"; shift 2 ;;
        --dry-run) DRY=1; shift ;;
        -h|--help) sed -n '2,20p' "$0"; exit 0 ;;
        *) echo "unknown argument $1"; exit 1 ;;
    esac
done
STEPS=(build smoke zeroshot ft_ball ft_noball final predict_eval summary)
declare -A CMD TIME DEP CAP
CMD[build]="jobs/build.sh";                 TIME[build]="00:30:00";        DEP[build]="";                              CAP[build]=0.4
CMD[smoke]="jobs/smoke.sh";                 TIME[smoke]="00:30:00";        DEP[smoke]="afterok:build";                 CAP[smoke]=0.4
CMD[zeroshot]="jobs/zeroshot.sh";           TIME[zeroshot]="00:30:00";     DEP[zeroshot]="afterok:smoke";              CAP[zeroshot]=0.4
CMD[ft_ball]="jobs/ft_fold0.sh ball";       TIME[ft_ball]="02:40:00";      DEP[ft_ball]="afterok:smoke";               CAP[ft_ball]=2.5
CMD[ft_noball]="jobs/ft_fold0.sh noball";   TIME[ft_noball]="02:40:00";    DEP[ft_noball]="afterok:smoke";             CAP[ft_noball]=2.5
CMD[final]="jobs/final.sh";                 TIME[final]="03:10:00";        DEP[final]="afterany:ft_ball:ft_noball";    CAP[final]=3.0
CMD[predict_eval]="jobs/predict_eval.sh";   TIME[predict_eval]="00:30:00"; DEP[predict_eval]="afterok:final";          CAP[predict_eval]=0.4
CMD[summary]="jobs/summary.sh";             TIME[summary]="00:10:00";      DEP[summary]="afterany:predict_eval:zeroshot"; CAP[summary]=0.1

for s in "$FROM" "$ONLY"; do
    if [ -n "$s" ] && [ -z "${CMD[$s]+x}" ]; then echo "unknown step '$s'; steps: ${STEPS[*]}"; exit 1; fi
done
cd "$HERE"
if [ "$DRY" = "0" ]; then
    mkdir -p "$HOME/logs" "$ROOT/runs" "$HOME/scratch"
    B2_ROOT="$ROOT" PYTHONPATH="$HERE/src" "$BOOT_PY" -m pathcrf_adapter.cycle init
fi
declare -A JID
started=0; n=0; total=0
for s in "${STEPS[@]}"; do
    if [ -n "$ONLY" ]; then [ "$s" = "$ONLY" ] || continue
    elif [ "$started" = "0" ]; then
        if [ -z "$FROM" ] || [ "$s" = "$FROM" ]; then started=1; else continue; fi
    fi
    n=$((n + 1))
    read -r -a C <<< "${CMD[$s]}"
    OPTS=(--parsable --kill-on-invalid-dep=yes --job-name="b2_$s" --time="${TIME[$s]}" --gres=gpu:1 --cpus-per-task=5 --mem=32G
          --output="$HOME/logs/%x_%j.log" --error="$HOME/logs/%x_%j.err")
    dep=""
    if [ -n "${DEP[$s]}" ] && [ -z "$ONLY" ]; then
        kind="${DEP[$s]%%:*}"; ids=""
        IFS=':' read -r -a names <<< "${DEP[$s]#*:}"
        for d in "${names[@]}"; do [ -n "${JID[$d]+x}" ] && ids="$ids:${JID[$d]}"; done
        [ -n "$ids" ] && dep="$kind$ids" && OPTS+=(--dependency="$dep")
    fi
    total=$(awk -v a="$total" -v b="${CAP[$s]}" 'BEGIN { printf "%.1f", a + b }')
    if [ "$DRY" = "1" ]; then
        jid="DRY$n"
        printf '[%d] %-12s cap %4s GPU h  limit %s  dep %-26s  %s\n' "$n" "$s" "${CAP[$s]}" "${TIME[$s]}" "${dep:-none}" "${C[*]}"
    else
        jid="$($SBATCH "${OPTS[@]}" "${C[@]}")"
        jid="${jid%%;*}"
        echo "[$n] $s -> job $jid ${dep:+(dependency $dep)}"
    fi
    JID[$s]="$jid"
done
if [ "$n" = "0" ]; then echo "nothing submitted"; exit 1; fi
if [ "$DRY" = "1" ]; then
    echo "dry run: $n jobs, caps sum $total GPU h (guard 10 h); every job: --gres=gpu:1 --cpus-per-task=5 --mem=32G; nothing submitted"
    exit 0
fi
echo "submitted $n jobs. Check: bash $HERE/status_step2.sh   (or squeue -u \$USER; logs ~/logs/b2_<step>_<jobid>.log)"
