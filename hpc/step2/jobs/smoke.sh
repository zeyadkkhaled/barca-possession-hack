#!/bin/bash
# g. Smoke test, before any long training: 300 fine-tuning steps on 2 fold-0 training matches for BOTH variants (ball,
# --no-ball), initialised from M1's fold-0 best.pt (dead windows kept, stride 3 s). Checks that the M1 checkpoint loads
# (the no-ball run must narrow the ball inputs), that the loss goes down, the time per step, peak GPU memory, and estimates
# minutes per epoch -> runs/smoke_report.json. Unhealthy -> exit 1, so Slurm cancels every later job.
source "$HOME/projects/barca-step2/code/jobs/common.sh"
step_begin smoke
M1F0="$(CY m1-ckpt fold0)"
[ -f "$M1F0" ] || { echo "M1 fold-0 checkpoint missing: $M1F0"; exit 1; }
TWO="$(python -c "import json, sys; F = json.load(open(sys.argv[1]))['folds']; print(' '.join(map(str, sorted(F[1])[1:3])))" "$BUILT/folds.json")"
for V in ball noball; do
    RUN="$(CY run-dir "smoke_$V")"
    FL=""; [ "$V" = "noball" ] && FL="--no-ball"
    rm -rf "$RUN"
    # shellcheck disable=SC2086
    python -m pathcrf_adapter.train --data "$BUILT" --run-dir "$RUN" --fold 0 --matches $TWO --init-from "$M1F0" \
        --keep-dead-windows --stride 3 --max-steps "${B2_SMOKE_STEPS:-300}" --max-epochs 1 --gpu-hours 0.15 --workers 4 $FL
    rm -f "$RUN"/checkpoints/*.pt
done
python -m pathcrf_adapter.cycle smoke-report
notes "$(CY run-dir smoke_ball)" "Smoke test (both variants): see runs/smoke_report.json."
