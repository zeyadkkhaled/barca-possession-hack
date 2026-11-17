#!/bin/bash
# d. Runs after BOTH fold-0 jobs ended (afterany). Picks the variant with the higher fold-0 spellF1 (runs/decision.json),
# then fine-tunes from M1's final_all best.pt on all 30 degraded + relinked matches: dead windows kept, stride 3 s,
# <= 6 epochs, training stops at 2.75 GPU h (job cap 3). No validation (every match is trained on): epochs.jsonl has the
# training loss and minutes. A checkpoint every epoch, auto-resume; the last epoch is the model; copied to final/.
source "$HOME/projects/barca-step2/code/jobs/common.sh"
step_begin final
for need in build smoke; do
    [ "$(CY is-done "$need")" = "yes" ] || { echo "step $need is not done: no final training"; exit 1; }
done
python -m pathcrf_adapter.cycle choose-variant
FL="$(CY decision flags)"
M1FA="$(CY m1-ckpt final)"
[ -f "$M1FA" ] || { echo "M1 final_all checkpoint missing: $M1FA"; exit 1; }
RUN="$(CY run-dir final)"
# shellcheck disable=SC2086
python -m pathcrf_adapter.train --data "$BUILT" --run-dir "$RUN" --all --init-from "$M1FA" \
    --keep-dead-windows --stride 3 --max-epochs 6 --gpu-hours "$(CY train-hours final)" --workers 4 $FL
[ -f "$RUN/checkpoints/best.pt" ] || { echo "no best.pt after training"; exit 1; }
python -m pathcrf_adapter.cycle promote "$RUN"
notes "$RUN" "Final model: variant $(CY decision variant) ($(CY decision why)); init $M1FA."
