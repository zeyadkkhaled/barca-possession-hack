#!/bin/bash
# b. Zero-shot on fold 0's 6 held-out matches (degraded + relinked): M1's fold-0 best.pt (clean PFF, ball input) and
# PathCRF's published weights (reference, no ball). Each combined with C3-T fold-0 P(D) + decoder, next to C3-T alone
# -> runs/score_zs_m1fold0.{json,md}, runs/score_zs_published.{json,md}.
source "$HOME/projects/barca-step2/code/jobs/common.sh"
step_begin zeroshot
ZS="$(CY run-dir zeroshot)"
C3T="$(CY run-dir c3t)"
HELD="$(fold0_held)"
# shellcheck disable=SC2086
python -m pathcrf_adapter.predict --data "$BUILT" --model "$(CY m1-ckpt fold0)" --out "$ZS/preds_m1fold0" --matches $HELD
python -m pathcrf_adapter.score --data "$BUILT" --c3t "$C3T" --preds "$ZS/preds_m1fold0" --out "$RUNS/score_zs_m1fold0.json" \
    --title "zero-shot, M1 fold-0 best.pt" --notes "$ZS"
# shellcheck disable=SC2086
python -m pathcrf_adapter.predict --data "$BUILT" --model published --out "$ZS/preds_published" --matches $HELD
python -m pathcrf_adapter.score --data "$BUILT" --c3t "$C3T" --preds "$ZS/preds_published" --out "$RUNS/score_zs_published.json" \
    --title "zero-shot, published weights" --notes "$ZS"
