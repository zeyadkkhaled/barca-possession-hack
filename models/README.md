# models

The frozen C3-T models that `run.py` uses. They are committed byte-exact (`.gitattributes`: `-text`) so a fresh clone
gives the same submission file (sha256 in the README).

| folder | files | what | trained by |
|---|---|---|---|
| `c3/` | `layer1_dead.txt`, `layer2_home.txt` (LightGBM text models), `decoder.json` (Viterbi costs C / w / beta, feature list, post-rules) | layer 1 P(dead), layer 2 P(home \| alive), 3-state decoder | `scripts/fit_predict.py` on the 30 degraded PFF learning matches |
| `bfp/` | `bfp_dx.txt`, `bfp_dy.txt` (LightGBM) | ball-from-players: fills ball gaps from the players' positions and motion | `scripts/ballfill_eval.py` on degraded PFF |

Nothing else belongs here: other weights are git-ignored.
