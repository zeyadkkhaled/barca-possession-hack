"""Write reports/step1b_report.md (step-1b checks) from the aggregate files in reports/auto and M1's saved C3 scores.

    .venv/Scripts/python.exe scripts/make_step1b_report.py [--m1-results <M1 c3v4 results_final.json>]

Inputs: reports/auto/cv_c3t.json (degraded, calibrated dead_weight), cv_c3t_clean.json, cv_c3t_dw1.json,
degrade_stats*.json, feature_importance.json; M1's results_final.json holds per-match score counts only.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from barca_labels.scoring import merge_had
from barca_possession.paths import AUTO, ROOT

M1_RESULTS = Path(r"Z:\GP\Task 2 - match sim\data\processed\m1_possession\c3v4\results_final.json")

LEAK = """\
**Verdict: no leak in the model inputs.** No smoother, ball-from-players or C3-T feature reads PFF's `ballsSmoothed`
or anything built from event tags. One input is placed with the help of labels on purpose: where the degradation
cuts ball gaps (see item 3).

Path of the ball that the features see, from the PFF file to the features:

| step | file:line | what it reads |
|---|---|---|
| 1 | M1 `src/football_fatigue/ingestion/loaders.py:475-493` | raw `balls[0]` x/y/z go to `BallSample.raw`; `ballsSmoothed` is kept in separate fields |
| 2 | M1 `src/football_fatigue/event_detection/inputs.py:122` | `pf.ball.raw` only (docstring line 6: ballsSmoothed is built from event labels, so it is not used) |
| 3 | M1 `scripts/m1/build_labels_v1.py:49` | raw-ball visibility from the `"balls":[{"visibility":…}` list (VISIBLE / ESTIMATED from the tracker, not from events) |
| 4 | M1 `scripts/m1/build_inputs.py:41` | M1 `smooth_ball` on the step-2 raw ball and its raw visibility; `src/m1_possession/smoother.py` is guarded by `tests/test_m1_smoother.py:116-122` (no `game_event`, `possession_event`, `ballsSmoothed` or `PlayersSmoothed` in the code) |
| 5 | `hpc/tools/export_hpc.py:75,103` | the package's `ball_x/y/vx/vy/flag` = that smoothed raw ball (flag 3 "estimated" = the tracker's ESTIMATED visibility) |
| 6 | `src/barca_train/degrade_pff.py:80-96` | `ball_true_x/y` = package ball where flag != 0 |
| 7 | `src/barca_train/degrade_pff.py:268` | model ball `bx/by` = `ball_true` with extra gaps cut |
| 8 | `src/barca_possession/ballfill.py:66,88,100` | smoother and fill read `m["bx"], m["by"]`, players `px/py/team_sign` |
| 9 | `src/barca_possession/c3t.py:56-57,105-111` | features read only the fill (`ball[...]`) and `t, period, px, py, team_sign, pitch` |

The true ball and the labels are read only as training targets or for scoring:
- `ballfill.py:67-72`: ball-from-players targets.
- `c3t.py:150-155`: decoder transition counts.
- `pipeline.py`: layer targets and scores.

The one label-driven input: `degrade_pff.py:193-208` (`degrade_ball`, `dead = lab == 2`) starts ball gaps
dead_weight times more often on answer-key D frames, so the ball is missing before restarts about as often as in the
hackathon match. This ties ball-missing to D in the training data; item 3 measures how much CV gains from it.

Note on the "raw" ball: the PFF package ball is the raw `balls` field after **M1's own smoother** (repeat and spike
removal, short gap fill, RTS). That is our code, not PFF's tag-based ball. It is still cleaner than the Metrica raw ball,
which our pipeline smooths only once.
"""


def m1_scores(path: Path):
    if not path.exists():
        return None
    res = json.loads(path.read_text())["final"]["res"]
    had = [v["had"] for v in res.values()]
    p = merge_had(had)
    pm = [merge_had([h]) for h in had]
    return dict(n=len(had), mean=(float(np.mean([x["spell_f1"] for x in pm])), float(np.mean([x["macro_f1"] for x in pm]))),
                pooled=(p["spell_f1"], p["macro_f1"]))


def cv(name):
    f = AUTO / name
    if not f.exists():
        return None
    r = json.loads(f.read_text())
    n = r["mean"]["none"]
    return dict(mean=(n["spell"], n["macro"]), pooled=(n["pooled"]["spell"], n["pooled"]["macro"]), n=n["n_matches"],
                folds=[(fr["fold"], fr["none"]["spell"], fr["none"]["macro"]) for fr in r["folds"]], raw=r)


def miss(name):
    f = AUTO / name
    if not f.exists():
        return None
    d = json.loads(f.read_text())["degraded"]
    return d["gap"]["missing_frac_graded"], d.get("missing_before_restart"), d["slots"]["mean_slots_total"]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--m1-results", default=str(M1_RESULTS))
    a = ap.parse_args()
    deg, clean, dw1 = cv("cv_c3t.json"), cv("cv_c3t_clean.json"), cv("cv_c3t_dw1.json")
    m1 = m1_scores(Path(a.m1_results))
    dw = json.loads((AUTO / "degrade_stats.json").read_text())["degraded"]["dead_weight"]
    fi = json.loads((AUTO / "feature_importance.json").read_text()) if (AUTO / "feature_importance.json").exists() else None
    L = ["# Step 1b: checks before step 2", "",
         "All numbers are aggregates from the scripts. CV = the same 5 folds of the 30 PFF learning matches (M1 folds.json),",
         "scorer `score_had` (= the official scorer without X), LOOSE → last team, no post-rules.", "",
         "## 1. Clean PFF vs degraded PFF vs M1's C3", "",
         "| run | input | spellF1 mean per match | macroF1 mean per match | spellF1 pooled | macroF1 pooled |",
         "|---|---|---|---|---|---|"]

    def row(label, inp, r):
        if r is None:
            L.append(f"| {label} | {inp} | not run | | | |")
        else:
            L.append(f"| {label} | {inp} | {r['mean'][0]:.4f} | {r['mean'][1]:.4f} | {r['pooled'][0]:.4f} | {r['pooled'][1]:.4f} |")
    row("C3-T, clean PFF", "25 fps, player identities kept, true ball (only PFF's own gaps)", clean)
    row(f"C3-T, degraded (dead_weight {dw:g}, step 1)", "Metrica form", deg)
    row("C3-T, degraded, dead_weight 1", "Metrica form, gaps placed without labels", dw1)
    row("M1 C3 (round 4 final)", "29.97 fps, player ids, smoothed raw ball; held-out per-match counts from M1", m1)
    L += ["", "Per fold (spellF1 / macroF1, mean per match, no post-rules):", "",
          "| fold | clean | degraded (step 1) | degraded, dead_weight 1 |", "|---|---|---|---|"]
    for k in range(5):
        cells = []
        for r in (clean, deg, dw1):
            fr = next((f for f in r["folds"] if f[0] == k), None) if r else None
            cells.append(f"{fr[1]:.4f} / {fr[2]:.4f}" if fr else "–")
        L.append(f"| {k} | " + " | ".join(cells) + " |")
    L += ["", "Ball missing share (graded frames) / missing before restarts / mean slots on the pitch:", ""]
    for nm, f in (("clean", "degrade_stats_clean.json"), (f"degraded, dead_weight {dw:g}", "degrade_stats.json"),
                  ("degraded, dead_weight 1", "degrade_stats_dw1.json")):
        v = miss(f)
        if v:
            L.append(f"- {nm}: {100 * v[0]:.1f} % / {100 * v[1]:.1f} % / {v[2]:.2f}")
    L += ["", "Reading:"]
    if m1:
        L.append(f"- Aggregation does not explain it: M1's C3 is {m1['pooled'][0]:.4f} pooled and {m1['mean'][0]:.4f} as a "
                 f"mean per match ({m1['n']} matches).")
    if clean and deg:
        L.append(f"- Clean minus degraded: spellF1 {clean['mean'][0] - deg['mean'][0]:+.4f} (mean per match), "
                 f"{clean['pooled'][0] - deg['pooled'][0]:+.4f} (pooled).")
    if clean and m1:
        L.append(f"- C3-T on clean PFF minus M1 C3, same matches, folds and scorer: spellF1 "
                 f"{clean['pooled'][0] - m1['pooled'][0]:+.4f} pooled. Two things still differ: 25 vs 29.97 fps (spells "
                 "are compared on the resampled frames), and C3-T's 3-state decoder has w / beta chosen on inner "
                 "out-of-fold spellF1 with a team-level layer 2, while M1's C3 decodes a player owner.")
    L += ["", "## 2. Leak check (raw ball only, no ballsSmoothed, no tags)", "", LEAK,
          "## 3. Without the dead-ball gap weighting", ""]
    if deg and dw1:
        L += [f"| | dead_weight {dw:g} (step 1) | dead_weight 1 | difference |", "|---|---|---|---|",
              f"| spellF1, mean per match | {deg['mean'][0]:.4f} | {dw1['mean'][0]:.4f} | {dw1['mean'][0] - deg['mean'][0]:+.4f} |",
              f"| macroF1, mean per match | {deg['mean'][1]:.4f} | {dw1['mean'][1]:.4f} | {dw1['mean'][1] - deg['mean'][1]:+.4f} |",
              f"| spellF1, pooled | {deg['pooled'][0]:.4f} | {dw1['pooled'][0]:.4f} | {dw1['pooled'][0] - deg['pooled'][0]:+.4f} |",
              f"| macroF1, pooled | {deg['pooled'][1]:.4f} | {dw1['pooled'][1]:.4f} | {dw1['pooled'][1] - deg['pooled'][1]:+.4f} |",
              "", "Both runs re-train ball-from-players on their own degraded matches (out of fold), then run the same CV."]
    else:
        L.append("not run")
    L += ["", "## 4. Top 10 features by gain (final models in models/c3, fit on all 30 degraded matches)", ""]
    if fi:
        L.append(f"Layer 1, P(D), {fi['n_features_l1']} features: " + ", ".join(f"`{r['name']}`" for r in fi["layer1"]) + ".")
        L.append("")
        L.append(f"Layer 2, P(H | alive), {fi['n_features_l2']} features: " + ", ".join(f"`{r['name']}`" for r in fi["layer2"]) + ".")
        L += ["", "`@+2` means the value 2 s later (offline, both sides of the frame)."]
    L += ["", "## Commands", "", "```powershell",
          ".\\.venv\\Scripts\\python.exe -m barca_train.degrade_pff --clean --out outputs\\pff_clean --auto-name degrade_stats_clean.json",
          ".\\.venv\\Scripts\\python.exe scripts\\ballfill_eval.py --deg outputs\\pff_clean --train-deg outputs\\pff_degraded --fill outputs\\ballfill_clean --no-final --out-name ballfill_eval_clean.json",
          ".\\.venv\\Scripts\\python.exe scripts\\cv_c3t.py --deg outputs\\pff_clean --fill outputs\\ballfill_clean --out-name cv_c3t_clean.json",
          ".\\.venv\\Scripts\\python.exe -m barca_train.degrade_pff --dead-weight 1 --out outputs\\pff_degraded_dw1 --auto-name degrade_stats_dw1.json",
          ".\\.venv\\Scripts\\python.exe scripts\\ballfill_eval.py --deg outputs\\pff_degraded_dw1 --fill outputs\\ballfill_dw1 --no-final --out-name ballfill_eval_dw1.json",
          ".\\.venv\\Scripts\\python.exe scripts\\cv_c3t.py --deg outputs\\pff_degraded_dw1 --fill outputs\\ballfill_dw1 --out-name cv_c3t_dw1.json",
          ".\\.venv\\Scripts\\python.exe scripts\\feature_importance.py",
          ".\\.venv\\Scripts\\python.exe scripts\\make_step1b_report.py", "```", ""]
    out = ROOT / "reports" / "step1b_report.md"
    out.write_text("\n".join(L), encoding="utf-8")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
