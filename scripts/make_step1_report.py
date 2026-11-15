"""Write reports/step1_report.md from the aggregate files in reports/auto/ (scores, counts, %; nothing per frame).

    .venv/Scripts/python.exe scripts/make_step1_report.py
"""
from __future__ import annotations

import json

from barca_possession.paths import AUTO, ROOT

FILES = """\
- `src/barca_possession/paths.py` – shared paths (outputs/, reports/auto/, models/, M1 PFF package; env overrides for tests).
- `src/barca_possession/load.py` – hackathon ATD loader (official games.py + starter baseline readers), orientation, noise stats.
- `src/barca_train/degrade_pff.py` – PFF package -> Metrica form (25 fps, coordinates / flip, anonymous slots with breaks /
  swaps / drop-outs, ball gaps copied from the measured histograms), H/A/D labels.
- `src/barca_possession/ballfill.py` – ball filling: M1 smoother at 25 fps (gaps <= 1 s), ball-from-players for longer gaps,
  ball_missing / gap_length_so_far / ball_source features.
- `src/barca_possession/c3t.py` – C3-T: team-level C3 features, layer 1 (D), layer 2 (H | alive) with stacking, 3-state Viterbi.
- `src/barca_possession/postrules.py` – the four post-rules and the simulated set-piece tags for PFF.
- `src/barca_train/pipeline.py` – H/A/D scoring, training one fold (inner OOF stacking + decoder choice), Smart Tagging tags.
- `scripts/ballfill_eval.py` – ball-from-players CV, out-of-fold fills, error by source, final model -> models/bfp/.
- `scripts/cv_c3t.py` – 5-fold CV of C3-T with / without each post-rule.
- `scripts/fit_predict.py` – fit on all 30, save models/c3/, predict the match, write outputs/submission_c3.csv, validate.
- `scripts/make_step1_report.py` – this report.
- `tests/test_step1.py` – unit tests of the decoder, row/frame mapping, post-rules, degradation rates (toy arrays only).
"""

COMMANDS = """\
Run in `D:\\barca_hack` (PowerShell), in this order. Each prints aggregate numbers only.

```powershell
.\\.venv\\Scripts\\python.exe -m barca_possession.load                       # ~1 min (first run parses the ATD, then cached)
.\\.venv\\Scripts\\python.exe -m barca_train.degrade_pff                # ~2 min, 30 PFF matches
.\\.venv\\Scripts\\python.exe scripts\\ballfill_eval.py --workers 4         # ~10 min
.\\.venv\\Scripts\\python.exe scripts\\cv_c3t.py --workers 4                 # ~1 h with --threads 1 (default)
.\\.venv\\Scripts\\python.exe scripts\\fit_predict.py --workers 4            # ~15 min; writes outputs\\submission_c3.csv
.\\.venv\\Scripts\\python.exe scripts\\make_step1_report.py
```
"""

FEATURES = """\
- **Removed (need persistent player identities):** C2's whole block (decision, rules, touches: time since / to, kind, release
  / arrival distance, duel, keeper, last / next actor; restarts), the 5-candidate owner ranker and its per-candidate
  features (distance, rank, speed, closing, margin, team, mates nearer, C2 flags, time since own touch), the touch-gated
  5-state decoder ("DEAD ends only at a touch", "CONTROL starts only at a touch", no direct team switch), the back-dating
  of CONTROL starts to touches and the owner fixed between touches, and the per-transition decoder offsets.
- **Kept from C3 base features (per frame, no identity):** ball position / speed / kick signal / turn / out-of-bounds / still
  ball / time since and until seen, pitch geometry, nearest-player distance, second distance, nearest team, margin to the
  nearest opponent, crowding counts, the nearest player's speed / closing / touchline distance, team activity at -4 ... +4 s,
  ball speed and nearest distance context.
- **Added (team level):** nearest home / away distance and their gap (with +-0.4 / 1 / 2 s context), time-to-ball per team and
  its difference (who gets there first), nearest team to the ball's position projected 0.5 s / 1 s ahead, team shape per
  side (centroid x / y, spread, deepest / highest player, players present, mean speed, centroid velocity, players in the
  opponent half) and differences, ball minus each centroid, ball_source / ball_missing / gap_length_so_far / estimated.
- **Stacking:** layer 2 gets layer 1's P(D) at -4 ... +4 s and its 4 s mean (inner out-of-fold for the training matches).
- **Ball:** the smoother rescaled to 25 fps (time-based; no repeat-frame rule; gaps <= 1 s), ball-from-players beyond that.
"""

DEVIATIONS = """\
- Module paths: `src/barca_possession/load.py` and `src/barca_train/degrade_pff.py` (inside packages, run with `python -m`), not `src/load.py`; degrade_pff and pipeline use the answer key, so they live in barca_train, outside the tag-free barca_possession package (M1 rule, enforced by tests/test_m1_labels.py).
- Decoder: 3 states (D / H / A) instead of C3's 5, on hybrid scaled likelihoods with transitions counted on the training labels;
  emission weight and switch scale chosen on the training matches' inner out-of-fold predictions (never the held-out fold).
- Smoother: `repeat_keep_every = 1` (PFF holds the ball two frames; nothing says Metrica does). load.py prints the share of
  identical consecutive ball positions to check this.
- Degraded PFF ball = PFF's *smoothed* ball minus the cut gaps (no raw measurement noise added), so the "seen" error is ~0 and
  says nothing about Metrica's raw ball noise. PFF itself already misses ~27 % of the ball.
- Unmeasurable degradation parameters are assumptions: identity swaps = 0.25 per track break, break gap 0.4 s, drop-outs take
  half of the measured slot-start rate (their length set by the measured share of absent players).
- Ball-from-players is trained directly on degraded PFF gap frames against the true ball (instead of M1's random hide times).
- Post-rule parameters were fixed a priori; after a smoke test on PFF fold 0 three were changed: kick-off searched in the first
  3 min only; set pieces anchored on the decoder's own restart and adding at most 8 s of D before the kick (an unseen ball is
  not a stopped ball); gap_same_team keeps D runs >= 2 s inside the gap. PFF only; the hackathon match was never used.
- The SET PIECES rule cannot be tested on PFF (no Smart Tagging): CV uses simulated tags from the answer key's restarts (60 %
  kept, interval -1..-4 s / +2..+6 s, 90 % right team, N/A near the touchline). Its CV effect is only indicative.
- 2nd-half kick-off: detected (first kick within 3 m of the centre spot, inside the Kick Off tag interval when present), not
  hard-coded (the rules forbid constants tied to frames of the game).
- Post-rules in the submission: kept automatically when their solo CV spellF1 gain is > 0 (cv_c3t.json "rules_kept").
- CV spellF1 / macroF1 are means over matches of the per-match scores (the hackathon is one match); pooled values are in
  reports/auto/cv_c3t.json. M1's C3 0.498 was pooled over matches on clean PFF, so it is not directly comparable.
- Coordinates: y kept as delivered (only centred and scaled); the 180-degree turn flips both axes.
"""


def j(name):
    p = AUTO / name
    return json.loads(p.read_text()) if p.exists() else None


def fmt_gap(g):
    lines = ["| split | " + " | ".join(g["bins"]) + " |", "|---|" + "---|" * len(g["bins"])]
    for k in [k for k in g if isinstance(g[k], dict) and "counts" in g[k]]:
        lines.append(f"| {k} | " + " | ".join(str(c) for c in g[k]["counts"]) + " |")
    return "\n".join(lines)


def main() -> int:
    out = ["# Step 1 report: Metrica loader, degraded PFF, ball filling, C3-T, post-rules, submission", "",
           "Aggregate numbers only (scores, counts, %). Generated by `scripts/make_step1_report.py` from `reports/auto/`.", "",
           "## Files created / changed", FILES, "## Commands (in order)", COMMANDS]
    ls = j("load_stats.json")
    out.append("## load.py statistics (hackathon match, graded frames)")
    if ls:
        g, s = ls["gap"], ls["slots"]
        o = ls.get("orientation", {})
        out += [f"- ball missing: **{100 * g['missing_frac_graded']:.1f} %** of graded frames ({100 * g['missing_frac_all']:.1f} % of all)",
                f"- mean slots present: home {s['mean_slots_home']:.2f}, away {s['mean_slots_away']:.2f} (total {s['mean_slots_total']:.2f}; "
                f"{s['n_slots']} slots; frames with a team > 11: {100 * s['frac_frames_team_over_11']:.1f} %)",
                "- attacking side: " + "; ".join(f"half {k}: {'home was on +x, flipped' if v['flipped'] else 'home on -x, kept'}"
                                                 for k, v in o.items()),
                f"- slot segment length: mean {s['mean_segment_s']:.1f} s (median {s['median_segment_s']:.1f} s); slot starts per minute: "
                f"{s['slot_starts_per_min']:.2f}",
                f"- ball: identical consecutive positions {100 * ls['ball']['repeat_frac']:.1f} %; third ball field p5 / median / p95: "
                f"{ls['ball']['third_field']['p5']} / {ls['ball']['third_field']['median']} / {ls['ball']['third_field']['p95']}",
                f"- ball-missing gaps: {g['n_gaps']} (counts per gap length)", "", fmt_gap(g), ""]
    else:
        out.append("_not run yet_\n")
    dg = j("degrade_stats.json")
    if dg:
        d = dg["degraded"]
        out += ["Degraded PFF against these targets (30 matches): ball missing "
                f"{100 * d['gap']['missing_frac_graded']:.1f} %, slots present {d['slots']['mean_slots_total']:.2f}, slot starts per "
                f"minute {d['slots']['slot_starts_per_min']:.2f}, mean segment {d['slots']['mean_segment_s']:.1f} s. Rates used: "
                + ", ".join(f"{k} {v:.4g}" for k, v in dg["rates"].items()), ""]
    out.append("## Ball position error on degraded PFF, by source (out-of-fold)")
    bf = j("ballfill_eval.json")
    if bf:
        out += ["| source | frames | median (m) | p90 (m) |", "|---|---|---|---|"]
        for k, v in bf["error"].items():
            if v["n"]:
                out.append(f"| {k} | {v['n']} | {v['median_m']:.2f} | {v['p90_m']:.2f} |")
        out += ["", "Share of in-period frames by source: " + ", ".join(f"{k} {100 * v:.1f} %" for k, v in bf["source_share"].items()), ""]
    else:
        out.append("_not run yet_\n")
    out += ["## C3 features removed / replaced", FEATURES, "## CV (5 folds, 30 degraded PFF learning matches; mean of per-match scores)"]
    cv = j("cv_c3t.json")
    if cv:
        out += ["| fold | spellF1 (no rules) | macroF1 (no rules) | spellF1 (all rules) | macroF1 (all rules) | decoder w / beta |",
                "|---|---|---|---|---|---|"]
        for fr in cv["folds"]:
            out.append(f"| {fr['fold']} | {fr['none']['spell']:.4f} | {fr['none']['macro']:.4f} | {fr['all']['spell']:.4f} | "
                       f"{fr['all']['macro']:.4f} | {fr['w']} / {fr['beta']} |")
        mn = cv["mean"]
        out += [f"| **mean** | **{mn['none']['spell']:.4f}** | **{mn['none']['macro']:.4f}** | **{mn['all']['spell']:.4f}** | "
                f"**{mn['all']['macro']:.4f}** | |", "",
                f"Pooled over matches (as M1 reported): no rules {mn['none']['pooled']['spell']:.4f} / {mn['none']['pooled']['macro']:.4f}; "
                f"all rules {mn['all']['pooled']['spell']:.4f} / {mn['all']['pooled']['macro']:.4f}. Threads {cv['threads']}, seed {cv['seed']}.", "",
                "### Score change from each post-rule alone (vs no rules)", "| rule | spellF1 | macroF1 |", "|---|---|---|"]
        for r, dv in cv["rule_delta"].items():
            out.append(f"| {r} | {dv['spell']:+.4f} | {dv['macro']:+.4f} |")
        out += ["", f"Rules kept for the submission (solo spellF1 gain > 0): {cv['rules_kept'] or 'none'}", ""]
    else:
        out.append("_not run yet_\n")
    out.append("## Submission")
    fp = j("fit_predict.json")
    if fp:
        out += [f"- validator: **{'PASS' if fp['validator_ok'] else 'FAIL'}** (exit {fp['validator_exit']}; reason {fp['validator_reason']}), "
                f"rows {fp['rows']}",
                f"- post-rules applied: {fp['rules'] or 'none'}; decoder w {fp['decoder']['w']}, beta {fp['decoder']['beta']}",
                "- graded label share: " + ", ".join(f"{k} {100 * v:.1f} %" for k, v in fp["label_share_graded"].items()),
                f"- predicted possessions: {fp['n_pred_spells']} (mean {fp['mean_spell_s']:.1f} s)",
                "- ball source on graded frames: " + ", ".join(f"{k} {100 * v:.1f} %" for k, v in fp["ball_source_share"].items()),
                f"- Smart Tagging: {fp['tags']}",
                f"- **public 5 min (sanity check only, never used to choose anything):** spellF1 {fp['public']['spell_f1']:.4f}, macroF1 "
                f"{fp['public']['macro_f1']:.4f} ({fp['public']['n_pred_spells']} predicted vs {fp['public']['n_gt_spells']} true "
                "possessions); baseline_smooth50 on the same window 0.3256 / 0.6207, its full-match spellF1 floor 0.2433.", ""]
    else:
        out.append("_not run yet_\n")
    out += ["## Deviations from the plan, problems, open questions", DEVIATIONS]
    (ROOT / "reports").mkdir(exist_ok=True)
    (ROOT / "reports" / "step1_report.md").write_text("\n".join(out), encoding="utf-8")
    print("wrote reports/step1_report.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
