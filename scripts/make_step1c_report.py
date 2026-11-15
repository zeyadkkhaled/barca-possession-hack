"""Write reports/step1c_report.md from the aggregate files in reports/auto (step 1c and 1c-v2).

    .venv/Scripts/python.exe scripts/make_step1c_report.py [--regenerated yes|no] [--chosen current|robust|trim11]
"""
from __future__ import annotations

import argparse
import json

from barca_possession.paths import AUTO, ROOT

ARMS = (("current", "cv_extra_current.json", "current team features"),
        ("robust", "cv_extra_robust.json", "A: median centre, IQR spread, p10 / p90 depth, median speeds"),
        ("trim11", "cv_extra_trim11.json", "B: trim each team to 11 per frame (newest-born slots dropped)"))


def j(name):
    f = AUTO / name
    return json.loads(f.read_text()) if f.exists() else None


def pct(x, d=1):
    return "–" if x is None else f"{100 * x:.{d}f} %"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--regenerated", default="no")
    ap.add_argument("--chosen", default="current")
    a = ap.parse_args()
    L = ["# Step 1c: extra slots, robust team features, stress matches", "",
         "Aggregate numbers only; nothing per frame was printed or read by the assistant. CV = 5 folds of the 30 PFF "
         "learning matches (M1 folds.json), scorer `score_had` (= the official scorer without X), LOOSE → last team, no "
         "post-rules. The 16 locked test matches were never read (`barca_train.stress` refuses them).", ""]
    ld, sf, dd = j("load_stats.json"), j("dup_safety.json"), j("extra_slot_diag.json")
    du = (ld or {}).get("dup", {})
    L += ["## 1. The 0.5 m duplicate rule (kept, minor)", "",
          "Rule: in a frame where a team has more than 11 slots, a same-team pair under 0.5 m apart with velocities "
          "differing by under 1 m/s is merged by dropping the shorter-lived slot.", "",
          "| | value |", "|---|---|",
          f"| match: graded frames with a team > 11, before → after | {pct(du.get('over11_before'))} → {pct(du.get('over11_after'))} |",
          f"| match: frames with a merge | {pct(du.get('frames_with_drop_frac'), 2)} |",
          f"| clean PFF: frames where the rule merges two real teammates | {pct((sf or {}).get('rule_merges_real_teammates_frac'), 3)} |",
          f"| clean PFF: same pair test without the >11 gate | {pct((sf or {}).get('close_pair_any_frame_frac'), 2)} |", "",
          "The extra slots are not near-copies (nearest same-team pair median 3.6 m on over-11 frames; "
          "reports/auto/dup_diag.json), so the rule barely acts. It stays in the pipeline (`--dedup`) because it is safe.", ""]
    if dd:
        c, o, t = dd["classification"], dd["other_team_count"], dd["total_count"]
        er, rr = dd["extra_runs"], dd["regular_runs"]
        ep = dd.get("episodes", {})
        L += ["## 2. What the extra slot is (match, frames where a team has > 11)", "",
              f"Graded frames with a team over 11: {pct(dd['frames_with_team_over11_frac'])} ({dd['team_frames_over11']} team-frames). "
              "The extra slot is the team's most recently born slot in that frame.", "",
              "| measure | value |", "|---|---|",
              f"| other team's slots ≤10 / 11 / >11 | {pct(o['<=10'])} / {pct(o['11'])} / {pct(o['>11'])} |",
              f"| total slots ≤22 / 23 / >23 | {pct(t['<=22'])} / {pct(t['23'])} / {pct(t['>23'])} |",
              f"| run share within 15 m of play, median: extra / regular | {er['share_near_play']['p50']:.2f} / {rr['share_near_play']['p50']:.2f} |",
              f"| run mean speed, median m/s: extra / regular | {er['mean_speed_mps']['p50']:.2f} / {rr['mean_speed_mps']['p50']:.2f} |",
              f"| run length, median s: extra / regular | {er['length_s']['p50']:.1f} / {rr['length_s']['p50']:.1f} |",
              "| team label flips within a slot | 0 % (the team is fixed per slot in the metadata XML) |",
              f"| run starts ≤2 m / ≤1 s after an OTHER-team slot ended: extra / regular | {pct(dd['handover_from_other_team']['extra'])} / {pct(dd['handover_from_other_team']['regular'])} |",
              f"| same, SAME-team slot: extra / regular | {pct(dd['handover_from_same_team']['extra'])} / {pct(dd['handover_from_same_team']['regular'])} |",
              "", "**Classification** (share of over-11 team-frames):", "",
              f"- (b) wrong team label (total 22, other team 10): **{pct(c['b_wrong_label'])}**",
              f"- (a) extra person (total 23, other team 11): **{pct(c['a_extra_person'])}**",
              f"- other (e.g. total ≤ 21 with the other team ≤ 9, or > 23): {pct(c['other'])}", ""]
        if ep:
            L += ["Over-11 episodes (consecutive frames, per team): " + "; ".join(
                f"{k}: {v['n']}, median {v['median_s']:.1f} s, mean {v['mean_s']:.1f} s, {pct(v['frame_frac'])} of graded frames"
                for k, v in ep.items()) + ".", "",
                  "Reading: mostly (b), a player of the other team carried under this team's label for a few seconds "
                  "(as hand-overs between slots happen to regular slots too, ~10 %, it looks like the tracker's team "
                  "assignment slipping near opponents). The extra slot moves like a player (same speed) and is somewhat "
                  "less often near play.", ""]
    dx = j("degrade_stats_extra.json")
    if dx:
        x = dx["degraded"]
        L += ["## 3. Simulation in degrade_pff (`--extra`)", "",
              "On top of the step-1 degradation (dead_weight 8), seeded per match:",
              "- (b) a real player's track moves for an episode into a new slot of the other team;",
              "- (a) a referee-like path is added to a team: the true ball smoothed with a 3 s lag, plus a fixed 5–12 m offset.",
              "- Episode lengths are drawn from the match's histograms and episodes start only where the receiving team "
              "already has ≥ 11 slots.", "",
              f"Result: frames with a team > 11 {pct(x['slots']['frac_frames_team_over_11'])} (match "
              f"{pct((ld or {}).get('slots', {}).get('frac_frames_team_over_11'))}); slots {x['slots']['mean_slots_total']:.2f} "
              f"(match {(ld or {}).get('slots', {}).get('mean_slots_total', float('nan')):.2f}); slot starts "
              f"{x['slots']['slot_starts_per_min']:.2f}/min (match {(ld or {}).get('slots', {}).get('slot_starts_per_min', float('nan')):.2f}); "
              f"wrong-label share of the episodes {x['extra']['share_b']:.2f}.", ""]
    L += ["## 4. CV on degraded PFF + extra slots (duplicate rule on in every arm)", "",
          "| features | spellF1 | macroF1 | spellF1 pooled | restart-team acc (matched share) | short-spell (<2 s) recall |",
          "|---|---|---|---|---|---|"]
    base = None
    for key, f, desc in ARMS:
        r = j(f)
        if r is None:
            L.append(f"| {desc} | not run | | | | |")
            continue
        n, e = r["mean"]["none"], r["extra"]
        base = base if base is not None else (n["spell"] if key == "current" else None)
        d = f" ({n['spell'] - base:+.4f})" if base is not None and key != "current" else ""
        L.append(f"| {desc} | {n['spell']:.4f}{d} | {n['macro']:.4f} | {n['pooled']['spell']:.4f} | "
                 f"{e['restart_team_acc']:.3f} ({e['restart_matched_share']:.3f}) | {e['short_spell_recall']:.3f} |")
    s1 = j("cv_c3t.json")
    if s1:
        L += ["", f"Step 1 for reference (degraded, no extra slots, no rule): spellF1 {s1['mean']['none']['spell']:.4f}, "
                  f"macroF1 {s1['mean']['none']['macro']:.4f}."]
    L += ["", "Decision rule: an option replaces the current features only if it beats them by ≥ 0.005 spellF1 (mean per match).",
          f"Chosen: **{a.chosen}**. Submission regenerated: **{a.regenerated}**.", ""]
    fp = j("fit_predict.json")
    if fp and a.regenerated == "yes":
        L += ["## 5. Submission", "",
              f"- validator: {'PASS' if fp['validator_ok'] else 'FAIL'} ({fp['rows']} rows)",
              "- graded label share: " + ", ".join(f"{k} {100 * v:.1f} %" for k, v in fp["label_share_graded"].items()),
              f"- predicted possessions: {fp['n_pred_spells']}, mean {fp['mean_spell_s']:.1f} s",
              f"- public 5 min (sanity only, never used for a choice): spellF1 {fp['public']['spell_f1']:.4f}, "
              f"macroF1 {fp['public']['macro_f1']:.4f}", ""]
    L += ["## 6. Stress-eval matches (6 M1 stress matches, never trained on)", "",
          "| model | input | spellF1 | macroF1 | spellF1 pooled | restart-team acc (matched) | short-spell recall |",
          "|---|---|---|---|---|---|---|"]
    for name, lab in (("stress_eval.json", "step 1 (current)"), ("stress_eval_final.json", "step 1c final")):
        r = j(name)
        if r is None:
            continue
        for v in ("clean", "degraded"):
            x, e = r[v], r[v]["extra"]
            L.append(f"| {lab} | {v} | {x['spell_mean']:.4f} | {x['macro_mean']:.4f} | {x['pooled']['spell']:.4f} | "
                     f"{e['restart_team_acc']:.3f} ({e['restart_matched_share']:.3f}) | {e['short_spell_recall']:.3f} |")
    L += ["", "\"degraded\" = the same degradation as the model's training data (the step-1 model: without extra slots).", ""]
    (ROOT / "reports" / "step1c_report.md").write_text("\n".join(L), encoding="utf-8")
    print("wrote reports/step1c_report.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
