"""Score per-frame predictions against the answer key of the HPC package (labels v2), with the same functions as our pipeline
(m1pc/m1score.py). Summary keys: acc, tol_0_2, tol_0_5, team, player, spell (LOOSE -> last team), spell_gap (LOOSE = gap),
rec_lt2 (spell recall < 2 s), restart_acc (restart team, matched restarts), loose_recall, poss_change_f1, and more."""
from __future__ import annotations

import numpy as np

from .m1score import merge, merge_had, merge_recall, recall_by_length, restart_metrics, score, score_had, tolerant_hits

FPS = 29.97


def label_of(d: dict) -> dict:
    return dict(state=d["label_state"], player_id=d["label_player_id"], team_id=d["label_team_id"], score_mask=d["label_score_mask"],
                period=d["period"])


def score_match(d: dict, state, player, team) -> dict:
    e = label_of(d)
    home = int(d["home_team_id"])
    state = np.asarray(state).copy()
    state[e["period"] <= 0] = 0
    s = score(e, state, player, team, FPS)
    s["had"] = score_had(e, state, team, home)
    s["had_gap"] = score_had(e, state, team, home, "gap")
    okm = e["score_mask"] & (e["state"] <= 3) & (e["period"] > 0)
    s["tol"] = dict(n=int(okm.sum()), **{f"{t:g}": int((tolerant_hits(e["state"], state, okm, int(round(t * FPS))) & okm).sum())
                                         for t in (0.2, 0.5)})
    s["rec_len"] = recall_by_length(e, state, team, home, FPS, "last")
    s["rec_len_gap"] = recall_by_length(e, state, team, home, FPS, "gap")
    s["restart"] = restart_metrics(e, state, team, home, FPS)
    return s


def summarize(scores: list[dict]) -> dict:
    m = merge(scores)
    c = np.asarray(m["confusion"], float)
    hl, hg = merge_had([s["had"] for s in scores]), merge_had([s["had_gap"] for s in scores])
    n = sum(s["tol"]["n"] for s in scores)
    rl = merge_recall([s["rec_len"] for s in scores])
    r = {k: sum(s["restart"][k] for s in scores) for k in ("n", "matched", "matched_ok", "at_ok", "extra_after_dead")}
    return dict(acc=m["accuracy"], tol_0_2=sum(s["tol"]["0.2"] for s in scores) / n, tol_0_5=sum(s["tol"]["0.5"] for s in scores) / n,
                team=m["team_acc"], player=m["player_acc"], spell=hl["spell_f1"], spell_gap=hg["spell_f1"], macro=hl["macro_f1"],
                rec_lt2=rl["< 2 s"]["recall"], rec_2_5=rl["2-5 s"]["recall"],
                restart_acc=r["matched_ok"] / max(r["matched"], 1), restart_matched_share=r["matched"] / max(r["n"], 1),
                extra_after_dead=r["extra_after_dead"],
                loose_recall=c[3, 3] / max(c[3].sum(), 1), poss_change_f1=m["possession_change"]["f1"],
                dead_end_0_3s=m["dead_end_0.3s"]["f1"], dead_end_2s=m["dead_end_2s"]["f1"], frames=m["frames"])


METRICS = [("acc", "frame accuracy"), ("tol_0_2", "tolerant accuracy ±0.2 s"), ("tol_0_5", "tolerant accuracy ±0.5 s"),
           ("team", "team accuracy (CONTROL)"), ("player", "player accuracy (CONTROL)"), ("spell", "spell-F1, LOOSE → last team"),
           ("spell_gap", "spell-F1, LOOSE → neither team"), ("rec_lt2", "spell recall < 2 s"), ("restart_acc", "restart-team accuracy"),
           ("loose_recall", "LOOSE recall"), ("poss_change_f1", "possession-change F1")]
