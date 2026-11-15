"""M1: what are the short possessions C3 misses, and what are its extra spells? (labels v2, held-out C3 predictions, 30
learning matches only; no training).

a. every missed true H / A spell (same view and fate rules as scripts/m1/diag_v2.py) gets one class from what the answer key has
   inside it, checked in this order:
   restart         the spell starts right after a DEAD stretch (a set piece that is lost again quickly);
   keeper catch    a goalkeeper (roster positionGroupType GK) is in CONTROL inside the spell;
   LOOSE only      no CONTROL / TRAVELLING frame of that team: the spell exists only because LOOSE is mapped to the last team;
   interception, one touch, lost again   the other team holds the ball right before and right after; one player in CONTROL
                   for <= 1 s;
   interception, carry / several touches, lost again   same, but longer CONTROL or several players;
   single touch    one player in CONTROL for <= 1 s, other context;
   other.
b. every extra (spurious) predicted spell gets a class from the answer key inside it: mostly DEAD, mostly the other team in
   CONTROL, mostly the other team's ball in flight (TRAVELLING), mostly LOOSE, unscored, mixed; plus the flags "starts right after
   a predicted DEAD" and "predicted LOOSE only".
c. spell-F1 with LOOSE mapped to (1) the last team in control (current) and (2) a gap that belongs to neither team.
d. spell recall by true spell length (< 2, 2-5, 5-15, > 15 s; matched = one-to-one greedy at IoU >= 0.5, as in spell-F1).

The answer key, rosters and PFF events are read only to describe and score; nothing here feeds a model.

    python scripts/m1/spells_v3.py [--pred-dir data/processed/m1_possession/c3v2/pred/combo] [--out .../spells_v3/combo.json]
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from c3_error_videos import load_events  # noqa: E402
from diag_v2 import spell_fate  # noqa: E402
from barca_labels.scoring import _spells, had, merge_had, score_had  # noqa: E402
from barca_possession.splits import LEARNING_MATCHES  # noqa: E402

BASE = Path("data/processed/m1_possession")
ROSTERS = Path("data/raw/FIFA World Cup 2022/Rosters")
LEN_BINS = (0.0, 2.0, 5.0, 15.0, np.inf)
LEN_NAMES = ("< 2 s", "2-5 s", "5-15 s", "> 15 s")
HOST = "epitome-staging.pff.com"
N_EX = 3


def keepers(mid: int) -> set[int]:
    r = json.loads((ROSTERS / f"{mid}.json").read_text(encoding="utf-8"))
    return {int(p["player"]["id"]) for p in r if p.get("positionGroupType") == "GK"}


def matched_true(lb, le, lc, pb, pe, pc, thr=0.5):
    """One-to-one greedy matching at IoU >= thr (the spell-F1 rule); returns a bool per true spell."""
    pairs = []
    for i, (b, e, c) in enumerate(zip(lb, le, lc)):
        lo, hi = np.searchsorted(pe, b, "right"), np.searchsorted(pb, e, "left")
        for j in range(lo, hi):
            if pc[j] == c:
                inter = min(e, pe[j]) - max(b, pb[j])
                if inter > 0:
                    pairs.append((inter / (max(e, pe[j]) - min(b, pb[j])), i, j))
    pairs.sort(reverse=True)
    ua, ub = set(), set()
    for iou, i, j in pairs:
        if iou < thr:
            break
        if i not in ua and j not in ub:
            ua.add(i); ub.add(j)
    out = np.zeros(len(lb), bool)
    out[list(ua)] = True
    return out


def recall_by_length(e, ps, pt, home, fps, loose="last"):
    """Per length bin: [matched, n_true] of true H / A spells."""
    per = e["period"]
    idx = np.flatnonzero(per > 0)
    lv = had(e["state"], e["team_id"], per, home, loose)[idx]
    pv = had(ps, pt, per, home, loose)[idx]
    lb, le, lc = _spells(lv, per[idx], (0, 1))
    pb, pe, pc = _spells(pv, per[idx], (0, 1))
    ok = matched_true(lb, le, lc, pb, pe, pc)
    b = np.digitize((le - lb) / fps, LEN_BINS[1:-1])
    return {LEN_NAMES[k]: [int(ok[b == k].sum()), int((b == k).sum())] for k in range(len(LEN_NAMES))}


def merge_recall(rows):
    tot = {k: [0, 0] for k in LEN_NAMES}
    for r in rows:
        for k, (a, n) in r.items():
            tot[k][0] += a; tot[k][1] += n
    return {k: dict(recall=a / max(n, 1), matched=a, n=n) for k, (a, n) in tot.items()}


def link(ev, t0):
    near = min((r for r in ev if r["url"]), key=lambda r: abs(r["t"] - t0))
    return re.sub(r"/[0-9.]+$", f"/{t0:.3f}", near["url"]).replace("epitome.pff.com", HOST), near["clock"]


def classify_missed(b, e, c, lv, st, pl, tm_side, gk, fps):
    """b, e, c: true spell in the in-period index space; lv: true H/A/D view; st, pl: true state / player; tm_side: true
    team as 0 H / 1 A / -1 (all already restricted to in-period frames)."""
    seg_st, seg_pl, seg_side = st[b:e], pl[b:e], tm_side[b:e]
    own = np.isin(seg_st, (1, 2)) & (seg_side == c)
    ctrl = (seg_st == 1) & (seg_side == c)
    players = set(seg_pl[ctrl].tolist())
    n_ctrl_s = ctrl.sum() / fps
    prev_v = lv[b - 1] if b > 0 else -1
    next_v = lv[e] if e < len(lv) else -1
    flags = dict(restart=bool(prev_v == 2), keeper=bool(players & gk), loose_only=bool(not own.any()),
                 sandwich=bool(prev_v == 1 - c and next_v == 1 - c), single=bool(len(players) == 1 and n_ctrl_s <= 1.0))
    if flags["restart"]:
        k = "restart"
    elif flags["keeper"]:
        k = "keeper catch"
    elif flags["loose_only"]:
        k = "LOOSE only (last-team mapping)"
    elif flags["sandwich"] and flags["single"]:
        k = "interception, one touch, lost again"
    elif flags["sandwich"]:
        k = "interception, carry / several touches, lost again"
    elif flags["single"]:
        k = "single touch, other context"
    else:
        k = "other"
    return k, flags


def classify_extra(b, e, c, st, tm_side, ok, pv, pst):
    seg = slice(b, e)
    n = e - b
    s, sd, okk = st[seg], tm_side[seg], ok[seg]
    shares = {"DEAD": np.mean(okk & (s == 0)), "other team CONTROL": np.mean(okk & (s == 1) & (sd == 1 - c)),
              "other team TRAVELLING": np.mean(okk & (s == 2) & (sd == 1 - c)), "LOOSE": np.mean(okk & (s == 3)),
              "same team": np.mean(okk & np.isin(s, (1, 2)) & (sd == c)), "unscored": np.mean(~okk)}
    top = max(shares, key=shares.get)
    k = top if shares[top] >= 0.5 else "mixed"
    flags = dict(after_pred_dead=bool(b > 0 and pv[b - 1] == 2), pred_loose_only=bool(not np.isin(pst[seg], (1, 2)).any()))
    return k, flags


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pred-dir", default=str(BASE / "c3v2" / "pred" / "combo"))
    ap.add_argument("--out", default=str(BASE / "spells_v3" / "combo.json"))
    a = ap.parse_args()
    home = json.loads((BASE / "home_team.json").read_text())
    rng = np.random.default_rng(20261109)
    miss_cls, miss_flags, miss_len, miss_rule = Counter(), Counter(), defaultdict(list), defaultdict(Counter)
    ext_cls, ext_flags, ext_len = Counter(), Counter(), defaultdict(list)
    miss_says = defaultdict(Counter)
    ex_m, ex_e = defaultdict(list), defaultdict(list)
    had_last, had_gap, rec_last, rec_gap = [], [], [], []
    n_miss = n_ext = 0
    for m in LEARNING_MATCHES:
        e = dict(np.load(BASE / "evalset_v2" / f"{m}.npz"))
        p = np.load(Path(a.pred_dir) / f"{m}.npz")
        z = np.load(BASE / "inputs" / f"{m}.npz")
        fps, t = float(z["fps"]), z["t"]
        ht = int(home[str(m)])
        per = e["period"]
        idx = np.flatnonzero(per > 0)
        st, pl, tm = e["state"][idx], e["player_id"][idx], e["team_id"][idx]
        tm_side = np.where(tm < 0, -1, np.where(tm == ht, 0, 1))
        ok = (e["score_mask"] & (e["state"] <= 3))[idx]
        lv = had(e["state"], e["team_id"], per, ht)[idx]
        pv = had(p["state"], p["team"], per, ht)[idx]
        pst = p["state"][idx]
        lb, le, lc = _spells(lv, per[idx], (0, 1))
        pb, pe, pc = _spells(pv, per[idx], (0, 1))
        tf, pf = spell_fate(lb, le, pb, pe, lc, pc)
        gk = keepers(m)
        rn = e["rule_names"][e["rule"][idx]]
        cand_m, cand_e = defaultdict(list), defaultdict(list)
        for f, b0, e0, c0 in zip(tf, lb, le, lc):
            if f != "missed":
                continue
            k, fl = classify_missed(b0, e0, c0, lv, st, pl, tm_side, gk, fps)
            n_miss += 1
            miss_cls[k] += 1
            miss_flags.update(x for x, v in fl.items() if v)
            miss_len[k].append((e0 - b0) / fps)
            miss_rule[k].update(rn[b0:e0].tolist())
            sg = pv[b0:e0]
            miss_says[k].update({"other team": int(((sg == 1 - c0)).sum()), "DEAD": int((sg == 2).sum()),
                                 "same team": int((sg == c0).sum())})
            cand_m[k].append((m, float(t[idx[b0]]), float(t[idx[e0 - 1]]), "H" if c0 == 0 else "A"))
        for f, b0, e0, c0 in zip(pf, pb, pe, pc):
            if f != "spurious":
                continue
            k, fl = classify_extra(b0, e0, c0, st, tm_side, ok, pv, pst)
            n_ext += 1
            ext_cls[k] += 1
            ext_flags.update(x for x, v in fl.items() if v)
            ext_len[k].append((e0 - b0) / fps)
            cand_e[k].append((m, float(t[idx[b0]]), float(t[idx[e0 - 1]]), "H" if c0 == 0 else "A"))
        ev = None
        for store, cand in ((ex_m, cand_m), (ex_e, cand_e)):
            for k, v in cand.items():
                for row in v:
                    store[k].append(row)
        had_last.append(score_had(e, p["state"], p["team"], ht, "last"))
        had_gap.append(score_had(e, p["state"], p["team"], ht, "gap"))
        rec_last.append(recall_by_length(e, p["state"], p["team"], ht, fps, "last"))
        rec_gap.append(recall_by_length(e, p["state"], p["team"], ht, fps, "gap"))
    ev_cache = {}

    def examples(store):
        out = {}
        for k, rows in store.items():
            pick = rng.choice(len(rows), size=min(N_EX, len(rows)), replace=False)
            out[k] = []
            for i in sorted(pick):
                m, t0, t1, side = rows[i]
                if m not in ev_cache:
                    ev_cache[m] = load_events(m)
                url, clock = link(ev_cache[m], t0)
                out[k].append(dict(match=m, video_start=round(t0, 1), video_end=round(t1, 1), team=side, clock=clock, url=url))
        return out

    res = dict(
        missed=dict(n=n_miss, share={k: v / n_miss for k, v in miss_cls.most_common()},
                    median_len_s={k: float(np.median(v)) for k, v in miss_len.items()},
                    flags_any={k: v / n_miss for k, v in miss_flags.items()},
                    top_rules={k: [(r, round(c / sum(v.values()), 3)) for r, c in v.most_common(4)] for k, v in miss_rule.items()},
                    c3_says={k: {a: round(b / sum(v.values()), 3) for a, b in v.items()} for k, v in miss_says.items()},
                    examples=examples(ex_m)),
        extra=dict(n=n_ext, share={k: v / n_ext for k, v in ext_cls.most_common()},
                   median_len_s={k: float(np.median(v)) for k, v in ext_len.items()},
                   flags_any={k: v / n_ext for k, v in ext_flags.items()}, examples=examples(ex_e)),
        spell_f1=dict(loose_last_team=merge_had(had_last), loose_gap=merge_had(had_gap)),
        recall_by_length=dict(loose_last_team=merge_recall(rec_last), loose_gap=merge_recall(rec_gap)))
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=1, default=float), encoding="utf-8")
    print(json.dumps({k: v for k, v in res.items()}, indent=1, default=float)[:12000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
