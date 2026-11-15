"""M1: how far is the final C3 really from the answer key (labels v2, held-out predictions, 30 learning matches)?

a. wrong frames by distance to the nearest state / team change in the answer key; timing-tolerant frame accuracy (a frame is
   right if the answer key has the predicted state within +-0.2 / +-0.5 / +-1.0 s).
b. team-level H / A spells (same H/A/D view as spell-F1): counts per match, what happens to each true spell (matched IoU >= 0.5,
   fragmented, merged, boundary, missed), what the unmatched predicted spells are, length histograms.
c. share of wrong frames where our smoothed ball is missing.

    python scripts/m1/diag_v2.py [--pred-dir data/processed/m1_possession/c3v2/pred/combo]
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np

from barca_labels.scoring import _spells, had
from barca_possession.smoother import FLAG_MISSING
from barca_possession.splits import LEARNING_MATCHES

BASE = Path("data/processed/m1_possession")
OUT = BASE / "diag_v2" / "diag.json"
TOLS = (0.2, 0.5, 1.0)
BINS = (0, 0.5, 1, 2, 5, 10, 20, 40, np.inf)


def tolerant_hits(true, pred, ok, k):
    """pred[i] == true[j] for some |i - j| <= k (j must also be a scored frame)."""
    hit = (pred == true) & ok
    n = len(true)
    for d in range(1, k + 1):
        for s in (d, -d):
            lo, hi = max(0, -s), min(n, n - s)
            idx = np.arange(lo, hi)
            hit[idx] |= (pred[idx] == true[idx + s]) & ok[idx + s]
    return hit


def spell_fate(lb, le, pb, pe, lc, pc):
    """Classify every true spell and every predicted spell (same team only)."""
    pairs_t, pairs_p = [[] for _ in lb], [[] for _ in pb]
    for i, (b, e, c) in enumerate(zip(lb, le, lc)):
        lo, hi = np.searchsorted(pe, b, "right"), np.searchsorted(pb, e, "left")
        for j in range(lo, hi):
            if pc[j] == c:
                inter = min(e, pe[j]) - max(b, pb[j])
                if inter > 0:
                    iou = inter / (max(e, pe[j]) - min(b, pb[j]))
                    pairs_t[i].append((j, inter, iou)); pairs_p[j].append((i, inter, iou))
    tf = []
    for i, ps in enumerate(pairs_t):
        L = le[i] - lb[i]
        cover = sum(x[1] for x in ps) / L
        if any(x[2] >= 0.5 for x in ps):
            tf.append("matched")
        elif cover < 0.5:
            tf.append("missed")
        elif len([x for x in ps if x[1] >= 0.1 * L]) >= 2:
            tf.append("fragmented")
        elif any(len(pairs_p[j]) >= 2 for j, _, _ in ps):
            tf.append("merged")
        else:
            tf.append("boundary")
    pf = []
    for j, ps in enumerate(pairs_p):
        L = pe[j] - pb[j]
        if any(x[2] >= 0.5 for x in ps):
            pf.append("matched")
        elif sum(x[1] for x in ps) / L < 0.5:
            pf.append("spurious")
        elif len(ps) >= 2:
            pf.append("merges several")
        elif ps and (le[ps[0][0]] - lb[ps[0][0]]) > L:
            pf.append("piece of a longer true spell")
        else:
            pf.append("boundary")
    return tf, pf


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred-dir", default=str(BASE / "c3v2" / "pred" / "combo"))
    a = ap.parse_args()
    home = json.loads((BASE / "home_team.json").read_text())
    acc = Counter()
    dist_hist = Counter()
    miss = Counter()
    per_match, tfate, pfate = {}, Counter(), Counter()
    tfate_frames, len_t, len_p = Counter(), [], []
    fate_len, missed_pred = {}, Counter()
    for m in LEARNING_MATCHES:
        e = dict(np.load(BASE / "evalset_v2" / f"{m}.npz"))
        p = np.load(Path(a.pred_dir) / f"{m}.npz")
        z = np.load(BASE / "inputs" / f"{m}.npz")
        fps = float(z["fps"])
        st, tm, per = e["state"], e["team_id"], e["period"]
        ok = e["score_mask"] & (st <= 3) & (per > 0)
        ps = p["state"]
        wrong = ok & (ps != st)
        acc["n"] += int(ok.sum()); acc["strict"] += int((ok & ~wrong).sum())
        for t in TOLS:
            acc[f"tol_{t}"] += int((tolerant_hits(st, ps, ok, int(round(t * fps))) & ok).sum())
        # a: distance of each wrong frame to the nearest answer-key state / team change (same period)
        ch = np.flatnonzero(((st[1:] != st[:-1]) | (tm[1:] != tm[:-1])) & (per[1:] == per[:-1])) + 1
        w = np.flatnonzero(wrong)
        if len(ch):
            j = np.clip(np.searchsorted(ch, w), 1, len(ch) - 1) if len(ch) > 1 else np.zeros(len(w), int)
            d = np.minimum(np.abs(w - ch[j - 1 if len(ch) > 1 else 0]), np.abs(w - ch[j])) / fps
        else:
            d = np.full(len(w), np.inf)
        for t in TOLS:
            dist_hist[f"within_{t}"] += int((d <= t).sum())
        dist_hist["wrong"] += len(w)
        # c: ball missing
        bm = z["bflag"] == FLAG_MISSING
        miss["wrong_missing"] += int((wrong & bm).sum()); miss["scored_missing"] += int((ok & bm).sum())
        miss["right_missing"] += int((ok & ~wrong & bm).sum())
        # b: spells
        ht = int(home[str(m)])
        idx = np.flatnonzero(per > 0)
        lv, pv = had(st, tm, per, ht)[idx], had(ps, p["team"], per, ht)[idx]
        lb, le, lc = _spells(lv, per[idx], (0, 1))
        pb, pe, pc = _spells(pv, per[idx], (0, 1))
        tf, pf = spell_fate(lb, le, pb, pe, lc, pc)
        tfate.update(tf); pfate.update(pf)
        for f, b0, e0, c0 in zip(tf, lb, le, lc):
            tfate_frames[f] += int(e0 - b0)
            fate_len.setdefault(f, []).append((e0 - b0) / fps)
            if f == "missed":
                seg = pv[b0:e0]
                missed_pred["other team"] += int(((seg != c0) & (seg >= 0) & (seg < 2)).sum())
                missed_pred["DEAD"] += int((seg == 2).sum()); missed_pred["same team"] += int((seg == c0).sum())
        for f, b0, e0 in zip(pf, pb, pe):
            fate_len.setdefault("pred " + f, []).append((e0 - b0) / fps)
        len_t += ((le - lb) / fps).tolist(); len_p += ((pe - pb) / fps).tolist()
        per_match[str(m)] = dict(n_true=len(lb), n_pred=len(pb), acc=float((ok & ~wrong).sum() / ok.sum()))
    ht_, _ = np.histogram(len_t, BINS); hp_, _ = np.histogram(len_p, BINS)
    res = dict(
        accuracy={k: v / acc["n"] for k, v in acc.items() if k != "n"}, frames=acc["n"],
        wrong_near_change={k: v / dist_hist["wrong"] for k, v in dist_hist.items() if k != "wrong"}, wrong=dist_hist["wrong"],
        ball_missing=dict(share_of_wrong=miss["wrong_missing"] / dist_hist["wrong"], share_of_scored=miss["scored_missing"] / acc["n"],
                          acc_when_missing=miss["right_missing"] / max(miss["scored_missing"], 1),
                          acc_when_seen=(acc["strict"] - miss["right_missing"]) / (acc["n"] - miss["scored_missing"])),
        true_spells=dict(tfate), true_spell_frames=dict(tfate_frames), pred_spells=dict(pfate), per_match=per_match,
        hist=dict(bins=[str(b) for b in BINS], true=ht_.tolist(), pred=hp_.tolist()),
        fate_median_len_s={k: float(np.median(v)) for k, v in fate_len.items()},
        missed_spell_predicted_as={k: v / sum(missed_pred.values()) for k, v in missed_pred.items()},
        median_len=dict(true=float(np.median(len_t)), pred=float(np.median(len_p))))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(res, indent=1))
    print(json.dumps({k: v for k, v in res.items() if k != "per_match"}, indent=1))
    nt = np.array([v["n_true"] for v in per_match.values()]); npd = np.array([v["n_pred"] for v in per_match.values()])
    print(f"spells per match: true median {np.median(nt):.0f} (range {nt.min()}-{nt.max()}), predicted median {np.median(npd):.0f} "
          f"(range {npd.min()}-{npd.max()}); predicted / true per match median {np.median(npd / nt):.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
