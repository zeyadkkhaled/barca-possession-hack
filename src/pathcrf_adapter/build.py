"""Build PathCRF inputs (and C3-T features) from the M1 PFF package, ON THE HPC. Nothing here needs the hackathon data:
the degradation reads only aggregate rates (reports/auto/*.json, shipped with the code).

Per learning match (seeded exactly like step 1c: degrade_pff seed 0 + match id, dead_weight 8, extra-slot simulation):
1. degrade_pff.degrade_match(..., extra=..., owner=True): Metrica-like slots (no ids), 25 fps, missing ball, extra slots;
   the owner map (true player per slot and frame) is kept for the targets only;
2. the duplicate rule (as C3-T: c3t.prepare_players(dedup=True));
3. ball: ballfill.fill_ball with models/bfp (filled x / y; seen = smoothed or seen sample);
4. C3-T features (current mode) -> <out>/c3t/<match>.npz (for the fold-0 C3-T on the HPC);
5. trim to 11 + Hungarian re-linking (barca_possession.relink) -> 22 lanes (0-10 home, 11-21 away);
6. targets per 25 fps frame (labels v2, as M1's adapter, but on lanes): CONTROL by player p -> self-link on the lane that
   holds p; TRAVELLING / LOOSE -> link from the last controller's lane to the next controller's lane, or to the outside
   node of the side where the ball is when the DEAD stretch starts; masked when a player is in no lane at that frame.
   -> <out>/pathcrf/<match>.npz
Coordinates: our convention (centre origin, metres, pitch 105.3 x 68, home attacks +x in every period), so the home team
is always PathCRF's left team. Writes <out>/build_stats.json (aggregates) and copies folds.json.

    python -m pathcrf_adapter.build --package DATA --out BUILT [--workers 4] [--matches ...]
"""
from __future__ import annotations

import argparse
import json
import multiprocessing
import shutil
import time
from pathlib import Path

import numpy as np

from barca_possession import ballfill as BF
from barca_possession import c3t as T
from barca_possession.paths import AUTO, FPS, MODELS
from barca_possession.relink import TEAM, link_stats, relink, window_stats
from barca_train import degrade_pff as DP

DEAD, CONTROL, TRAV, LOOSE = 0, 1, 2, 3
K_MASK, K_SELF, K_LINK, K_OUT = 0, 1, 2, 3
SEED = 0


def config() -> dict:
    """The step-1c training degradation, from the aggregate files in reports/auto."""
    deg = json.loads((AUTO / "degrade_stats_extra.json").read_text())
    return dict(stats=json.loads((AUTO / "load_stats.json").read_text()), dead_weight=float(deg["degraded"]["dead_weight"]),
                extra=deg["degraded"]["extra"], seed=int(deg.get("seed", SEED)))


def side_out(bx, by, L, W):
    """Outside node nearest to a ball position (centre origin): 0 left, 1 right, 2 bottom, 3 top; -1 if unknown."""
    if not np.isfinite(bx) or not np.isfinite(by):
        return -1
    return int(np.argmin((bx + L / 2, L / 2 - bx, by + W / 2, W / 2 - by)))


def player_targets(state, player, scored, period, bx, by, L, W):
    """Per frame, in true-player space: kind, a (player), b (player or outside index), tl (0 travelling / 1 loose / -1)."""
    n = len(state)
    last = np.full(n, -1); nxt = np.full(n, -1); nxt_out = np.full(n, -1)
    cur = -1
    for i in range(n):
        if i and period[i] != period[i - 1]:
            cur = -1
        if state[i] == DEAD:
            cur = -1
        elif state[i] == CONTROL and player[i] >= 0:
            cur = player[i]
        last[i] = cur
    cur, cur_out = -1, -1
    for i in range(n - 1, -1, -1):
        if i < n - 1 and period[i] != period[i + 1]:
            cur, cur_out = -1, -1
        if state[i] == DEAD:
            cur, cur_out = -1, side_out(bx[i], by[i], L, W)
        elif state[i] == CONTROL and player[i] >= 0:
            cur, cur_out = player[i], -1
        nxt[i], nxt_out[i] = cur, cur_out
    kind = np.zeros(n, np.int8); a = np.full(n, -1); b = np.full(n, -1)
    ctrl = scored & (state == CONTROL) & (player >= 0)
    kind[ctrl], a[ctrl], b[ctrl] = K_SELF, player[ctrl], player[ctrl]
    mid = scored & np.isin(state, (TRAV, LOOSE)) & (last >= 0)
    to_p, to_o = mid & (nxt >= 0), mid & (nxt < 0) & (nxt_out >= 0)
    kind[to_p], a[to_p], b[to_p] = K_LINK, last[to_p], nxt[to_p]
    kind[to_o], a[to_o], b[to_o] = K_OUT, last[to_o], nxt_out[to_o]
    tl = np.where(scored & (state == TRAV), 0, np.where(scored & (state == LOOSE), 1, -1)).astype(np.int8)
    return kind, a, b, tl


def lane_of(lane_owner, who):
    """Lane holding true player who[f] at frame f (-1 if none / who < 0)."""
    eq = (lane_owner == who[:, None]) & (who[:, None] >= 0)
    return np.where(eq.any(1), eq.argmax(1), -1)


def lanes_to_targets(kind, a, b, lane_owner):
    """Player-space targets -> lane / node space (outside nodes are 22 + side). Unmappable frames are masked."""
    la = lane_of(lane_owner, a)
    lb = np.where(kind == K_OUT, 2 * TEAM + b, lane_of(lane_owner, np.where(kind == K_OUT, -1, b)))
    ok = (kind != K_MASK) & (la >= 0) & (lb >= 0)
    return np.where(ok, kind, K_MASK).astype(np.int8), np.where(ok, la, -1).astype(np.int16), np.where(ok, lb, -1).astype(np.int16)


def pathcrf_arrays(m: dict, ball: dict, owner=None) -> tuple[dict, dict]:
    """The lane input for one match in our Metrica form (also the hackathon match: owner=None, no labels)."""
    r = relink(m["px"], m["py"], m["team_sign"], m["period"], FPS)
    seen = np.isin(ball["source"], (BF.SRC_SEEN, BF.SRC_SMOOTHED))
    out = dict(lane_x=r["lane_x"], lane_y=r["lane_y"], period=m["period"].astype(np.int8), t=m["t"].astype(np.float64),
               ball_x=ball["x"].astype(np.float32), ball_y=ball["y"].astype(np.float32), ball_seen=seen.astype(np.int8),
               pitch=np.asarray(m["pitch"], np.float32))
    st = dict(window=window_stats(r["lane_slot"], m["period"], r["links"], FPS), links=link_stats(r["links"]),
              trimmed_share=r["trimmed_share"])
    if owner is not None:
        rows = np.arange(len(m["period"]))[:, None]
        lane_owner = np.where(r["lane_slot"] >= 0, owner[rows, np.maximum(r["lane_slot"], 0)], -1)
        L, W = (float(x) for x in m["pitch"])
        scored = m["label_mask"].astype(bool)
        kind, a, b, tl = player_targets(m["label_state"].astype(int), m["label_player"].astype(int), scored,
                                        m["period"].astype(int), m["ball_true_x"], m["ball_true_y"], L, W)
        k2, la, lb = lanes_to_targets(kind, a, b, lane_owner)
        out.update(kind=k2, a=la, b=lb, tl=tl, label_had=m["label_had"].astype(np.int8), label_mask=scored,
                   live_scored=(scored & (m["label_had"] != 2)), match_id=np.int64(m["match_id"]), fold=np.int64(m["fold"]))
        need = kind != K_MASK
        st["targets"] = dict(frames_with_target=int(need.sum()), mapped_share=float((k2[need] != K_MASK).mean()) if need.any() else None)
        st["purity"] = _purity(lane_owner, m["period"])
    return out, st


def _purity(lane_owner, period, win: int = 250):
    """Share of lane-frames whose true player is the lane's most frequent one inside each 10 s window."""
    tot = same = 0
    for s in range(0, len(period) - win + 1, win):
        if period[s] <= 0:
            continue
        lo = lane_owner[s:s + win]
        for k in range(lo.shape[1]):
            v = lo[:, k][lo[:, k] >= 0]
            if len(v):
                tot += len(v); same += int(np.bincount(v).max())
    return same / max(tot, 1)


def build_one(args):
    path, out, cfg = args
    t0 = time.time()
    import lightgbm as lgb
    bfp = [lgb.Booster(model_file=str(MODELS / "bfp" / nm)) for nm in ("bfp_dx.txt", "bfp_dy.txt")]
    d = DP.load_package_match(path)
    m = DP.degrade_match(d, cfg["stats"], cfg["seed"], dead_weight=cfg["dead_weight"], extra=cfg["extra"], owner=True)
    m = T.prepare_players(m, "current", dedup=True)
    owner = np.where(np.isfinite(m["px"]), m["owner"], -1).astype(np.int16)
    ball = BF.fill_ball(m, bfp)
    rows, X, names = T.features(m, ball, mode="current")
    gt, gmask = m["label_had"], m["label_mask"]
    np.savez_compressed(out / "c3t" / f"{m['match_id']}.npz", rows=rows, X=X.astype(np.float32), lab=gt[rows], mask=gmask[rows],
                        prow=m["period"][rows], gt=gt, gmask=gmask, period=m["period"])
    arr, st = pathcrf_arrays(m, ball, owner)
    np.savez_compressed(out / "pathcrf" / f"{m['match_id']}.npz", **arr)
    st.update(match_id=int(m["match_id"]), frames=int(len(m["t"])), seconds=time.time() - t0)
    return int(m["match_id"]), st, names


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--package", required=True, help="the M1 PFF package (matches/*.npz + folds.json)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--matches", type=int, nargs="*")
    a = ap.parse_args(argv)
    t0 = time.time()
    pkg, out = Path(a.package), Path(a.out)
    for sub in ("c3t", "pathcrf"):
        (out / sub).mkdir(parents=True, exist_ok=True)
    shutil.copy(pkg / "folds.json", out / "folds.json")
    folds = json.loads((pkg / "folds.json").read_text())["folds"]
    learn = {m for f in folds for m in f}
    files = sorted(p for p in (pkg / "matches").glob("*.npz") if int(p.stem) in learn)
    if a.matches:
        files = [p for p in files if int(p.stem) in set(a.matches)]
    cfg = config()
    print(f"{len(files)} matches; dead_weight {cfg['dead_weight']:g}, extra slots target {100 * cfg['extra']['target_frac']:.1f} %, "
          f"seed {cfg['seed']}", flush=True)
    with multiprocessing.Pool(a.workers) as pool:
        res = []
        for mid, st, names in pool.imap_unordered(build_one, [(p, out, cfg) for p in files]):
            res.append(st)
            w = st["window"]
            print(f"  {mid}: windows both teams >= 10 lanes {100 * w['share_both_teams_ge10']:.1f} %, links/window "
                  f"{w['mean_links_per_window']:.2f}, targets mapped {100 * (st['targets']['mapped_share'] or 0):.1f} %, "
                  f"lane purity {100 * st['purity']:.1f} % ({st['seconds']:.0f} s)", flush=True)
    (out / "c3t" / "features.json").write_text(json.dumps(names))
    agg = dict(n_matches=len(res), config=dict(dead_weight=cfg["dead_weight"], seed=cfg["seed"], extra_target=cfg["extra"]["target_frac"]),
               share_both_teams_ge10=float(np.mean([r["window"]["share_both_teams_ge10"] for r in res])),
               mean_links_per_window=float(np.mean([r["window"]["mean_links_per_window"] for r in res])),
               forced_link_share=float(np.mean([r["links"].get("forced_share", 0) for r in res])),
               targets_mapped_share=float(np.mean([r["targets"]["mapped_share"] for r in res])),
               lane_purity=float(np.mean([r["purity"] for r in res])), per_match=sorted(res, key=lambda r: r["match_id"]),
               seconds=time.time() - t0)
    (out / "build_stats.json").write_text(json.dumps(agg, indent=1, default=float))
    print(f"built {len(res)} matches in {time.time() - t0:.0f} s: both teams >= 10 lanes {100 * agg['share_both_teams_ge10']:.1f} % of "
          f"windows, links/window {agg['mean_links_per_window']:.2f}, targets mapped {100 * agg['targets_mapped_share']:.1f} %, "
          f"lane purity {100 * agg['lane_purity']:.1f} %", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
