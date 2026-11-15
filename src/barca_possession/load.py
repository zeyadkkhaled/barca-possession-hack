"""Load the hackathon tracking (Metrica FIFA/EPTS ATD) into per-frame arrays, and measure the noise statistics that
degrade_pff.py copies onto PFF.

Written from DATA.md, games.py and starter/baseline.py only (the data itself was never opened while writing it).
- one row per video frame: frame = frame_id - 1 (DATA.md, "Frame alignment");
- track slots and their team from the metadata XML via the official starter (baseline.team_of_each_track: the first team
  in the file is home);
- coordinates: normalised 0-1 -> metres with the pitch centre at (0, 0): x_m = (x - 0.5) * 105.3, y_m = (y - 0.5) * 68
  (the y direction is kept as delivered);
- period: 1 / 2 on the graded halves (games.half_video_blocks), 0 elsewhere (never scored);
- orientation: per half, the median x of each team over the whole half; if home is on the +x side, x and y are
  negated for that half, so home always attacks +x (a 180-degree turn, which keeps left / right).

    python -m barca_possession.load            # prints aggregate statistics only; writes outputs/load_stats.json

Only aggregate numbers are printed (percentages, counts, means). Nothing per frame.
"""
from __future__ import annotations

import json
import time

import numpy as np

from .paths import AUTO, FPS, OUT, PITCH, official

GAP_EDGES_S = (0.0, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, np.inf)
GAP_BIN_NAMES = ("<0.2", "0.2-0.5", "0.5-1", "1-2", "2-5", "5-10", "10-30", "30+")
SLOW_MPS = 5.0
SIDE_EDGES_M = (0.0, 3.0, 10.0, np.inf)
SIDE_BIN_NAMES = ("<3 m", "3-10 m", ">=10 m")
ORIENT_S = 10.0
CACHE = OUT / "cache" / "metrica_tactical.npz"


def _parse_raw(raw_path, n_slots: int, n_frames: int):
    """Raw ATD -> arrays. Line: '<frame_id>:<slot>;<slot>;...:<ball>', each group 'x,y,s' (NaN = missing)."""
    X = np.full((n_frames, n_slots), np.nan, np.float32)
    Y = np.full_like(X, np.nan)
    S = np.full_like(X, np.nan)
    B = np.full((n_frames, 3), np.nan, np.float32)
    bad = 0
    with open(raw_path) as fh:
        for line in fh:
            parts = line.strip().split(":")
            if len(parts) != 3:
                bad += 1
                continue
            row = int(parts[0]) - 1                       # frame_id counts from 1, video frames from 0
            if not 0 <= row < n_frames:
                bad += 1
                continue
            groups = [g for g in parts[1].split(";") if g]
            if len(groups) == n_slots:
                v = np.array(",".join(groups).split(","), dtype=np.float64)
                if len(v) == 3 * n_slots:
                    v = v.reshape(n_slots, 3)
                    X[row], Y[row], S[row] = v[:, 0], v[:, 1], v[:, 2]
                else:
                    for k, g in enumerate(groups):
                        vals = np.array(g.split(","), dtype=np.float64)
                        X[row, k], Y[row, k] = vals[0], vals[1]
                        if len(vals) > 2:
                            S[row, k] = vals[2]
            else:
                bad += 1
            bv = np.array(parts[2].split(","), dtype=np.float64)
            B[row, :min(3, len(bv))] = bv[:3]
    return X, Y, S, B, bad


def read_metrica(data_dir=None, use_cache: bool = True) -> dict:
    """The tactical ATD as per-frame arrays in metres, centre origin, home attacking +x in both halves."""
    games = official()
    from baseline import team_names, team_of_each_track          # official starter (numpy only)

    meta, raw = games.require(["atd_meta", "atd_raw"], data_dir=data_dir)
    n = games.source_frames("tactical")
    team_of_track = team_of_each_track(meta)                       # 0 home, 1 away per slot
    if use_cache and CACHE.exists() and CACHE.stat().st_mtime > max(_mtime(meta), _mtime(raw)):
        with np.load(CACHE) as z:
            X, Y, S, B, bad = z["X"], z["Y"], z["S"], z["B"], int(z["bad"])
    else:
        X, Y, S, B, bad = _parse_raw(raw, len(team_of_track), n)
        CACHE.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(CACHE, X=X, Y=Y, S=S, B=B, bad=bad)
    L, W = PITCH
    px, py = (X - 0.5) * L, (Y - 0.5) * W
    bx, by = (B[:, 0] - 0.5) * L, (B[:, 1] - 0.5) * W
    period = np.zeros(n, np.int8)
    for half, a, b in games.half_video_blocks(games.GAME):
        period[a:b + 1] = half
    team_sign = np.where(team_of_track == 0, 1.0, -1.0).astype(np.float32)
    flip = {}
    for half in (1, 2):
        sel = np.flatnonzero(period == half)
        # Whole-half median x per team (as M1's orient). The plan's "first 10 s" fails on the 2nd half: it is graded
        # ~2 min before its kick-off, while the players walk out (checked 2026-11-15 against the Kick Off tags).
        with np.errstate(all="ignore"):
            hx = np.nanmedian(px[sel][:, team_sign > 0])
            ax = np.nanmedian(px[sel][:, team_sign < 0])
        f = bool(np.isfinite(hx) and np.isfinite(ax) and hx > ax)
        flip[half] = dict(flipped=f, home_mean_x=float(hx), away_mean_x=float(ax))
        if f:
            px[sel], py[sel], bx[sel], by[sel] = -px[sel], -py[sel], -bx[sel], -by[sel]
    return dict(frame=np.arange(n), t=np.arange(n) / FPS, period=period, px=px.astype(np.float32), py=py.astype(np.float32),
                ps=S, team_sign=team_sign, bx=bx.astype(np.float32), by=by.astype(np.float32), b3=B[:, 2], pitch=PITCH,
                flip=flip, teams=team_names(meta), bad_lines=bad)


def _mtime(p):
    import os
    return os.path.getmtime(p)


# ---------------------------------------------------------------- statistics (shared with degraded PFF)

def runs(mask: np.ndarray):
    """[(start, stop)] half-open runs where mask is True."""
    e = np.diff(np.r_[0, mask.astype(np.int8), 0])
    return list(zip(np.flatnonzero(e == 1), np.flatnonzero(e == -1)))


def ball_speed_before(t, bx, by, a, k: int = 3):
    """Ball speed (m/s) from the last k+1 seen frames ending at a-1 (all of them seen), else NaN."""
    i0 = a - 1 - k
    if i0 < 0 or not np.all(np.isfinite(bx[i0:a])):
        return np.nan
    return float(np.hypot(bx[a - 1] - bx[i0], by[a - 1] - by[i0]) / (t[a - 1] - t[i0]))


def gap_stats(t, period, bx, by, half_width: float) -> dict:
    """Ball-missing runs inside graded frames: length histogram, split by ball speed before the gap and by the distance to
    the sideline where it starts. Counts and frame shares only."""
    miss = ~np.isfinite(bx)
    graded = period > 0
    gaps = []
    for p in np.unique(period[graded]):
        idx = np.flatnonzero(period == p)
        a0 = idx[0]
        for a, b in runs(miss[idx]):
            a, b = a + a0, b + a0
            sp = ball_speed_before(t, bx, by, a) if a > a0 else np.nan
            side = half_width - abs(float(by[a - 1])) if a > a0 and np.isfinite(by[a - 1]) else np.nan
            gaps.append(((b - a) / FPS, sp, side))
    g = np.array(gaps, float).reshape(-1, 3)
    L, SP, SD = g[:, 0], g[:, 1], g[:, 2]

    def hist(sel):
        c, _ = np.histogram(L[sel], bins=GAP_EDGES_S)
        fr = np.array([L[sel & (L >= lo) & (L < hi)].sum() for lo, hi in zip(GAP_EDGES_S[:-1], GAP_EDGES_S[1:])])
        return dict(counts=c.tolist(), seconds=fr.round(2).tolist())

    out = dict(bins=list(GAP_BIN_NAMES), edges_s=[float(x) for x in GAP_EDGES_S[:-1]] + ["inf"], n_gaps=int(len(L)),
               all=hist(np.ones(len(L), bool)),
               slow=hist(SP < SLOW_MPS), fast=hist(SP >= SLOW_MPS), speed_unknown=hist(~np.isfinite(SP)))
    for (lo, hi), nm in zip(zip(SIDE_EDGES_M[:-1], SIDE_EDGES_M[1:]), SIDE_BIN_NAMES):
        out[f"sideline {nm}"] = hist((SD >= lo) & (SD < hi))
    out["sideline unknown"] = hist(~np.isfinite(SD))
    out["missing_frac_graded"] = float(miss[graded].mean())
    out["missing_frac_all"] = float(miss.mean())
    return out


def slot_stats(period, px, team_sign) -> dict:
    graded = period > 0
    pres = np.isfinite(px)
    seg_len, starts = [], 0
    minutes = graded.sum() / FPS / 60
    for p in np.unique(period[graded]):
        idx = np.flatnonzero(period == p)
        for s in range(px.shape[1]):
            for a, b in runs(pres[idx, s]):
                seg_len.append((b - a) / FPS)
                starts += a > 0
    ph = pres[graded][:, team_sign > 0].sum(1)
    pa = pres[graded][:, team_sign < 0].sum(1)
    return dict(n_slots=int(px.shape[1]), mean_slots_home=float(ph.mean()), mean_slots_away=float(pa.mean()),
                mean_slots_total=float((ph + pa).mean()), frac_frames_team_over_11=float(((ph > 11) | (pa > 11)).mean()),
                mean_segment_s=float(np.mean(seg_len)) if seg_len else 0.0,
                median_segment_s=float(np.median(seg_len)) if seg_len else 0.0, n_segments=len(seg_len),
                slot_starts_per_min=float(starts / max(minutes, 1e-9)), graded_minutes=float(minutes))


def ball_misc(period, bx, by, b3) -> dict:
    g = period > 0
    seen = np.isfinite(bx) & g
    both = seen[1:] & seen[:-1]
    rep = (bx[1:] == bx[:-1]) & (by[1:] == by[:-1]) & both
    v = b3[seen & np.isfinite(b3)]
    return dict(repeat_frac=float(rep.sum() / max(both.sum(), 1)),
                third_field=dict(n=int(len(v)), p5=float(np.percentile(v, 5)) if len(v) else None,
                                 median=float(np.median(v)) if len(v) else None,
                                 p95=float(np.percentile(v, 95)) if len(v) else None))


def all_stats(m: dict) -> dict:
    hw = m["pitch"][1] / 2
    return dict(source="metrica tactical", gap=gap_stats(m["t"], m["period"], m["bx"], m["by"], hw),
                slots=slot_stats(m["period"], m["px"], m["team_sign"]), ball=ball_misc(m["period"], m["bx"], m["by"], m["b3"]))


def print_stats(st: dict, extra: dict | None = None) -> None:
    g, s = st["gap"], st["slots"]
    print(f"ball missing: {100 * g['missing_frac_graded']:.1f} % of graded frames ({100 * g['missing_frac_all']:.1f} % of all)")
    print(f"slots: {s['n_slots']}; mean present per graded frame: home {s['mean_slots_home']:.2f}, away {s['mean_slots_away']:.2f}, "
          f"total {s['mean_slots_total']:.2f}; frames with a team > 11: {100 * s['frac_frames_team_over_11']:.1f} %")
    print(f"slot segments: {s['n_segments']}, mean {s['mean_segment_s']:.1f} s, median {s['median_segment_s']:.1f} s; "
          f"slot starts per minute: {s['slot_starts_per_min']:.2f}")
    for k, v in (extra or {}).items():
        print(f"{k}: {v}")
    print(f"ball gaps: {g['n_gaps']}")
    head = "split".ljust(22) + "".join(b.rjust(9) for b in g["bins"])
    print(head)
    for key in [k for k in g if isinstance(g[k], dict) and "counts" in g[k]]:
        print(f"{key:22s}" + "".join(f"{c:9d}" for c in g[key]["counts"]) + "   gaps")
        print(f"{'':22s}" + "".join(f"{x:9.0f}" for x in g[key]["seconds"]) + "   s missing")
    if "ball" in st:
        b = st["ball"]
        print(f"ball: identical consecutive positions {100 * b['repeat_frac']:.1f} %; third ball field p5/median/p95 = "
              f"{b['third_field']['p5']}/{b['third_field']['median']}/{b['third_field']['p95']}")


def smart_tags(home_name: str, fps: float = FPS):
    """SET PIECES tags as {start_frame, end_frame, team (0 H / 1 A / None for N/A), type} and the Kick Off (Start)
    intervals per half, from the official starter reader (Smart Tagging is an input, not ground truth)."""
    official()
    import load_smart_tagging as LST
    home_key = (home_name or "").strip().lower()
    tags, ko = [], {}
    for e in LST.load():
        if (e["code"] or "").strip().lower() != "set pieces":
            continue
        team_txt = (e["tags"].get("Team") or "").strip().lower()
        if team_txt in ("", "n/a", "na"):
            team = None
        elif home_key and (home_key in team_txt or team_txt in home_key):
            team = 0
        else:
            team = 1
        typ = (e["tags"].get("Type") or "").strip().lower()
        tags.append(dict(start_frame=e["start_frame"], end_frame=e["end_frame"], team=team, type=typ))
        if typ == "kick off (start)":
            half = (e["tags"].get("Half") or "").strip().lower()
            p = 1 if half.startswith("1") else 2 if half.startswith("2") else None
            if p:
                ko[p] = (e["start_frame"], e["end_frame"])
    return tags, ko


# window before a restart where the ball-missing share is compared between the match (relative to the SET PIECES tag
# start) and degraded PFF (relative to the answer key's restart, ~2.5 s later on average: the tag interval contains it)
BEFORE_TAG_S = (-10.0, -2.0)
BEFORE_RESTART_S = (-12.5, -4.5)


def missing_before(miss, period, anchors, window_s, fps: float = FPS) -> float:
    v = []
    for a in anchors:
        lo, hi = int(a + window_s[0] * fps), int(a + window_s[1] * fps)
        if lo >= 0 and hi <= len(miss) and period[lo] > 0 and period[lo] == period[hi - 1]:
            v.append(miss[lo:hi].mean())
    return float(np.mean(v)) if v else float("nan")


def main() -> int:
    t0 = time.time()
    m = read_metrica()
    st = all_stats(m)
    tags, ko = smart_tags(m["teams"][0])
    st["missing_before_set_piece"] = missing_before(~np.isfinite(m["bx"]), m["period"], [t["start_frame"] for t in tags],
                                                    BEFORE_TAG_S)
    st["n_set_piece_tags"] = len(tags)
    print(f"ball missing in the {BEFORE_TAG_S} s window before SET PIECES tags: "
          f"{100 * st['missing_before_set_piece']:.1f} % ({len(tags)} tags)")
    st["orientation"] = {str(k): v for k, v in m["flip"].items()}
    st["bad_lines"] = m["bad_lines"]
    side = {f"half {k}": ("home was on +x -> flipped" if v["flipped"] else "home on -x (attacks +x) -> kept")
            for k, v in m["flip"].items()}
    print(f"teams: H = {m['teams'][0]}, A = {m['teams'][1]}; unparsed lines: {m['bad_lines']}")
    print_stats(st, extra={"attacking side": side})
    for d in (OUT, AUTO):
        d.mkdir(parents=True, exist_ok=True)
        (d / "load_stats.json").write_text(json.dumps(st, indent=1))
    print(f"wrote outputs/load_stats.json and reports/auto/load_stats.json ({time.time() - t0:.0f} s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
