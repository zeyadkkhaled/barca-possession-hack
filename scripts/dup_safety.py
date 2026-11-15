"""Step 1c, task 2: on clean PFF (true player identities, so every slot is a real player) how often would the duplicate
rule merge two real teammates? Aggregates only.

    .venv/Scripts/python.exe scripts/dup_safety.py [--deg outputs/pff_clean]

(a) the rule as used (only frames where a team has more than 11 slots); (b) the same pair test on every frame, i.e. how
often two real teammates are closer than 0.5 m with velocities within 1 m/s (the risk if the >11 gate is wrong).
Writes reports/auto/dup_safety.json.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from barca_possession.load import DUP_MAX_DIST_M, DUP_MAX_DV_MPS, remove_duplicates, slot_velocity, team_over
from barca_possession.paths import AUTO, OUT


def close_pair_frames(px, py, ts, per) -> np.ndarray:
    """Per frame: some same-team pair passes the distance and velocity test (no >11 gate)."""
    vx, vy = slot_velocity(px, per), slot_velocity(py, per)
    hit = np.zeros(len(px), bool)
    for side in (1.0, -1.0):
        c = np.flatnonzero(ts == side)
        i, j = np.triu_indices(len(c), 1)
        a, b = c[i], c[j]
        for k0 in range(0, len(a), 64):                      # chunks of pairs keep memory small
            sa, sb = a[k0:k0 + 64], b[k0:k0 + 64]
            d = np.hypot(px[:, sa] - px[:, sb], py[:, sa] - py[:, sb])
            dv = np.hypot(vx[:, sa] - vx[:, sb], vy[:, sa] - vy[:, sb])
            hit |= ((d < DUP_MAX_DIST_M) & (dv < DUP_MAX_DV_MPS)).any(1)
    return hit


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--deg", default=str(OUT / "pff_clean"))
    a = ap.parse_args()
    files = sorted(Path(a.deg).glob("*.npz"))
    n_in = merged = close = 0
    over = []
    for f in files:
        with np.load(f) as z:
            px, py, ts, per = z["px"], z["py"], z["team_sign"], z["period"]
        g = per > 0
        _, _, dropped = remove_duplicates(px, py, ts, per)
        n_in += int(g.sum())
        merged += int(dropped[g].any(1).sum())
        close += int(close_pair_frames(px, py, ts, per)[g].sum())
        over.append(team_over(px, ts, per))
    res = dict(n_matches=len(files), frames=n_in, rule_merges_real_teammates_frac=merged / max(n_in, 1),
               close_pair_any_frame_frac=close / max(n_in, 1), team_over_11_frac=float(np.mean(over)))
    print(f"clean PFF, {len(files)} matches, {n_in} in-period frames:")
    print(f"  rule as used (team > 11 gate): merges two real teammates on {100 * res['rule_merges_real_teammates_frac']:.3f} % of frames")
    print(f"  same pair test without the gate: {100 * res['close_pair_any_frame_frac']:.2f} % of frames")
    print(f"  frames with a team > 11 on clean PFF: {100 * res['team_over_11_frac']:.3f} %")
    AUTO.mkdir(parents=True, exist_ok=True)
    (AUTO / "dup_safety.json").write_text(json.dumps(res, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
