"""Step 2a: re-link the hackathon match's slots into 22 lanes (barca_possession.relink), after the duplicate rule (as
C3-T's input). Aggregates only: share of 10 s windows where both teams have >= 10 (and 11) lanes present on >= 80 % of
the frames, mean lanes per team, links per window, link gaps and distances. Writes reports/auto/relink_match.json.

    .venv/Scripts/python.exe scripts/relink_match.py
"""
from __future__ import annotations

import json

from barca_possession import c3t as T
from barca_possession.load import read_metrica
from barca_possession.paths import AUTO, FPS
from barca_possession.relink import link_stats, relink, window_stats


def main() -> int:
    m = T.prepare_players(read_metrica(), "current", dedup=True)
    r = relink(m["px"], m["py"], m["team_sign"], m["period"], FPS)
    res = dict(window=window_stats(r["lane_slot"], m["period"], r["links"], FPS),
               window_any=window_stats(r["lane_slot"], m["period"], r["links"], FPS, cover=1e-9),
               links=link_stats(r["links"]), trimmed_share=r["trimmed_share"], slots_in=int(m["px"].shape[1]))
    w, wa, lk = res["window"], res["window_any"], res["links"]
    print(f"slots in: {res['slots_in']}; trimmed (team > 11): {100 * r['trimmed_share']:.2f} % of slot-frames")
    print(f"10 s windows: {w['windows']}; both teams >= 10 lanes present on >= 80 % of frames: "
          f"{100 * w['share_both_teams_ge10']:.1f} % (all 11: {100 * w['share_both_teams_eq11']:.1f} %); "
          f"mean lanes per team {w['mean_lanes_per_team']:.2f}")
    print(f"... lanes present on any frame of the window: both teams >= 10 {100 * wa['share_both_teams_ge10']:.1f} %, "
          f"mean lanes per team {wa['mean_lanes_per_team']:.2f}")
    print(f"links: {lk['n_links']}, mean per 10 s window {w['mean_links_per_window']:.2f}; forced (outside the gate) "
          f"{100 * lk.get('forced_share', 0):.1f} %; gap median {lk.get('gap_s_p50', float('nan')):.2f} s; distance median "
          f"{lk.get('dist_m_p50', float('nan')):.2f} m, p90 {lk.get('dist_m_p90', float('nan')):.2f} m")
    AUTO.mkdir(parents=True, exist_ok=True)
    (AUTO / "relink_match.json").write_text(json.dumps(res, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
