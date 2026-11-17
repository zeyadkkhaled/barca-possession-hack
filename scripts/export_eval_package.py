"""Step 2b: the eval package m1_hpc_v2_eval: the 6 stress-eval + 16 locked PFF matches in the m1_hpc_v1 format, so the
HPC can degrade + relink them and the final PathCRF can PREDICT them (the chain never scores them). PFF data only.

Built like M1's inputs (scripts/m1/build_inputs.py) and package (export_hpc.py), from M1's processed files:
positions cache (raw players + raw ball), ball visibility (frames/), M1's ball smoother, PFF metadata (pitch, fps, team
ids), labels v2 (the package format carries them; the degradation reads them only for its dead-ball gap weight).
Difference to m1_hpc_v1: player_vis is 1 where a player has x/y (the raw visibility pass is not redone); our pipeline
never reads player_vis.

Run with M1's venv from the M1 root (it needs football_fatigue + m1_possession):
    cd "Z:/GP/Task 2 - match sim"
    $env:PYTHONPATH="src"; .venv/Scripts/python.exe D:/barca_hack/scripts/export_eval_package.py [--out D:/m1_tmp/hpc_step2/m1_hpc_v2_eval]
Prints per match only frame counts and sizes; zips to <out>.zip (layout eval_data/matches/*.npz, eval_data/README.md).
"""
from __future__ import annotations

import argparse
import json
import zipfile
from pathlib import Path

import numpy as np

from football_fatigue.ingestion import Dataset
from m1_possession import SmootherConfig, smooth_ball

STRESS_EVAL = (3816, 3829, 3847, 3852, 10512, 10514)
LOCKED_TEST = (3822, 3825, 3827, 3828, 3834, 3836, 3842, 3844, 3848, 10503, 10505, 10508, 10509, 10510, 10511, 10513)
RAW = "data/raw/FIFA World Cup 2022"
POS = Path("data/processed/event_detection/cache")
BASE = Path("data/processed/m1_possession")
VIS_NAMES = np.array([None, "VISIBLE", "ESTIMATED", None], dtype=object)


def export_one(mid: int, out: Path, ds: Dataset) -> dict:
    meta = ds.metadata(str(mid))
    z = dict(np.load(sorted(POS.glob(f"{mid}*.positions.npz"))[0]))
    pm = json.loads(str(z["meta"]))
    fr = dict(np.load(BASE / "frames" / f"{mid}.npz"))
    lab = np.load(BASE / "labels" / "v2" / f"{mid}.npz")
    fnum = z["frame_num"]
    fi = {int(f): i for i, f in enumerate(fr["frame_num"])}
    vis = np.array([fr["ball_visibility"][fi[int(f)]] if int(f) in fi else 0 for f in fnum])
    per = z["period"].astype(np.int8)
    b = smooth_ball(z["t"], z["ball"][:, 0], z["ball"][:, 1], z["ball"][:, 2], visibility=VIS_NAMES[vis], period=per,
                    cfg=SmootherConfig())
    li = {int(f): i for i, f in enumerate(lab["frame_num"])}
    il = np.array([li[int(f)] for f in fnum])
    px, py = z["px"], z["py"]
    pid = z["player_id"]
    lpid = lab["player_id"][il]
    slot = np.full(len(fnum), -1, np.int8)
    for c, p in enumerate(pid):
        if p >= 0:
            slot[lpid == p] = c
    side = z["player_side"].astype(np.int8)
    team = np.where(side == 0, meta.home_team_id, meta.away_team_id).astype(np.int32)
    jersey = np.array([int(k.split(":")[1]) for k in pm["player_key"]], np.int16)
    f16 = lambda a: np.asarray(a, np.float16)
    dst = out / "matches" / f"{mid}.npz"
    np.savez_compressed(
        dst, match_id=np.int32(mid), fold=np.int8(-1), fps=np.float32(meta.fps),
        pitch=np.array([meta.pitch_length_m, meta.pitch_width_m], np.float32), home_team_id=np.int32(meta.home_team_id),
        home_attacks_right=np.array([bool(pm["home_attacks_right"].get(str(p), False)) for p in (1, 2, 3, 4)]),
        frame_num=fnum.astype(np.int32), t=z["t"].astype(np.float32), period=per,
        ball_x=f16(b.x), ball_y=f16(b.y), ball_z=f16(b.z), ball_vx=f16(b.vx), ball_vy=f16(b.vy), ball_flag=b.flag.astype(np.int8),
        player_x=f16(px), player_y=f16(py), player_vis=np.isfinite(px).astype(np.int8), player_side=side, player_team_id=team,
        player_jersey=jersey, player_id=pid.astype(np.int32),
        label_state=lab["state"][il].astype(np.int8), label_player_id=lpid.astype(np.int32), label_player_slot=slot,
        label_team_id=lab["team_id"][il].astype(np.int32), label_score_mask=lab["score_mask"][il],
        label_mask_rule=lab["mask_rule"][il].astype(np.int8))
    return dict(match=mid, split="stress_eval" if mid in STRESS_EVAL else "locked_test", frames=int(len(fnum)),
                mb=dst.stat().st_size / 2**20)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="D:/m1_tmp/hpc_step2/m1_hpc_v2_eval")
    a = ap.parse_args()
    out = Path(a.out)
    (out / "matches").mkdir(parents=True, exist_ok=True)
    ds = Dataset(RAW)
    rows = []
    for mid in STRESS_EVAL + LOCKED_TEST:
        r = export_one(mid, out, ds)
        rows.append(r)
        print(f"  {mid} ({r['split']}): {r['frames']} frames, {r['mb']:.1f} MB", flush=True)
    (out / "matches.json").write_text(json.dumps(dict(stress_eval=list(STRESS_EVAL), locked_test=list(LOCKED_TEST), rows=rows), indent=1))
    (out / "README.md").write_text(
        "# m1_hpc_v2_eval\n\nThe 6 M1 stress-eval and 16 locked-test PFF World Cup 2022 matches in the m1_hpc_v1 format "
        "(see scripts/export_eval_package.py). For PathCRF PREDICTIONS ONLY in step 2b: the chain degrades + relinks them "
        "and predicts; it never scores them. player_vis = 1 where a player has x/y. PFF data: never commit, never share.\n",
        encoding="utf-8")
    zp = out.with_suffix(".zip")
    with zipfile.ZipFile(zp, "w", zipfile.ZIP_STORED) as zf:                # npz files are already compressed
        for f in sorted((out / "matches").glob("*.npz")):
            zf.write(f, f"eval_data/matches/{f.name}")
        zf.write(out / "README.md", "eval_data/README.md")
        zf.write(out / "matches.json", "eval_data/matches.json")
    print(f"{len(rows)} matches -> {out}; zip {zp}: {zp.stat().st_size / 2**20:.1f} MB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
