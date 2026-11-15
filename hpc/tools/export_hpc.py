"""M1: data package for GPU training of sequence models on the HPC (30 learning matches only; never test / stress).

Per match one compressed .npz (format in scripts/m1/hpc/README.md): time, period, our smoothed raw ball, every raw player
column (x, y float16; team, jersey, player id per column; raw visibility per frame) and labels v2 (state, player, team,
score_mask, mask_rule). Plus folds.json, the loader and the README.

Stage 1 (vis): one streaming pass over each raw tracking file reads ONLY the raw homePlayers / awayPlayers lists (never the
*Smoothed lists or the event overlays) and keeps (jersey, visibility) per player and frame -> D:\\m1_tmp\\hpc_vis\\<match>.npz.
Stage 2 (export): writes <out>/matches/<match>.npz, folds.json, README.md, m1_hpc_loader.py, sizes.json.

    python scripts/m1/export_hpc.py vis --workers 4
    python scripts/m1/export_hpc.py export [--out D:/m1_tmp/hpc/m1_hpc_v1]
"""
from __future__ import annotations

import argparse
import bz2
import json
import multiprocessing
import re
import shutil
from pathlib import Path

import numpy as np

from barca_possession.config import use_scratch
from barca_possession.splits import FOLDS, LEARNING_MATCHES

RAW = Path("data/raw/FIFA World Cup 2022/Tracking Data")
BASE = Path("data/processed/m1_possession")
POS = Path("data/processed/event_detection/cache")
HERE = Path(__file__).parent / "hpc"
VIS_DIR = Path("D:/m1_tmp/hpc_vis")
OUT = Path("D:/m1_tmp/hpc/m1_hpc_v1")
FN = re.compile(r'"frameNum":(\d+)')
LISTS = (("home", re.compile(r'"homePlayers":\[(.*?)\],"homePlayersSmoothed"')),
         ("away", re.compile(r'"awayPlayers":\[(.*?)\],"awayPlayersSmoothed"')))
OBJ = re.compile(r"\{[^{}]*\}")
JER = re.compile(r'"jerseyNum":"?(\d+)"?')
VIS = re.compile(r'"visibility":"(\w+)"')
VCODE = {"VISIBLE": 1, "ESTIMATED": 2}   # 0 = not in the raw list this frame, 3 = other / null


def vis_pass(mid: int) -> str:
    out = VIS_DIR / f"{mid}.npz"
    if out.exists():
        return f"{mid} exists"
    keys = json.loads(str(np.load(sorted(POS.glob(f"{mid}*.positions.npz"))[0])["meta"]))["player_key"]
    col = {k: i for i, k in enumerate(keys)}
    fn, rows, unk = [], [], 0
    with bz2.open(RAW / f"{mid}.jsonl.bz2", "rt", encoding="utf-8") as fh:
        for line in fh:
            m = FN.search(line)
            if not m:
                continue
            r = np.zeros(len(keys), np.int8)
            for side, rx in LISTS:
                g = rx.search(line)
                if not g:
                    continue
                for o in OBJ.findall(g.group(1)):
                    j, v = JER.search(o), VIS.search(o)
                    c = col.get(f"{side}:{int(j.group(1))}") if j else None
                    if c is None:
                        unk += 1
                        continue
                    r[c] = VCODE.get(v.group(1), 3) if v else 3
            fn.append(int(m.group(1))); rows.append(r)
    VIS_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, frame_num=np.array(fn, np.int64), vis=np.array(rows, np.int8))
    return f"{mid} ok ({len(fn)} frames, {unk} player entries without a column)"


def export_one(mid: int, out: Path, fold: int, home: int) -> dict:
    z = np.load(BASE / "inputs" / f"{mid}.npz")
    lab = np.load(BASE / "labels" / "v2" / f"{mid}.npz")
    pos = np.load(sorted(POS.glob(f"{mid}*.positions.npz"))[0])
    meta = json.loads(str(pos["meta"]))
    v = np.load(VIS_DIR / f"{mid}.npz")
    fnum = z["frame_num"]
    li = {int(f): i for i, f in enumerate(lab["frame_num"])}
    il = np.array([li[int(f)] for f in fnum])
    vi = {int(f): i for i, f in enumerate(v["frame_num"])}
    iv = np.array([vi.get(int(f), -1) for f in fnum])
    vis = np.where(iv[:, None] >= 0, v["vis"][np.maximum(iv, 0)], 0).astype(np.int8)
    px, py = z["px"], z["py"]
    present = np.isfinite(px)
    vis[~present] = 0                                   # a column with no raw x/y is absent, whatever the list said
    pid = z["player_id"]
    lpid = lab["player_id"][il]
    slot = np.full(len(fnum), -1, np.int8)
    for c, p in enumerate(pid):
        if p >= 0:
            slot[lpid == p] = c
    jersey = np.array([int(k.split(":")[1]) for k in meta["player_key"]], np.int16)
    side = np.array([0 if k.startswith("home") else 1 for k in meta["player_key"]], np.int8)
    f16 = lambda a: np.asarray(a, np.float16)
    dst = out / "matches" / f"{mid}.npz"
    np.savez_compressed(
        dst, match_id=np.int32(mid), fold=np.int8(fold), fps=np.float32(z["fps"]), pitch=z["pitch"].astype(np.float32),
        home_team_id=np.int32(home), home_attacks_right=np.array([bool(meta["home_attacks_right"].get(str(p), False)) for p in (1, 2, 3, 4)]),
        frame_num=fnum.astype(np.int32), t=z["t"].astype(np.float32), period=z["period"].astype(np.int8),
        ball_x=f16(z["bx"]), ball_y=f16(z["by"]), ball_z=f16(z["bz"]), ball_vx=f16(z["bvx"]), ball_vy=f16(z["bvy"]),
        ball_flag=z["bflag"].astype(np.int8),
        player_x=f16(px), player_y=f16(py), player_vis=vis, player_side=side, player_team_id=z["team_id"].astype(np.int32),
        player_jersey=jersey, player_id=pid.astype(np.int32),
        label_state=lab["state"][il].astype(np.int8), label_player_id=lpid.astype(np.int32), label_player_slot=slot,
        label_team_id=lab["team_id"][il].astype(np.int32), label_score_mask=lab["score_mask"][il],
        label_mask_rule=lab["mask_rule"][il].astype(np.int8))
    n_on = present.sum(1)
    return dict(match=mid, fold=fold, frames=int(len(fnum)), mb=dst.stat().st_size / 2**20,
                players_on_median=float(np.median(n_on)), frames_without_vis_row=int((iv < 0).sum()),
                present_but_not_listed=int((present & (vis == 0)).sum()),
                label_player_without_slot=int(((lab["state"][il] == 1) & (lpid >= 0) & (slot < 0)).sum()))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=("vis", "export"))
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args()
    use_scratch()
    if a.stage == "vis":
        with multiprocessing.Pool(a.workers, maxtasksperchild=1) as pool:
            for r in pool.imap_unordered(vis_pass, LEARNING_MATCHES):
                print(r, flush=True)
        return 0
    out = Path(a.out)
    (out / "matches").mkdir(parents=True, exist_ok=True)
    home = json.loads((BASE / "home_team.json").read_text())
    fold_of = {m: k for k, ms in enumerate(FOLDS) for m in ms}
    rows = []
    for m in LEARNING_MATCHES:
        r = export_one(m, out, fold_of[m], int(home[str(m)]))
        rows.append(r)
        print(json.dumps(r), flush=True)
    (out / "folds.json").write_text(json.dumps({"folds": [list(map(int, f)) for f in FOLDS],
                                                "note": "30 learning matches, 5 folds (src/m1_possession/splits.py). "
                                                        "Test and stress matches are not in this package."}, indent=1))
    shutil.copy(HERE / "README.md", out / "README.md")
    shutil.copy(HERE / "m1_hpc_loader.py", out / "m1_hpc_loader.py")
    tot = sum(r["mb"] for r in rows)
    (out / "sizes.json").write_text(json.dumps(dict(total_mb=tot, matches=rows), indent=1))
    print(f"total {tot:.0f} MB in {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
