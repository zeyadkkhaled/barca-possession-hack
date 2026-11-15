"""Build the M1 x PathCRF HPC bundle: D:/m1_tmp/hpc/m1_hpc_bundle/ and D:/m1_tmp/hpc/m1_hpc_bundle.zip.

Zip layout (unzip into ~/projects/m1-possession):
  code/      m1pc/ (adapter, model, train, predict, scorer, combine, cycle), pathcrf/ (vendored, MPL-2.0 licence, published model 140
             weights), env/setup_env.sh, jobs/*.sh, run_cycle.sh, README_HPC.md
  data/      matches/*.npz + folds.json + README.md + m1_hpc_loader.py (the m1_hpc_v1 package: 30 learning matches, no test / stress)
             c3_fold0/<match>.npz   the frozen C3's held-out predictions for fold 0 (state, player, team per frame; fold-0 model
                                    trained on folds 1-4 only), so the HPC compares PathCRF with C3 by itself
  README_HPC.md

    python scripts/m1/hpc_pathcrf/build_bundle.py [--pathcrf D:/m1_tmp/ext/pathcrf] [--package D:/m1_tmp/hpc/m1_hpc_v1]
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
OUT = Path("D:/m1_tmp/hpc/m1_hpc_bundle")
ZIP = Path("D:/m1_tmp/hpc/m1_hpc_bundle.zip")
PATHCRF_KEEP = ["LICENSE", "README.md", "requirements.txt", "setup.cfg", "dataset.py", "train.py", "inference.py", "evaluate.py",
                "models", "set_transformer", "datatools", "saved/140"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pathcrf", default="D:/m1_tmp/ext/pathcrf")
    ap.add_argument("--package", default="D:/m1_tmp/hpc/m1_hpc_v1")
    a = ap.parse_args()
    src_pc, pkg = Path(a.pathcrf), Path(a.package)
    if subprocess.run([sys.executable, str(HERE / "make_m1score.py"), "--check"]).returncode:
        raise SystemExit("m1score.py is out of date: run scripts/m1/hpc_pathcrf/make_m1score.py")
    shutil.rmtree(OUT, ignore_errors=True)
    code, data = OUT / "code", OUT / "data"
    shutil.copytree(HERE / "code", code, ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "pathcrf"))
    # vendored PathCRF
    commit = subprocess.run(["git", "-C", str(src_pc), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    for item in PATHCRF_KEEP:
        s, d = src_pc / item, code / "pathcrf" / item
        if s.is_dir():
            shutil.copytree(s, d, ignore=shutil.ignore_patterns("__pycache__", "*.pyc", ".ipynb_checkpoints"))
        else:
            d.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(s, d)
    (code / "pathcrf" / "VENDORED.txt").write_text(
        f"PathCRF (Kim et al., KDD 2026), https://github.com/hyunsungkim-ds/pathcrf, commit {commit}.\n"
        "Licence: Mozilla Public License 2.0 (LICENSE). Vendored unchanged; only the files listed in build_bundle.py are included\n"
        "(code, requirements, licence, published model 140 = saved/140). Our changes live in ../m1pc and do not edit these files.\n")
    # data package + C3 fold-0 predictions
    shutil.copytree(pkg / "matches", data / "matches")
    for f in ("folds.json", "README.md", "m1_hpc_loader.py", "sizes.json"):
        if (pkg / f).exists():
            shutil.copy2(pkg / f, data / f)
    folds = json.loads((pkg / "folds.json").read_text())["folds"]
    (data / "c3_fold0").mkdir()
    for m in folds[0]:
        shutil.copy2(REPO / "data/processed/m1_possession/c3v4/pred/base_pt" / f"k0_{m}.npz", data / "c3_fold0" / f"{m}.npz")
    (data / "c3_fold0" / "README.txt").write_text(
        "Frozen C3 (C3 + per-transition decoder, round 3 'alt' = round 4 'base_pt') held-out predictions for the 6 fold-0 matches:\n"
        "state (0 DEAD, 1 CONTROL, 2 TRAVELLING, 3 LOOSE), player (PFF player id, -1 none), team (PFF team id, -1 none) per frame\n"
        "(29.97 fps, same frames as matches/<match>.npz). Made by the fold-0 model, trained on folds 1-4 only.\n")
    shutil.copy2(code / "README_HPC.md", OUT / "README_HPC.md")
    for p in OUT.rglob("*.sh"):                                          # LF line endings for bash
        p.write_bytes(p.read_bytes().replace(b"\r\n", b"\n"))
    ZIP.unlink(missing_ok=True)
    with zipfile.ZipFile(ZIP, "w", zipfile.ZIP_DEFLATED) as z:
        for p in sorted(OUT.rglob("*")):
            if p.is_file():
                info = zipfile.ZipInfo.from_file(p, p.relative_to(OUT).as_posix())
                if p.suffix == ".sh":
                    info.external_attr = 0o100755 << 16
                info.compress_type = zipfile.ZIP_STORED if p.suffix in (".npz", ".pt") else zipfile.ZIP_DEFLATED
                with open(p, "rb") as fh:
                    z.writestr(info, fh.read())
    size = ZIP.stat().st_size
    n = sum(1 for p in OUT.rglob("*") if p.is_file())
    print(f"bundle: {ZIP} ({size / 2**20:.1f} MB, {n} files; staging folder {OUT}); PathCRF commit {commit[:7]}")
    if size >= 2 * 2**30:
        raise SystemExit("bundle >= 2 GB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
