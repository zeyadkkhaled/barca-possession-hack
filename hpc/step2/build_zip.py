"""Package step 2b for the HPC: D:/m1_tmp/hpc_step2/barca_step2.zip (unzip into ~/projects/barca-step2/).

Zip layout:
  code/src/{barca_possession,barca_train,barca_labels,pathcrf_adapter}/*.py    our code (Python files only)
  code/vendor/pathcrf/                                                       PathCRF, vendored with its MPL-2.0 LICENSE and the
                                                                             published model 140 (saved/140)
  code/models/bfp/bfp_d{x,y}.txt                                             ball-from-players model (trained on degraded PFF)
  code/reports/auto/{load_stats,degrade_stats_extra,extra_slot_diag}.json   aggregate rates that drive the degradation
  code/jobs/*.sh, code/run_step2.sh, code/status_step2.sh, code/HPC_STEP2.md
  HPC_STEP2.md
No data: no hackathon data (only the aggregate rate files above) and no PFF data (the HPC builds it from the existing
~/projects/m1-possession/data). Shell scripts are written with LF line endings.

    .venv/Scripts/python.exe hpc/step2/build_zip.py [--out D:/m1_tmp/hpc_step2/barca_step2.zip]
"""
from __future__ import annotations

import argparse
import hashlib
import zipfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
PACKAGES = ("barca_possession", "barca_train", "barca_labels", "pathcrf_adapter")
AUTO_FILES = ("load_stats.json", "degrade_stats_extra.json", "extra_slot_diag.json")


def entries():
    """(source path, name in the zip)."""
    out = []
    for pk in PACKAGES:
        for f in sorted((REPO / "src" / pk).glob("*.py")):
            out.append((f, f"code/src/{pk}/{f.name}"))
    for f in sorted((REPO / "vendor" / "pathcrf").rglob("*")):
        if f.is_file() and "__pycache__" not in f.parts and f.suffix != ".pyc":
            out.append((f, "code/vendor/pathcrf/" + f.relative_to(REPO / "vendor" / "pathcrf").as_posix()))
    for nm in ("bfp_dx.txt", "bfp_dy.txt"):
        out.append((REPO / "models" / "bfp" / nm, f"code/models/bfp/{nm}"))
    for nm in AUTO_FILES:
        out.append((REPO / "reports" / "auto" / nm, f"code/reports/auto/{nm}"))
    for f in sorted((HERE / "jobs").glob("*.sh")):
        out.append((f, f"code/jobs/{f.name}"))
    out.append((HERE / "run_step2.sh", "code/run_step2.sh"))
    out.append((HERE / "status_step2.sh", "code/status_step2.sh"))
    out.append((HERE / "HPC_STEP2.md", "code/HPC_STEP2.md"))
    out.append((HERE / "HPC_STEP2.md", "HPC_STEP2.md"))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="D:/m1_tmp/hpc_step2/barca_step2.zip")
    a = ap.parse_args()
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    ents = entries()
    missing = [str(s) for s, _ in ents if not s.exists()]
    if missing:
        raise SystemExit(f"missing: {missing}")
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        for src, name in ents:
            data = src.read_bytes()
            if name.endswith((".sh", ".py", ".md", ".json", ".txt")) and b"\r\n" in data and not name.startswith("code/vendor/"):
                data = data.replace(b"\r\n", b"\n")
            info = zipfile.ZipInfo(name, date_time=(2026, 11, 17, 0, 0, 0))
            info.external_attr = (0o755 if name.endswith(".sh") else 0o644) << 16
            info.compress_type = zipfile.ZIP_DEFLATED
            z.writestr(info, data)
    h = hashlib.sha256(out.read_bytes()).hexdigest()
    print(f"{out}: {len(ents)} files, {out.stat().st_size / 2**20:.1f} MB, sha256 {h[:16]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
