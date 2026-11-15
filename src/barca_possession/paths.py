"""Paths shared by the hackathon pipeline (one place)."""
from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
REF = ROOT / "reference" / "more-than-a-hack-2027"
OUT = Path(os.environ.get("BARCA_OUT", ROOT / "outputs"))        # git-ignored; data-derived files (never shown to an assistant)
AUTO = Path(os.environ.get("BARCA_AUTO", ROOT / "reports" / "auto"))   # aggregate numbers only, read by make_step1_report.py
MODELS = Path(os.environ.get("BARCA_MODELS", ROOT / "models"))
PFF_PACKAGE = Path(os.environ.get("M1_HPC_PACKAGE", r"D:\m1_tmp\hpc\m1_hpc_v1"))   # M1 HPC package (30 learning matches)
FPS = 25.0
PITCH = (105.3, 68.0)                         # games.py, measured from the corrected tracking


def data_dir() -> Path:
    """The hackathon bundle: MTAH_DATA_DIR, else data/ or the unpacked download folder inside it (whichever has tactical/)."""
    if os.environ.get("MTAH_DATA_DIR"):
        return Path(os.environ["MTAH_DATA_DIR"])
    for d in (ROOT / "data", ROOT / "data" / "More Than A Hack 2027 - Initial Challenge - Dataset"):
        if (d / "tactical").is_dir():
            return d
    return ROOT / "data"


def official():
    """Import the official games / starter / evaluation modules (read-only submodule). Sets MTAH_DATA_DIR first, because
    games.py reads it at import."""
    os.environ.setdefault("MTAH_DATA_DIR", str(data_dir()))
    for p in (REF, REF / "starter", REF / "evaluation"):
        if str(p) not in sys.path:
            sys.path.insert(0, str(p))
    import games  # noqa: F401
    return games
