"""Central path and constant definitions for the DeepMeaning app."""

from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent      # .../deep-meaning-app/app
REPO_ROOT = PROJECT_ROOT.parent                            # .../deep-meaning-app
DATA_DIR = PROJECT_ROOT / "data"
GRID_DIR = DATA_DIR / "grid"
MODEL_DIR = DATA_DIR / "DeepMeaning"
PATCH_TEMP_DIR = DATA_DIR / "patch_temp"

DEFAULT_INPUT_DIR = REPO_ROOT / "input"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "output"

PATCH_SIZE = 128
DEFAULT_OVERLAP = 0.73
SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
