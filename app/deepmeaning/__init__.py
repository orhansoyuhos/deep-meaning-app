"""DeepMeaning package.

Bundles the original Hayes & Henderson (2025) DeepMeaning code together with
the desktop app and CLI wrappers built on top of it.
"""

from .paths import (
    DATA_DIR,
    DEFAULT_INPUT_DIR,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_OVERLAP,
    GRID_DIR,
    MODEL_DIR,
    PATCH_SIZE,
    PATCH_TEMP_DIR,
    PROJECT_ROOT,
    REPO_ROOT,
    SUPPORTED_EXTENSIONS,
)
from .pipeline import (
    GridMatch,
    GridRequiredError,
    InputInspection,
    ProcessingResult,
    find_matching_grids,
    generate_grid_file,
    inspect_input_folder,
    load_grid_match,
    run_meaning_map_job,
)

__all__ = [
    "DATA_DIR",
    "DEFAULT_INPUT_DIR",
    "DEFAULT_OUTPUT_DIR",
    "DEFAULT_OVERLAP",
    "GRID_DIR",
    "GridMatch",
    "GridRequiredError",
    "InputInspection",
    "MODEL_DIR",
    "PATCH_SIZE",
    "PATCH_TEMP_DIR",
    "PROJECT_ROOT",
    "ProcessingResult",
    "REPO_ROOT",
    "SUPPORTED_EXTENSIONS",
    "find_matching_grids",
    "generate_grid_file",
    "inspect_input_folder",
    "load_grid_match",
    "run_meaning_map_job",
]
