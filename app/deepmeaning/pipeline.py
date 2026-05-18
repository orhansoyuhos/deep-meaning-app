"""Image-only processing pipeline for DeepMeaning.

This is the single source of truth used by both the desktop GUI and the
batch CLI. It composes the paper-authors' CoCa + linear-model code into a
reusable job that takes a folder of images, a grid, and a scene category
and writes one ``.npz`` meaning map per image.
"""

from __future__ import annotations

import pickle
import shutil
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Callable, Optional, Sequence, Union

import numpy as np
import pandas as pd
import torch
from PIL import Image

from .coca import get_embeddings as get_coca_embeddings
from .paths import (
    DEFAULT_OVERLAP,
    GRID_DIR,
    MODEL_DIR,
    PATCH_SIZE,
    PATCH_TEMP_DIR,
    SUPPORTED_EXTENSIONS,
)
from .utils import (
    create_patches_scene,
    define_fuzzy_grid,
    get_scene_from_path,
    load_grid,
)

ProgressCallback = Callable[[float, str], None]
StatusCallback = Callable[[str], None]
PathLike = Union[str, Path]


@dataclass(frozen=True)
class GridMatch:
    name: str
    path: Path
    width: int
    height: int
    patch_size: int
    x_grid: np.ndarray
    y_grid: np.ndarray


@dataclass(frozen=True)
class InputInspection:
    input_dir: Path
    image_paths: tuple[Path, ...]
    width: int
    height: int
    matching_grids: tuple[GridMatch, ...]


@dataclass(frozen=True)
class ProcessingResult:
    category: str
    grid: GridMatch
    output_dir: Path
    output_files: tuple[Path, ...]
    image_count: int


class GridRequiredError(RuntimeError):
    """Raised when no grid matches the selected image size."""

    def __init__(self, width: int, height: int) -> None:
        super().__init__(f"No grid exists for images sized {width}x{height}.")
        self.width = width
        self.height = height


def _emit_progress(callback: Optional[ProgressCallback], fraction: float, message: str) -> None:
    if callback is not None:
        fraction = max(0.0, min(1.0, fraction))
        callback(fraction * 100.0, message)


def _emit_status(callback: Optional[StatusCallback], message: str) -> None:
    if callback is not None:
        callback(message)


def _list_supported_images(input_dir: Path) -> list[Path]:
    return [
        path for path in sorted(input_dir.iterdir())
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    ]


def _infer_grid_size(x_grid: np.ndarray, y_grid: np.ndarray, patch_size: int = PATCH_SIZE) -> tuple[int, int]:
    return int(np.max(x_grid) + patch_size), int(np.max(y_grid) + patch_size)


def load_grid_match(grid_path: PathLike, patch_size: int = PATCH_SIZE) -> GridMatch:
    grid_path = Path(grid_path)
    x_grid, y_grid = load_grid(str(grid_path))
    width, height = _infer_grid_size(x_grid, y_grid, patch_size)
    return GridMatch(
        name=grid_path.name,
        path=grid_path,
        width=width,
        height=height,
        patch_size=patch_size,
        x_grid=x_grid,
        y_grid=y_grid,
    )


def find_matching_grids(
    width: int,
    height: int,
    grid_dir: PathLike = GRID_DIR,
    patch_size: int = PATCH_SIZE,
) -> list[GridMatch]:
    grid_dir = Path(grid_dir)
    matches: list[GridMatch] = []

    for grid_path in sorted(grid_dir.glob("*.npz")):
        try:
            match = load_grid_match(grid_path, patch_size=patch_size)
        except Exception:
            continue
        if match.width == width and match.height == height:
            matches.append(match)

    preferred_prefix = f"{width}x{height}_"
    matches.sort(key=lambda item: (0 if item.name.startswith(preferred_prefix) else 1, item.name.lower()))
    return matches


def inspect_input_folder(input_dir: PathLike) -> InputInspection:
    input_dir = Path(input_dir)
    if not input_dir.exists():
        raise FileNotFoundError(f"Input folder does not exist: {input_dir}")
    if not input_dir.is_dir():
        raise NotADirectoryError(f"Input path is not a folder: {input_dir}")

    image_paths = _list_supported_images(input_dir)
    if not image_paths:
        raise ValueError(
            "No supported images were found in the selected folder. "
            "Supported formats: PNG, JPG, JPEG, BMP, TIF, TIFF."
        )

    sizes: dict[tuple[int, int], list[str]] = {}
    for image_path in image_paths:
        with Image.open(image_path) as image:
            size = image.size
        sizes.setdefault(size, []).append(image_path.name)

    if len(sizes) != 1:
        size_summary = "; ".join(
            f"{width}x{height} ({len(names)} file(s))"
            for (width, height), names in sorted(sizes.items())
        )
        raise ValueError(
            "All input images must have the same size. "
            f"Found multiple sizes: {size_summary}"
        )

    width, height = next(iter(sizes.keys()))
    matching_grids = tuple(find_matching_grids(width, height))

    return InputInspection(
        input_dir=input_dir,
        image_paths=tuple(image_paths),
        width=width,
        height=height,
        matching_grids=matching_grids,
    )


def generate_grid_file(
    width: int,
    height: int,
    grid_dir: PathLike = GRID_DIR,
    patch_size: int = PATCH_SIZE,
    percent_overlap: float = DEFAULT_OVERLAP,
) -> GridMatch:
    grid_dir = Path(grid_dir)
    grid_dir.mkdir(parents=True, exist_ok=True)

    overlap_label = int(round(percent_overlap * 100))
    grid_name = f"{width}x{height}_{patch_size}px_{overlap_label}over.npz"
    grid_path = grid_dir / grid_name

    if grid_path.exists():
        try:
            existing_grid = load_grid_match(grid_path, patch_size=patch_size)
            if existing_grid.width == width and existing_grid.height == height:
                return existing_grid
        except Exception:
            pass

    x_grid, y_grid = define_fuzzy_grid(
        width=width,
        height=height,
        tile_size=patch_size,
        percent_overlap=percent_overlap,
        show_grid=False,
    )
    np.savez_compressed(grid_path, x_grid=x_grid, y_grid=y_grid)
    return load_grid_match(grid_path, patch_size=patch_size)


@lru_cache(maxsize=2)
def _load_model(category: str):
    if category not in {"indoor", "outdoor"}:
        raise ValueError("Category must be 'indoor' or 'outdoor'.")

    model_path = MODEL_DIR / f"DeepMeaning_{category}_ensemble.pkl"
    with open(model_path, "rb") as handle:
        return pickle.load(handle)["model"]


def _reset_patch_temp_dir() -> None:
    """Empty the patch scratch folder without removing the folder itself.

    On Windows with Box Sync / OneDrive watching the project, the agent holds
    a handle on the directory itself, so ``rmtree(PATCH_TEMP_DIR)`` fails with
    WinError 5 ("Access is denied"). File-level deletes inside the folder are
    still allowed, so we just iterate and unlink instead.
    """
    PATCH_TEMP_DIR.mkdir(parents=True, exist_ok=True)
    for entry in PATCH_TEMP_DIR.iterdir():
        try:
            if entry.is_dir():
                shutil.rmtree(entry, ignore_errors=True)
            else:
                entry.unlink()
        except OSError:
            # Best-effort: a leftover file is preferable to aborting the run.
            pass


def _extract_patches(
    inspection: InputInspection,
    grid: GridMatch,
    progress_callback: Optional[ProgressCallback] = None,
    status_callback: Optional[StatusCallback] = None,
) -> None:
    _emit_status(status_callback, "Extracting image patches...")
    total_images = len(inspection.image_paths)

    for index, image_path in enumerate(inspection.image_paths, start=1):
        create_patches_scene(
            (
                image_path.name,
                str(inspection.input_dir),
                str(PATCH_TEMP_DIR),
                grid.x_grid,
                grid.y_grid,
                grid.patch_size,
            )
        )
        _emit_progress(
            progress_callback,
            0.05 + (0.30 * index / total_images),
            f"Extracted patches for {image_path.name} ({index}/{total_images})",
        )


def _build_predictions_dataframe(
    filename_mapping: dict[str, int],
    predictions: Sequence[float],
    category: str,
) -> pd.DataFrame:
    rows = []
    for filename, index in sorted(filename_mapping.items(), key=lambda item: item[1]):
        rows.append(
            {
                "model": "coca",
                "category": category,
                "scene": get_scene_from_path(filename),
                "image": filename,
                "predicted": float(predictions[index]),
            }
        )
    return pd.DataFrame(rows)


def _save_scene_maps(
    predictions_df: pd.DataFrame,
    output_dir: Path,
    grid: GridMatch,
    progress_callback: Optional[ProgressCallback] = None,
    status_callback: Optional[StatusCallback] = None,
) -> tuple[Path, ...]:
    _emit_status(status_callback, "Reconstructing scene meaning maps...")
    output_dir.mkdir(parents=True, exist_ok=True)

    scenes = list(predictions_df["scene"].drop_duplicates())
    output_files: list[Path] = []

    for scene_index, scene in enumerate(scenes, start=1):
        scene_predictions = predictions_df[predictions_df["scene"] == scene]
        prediction_lookup = dict(zip(scene_predictions["image"], scene_predictions["predicted"]))

        sum_map = np.zeros((grid.height, grid.width), dtype=np.float64)
        count_map = np.zeros((grid.height, grid.width), dtype=np.uint16)

        patch_index = 0
        for y_coord in grid.y_grid:
            for x_coord in grid.x_grid:
                patch_name = f"{scene}_{patch_index}.png"
                prediction = prediction_lookup.get(patch_name)
                if prediction is not None:
                    sum_map[y_coord:y_coord + grid.patch_size, x_coord:x_coord + grid.patch_size] += prediction
                    count_map[y_coord:y_coord + grid.patch_size, x_coord:x_coord + grid.patch_size] += 1
                patch_index += 1

        average_map = np.zeros_like(sum_map)
        valid_pixels = count_map > 0
        average_map[valid_pixels] = sum_map[valid_pixels] / count_map[valid_pixels]

        output_path = output_dir / f"{scene}.npz"
        np.savez_compressed(output_path, array=average_map)
        output_files.append(output_path)

        _emit_progress(
            progress_callback,
            0.82 + (0.18 * scene_index / len(scenes)),
            f"Saved meaning map for {scene} ({scene_index}/{len(scenes)})",
        )

    return tuple(output_files)


def run_meaning_map_job(
    input_folder: PathLike,
    output_folder: PathLike,
    category: str,
    grid_path: Optional[PathLike] = None,
    generate_missing_grid: bool = False,
    patch_size: int = PATCH_SIZE,
    percent_overlap: float = DEFAULT_OVERLAP,
    progress_callback: Optional[ProgressCallback] = None,
    status_callback: Optional[StatusCallback] = None,
) -> ProcessingResult:
    """Run the full DeepMeaning pipeline on a folder of same-sized images.

    Used by both the GUI (deepmeaning.gui) and the batch CLI
    (batch_deep_meaning.py at the project root).
    """
    category = category.lower().strip()
    if category not in {"indoor", "outdoor"}:
        raise ValueError("Category must be 'indoor' or 'outdoor'.")

    inspection = inspect_input_folder(input_folder)
    _emit_progress(progress_callback, 0.02, "Validated input folder")
    _emit_status(
        status_callback,
        f"Found {len(inspection.image_paths)} image(s) sized {inspection.width}x{inspection.height}.",
    )

    if grid_path is not None:
        grid = load_grid_match(grid_path, patch_size=patch_size)
        if grid.width != inspection.width or grid.height != inspection.height:
            raise ValueError(
                f"Selected grid {grid.name} is for {grid.width}x{grid.height}, "
                f"but the images are {inspection.width}x{inspection.height}."
            )
    elif inspection.matching_grids:
        grid = inspection.matching_grids[0]
    elif generate_missing_grid:
        _emit_status(
            status_callback,
            f"Generating a new grid for {inspection.width}x{inspection.height} images...",
        )
        grid = generate_grid_file(
            inspection.width,
            inspection.height,
            patch_size=patch_size,
            percent_overlap=percent_overlap,
        )
    else:
        raise GridRequiredError(inspection.width, inspection.height)

    _emit_progress(progress_callback, 0.05, f"Using grid {grid.name}")
    _emit_status(status_callback, f"Using grid file: {grid.path}")

    output_dir = Path(output_folder)
    output_dir.mkdir(parents=True, exist_ok=True)

    _reset_patch_temp_dir()
    try:
        _extract_patches(inspection, grid, progress_callback, status_callback)

        device = "cuda" if torch.cuda.is_available() else "cpu"
        _emit_status(status_callback, f"Embedding device: {device.upper()}")

        embeddings, filename_mapping = get_coca_embeddings(
            str(PATCH_TEMP_DIR),
            save_dir=str(PATCH_TEMP_DIR),
            batch_size=32,
            device=device,
            force_compute=True,
            num_workers=0,
            progress_callback=lambda current, total: _emit_progress(
                progress_callback,
                0.35 + (0.40 * current / total),
                f"Computed CoCa embeddings ({current}/{total} batches)",
            ),
            status_callback=status_callback,
        )

        _emit_status(status_callback, f"Predicting patch meaning with the {category} model...")
        model = _load_model(category)
        predictions = model.predict(embeddings)
        _emit_progress(progress_callback, 0.80, "Predicted patch meaning scores")

        predictions_df = _build_predictions_dataframe(filename_mapping, predictions, category)
        output_files = _save_scene_maps(predictions_df, output_dir, grid, progress_callback, status_callback)
        _emit_progress(progress_callback, 1.0, "Finished generating meaning maps")

        return ProcessingResult(
            category=category,
            grid=grid,
            output_dir=output_dir,
            output_files=output_files,
            image_count=len(inspection.image_paths),
        )
    finally:
        _reset_patch_temp_dir()
