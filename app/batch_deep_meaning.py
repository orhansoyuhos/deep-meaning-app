#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Batch CLI for generating DeepMeaning maps.

By default the CLI reads from ``<repo>/input/`` and writes to
``<repo>/output/`` (siblings of the ``app/`` folder). If the input folder
contains ``indoor/`` and/or ``outdoor/`` subfolders, both are processed
automatically; otherwise the input is treated as a single flat folder and
``--category`` is required.

Examples
--------
    # Process the default top-level input/ folder (indoor scenes)
    python batch_deep_meaning.py --category indoor

    # Process a flat folder of indoor scenes with an explicit grid
    python batch_deep_meaning.py --input my_scenes/ --output my_maps/ \\
        --category indoor --grid 1024x768_128px_73over.npz

    # Allow a new grid to be generated for an unseen image size
    python batch_deep_meaning.py --input my_scenes/ --category outdoor --generate-grid

Both the GUI (app.py) and this CLI call ``deepmeaning.pipeline.run_meaning_map_job``,
so the processing path is identical.
"""

from __future__ import annotations

import argparse
import sys
from multiprocessing import freeze_support
from pathlib import Path
from typing import Optional

from deepmeaning import (
    DEFAULT_INPUT_DIR,
    DEFAULT_OUTPUT_DIR,
    GRID_DIR,
    GridRequiredError,
    run_meaning_map_job,
)


DEFAULT_INPUT_ROOT = DEFAULT_INPUT_DIR
DEFAULT_OUTPUT_ROOT = DEFAULT_OUTPUT_DIR


def _print_progress(percent: float, message: str) -> None:
    print(f"[{percent:6.2f}%] {message}", flush=True)


def _print_status(message: str) -> None:
    print(f"  - {message}", flush=True)


def _resolve_grid_path(grid_arg: Optional[str]) -> Optional[str]:
    if grid_arg is None:
        return None
    candidate = Path(grid_arg)
    if candidate.is_file():
        return str(candidate)
    candidate_in_grid_dir = GRID_DIR / grid_arg
    if candidate_in_grid_dir.is_file():
        return str(candidate_in_grid_dir)
    raise FileNotFoundError(
        f"Grid file not found: tried '{candidate}' and '{candidate_in_grid_dir}'."
    )


def _run_one(
    input_folder: Path,
    output_folder: Path,
    category: str,
    grid_path: Optional[str],
    generate_missing_grid: bool,
) -> None:
    print(f"\n=== Processing {category} scenes ===")
    print(f"  Input : {input_folder}")
    print(f"  Output: {output_folder}")

    result = run_meaning_map_job(
        input_folder=input_folder,
        output_folder=output_folder,
        category=category,
        grid_path=grid_path,
        generate_missing_grid=generate_missing_grid,
        progress_callback=_print_progress,
        status_callback=_print_status,
    )
    print(
        f"  Done. {len(result.output_files)} map(s) -> {result.output_dir} "
        f"(grid: {result.grid.name})"
    )


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Generate DeepMeaning maps for a folder of scenes.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT_ROOT,
        help=(
            "Input folder. If it contains 'indoor/' and/or 'outdoor/' subfolders, "
            "both are processed automatically. Otherwise treated as a single flat "
            "folder and --category is required. Default: %(default)s"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help=(
            "Output folder. When the indoor/outdoor layout is detected, maps are "
            "written into 'indoor/' and 'outdoor/' subfolders. Default: %(default)s"
        ),
    )
    parser.add_argument(
        "--category",
        choices=("indoor", "outdoor"),
        help="Required when --input is a flat folder of scenes.",
    )
    parser.add_argument(
        "--grid",
        help=(
            "Grid file name (looked up in data/grid/) or full path. "
            "If omitted, a matching grid is auto-detected from the image size."
        ),
    )
    parser.add_argument(
        "--generate-grid",
        action="store_true",
        help="Allow a new grid to be generated when no matching grid exists.",
    )
    args = parser.parse_args(argv)

    try:
        grid_path = _resolve_grid_path(args.grid)
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    input_root: Path = args.input
    output_root: Path = args.output

    if not input_root.exists():
        print(f"ERROR: Input folder does not exist: {input_root}", file=sys.stderr)
        return 2

    indoor_subdir = input_root / "indoor"
    outdoor_subdir = input_root / "outdoor"
    has_layout = indoor_subdir.is_dir() or outdoor_subdir.is_dir()

    try:
        if has_layout and args.category is None:
            if indoor_subdir.is_dir():
                _run_one(
                    input_folder=indoor_subdir,
                    output_folder=output_root / "indoor",
                    category="indoor",
                    grid_path=grid_path,
                    generate_missing_grid=args.generate_grid,
                )
            if outdoor_subdir.is_dir():
                _run_one(
                    input_folder=outdoor_subdir,
                    output_folder=output_root / "outdoor",
                    category="outdoor",
                    grid_path=grid_path,
                    generate_missing_grid=args.generate_grid,
                )
        else:
            if args.category is None:
                print(
                    "ERROR: --category is required when --input is a flat folder "
                    "(no 'indoor/' or 'outdoor/' subfolder detected).",
                    file=sys.stderr,
                )
                return 2
            _run_one(
                input_folder=input_root,
                output_folder=output_root,
                category=args.category,
                grid_path=grid_path,
                generate_missing_grid=args.generate_grid,
            )
    except GridRequiredError as exc:
        print(
            f"ERROR: {exc} Use --generate-grid to create a new one in {GRID_DIR}, "
            "or pass --grid <existing.npz>.",
            file=sys.stderr,
        )
        return 1
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    freeze_support()
    sys.exit(main())
