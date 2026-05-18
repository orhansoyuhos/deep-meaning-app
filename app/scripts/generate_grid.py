#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Generate a patch grid (.npz) for a given scene size.

Wraps ``deepmeaning.pipeline.generate_grid_file`` with a small CLI. The grid is
written into ``data/grid/`` and is usable by both the GUI and the batch CLI.

Examples
--------
    python scripts/generate_grid.py --width 1024 --height 768
    python scripts/generate_grid.py --width 1440 --height 1080 --overlap 0.73 --show
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = THIS_FILE.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from deepmeaning.paths import GRID_DIR, PATCH_SIZE, DEFAULT_OVERLAP  # noqa: E402
from deepmeaning.pipeline import generate_grid_file  # noqa: E402
from deepmeaning.utils import define_fuzzy_grid  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    parser.add_argument("--width", type=int, required=True, help="Scene width in pixels.")
    parser.add_argument("--height", type=int, required=True, help="Scene height in pixels.")
    parser.add_argument(
        "--patch-size",
        type=int,
        default=PATCH_SIZE,
        help=f"Patch size in pixels (default: {PATCH_SIZE}).",
    )
    parser.add_argument(
        "--overlap",
        type=float,
        default=DEFAULT_OVERLAP,
        help=f"Patch overlap fraction in [0, 1) (default: {DEFAULT_OVERLAP}).",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Render the grid in a matplotlib window before saving.",
    )
    args = parser.parse_args(argv)

    if args.show:
        # The interactive preview uses define_fuzzy_grid directly; the saved
        # grid is still produced by generate_grid_file so naming is consistent.
        define_fuzzy_grid(
            width=args.width,
            height=args.height,
            tile_size=args.patch_size,
            percent_overlap=args.overlap,
            show_grid=True,
        )

    grid = generate_grid_file(
        width=args.width,
        height=args.height,
        patch_size=args.patch_size,
        percent_overlap=args.overlap,
    )
    print(f"Wrote grid: {grid.path}")
    print(f"  Inferred image size: {grid.width} x {grid.height}")
    print(f"  x_grid ({len(grid.x_grid)} cols): {grid.x_grid.tolist()}")
    print(f"  y_grid ({len(grid.y_grid)} rows): {grid.y_grid.tolist()}")
    print(f"  Grid directory: {GRID_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
