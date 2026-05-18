#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Quick standalone viewer for a single DeepMeaning ``.npz`` map.

The GUI app already provides an interactive overlay viewer; this script is
useful for one-off CLI inspection of a generated map next to its source image.

Example
-------
    python scripts/visualize_map.py \\
        --image ../input/familiar_83.png \\
        --map   ../output/familiar_83.npz
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


_CANDIDATE_KEYS = ("array", "average_map", "meaning_map", "map", "arr_0")


def _load_meaning_map(npz_path: Path) -> tuple[np.ndarray, str]:
    data = np.load(npz_path)
    for key in _CANDIDATE_KEYS:
        if key in data.files:
            arr = data[key]
            if arr.ndim == 2:
                return arr, key
    for key in data.files:
        arr = data[key]
        if arr.ndim == 2:
            return arr, key
    raise ValueError(f"No 2D array found in {npz_path}. Keys: {data.files}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    parser.add_argument("--image", type=Path, required=True, help="Original scene image.")
    parser.add_argument("--map", dest="npz", type=Path, required=True, help="Meaning map .npz file.")
    parser.add_argument("--cmap", default="viridis", help="Matplotlib colormap (default: viridis).")
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.65,
        help="Overlay alpha for the combined plot (default: 0.65).",
    )
    args = parser.parse_args(argv)

    if not args.image.is_file():
        print(f"ERROR: image not found: {args.image}", file=sys.stderr)
        return 2
    if not args.npz.is_file():
        print(f"ERROR: map not found: {args.npz}", file=sys.stderr)
        return 2

    meaning_map, used_key = _load_meaning_map(args.npz)
    print(f"Loaded map key '{used_key}', shape {meaning_map.shape}, "
          f"range {meaning_map.min():.3f}..{meaning_map.max():.3f}")

    img = Image.open(args.image)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].imshow(img)
    axes[0].set_title("Original Image")
    axes[0].axis("off")

    im = axes[1].imshow(meaning_map, cmap=args.cmap, origin="upper")
    axes[1].set_title(f"DeepMeaning Map ({used_key})")
    axes[1].axis("off")
    fig.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04).set_label("Meaning")
    plt.tight_layout()

    fig2, ax2 = plt.subplots(figsize=(8, 5))
    ax2.imshow(img)
    im2 = ax2.imshow(meaning_map, cmap=args.cmap, alpha=args.alpha)
    ax2.set_title("Original Image + DeepMeaning Overlay")
    ax2.axis("off")
    fig2.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04).set_label("Meaning")
    plt.tight_layout()
    plt.show()
    return 0


if __name__ == "__main__":
    sys.exit(main())
