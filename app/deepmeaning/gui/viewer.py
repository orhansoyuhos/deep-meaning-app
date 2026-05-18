"""Overlay results viewer used by the DeepMeaning GUI."""

from __future__ import annotations

import tkinter as tk
from pathlib import Path
from tkinter import ttk
from typing import Optional, Sequence

import numpy as np
from matplotlib import cm
from PIL import Image, ImageTk

from ..pipeline import ProcessingResult


RESAMPLING = getattr(Image, "Resampling", Image)
DISPLAY_RESAMPLE = RESAMPLING.LANCZOS
OVERLAY_ALPHA_DEFAULT = 0.65
DEFAULT_COLORMAP = "inferno"
COLORMAP_OPTIONS = ("viridis", "plasma", "inferno", "magma")
DISPLAY_MAX_SIZE = (1120, 720)


class ResultViewer:
    """Toplevel window for browsing meaning-map overlays."""

    def __init__(
        self,
        parent: tk.Tk,
        result: ProcessingResult,
        input_images: Sequence[Path],
    ) -> None:
        self.window = tk.Toplevel(parent)
        self.window.title("DeepMeaning Results Viewer")
        self.window.geometry("1280x940")
        self.window.minsize(980, 780)

        self.image_lookup = {image_path.stem: image_path for image_path in input_images}
        self.items: list[tuple[Path, Path]] = []
        for output_path in result.output_files:
            source_path = self.image_lookup.get(output_path.stem)
            if source_path is not None and source_path.exists():
                self.items.append((output_path, source_path))

        if not self.items:
            self.window.destroy()
            raise ValueError("No original images were available for overlay viewing.")

        self.index = 0
        self.alpha_var = tk.DoubleVar(value=OVERLAY_ALPHA_DEFAULT)
        self.colormap_var = tk.StringVar(value=DEFAULT_COLORMAP)
        self.selection_var = tk.StringVar(value=self.items[0][0].stem)
        self.info_var = tk.StringVar()

        self.image_label: Optional[ttk.Label] = None
        self.photo_image: Optional[ImageTk.PhotoImage] = None
        self.overlay_cache: dict = {}
        self.source_cache: dict = {}

        self._build_ui()
        self._render_current()

    def _build_ui(self) -> None:
        self.window.columnconfigure(0, weight=1)
        self.window.rowconfigure(1, weight=1)

        controls = ttk.Frame(self.window, padding=12)
        controls.grid(row=0, column=0, sticky="ew")
        controls.columnconfigure(3, weight=1)
        controls.columnconfigure(5, weight=1)
        controls.columnconfigure(7, weight=1)

        ttk.Button(controls, text="Previous", command=self._show_previous).grid(row=0, column=0, padx=(0, 8))
        ttk.Button(controls, text="Next", command=self._show_next).grid(row=0, column=1, padx=(0, 12))

        ttk.Label(controls, text="Image").grid(row=0, column=2, sticky="w")
        selector = ttk.Combobox(
            controls,
            textvariable=self.selection_var,
            state="readonly",
            values=[item[0].stem for item in self.items],
            width=32,
        )
        selector.grid(row=0, column=3, sticky="ew", padx=(8, 16))
        selector.bind("<<ComboboxSelected>>", self._on_selection_changed)

        ttk.Label(controls, text="Overlay alpha").grid(row=0, column=4, sticky="w")
        alpha_scale = ttk.Scale(
            controls,
            variable=self.alpha_var,
            from_=0.0,
            to=1.0,
            command=self._on_alpha_changed,
        )
        alpha_scale.grid(row=0, column=5, sticky="ew", padx=(8, 16))

        ttk.Label(controls, text="Colormap").grid(row=0, column=6, sticky="w")
        colormap_selector = ttk.Combobox(
            controls,
            textvariable=self.colormap_var,
            state="readonly",
            values=COLORMAP_OPTIONS,
            width=12,
        )
        colormap_selector.grid(row=0, column=7, sticky="ew", padx=(8, 0))
        colormap_selector.bind("<<ComboboxSelected>>", self._on_colormap_changed)

        ttk.Label(controls, textvariable=self.info_var, justify="left").grid(
            row=1,
            column=0,
            columnspan=8,
            sticky="w",
            pady=(10, 0),
        )

        canvas_frame = ttk.Frame(self.window, padding=(12, 0, 12, 12))
        canvas_frame.grid(row=1, column=0, sticky="nsew")
        canvas_frame.columnconfigure(0, weight=1)
        canvas_frame.rowconfigure(0, weight=1)

        self.image_label = ttk.Label(canvas_frame, anchor="center")
        self.image_label.grid(row=0, column=0, sticky="nsew")

    def _current_item(self) -> tuple[Path, Path]:
        return self.items[self.index]

    def _on_selection_changed(self, _event=None) -> None:
        selected_name = self.selection_var.get().strip()
        for index, (output_path, _source_path) in enumerate(self.items):
            if output_path.stem == selected_name:
                self.index = index
                break
        self._render_current()

    def _on_alpha_changed(self, _value=None) -> None:
        self._render_current()

    def _on_colormap_changed(self, _event=None) -> None:
        self._render_current()

    def _show_previous(self) -> None:
        self.index = (self.index - 1) % len(self.items)
        self.selection_var.set(self.items[self.index][0].stem)
        self._render_current()

    def _show_next(self) -> None:
        self.index = (self.index + 1) % len(self.items)
        self.selection_var.set(self.items[self.index][0].stem)
        self._render_current()

    def _load_source_image(self, source_path: Path) -> Image.Image:
        cached = self.source_cache.get(source_path)
        if cached is None:
            with Image.open(source_path) as image:
                cached = image.convert("RGB")
            self.source_cache[source_path] = cached
        return cached.copy()

    def _load_overlay_image(self, output_path: Path, target_size: tuple[int, int]):
        colormap_name = self.colormap_var.get().strip() or DEFAULT_COLORMAP
        cache_key = (output_path, target_size, colormap_name)
        cached = self.overlay_cache.get(cache_key)
        if cached is not None:
            return cached

        map_data = np.load(output_path)["array"]
        map_min = float(np.min(map_data))
        map_max = float(np.max(map_data))
        if map_max > map_min:
            normalized = (map_data - map_min) / (map_max - map_min)
        else:
            normalized = np.zeros_like(map_data)

        colored = (cm.get_cmap(colormap_name)(normalized)[..., :3] * 255).astype(np.uint8)
        overlay = Image.fromarray(colored, mode="RGB")
        if overlay.size != target_size:
            overlay = overlay.resize(target_size, DISPLAY_RESAMPLE)

        cached = (overlay, map_data.shape, map_min, map_max)
        self.overlay_cache[cache_key] = cached
        return cached

    def _render_current(self) -> None:
        if self.image_label is None:
            return

        output_path, source_path = self._current_item()
        base_image = self._load_source_image(source_path)
        overlay_image, map_shape, map_min, map_max = self._load_overlay_image(output_path, base_image.size)

        alpha = float(self.alpha_var.get())
        colormap_name = self.colormap_var.get().strip() or DEFAULT_COLORMAP
        composite = Image.blend(base_image, overlay_image, alpha)
        display_image = composite.copy()
        display_image.thumbnail(DISPLAY_MAX_SIZE, DISPLAY_RESAMPLE)

        self.photo_image = ImageTk.PhotoImage(display_image)
        self.image_label.configure(image=self.photo_image)
        self.info_var.set(
            f"Image: {source_path.name}    Map: {output_path.name}    "
            f"Map shape: {map_shape[1]} x {map_shape[0]}    Alpha: {alpha:.2f}    "
            f"Colormap: {colormap_name}    Meaning range: {map_min:.3f} to {map_max:.3f}"
        )
