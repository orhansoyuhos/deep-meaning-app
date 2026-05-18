"""Tkinter desktop app for generating DeepMeaning maps."""

from __future__ import annotations

import queue
import threading
import tkinter as tk
import webbrowser
from pathlib import Path
from tkinter import filedialog, messagebox, scrolledtext, ttk
from typing import Optional

from ..paths import GRID_DIR
from ..pipeline import (
    GridRequiredError,
    InputInspection,
    ProcessingResult,
    inspect_input_folder,
    run_meaning_map_job,
)
from .viewer import ResultViewer


CITATION_INTRO = (
    "This app uses the DeepMeaning model developed by Hayes & Henderson "
    "(2025). The CoCa-based feature extraction, the leave-one-scene-out "
    "linear ensembles for indoor and outdoor scenes, and the released model "
    "weights are all their work. The original code is available on OSF: "
    "https://osf.io/hcnfx\n\n"
    "This GUI / CLI wraps that pipeline so that patch grid management, "
    "CoCa embedding, model inference, and map reconstruction are handled "
    "automatically.\n\n"
    "If you use this tool in your research, please cite both papers below."
)

CITATION_HAYES = (
    "Hayes, T. R., & Henderson, J. M. (2025). DeepMeaning: Estimating and "
    "Interpreting Scene Meaning for Attention Using a Vision-Language "
    "Transformer. Open Mind, 9, 1020-1036."
)
CITATION_HAYES_DOI = "https://doi.org/10.1162/opmi.a.6"

CITATION_SOYUHOS = (
    "Soyuhos, O., Hayes, T. R., Hu, W., Hamel, T. P., Sevak, B., "
    "Henderson, J. M., & Chen, X. (2026). Meaning-based guidance of "
    "attention in rhesus monkeys during naturalistic scene viewing. "
    "bioRxiv, 2026-03."
)
CITATION_SOYUHOS_DOI = "https://doi.org/10.64898/2026.03.11.711223"


class MeaningMapApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("DeepMeaning Map Generator")
        self.root.geometry("980x860")
        self.root.minsize(900, 800)

        self.event_queue: queue.Queue = queue.Queue()
        self.worker_thread: Optional[threading.Thread] = None

        self.inspection: Optional[InputInspection] = None
        self.result_viewer: Optional[ResultViewer] = None
        self.latest_result: Optional[ProcessingResult] = None
        self.image_running = False

        self.input_var = tk.StringVar()
        self.output_var = tk.StringVar()
        self.category_var = tk.StringVar(value="indoor")
        self.size_var = tk.StringVar(value="No input folder selected")
        self.image_count_var = tk.StringVar(value="")
        self.grid_status_var = tk.StringVar(value="Matching grids will appear here.")
        self.grid_choice_var = tk.StringVar()
        self.progress_var = tk.DoubleVar(value=0.0)
        self.progress_text_var = tk.StringVar(value="Idle")

        self.run_button: Optional[ttk.Button] = None
        self.open_results_button: Optional[ttk.Button] = None
        self.input_button: Optional[ttk.Button] = None
        self.output_button: Optional[ttk.Button] = None
        self.grid_combo: Optional[ttk.Combobox] = None
        self.log_text: Optional[scrolledtext.ScrolledText] = None

        self._build_ui()
        self.root.after(100, self._poll_events)

    def _build_ui(self) -> None:
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)

        notebook = ttk.Notebook(self.root, padding=12)
        notebook.grid(row=0, column=0, sticky="nsew")

        generate_tab = ttk.Frame(notebook)
        citations_tab = ttk.Frame(notebook)
        notebook.add(generate_tab, text="Generate")
        notebook.add(citations_tab, text="About")

        self._build_generate_tab(generate_tab)
        self._build_citations_tab(citations_tab)

    def _build_generate_tab(self, parent: ttk.Frame) -> None:
        container = ttk.Frame(parent, padding=4)
        container.grid(row=0, column=0, sticky="nsew")
        parent.columnconfigure(0, weight=1)
        parent.rowconfigure(0, weight=1)
        container.columnconfigure(0, weight=1)
        container.rowconfigure(3, weight=1)

        paths_frame = ttk.LabelFrame(container, text="Folders", padding=12)
        paths_frame.grid(row=0, column=0, sticky="ew")
        paths_frame.columnconfigure(1, weight=1)

        ttk.Label(paths_frame, text="Input folder").grid(row=0, column=0, sticky="w", pady=(0, 8))
        ttk.Entry(paths_frame, textvariable=self.input_var).grid(row=0, column=1, sticky="ew", padx=(10, 10), pady=(0, 8))
        self.input_button = ttk.Button(paths_frame, text="Browse...", command=self._choose_input_folder)
        self.input_button.grid(row=0, column=2, sticky="ew", pady=(0, 8))

        ttk.Label(paths_frame, text="Output folder").grid(row=1, column=0, sticky="w")
        ttk.Entry(paths_frame, textvariable=self.output_var).grid(row=1, column=1, sticky="ew", padx=(10, 10))
        self.output_button = ttk.Button(paths_frame, text="Browse...", command=self._choose_output_folder)
        self.output_button.grid(row=1, column=2, sticky="ew")

        options_frame = ttk.LabelFrame(container, text="Options", padding=12)
        options_frame.grid(row=1, column=0, sticky="ew", pady=(12, 0))
        options_frame.columnconfigure(1, weight=1)

        ttk.Label(options_frame, text="Scene category").grid(row=0, column=0, sticky="w")
        category_frame = ttk.Frame(options_frame)
        category_frame.grid(row=0, column=1, sticky="w")
        ttk.Radiobutton(category_frame, text="Indoor", variable=self.category_var, value="indoor").pack(side="left", padx=(0, 10))
        ttk.Radiobutton(category_frame, text="Outdoor", variable=self.category_var, value="outdoor").pack(side="left")

        ttk.Label(options_frame, text="Detected image size").grid(row=1, column=0, sticky="w", pady=(10, 0))
        ttk.Label(options_frame, textvariable=self.size_var).grid(row=1, column=1, sticky="w", pady=(10, 0))

        ttk.Label(options_frame, text="Image count").grid(row=2, column=0, sticky="w", pady=(10, 0))
        ttk.Label(options_frame, textvariable=self.image_count_var).grid(row=2, column=1, sticky="w", pady=(10, 0))

        ttk.Label(options_frame, text="Matching grid").grid(row=3, column=0, sticky="w", pady=(10, 0))
        self.grid_combo = ttk.Combobox(options_frame, textvariable=self.grid_choice_var, state="disabled")
        self.grid_combo.grid(row=3, column=1, sticky="ew", pady=(10, 0))

        ttk.Label(options_frame, textvariable=self.grid_status_var, wraplength=780, justify="left").grid(
            row=4,
            column=0,
            columnspan=2,
            sticky="w",
            pady=(10, 0),
        )

        progress_frame = ttk.LabelFrame(container, text="Progress", padding=12)
        progress_frame.grid(row=2, column=0, sticky="ew", pady=(12, 0))
        progress_frame.columnconfigure(0, weight=1)
        ttk.Progressbar(progress_frame, variable=self.progress_var, maximum=100).grid(row=0, column=0, sticky="ew")
        ttk.Label(progress_frame, textvariable=self.progress_text_var).grid(row=1, column=0, sticky="w", pady=(8, 0))

        log_frame = ttk.LabelFrame(container, text="Status Log", padding=12)
        log_frame.grid(row=3, column=0, sticky="nsew", pady=(12, 0))
        log_frame.columnconfigure(0, weight=1)
        log_frame.rowconfigure(0, weight=1)
        self.log_text = scrolledtext.ScrolledText(log_frame, height=14, state="disabled", wrap="word")
        self.log_text.grid(row=0, column=0, sticky="nsew")

        actions = ttk.Frame(container)
        actions.grid(row=4, column=0, sticky="e", pady=(12, 0))
        self.open_results_button = ttk.Button(
            actions,
            text="Open Results Viewer",
            command=self._reopen_result_viewer,
            state="disabled",
        )
        self.open_results_button.pack(side="right", padx=(0, 10))
        self.run_button = ttk.Button(actions, text="Generate Meaning Maps", command=self._start_image_run)
        self.run_button.pack(side="right")

    def _build_citations_tab(self, parent: ttk.Frame) -> None:
        parent.columnconfigure(0, weight=1)
        parent.rowconfigure(2, weight=1)

        ttk.Label(
            parent,
            text=CITATION_INTRO,
            wraplength=860,
            justify="left",
        ).grid(row=0, column=0, sticky="ew", padx=4, pady=(8, 16))

        self._build_citation_block(
            parent,
            row=1,
            title="DeepMeaning model — Hayes & Henderson (2025)",
            body=CITATION_HAYES,
            url=CITATION_HAYES_DOI,
            url_label="Open DOI",
        )

        self._build_citation_block(
            parent,
            row=2,
            title="DeepMeaning App — Soyuhos et al. (2026)",
            body=CITATION_SOYUHOS,
            url=CITATION_SOYUHOS_DOI,
            url_label="Open DOI",
        )

    def _build_citation_block(
        self,
        parent: ttk.Frame,
        row: int,
        title: str,
        body: str,
        url: Optional[str],
        url_label: str = "Open link",
    ) -> None:
        frame = ttk.LabelFrame(parent, text=title, padding=12)
        frame.grid(row=row, column=0, sticky="ew", padx=4, pady=(0, 12))
        frame.columnconfigure(0, weight=1)

        text_widget = tk.Text(
            frame,
            wrap="word",
            height=4,
            relief="flat",
            font=("Segoe UI", 10),
            highlightthickness=0,
            borderwidth=0,
        )
        contents = body if url is None else f"{body}\n{url}"
        text_widget.insert("1.0", contents)
        text_widget.configure(state="disabled")
        text_widget.grid(row=0, column=0, sticky="ew")

        button_row = ttk.Frame(frame)
        button_row.grid(row=1, column=0, sticky="e", pady=(8, 0))
        ttk.Button(
            button_row,
            text="Copy citation",
            command=lambda: self._copy_to_clipboard(body),
        ).pack(side="right")
        if url is not None:
            ttk.Button(
                button_row,
                text=url_label,
                command=lambda: webbrowser.open(url),
            ).pack(side="right", padx=(0, 8))

    def _copy_to_clipboard(self, text: str) -> None:
        self.root.clipboard_clear()
        self.root.clipboard_append(text)
        self.root.update()
        self._append_log("Copied citation to clipboard.")

    def _append_log(self, message: str) -> None:
        if self.log_text is None:
            return
        self.log_text.configure(state="normal")
        self.log_text.insert("end", message.rstrip() + "\n")
        self.log_text.see("end")
        self.log_text.configure(state="disabled")

    def _choose_input_folder(self) -> None:
        selected = filedialog.askdirectory(
            title="Select the input image folder",
            initialdir=self.input_var.get() or str(Path.cwd()),
        )
        if selected:
            self.input_var.set(selected)
            self._inspect_input_folder()

    def _choose_output_folder(self) -> None:
        selected = filedialog.askdirectory(
            title="Select the output folder",
            initialdir=self.output_var.get() or str(Path.cwd()),
        )
        if selected:
            self.output_var.set(selected)

    def _inspect_input_folder(self) -> bool:
        input_folder = self.input_var.get().strip()
        if not input_folder:
            return False
        try:
            inspection = inspect_input_folder(input_folder)
        except Exception as exc:
            self.inspection = None
            self.size_var.set("Unable to inspect selected folder")
            self.image_count_var.set("")
            self.grid_choice_var.set("")
            if self.grid_combo is not None:
                self.grid_combo["values"] = ()
                self.grid_combo.configure(state="disabled")
            self.grid_status_var.set(str(exc))
            self._append_log(f"Input inspection failed: {exc}")
            messagebox.showerror("Input Folder Error", str(exc))
            return False

        self.inspection = inspection
        self.size_var.set(f"{inspection.width} x {inspection.height}")
        self.image_count_var.set(str(len(inspection.image_paths)))

        if inspection.matching_grids:
            grid_names = [match.name for match in inspection.matching_grids]
            if self.grid_combo is not None:
                self.grid_combo["values"] = grid_names
                self.grid_combo.configure(state="readonly")
            self.grid_choice_var.set(grid_names[0])
            self.grid_status_var.set(
                f"Found {len(grid_names)} matching grid(s) in {GRID_DIR}. "
                "You can keep the default selection or choose another match."
            )
            self._append_log(
                f"Detected {len(inspection.image_paths)} image(s) sized "
                f"{inspection.width}x{inspection.height}. Found {len(grid_names)} matching grid(s)."
            )
        else:
            if self.grid_combo is not None:
                self.grid_combo["values"] = ()
                self.grid_combo.configure(state="disabled")
            self.grid_choice_var.set("")
            self.grid_status_var.set(
                f"No grid currently matches {inspection.width}x{inspection.height}. "
                "The app can generate a new grid in the existing grid folder when you run it."
            )
            self._append_log(
                f"Detected {len(inspection.image_paths)} image(s) sized "
                f"{inspection.width}x{inspection.height}. No matching grid exists yet."
            )
        return True

    def _selected_grid_path(self) -> Optional[str]:
        if self.inspection is None:
            return None
        selected_name = self.grid_choice_var.get().strip()
        for match in self.inspection.matching_grids:
            if match.name == selected_name:
                return str(match.path)
        return None

    def _set_image_running(self, running: bool) -> None:
        self.image_running = running
        state = "disabled" if running else "normal"
        combo_state = "disabled"
        if not running and self.inspection and self.inspection.matching_grids:
            combo_state = "readonly"
        open_results_state = "normal" if (not running and self.latest_result is not None) else "disabled"

        if self.run_button is not None:
            self.run_button.configure(state=state)
        if self.open_results_button is not None:
            self.open_results_button.configure(state=open_results_state)
        if self.input_button is not None:
            self.input_button.configure(state=state)
        if self.output_button is not None:
            self.output_button.configure(state=state)
        if self.grid_combo is not None:
            self.grid_combo.configure(state=combo_state)

    def _start_image_run(self) -> None:
        if self.image_running:
            return

        input_folder = self.input_var.get().strip()
        output_folder = self.output_var.get().strip()
        category = self.category_var.get().strip()
        if not input_folder:
            messagebox.showwarning("Missing Input Folder", "Please choose an input folder first.")
            return
        if not output_folder:
            messagebox.showwarning("Missing Output Folder", "Please choose an output folder first.")
            return
        if self.inspection is None or Path(input_folder) != self.inspection.input_dir:
            if not self._inspect_input_folder():
                return

        generate_missing_grid = False
        selected_grid_path = self._selected_grid_path()
        if selected_grid_path is None and self.inspection is not None:
            generate_missing_grid = messagebox.askyesno(
                "Generate New Grid",
                "No matching grid exists for these image dimensions.\n\n"
                f"Generate a new 128px / 73% overlap grid in:\n{GRID_DIR}",
            )
            if not generate_missing_grid:
                self._append_log("Run cancelled because no matching grid was selected or generated.")
                return

        self.progress_var.set(0.0)
        self.progress_text_var.set("Starting...")
        self._set_image_running(True)
        self._append_log(
            f"Starting run with category '{category}', input '{input_folder}', and output '{output_folder}'."
        )

        self.worker_thread = threading.Thread(
            target=self._run_image_worker,
            args=(input_folder, output_folder, category, selected_grid_path, generate_missing_grid),
            daemon=True,
        )
        self.worker_thread.start()

    def _run_image_worker(
        self,
        input_folder: str,
        output_folder: str,
        category: str,
        selected_grid_path: Optional[str],
        generate_missing_grid: bool,
    ) -> None:
        try:
            result = run_meaning_map_job(
                input_folder=input_folder,
                output_folder=output_folder,
                category=category,
                grid_path=selected_grid_path,
                generate_missing_grid=generate_missing_grid,
                progress_callback=lambda percent, message: self.event_queue.put(("progress", percent, message)),
                status_callback=lambda message: self.event_queue.put(("log", message)),
            )
        except GridRequiredError as exc:
            self.event_queue.put(("error", f"No matching grid exists for {exc.width}x{exc.height}."))
        except Exception as exc:
            self.event_queue.put(("error", str(exc)))
        else:
            self.event_queue.put(("done", result))

    def _open_result_viewer(self, result: ProcessingResult) -> None:
        if self.inspection is None:
            return
        try:
            if self.result_viewer is not None and self.result_viewer.window.winfo_exists():
                self.result_viewer.window.destroy()
            self.result_viewer = ResultViewer(self.root, result, self.inspection.image_paths)
            self.result_viewer.window.focus_force()
        except Exception as exc:
            self._append_log(f"Results viewer could not be opened: {exc}")
            messagebox.showwarning("Viewer Error", str(exc))

    def _reopen_result_viewer(self) -> None:
        if self.latest_result is None:
            messagebox.showinfo("No Results Yet", "Run the generator first to open the results viewer.")
            return
        self._open_result_viewer(self.latest_result)

    def _poll_events(self) -> None:
        while True:
            try:
                event = self.event_queue.get_nowait()
            except queue.Empty:
                break

            kind = event[0]
            if kind == "progress":
                _, percent, message = event
                self.progress_var.set(percent)
                self.progress_text_var.set(message)
            elif kind == "log":
                _, message = event
                self._append_log(message)
            elif kind == "error":
                _, message = event
                self._set_image_running(False)
                self.progress_text_var.set("Run failed")
                self._append_log(f"Run failed: {message}")
                messagebox.showerror("DeepMeaning Run Failed", message)
            elif kind == "done":
                _, result = event
                self.latest_result = result
                self.progress_var.set(100.0)
                self.progress_text_var.set("Finished")
                self._append_log(
                    f"Finished. Saved {len(result.output_files)} map(s) to {result.output_dir} "
                    f"using grid {result.grid.name}. Opening results viewer."
                )
                self._inspect_input_folder()
                self._set_image_running(False)
                self._open_result_viewer(result)

        self.root.after(100, self._poll_events)


def launch_app() -> None:
    root = tk.Tk()
    style = ttk.Style(root)
    if "vista" in style.theme_names():
        style.theme_use("vista")
    app = MeaningMapApp(root)
    app._append_log("DeepMeaning app ready. Select an input folder to begin.")
    root.mainloop()
