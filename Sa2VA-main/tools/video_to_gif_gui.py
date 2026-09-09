#!/usr/bin/env python3
"""Simple GUI tool to convert a local video file to GIF."""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Optional

import tkinter as tk
from tkinter import filedialog, messagebox


VIDEO_FILETYPES = [
    ("Video files", "*.mp4 *.avi *.mov *.mkv *.webm *.flv *.wmv *.m4v"),
    ("All files", "*.*"),
]

GIF_FILETYPES = [
    ("GIF file", "*.gif"),
    ("All files", "*.*"),
]


class VideoToGifApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Video to GIF Converter")
        self.root.geometry("760x300")
        self.root.minsize(700, 280)

        self.input_path_var = tk.StringVar()
        self.output_path_var = tk.StringVar()
        self.fps_var = tk.StringVar(value="10")
        self.width_var = tk.StringVar(value="480")
        self.status_var = tk.StringVar(value="Ready")
        self.is_converting = False

        self._build_ui()

    def _build_ui(self) -> None:
        frame = tk.Frame(self.root, padx=14, pady=14)
        frame.pack(fill=tk.BOTH, expand=True)

        title = tk.Label(
            frame,
            text="Convert Local Video to GIF",
            font=("Arial", 14, "bold"),
            anchor="w",
        )
        title.grid(row=0, column=0, columnspan=3, sticky="w", pady=(0, 12))

        tk.Label(frame, text="Input video:", anchor="w").grid(
            row=1, column=0, sticky="w", pady=6
        )
        tk.Entry(frame, textvariable=self.input_path_var, width=72).grid(
            row=1, column=1, sticky="we", pady=6, padx=(8, 8)
        )
        tk.Button(frame, text="Browse", command=self.choose_input).grid(
            row=1, column=2, sticky="e", pady=6
        )

        tk.Label(frame, text="Output GIF:", anchor="w").grid(
            row=2, column=0, sticky="w", pady=6
        )
        tk.Entry(frame, textvariable=self.output_path_var, width=72).grid(
            row=2, column=1, sticky="we", pady=6, padx=(8, 8)
        )
        tk.Button(frame, text="Browse", command=self.choose_output).grid(
            row=2, column=2, sticky="e", pady=6
        )

        tk.Label(frame, text="FPS:", anchor="w").grid(row=3, column=0, sticky="w", pady=6)
        tk.Entry(frame, textvariable=self.fps_var, width=10).grid(
            row=3, column=1, sticky="w", pady=6, padx=(8, 0)
        )

        tk.Label(frame, text="Max width (px):", anchor="w").grid(
            row=4, column=0, sticky="w", pady=6
        )
        tk.Entry(frame, textvariable=self.width_var, width=10).grid(
            row=4, column=1, sticky="w", pady=6, padx=(8, 0)
        )

        btn_frame = tk.Frame(frame)
        btn_frame.grid(row=5, column=0, columnspan=3, sticky="w", pady=(14, 8))
        self.convert_btn = tk.Button(btn_frame, text="Convert to GIF", command=self.on_convert)
        self.convert_btn.pack(side=tk.LEFT)
        tk.Button(btn_frame, text="Exit", command=self.root.destroy).pack(side=tk.LEFT, padx=(8, 0))

        tk.Label(frame, textvariable=self.status_var, fg="#1155cc", anchor="w").grid(
            row=6, column=0, columnspan=3, sticky="w", pady=(8, 0)
        )

        frame.grid_columnconfigure(1, weight=1)

    def choose_input(self) -> None:
        chosen = filedialog.askopenfilename(
            title="Select input video",
            filetypes=VIDEO_FILETYPES,
        )
        if chosen:
            self.input_path_var.set(chosen)
            if not self.output_path_var.get().strip():
                in_path = Path(chosen)
                self.output_path_var.set(str(in_path.with_suffix(".gif")))

    def choose_output(self) -> None:
        initial = self.output_path_var.get().strip()
        chosen = filedialog.asksaveasfilename(
            title="Choose output GIF path",
            defaultextension=".gif",
            filetypes=GIF_FILETYPES,
            initialfile=Path(initial).name if initial else "output.gif",
        )
        if chosen:
            output = Path(chosen)
            if output.suffix.lower() != ".gif":
                output = output.with_suffix(".gif")
            self.output_path_var.set(str(output))

    def on_convert(self) -> None:
        if self.is_converting:
            return
        try:
            input_path, output_path, fps, width = self._validate_inputs()
        except ValueError as exc:
            messagebox.showerror("Invalid Input", str(exc))
            return

        self.is_converting = True
        self.convert_btn.config(state=tk.DISABLED)
        self.status_var.set("Converting... please wait")

        worker = threading.Thread(
            target=self._convert_worker,
            args=(input_path, output_path, fps, width),
            daemon=True,
        )
        worker.start()

    def _validate_inputs(self) -> tuple[Path, Path, int, Optional[int]]:
        input_path_str = self.input_path_var.get().strip().strip('"')
        output_path_str = self.output_path_var.get().strip().strip('"')

        if not input_path_str:
            raise ValueError("Please choose an input video file.")
        if not output_path_str:
            raise ValueError("Please choose an output GIF path.")

        input_path = Path(input_path_str)
        output_path = Path(output_path_str)

        if not input_path.exists() or not input_path.is_file():
            raise ValueError("Input video file does not exist.")

        if output_path.suffix.lower() != ".gif":
            output_path = output_path.with_suffix(".gif")
            self.output_path_var.set(str(output_path))

        output_parent = output_path.parent
        if not output_parent.exists():
            raise ValueError("Output folder does not exist.")

        try:
            fps = int(self.fps_var.get().strip())
        except ValueError as exc:
            raise ValueError("FPS must be an integer.") from exc
        if fps <= 0:
            raise ValueError("FPS must be greater than 0.")

        width_raw = self.width_var.get().strip()
        width: Optional[int] = None
        if width_raw:
            try:
                width = int(width_raw)
            except ValueError as exc:
                raise ValueError("Max width must be an integer.") from exc
            if width <= 0:
                raise ValueError("Max width must be greater than 0.")

        return input_path, output_path, fps, width

    def _convert_worker(self, input_path: Path, output_path: Path, fps: int, width: Optional[int]) -> None:
        try:
            self._convert_video_to_gif(input_path, output_path, fps, width)
        except Exception as exc:
            self.root.after(0, self._on_convert_failed, str(exc))
            return
        self.root.after(0, self._on_convert_success, output_path)

    @staticmethod
    def _convert_video_to_gif(input_path: Path, output_path: Path, fps: int, width: Optional[int]) -> None:
        try:
            from moviepy.editor import VideoFileClip
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "Missing dependency: moviepy. Install it with:\n"
                "python -m pip install moviepy imageio imageio-ffmpeg"
            ) from exc

        clip = VideoFileClip(str(input_path))
        try:
            if width is not None:
                clip = clip.resize(width=width)
            clip.write_gif(
                str(output_path),
                fps=fps,
                program="ffmpeg",
                logger=None,
            )
        finally:
            clip.close()

    def _on_convert_success(self, output_path: Path) -> None:
        self.is_converting = False
        self.convert_btn.config(state=tk.NORMAL)
        self.status_var.set(f"Done: {output_path}")
        messagebox.showinfo("Success", f"GIF saved to:\n{output_path}")

    def _on_convert_failed(self, error_text: str) -> None:
        self.is_converting = False
        self.convert_btn.config(state=tk.NORMAL)
        self.status_var.set("Conversion failed")
        messagebox.showerror("Conversion Failed", error_text)


def main() -> None:
    root = tk.Tk()
    app = VideoToGifApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
