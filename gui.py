#!/usr/bin/env python3
"""entropy_wall GUI — Tkinter front end for the CLI in entropy_wall.py.

Three tabs (Photo / Video / Webcam), shared options (seed length,
noise-extract), and a results panel with copy/save. Webcam has a live
preview and a Stream mode that keeps emitting fresh seeds on an interval,
mirroring `entropy_wall.py webcam --stream`.

Capture/hashing runs on a background thread so the window never freezes;
the background thread only ever talks back to the UI thread through a
queue.Queue, polled with `root.after(...)` — the one safe way to update
Tkinter widgets from outside the main thread.
"""

from __future__ import annotations

import queue
import sys
import threading
import time
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from entropy import EntropyPool, run_pool, shannon_entropy_bits_per_byte
from entropy.capture import extract_noise, frames_from_image, frames_from_video, frames_from_webcam

try:
    import cv2
except ImportError:
    cv2 = None

try:
    from PIL import Image, ImageTk
except ImportError:
    Image = ImageTk = None


PREVIEW_SIZE = (320, 240)


class EntropyWallGUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Entropy Wall")
        self.resizable(False, False)

        self._queue: "queue.Queue[tuple]" = queue.Queue()
        self._busy = False
        self._last_seed: bytes | None = None
        self._stream_stop = threading.Event()
        self._streaming = False
        self._preview_photo = None  # keep a reference or Tk garbage-collects it

        self._build_ui()
        self.after(50, self._poll_queue)
        self.protocol("WM_DELETE_WINDOW", self._on_close)

    # ---------------------------------------------------------------- UI --

    def _build_ui(self) -> None:
        self.notebook = ttk.Notebook(self)
        self.notebook.pack(padx=10, pady=(10, 0), fill="x")

        self.photo_tab = self._build_photo_tab()
        self.video_tab = self._build_video_tab()
        self.webcam_tab = self._build_webcam_tab()
        self.notebook.add(self.photo_tab, text="Photo")
        self.notebook.add(self.video_tab, text="Video")
        self.notebook.add(self.webcam_tab, text="Webcam")
        self.notebook.bind("<<NotebookTabChanged>>", lambda e: self._sync_generate_button())

        # --- shared options ---
        opts = ttk.LabelFrame(self, text="Options")
        opts.pack(padx=10, pady=8, fill="x")

        ttk.Label(opts, text="Seed length (bytes):").grid(row=0, column=0, sticky="w", padx=5, pady=5)
        self.length_var = tk.IntVar(value=32)
        ttk.Spinbox(opts, from_=8, to=1024, textvariable=self.length_var, width=8).grid(
            row=0, column=1, sticky="w", padx=5, pady=5
        )

        self.noise_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            opts, text="Noise-extract (frame delta / LSB plane instead of raw pixels)",
            variable=self.noise_var,
        ).grid(row=0, column=2, sticky="w", padx=5, pady=5)

        # --- action row ---
        actions = ttk.Frame(self)
        actions.pack(padx=10, pady=(0, 8), fill="x")
        self.generate_btn = ttk.Button(actions, text="Generate", command=self._on_generate)
        self.generate_btn.pack(side="left")
        self.status_var = tk.StringVar(value="Ready.")
        ttk.Label(actions, textvariable=self.status_var).pack(side="left", padx=10)

        # --- results ---
        results = ttk.LabelFrame(self, text="Result")
        results.pack(padx=10, pady=(0, 10), fill="both")

        self.info_var = tk.StringVar(value="")
        ttk.Label(results, textvariable=self.info_var, justify="left").pack(
            anchor="w", padx=8, pady=(6, 2)
        )

        seed_row = ttk.Frame(results)
        seed_row.pack(fill="x", padx=8, pady=(0, 8))
        self.seed_entry = ttk.Entry(seed_row, font=("Consolas", 9))
        self.seed_entry.pack(side="left", fill="x", expand=True)
        ttk.Button(seed_row, text="Copy", command=self._copy_seed, width=6).pack(side="left", padx=(4, 0))
        ttk.Button(seed_row, text="Save...", command=self._save_seed, width=7).pack(side="left", padx=(4, 0))

        self.log = tk.Text(results, height=6, width=64, font=("Consolas", 9), state="disabled")
        self.log.pack(padx=8, pady=(0, 8), fill="both")

    def _build_photo_tab(self) -> ttk.Frame:
        f = ttk.Frame(self.notebook, padding=10)
        ttk.Label(f, text="Photo file:").grid(row=0, column=0, sticky="w")
        self.photo_path = tk.StringVar()
        ttk.Entry(f, textvariable=self.photo_path, width=48).grid(row=1, column=0, sticky="w")
        ttk.Button(f, text="Browse...", command=self._pick_photo).grid(row=1, column=1, padx=(6, 0))
        return f

    def _build_video_tab(self) -> ttk.Frame:
        f = ttk.Frame(self.notebook, padding=10)
        ttk.Label(f, text="Video file:").grid(row=0, column=0, sticky="w")
        self.video_path = tk.StringVar()
        ttk.Entry(f, textvariable=self.video_path, width=48).grid(row=1, column=0, sticky="w")
        ttk.Button(f, text="Browse...", command=self._pick_video).grid(row=1, column=1, padx=(6, 0))
        ttk.Label(f, text="Use every Nth frame:").grid(row=2, column=0, sticky="w", pady=(8, 0))
        self.sample_every = tk.IntVar(value=1)
        ttk.Spinbox(f, from_=1, to=100, textvariable=self.sample_every, width=6).grid(
            row=3, column=0, sticky="w"
        )
        return f

    def _build_webcam_tab(self) -> ttk.Frame:
        f = ttk.Frame(self.notebook, padding=10)

        left = ttk.Frame(f)
        left.grid(row=0, column=0, sticky="n")

        ttk.Label(left, text="Device index:").grid(row=0, column=0, sticky="w")
        self.device_var = tk.IntVar(value=0)
        ttk.Spinbox(left, from_=0, to=8, textvariable=self.device_var, width=6).grid(
            row=1, column=0, sticky="w"
        )

        ttk.Label(left, text="Capture seconds:").grid(row=2, column=0, sticky="w", pady=(8, 0))
        self.seconds_var = tk.DoubleVar(value=5.0)
        ttk.Spinbox(
            left, from_=0.5, to=120, increment=0.5, textvariable=self.seconds_var, width=6
        ).grid(row=3, column=0, sticky="w")

        ttk.Separator(left, orient="horizontal").grid(row=4, column=0, sticky="ew", pady=10)

        self.stream_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            left, text="Stream mode (run continuously)", variable=self.stream_var,
            command=self._sync_generate_button,
        ).grid(row=5, column=0, sticky="w")

        ttk.Label(left, text="Emit interval (s):").grid(row=6, column=0, sticky="w", pady=(6, 0))
        self.interval_var = tk.DoubleVar(value=2.0)
        ttk.Spinbox(
            left, from_=0.5, to=60, increment=0.5, textvariable=self.interval_var, width=6
        ).grid(row=7, column=0, sticky="w")

        right = ttk.Frame(f)
        right.grid(row=0, column=1, sticky="n", padx=(16, 0))
        self.preview_canvas = tk.Canvas(
            right, width=PREVIEW_SIZE[0], height=PREVIEW_SIZE[1], background="#222"
        )
        self.preview_canvas.pack()
        if cv2 is None or Image is None:
            self.preview_canvas.create_text(
                PREVIEW_SIZE[0] // 2, PREVIEW_SIZE[1] // 2,
                text="preview needs opencv-python\n+ Pillow installed",
                fill="white", justify="center",
            )

        return f

    # ------------------------------------------------------------- utils --

    def _pick_photo(self) -> None:
        path = filedialog.askopenfilename(
            title="Choose a photo",
            filetypes=[("Images", "*.jpg *.jpeg *.png *.bmp *.webp"), ("All files", "*.*")],
        )
        if path:
            self.photo_path.set(path)

    def _pick_video(self) -> None:
        path = filedialog.askopenfilename(
            title="Choose a video",
            filetypes=[("Videos", "*.mp4 *.avi *.mov *.mkv"), ("All files", "*.*")],
        )
        if path:
            self.video_path.set(path)

    def _sync_generate_button(self) -> None:
        on_webcam = self.notebook.index(self.notebook.select()) == 2
        if on_webcam and self.stream_var.get():
            self.generate_btn.config(text="Stop Streaming" if self._streaming else "Start Streaming")
        else:
            self.generate_btn.config(text="Generate")

    def _log(self, line: str) -> None:
        self.log.config(state="normal")
        self.log.insert("end", line + "\n")
        self.log.see("end")
        self.log.config(state="disabled")

    def _copy_seed(self) -> None:
        if not self._last_seed:
            return
        self.clipboard_clear()
        self.clipboard_append(self._last_seed.hex())
        self.status_var.set("Seed copied to clipboard.")

    def _save_seed(self) -> None:
        if not self._last_seed:
            messagebox.showinfo("Nothing to save", "Generate a seed first.")
            return
        path = filedialog.asksaveasfilename(
            title="Save seed bytes", defaultextension=".bin", filetypes=[("Binary", "*.bin")]
        )
        if not path:
            return
        with open(path, "wb") as fh:
            fh.write(self._last_seed)
        self.status_var.set(f"Saved to {path}")

    def _set_busy(self, busy: bool) -> None:
        self._busy = busy
        state = "disabled" if busy else "normal"
        for tab in (self.photo_tab, self.video_tab):
            for child in tab.winfo_children():
                try:
                    child.config(state=state)
                except tk.TclError:
                    pass

    # -------------------------------------------------------- dispatch --

    def _on_generate(self) -> None:
        if self._busy:
            return
        tab_index = self.notebook.index(self.notebook.select())
        if tab_index == 2 and self.stream_var.get():
            self._toggle_stream()
            return

        try:
            if tab_index == 0:
                path = self.photo_path.get().strip()
                if not path:
                    messagebox.showwarning("No file", "Choose a photo first.")
                    return
                frames_fn = lambda: frames_from_image(path)
                tag = "image"
            elif tab_index == 1:
                path = self.video_path.get().strip()
                if not path:
                    messagebox.showwarning("No file", "Choose a video first.")
                    return
                # Read the Tk variable here, on the main thread — Tk/Tcl
                # variables aren't safe to touch from a background thread
                # (it can deadlock the Tcl interpreter), so only plain
                # Python values get captured into the worker closure.
                sample_every = self.sample_every.get()
                frames_fn = lambda: frames_from_video(path, sample_every=sample_every)
                tag = "video"
            else:
                if cv2 is None:
                    messagebox.showerror("Missing dependency", "opencv-python is required for webcam capture.")
                    return
                device, seconds = self.device_var.get(), self.seconds_var.get()
                frames_fn = lambda: frames_from_webcam(device=device, seconds=seconds)
                tag = "webcam"
        except Exception as exc:  # bad user input, e.g. non-numeric spinbox
            messagebox.showerror("Invalid input", str(exc))
            return

        noise_extract = self.noise_var.get()
        length = self.length_var.get()

        def work():
            try:
                pool, sample = run_pool(frames_fn(), noise_extract=noise_extract, tag=tag)
                self._queue.put(("result", pool, sample, length))
            except Exception as exc:
                self._queue.put(("error", str(exc)))

        self._set_busy(True)
        self.generate_btn.config(state="disabled")
        self.status_var.set("Working...")
        threading.Thread(target=work, daemon=True).start()

    # ------------------------------------------------------ stream mode --

    def _toggle_stream(self) -> None:
        if self._streaming:
            self._stream_stop.set()
            return

        if cv2 is None:
            messagebox.showerror("Missing dependency", "opencv-python is required for webcam capture.")
            return

        device = self.device_var.get()
        interval = self.interval_var.get()
        noise_extract = self.noise_var.get()
        length = self.length_var.get()

        self._stream_stop.clear()
        self._streaming = True
        self.generate_btn.config(text="Stop Streaming")
        self.status_var.set(f"Streaming from device {device}...")

        def work():
            pool = EntropyPool()
            previous = None
            last_emit = time.monotonic()
            i = 0
            try:
                for frame in frames_from_webcam(device=device, seconds=None):
                    if self._stream_stop.is_set():
                        break
                    payload = extract_noise(frame, previous) if noise_extract else frame
                    previous = frame
                    pool.absorb(payload.tobytes(), tag=f"webcam:{i}")
                    i += 1
                    now = time.monotonic()
                    do_emit = now - last_emit >= interval
                    self._queue.put(("stream_frame", frame, pool if do_emit else None, length))
                    if do_emit:
                        last_emit = now
            except Exception as exc:
                self._queue.put(("error", str(exc)))
            finally:
                self._queue.put(("stream_stopped",))

        threading.Thread(target=work, daemon=True).start()

    # ----------------------------------------------------------- queue --

    def _poll_queue(self) -> None:
        try:
            while True:
                msg = self._queue.get_nowait()
                self._handle_msg(msg)
        except queue.Empty:
            pass
        self.after(50, self._poll_queue)

    def _handle_msg(self, msg: tuple) -> None:
        kind = msg[0]
        if kind == "result":
            _, pool, sample, length = msg
            self._show_result(pool, sample, length)
            self._set_busy(False)
            self.generate_btn.config(state="normal")
            self.status_var.set("Done.")
        elif kind == "error":
            self._set_busy(False)
            self.generate_btn.config(state="normal")
            self.status_var.set("Error.")
            messagebox.showerror("Error", msg[1])
        elif kind == "stream_frame":
            _, frame, pool, length = msg
            self._update_preview(frame)
            if pool is not None:
                seed = pool.derive(length)
                self._last_seed = seed
                self.seed_entry.delete(0, "end")
                self.seed_entry.insert(0, seed.hex())
                self._log(f"[{pool.frames_absorbed} frames] {seed.hex()}")
        elif kind == "stream_stopped":
            self._streaming = False
            self.generate_btn.config(text="Start Streaming")
            self.status_var.set("Stream stopped.")

    def _show_result(self, pool: EntropyPool, sample: bytes, length: int) -> None:
        seed = pool.derive(length)
        self._last_seed = seed
        entropy_bits = shannon_entropy_bits_per_byte(sample)
        self.info_var.set(
            f"Frames absorbed: {pool.frames_absorbed}    "
            f"Source entropy: ~{entropy_bits:.2f} bits/byte (diagnostic only)"
        )
        self.seed_entry.delete(0, "end")
        self.seed_entry.insert(0, seed.hex())
        self._log(f"[{pool.frames_absorbed} frames] {seed.hex()}")

    def _update_preview(self, frame) -> None:
        if Image is None or cv2 is None:
            return
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb).resize(PREVIEW_SIZE)
        self._preview_photo = ImageTk.PhotoImage(img)
        self.preview_canvas.create_image(0, 0, anchor="nw", image=self._preview_photo)

    def _on_close(self) -> None:
        self._stream_stop.set()
        self.destroy()


def main() -> None:
    app = EntropyWallGUI()
    app.mainloop()


if __name__ == "__main__":
    main()
