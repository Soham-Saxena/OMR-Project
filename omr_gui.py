# =============================================================================
#  omr_gui.py  --  desktop interface (Tkinter) for the OMR pipeline
#  Version: 1.0.0
#
#  Changelog
#  ---------
#  1.0.0  (2026-10-25)
#    - Open a page image, set tempo / beat unit / time signature / instrument
#      / deskew, convert, view the page or the recognition overlay (zoom),
#      read a summary (staves, clefs, keys, notes, measure check, timings),
#      save the MIDI file and the overlay PNG.
#    - The pipeline runs on a worker thread; the window polls for the result
#      with after(), so it never freezes during a conversion.
#    - All non-Tk logic sits in OmrController, which can be used and tested
#      without a display.
#
#  Usage:  python omr_gui.py [page.png]
# =============================================================================

from __future__ import annotations

import base64
import os
import queue
import sys
import threading
import traceback

import cv2
import numpy as np

from env_config import BEAT_UNITS, GENERAL_MIDI, TRANSPOSITION, parse_bpm, parse_time
from env_detect import Clef, key_name
from interpret import interpret
from omr import PipelineResult, overlay, run_pipeline, write_midi

AUTO_INSTRUMENT = "(as detected)"


class OmrController:
    """State and actions behind the window; no Tk inside."""

    def __init__(self):
        self.path: str | None = None
        self.image: np.ndarray | None = None
        self.result: PipelineResult | None = None
        self.overlay_rgb: np.ndarray | None = None

    def load(self, path: str) -> np.ndarray:
        image = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if image is None:
            raise ValueError(f"Cannot read {path}")
        self.path, self.image, self.result, self.overlay_rgb = path, image, None, None
        return image

    def convert(self, bpm: str, beat_unit: str, time_signature: str, instrument: str,
                deskew: bool) -> PipelineResult:
        if self.image is None:
            raise ValueError("Open an image first.")
        res = run_pipeline(self.image, straighten=deskew, time_signature=parse_time(time_signature),
                           bpm=parse_bpm(bpm), beat_unit=beat_unit)
        if instrument and instrument != AUTO_INSTRUMENT:
            program = GENERAL_MIDI.index(instrument)
            for part in res.env.parts:
                part.instrument, part.program = instrument, program
                part.transpose = TRANSPOSITION.get(instrument, 0)
            res.music = interpret(res.recognition, res.removal, res.env)   # transposition may change pitches
        self.result = res
        self.overlay_rgb = overlay(res)
        return res

    def summary(self) -> str:
        res = self.result
        if res is None:
            return ""
        lines = []
        if res.staff.get("skew_deg"):
            lines.append(f"Page levelled by {res.staff['skew_deg']:+.2f} degrees")
        lines.append(f"Staves: {len(res.staff['staves'])}  (staff space U = {res.staff['interline']:.1f} px)")
        lines.append(f"Systems: {[len(s) for s in res.page_env.systems]}")
        for s in res.env.staves:
            lines.append(f"  staff {s.index}: {Clef.parse(s.clef).name}, {key_name(s.key)}")
        lines.append(f"Time signature: {res.env.time_signature[0]}/{res.env.time_signature[1]}, "
                     f"tempo {res.env.quarter_bpm:g} quarter notes per minute")
        for p in res.env.parts:
            lines.append(f"Part {p.slot + 1}: {p.name}, {p.instrument} (program {p.program})")
        good, total = res.music.measures_ok
        lines.append(f"Notes: {len(res.music.notes)}   rests: {len(res.music.rests)}")
        lines.append(f"Measures matching the time signature: {good}/{total}")
        lines.append("Time: " + ", ".join(f"{k} {v:.2f}s" for k, v in res.timings.items()))
        return "\n".join(lines)

    def save_midi(self, path: str) -> str:
        if self.result is None:
            raise ValueError("Convert a page first.")
        stem = path[:-4] if path.lower().endswith(".mid") else path
        return write_midi(self.result, stem)

    def save_overlay(self, path: str):
        if self.overlay_rgb is None:
            raise ValueError("Convert a page first.")
        cv2.imwrite(path, cv2.cvtColor(self.overlay_rgb, cv2.COLOR_RGB2BGR))


def png_data(image: np.ndarray, zoom: float) -> str:
    """Base64 PNG of `image` scaled by `zoom`, for tk.PhotoImage(data=...)."""
    if image.ndim == 3 and image.shape[2] == 3:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    if zoom != 1.0:
        image = cv2.resize(image, None, fx=zoom, fy=zoom, interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".png", image)
    if not ok:
        raise ValueError("PNG encoding failed")
    return base64.b64encode(buf.tobytes()).decode("ascii")


def main():
    import tkinter as tk
    from tkinter import filedialog, messagebox, ttk

    ctl = OmrController()
    root = tk.Tk()
    root.title("OMR: sheet music to MIDI")
    root.geometry("1280x800")

    # ---- left: controls ---------------------------------------------------
    side = ttk.Frame(root, padding=8)
    side.pack(side=tk.LEFT, fill=tk.Y)
    file_label = ttk.Label(side, text="No image", width=34, wraplength=250)

    settings = ttk.LabelFrame(side, text="Settings", padding=6)
    bpm_var = tk.StringVar(value="100")
    unit_var = tk.StringVar(value="quarter")
    time_var = tk.StringVar(value="4/4")
    inst_var = tk.StringVar(value=AUTO_INSTRUMENT)
    deskew_var = tk.BooleanVar(value=True)
    ttk.Label(settings, text="Tempo (beats per minute)").grid(row=0, column=0, sticky="w")
    ttk.Spinbox(settings, from_=10, to=400, textvariable=bpm_var, width=8).grid(row=0, column=1, sticky="w")
    ttk.Label(settings, text="Beat unit").grid(row=1, column=0, sticky="w")
    ttk.Combobox(settings, textvariable=unit_var, values=list(BEAT_UNITS), state="readonly",
                 width=14).grid(row=1, column=1, sticky="w")
    ttk.Label(settings, text="Time signature").grid(row=2, column=0, sticky="w")
    ttk.Entry(settings, textvariable=time_var, width=8).grid(row=2, column=1, sticky="w")
    ttk.Label(settings, text="Instrument").grid(row=3, column=0, sticky="w")
    ttk.Combobox(settings, textvariable=inst_var, values=[AUTO_INSTRUMENT] + GENERAL_MIDI, state="readonly",
                 width=24).grid(row=4, column=0, columnspan=2, sticky="we")
    ttk.Checkbutton(settings, text="Straighten tilted pages", variable=deskew_var).grid(row=5, column=0,
                                                                                       columnspan=2, sticky="w")

    view_var = tk.StringVar(value="page")
    zoom_var = tk.StringVar(value="50%")
    status = tk.StringVar(value="Open a sheet-music image to begin.")
    log = tk.Text(side, width=40, height=22, wrap="word", font=("TkFixedFont", 9))

    # ---- right: image -----------------------------------------------------
    viewer = ttk.Frame(root)
    viewer.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
    canvas = tk.Canvas(viewer, background="#808080")
    xs = ttk.Scrollbar(viewer, orient=tk.HORIZONTAL, command=canvas.xview)
    ys = ttk.Scrollbar(viewer, orient=tk.VERTICAL, command=canvas.yview)
    canvas.configure(xscrollcommand=xs.set, yscrollcommand=ys.set)
    ys.pack(side=tk.RIGHT, fill=tk.Y)
    xs.pack(side=tk.BOTTOM, fill=tk.X)
    canvas.pack(fill=tk.BOTH, expand=True)
    photo = {"img": None}

    def show():
        image = ctl.overlay_rgb if view_var.get() == "overlay" and ctl.overlay_rgb is not None else ctl.image
        if image is None:
            return
        zoom = int(zoom_var.get().rstrip("%")) / 100.0
        photo["img"] = tk.PhotoImage(data=png_data(image, zoom))
        canvas.delete("all")
        canvas.create_image(0, 0, image=photo["img"], anchor="nw")
        canvas.configure(scrollregion=(0, 0, photo["img"].width(), photo["img"].height()))

    def open_image(path=None):
        path = path or filedialog.askopenfilename(
            title="Open sheet music", filetypes=[("Images", "*.png *.jpg *.jpeg *.bmp *.tif *.tiff"), ("All", "*.*")])
        if not path:
            return
        try:
            ctl.load(path)
        except ValueError as e:
            messagebox.showerror("Open", str(e))
            return
        file_label.configure(text=os.path.basename(path))
        view_var.set("page")
        log.delete("1.0", tk.END)
        status.set("Image loaded. Check the settings, then Convert.")
        show()

    results: queue.Queue = queue.Queue()

    def convert():
        if ctl.image is None:
            messagebox.showinfo("Convert", "Open an image first.")
            return
        convert_btn.state(["disabled"])
        status.set("Converting...")
        args = (bpm_var.get(), unit_var.get(), time_var.get(), inst_var.get(), deskew_var.get())

        def work():
            try:
                ctl.convert(*args)
                results.put(("ok", None))
            except Exception as e:          # report any failure in the window
                results.put(("error", f"{e}\n\n{traceback.format_exc()}"))
        threading.Thread(target=work, daemon=True).start()
        root.after(100, poll)

    def poll():
        try:
            kind, payload = results.get_nowait()
        except queue.Empty:
            root.after(100, poll)
            return
        convert_btn.state(["!disabled"])
        if kind == "error":
            status.set("Conversion failed.")
            messagebox.showerror("Convert", payload.split("\n\n")[0])
            log.delete("1.0", tk.END)
            log.insert(tk.END, payload)
            return
        status.set("Done. Save the MIDI file, or inspect the overlay.")
        log.delete("1.0", tk.END)
        log.insert(tk.END, ctl.summary())
        view_var.set("overlay")
        show()

    def save_midi():
        try:
            path = filedialog.asksaveasfilename(defaultextension=".mid", filetypes=[("MIDI", "*.mid")],
                                                initialfile=os.path.splitext(os.path.basename(ctl.path or "page"))[0])
            if path:
                status.set(f"Saved {ctl.save_midi(path)}")
        except ValueError as e:
            messagebox.showinfo("Save MIDI", str(e))

    def save_overlay():
        try:
            path = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG", "*.png")])
            if path:
                ctl.save_overlay(path)
                status.set(f"Saved {path}")
        except ValueError as e:
            messagebox.showinfo("Save overlay", str(e))

    ttk.Button(side, text="Open image...", command=open_image).pack(fill=tk.X)
    file_label.pack(fill=tk.X, pady=(4, 8))
    settings.pack(fill=tk.X)
    convert_btn = ttk.Button(side, text="Convert", command=convert)
    convert_btn.pack(fill=tk.X, pady=(8, 2))
    ttk.Button(side, text="Save MIDI...", command=save_midi).pack(fill=tk.X, pady=2)
    ttk.Button(side, text="Save overlay...", command=save_overlay).pack(fill=tk.X, pady=2)
    view = ttk.LabelFrame(side, text="View", padding=6)
    view.pack(fill=tk.X, pady=(8, 4))
    ttk.Radiobutton(view, text="Page", value="page", variable=view_var, command=show).grid(row=0, column=0)
    ttk.Radiobutton(view, text="Recognised", value="overlay", variable=view_var, command=show).grid(row=0, column=1)
    zoom_box = ttk.Combobox(view, textvariable=zoom_var, values=["25%", "50%", "75%", "100%"], state="readonly",
                            width=6)
    zoom_box.grid(row=0, column=2, padx=(8, 0))
    zoom_box.bind("<<ComboboxSelected>>", lambda e: show())
    ttk.Label(side, textvariable=status, wraplength=260).pack(fill=tk.X, pady=4)
    log.pack(fill=tk.BOTH, expand=True)

    if len(sys.argv) > 1:
        root.after(50, lambda: open_image(sys.argv[1]))
    root.mainloop()


if __name__ == "__main__":
    main()
