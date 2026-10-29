# =============================================================================
#  omr.py  --  end-to-end pipeline: sheet-music image -> MIDI file
#  Version: 1.3.0
#
#  Changelog
#  ---------
#  1.3.0  (2026-10-28)
#    - Several pages: `python omr.py p1.png p2.png ...` or a PDF (every page,
#      rendered at PDF_DPI = 300 dpi with PyMuPDF, or poppler's pdftoppm if
#      that is what is installed). Pages are read one after the other, the
#      time signature and tempo of the first page carry over, and one MIDI
#      file is written with each page starting where the previous one ended
#      (load_pages, run_pages, write_midi_pages). A PDF exported from
#      MuseScore is the best input: it is sharp at any size, unlike a
#      screenshot.
#    - Very large pages (staff space above 44 px, e.g. 600 dpi scans) are
#      reduced to 32 px first; recognition is the same, only faster.
#    - Screenshots: everything around the page (grey panels, the taskbar) is
#      painted white first (preprocess.clean_page).
#  1.2.0  (2026-10-28)
#    - Small pages are enlarged first: when the staff space is below MIN_U
#      (16 px), the grey page is resized (bicubic) so that it becomes
#      TARGET_U (20 px). On a Leland (MuseScore) page with a 10-12 px staff
#      space, about half the noteheads were missed before (recall 0.44 at
#      10 px, 0.57 at 12 px). staff["scale"] records the factor; overlay
#      and note coordinates refer to the enlarged page. upscale=False
#      switches it off. The smallest staff decides the factor (a solo part
#      printed small above a piano part needs more enlargement).
#  1.1.0  (2026-10-28)
#    - Time signature check: when fewer than half of the measures add up to
#      the time signature in use (for example the 4/4 default on a piece in
#      12/8) and most measures agree on another length, that length is taken
#      as the time signature and the timing is redone (infer_time_signature).
#      A length that fits both simple and compound time (6/8 or 3/4, 12/8 or
#      6/4) is decided by the beam groups: groups of 3 or 6 notes mean
#      compound time. The change is printed and stored in the result
#      (PipelineResult.time_note); infer_time=False switches it off.
#  1.0.0  (2026-10-25)
#    - run_pipeline(): preprocessing (preprocess: grey, deskew) -> staff
#      detection (staff_detect) -> staff removal
#      (staff_remove) -> environment (env_detect + env_config) -> region
#      localisation (detect) -> template matching (symbol_match) -> pitch and
#      rhythm (interpret) -> MIDI (writer). Returns every intermediate.
#    - Environment: loaded from an env_config JSON, or built from the
#      detected clefs/keys with defaults (--yes) or by asking in the terminal;
#      --time / --bpm / --beat-unit override the answers.
#    - Saves the .mid, an annotated overlay PNG and a JSON list of notes.
#
#  Usage:
#    python omr.py page.png --yes --time 3/4 --bpm 90 --out page
#    python omr.py page.png --env page.env.json --out page
# =============================================================================

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass

import cv2
import numpy as np

from detect import SymbolDetection, detect_symbols
from env_config import Environment, Prompter, build_environment, parse_beat_unit, parse_bpm, parse_time
from env_detect import PageEnvironment, detect_environment
from interpret import Interpretation, interpret
from preprocess import clean_page, deskew, to_gray
from staff_detect import binarize, detect_staves, estimate_staff_metrics
from staff_remove import StaffRemoval, remove_from_detection
from symbol_match import Recognition, draw_recognition, recognise
from writer import MIDIWriter


@dataclass
class PipelineResult:
    gray: np.ndarray
    staff: dict
    removal: StaffRemoval
    page_env: PageEnvironment
    env: Environment
    regions: SymbolDetection
    recognition: Recognition
    music: Interpretation
    timings: dict
    time_note: str = ""


def _time_from_ticks(length: int, compound: bool) -> tuple[int, int] | None:
    eighths, rem = divmod(length, 240)
    if rem or eighths < 2:
        return None
    if compound and eighths % 3 == 0 and eighths >= 6:
        return eighths, 8
    if eighths % 2 == 0:
        return eighths // 2, 4
    return eighths, 8


def infer_time_signature(music: Interpretation, rec: Recognition, current: tuple) -> tuple | None:
    """A time signature that the measures actually add up to, or None when the
    current one already fits (at least half the measures) or no length is
    shared by most measures."""
    lengths = [m[2] for m in music.measures]
    if len(lengths) < 3:
        return None
    want = music.measures[0][3]
    if sum(1 for x in lengths if x == want) >= len(lengths) / 2:
        return None
    best = max(set(lengths), key=lengths.count)
    if lengths.count(best) < 0.6 * len(lengths) or best == want:
        return None
    # beam groups of 3 or 6 columns are the beat groups of compound time
    sizes: dict[int, set] = {}
    for ss in rec.staves:
        for c in ss.chords:
            if c.beam_group >= 0:
                sizes.setdefault(c.beam_group, set()).add(round(c.x / max(rec.u, 1) * 2))
    n_cols = [len(v) for v in sizes.values()]
    triple = sum(1 for n in n_cols if n % 3 == 0)
    duple = sum(1 for n in n_cols if n % 2 == 0 and n % 3 != 0)
    ts = _time_from_ticks(best, compound=triple >= duple)
    return ts if ts and tuple(ts) != tuple(current) else None


def make_environment(page_env: PageEnvironment, env_json: str | None = None, ask: bool = False,
                     time_signature: tuple | None = None, bpm: float | None = None,
                     beat_unit: str | None = None, page: str = "", quiet: bool = True) -> Environment:
    if env_json:
        env = Environment.from_json(env_json)
    else:
        prompt = Prompter(auto=not ask, out=(lambda *a, **k: None) if quiet and not ask else print)
        env = build_environment(page_env, prompt, page=page)
    if time_signature:
        env.time_signature = tuple(time_signature)
    if beat_unit:
        env.beat_unit = beat_unit
    if bpm:
        env.bpm = float(bpm)
    return env


MIN_U = 16.0        # pages with a smaller staff space are enlarged ...
TARGET_U = 20.0     # ... to this staff space (pixels)
MAX_SCALE = 4.0     # at most this enlargement
MAX_U = 44.0        # pages with a larger staff space are reduced ...
TARGET_BIG_U = 32.0  # ... to this one


def run_pipeline(gray: np.ndarray, env: Environment | None = None, straighten: bool = True,
                 infer_time: bool = True, upscale: bool = True, clean: bool = True,
                 **env_kwargs) -> PipelineResult:
    """Runs every stage on one page. `env_kwargs` go to make_environment.
    The page is converted to grey and levelled first (preprocess.deskew)."""
    timings = {}
    t0 = time.perf_counter()
    gray = to_gray(gray)
    page_box = None
    if clean:
        gray, page_box = clean_page(gray)
    skew = 0.0
    if straighten:
        gray, skew = deskew(gray)
    # small pages (screenshots, low-dpi scans): enlarge so that one staff
    # space is about TARGET_U pixels; noteheads of 9-12 px lose their shape
    # in the templates and staff removal otherwise
    scale = 1.0
    if upscale:
        u_page = float(estimate_staff_metrics(binarize(gray)).interline)
        u_small = u_page
        if 4.0 <= u_page < 2 * MIN_U:
            # the smallest staff decides (a small solo staff above a piano part)
            try:
                spacings = [s.spacing for s in detect_staves(gray)["staves"]]
                if spacings:
                    u_small = min(u_page, float(min(spacings)))
            except Exception:
                pass
        if 3.0 <= u_small < MIN_U:
            scale = min(TARGET_U / u_small, MAX_SCALE)
            gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        elif u_page > MAX_U:
            # very high resolution (600 dpi scans): reduce, it only costs time
            scale = TARGET_BIG_U / u_page
            gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    timings["preprocessing"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    staff = detect_staves(gray)
    staff["scale"] = scale
    staff["page_box"] = page_box
    staff["skew_deg"] = skew
    staff["gray"] = gray                       # env_detect.plot_environment reads it
    timings["staff detection"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    removal = remove_from_detection(staff)
    timings["staff removal"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    page_env = detect_environment(staff, removal)
    env = env or make_environment(page_env, **env_kwargs)
    timings["environment"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    regions = detect_symbols(staff, removal)
    timings["region localisation"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    rec = recognise(staff, removal, page_env, regions.regions)
    timings["template matching"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    music = interpret(rec, removal, env)
    note = ""
    if infer_time:
        ts = infer_time_signature(music, rec, env.time_signature)
        if ts:
            old_ts = tuple(env.time_signature)
            env.time_signature = ts
            music2 = interpret(rec, removal, env)
            if music2.measures_ok[0] > music.measures_ok[0]:
                music = music2
                note = (f"time signature {old_ts[0]}/{old_ts[1]} does not fit the measures; "
                        f"using {ts[0]}/{ts[1]} (give --time to override)")
            else:
                env.time_signature = old_ts
    timings["pitch and rhythm"] = time.perf_counter() - t0
    return PipelineResult(gray, staff, removal, page_env, env, regions, rec, music, timings, note)


PDF_DPI = 300


def load_pages(path: str, dpi: int = PDF_DPI) -> list[np.ndarray]:
    """The page images in a file: one for an image, every page for a PDF."""
    if not path.lower().endswith(".pdf"):
        image = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if image is None:
            raise ValueError(f"cannot read {path}")
        return [image]
    try:
        import fitz                                    # PyMuPDF
        pages = []
        with fitz.open(path) as doc:
            for page in doc:
                pix = page.get_pixmap(dpi=dpi, colorspace=fitz.csGRAY)
                pages.append(np.frombuffer(pix.samples, np.uint8).reshape(pix.height, pix.width).copy())
        return pages
    except ImportError:
        pass
    if shutil.which("pdftoppm"):
        with tempfile.TemporaryDirectory() as tmp:
            subprocess.run(["pdftoppm", "-r", str(dpi), "-gray", "-png", path, os.path.join(tmp, "p")], check=True)
            names = sorted(f for f in os.listdir(tmp) if f.endswith(".png"))
            return [cv2.imread(os.path.join(tmp, f), cv2.IMREAD_GRAYSCALE) for f in names]
    raise RuntimeError("reading a PDF needs PyMuPDF (pip install pymupdf) or poppler's pdftoppm; "
                       "or export the score as PNG at 300 dpi")


def run_pages(images: list[np.ndarray], **kwargs) -> list[PipelineResult]:
    """run_pipeline on every page. The first page's time signature, tempo and
    beat unit carry over to the others (later pages do not repeat them), and
    only the first page asks questions."""
    results = []
    for image in images:
        kw = dict(kwargs)
        if results:
            first = results[0].env
            kw["time_signature"] = kw.get("time_signature") or tuple(first.time_signature)
            kw["bpm"] = kw.get("bpm") or first.bpm
            kw["beat_unit"] = kw.get("beat_unit") or first.beat_unit
            kw["ask"] = False
            kw.pop("env_json", None)
        results.append(run_pipeline(image, **kw))
    return results


def write_midi_pages(results: list[PipelineResult], out_stem: str) -> str:
    """One MIDI file for several pages: each page starts where the previous
    page's music ended. Parts are matched by their position in the system."""
    from events import NoteEvent
    from interpret import midi_to_note
    from sequences import Sequences
    env = results[0].env
    parts = sorted(env.parts, key=lambda p: p.slot)
    seqs = {p.slot: Sequences.absolute() for p in parts}
    offset = 0
    for res in results:
        for n in sorted(res.music.notes, key=lambda n: (n.onset, n.midi)):
            if n.part in seqs:
                seqs[n.part].add_event(NoteEvent(midi_to_note(n.midi, env.default_velocity), duration_ticks=n.ticks,
                                                 timestamp_ticks=n.onset + offset))
        offset += max(res.music.end_ticks, max((n.onset + n.ticks for n in res.music.notes), default=0))
    writer = MIDIWriter(out_stem, bpm=env.quarter_bpm)
    writer.time_signature = tuple(env.time_signature)
    for part in parts:
        writer.addSequence(seqs[part.slot], program=part.program, channel=part.channel, name=part.name)
    writer.compile()
    return out_stem + ".mid"


def write_midi(res: PipelineResult, out_stem: str) -> str:
    writer = MIDIWriter(out_stem, bpm=res.env.quarter_bpm)
    writer.time_signature = tuple(res.env.time_signature)
    for part, seq in zip(sorted(res.env.parts, key=lambda p: p.slot), res.music.sequences):
        writer.addSequence(seq, program=part.program, channel=part.channel, name=part.name)
    writer.compile()
    return out_stem + ".mid"


def overlay(res: PipelineResult) -> np.ndarray:
    return draw_recognition(res.gray, res.recognition, res.music.labels)


def notes_json(res: PipelineResult) -> dict:
    return {
        "time_signature": list(res.env.time_signature),
        "quarter_bpm": res.env.quarter_bpm,
        "parts": [{"slot": p.slot, "name": p.name, "instrument": p.instrument, "clef": p.clef, "key": p.key}
                  for p in res.env.parts],
        "notes": [{"part": n.part, "staff": n.staff, "onset": n.onset, "ticks": n.ticks, "midi": n.midi,
                   "name": n.name, "x": round(n.x, 1), "y": round(n.y, 1)} for n in res.music.notes],
        "rests": [{"part": p, "staff": s, "onset": o, "ticks": d, "kind": k} for p, s, o, d, k in res.music.rests],
        "measures_ok": list(res.music.measures_ok),
    }


def _report(res: PipelineResult, label: str = "") -> None:
    good, total = res.music.measures_ok
    if res.staff.get("page_box"):
        print(f"{label}screenshot: only the page area {res.staff['page_box']} was read")
    if res.staff["skew_deg"]:
        print(f"{label}page levelled by {res.staff['skew_deg']:+.2f} degrees")
    if res.staff.get("scale", 1.0) != 1.0:
        tip = "; a PDF or a 300 dpi export reads better" if res.staff["scale"] > 1.6 else ""
        print(f"{label}small page: enlarged x{res.staff['scale']:.2f} before recognition "
              f"(overlay is at that size){tip}")
    if res.time_note:
        print(label + res.time_note)
    for w in res.page_env.warnings:
        if "matched to parts" in w:
            print(f"{label}{w}")
    print(f"{label}staves {len(res.staff['staves'])}, notes {len(res.music.notes)}, rests {len(res.music.rests)}, "
          f"measures matching the time signature {good}/{total}")
    print(f"{label}timings: " + ", ".join(f"{k} {v:.2f}s" for k, v in res.timings.items()))


def main(argv=None):
    p = argparse.ArgumentParser(description="Sheet-music image(s) or PDF to MIDI")
    p.add_argument("image", nargs="+", help="page image(s) in order, or a PDF")
    p.add_argument("--out", help="output stem (default: first file's name)")
    p.add_argument("--env", help="environment JSON from env_config.py (first page)")
    p.add_argument("-y", "--yes", action="store_true", help="accept detected/default settings without asking")
    p.add_argument("--time", type=parse_time, help="time signature, e.g. 3/4 or C")
    p.add_argument("--bpm", type=parse_bpm, help="tempo in beat units per minute")
    p.add_argument("--beat-unit", type=parse_beat_unit, help="quarter, eighth, half, dotted quarter ...")
    p.add_argument("--no-overlay", action="store_true")
    p.add_argument("--no-deskew", action="store_true", help="skip skew correction")
    p.add_argument("--dpi", type=int, default=PDF_DPI, help="resolution for PDF pages")
    args = p.parse_args(argv)

    pages = []
    for path in args.image:
        try:
            pages += load_pages(path, args.dpi)
        except (ValueError, RuntimeError) as e:
            p.error(str(e))
    stem = args.out or args.image[0].rsplit(".", 1)[0]
    kw = dict(straighten=not args.no_deskew, env_json=args.env, ask=not args.yes and not args.env,
              time_signature=args.time, bpm=args.bpm, beat_unit=args.beat_unit, page=args.image[0])
    if len(pages) == 1:
        res = run_pipeline(pages[0], **kw)
        path = write_midi(res, stem)
        with open(stem + ".notes.json", "w") as f:
            json.dump(notes_json(res), f, indent=1)
        if not args.no_overlay:
            cv2.imwrite(stem + ".overlay.png", cv2.cvtColor(overlay(res), cv2.COLOR_RGB2BGR))
        _report(res)
        print(f"wrote {path}")
        return res
    results = run_pages(pages, **kw)
    path = write_midi_pages(results, stem)
    with open(stem + ".notes.json", "w") as f:
        json.dump([notes_json(r) for r in results], f, indent=1)
    for i, res in enumerate(results, 1):
        if not args.no_overlay:
            cv2.imwrite(f"{stem}.p{i}.overlay.png", cv2.cvtColor(overlay(res), cv2.COLOR_RGB2BGR))
        _report(res, f"page {i}: ")
    print(f"wrote {path} ({len(results)} pages)")
    return results


if __name__ == "__main__":
    main()
