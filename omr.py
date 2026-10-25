# =============================================================================
#  omr.py  --  end-to-end pipeline: sheet-music image -> MIDI file
#  Version: 1.0.0
#
#  Changelog
#  ---------
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
import time
from dataclasses import dataclass

import cv2
import numpy as np

from detect import SymbolDetection, detect_symbols
from env_config import Environment, Prompter, build_environment, parse_beat_unit, parse_bpm, parse_time
from env_detect import PageEnvironment, detect_environment
from interpret import Interpretation, interpret
from preprocess import deskew, to_gray
from staff_detect import detect_staves
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


def run_pipeline(gray: np.ndarray, env: Environment | None = None, straighten: bool = True,
                 **env_kwargs) -> PipelineResult:
    """Runs every stage on one page. `env_kwargs` go to make_environment.
    The page is converted to grey and levelled first (preprocess.deskew)."""
    timings = {}
    t0 = time.perf_counter()
    gray = to_gray(gray)
    skew = 0.0
    if straighten:
        gray, skew = deskew(gray)
    timings["preprocessing"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    staff = detect_staves(gray)
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
    timings["pitch and rhythm"] = time.perf_counter() - t0
    return PipelineResult(gray, staff, removal, page_env, env, regions, rec, music, timings)


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


def main(argv=None):
    p = argparse.ArgumentParser(description="Sheet-music image to MIDI")
    p.add_argument("image")
    p.add_argument("--out", help="output stem (default: image name)")
    p.add_argument("--env", help="environment JSON from env_config.py")
    p.add_argument("-y", "--yes", action="store_true", help="accept detected/default settings without asking")
    p.add_argument("--time", type=parse_time, help="time signature, e.g. 3/4 or C")
    p.add_argument("--bpm", type=parse_bpm, help="tempo in beat units per minute")
    p.add_argument("--beat-unit", type=parse_beat_unit, help="quarter, eighth, half, dotted quarter ...")
    p.add_argument("--no-overlay", action="store_true")
    p.add_argument("--no-deskew", action="store_true", help="skip skew correction")
    args = p.parse_args(argv)

    gray = cv2.imread(args.image, cv2.IMREAD_UNCHANGED)
    if gray is None:
        p.error(f"cannot read {args.image}")
    stem = args.out or args.image.rsplit(".", 1)[0]
    res = run_pipeline(gray, straighten=not args.no_deskew, env_json=args.env, ask=not args.yes and not args.env,
                       time_signature=args.time, bpm=args.bpm, beat_unit=args.beat_unit, page=args.image)
    path = write_midi(res, stem)
    with open(stem + ".notes.json", "w") as f:
        json.dump(notes_json(res), f, indent=1)
    if not args.no_overlay:
        cv2.imwrite(stem + ".overlay.png", cv2.cvtColor(overlay(res), cv2.COLOR_RGB2BGR))
    good, total = res.music.measures_ok
    if res.staff["skew_deg"]:
        print(f"page levelled by {res.staff['skew_deg']:+.2f} degrees")
    print(f"staves {len(res.staff['staves'])}, notes {len(res.music.notes)}, rests {len(res.music.rests)}, "
          f"measures matching the time signature {good}/{total}")
    print("timings: " + ", ".join(f"{k} {v:.2f}s" for k, v in res.timings.items()))
    print(f"wrote {path}")
    return res


if __name__ == "__main__":
    main()
