# =============================================================================
#  evaluate.py  --  scores the pipeline on engraved pages with ground truth
#  Version: 1.0.0
#
#  Changelog
#  ---------
#  1.0.0  (2026-10-25)
#    - Per page: staff count and line position error, staff-removal pixel
#      precision/recall/F1 (staff_remove.evaluate_removal against the
#      staff-free rendering), clef and key accuracy per staff, notehead
#      precision/recall/F1, notehead type, pitch, duration and accidental
#      accuracy on matched heads, rest detection, measure check, and an
#      end-to-end comparison of the written MIDI file with the ground truth
#      (onset+pitch F1 and pitch-sequence accuracy by edit distance).
#    - Writes one JSON per page and a summary table (CSV + printed).
#    - Measure-level note F1: pitch, measure number and offset in the
#      measure all right (localises timing errors to their measure).
#    - --confirmed: clef/key/time taken from the ground truth (what a user
#      confirms in env_config), to separate recognition errors from
#      env_detect's clef/key errors.
#
#  Usage:  python eval/evaluate.py eval/pages --out eval/results
# =============================================================================

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import mido  # noqa: E402
from omr import run_pipeline, write_midi  # noqa: E402
from env_config import Environment, Part, StaffSetting  # noqa: E402
from staff_remove import evaluate_removal  # noqa: E402

HEAD_TOL_U = 0.5
REST_TOL_U = 0.8


def greedy_match(gt_pts: list, det_pts: list, tol: float) -> list[tuple[int, int]]:
    """One-to-one matching by increasing distance, within tol."""
    pairs = []
    for i, (gx, gy) in enumerate(gt_pts):
        for j, (dx, dy) in enumerate(det_pts):
            d = np.hypot(gx - dx, gy - dy)
            if d <= tol:
                pairs.append((d, i, j))
    pairs.sort()
    used_g, used_d, out = set(), set(), []
    for d, i, j in pairs:
        if i not in used_g and j not in used_d:
            used_g.add(i)
            used_d.add(j)
            out.append((i, j))
    return out


def prf(tp: int, n_gt: int, n_det: int) -> dict:
    p = tp / n_det if n_det else 0.0
    r = tp / n_gt if n_gt else 0.0
    return {"precision": p, "recall": r, "f1": 2 * p * r / (p + r) if p + r else 0.0, "tp": tp,
            "gt": n_gt, "det": n_det}


def midi_notes(path: str) -> list[tuple[int, int, int, int]]:
    """(track, onset, pitch, duration) from a MIDI file."""
    out = []
    for ti, track in enumerate(mido.MidiFile(path).tracks):
        now, active = 0, {}
        for msg in track:
            now += msg.time
            if msg.is_meta:
                continue
            if msg.type == "note_on" and msg.velocity > 0:
                active.setdefault(msg.note, []).append(now)
            elif msg.type in ("note_off", "note_on") and msg.note in active and active[msg.note]:
                start = active[msg.note].pop(0)
                out.append((ti, start, msg.note, now - start))
    return out


def edit_distance(a: list, b: list) -> int:
    prev = list(range(len(b) + 1))
    for i in range(1, len(a) + 1):
        cur = [i] + [0] * len(b)
        for j in range(1, len(b) + 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (a[i - 1] != b[j - 1]))
        prev = cur
    return prev[-1]


def confirmed_environment(gray: np.ndarray, gt: dict) -> Environment:
    """The environment a user would confirm in env_config: clef and key of
    every staff and the time signature, taken from the ground truth. Used to
    score recognition separately from env_detect's clef/key reading."""
    from staff_detect import detect_staves
    staves = detect_staves(gray)["staves"]
    gt_tops = [s["lines"][0] for s in gt["staves"]]
    settings = []
    for i, st in enumerate(staves):
        g = gt["staves"][int(np.argmin([abs(st.top - t) for t in gt_tops]))]
        settings.append(StaffSetting(i, g["system"], g["slot"], g["clef"], g["key"], {"clef": "user", "key": "user"}))
    parts = []
    for slot in range(gt["parts"]):
        clef = next(s["clef"] for s in gt["staves"] if s["slot"] == slot)
        parts.append(Part(f"Part {slot + 1}", slot, "Acoustic Grand Piano", 0, slot, 0, clef, gt["fifths"]))
    return Environment(100.0, "quarter", tuple(gt["time_signature"]), "mf", parts, settings)


def evaluate_page(png: str, out_dir: str, confirmed: bool = False, straighten: bool = True) -> dict:
    gt = json.load(open(png.replace(".png", ".json")))
    gray = cv2.imread(png, cv2.IMREAD_GRAYSCALE)
    nostaff = cv2.imread(png.replace(".png", "_nostaff.png"), cv2.IMREAD_GRAYSCALE)
    u = gt["u"]
    if confirmed:
        res = run_pipeline(gray, env=confirmed_environment(gray, gt), straighten=straighten)
    else:
        res = run_pipeline(gray, time_signature=tuple(gt["time_signature"]), straighten=straighten)
    stem = os.path.join(out_dir, os.path.basename(png).replace(".png", ""))
    midi_path = write_midi(res, stem)

    r: dict = {"page": os.path.basename(png), "mode": gt["mode"], "u": u, "degrade": gt["degrade"],
               "timings": res.timings, "skew_deg": res.staff.get("skew_deg", 0.0)}

    # staves
    det_staves = res.staff["staves"]
    gt_tops = [s["lines"][0] for s in gt["staves"]]
    staff_map = {}
    errs = []
    for di, s in enumerate(det_staves):
        gi = int(np.argmin([abs(s.top - t) for t in gt_tops]))
        if abs(s.top - gt_tops[gi]) < 2 * u:
            staff_map[di] = gi
            errs += list(np.abs(np.asarray(s.lines) - np.asarray(gt["staves"][gi]["lines"])))
    r["staves"] = {"gt": len(gt_tops), "det": len(det_staves), "matched": len(staff_map),
                   "line_err_px": float(np.mean(errs)) if errs else None}

    # staff removal
    r["removal"] = evaluate_removal(res.staff["binary"], res.removal.cleaned, nostaff)

    # clef / key
    clef_ok = key_ok = n = 0
    for s in res.page_env.staves:
        if s.index in staff_map:
            g = gt["staves"][staff_map[s.index]]
            n += 1
            clef_ok += int(s.clef.value is not None and s.clef.value.code == g["clef"])
            key_ok += int(s.key.value == g["key"])
    r["clef_acc"] = clef_ok / n if n else 0.0
    r["key_acc"] = key_ok / n if n else 0.0

    # noteheads
    gt_heads = gt["heads"]
    det_heads = [(h, c) for ss in res.recognition.staves for c in ss.chords for h in c.heads]
    note_by_head = {id(nt.head): nt for nt in res.music.notes}
    pairs = greedy_match([(h["cx"], h["cy"]) for h in gt_heads],
                         [(h.match.cx, h.match.cy) for h, _ in det_heads], HEAD_TOL_U * u)
    r["heads"] = prf(len(pairs), len(gt_heads), len(det_heads))
    kind_ok = pitch_ok = dur_ok = acc_tot = acc_ok = 0
    for gi, dj in pairs:
        g, (h, c) = gt_heads[gi], det_heads[dj]
        hollow_gt = g["kind"] in ("hollow", "whole")
        kind_ok += int(hollow_gt == c.hollow)
        nt = note_by_head.get(id(h))
        pitch_ok += int(nt is not None and nt.midi == g["pitch"])
        dur_ok += int(nt is not None and nt.ticks == g["ticks"])
        if g.get("accidental"):
            acc_tot += 1
            acc_ok += int(h.accidental == g["accidental"])
    m = len(pairs) or 1
    r["head_type_acc"] = kind_ok / m
    r["pitch_acc"] = pitch_ok / m
    r["duration_acc"] = dur_ok / m
    r["accidental_acc"] = (acc_ok / acc_tot) if acc_tot else None
    r["accidental_n"] = acc_tot

    # rests
    gt_rests = gt["rests"]
    det_rests = [rr for ss in res.recognition.staves for rr in ss.rests]
    rp = greedy_match([(x["x"] + x["w"] / 2, x["y"] + x["h"] / 2) for x in gt_rests],
                      [(x.match.cx, x.match.cy) for x in det_rests], REST_TOL_U * u)
    r["rests"] = prf(len(rp), len(gt_rests), len(det_rests))
    r["rest_kind_acc"] = (sum(gt_rests[i]["kind"] == det_rests[j].kind for i, j in rp) / len(rp)) if rp else None

    # measures
    good, total = res.music.measures_ok
    r["measures"] = {"ok": good, "total": total,
                     "gt_total": sum(len(s["barlines"]) for s in gt["staves"])}

    # end to end: MIDI file vs ground truth
    det = midi_notes(midi_path)
    gt_notes = [(h["part"], h["onset"], h["pitch"], h["ticks"]) for h in gt_heads]
    det_set = {}
    for t, o, p, d in det:
        det_set.setdefault((t, o, p), []).append(d)
    tp_on = tp_full = 0
    for part, o, p, d in gt_notes:
        lst = det_set.get((part, o, p))
        if lst:
            tp_on += 1
            if d in lst:
                tp_full += 1
            lst.pop(0)
            if not lst:
                del det_set[(part, o, p)]
    # measure-level: right pitch at the right place inside the right measure.
    # A duration error then costs only its own measure, not the rest of the part.
    mt = int(gt["time_signature"][0] * 4 / gt["time_signature"][1] * 480)
    gt_m = {}
    for h in gt_heads:
        key_ = (h["part"], h["onset"] // mt, h["onset"] % mt, h["pitch"])
        gt_m[key_] = gt_m.get(key_, 0) + 1
    tp_m = 0
    for nt in res.music.notes:
        key_ = (nt.part, nt.measure, nt.offset, nt.midi)
        if gt_m.get(key_, 0) > 0:
            gt_m[key_] -= 1
            tp_m += 1
    r["measure_onset_pitch"] = prf(tp_m, len(gt_heads), len(res.music.notes))
    r["e2e_onset_pitch"] = prf(tp_on, len(gt_notes), len(det))
    r["e2e_with_duration"] = prf(tp_full, len(gt_notes), len(det))
    seq_acc = []
    for part in range(gt["parts"]):
        g_seq = [p for _, o, p, _ in sorted(x for x in gt_notes if x[0] == part)]
        d_seq = [p for _, o, p, _ in sorted(x for x in det if x[0] == part)]
        if g_seq:
            seq_acc.append((len(g_seq), edit_distance(g_seq, d_seq)))
    r["pitch_sequence_acc"] = 1 - sum(e for _, e in seq_acc) / max(1, sum(n for n, _ in seq_acc))
    with open(stem + ".eval.json", "w") as f:
        json.dump(r, f, indent=1, default=float)
    return r


SUMMARY_COLS = [
    ("page", lambda r: r["page"]), ("mode", lambda r: r["mode"]), ("U", lambda r: r["u"]),
    ("deg", lambda r: r["degrade"]),
    ("staves", lambda r: f'{r["staves"]["det"]}/{r["staves"]["gt"]}'),
    ("line_err", lambda r: f'{r["staves"]["line_err_px"]:.2f}'),
    ("rm_F1", lambda r: f'{r["removal"]["f1"]:.3f}'),
    ("clef", lambda r: f'{r["clef_acc"]:.2f}'), ("key", lambda r: f'{r["key_acc"]:.2f}'),
    ("head_P", lambda r: f'{r["heads"]["precision"]:.3f}'), ("head_R", lambda r: f'{r["heads"]["recall"]:.3f}'),
    ("type", lambda r: f'{r["head_type_acc"]:.3f}'), ("pitch", lambda r: f'{r["pitch_acc"]:.3f}'),
    ("dur", lambda r: f'{r["duration_acc"]:.3f}'),
    ("acc", lambda r: "-" if r["accidental_acc"] is None else f'{r["accidental_acc"]:.2f}'),
    ("rest_F1", lambda r: f'{r["rests"]["f1"]:.3f}'),
    ("meas", lambda r: f'{r["measures"]["ok"]}/{r["measures"]["total"]}'),
    ("e2e_F1", lambda r: f'{r["e2e_onset_pitch"]["f1"]:.3f}'),
    ("meas_F1", lambda r: f'{r["measure_onset_pitch"]["f1"]:.3f}'),
    ("seq", lambda r: f'{r["pitch_sequence_acc"]:.3f}'),
]


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("pages")
    p.add_argument("--out", default="eval/results")
    p.add_argument("--confirmed", action="store_true",
                   help="clef/key/time from the ground truth, as a user would confirm them in env_config")
    args = p.parse_args(argv)
    os.makedirs(args.out, exist_ok=True)
    rows = []
    for png in sorted(glob.glob(os.path.join(args.pages, "synth_*.png"))):
        if png.endswith("_nostaff.png"):
            continue
        rows.append(evaluate_page(png, args.out, args.confirmed))
        print("  ".join(f"{name}={fn(rows[-1])}" for name, fn in SUMMARY_COLS))
    with open(os.path.join(args.out, "summary.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([n for n, _ in SUMMARY_COLS])
        for r in rows:
            w.writerow([fn(r) for _, fn in SUMMARY_COLS])
    json.dump(rows, open(os.path.join(args.out, "all.json"), "w"), indent=1, default=float)


if __name__ == "__main__":
    main()
