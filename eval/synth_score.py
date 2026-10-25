# =============================================================================
#  synth_score.py  --  printed test pages with exact ground truth
#  Version: 1.0.0
#
#  Changelog
#  ---------
#  1.0.0  (2026-10-25)
#    - Engraves random but musically valid single-part and piano (2-staff)
#      pages: clefs, key and time signatures, notes from whole to sixteenth,
#      dots, beams, flags, chords, ledger lines, accidentals, rests, barlines.
#    - Glyphs come from the FreeSerif font (GNU FreeFont), whose Unicode
#      "Musical Symbols" block supplies clefs, noteheads, rests and
#      accidentals. Stems, beams, flags, ledger lines and barlines are drawn.
#    - Per-glyph jitter (scale, rotation) and page degradations (blur, noise,
#      rotation, contrast) so templates never see their exact source pixels.
#    - Writes, per page: the image, the same page without staff lines (for
#      pixel-level staff-removal scoring), and a JSON ground truth with staff
#      geometry, clef/key per staff, every notehead (box, step, pitch, onset,
#      duration) and every rest.
#
#  Usage:  python eval/synth_score.py --out eval/pages --seeds 0-11
# =============================================================================

from __future__ import annotations

import argparse
import json
import math
import os
import random
from dataclasses import dataclass, field

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

FONT = "/usr/share/fonts/truetype/freefont/FreeSerif.ttf"
FONT_BOLD = "/usr/share/fonts/truetype/freefont/FreeSerifBold.ttf"
SS = 2                      # supersampling factor
TPB = 480                   # ticks per quarter note (matches events.TICKS_PER_BEAT)

GLYPH = {"gclef": 0x1D11E, "fclef": 0x1D122, "black": 0x1D158, "void": 0x1D157, "whole": 0x1D15D,
         "r_q": 0x1D13D, "r_8": 0x1D13E, "r_16": 0x1D13F,
         "sharp": 0x266F, "flat": 0x266D, "natural": 0x266E}
# glyph heights in staff spaces (U)
HEIGHT_U = {"gclef": 7.0, "fclef": 3.3, "black": 1.0, "void": 1.0, "whole": 1.0,
            "r_q": 2.9, "r_8": 1.9, "r_16": 2.8, "sharp": 2.7, "flat": 2.3, "natural": 2.7}

LETTERS = "CDEFGAB"
SHARP_ORDER, FLAT_ORDER = "FCGDAEB", "BEADGCF"
KEY_POSITIONS = {   # same convention as env_detect.KEY_POSITIONS
    "G2": {"sharp": [0, 3, -1, 2, 5, 1, 4], "flat": [4, 1, 5, 2, 6, 3, 7]},
    "F4": {"sharp": [2, 5, 1, 4, 7, 3, 6], "flat": [6, 3, 7, 4, 8, 5, 9]},
}
CLEF_REF = {"G2": ("G", 4, 6), "F4": ("F", 3, 2)}   # letter, octave, step of that line
PC = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}


def step_to_pitch(clef: str, step: int) -> tuple[str, int]:
    letter, octave, ref = CLEF_REF[clef]
    index = LETTERS.index(letter) + octave * 7 + (ref - step)
    return LETTERS[index % 7], index // 7


def midi_number(letter: str, octave: int, alter: int) -> int:
    return 12 * (octave + 1) + PC[letter] + alter


# --------------------------------------------------------------------------- #
#  Glyph rendering
# --------------------------------------------------------------------------- #
_glyph_cache: dict = {}


def glyph_mask(name: str, height: float, angle: float = 0.0) -> np.ndarray:
    """Tight float mask (1 = ink) of a glyph scaled to `height` pixels."""
    key = (name, round(height, 1), round(angle, 1))
    if key in _glyph_cache:
        return _glyph_cache[key]
    font = ImageFont.truetype(FONT, 400)
    im = Image.new("L", (900, 900), 0)
    d = ImageDraw.Draw(im)
    d.text((200, 100), chr(GLYPH[name]), font=font, fill=255)
    a = np.asarray(im)
    ys, xs = np.nonzero(a > 0)
    a = a[ys.min():ys.max() + 1, xs.min():xs.max() + 1].astype(np.float32) / 255
    scale = height / a.shape[0]
    a = cv2.resize(a, (max(1, round(a.shape[1] * scale)), max(1, round(height))), interpolation=cv2.INTER_AREA)
    if angle:
        h, w = a.shape
        pad = int(0.15 * max(h, w)) + 2
        a = np.pad(a, pad)
        m = cv2.getRotationMatrix2D((a.shape[1] / 2, a.shape[0] / 2), angle, 1.0)
        a = cv2.warpAffine(a, m, (a.shape[1], a.shape[0]), flags=cv2.INTER_LINEAR)
        ys, xs = np.nonzero(a > 0.05)
        a = a[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    _glyph_cache[key] = a
    return a


def digit_mask(ch: str, height: float) -> np.ndarray:
    font = ImageFont.truetype(FONT_BOLD, 400)
    im = Image.new("L", (500, 600), 0)
    ImageDraw.Draw(im).text((50, 50), ch, font=font, fill=255)
    a = np.asarray(im)
    ys, xs = np.nonzero(a > 0)
    a = a[ys.min():ys.max() + 1, xs.min():xs.max() + 1].astype(np.float32) / 255
    s = height / a.shape[0]
    return cv2.resize(a, (max(1, round(a.shape[1] * s)), max(1, round(height))), interpolation=cv2.INTER_AREA)


# --------------------------------------------------------------------------- #
#  Canvas: two layers, so the page can be written with and without staff lines
# --------------------------------------------------------------------------- #
class Canvas:
    def __init__(self, width: int, height: int):
        self.symbols = np.zeros((height * SS, width * SS), np.float32)
        self.staff = np.zeros_like(self.symbols)

    def paste(self, mask: np.ndarray, x: float, y: float, layer: str = "symbols"):
        """Paste `mask` with its top-left corner at page coords (x, y)."""
        target = self.symbols if layer == "symbols" else self.staff
        x, y = int(round(x * SS)), int(round(y * SS))
        h, w = mask.shape
        y0, x0 = max(0, y), max(0, x)
        y1, x1 = min(target.shape[0], y + h), min(target.shape[1], x + w)
        if y1 <= y0 or x1 <= x0:
            return
        sub = mask[y0 - y:y1 - y, x0 - x:x1 - x]
        np.maximum(target[y0:y1, x0:x1], sub, out=target[y0:y1, x0:x1])

    def polygon(self, pts, layer: str = "symbols"):
        target = self.symbols if layer == "symbols" else self.staff
        p = np.round(np.asarray(pts, np.float64) * SS * 4).astype(np.int32)
        cv2.fillPoly(target, [p], 1.0, lineType=cv2.LINE_AA, shift=2)

    def rect(self, x0, y0, x1, y1, layer: str = "symbols"):
        self.polygon([(x0, y0), (x1, y0), (x1, y1), (x0, y1)], layer)

    def thick_line(self, x0, y0, x1, y1, width, layer="symbols"):
        dx, dy = x1 - x0, y1 - y0
        n = math.hypot(dx, dy) or 1.0
        ox, oy = -dy / n * width / 2, dx / n * width / 2
        self.polygon([(x0 + ox, y0 + oy), (x1 + ox, y1 + oy), (x1 - ox, y1 - oy), (x0 - ox, y0 - oy)], layer)

    def disc(self, cx, cy, r):
        cv2.circle(self.symbols, (int(round(cx * SS * 4)), int(round(cy * SS * 4))), int(round(r * SS * 4)),
                   1.0, -1, lineType=cv2.LINE_AA, shift=2)

    def render(self, with_staff: bool = True) -> np.ndarray:
        ink = np.maximum(self.symbols, self.staff) if with_staff else self.symbols
        small = cv2.resize(ink, (ink.shape[1] // SS, ink.shape[0] // SS), interpolation=cv2.INTER_AREA)
        return np.clip(255 * (1 - small), 0, 255).astype(np.uint8)


# --------------------------------------------------------------------------- #
#  Music content
# --------------------------------------------------------------------------- #
@dataclass
class Item:
    kind: str                     # "note" | "rest" | "mrest"
    ticks: int                    # duration
    steps: list = field(default_factory=list)   # notehead steps (chords: several)
    dots: int = 0
    acc: dict = field(default_factory=dict)     # step -> "sharp" | "flat" | "natural"
    beam_group: int = -1


BASE = {1920: "whole", 960: "half", 480: "quarter", 240: "eighth", 120: "sixteenth"}


def random_measure(rng: random.Random, beats: int, chord_p: float) -> list[Item]:
    """One measure as a list of items. Beats are quarter notes."""
    items: list[Item] = []
    left = beats
    if rng.random() < 0.06:
        return [Item("mrest", beats * TPB)]
    if beats == 4 and rng.random() < 0.07:
        return [Item("note", 1920, [0])]
    while left > 0:
        options = ["q", "q", "q", "qr", "8pair", "8pair", "16four", "8r8"]
        if left >= 2:
            options += ["h", "h", "hr", "dq8"]
        if left >= 3:
            options += ["dh"]
        choice = rng.choice(options)
        if choice == "q":
            items.append(Item("note", 480, [0])); left -= 1
        elif choice == "qr":
            items.append(Item("rest", 480)); left -= 1
        elif choice == "8pair":
            g = rng.randrange(1 << 30)
            items += [Item("note", 240, [0], beam_group=g), Item("note", 240, [0], beam_group=g)]; left -= 1
        elif choice == "16four":
            g = rng.randrange(1 << 30)
            items += [Item("note", 120, [0], beam_group=g) for _ in range(4)]; left -= 1
        elif choice == "8r8":
            items += [Item("rest", 240), Item("note", 240, [0])]; left -= 1
        elif choice == "h":
            items.append(Item("note", 960, [0])); left -= 2
        elif choice == "hr":
            items.append(Item("rest", 960)); left -= 2
        elif choice == "dq8":
            items += [Item("note", 720, [0], dots=1), Item("note", 240, [0])]; left -= 2
        elif choice == "dh":
            items.append(Item("note", 1440, [0], dots=1)); left -= 3
    for it in items:
        if it.kind == "note" and it.beam_group < 0 and rng.random() < chord_p:
            it.steps = [0, 0]      # filled in by pitch assignment
    return items


def assign_pitches(rng: random.Random, items: list[Item], fifths: int, clef: str, acc_p: float):
    key_letters = set(SHARP_ORDER[:fifths]) if fifths > 0 else set(FLAT_ORDER[:-fifths])
    prev = rng.randint(1, 7)
    for it in items:
        if it.kind != "note":
            continue
        step = int(np.clip(prev + rng.choice([-3, -2, -1, -1, 1, 1, 2, 3, 0]), -4, 12))
        if it.beam_group < 0 and rng.random() < 0.08:      # big leaps only outside beamed groups
            step = rng.choice([-4, -3, -2, 10, 11, 12])
        prev = step
        if len(it.steps) == 2:
            span = rng.choice([[0, 2], [0, 4], [0, 2, 4], [0, 3], [0, 5]])
            base = int(np.clip(step, 0, 12 - max(span)))
            it.steps = [base + s for s in span]
        else:
            it.steps = [step]
        for s in it.steps:
            if rng.random() < acc_p:
                letter = step_to_pitch(clef, s)[0]
                if letter in key_letters:
                    it.acc[s] = "natural"
                else:
                    it.acc[s] = rng.choice(["sharp", "flat"])


# --------------------------------------------------------------------------- #
#  Engraving one staff
# --------------------------------------------------------------------------- #
class Engraver:
    def __init__(self, canvas: Canvas, rng: random.Random, u: float, t: float, jitter: float):
        self.c, self.rng, self.u, self.t, self.jitter = canvas, rng, u, t, jitter
        self.stem_w = max(1.2, 0.11 * u)
        self.heads: list[dict] = []
        self.rests: list[dict] = []

    def y(self, top: float, step: float) -> float:
        return top + step * self.u / 2

    def _jit(self) -> tuple[float, float]:
        j = self.jitter
        return 1 + self.rng.uniform(-j, j), self.rng.uniform(-25 * j, 25 * j)

    def glyph(self, name: str, cx: float, cy: float, *, height_u: float | None = None,
              anchor: str = "center", jitter: bool = True) -> tuple[float, float, float, float]:
        s, a = self._jit() if jitter else (1.0, 0.0)
        h = (height_u or HEIGHT_U[name]) * self.u * s * SS
        m = glyph_mask(name, h, a)
        mh, mw = m.shape[0] / SS, m.shape[1] / SS
        if anchor == "center":
            x0, y0 = cx - mw / 2, cy - mh / 2
        elif anchor == "flat":           # loop centre at (cx, cy): loop is the lower ~45 %
            x0, y0 = cx - mw / 2, cy - mh + 0.5 * self.u
        elif anchor == "bottom":
            x0, y0 = cx - mw / 2, cy - mh
        else:
            raise ValueError(anchor)
        self.c.paste(m, x0, y0)
        return x0, y0, mw, mh

    def staff_lines(self, top: float, x0: float, x1: float):
        for k in range(5):
            yy = top + k * self.u
            self.c.rect(x0, yy - self.t / 2, x1, yy + self.t / 2, layer="staff")

    def draw_clef(self, clef: str, top: float, x: float) -> float:
        name = "gclef" if clef == "G2" else "fclef"
        h = HEIGHT_U[name] * self.u * SS
        m = glyph_mask(name, h)
        mh, mw = m.shape[0] / SS, m.shape[1] / SS
        if clef == "G2":
            y0 = self.y(top, 11) - mh            # bottom 1.5U below the staff
        else:
            y0 = self.y(top, -0.1)               # top on the top line
        self.c.paste(m, x, y0)
        return x + mw

    def key_signature(self, clef: str, fifths: int, top: float, x: float) -> float:
        if fifths == 0:
            return x
        kind = "sharp" if fifths > 0 else "flat"
        for i in range(abs(fifths)):
            step = KEY_POSITIONS[clef][kind][i]
            cx = x + 0.6 * self.u
            x0, _, w, _ = self.glyph(kind, cx, self.y(top, step), anchor="flat" if kind == "flat" else "center",
                                     jitter=False)
            x = x0 + w + 0.15 * self.u
        return x

    def time_signature(self, beats: int, beat_type: int, top: float, x: float) -> float:
        h = 2.0 * self.u
        up, lo = digit_mask(str(beats), h * SS), digit_mask(str(beat_type), h * SS)
        w = max(up.shape[1], lo.shape[1]) / SS
        self.c.paste(up, x + (w - up.shape[1] / SS) / 2, top)
        self.c.paste(lo, x + (w - lo.shape[1] / SS) / 2, top + 2 * self.u)
        return x + w

    def ledger(self, top: float, cx: float, step: int, half_width: float):
        rng = []
        if step <= -2:
            rng = range(-2, step - 1, -2)
        elif step >= 10:
            rng = range(10, step + 1, 2)
        for s in rng:
            yy = self.y(top, s)
            self.c.rect(cx - half_width, yy - self.t / 2, cx + half_width, yy + self.t / 2)

    def notehead(self, kind: str, cx: float, top: float, step: int) -> dict:
        name = {"filled": "black", "hollow": "void", "whole": "whole"}[kind]
        cy = self.y(top, step)
        x0, y0, w, h = self.glyph(name, cx, cy)
        self.ledger(top, cx, step, 0.5 * w + 0.35 * self.u)
        return {"x": x0, "y": y0, "w": w, "h": h, "cx": cx, "cy": cy, "step": step, "kind": kind}

    def flag(self, x: float, y: float, up: bool, count: int):
        u = self.u
        for k in range(count):
            off = k * 0.75 * u * (1 if up else -1)
            sgn = 1 if up else -1
            pts_top = [(x, y + off), (x + 0.35 * u, y + off + sgn * 0.7 * u), (x + 0.95 * u, y + off + sgn * 1.5 * u),
                       (x + 0.75 * u, y + off + sgn * 2.4 * u)]
            pts_bot = [(x + 0.65 * u, y + off + sgn * 2.2 * u), (x + 0.7 * u, y + off + sgn * 1.6 * u),
                       (x + 0.25 * u, y + off + sgn * 1.05 * u), (x, y + off + sgn * 0.65 * u)]
            self.c.polygon(pts_top + pts_bot)

    def dot(self, x: float, cy: float, step: int, top: float):
        yy = cy if step % 2 else self.y(top, step - 1)   # on a line: dot moves to the space above
        self.c.disc(x, yy, 0.2 * self.u)

    def rest(self, kind: str, cx: float, top: float, measure_ticks: int, ticks: int) -> dict:
        u = self.u
        if kind in ("whole", "half"):
            w, h = 1.15 * u, 0.5 * u
            if kind == "whole":
                y0 = self.y(top, 2) - self.t / 2
            else:
                y0 = self.y(top, 4) - h + self.t / 2
            self.c.rect(cx - w / 2, y0, cx + w / 2, y0 + h)
            box = (cx - w / 2, y0, w, h)
        else:
            name = {"quarter": "r_q", "eighth": "r_8", "sixteenth": "r_16"}[kind]
            cy = self.y(top, 4) if kind == "quarter" else self.y(top, 4.5)
            box = self.glyph(name, cx, cy)
        return {"x": box[0], "y": box[1], "w": box[2], "h": box[3], "kind": kind, "ticks": ticks}


def ticks_kind(ticks: int, dots: int) -> str:
    base = int(ticks / 1.5) if dots else ticks
    return BASE[base]


# --------------------------------------------------------------------------- #
#  Page
# --------------------------------------------------------------------------- #
def engrave_page(seed: int, mode: str | None = None, u: float | None = None, degrade: int | None = None,
                 rotation: float | None = None) -> tuple[np.ndarray, np.ndarray, dict]:
    rng = random.Random(seed)
    mode = mode or ("piano" if seed % 3 == 2 else "single")
    u = u or rng.choice([18, 20, 22, 24, 26, 28])
    t = max(1.6, round(0.1 * u, 1))
    degrade = rng.choice([0, 1, 2]) if degrade is None else degrade
    jitter = 0.06 if degrade == 0 else 0.10
    width = 2400
    margin = 3.0 * u
    beats, beat_type = rng.choice([(4, 4), (4, 4), (3, 4)])
    fifths = rng.choice([-3, -2, -1, 0, 0, 1, 2, 3])
    measure_ticks = beats * TPB
    single_clef = rng.choice(["G2", "G2", "F4"])
    n_systems = 6 if mode == "single" else 4
    staff_gap = 9.0 * u if mode == "single" else 7.5 * u     # top-to-top inside a piano system
    system_gap = 11.0 * u
    clefs_per_system = [single_clef] if mode == "single" else ["G2", "F4"]
    height = int(margin * 2 + n_systems * (len(clefs_per_system) * staff_gap) + n_systems * 2 * u + 8 * u)
    canvas = Canvas(width, height)
    eng = Engraver(canvas, rng, u, t, jitter)

    gt = {"seed": seed, "mode": mode, "u": u, "t": t, "degrade": degrade, "time_signature": [beats, beat_type],
          "fifths": fifths, "staves": [], "heads": [], "rests": [], "parts": len(clefs_per_system),
          "barlines": []}
    part_clock = [0] * len(clefs_per_system)
    y_cursor = margin + 3 * u
    staff_index = 0
    for system in range(n_systems):
        tops = [y_cursor + i * staff_gap for i in range(len(clefs_per_system))]
        # music: same measures count for both staves of a system
        n_measures = rng.choice([3, 4]) if beats == 4 else rng.choice([4, 5])
        content = []
        for slot, clef in enumerate(clefs_per_system):
            ms = []
            for _ in range(n_measures):
                m = random_measure(rng, beats, chord_p=0.18 if mode == "piano" else 0.08)
                assign_pitches(rng, m, fifths, clef, acc_p=0.10)
                ms.append(m)
            content.append(ms)

        x_start = margin
        # header
        header_end = []
        for slot, clef in enumerate(clefs_per_system):
            x = eng.draw_clef(clef, tops[slot], x_start + 0.6 * u) + 0.6 * u
            x = eng.key_signature(clef, fifths, tops[slot], x) + 0.5 * u
            if system == 0:
                x = eng.time_signature(beats, beat_type, tops[slot], x) + 0.4 * u
            header_end.append(x)
        x = max(header_end) + 1.4 * u

        # shared x positions per measure: lay out the union of onsets across slots
        barline_xs = []
        for mi in range(n_measures):
            onsets = sorted({o for slot in range(len(clefs_per_system))
                             for o in _onsets(content[slot][mi])})
            acc_extra = {o: any(it.acc for slot in range(len(clefs_per_system))
                                for oo, it in _timed(content[slot][mi]) if oo == o) for o in onsets}
            xpos = {}
            for k, o in enumerate(onsets):
                if acc_extra[o]:
                    x += 1.3 * u
                xpos[o] = x
                nxt = onsets[k + 1] if k + 1 < len(onsets) else measure_ticks
                dur = nxt - o
                x += u * max(1.9, 2.3 + 1.5 * math.log2(dur / 240))
            x += 0.3 * u
            for slot, clef in enumerate(clefs_per_system):
                top = tops[slot]
                _engrave_measure(eng, content[slot][mi], xpos, top, clef, fifths, measure_ticks,
                                 x_bar_left=(xpos[onsets[0]] - 1.2 * u), x_bar_right=x,
                                 clock=part_clock[slot], part=slot, staff=staff_index + slot, gt=gt)
                part_clock[slot] += measure_ticks
            barline_xs.append(x)
            x += 1.6 * u
        x_end = barline_xs[-1]
        # barlines (spanning the whole system in piano mode), final double bar on the last system
        y_top, y_bot = tops[0], tops[-1] + 4 * u
        bw = max(1.6, 0.16 * u)
        for k, bx in enumerate(barline_xs):
            canvas.rect(bx - bw / 2, y_top - t / 2, bx + bw / 2, y_bot + t / 2)
            if system == n_systems - 1 and k == len(barline_xs) - 1:
                canvas.rect(bx - bw / 2 - 0.55 * u - 0.5 * u, y_top - t / 2, bx - bw / 2 - 0.55 * u, y_bot + t / 2)
        if mode == "piano":   # system start line
            canvas.rect(margin - bw / 2, y_top - t / 2, margin + bw / 2, y_bot + t / 2)
        for slot, clef in enumerate(clefs_per_system):
            eng.staff_lines(tops[slot], margin, x_end + bw / 2)
            gt["staves"].append({"index": staff_index + slot, "system": system, "slot": slot,
                                 "lines": [tops[slot] + k * u for k in range(5)], "clef": clef, "key": fifths,
                                 "time_present": system == 0, "x0": margin, "x1": x_end,
                                 "barlines": barline_xs})
        staff_index += len(clefs_per_system)
        y_cursor = tops[-1] + staff_gap + (system_gap - staff_gap if mode == "piano" else 0)

    gt["heads"] = eng.heads
    gt["rests"] = eng.rests
    gt["end_ticks"] = part_clock
    image = canvas.render(True)
    nostaff = canvas.render(False)
    image = degrade_page(image, degrade, rng, rotation)
    if rotation:
        nostaff = rotate(nostaff, rotation)
    return image, nostaff, gt


def _timed(items: list[Item]):
    clock = 0
    for it in items:
        yield clock, it
        clock += it.ticks


def _onsets(items: list[Item]):
    return [o for o, _ in _timed(items)]


def _engrave_measure(eng: Engraver, items: list[Item], xpos: dict, top: float, clef: str, fifths: int,
                     measure_ticks: int, x_bar_left: float, x_bar_right: float, clock: int, part: int,
                     staff: int, gt: dict):
    u = eng.u
    key_letters = {l: 1 for l in SHARP_ORDER[:fifths]} if fifths > 0 else {l: -1 for l in FLAT_ORDER[:-fifths]}
    measure_acc: dict[tuple[str, int], int] = {}
    # beam groups
    groups: dict[int, list] = {}
    for o, it in _timed(items):
        if it.kind == "note" and it.beam_group >= 0:
            groups.setdefault(it.beam_group, []).append((o, it))
    stem_info = {}   # id(item) -> (stem_x, stem_end_y, up)

    for g, members in groups.items():
        mean_step = np.mean([s for _, it in members for s in it.steps])
        up = mean_step >= 4
        xs = [xpos[o] for o, _ in members]
        steps = [it.steps[0] for _, it in members]
        head_w = 1.3 * u
        stem_x = [x + (head_w / 2 - eng.stem_w / 2) if up else x - (head_w / 2 - eng.stem_w / 2) for x in xs]
        ys = [eng.y(top, s) for s in steps]
        if up:
            slope = np.clip((ys[-1] - ys[0]) * 0.25, -0.6 * u, 0.6 * u)
            ends =[min(ys) - 3.3 * u + slope * (sx - stem_x[0]) / max(stem_x[-1] - stem_x[0], 1) for sx in stem_x]
            shift = min(0, min(y_ - 2.6 * u - e for y_, e in zip(ys, ends)))
            ends = [e + shift for e in ends]
        else:
            slope = np.clip((ys[-1] - ys[0]) * 0.25, -0.6 * u, 0.6 * u)
            ends = [max(ys) + 3.3 * u + slope * (sx - stem_x[0]) / max(stem_x[-1] - stem_x[0], 1) for sx in stem_x]
            shift = max(0, max(y_ + 2.6 * u - e for y_, e in zip(ys, ends)))
            ends = [e + shift for e in ends]
        n_beams = 1 if members[0][1].ticks == 240 else 2
        bt = 0.48 * u
        for b in range(n_beams):
            off = b * 0.75 * u * (1 if up else -1)
            x0, x1 = stem_x[0] - eng.stem_w / 2, stem_x[-1] + eng.stem_w / 2
            y0, y1 = ends[0] + off, ends[-1] + off
            if up:
                eng.c.polygon([(x0, y0), (x1, y1), (x1, y1 + bt), (x0, y0 + bt)])
            else:
                eng.c.polygon([(x0, y0 - bt), (x1, y1 - bt), (x1, y1), (x0, y0)])
        for (o, it), sx, e in zip(members, stem_x, ends):
            stem_info[id(it)] = (sx, e, up)

    for o, it in _timed(items):
        x = xpos[o]
        if it.kind in ("rest", "mrest"):
            if it.kind == "mrest":
                kind, ticks = "whole", measure_ticks
                cx = (x_bar_left + x_bar_right) / 2
            else:
                kind, ticks = BASE[it.ticks], it.ticks
                cx = x
            r = eng.rest(kind, cx, top, measure_ticks, ticks)
            r.update({"part": part, "staff": staff, "onset": clock + o})
            eng.rests.append(r)
            continue
        base_kind = ticks_kind(it.ticks, it.dots)
        head_kind = "whole" if base_kind == "whole" else "hollow" if base_kind == "half" else "filled"
        heads = []
        for s in sorted(it.steps):
            if s in it.acc:
                name = it.acc[s]
                acx = x - 1.45 * u
                eng.glyph(name, acx, eng.y(top, s), anchor="flat" if name == "flat" else "center")
            heads.append(eng.notehead(head_kind, x, top, s))
        # stems
        if base_kind != "whole":
            if id(it) in stem_info:
                sx, end, up = stem_info[id(it)]
                ys = [h["cy"] for h in heads]
                y_from = max(ys) if up else min(ys)
                eng.c.rect(sx - eng.stem_w / 2, min(y_from, end), sx + eng.stem_w / 2, max(y_from, end))
            else:
                mean_step = np.mean(it.steps)
                up = mean_step >= 4
                hw = max(h["w"] for h in heads) / 2 - eng.stem_w / 2
                sx = x + hw if up else x - hw
                ys = [h["cy"] for h in heads]
                if up:
                    end = min(min(ys) - 3.4 * u, eng.y(top, 4))
                    eng.c.rect(sx - eng.stem_w / 2, end, sx + eng.stem_w / 2, max(ys))
                else:
                    end = max(max(ys) + 3.4 * u, eng.y(top, 4))
                    eng.c.rect(sx - eng.stem_w / 2, min(ys), sx + eng.stem_w / 2, end)
                if base_kind in ("eighth", "sixteenth"):
                    eng.flag(sx + eng.stem_w / 2 - 0.5, end, up, 1 if base_kind == "eighth" else 2)
        if it.dots:
            right = max(h["x"] + h["w"] for h in heads)
            for h in heads:
                eng.dot(right + 0.45 * u, h["cy"], h["step"], top)
        for h in heads:
            letter, octave = step_to_pitch(clef, h["step"])
            if h["step"] in it.acc:
                alter = {"sharp": 1, "flat": -1, "natural": 0}[it.acc[h["step"]]]
                measure_acc[(letter, octave)] = alter
            else:
                alter = measure_acc.get((letter, octave), key_letters.get(letter, 0))
            h.update({"part": part, "staff": staff, "onset": clock + o, "ticks": it.ticks,
                      "pitch": midi_number(letter, octave, alter), "name": f"{letter}{octave}",
                      "alter": alter, "accidental": it.acc.get(h["step"])})
            eng.heads.append(h)


# --------------------------------------------------------------------------- #
#  Degradation
# --------------------------------------------------------------------------- #
def rotate(img: np.ndarray, degrees: float) -> np.ndarray:
    h, w = img.shape
    m = cv2.getRotationMatrix2D((w / 2, h / 2), degrees, 1.0)
    return cv2.warpAffine(img, m, (w, h), flags=cv2.INTER_LINEAR, borderValue=255)


def degrade_page(img: np.ndarray, level: int, rng: random.Random, rotation: float | None) -> np.ndarray:
    out = img.astype(np.float32)
    nrng = np.random.default_rng(rng.randrange(1 << 30))
    if level >= 1:
        out = cv2.GaussianBlur(out, (0, 0), 0.6 if level == 1 else 1.0)
        out = 40 + out * (215 / 255)                      # grey paper, softer ink
        out += nrng.normal(0, 6 if level == 1 else 12, out.shape)
    if level >= 2:
        sp = nrng.random(out.shape)
        out[sp < 0.002] = 0
        out[sp > 0.998] = 255
    out = np.clip(out, 0, 255).astype(np.uint8)
    if rotation:
        out = rotate(out, rotation)
    return out


def parse_seeds(text: str) -> list[int]:
    out = []
    for part in text.split(","):
        if "-" in part:
            a, b = part.split("-")
            out += list(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description="Engrave printed test pages with ground truth")
    p.add_argument("--out", default="eval/pages")
    p.add_argument("--seeds", default="0-11")
    p.add_argument("--mode", choices=["single", "piano"])
    p.add_argument("--u", type=float)
    p.add_argument("--degrade", type=int, choices=[0, 1, 2])
    p.add_argument("--rotation", type=float)
    args = p.parse_args(argv)
    os.makedirs(args.out, exist_ok=True)
    for seed in parse_seeds(args.seeds):
        img, nostaff, gt = engrave_page(seed, args.mode, args.u, args.degrade, args.rotation)
        stem = os.path.join(args.out, f"synth_{seed:03d}")
        cv2.imwrite(stem + ".png", img)
        cv2.imwrite(stem + "_nostaff.png", nostaff)
        with open(stem + ".json", "w") as f:
            json.dump(gt, f, indent=1, default=float)
        print(f"{stem}.png  mode={gt['mode']} U={gt['u']} degrade={gt['degrade']} "
              f"staves={len(gt['staves'])} heads={len(gt['heads'])} rests={len(gt['rests'])}")


if __name__ == "__main__":
    main()
