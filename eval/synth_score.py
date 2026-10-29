# =============================================================================
#  synth_score.py  --  printed test pages with exact ground truth
#  Version: 1.2.1
#
#  Changelog
#  ---------
#  1.2.1  (2026-10-29)
#    - Alto clef (C3) for scored pieces with a SMuFL font;
#      eval/scores/megalovania_excerpt.json.
#  1.2.0  (2026-10-28)
#    - --music-font: engrave with a SMuFL music font (e.g. Leland, the
#      MuseScore 4 font, or Bravura; both SIL OFL, from GitHub) at its natural
#      size, glyphs placed by their SMuFL origin. The default stays FreeSerif.
#    - --score: engrave a fixed piece from a JSON file (eval/scores/), with
#      chord symbols above the staff, a title, composer and part names, as on
#      a MuseScore export. eval/scores/happy_birthday.json is the piano
#      arrangement from the user's screenshot.
#    - Chords containing a second put one head on the other side of the stem,
#      with extra space before the chord when a head moves left.
#    - Scores may use small staves ("size"), staff groups for barlines,
#      pickup measures, whole-measure rests ("r:m"), dynamics, 8va lines,
#      triplets ("8t", "16t") and systems with hidden staves ("hide").
#    - Random pages are unchanged by these additions.
#  1.1.0  (2026-10-25)
#    - --extended pages carry the features a real printed piano score showed
#      the pipeline could not handle: compound time (6/8, 12/8, beamed in
#      dotted-quarter groups), keys up to five sharps or flats, a tempo mark
#      (note = number) above the first staff, ties between repeated notes,
#      and mid-staff clef changes in the left hand (change-size G clef, some
#      with an octave 8 below, then back to the F clef). The ground truth
#      records the sounding duration of tied notes and every clef change.
#    - FIX: a system whose measures did not fit was drawn past the right edge
#      of the page (notes outside the image but in the ground truth). The
#      spacing is now compressed, and a measure dropped if that is not
#      enough. Pages that fitted are unchanged.
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
#  Usage:  python eval/synth_score.py --out eval/pages --seeds 0-11 [--extended]
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
    "C3": {"sharp": [1, 4, 0, 3, 6, 2, 5], "flat": [5, 2, 6, 3, 7, 4, 8]},
}
CLEF_REF = {"G2": ("G", 4, 6), "F4": ("F", 3, 2), "C3": ("C", 4, 4)}   # letter, octave, step of that line
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
    im = Image.new("L", (200 + 300 * len(ch), 600), 0)
    ImageDraw.Draw(im).text((50, 50), ch, font=font, fill=255)
    a = np.asarray(im)
    ys, xs = np.nonzero(a > 0)
    a = a[ys.min():ys.max() + 1, xs.min():xs.max() + 1].astype(np.float32) / 255
    s = height / a.shape[0]
    return cv2.resize(a, (max(1, round(a.shape[1] * s)), max(1, round(height))), interpolation=cv2.INTER_AREA)


# --------------------------------------------------------------------------- #
#  SMuFL music fonts (Leland, Bravura, ...): glyphs at their natural size
# --------------------------------------------------------------------------- #
SMUFL = {"gclef": 0xE050, "fclef": 0xE062, "cclef": 0xE05C, "black": 0xE0A4, "void": 0xE0A3, "whole": 0xE0A2,
         "r_q": 0xE4E5, "r_8": 0xE4E6, "r_16": 0xE4E7, "sharp": 0xE262, "flat": 0xE260, "natural": 0xE261}
SMUFL_DIGIT0 = 0xE080
MUSIC_FONT: str | None = None          # path of a SMuFL font; None = FreeSerif glyphs
TEXT_FONT = FONT                       # chord symbols, titles
_smufl_cache: dict = {}


def smufl_mask(code_point: int, em_px: float, angle: float = 0.0) -> tuple[np.ndarray, float, float]:
    """A SMuFL glyph rendered with 1 em = em_px (4 staff spaces). Returns the
    float mask and the position (x, y) of the glyph origin inside it."""
    key = (code_point, round(em_px, 1), round(angle, 1))
    if key in _smufl_cache:
        return _smufl_cache[key]
    font = ImageFont.truetype(MUSIC_FONT, max(4, int(round(em_px))))
    l, t, r, b = font.getbbox(chr(code_point), anchor="ls")
    pad = 4
    im = Image.new("L", (r - l + 2 * pad, b - t + 2 * pad), 0)
    ImageDraw.Draw(im).text((pad - l, pad - t), chr(code_point), font=font, fill=255, anchor="ls")
    a = np.asarray(im).astype(np.float32) / 255
    ox, oy = pad - l, pad - t
    if angle:
        h, w = a.shape
        m = cv2.getRotationMatrix2D((ox, oy), angle, 1.0)
        a = cv2.warpAffine(a, m, (w, h), flags=cv2.INTER_LINEAR)
    _smufl_cache[key] = (a, float(ox), float(oy))
    return _smufl_cache[key]


FONT_BOLD_ITALIC = "/usr/share/fonts/truetype/freefont/FreeSerifBoldItalic.ttf"


def text_mask(text: str, height_px: float, bold: bool = False, italic: bool = False) -> np.ndarray:
    font = ImageFont.truetype(FONT_BOLD_ITALIC if italic else FONT_BOLD if bold else TEXT_FONT, 200)
    l, t, r, b = font.getbbox(text)
    im = Image.new("L", (r - l + 20, b - t + 20), 0)
    ImageDraw.Draw(im).text((10 - l, 10 - t), text, font=font, fill=255)
    a = np.asarray(im).astype(np.float32) / 255
    ys, xs = np.nonzero(a > 0)
    a = a[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    s = height_px / 200 * 1.0
    return cv2.resize(a, (max(1, round(a.shape[1] * s)), max(1, round(a.shape[0] * s))), interpolation=cv2.INTER_AREA)


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
    tie_next: bool = False        # tied to the next item (same pitch)


BASE = {1920: "whole", 960: "half", 480: "quarter", 240: "eighth", 120: "sixteenth",
        160: "eighth", 80: "sixteenth"}            # 160 / 80: triplet eighths / sixteenths


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


def random_compound_measure(rng: random.Random, groups: int, chord_p: float) -> list[Item]:
    """One 6/8 or 12/8 measure: `groups` dotted-quarter beats."""
    if rng.random() < 0.05:
        return [Item("mrest", groups * 720)]
    items: list[Item] = []
    left = groups
    while left > 0:
        options = ["3e", "3e", "3e", "dq", "q8", "qr8"]
        if left >= 2:
            options += ["dh"]
        choice = rng.choice(options)
        if choice == "3e":
            g = rng.randrange(1 << 30)
            items += [Item("note", 240, [0], beam_group=g) for _ in range(3)]; left -= 1
        elif choice == "dq":
            items.append(Item("note", 720, [0], dots=1)); left -= 1
        elif choice == "q8":
            items += [Item("note", 480, [0]), Item("note", 240, [0])]; left -= 1
        elif choice == "qr8":
            items += [Item("rest", 480), Item("note", 240, [0])]; left -= 1
        elif choice == "dh":
            items.append(Item("note", 1440, [0], dots=1)); left -= 2
    for it in items:
        if it.kind == "note" and it.beam_group < 0 and rng.random() < chord_p:
            it.steps = [0, 0]
    return items


def add_ties(rng: random.Random, items: list[Item], p: float):
    """Tie some single, undotted notes to the next note, which takes the same pitch."""
    for a, b in zip(items, items[1:]):
        if a.kind == b.kind == "note" and len(a.steps) == 1 and len(b.steps) == 1 and not a.dots \
                and rng.random() < p:
            a.tie_next = True
            b.steps = list(a.steps)
            b.acc = {}


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
        self.smufl = MUSIC_FONT is not None
        self.stem_w = max(1.2, (0.12 if self.smufl else 0.11) * u)
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
        if self.smufl and name in SMUFL:
            # natural size; the origin is the glyph's own anchor (noteheads and
            # accidentals: pitch centre; rests: middle line)
            scale = (height_u / HEIGHT_U.get(name, 1.0)) if height_u and name in ("black", "void") else 1.0
            m, ox, oy = smufl_mask(SMUFL[name], 4 * self.u * s * scale * SS, a)
            mh, mw = m.shape[0] / SS, m.shape[1] / SS
            base = cy
            if name in ("r_8", "r_16"):
                base = cy - 0.25 * self.u
            x0, y0 = cx - mw / 2, base - oy / SS
            self.c.paste(m, x0, y0)
            ys, xs = np.nonzero(m > 0.3)
            return x0 + xs.min() / SS, y0 + ys.min() / SS, (xs.max() - xs.min() + 1) / SS, (ys.max() - ys.min() + 1) / SS
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

    def draw_clef(self, clef: str, top: float, x: float, size: float = 1.0) -> float:
        name = {"G2": "gclef", "F4": "fclef", "C3": "cclef"}[clef]
        if self.smufl:
            m, ox, oy = smufl_mask(SMUFL[name], 4 * self.u * size * SS)
            base = self.y(top, {"G2": 6, "F4": 2, "C3": 4}[clef])
            self.c.paste(m, x, base - oy / SS)
            return x + m.shape[1] / SS
        h = HEIGHT_U[name] * self.u * SS
        m = glyph_mask(name, h)
        mh, mw = m.shape[0] / SS, m.shape[1] / SS
        if clef == "G2":
            y0 = self.y(top, 11) - mh            # bottom 1.5U below the staff
        else:
            y0 = self.y(top, -0.1)               # top on the top line
        self.c.paste(m, x, y0)
        return x + mw

    def change_clef(self, clef: str, top: float, x: float, octave: int) -> dict:
        """Change-size clef (80 % of the full one) inside the staff; for a G clef
        with octave -1 a small 8 hangs below it."""
        if self.smufl:
            x_right = self.draw_clef(clef, top, x, size=0.8)
            m, ox, oy = smufl_mask(SMUFL["gclef" if clef == "G2" else "fclef"], 4 * self.u * 0.8 * SS)
            mh, mw = m.shape[0] / SS, m.shape[1] / SS
            y0 = self.y(top, 6 if clef == "G2" else 2) - oy / SS
            if octave:
                eight = digit_mask("8", 1.0 * self.u * SS)
                ew = eight.shape[1] / SS
                self.c.paste(eight, x + mw * 0.55 - ew / 2, y0 + mh - 0.1 * self.u)
            return {"x": x, "y": y0, "w": mw, "h": mh, "cx": x + mw / 2, "clef": clef, "octave": octave}
        if clef == "G2":
            h = 5.6 * self.u
            m = glyph_mask("gclef", h * SS)
            y0 = self.y(top, 6) - 0.643 * h          # the curl stays on the G line
        else:
            h = 2.65 * self.u
            m = glyph_mask("fclef", h * SS)
            y0 = self.y(top, 2) - 0.318 * h          # the dots straddle the F line
        mh, mw = m.shape[0] / SS, m.shape[1] / SS
        self.c.paste(m, x, y0)
        if octave:
            eight = digit_mask("8", 1.0 * self.u * SS)
            ew = eight.shape[1] / SS
            self.c.paste(eight, x + mw * 0.55 - ew / 2, y0 + mh - 0.1 * self.u)
        return {"x": x, "y": y0, "w": mw, "h": mh, "cx": x + mw / 2, "clef": clef, "octave": octave}

    def tempo_mark(self, x: float, top: float, bpm: int) -> dict:
        """A quarter note and '= bpm' well above the staff (no ledger lines)."""
        u = self.u
        cy = top - 4.2 * u
        x0, y0, w, h = self.glyph("black", x, cy, height_u=0.85, jitter=False)
        self.c.rect(x0 + w - self.stem_w, cy - 2.6 * u, x0 + w, cy)
        text = digit_mask(f"= {bpm}", 1.5 * u * SS)
        self.c.paste(text, x0 + w + 0.5 * u, cy - 1.1 * u)
        return {"x": x0, "y": cy - 2.6 * u, "w": w + 0.5 * u + text.shape[1] / SS, "h": 3.1 * u, "bpm": bpm}

    def text(self, text: str, x: float, y_bottom: float, height_u: float, bold: bool = False, centre: bool = False,
             italic: bool = False):
        m = text_mask(text, height_u * self.u * SS, bold, italic)
        w = m.shape[1] / SS
        self.c.paste(m, x - w / 2 if centre else x, y_bottom - m.shape[0] / SS)
        return w

    def tie(self, x0: float, x1: float, cy: float, below: bool, on_line: bool):
        """Thin arc from just right of one head to just left of the next, kept
        inside a staff space as engravers do (a head on a line: a flat arc
        between that line and the next)."""
        u, sgn = self.u, (1 if below else -1)
        if on_line:
            ya, sag = cy + sgn * 0.58 * u, sgn * 0.25 * u
        else:
            ya, sag = cy + sgn * 0.7 * u, sgn * min(0.6 * u, 0.15 * (x1 - x0) + 0.25 * u)
        n = 24
        outer, inner = [], []
        for k in range(n + 1):
            f = k / n
            xx = x0 + f * (x1 - x0)
            bulge = 4 * f * (1 - f)
            outer.append((xx, ya + sag * bulge))
            inner.append((xx, ya + (sag - sgn * 0.2 * u) * bulge))
        self.c.polygon(outer + inner[::-1])

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
        if self.smufl:
            rows = []
            for text, step in ((str(beats), 2), (str(beat_type), 6)):
                parts = [smufl_mask(SMUFL_DIGIT0 + int(ch), 4 * self.u * SS) for ch in text]
                rows.append((parts, step, sum(p[0].shape[1] for p in parts) / SS))
            w = max(r[2] for r in rows)
            for parts, step, rw in rows:
                xx = x + (w - rw) / 2
                for m, ox, oy in parts:
                    self.c.paste(m, xx, self.y(top, step) - oy / SS)
                    xx += m.shape[1] / SS
            return x + w
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
DUR = {"m": (0, 0), "8t": (160, 0), "16t": (80, 0), "w": (1920, 0), "h.": (1440, 1), "h": (960, 0), "q.": (720, 1), "q": (480, 0), "8.": (360, 1),
       "8": (240, 0), "16": (120, 0)}


def pitch_step(clef: str, name: str) -> int:
    letter, octave = name[0], int(name[1:].lstrip("#b"))
    ref_letter, ref_oct, ref_step = CLEF_REF[clef]
    return ref_step - ((LETTERS.index(letter) + 7 * octave) - (LETTERS.index(ref_letter) + 7 * ref_oct))


def parse_items(tokens: list[str], clef: str, fifths: int = 0) -> list[Item]:
    """'G4:8b1' (pitch:duration, b<n> = beam group), 'C3+E3+G3:h', 'r:q',
    'C#4' / 'Bb3' for an accidental on the note."""
    items = []
    for tok in tokens:
        what, _, rest = tok.partition(":")
        dur, _, beam = rest.partition("b")
        ticks, dots = DUR[dur]
        if what == "r":
            items.append(Item("mrest" if dur == "m" else "rest", ticks))
            continue
        it = Item("note", ticks, [], dots=dots, beam_group=int(beam) if beam else -1)
        in_key = {l: 1 for l in SHARP_ORDER[:fifths]} if fifths > 0 else {l: -1 for l in FLAT_ORDER[:-fifths]}
        for p_ in what.split("+"):
            st = pitch_step(clef, p_)
            it.steps.append(st)
            alter = 1 if "#" in p_ else -1 if len(p_) > 2 and p_[1] == "b" else 0
            if alter != in_key.get(p_[0], 0):      # only what the key does not already say
                it.acc[st] = {1: "sharp", -1: "flat", 0: "natural"}[alter]
        items.append(it)
    return items


def engrave_page(seed: int, mode: str | None = None, u: float | None = None, degrade: int | None = None,
                 rotation: float | None = None, extended: bool = False,
                 score: dict | None = None) -> tuple[np.ndarray, np.ndarray, dict]:
    rng = random.Random(seed)
    if score:
        mode = "piano"
    mode = mode or ("piano" if seed % 3 == 2 else "single")
    u = u or rng.choice([18, 20, 22, 24, 26, 28])
    t = max(1.6, round(0.1 * u, 1))
    degrade = rng.choice([0, 1, 2]) if degrade is None else degrade
    jitter = 0.06 if degrade == 0 else 0.10
    width = 2400
    margin = 3.0 * u
    beats, beat_type = rng.choice([(4, 4), (4, 4), (3, 4)])
    fifths = rng.choice([-3, -2, -1, 0, 0, 1, 2, 3])
    if extended:
        beats, beat_type = rng.choice([(4, 4), (3, 4), (6, 8), (12, 8), (12, 8)])
        fifths = rng.choice([-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5])
    if score:
        beats, beat_type = score["time"]
        fifths = score.get("fifths", 0)
    compound = beat_type == 8
    measure_ticks = int(beats * 4 / beat_type * TPB)
    single_clef = rng.choice(["G2", "G2", "F4"])
    n_systems = 6 if mode == "single" else 4
    if score:
        n_systems = len(score["systems"])
    staff_gap = 9.0 * u if mode == "single" else 7.5 * u     # top-to-top inside a piano system
    system_gap = 11.0 * u
    clefs_per_system = [single_clef] if mode == "single" else ["G2", "F4"]
    sizes = [1.0] * len(clefs_per_system)
    gaps = [staff_gap] * len(clefs_per_system)            # top of each staff below the previous one
    groups = [list(range(len(clefs_per_system)))]          # staves joined by barlines
    if score and score.get("staves"):
        clefs_per_system = [st["clef"] for st in score["staves"]]
        sizes = [st.get("size", 1.0) for st in score["staves"]]
        gaps = [st.get("gap", 7.5) * u for st in score["staves"]]
        groups = score.get("groups", [list(range(len(clefs_per_system)))])
        staff_gap = float(np.mean(gaps))
    tempo_room = 4.0 * u if extended or score else 0.0
    if score:
        margin = 6.0 * u                     # room for the part names
    height = int(margin * 2 + n_systems * (len(clefs_per_system) * staff_gap) + n_systems * 2 * u + 8 * u
                 + tempo_room)
    if score and score.get("staves"):
        height = int(margin * 2 + tempo_room + n_systems * (sum(gaps[1:]) + 4 * u + score.get("system_gap", 11) * u)
                     + 8 * u)
    canvas = Canvas(width, height)
    eng = Engraver(canvas, rng, u, t, jitter)
    engs = []                                   # one engraver per staff slot (small staves: smaller U)
    for sz in sizes:
        e = eng if sz == 1.0 else Engraver(canvas, rng, u * sz, t, jitter)
        e.heads, e.rests = eng.heads, eng.rests
        engs.append(e)

    gt = {"seed": seed, "mode": mode, "u": u, "t": t, "degrade": degrade, "time_signature": [beats, beat_type],
          "fifths": fifths, "staves": [], "heads": [], "rests": [], "parts": len(clefs_per_system),
          "sizes": sizes,
          "barlines": [], "extended": extended}
    part_clock = [0] * len(clefs_per_system)
    y_cursor = margin + 3 * u + tempo_room
    staff_index = 0
    for system in range(n_systems):
        hidden = set(score["systems"][system].get("hide", [])) if score else set()
        tops, y_ = [], y_cursor
        for i in range(len(clefs_per_system)):
            if i > 0 and (i - 1) not in hidden and tops:
                y_ += gaps[i]
            elif i > 0 and tops:
                y_ += 0
            tops.append(y_)
        if hidden:                                  # hidden staves take no room
            shown = [i for i in range(len(clefs_per_system)) if i not in hidden]
            y_ = y_cursor
            for k, i in enumerate(shown):
                if k:
                    y_ += gaps[i]
                tops[i] = y_
            for i in hidden:
                tops[i] = -1e6
        # music: same measures count for both staves of a system
        ottava_of = None
        if score:
            spec = score["systems"][system]["measures"]
            n_measures = len(spec)
            clef_of = [[clef] * n_measures for clef in clefs_per_system]
            octave_of = [[0] * n_measures for _ in clefs_per_system]
            ottava_of = [[0] * n_measures for _ in clefs_per_system]
            content = [[parse_items(m["staves"][slot], clefs_per_system[slot], fifths) for m in spec]
                       for slot in range(len(clefs_per_system))]
            for m, mi in zip(spec, range(n_measures)):
                for slot in m.get("8va", []):
                    ottava_of[slot][mi] = 1          # written an octave lower, sounds as notated + 12
        elif compound:
            n_measures = rng.choice([2, 3]) if beats == 12 else rng.choice([3, 4])
        else:
            n_measures = rng.choice([3, 4]) if beats == 4 else rng.choice([4, 5])
        # clef in force per (slot, measure): a left-hand passage in the G clef
        if not score:
            clef_of = [[clef] * n_measures for clef in clefs_per_system]
            octave_of = [[0] * n_measures for _ in clefs_per_system]
        if ottava_of is None:
            ottava_of = [[0] * n_measures for _ in clefs_per_system]
        if not score and extended and mode == "piano" and n_measures >= 2 and rng.random() < 0.6:
            mi = rng.randrange(1, n_measures)
            mj = rng.randrange(mi, n_measures)
            octave = -1 if rng.random() < 0.4 else 0
            for k in range(mi, mj + 1):
                clef_of[1][k], octave_of[1][k] = "G2", octave
        for slot in range(len(clefs_per_system) if not score else 0):
            if slot == 0:
                content = []
            ms = []
            for mi in range(n_measures):
                cp = 0.18 if mode == "piano" else 0.08
                m = random_compound_measure(rng, beats // 3, cp) if compound else random_measure(rng, beats, cp)
                assign_pitches(rng, m, fifths, clef_of[slot][mi], acc_p=0.10)
                if extended:
                    add_ties(rng, m, 0.10)
                ms.append(m)
            content.append(ms)

        x_start = margin
        # header
        header_end = []
        for slot, clef in enumerate(clefs_per_system):
            e = engs[slot]
            if slot in hidden:
                continue
            x = e.draw_clef(clef, tops[slot], x_start + 0.6 * u) + 0.6 * e.u
            x = e.key_signature(clef, fifths, tops[slot], x) + 0.5 * e.u
            if system == 0:
                x = e.time_signature(beats, beat_type, tops[slot], x) + 0.4 * e.u
            header_end.append(x)
        x_first = max(header_end) + 1.4 * u
        if extended and system == 0:
            gt["tempo"] = eng.tempo_mark(x_first, tops[0], rng.choice([72, 88, 95, 108, 120]))
        if score and system == 0 and score.get("tempo"):
            gt["tempo"] = engs[0].tempo_mark(x_first, tops[0], score["tempo"])

        # clef changes (slot, measure) -> new clef drawn just after the barline before it
        changes = {}
        for slot in range(len(clefs_per_system)):
            for mi in range(1, n_measures):
                if (clef_of[slot][mi], octave_of[slot][mi]) != (clef_of[slot][mi - 1], octave_of[slot][mi - 1]):
                    changes[(slot, mi)] = (clef_of[slot][mi], octave_of[slot][mi])

        # shared x positions per measure: lay out the union of onsets across slots,
        # compressing the spacing (or dropping a measure) if the system would overflow
        def measure_len(mi: int) -> int:
            """A pickup measure is as long as its music; others follow the time signature."""
            if score and score["systems"][system]["measures"][mi].get("pickup"):
                return max(sum(it.ticks for it in content[slot][mi] if it.kind != "mrest")
                           for slot in range(len(clefs_per_system)))
            return measure_ticks

        def layout(n: int, scale: float):
            x = x_first
            plan = []
            for mi in range(n):
                if any((slot, mi) in changes for slot in range(len(clefs_per_system))):
                    x += 2.8 * u
                onsets = sorted({o for slot in range(len(clefs_per_system)) for o in _onsets(content[slot][mi])})
                m_len = measure_len(mi)
                acc_extra = {o: any((it.acc or _has_second(it)) for slot in range(len(clefs_per_system))
                                    for oo, it in _timed(content[slot][mi]) if oo == o) for o in onsets}
                xpos = {}
                for k, o in enumerate(onsets):
                    if acc_extra[o]:
                        x += 1.3 * u
                    xpos[o] = x
                    nxt = onsets[k + 1] if k + 1 < len(onsets) else m_len
                    x += scale * u * max(1.9, 2.3 + 1.5 * math.log2(max(nxt - o, 120) / 240))
                x += 0.3 * u
                plan.append((xpos, x, onsets))
                x += 1.6 * u
            return plan

        right_edge = width - margin
        scale = 1.0
        plan = layout(n_measures, scale)
        while plan[-1][1] > right_edge:
            scale *= 0.92
            if scale < 0.75:
                n_measures -= 1
                scale = 1.0
            plan = layout(n_measures, scale)

        barline_xs = []
        clef_changes = {slot: [] for slot in range(len(clefs_per_system))}
        for mi in range(n_measures):
            xpos, x_bar, onsets = plan[mi]
            m_len = measure_len(mi)
            for slot, clef in enumerate(clefs_per_system):
                top = tops[slot]
                e = engs[slot]
                if (slot, mi) in changes:
                    new_clef, octave = changes[(slot, mi)]
                    cc = e.change_clef(new_clef, top, barline_xs[-1] + 0.5 * u, octave)
                    clef_changes[slot].append(cc)
                for it in content[slot][mi]:
                    if it.kind == "mrest":
                        it.ticks = m_len
                if slot in hidden:
                    part_clock[slot] += m_len
                    continue
                _engrave_measure(e, content[slot][mi], xpos, top, clef_of[slot][mi], fifths, m_len,
                                 x_bar_left=(xpos[onsets[0]] - 1.2 * u), x_bar_right=x_bar,
                                 clock=part_clock[slot], part=slot,
                                 staff=staff_index + sum(1 for i in range(slot) if i not in hidden), gt=gt,
                                 octave=octave_of[slot][mi] + ottava_of[slot][mi])
                part_clock[slot] += m_len
            barline_xs.append(x_bar)
        if score:
            ms_spec = score["systems"][system]["measures"]
            for mi, m in enumerate(ms_spec[:n_measures]):
                xpos, _, onsets = plan[mi]
                if m.get("chord"):
                    eng.text(m["chord"], xpos[onsets[0]], tops[0] - 2.6 * u, 1.6)
                for slot, dyn in m.get("dyn", []):
                    e = engs[slot]
                    e.text(dyn, xpos[onsets[0]] - 0.3 * u, tops[slot] + 4 * e.u + 2.6 * e.u, 1.3, italic=True)
            # 8va: "8va" and a dashed line over the run of measures, above the staff
            for slot in range(len(clefs_per_system)):
                runs, start = [], None
                for mi in range(n_measures + 1):
                    on = mi < n_measures and ottava_of[slot][mi] == 1
                    if on and start is None:
                        start = mi
                    if not on and start is not None:
                        runs.append((start, mi - 1))
                        start = None
                for a, b in runs:
                    e = engs[slot]
                    xa = plan[a][0][plan[a][2][0]] - 0.6 * u
                    xb = plan[b][1] - 0.4 * u
                    y = tops[slot] - 3.2 * e.u
                    w8 = e.text("8va", xa, y + 0.6 * e.u, 1.1, italic=True)
                    xx = xa + w8 + 0.4 * u
                    while xx < xb:
                        canvas.rect(xx, y - 0.05 * e.u, min(xx + 0.45 * e.u, xb), y + 0.07 * e.u)
                        xx += 0.9 * e.u
                    canvas.rect(xb - 0.07 * e.u, y - 0.05 * e.u, xb + 0.05 * e.u, y + 0.8 * e.u)
            if score.get("staves"):
                for st, top, e in zip(score["staves"], tops, engs):
                    name = st.get("name" if system == 0 else "short", "")
                    if name:
                        w_lbl = text_mask(name, 1.1 * e.u * SS).shape[1] / SS
                        e.text(name, margin - 1.0 * u - w_lbl, top + 2.6 * e.u, 1.1)
            else:
                label = score.get("parts_label", ["Piano", "Pno."])[0 if system == 0 else 1]
                w_lbl = text_mask(label, 1.4 * u * SS).shape[1] / SS
                eng.text(label, margin - 1.2 * u - w_lbl, (tops[0] + tops[-1] + 4 * u) / 2 + 0.7 * u, 1.4)
            if system == 0 and score.get("title"):
                eng.text(score["title"], width / 2, margin, 2.2, centre=True)
            if system == 0 and score.get("composer"):
                eng.text(score["composer"], width - margin - 14 * u, margin + 2.2 * u, 1.3)
        x_end = barline_xs[-1]
        # barlines (spanning the whole system in piano mode), final double bar on the last system
        shown = [i for i in range(len(clefs_per_system)) if i not in hidden]
        y_top, y_bot = tops[shown[0]], tops[shown[-1]] + 4 * engs[shown[-1]].u
        bw = max(1.6, 0.16 * u)
        for grp in groups:
            grp = [i for i in grp if i not in hidden]
            if not grp:
                continue
            g_top, g_bot = tops[grp[0]], tops[grp[-1]] + 4 * engs[grp[-1]].u
            for k, bx in enumerate(barline_xs):
                canvas.rect(bx - bw / 2, g_top - t / 2, bx + bw / 2, g_bot + t / 2)
                if system == n_systems - 1 and k == len(barline_xs) - 1:
                    canvas.rect(bx - bw / 2 - 0.55 * u - 0.5 * u, g_top - t / 2, bx - bw / 2 - 0.55 * u, g_bot + t / 2)
        if mode == "piano":   # system start line
            canvas.rect(margin - bw / 2, y_top - t / 2, margin + bw / 2, y_bot + t / 2)
        for slot, clef in enumerate(clefs_per_system):
            if slot in hidden:
                continue
            engs[slot].staff_lines(tops[slot], margin, x_end + bw / 2)
            gt["staves"].append({"index": staff_index + sum(1 for i in range(slot) if i not in hidden),
                                 "system": system, "slot": slot,
                                 "lines": [tops[slot] + k * engs[slot].u for k in range(5)], "clef": clef,
                                 "key": fifths, "size": sizes[slot],
                                 "time_present": system == 0, "x0": margin, "x1": x_end,
                                 "barlines": barline_xs, "clef_changes": clef_changes[slot]})
        staff_index += len(clefs_per_system) - len(hidden)
        y_cursor = tops[shown[-1]] + staff_gap + (system_gap - staff_gap if mode == "piano" else 0)
        if score and score.get("staves"):
            y_cursor = tops[shown[-1]] + score.get("system_gap", 11.0) * u

    # ties: the first note sounds for the whole tied length, the others do not start a note
    by_key = {}
    for h in eng.heads:
        by_key.setdefault((h["part"], h["staff"]), []).append(h)
    for h in eng.heads:
        h["sound_ticks"] = h["ticks"]
    for hs in by_key.values():
        hs.sort(key=lambda h: (h["onset"], h["step"]))
        for i in range(len(hs) - 1, -1, -1):
            h = hs[i]
            if h.get("tie_next"):
                nxt = next((g for g in hs[i + 1:] if g["onset"] == h["onset"] + h["ticks"] and g["step"] == h["step"]),
                           None)
                if nxt is not None:
                    nxt["tied_from_prev"] = True
                    h["sound_ticks"] = h["ticks"] + nxt["sound_ticks"]
    gt["heads"] = eng.heads
    gt["rests"] = eng.rests
    gt["end_ticks"] = part_clock
    image = canvas.render(True)
    nostaff = canvas.render(False)
    image = degrade_page(image, degrade, rng, rotation)
    if rotation:
        nostaff = rotate(nostaff, rotation)
    return image, nostaff, gt


def _has_second(it: Item) -> bool:
    st = sorted(it.steps)
    return it.kind == "note" and any(b - a == 1 for a, b in zip(st, st[1:]))


def _timed(items: list[Item]):
    clock = 0
    for it in items:
        yield clock, it
        clock += it.ticks


def _onsets(items: list[Item]):
    return [o for o, _ in _timed(items)]


def _engrave_measure(eng: Engraver, items: list[Item], xpos: dict, top: float, clef: str, fifths: int,
                     measure_ticks: int, x_bar_left: float, x_bar_right: float, clock: int, part: int,
                     staff: int, gt: dict, octave: int = 0):
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
        n_beams = 1 if members[0][1].ticks in (240, 160) else 2
        if members[0][1].ticks in (160, 80):             # triplet: a "3" over the beam
            mid_x = (xs[0] + xs[-1]) / 2
            eng.text("3", mid_x, (min(ends) - 0.4 * u) if up else (max(ends) + 1.6 * u), 1.0, italic=True,
                     centre=True)
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
        up_dir = stem_info[id(it)][2] if id(it) in stem_info else np.mean(it.steps) >= 4
        # a second in a chord puts one head on the other side of the stem:
        # the upper one to the right (stem up) or the lower one to the left
        shift = {}
        order = sorted(it.steps, reverse=up_dir)          # from the stem's far end
        moved_prev = False
        for a, b in zip(order, order[1:]):
            if abs(a - b) == 1 and not moved_prev:
                shift[b] = 1 if up_dir else -1
                moved_prev = True
            else:
                moved_prev = False
        acc_extra = 1.2 * u if any(v < 0 for v in shift.values()) else 0.0
        for s in sorted(it.steps):
            if s in it.acc:
                name = it.acc[s]
                acx = x - 1.45 * u - acc_extra
                eng.glyph(name, acx, eng.y(top, s), anchor="flat" if name == "flat" else "center")
            hx = x + shift.get(s, 0) * (1.18 * u - eng.stem_w)
            heads.append(eng.notehead(head_kind, hx, top, s))
        if shift:
            normal = [h for h in heads if h["step"] not in shift] or heads
            heads_for_stem = normal
        else:
            heads_for_stem = heads
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
                up = up_dir
                hw = max(h["w"] for h in heads_for_stem) / 2 - eng.stem_w / 2
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
        if it.tie_next:
            nxt_o = o + it.ticks
            if nxt_o in xpos:
                up = stem_info[id(it)][2] if id(it) in stem_info else np.mean(it.steps) >= 4
                eng.tie(x + 0.35 * u, xpos[nxt_o] - 0.35 * u, heads[0]["cy"], below=up,
                        on_line=heads[0]["step"] % 2 == 0)   # opposite the stem
        for h in heads:
            letter, oct_ = step_to_pitch(clef, h["step"])
            if h["step"] in it.acc:
                alter = {"sharp": 1, "flat": -1, "natural": 0}[it.acc[h["step"]]]
                measure_acc[(letter, oct_)] = alter
            else:
                alter = measure_acc.get((letter, oct_), key_letters.get(letter, 0))
            h.update({"part": part, "staff": staff, "onset": clock + o, "ticks": it.ticks,
                      "pitch": midi_number(letter, oct_ + octave, alter), "name": f"{letter}{oct_ + octave}",
                      "alter": alter, "accidental": it.acc.get(h["step"]), "clef": clef,
                      "tie_next": it.tie_next})
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
    p.add_argument("--extended", action="store_true",
                   help="compound time, keys to 5 accidentals, tempo mark, ties, clef changes")
    p.add_argument("--music-font", help="SMuFL font (e.g. Leland.otf, Bravura.otf) for the music glyphs")
    p.add_argument("--score", help="JSON file with a fixed piece to engrave instead of random music")
    args = p.parse_args(argv)
    global MUSIC_FONT
    MUSIC_FONT = args.music_font
    score = json.load(open(args.score)) if args.score else None
    os.makedirs(args.out, exist_ok=True)
    for seed in parse_seeds(args.seeds):
        img, nostaff, gt = engrave_page(seed, args.mode, args.u, args.degrade, args.rotation, args.extended, score)
        stem = os.path.join(args.out, f"synth_{seed:03d}")
        cv2.imwrite(stem + ".png", img)
        cv2.imwrite(stem + "_nostaff.png", nostaff)
        with open(stem + ".json", "w") as f:
            json.dump(gt, f, indent=1, default=float)
        print(f"{stem}.png  mode={gt['mode']} U={gt['u']} degrade={gt['degrade']} "
              f"staves={len(gt['staves'])} heads={len(gt['heads'])} rests={len(gt['rests'])}")


if __name__ == "__main__":
    main()
