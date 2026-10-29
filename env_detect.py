# =============================================================================
#  env_detect.py  --  musical environment read from the page
#  Version: 1.0.3
#
#  Changelog
#  ---------
#  1.0.3  (2026-10-29)
#    - C clef (alto, tenor): its bars, cut from the body by staff removal,
#      are joined back onto a body that stays inside the staff. A thick bar
#      alone was taken for a barline and the body alone read as a bass clef,
#      so every pitch of an alto-clef part (viola) came out wrong.
#  1.0.2  (2026-10-28)
#    - FRAGMENT_GAP_U 0.15 -> 0.35 for loop-sized pieces (at most 1.4U x
#      0.8U and at least 0.08U^2; others keep 0.15): the thin left side of a MuseScore
#      (Leland) flat's loop lies on a staff line and is removed with it, so
#      the loop starts further from the stem; key signatures of flats were
#      read as sharps on such pages.
#    - Systems: a straight vertical stroke from the top of one staff to the
#      bottom of the next joins them too, even when it touches a clef or a bracket (its
#      connected component is then too wide for the old rule). A violin
#      staff and the piano below it were split into separate systems.
#    - Systems with fewer staves than the fullest one (empty staves hidden)
#      are matched to it by clef and staff size (_align_slots), so a staff
#      keeps its part; before, the staff position alone decided.
#    - A clef must be at least CLEF_MIN_W_U (0.6U) wide: a piece of the
#      system's opening line (thicker on enlarged pages) was read as a C clef.
#    - Clef, key and time signature of each staff are read at that staff's
#      own staff space (small staves, staff_detect 3.3.0).
#    - Only strokes up to ACC_STROKE_MAX_U (2.9U) are re-joined with
#      fragments (accidentals); pieces of a system's opening double line
#      (about 3U after staff removal) were joined into a clef-like glyph.
#    - Joined time-signature digits: two loops (4 over 4, 6 over 8) or a
#      dense glyph (about 40 % ink) are accepted too, and blurred digits up
#      to 80 % ink (enlarged small pages).
#  1.0.1  (2026-10-25)
#    - FIX: key signatures stopped after one or two flats on printed scores.
#      Staff removal cuts the thin join between a flat's stem and its loop
#      where it crosses a line, and the 2.5U stem was then treated as a
#      barline ("thin vertical") and never merged back. A stroke now counts
#      as barline-like only above BARLINE_MIN_U (3.2U; a barline spans 4U),
#      and a piece just right of a short stroke, inside its height, is
#      joined to it first (re-forms flats and naturals).
#    - A sharp (or natural) whose slanted bars lay on staff lines and were
#      removed with them leaves two short strokes of equal height; they are
#      joined again (key signatures stopped at such a sharp).
#    - A bass clef with only one dot found (the other lost to noise) is the
#      usual F clef on the fourth line; the single dot's space was rounded to
#      the third line before.
#    - Time signatures whose digits touch (12 over 8) form one staff-high
#      glyph; such a glyph right after the key signature is now located.
#      It must be wide in both its upper and lower third, so the first
#      note of a staff (head plus a staff-high stem) is not taken for one,
#      and either no vertical stroke is longer than 3.3U (a stem is) or the
#      glyph has an enclosed loop (6, 8, 4, 9; a beamed group of filled
#      notes has none) -- rejects beamed note groups, keeps blurred digits.
#  1.0.0  (2026-10-23)
#    - Systems: which staves are played together, from narrow ink strokes
#      (barlines, braces, brackets, system lines) that reach into both of two
#      neighbouring staves.
#    - Clefs: G / F / C, with the line they sit on (MusicXML sign + line), from
#      the first glyph's vertical extent, its longest straight vertical stroke
#      and, when present, the two dots that flank the reference line.
#    - Key signatures: tests "these are sharps" and "these are flats" against
#      the conventional positions for the clef, using shape only to break
#      ties. Reported as fifths (+sharps / -flats).
#    - Time signatures: located (two glyphs stacked at the same x), but the
#      digits are NOT read -- that needs the Stage 7 glyph classifier. The
#      confidence says so, and env_config asks the user for them.
#    - Every result carries a confidence and a short reason, so env_config can
#      ask only about what is missing or uncertain.
#    - Diagnostic figure in its own window.
#
#  Depends on staff_detect.py >= 3.3.0 and staff_remove.py >= 1.2.0
# =============================================================================

from __future__ import annotations

import argparse
import random
from dataclasses import dataclass, field
from glob import glob

import cv2
import numpy as np
from itertools import combinations
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from staff_detect import detect_staves
from staff_remove import StaffRemoval, remove_from_detection

# --------------------------------------------------------------------------- #
#  Tunables -- all lengths in units of U (staff line pitch) unless noted.
#  Steps are half-spaces counted DOWN from the top line: top line = 0,
#  bottom line = 8. A step of 1 is U/2.
# --------------------------------------------------------------------------- #
CONNECT_PENETRATION_U = 1.0   # a connector must reach this far into both staves
CONNECT_MAX_WIDTH_U   = 4.0   # wider strokes spanning a gap are slurs/beams
BRACKET_MIN_HEIGHT_U  = 10.0  # taller components are brackets/system lines...
BRACKET_CUT_RUN_U     = 9.0   # ...whose vertical runs this long get cut away
                              # (longer than any clef, so treble stems survive)

SCAN_START_U   = -1.5   # look for glyphs from this far left of the staff start
SCAN_END_U     = 16.0   # ...to this far right
MERGE_GAP_U    = 0.30   # side-by-side pieces closer than this form one glyph
MERGE_VOVERLAP = 0.60   # ...if they overlap vertically by this share
TWIN_GAP_U = 0.6       # two short strokes this close, of one height: a sharp's stems
FRAGMENT_GAP_U = 0.35   # a flat-loop-sized piece this close right of a short stroke ...
C_BAR_GAP_U = 0.35         # bars of a C clef this close left of its body belong to it
CLEF_MIN_W_U = 0.6         # clefs are wide; a narrower piece is a bracket or system line
ACC_STROKE_MAX_U = 2.9     # strokes of accidentals are shorter than this (system brackets are longer)
FRAGMENT_GAP_NEAR_U = 0.15  # (a larger piece: this close; 1.0.1 used 0.15 for all)
FRAGMENT_INSIDE = 0.85  # ...and this much inside its height is part of it
BARLINE_MIN_U  = 3.2    # thin strokes taller than this are barline-like
MIN_GLYPH_AREA = 0.03   # in U^2

TREBLE_MIN_HEIGHT_U = 4.8
TREBLE_MIN_BOTTOM   = 9.75   # trebles reach >= 9.9 steps on the samples; bass <= 9.6
C_CLEF_MIN_VRUN     = 0.75   # longest vertical run / height for a C clef bar...
C_CLEF_MAX_VRUN_X   = 0.45   # ...which sits on the LEFT of the glyph
F_CLEF_MIN_VRUN_X   = 0.60   # a bass clef's long stroke, if any, is its right-hand tail
F_DOT_STEPS         = (0.5, 3.5)  # bass clef dots straddle the F line (step 2)
C_BODY_MIN_WIDTH_U  = 0.35   # rows narrower than this are not part of a C clef body
CLEF_MIN_HEIGHT_U   = 2.5
CLEF_MAX_HEIGHT_U   = 9.5
CLEF_MIN_ASPECT     = 0.15   # width/height; below this it is a line, not a clef
DOT_MAX_SIZE_U      = 0.55
DOT_MAX_DISTANCE_U  = 1.4

ACC_MIN_HEIGHT_U = 1.1
ACC_MAX_HEIGHT_U = 3.8   # sharps up to 3.65U on the samples; stemmed notes ~4U+
ACC_MAX_WIDTH_U  = 2.2   # some writers draw sharps 2U wide
ACC_MAX_GAP_U    = 1.3   # largest gap between consecutive accidentals (<= 0.83U seen)
ACC_FIRST_GAP_U  = 3.5   # largest gap between clef and first accidental
ACC_STEP_TOL     = 2.0   # allowed |anchor - expected| in steps
ACC_GAP_GROWTH   = 2.0   # a gap may be at most this many times the mean so far
FLAT_MIN_SHAPE   = 0.0   # _flatness above this to count as a flat...
SHARP_MAX_SHAPE  = 0.2   # ...and below this to count as a sharp
KEY_START_TRIES  = 3     # first accidental may follow up to this many stray glyphs
TIME_MAX_GAP_U   = 4.5   # time signature must start this close to clef/key
TIME_UPPER_MAX_TOP    = 1.5   # upper digit starts within this many steps of the top line
TIME_LOWER_MIN_BOTTOM = 6.0   # lower digit reaches at least this far down (6.6+ seen)
TIME_MIN_GAP_U        = -0.05 # digits don't overlap (+0.10..+0.35U seen; a chord: -0.14U)
TIME_MAX_X_OFFSET_U   = 0.45  # digits are centred over each other

LETTERS = "CDEFGAB"


# --------------------------------------------------------------------------- #
#  Clefs, in MusicXML terms
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Clef:
    """A clef as MusicXML describes it: a sign and the staff line it marks.

    Attributes:
        sign (str): `"G"`, `"F"` or `"C"`.
        line (int): Staff line the clef marks, counted 1-5 from the BOTTOM.
    """
    sign: str
    line: int

    NAMES = {("G", 2): "treble", ("F", 4): "bass", ("C", 3): "alto",
             ("C", 4): "tenor", ("C", 1): "soprano", ("C", 2): "mezzo-soprano",
             ("C", 5): "baritone", ("F", 3): "baritone (F)", ("G", 1): "French violin"}
    REFERENCE = {"G": ("G", 4), "F": ("F", 3), "C": ("C", 4)}

    @property
    def name(self) -> str:
        return self.NAMES.get((self.sign, self.line), f"{self.sign} clef on line {self.line}")

    @property
    def code(self) -> str:
        """Compact form, e.g. `G2`."""
        return f"{self.sign}{self.line}"

    @property
    def reference_step(self) -> int:
        """Step (half-spaces down from the top line) of the line the clef marks."""
        return 2 * (5 - self.line)

    def step_to_pitch(self, step: int) -> tuple[str, int]:
        """Letter and octave (scientific pitch, middle C = C4) of a staff step.
        Args:
            step: Half-spaces down from the top line (0 = top line, 8 = bottom).
        """
        letter, octave = self.REFERENCE[self.sign]
        index = LETTERS.index(letter) + octave * 7 + (self.reference_step - int(step))
        return LETTERS[index % 7], index // 7

    def letter_step(self, letter: str, near: float) -> int:
        """Step at which `letter` sits closest to step `near`."""
        best = None
        for step in range(int(near) - 7, int(near) + 8):
            if self.step_to_pitch(step)[0] == letter:
                if best is None or abs(step - near) < abs(best - near):
                    best = step
        return best

    @classmethod
    def parse(cls, text: str) -> "Clef":
        """Accepts `"G2"`, `"treble"`, `"bass"`, `"alto"`, `"tenor"`, ..."""
        text = text.strip()
        for (sign, line), name in cls.NAMES.items():
            if text.lower() == name:
                return cls(sign, line)
        if len(text) == 2 and text[0].upper() in "GFC" and text[1] in "12345":
            return cls(text[0].upper(), int(text[1]))
        raise ValueError(f"Unknown clef '{text}'. Try treble, bass, alto, tenor, or G2/F4/C3.")


TREBLE, BASS, ALTO, TENOR = Clef("G", 2), Clef("F", 4), Clef("C", 3), Clef("C", 4)

SHARP_ORDER = "FCGDAEB"
FLAT_ORDER = "BEADGCF"

# Conventional key signature positions (steps from the top line). Bass and alto
# are the treble pattern shifted down 2 and 1 steps. Tenor sharps break the
# zigzag (F# is written low, on step 6) to stay inside the staff.
KEY_POSITIONS = {
    "G2": {"sharp": [0, 3, -1, 2, 5, 1, 4], "flat": [4, 1, 5, 2, 6, 3, 7]},
    "F4": {"sharp": [2, 5, 1, 4, 7, 3, 6], "flat": [6, 3, 7, 4, 8, 5, 9]},
    "C3": {"sharp": [1, 4, 0, 3, 6, 2, 5], "flat": [5, 2, 6, 3, 7, 4, 8]},
    "C4": {"sharp": [6, 2, 5, 1, 4, 0, 3], "flat": [3, 0, 4, 1, 5, 2, 6]},
}


def expected_key_steps(clef: Clef, kind: str) -> list[int]:
    """Where each accidental of a key signature sits, for any clef.
    - Uses the conventional table for treble, bass, alto and tenor.
    - For rarer clefs, places each letter at the step nearest the position it
      would have in treble, keeping the familiar zigzag inside the staff.
    """
    if clef.code in KEY_POSITIONS:
        return KEY_POSITIONS[clef.code][kind]
    order = SHARP_ORDER if kind == "sharp" else FLAT_ORDER
    return [clef.letter_step(letter, near) for letter, near in zip(order, KEY_POSITIONS["G2"][kind])]


# --------------------------------------------------------------------------- #
#  Results
# --------------------------------------------------------------------------- #
@dataclass
class Found:
    """A detected value with how sure we are and why.

    Attributes:
        value: The detected value, or `None` if not determined.
        confidence (str): `"high"`, `"medium"`, `"low"` or `"none"`.
        reason (str): One line explaining the decision.
        boxes (list): `(x, y, w, h)` boxes of the glyphs it came from.
    """
    value: object = None
    confidence: str = "none"
    reason: str = ""
    boxes: list = field(default_factory=list)

    @property
    def certain(self) -> bool:
        return self.value is not None and self.confidence in ("high", "medium")


@dataclass
class StaffEnvironment:
    """What the start of one staff says.

    Attributes:
        index (int): Staff index on the page, top to bottom.
        system (int): System it belongs to.
        slot (int): Position within its system (0 = top). Staves in the same
            slot across systems are the same part.
        clef (Found): `Clef` or None.
        key (Found): fifths (int) or None.
        time (Found): `(beats, beat_type)` or None. `present` in `time.reason`
            when a time signature was located but not read.
        time_present (bool): Whether a time signature was located.
    """
    index: int
    system: int
    slot: int
    clef: Found
    key: Found
    time: Found
    time_present: bool = False


@dataclass
class PageEnvironment:
    """Everything env_detect read from one page.

    Attributes:
        systems (list[list[int]]): Staff indices of each system.
        staves (list[StaffEnvironment]): Per-staff findings.
        slots (int): Staves per system (if consistent).
        consistent (bool): Every system has the same number of staves, so
            slot = part is safe.
        warnings (list[str]): Anything worth a human look.
    """
    systems: list
    staves: list
    slots: int
    consistent: bool
    warnings: list


# --------------------------------------------------------------------------- #
#  Geometry helpers
# --------------------------------------------------------------------------- #
def y_to_step(paths: np.ndarray, x: float, y: float) -> float:
    """Step (half-spaces down from the top line) of point (x, y) on a staff.
    - Uses the tracked line paths at column x, so curved staves are handled.
    - Beyond the staff, extrapolates with the local line spacing.
    """
    col = int(np.clip(round(x), 0, paths.shape[1] - 1))
    ys = paths[:, col]
    spacing = (ys[-1] - ys[0]) / (len(ys) - 1)
    if y <= ys[0]:
        return 2.0 * (y - ys[0]) / spacing
    if y >= ys[-1]:
        return 2.0 * (len(ys) - 1) + 2.0 * (y - ys[-1]) / spacing
    return float(np.interp(y, ys, np.arange(0, 2 * len(ys), 2)))


def longest_vertical_run(mask: np.ndarray) -> tuple[int, float]:
    """Longest vertical run of ink in a boolean mask, and where it is.
    Returns:
        `(length, x)` with x relative to the mask width (0 = left, 1 = right).
    """
    padded = np.pad(mask.astype(np.int8), ((1, 1), (0, 0)))
    edges = np.diff(padded, axis=0)
    best, best_col = 0, 0
    for col in range(mask.shape[1]):
        starts = np.flatnonzero(edges[:, col] == 1)
        ends = np.flatnonzero(edges[:, col] == -1)
        if starts.size and (ends - starts).max() > best:
            best, best_col = int((ends - starts).max()), col
    return best, best_col / max(mask.shape[1] - 1, 1)


def body_rows(mask: np.ndarray, min_width: float) -> tuple[int, int]:
    """First and last row whose ink spans at least `min_width` pixels -- the
    body of a glyph without thin attachments (a bracket hook, a stray tail)
    above or below it. A C clef's waist is narrow, so this deliberately spans
    across narrow rows in the middle."""
    rows = mask.any(axis=1)
    left = np.where(rows, mask.argmax(axis=1), 0)
    right = np.where(rows, mask.shape[1] - 1 - mask[:, ::-1].argmax(axis=1), -1)
    wide = np.flatnonzero((right - left + 1) >= min_width)
    if wide.size == 0:
        return 0, mask.shape[0] - 1
    return int(wide[0]), int(wide[-1])


@dataclass
class Glyph:
    """One or more connected components treated as a single symbol."""
    x: int
    y: int
    w: int
    h: int
    area: int
    mask: np.ndarray            # boolean, (h, w)
    top_step: float = 0.0
    bottom_step: float = 0.0

    @property
    def right(self) -> int:
        return self.x + self.w

    @property
    def box(self) -> tuple[int, int, int, int]:
        return (self.x, self.y, self.w, self.h)

    @property
    def centre_step(self) -> float:
        return (self.top_step + self.bottom_step) / 2.0


def _two_wide_halves(g: Glyph) -> bool:
    """Two stacked digits are wide in both the upper and the lower third;
    a notehead with its stem is wide at one end only."""
    third = max(g.h // 3, 1)
    def span(rows: np.ndarray) -> int:
        cols = np.flatnonzero(rows.any(axis=0))
        return int(cols[-1] - cols[0] + 1) if cols.size else 0
    return span(g.mask[:third]) >= 0.45 * g.w and span(g.mask[-third:]) >= 0.45 * g.w


def _holes(mask: np.ndarray, min_area: float) -> int:
    """Enclosed background regions (the loops of 6, 8, 9, 0, 4)."""
    bg = np.pad(~mask, 1, constant_values=True).astype(np.uint8)
    n, lab, st, _ = cv2.connectedComponentsWithStats(bg, connectivity=4)
    outside = lab[0, 0]
    return sum(1 for k in range(1, n) if k != outside and st[k, cv2.CC_STAT_AREA] >= min_area)


def _longest_vertical_run(mask: np.ndarray) -> int:
    best = 0
    for col in mask.T:
        padded = np.concatenate(([0], col.astype(np.int8), [0]))
        d = np.diff(padded)
        starts, ends = np.flatnonzero(d == 1), np.flatnonzero(d == -1)
        if starts.size:
            best = max(best, int((ends - starts).max()))
    return best


def _merge(a: Glyph, b: Glyph) -> Glyph:
    x, y = min(a.x, b.x), min(a.y, b.y)
    r, btm = max(a.right, b.right), max(a.y + a.h, b.y + b.h)
    mask = np.zeros((btm - y, r - x), dtype=bool)
    mask[a.y - y:a.y - y + a.h, a.x - x:a.x - x + a.w] |= a.mask
    mask[b.y - y:b.y - y + b.h, b.x - x:b.x - x + b.w] |= b.mask
    return Glyph(x, y, r - x, btm - y, a.area + b.area, mask)


def _cut_brackets(cleaned: np.ndarray, u: float) -> np.ndarray:
    """Removes the long vertical strokes of very tall components (brackets,
    system lines) so that clefs drawn touching them become separate glyphs."""
    image = cleaned > 0
    n, labels, stats, _ = cv2.connectedComponentsWithStats(image.astype(np.uint8), connectivity=8)
    tall = np.flatnonzero(stats[:, cv2.CC_STAT_HEIGHT] >= BRACKET_MIN_HEIGHT_U * u)
    tall = tall[tall > 0]
    if tall.size == 0:
        return image

    out = image.copy()
    limit = BRACKET_CUT_RUN_U * u
    for c in tall:
        x, y, w, h = stats[c, :4]
        sub = labels[y:y + h, x:x + w] == c
        padded = np.pad(sub.astype(np.int8), ((1, 1), (0, 0)))
        edges = np.diff(padded, axis=0)
        for col in range(w):
            starts = np.flatnonzero(edges[:, col] == 1)
            ends = np.flatnonzero(edges[:, col] == -1)
            for s, e in zip(starts, ends):
                if e - s >= limit:
                    out[y + s:y + e, x + col] = False
    return out


# --------------------------------------------------------------------------- #
#  Systems
# --------------------------------------------------------------------------- #
def _vertical_through(ink: np.ndarray, y0: float, y1: float, u: float) -> int:
    """Number of narrow vertical strokes covering every row from y0 to y1."""
    y0, y1 = int(max(0, np.floor(y0))), int(min(ink.shape[0], np.ceil(y1)))
    if y1 - y0 < 2:
        return 0
    full = ink[y0:y1].all(axis=0)
    cols = np.flatnonzero(full)
    if cols.size == 0:
        return 0
    runs = np.split(cols, np.flatnonzero(np.diff(cols) > 1) + 1)
    return sum(1 for r in runs if r.size <= CONNECT_MAX_WIDTH_U * u / 4)


def group_systems(result: dict, removal: StaffRemoval) -> tuple[list[list[int]], list[int]]:
    """Groups staves into systems.

    Two neighbouring staves are in the same system when some narrow ink stroke
    reaches at least CONNECT_PENETRATION_U into both: a barline drawn through,
    a brace, a bracket or the system's opening line. Long slurs and beams can
    also cross a gap, but they are wide, so width rules them out.

    Returns:
        `(systems, connector_counts)` -- staff indices per system, and how many
        connectors joined each neighbouring pair (for diagnostics).
    """
    staves, u = result["staves"], result["interline"]
    if not staves:
        return [], []

    n, _, stats, _ = cv2.connectedComponentsWithStats((removal.cleaned > 0).astype(np.uint8), connectivity=8)
    tops = stats[:, cv2.CC_STAT_TOP]
    bottoms = tops + stats[:, cv2.CC_STAT_HEIGHT]
    narrow = stats[:, cv2.CC_STAT_WIDTH] <= CONNECT_MAX_WIDTH_U * u
    narrow[0] = False

    systems, counts = [[0]], []
    for i in range(len(staves) - 1):
        reach = CONNECT_PENETRATION_U * u
        joined = narrow & (tops <= staves[i].bottom - reach) & (bottoms >= staves[i + 1].top + reach)
        # or a straight vertical stroke through the whole gap (the system's
        # opening line, when a clef or a bracket touches it and makes its
        # component wide)
        # (it runs from the top of the upper staff to the bottom of the lower
        # one, which a chain of stems and ledger notes does not)
        through = _vertical_through(removal.cleaned > 0, staves[i].top + 0.3 * u, staves[i + 1].bottom - 0.3 * u, u)
        counts.append(int(joined.sum()) + through)
        if joined.any() or through:
            systems[-1].append(i + 1)
        else:
            systems.append([i + 1])
    return systems, counts


# --------------------------------------------------------------------------- #
#  Glyphs at the start of a staff
# --------------------------------------------------------------------------- #
def staff_start_glyphs(image: np.ndarray, labels: np.ndarray, stats: np.ndarray,
                       owner: np.ndarray, staff_index: int, paths: np.ndarray,
                       x0: int, u: float) -> list[Glyph]:
    """Glyphs near the left end of one staff, merged and sorted left to right."""
    lo, hi = x0 + SCAN_START_U * u, x0 + SCAN_END_U * u
    pieces = []
    for c in np.flatnonzero(owner == staff_index):
        x, y, w, h, area = stats[c]
        if not (lo <= x <= hi) or area < MIN_GLYPH_AREA * u * u:
            continue
        pieces.append(Glyph(int(x), int(y), int(w), int(h), int(area),
                            labels[y:y + h, x:x + w] == c))
    pieces.sort(key=lambda g: g.x)

    # 1. re-join fragments cut off a short stroke by staff removal (flat loops)
    merged = True
    while merged:
        merged = False
        for i in range(len(pieces)):
            a = pieces[i]
            if not (a.w < 0.45 * u and a.h <= ACC_STROKE_MAX_U * u):   # accidental strokes, not bracket lines
                continue
            for j in range(len(pieces)):
                b = pieces[j]
                if j == i or b.x < a.x:
                    continue
                gap = b.x - a.right
                inside = min(a.y + a.h, b.y + b.h) - max(a.y, b.y)
                # the two strokes of a sharp or natural whose cross bars were
                # removed with the staff lines
                twin = (b.w < 0.45 * u and abs(b.h - a.h) <= 0.4 * u and gap <= TWIN_GAP_U * u
                        and inside >= 0.8 * min(a.h, b.h))
                if b.h >= a.h and not twin:
                    continue
                # a small loop may start further away (its thin join lay on a line)
                reach = FRAGMENT_GAP_U if (b.h <= 1.4 * u and b.w <= 0.8 * u and b.area >= 0.08 * u * u) \
                    else FRAGMENT_GAP_NEAR_U         # (not specks of noise)
                if twin or (gap <= reach * u and inside >= FRAGMENT_INSIDE * b.h):
                    pieces[i] = _merge(a, b)
                    del pieces[j]
                    merged = True
                    break
            if merged:
                break
    pieces.sort(key=lambda g: g.x)

    # 2. side-by-side halves of one glyph
    merged = True
    while merged:
        merged = False
        for i in range(len(pieces)):
            for j in range(i + 1, len(pieces)):
                a, b = pieces[i], pieces[j]
                if _is_thin_vertical(a, u) or _is_thin_vertical(b, u):
                    continue   # barlines drawn against a clef stay separate
                gap = max(a.x, b.x) - min(a.right, b.right)
                overlap = min(a.y + a.h, b.y + b.h) - max(a.y, b.y)
                # relative to the TALLER piece: halves of one glyph are similar
                # in height, whereas a sharp beside a treble clef is not
                if gap <= MERGE_GAP_U * u and overlap >= MERGE_VOVERLAP * max(a.h, b.h):
                    pieces[i] = _merge(a, b)
                    del pieces[j]
                    merged = True
                    break
            if merged:
                break

    for g in pieces:
        cx = g.x + g.w / 2
        g.top_step = y_to_step(paths, cx, g.y)
        g.bottom_step = y_to_step(paths, cx, g.y + g.h)
    pieces.sort(key=lambda g: g.x)
    return pieces


def _is_thin_vertical(g: Glyph, u: float) -> bool:
    """Barlines, double bars and bracket remnants at the start of a staff."""
    return g.w < 0.45 * u and g.h > BARLINE_MIN_U * u


def _is_dot(g: Glyph, u: float) -> bool:
    return g.w <= DOT_MAX_SIZE_U * u and g.h <= DOT_MAX_SIZE_U * u


# --------------------------------------------------------------------------- #
#  Clef
# --------------------------------------------------------------------------- #
def detect_clef(glyphs: list[Glyph], paths: np.ndarray, u: float) -> tuple[Found, int]:
    """Finds and classifies the clef at the start of a staff.

    Decision, in order (thresholds measured on the 15 sample pages):
      1. Treble (G2): reaches far below the staff (bottom >= TREBLE_MIN_BOTTOM
         steps) and is tall, with no bass-clef dots.
      2. Bass (F): two dots, or one, between steps 0.5 and 3.5 to its right --
         they flank the F line.
      3. C clef: a straight bar running most of its height on its LEFT side.
         The line it marks is the centre of its body, snapped to a line.
      4. Bass (F), less sure: curved, any long stroke is its right-hand tail.

    Returns:
        `(found, end_x)` -- the clef, and the x where the clef (with its dots)
        ends, so the key signature search starts after it.
    """
    candidates = [g for g in glyphs
                  if not _is_thin_vertical(g, u) and not _is_dot(g, u) and g.w >= CLEF_MIN_W_U * u
                  and CLEF_MIN_HEIGHT_U * u <= g.h <= CLEF_MAX_HEIGHT_U * u
                  and g.w >= CLEF_MIN_ASPECT * g.h
                  and -6 <= g.centre_step <= 14]
    if not candidates:
        return Found(None, "none", "no clef-sized glyph at the start of the staff"), \
            (glyphs[0].x if glyphs else 0)

    g = candidates[0]
    # a C clef's two bars are separate strokes once staff removal cuts the
    # thin join at the middle line, and a thick bar alone looks like a
    # barline: join bars standing just left of the body back onto it
    # (only for a body that stays inside the staff, as a C clef does; a G or
    # F clef next to the system's opening line is left alone)
    inside = g.top_step >= -1.5 and g.bottom_step <= 9.5
    for b in sorted([b for b in glyphs if b.right <= g.x + 1 and b.x < g.x], key=lambda b: -b.right):
        if not inside:
            break
        overlap = min(b.y + b.h, g.y + g.h) - max(b.y, g.y)
        if g.x - b.right <= C_BAR_GAP_U * u and b.w <= 0.8 * u and b.h >= 0.7 * g.h and overlap >= 0.7 * g.h \
                and b.top_step >= -1.5 and b.bottom_step <= 9.5:
            top, bottom = min(g.top_step, b.top_step), max(g.bottom_step, b.bottom_step)
            g = _merge(b, g)
            g.top_step, g.bottom_step = top, bottom
        elif g.x - b.right > C_BAR_GAP_U * u:
            break
    height_u = g.h / u
    run, run_x = longest_vertical_run(g.mask)
    vrun = run / g.h

    dots = [d for d in glyphs
            if _is_dot(d, u) and 0 <= d.x - g.right <= DOT_MAX_DISTANCE_U * u
            and -2 <= d.centre_step <= 9]
    f_dots = [d for d in dots if F_DOT_STEPS[0] <= d.centre_step <= F_DOT_STEPS[1]]
    end_x = max([g.right] + [d.right for d in dots])
    shape = (f"{height_u:.1f}U tall, steps {g.top_step:.1f}..{g.bottom_step:.1f}, "
             f"longest straight stroke {vrun:.0%} at x={run_x:.2f}")

    if g.bottom_step >= TREBLE_MIN_BOTTOM and height_u >= TREBLE_MIN_HEIGHT_U and not f_dots:
        conf = "high" if g.bottom_step >= TREBLE_MIN_BOTTOM + 0.5 else "medium"
        return Found(TREBLE, conf, f"reaches well below the staff ({shape})", [g.box]), end_x

    if f_dots:
        dot_step = float(np.mean([d.centre_step for d in f_dots]))
        line_step = int(np.clip(2 * round(dot_step / 2), 0, 8))
        if len(f_dots) == 1:
            # one dot (the other lost to noise or merged): it sits in a space
            # next to the F line, so it cannot tell which; take the usual one
            line_step = 2
        conf = "high" if len(f_dots) >= 2 else "medium"
        return Found(Clef("F", 5 - line_step // 2), conf,
                     f"{len(f_dots)} dot(s) around step {dot_step:.1f} ({shape})", [g.box]), end_x

    if vrun >= C_CLEF_MIN_VRUN and run_x <= C_CLEF_MAX_VRUN_X:
        top, bottom = body_rows(g.mask, C_BODY_MIN_WIDTH_U * u)
        cx = g.x + g.w / 2
        centre = (y_to_step(paths, cx, g.y + top) + y_to_step(paths, cx, g.y + bottom)) / 2
        line_step = int(np.clip(2 * round(centre / 2), 0, 8))
        off = abs(centre - line_step)
        conf = "high" if off <= 0.6 else "medium" if off <= 1.0 else "low"
        return Found(Clef("C", 5 - line_step // 2), conf,
                     f"bar on the left, body centred on step {centre:.1f} ({shape})", [g.box]), end_x

    if run_x >= F_CLEF_MIN_VRUN_X or vrun < C_CLEF_MIN_VRUN:
        return Found(BASS, "medium", f"curved, no dots ({shape})", [g.box]), end_x

    return Found(BASS, "low", f"unclear shape, assuming bass ({shape})", [g.box]), end_x


# --------------------------------------------------------------------------- #
#  Time signature (located, not read)
# --------------------------------------------------------------------------- #
def find_stacked_pairs(glyphs: list[Glyph], u: float) -> list[tuple[Glyph, Glyph]]:
    """Pairs of glyphs stacked at the same x, upper one in the top half of the
    staff and lower one in the bottom half -- the shape of a time signature."""
    pairs = []
    for i, a in enumerate(glyphs):
        for b in glyphs[i + 1:]:
            upper, lower = (a, b) if a.centre_step < b.centre_step else (b, a)
            overlap = min(a.right, b.right) - max(a.x, b.x)
            if overlap < 0.5 * min(a.w, b.w):
                continue
            if not (-1.0 <= upper.centre_step <= 4.0 and 4.0 <= lower.centre_step <= 9.0):
                continue
            if upper.top_step > TIME_UPPER_MAX_TOP or lower.bottom_step < TIME_LOWER_MIN_BOTTOM:
                continue   # digits fill their half of the staff; chords rarely do
            if abs((upper.x + upper.w / 2) - (lower.x + lower.w / 2)) > TIME_MAX_X_OFFSET_U * u:
                continue
            if lower.y - (upper.y + upper.h) < TIME_MIN_GAP_U * u:
                continue
            if not all(1.0 * u <= g.h <= 3.0 * u for g in (upper, lower)):
                continue
            pairs.append((upper, lower))
    return pairs


# --------------------------------------------------------------------------- #
#  Key signature
# --------------------------------------------------------------------------- #
def _flatness(g: Glyph) -> float:
    """> 0 looks like a flat (thin top, wide loop low), < 0 like a sharp."""
    h = g.mask.shape[0]

    def width(r0: float, r1: float) -> float:
        rows = g.mask[int(r0 * h):max(int(r1 * h), int(r0 * h) + 1)]
        cols = np.flatnonzero(rows.any(axis=0))
        return (cols.max() - cols.min() + 1) / g.w if cols.size else 0.0

    ratio = width(0.0, 0.35) / max(width(0.65, 1.0), 1e-3)
    cy = np.nonzero(g.mask)[0].mean() / h
    return (0.9 - ratio) + 2.0 * (cy - 0.5)


def _anchor(g: Glyph, kind: str) -> float:
    """The staff step an accidental marks: a sharp's centre, a flat's loop."""
    return g.centre_step if kind == "sharp" else g.bottom_step - 1.0


def detect_key(glyphs: list[Glyph], clef: Clef | None, start_x: int,
               exclude: set, u: float) -> tuple[Found, int]:
    """Reads a key signature after the clef.

    Both hypotheses -- "these are sharps" and "these are flats" -- are walked
    left to right against the conventional positions for this clef. Each keeps
    going while the next glyph is accidental-sized, close to the previous one
    and sits within ACC_STEP_TOL steps of where the next accidental belongs.
    The longer run wins; on a tie, glyph shape decides.

    Returns:
        `(found, end_x)` -- fifths, and where the key signature ends.
    """
    if clef is None:
        return Found(None, "none", "no clef, so accidental positions are unknown"), start_x

    pool = [g for g in glyphs
            if g.x + g.w / 2 >= start_x - 0.1 * u
            and not _is_dot(g, u) and not _is_thin_vertical(g, u)]

    def fits_shape(g: Glyph, kind: str) -> bool:
        f = _flatness(g)
        return f > FLAT_MIN_SHAPE if kind == "flat" else f < SHARP_MAX_SHAPE

    def walk_from(kind: str, start: int) -> list[Glyph]:
        expected = expected_key_steps(clef, kind)
        taken, last_right, gaps = [], start_x, []
        for g in pool[start:]:
            if len(taken) == 7 or id(g) in exclude:
                break   # a time signature always comes after the key signature
            gap = g.x - last_right
            limit = ACC_FIRST_GAP_U * u if not taken else ACC_MAX_GAP_U * u
            if gaps:
                limit = min(limit, max(1.0 * u, ACC_GAP_GROWTH * float(np.mean(gaps))))
            if gap > limit:
                break
            if not (ACC_MIN_HEIGHT_U * u <= g.h <= ACC_MAX_HEIGHT_U * u and g.w <= ACC_MAX_WIDTH_U * u):
                break
            if abs(_anchor(g, kind) - expected[len(taken)]) > ACC_STEP_TOL:
                break
            if not fits_shape(g, kind):
                break
            if taken:
                gaps.append(max(gap, 0))
            taken.append(g)
            last_right = g.right
        return taken

    def walk(kind: str) -> list[Glyph]:
        best: list[Glyph] = []
        for start in range(min(KEY_START_TRIES, len(pool))):
            if pool[start].x - start_x > ACC_FIRST_GAP_U * u or id(pool[start]) in exclude:
                break
            run = walk_from(kind, start)
            if len(run) > len(best):
                best = run
        return best

    sharps, flats = walk("sharp"), walk("flat")
    if not sharps and not flats:
        return Found(0, "medium", "no accidentals where a key signature would start"), start_x

    shape = {k: float(np.mean([_flatness(g) for g in run])) if run else 0.0
             for k, run in (("sharp", sharps), ("flat", flats))}
    if len(sharps) != len(flats):
        kind = "sharp" if len(sharps) > len(flats) else "flat"
    else:
        kind = "flat" if shape["flat"] > 0 else "sharp"
    run = sharps if kind == "sharp" else flats
    count = len(run)
    fifths = count if kind == "sharp" else -count

    shape_agrees = (shape[kind] > 0) == (kind == "flat")
    if shape_agrees and len(sharps) != len(flats):
        conf = "high"
    elif shape_agrees or len(sharps) != len(flats):
        conf = "medium"
    else:
        conf = "low"
    word = "sharp" if kind == "sharp" else "flat"
    reason = (f"{count} {word}{'s' if count != 1 else ''} at the conventional "
              f"{clef.name} positions (sharps fit {len(sharps)}, flats fit {len(flats)}; "
              f"shape {'agrees' if shape_agrees else 'disagrees'})")
    return Found(fifths, conf, reason, [g.box for g in run]), run[-1].right


# --------------------------------------------------------------------------- #
#  Whole page
# --------------------------------------------------------------------------- #
def _flag_key_disagreement(staves: list, warnings: list):
    """Lowers confidence where key signatures on one page disagree.

    Parts can legitimately differ (transposing instruments), but on the sample
    pages every high-confidence key error showed up as a disagreement. So:
    with a clear page majority, the odd ones out become "low"; with no clear
    majority, every staff becomes "low". "low" makes env_config ask.
    """
    values = [s.key.value for s in staves if s.key.value is not None]
    if len(set(values)) <= 1:
        return
    winner = max(set(values), key=values.count)
    clear = values.count(winner) > len(values) / 2
    for s in staves:
        if s.key.value is None or (clear and s.key.value == winner):
            continue
        note = (f"differs from {values.count(winner)}/{len(values)} staves reading {winner}"
                if clear else "staves on this page disagree")
        s.key = Found(s.key.value, "low", f"{s.key.reason}; {note}", s.key.boxes)
    warnings.append(f"key signatures differ between staves: {sorted(set(values))} "
                    f"({'odd ones out' if clear else 'all'} marked low confidence; "
                    "transposing instruments, or a misread)")


def _harmonise_keys(staves: list, warnings: list):
    """A part repeats its key signature at the start of every staff, so a
    staff that disagrees with a clear majority of its part is a misread.

    Within each slot (part), if more than half of the staves agree on a key,
    staves that read something else are set to it, with confidence "medium"
    and a reason saying so. Staves with a confident detection that the
    majority overrules are reported as warnings, since key changes do happen.
    """
    by_slot: dict[int, list] = {}
    for s in staves:
        by_slot.setdefault(s.slot, []).append(s)
    for slot, members in by_slot.items():
        if len(members) < 3:
            continue
        values = [s.key.value for s in members if s.key.value is not None]
        if not values:
            continue
        winner = max(set(values), key=values.count)
        if values.count(winner) <= len(members) / 2:
            continue
        for s in members:
            if s.key.value != winner:
                if s.key.confidence == "high":
                    warnings.append(f"staff {s.index}: read key {s.key.value}, part majority says "
                                    f"{winner}; using {winner} (check for a key change)")
                s.key = Found(winner, "medium",
                              f"set to the part's majority ({values.count(winner)}/{len(members)} staves); "
                              f"this staff read {s.key.value}: {s.key.reason}", s.key.boxes)


def _align_slots(out: list, systems: list, staves: list, page_u: float, warnings: list) -> None:
    """Systems with fewer staves (empty staves hidden, as MuseScore does by
    default) are matched to the fullest system by clef and staff size, in
    order, so each staff keeps its part. Without this, the piano right hand
    of a violin + piano score became "the violin" on such systems."""
    by_index = {s.index: s for s in out}

    def sig(si: int) -> tuple:
        e = by_index.get(si)
        clef = e.clef.value.sign if e is not None and e.clef.value is not None else None
        sp = float(getattr(staves[si], "spacing", page_u) or page_u)
        return clef, sp < 0.9 * page_u

    ref = max(systems, key=len)
    ref_sig = [sig(si) for si in ref]
    for members in systems:
        if len(members) >= len(ref):
            continue
        own = [sig(si) for si in members]
        best = None
        for slots in combinations(range(len(ref)), len(members)):
            score = sum((a[0] == b[0] and a[0] is not None) * 2 + (a[1] == b[1]) * 2
                        for a, b in zip(own, (ref_sig[k] for k in slots)))
            score += 0.1 * sum(slots)                # ties: the lower staves (hidden solo parts are on top)
            if best is None or score > best[0]:
                best = (score, slots)
        for si, slot in zip(members, best[1]):
            if si in by_index:
                by_index[si].slot = slot
        warnings.append(f"system with staves {members}: matched to parts {list(best[1])} by clef and size")


def detect_environment(result: dict, removal: StaffRemoval) -> PageEnvironment:
    """Reads systems, clefs, key signatures and time-signature positions.
    Args:
        result: Output of `staff_detect.detect_staves`.
        removal: Output of `staff_remove.remove_from_detection`.
    Returns:
        A `PageEnvironment`.
    """
    staves, u = result["staves"], result["interline"]
    warnings: list[str] = []

    systems, _ = group_systems(result, removal)
    sizes = [len(s) for s in systems]
    consistent = len(set(sizes)) <= 1
    if not consistent:
        warnings.append(f"systems have different staff counts {sizes}: "
                        "slot-to-part mapping needs checking")

    image = _cut_brackets(removal.cleaned, u)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(image.astype(np.uint8), connectivity=8)

    # each component belongs to the staff whose centre is nearest its own
    centres = stats[:, cv2.CC_STAT_TOP] + stats[:, cv2.CC_STAT_HEIGHT] / 2.0
    staff_mid = np.array([(s.top + s.bottom) / 2.0 for s in staves])
    owner = np.argmin(np.abs(centres[:, None] - staff_mid[None, :]), axis=1)
    owner[0] = -1

    out = []
    page_u = u
    for system_index, members in enumerate(systems):
        for slot, si in enumerate(members):
            # each staff at its own size (a small solo staff above a piano part)
            sp = float(getattr(staves[si], "spacing", 0.0) or 0.0)
            u = sp if 0.5 * page_u <= sp <= 1.5 * page_u else page_u
            paths = removal.line_paths[si]
            x0 = int(removal.line_extents[si][:, 0].min())
            glyphs = staff_start_glyphs(image, labels, stats, owner, si, paths, x0, u)

            clef, after_clef = detect_clef(glyphs, paths, u)

            later = [g for g in glyphs if g.x >= after_clef - 0.2 * u]
            pairs = find_stacked_pairs(later, u)
            stacked = {id(g) for pair in pairs for g in pair}

            key, after_key = detect_key(glyphs, clef.value, after_clef, stacked, u)

            time = Found(None, "none", "no time signature here")
            present = False
            if not pairs:
                # digits that touch (e.g. 12 over 8) form one staff-high glyph
                for g in later:
                    if g.x - after_key > TIME_MAX_GAP_U * u:
                        break
                    if id(g) in {id(x) for x in key.boxes} or _is_thin_vertical(g, u):
                        continue
                    fill = g.area / max(g.w * g.h, 1)
                    # bold digits fill 25-65 % of their box; a note with a stem is sparser,
                    # a thick (repeat) barline is denser and narrower
                    if g.top_step <= 0.6 and g.bottom_step >= 7.4 and 3.6 * u <= g.h <= 4.8 * u \
                            and 1.2 * u <= g.w <= 3.6 * u and 0.25 <= fill <= 0.8 \
                            and g.x >= after_key - 0.2 * u \
                            and (_two_wide_halves(g) or _holes(g.mask, 0.05 * u * u) >= 2) \
                            and (_longest_vertical_run(g.mask) <= 3.3 * u or _holes(g.mask, 0.05 * u * u) >= 1
                                 or fill >= 0.4):
                        present = True
                        time = Found(None, "none", "time signature located (one joined glyph); digits not read",
                                     [g.box])
                        break
            for upper, lower in pairs:
                if upper.x - after_key <= TIME_MAX_GAP_U * u:
                    present = True
                    box = (min(upper.x, lower.x), upper.y,
                           max(upper.right, lower.right) - min(upper.x, lower.x),
                           lower.y + lower.h - upper.y)
                    time = Found(None, "none", "time signature located; digits not read "
                                 "(needs the glyph classifier)", [box])
                    break

            out.append(StaffEnvironment(si, system_index, slot, clef, key, time, present))

    out.sort(key=lambda s: s.index)
    if not consistent:
        _align_slots(out, systems, staves, page_u, warnings)
    _harmonise_keys(out, warnings)

    for s in out:
        if s.clef.value is None:
            warnings.append(f"staff {s.index}: clef not found")
        elif s.clef.confidence == "low":
            warnings.append(f"staff {s.index}: clef uncertain ({s.clef.reason})")

    _flag_key_disagreement(out, warnings)

    return PageEnvironment(systems, out, max(sizes) if sizes else 0, consistent, warnings)


# --------------------------------------------------------------------------- #
#  Reporting
# --------------------------------------------------------------------------- #
def key_name(fifths: int | None) -> str:
    """`2` -> `2 sharps (D major / B minor)`."""
    if fifths is None:
        return "unknown"
    majors = ["Cb", "Gb", "Db", "Ab", "Eb", "Bb", "F", "C", "G", "D", "A", "E", "B", "F#", "C#"]
    minors = ["Ab", "Eb", "Bb", "F", "C", "G", "D", "A", "E", "B", "F#", "C#", "G#", "D#", "A#"]
    count = abs(fifths)
    word = "no sharps or flats" if fifths == 0 else \
        f"{count} {'sharp' if fifths > 0 else 'flat'}{'s' if count != 1 else ''}"
    return f"{word} ({majors[fifths + 7]} major / {minors[fifths + 7]} minor)"


def summarise(env: PageEnvironment):
    print(f"systems          : {len(env.systems)}  {[len(s) for s in env.systems]} staves each")
    print()
    print(f"{'staff':>5} {'sys':>3} {'slot':>4}  {'clef':<16} {'key':<36} time")
    for s in env.staves:
        clef = f"{s.clef.value.name} ({s.clef.value.code})" if s.clef.value else "?"
        clef = f"{clef} [{s.clef.confidence[0]}]"
        key = f"{key_name(s.key.value)} [{s.key.confidence[0]}]"
        time = "located, not read" if s.time_present else "-"
        print(f"{s.index:>5} {s.system:>3} {s.slot:>4}  {clef:<16} {key:<36} {time}")
    for w in env.warnings:
        print(f"WARNING: {w}")


def plot_environment(result: dict, env: PageEnvironment, width_in: float = 16.0):
    """The page with systems, clefs, key signatures and time signatures marked."""
    gray = result["gray"]
    height_px, width_px = gray.shape[:2]
    fig, ax = plt.subplots(figsize=(width_in, width_in * height_px / width_px))
    ax.imshow(255 - result["binary"], cmap="gray", vmin=0, vmax=255)

    colours = {"clef": "tab:blue", "key": "tab:orange", "time": "tab:green"}
    staves = result["staves"]
    for system_index, members in enumerate(env.systems):
        top, bottom = staves[members[0]].top, staves[members[-1]].bottom
        ax.plot([6, 6], [top, bottom], color="tab:purple", lw=4)
        ax.text(14, top, f"sys {system_index}", color="tab:purple", fontsize=8, va="bottom")

    for s in env.staves:
        for kind, found in (("clef", s.clef), ("key", s.key), ("time", s.time)):
            for (x, y, w, h) in found.boxes:
                ax.add_patch(mpatches.Rectangle((x, y), w, h, fill=False,
                                                edgecolor=colours[kind], lw=1.2))
        label = s.clef.value.code if s.clef.value else "?"
        if s.key.value:
            label += f" {abs(s.key.value)}{'#' if s.key.value > 0 else 'b'}"
        if s.time_present:
            label += " t"
        y = staves[s.index].top
        x = s.clef.boxes[0][0] if s.clef.boxes else 0
        ax.text(x, y - 0.6 * result["interline"], label, color="tab:red", fontsize=8,
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.75, pad=1))

    handles = [mpatches.Patch(edgecolor=c, facecolor="none", label=k) for k, c in colours.items()]
    handles.append(mpatches.Patch(color="tab:purple", label="system"))
    ax.legend(handles=handles, loc="upper right", fontsize=8)
    ax.set_title("Detected environment")
    ax.axis("off")
    fig.tight_layout()
    return fig


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Read systems, clefs and key signatures from a page")
    parser.add_argument("image", nargs="?", help="page image (default: random from ./Separate_Set)")
    args = parser.parse_args()

    path = args.image
    if path is None:
        random.seed(35)
        files = glob("./Separate_Set/*.png")
        if not files:
            raise SystemExit("No images found in ./Separate_Set/")
        path = random.sample(files, 1)[0]
    print(f"page: {path}\n")

    result = detect_staves(cv2.imread(path, 0))
    removal = remove_from_detection(result)
    env = detect_environment(result, removal)
    summarise(env)
    plot_environment(result, env)
    plt.show()
