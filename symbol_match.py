# =============================================================================
#  symbol_match.py  --  Stage 5: symbol recognition by template matching
#  Version: 1.2.0
#
#  Changelog
#  ---------
#  1.2.0  (2026-10-28)
#    - Each staff is read at its own staff space (staff_detect 3.3.0 finds
#      small staves): templates are scaled per staff, and every distance in
#      steps 3-5 (stems, beams, dots, rests, barlines, ...) uses that staff's
#      U. A violin part printed at 75 % above a piano part was not read.
#    - Octave lines: "8va" (or "8vb") with a dashed line above (below) a
#      staff are found (find_ottavas -> StaffSymbols.ottavas); interpret.py
#      moves the notes under them up (down) an octave.
#    - Rests by shape (shape_rests) when the templates miss them on blurred,
#      enlarged pages: a filled bar touching a line (whole / half rest) or a
#      staff-centred, stemless zigzag (quarter rest), if no other symbol
#      claims that ink.
#    - A stemless "whole note" that is a solid wide rectangle is read as the
#      rest block it is (blurred or small pages).
#    - A notehead match in the middle of a beam (beam ink continuing 1.3U to
#      both sides of its centre) is dropped; staff removal leaves holes in
#      beams that cross lines, which looked like half notes on small staves.
#    - Overlay: only the symbols the music is built from are boxed in colour
#      (half/whole by the hole test, not by template); matches that were
#      found and then rejected (time-signature digits, chord-symbol text,
#      flags) are drawn as thin grey boxes. Before, every match was boxed in
#      colour, so a rejected "3" looked like a recognised note.
#    - Half notes whose thin rim was cut by staff removal (rim on a staff
#      line, common for MuseScore/Leland noteheads and small pages): hollow
#      matches from WEAK_HOLLOW (0.40) up to the threshold are kept when the
#      head has a stem and its rim is closed in the page with staff lines
#      (rim_closed: the paper at its centre is an enclosed, hole-sized
#      region), and another head next to it on the same stem (chord notes a
#      second or third apart, whose rims staff removal cuts; a lone weak
#      match is more often a flag or a digit).
#    - Beam groups: chords whose stem tips are joined by beam ink get the
#      same Chord.beam_group, also across the two staves of a piano system.
#      interpret.py uses it to repair measures whose beam counts were
#      misread (a second voice's beam lying on the group thickens it).
#  1.1.0  (2026-10-25)
#    - Mid-staff clef changes: G and F clef templates (full and change size)
#      matched inside the staff, with an octave "8" looked for below/above a
#      G clef (also for the clef at the start of the staff). Matches inside a
#      clef (and its 8) are discarded.
#    - Fewer false noteheads: a head outside the staff needs the ledger
#      lines its position implies (rejects tempo marks, text, dynamics); a
#      filled head needs a stem; a stemless hollow head must be an isolated
#      component no taller than a notehead (rejects time-signature digits).
#    - Whole vs half rest from which edge of the block touches a staff line
#      (hangs = whole, sits = half), so displaced second-voice rests work.
#    - Search band widened to 6U above/below the staff and notehead steps
#      -12..20 allowed (cross-staff notes with four or five ledger lines).
#      Ledger positions that fall within 1.2U of another staff are skipped.
#    - Ties: thin arcs that start just right of one notehead and end just
#      left of the next notehead of the same staff position, with no chord
#      between them, are returned as tie pairs. Arc pieces cut apart where
#      the arc runs along a staff line are joined first. Searched in the
#      image before ledger-line removal, which would cut arcs above or below
#      the staff.
#    - The gap between the thick and thin strokes of a final (or double)
#      barline is no longer read as a hollow head: a "head" with a
#      staff-high stroke on both sides, each nearer than a stem would be
#      (0.5U), is dropped.
#
#  1.0.0  (2026-10-25)
#    - Ledger line removal: the staff-removal run-length rule applied at the
#      extrapolated ledger positions above and below each staff.
#    - Optional median filter on the image used for matching (MEDIAN_KSIZE;
#      off by default: 3x3 lowered head-type accuracy on the dev pages).
#    - Template matching with normalised cross-correlation
#      (cv2.TM_CCOEFF_NORMED) per staff band, templates rescaled from
#      templates/ (drawn at U_REF) to the page's staff space U.
#    - Peak picking (local maxima above a per-class threshold) and greedy
#      non-maximum suppression across all templates and staff bands;
#      noteheads suppress each other only when their centres nearly coincide
#      (chord notes a third apart touch).
#    - Music starts after the clef, a confident key signature and the time
#      signature boxes reported by env_detect.
#    - Head type (filled / hollow) from a grey-level hole test: lightness
#      inside the head, measured beside the staff line when the head sits on
#      one. Falls back to the template class without a grey image.
#    - Filled heads must be ~1U tall at their centre (rejects beam ends).
#    - Notehead analysis on the staff-free image: stem search beside the head,
#      beam/flag count near the stem tip, augmentation dots from the
#      connected regions of detect.py, accidental-to-notehead association.
#    - Barlines: columns whose ink run crosses the whole staff, away from
#      noteheads, grouped into thin strokes (robust to a beam touching).
#    - Staff assignment by pitch anchor; a notehead between two staves goes
#      to the staff whose ledger lines are present beside it.
#    - Rests and accidentals must be isolated connected components
#      (rejects beams, stems and flags), checked before NMS.
#    - Whole vs half rest from the rest block's position on the staff.
#
#  Depends on staff_detect.py >= 3.2.0, staff_remove.py >= 1.1.0,
#  env_detect.py >= 1.0.0, detect.py (kaushik_detection)
# =============================================================================

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field

import cv2
import numpy as np

from env_detect import PageEnvironment, y_to_step
from staff_remove import StaffRemoval, vertical_run_maps

# --------------------------------------------------------------------------- #
#  Tunables -- lengths in units of U (staff line pitch)
# --------------------------------------------------------------------------- #
THRESHOLDS = {                 # minimum NCC score per template class (tuned on dev pages 100-111)
    "notehead_filled": 0.64,
    "notehead_hollow": 0.50,
    "rest_block": 0.70,
    "rest_quarter": 0.55,
    "rest_eighth": 0.55,
    "rest_sixteenth": 0.55,
    "sharp": 0.50,
    "flat": 0.50,
    "natural": 0.50,
    "clef_g": 0.50,
    "clef_f": 0.55,
}
BAND_ABOVE_U = 6.0             # search this far above the top line ...
BAND_BELOW_U = 6.0             # ... and below the bottom line (ledger notes; the
                               #     ledger-line check rejects anything unsupported)
PEAK_RADIUS_U = 0.35           # local-maximum neighbourhood
NMS_IOU = 0.30                 # suppress a box overlapping a better one by more than this ...
NMS_IOS = 0.60                 # ... or covering this share of the smaller box
NMS_HEAD_DX_U = 0.6            # two noteheads closer than this in x ...
NMS_HEAD_DY_U = 0.6            # ... and this in y are the same notehead
LEDGER_STEPS = (-2, -4, -6, -8, -10, 10, 12, 14, 16, 18)
HEADER_PAD_U = 0.3             # skip this much after the clef/key/time boxes
MEDIAN_KSIZE = 1               # median filter on the matching image (1 = off; 3 tested, no gain)

STEM_ZONE_U = (0.30, 1.00)     # stem searched this far either side of the head centre
STEM_MIN_U = 2.0               # a stem extends at least this far past the head centre
STEM_ROWS_U = 0.35             # stem run must pass within this of the head centre row
BEAM_OFFSETS_U = (0.35, 0.55)  # columns beside the stem sampled for beams/flags
BEAM_RUN_U = (0.20, 0.85)      # an ink run of this height counts as one beam/flag
BEAM_CLEAR_U = 0.9             # stop the beam window this far before the nearest head
DOT_MAX_U = 0.6                # augmentation dot: at most this wide/high ...
DOT_MIN_AREA_U2 = 0.04         # ... and at least this much ink
DOT_DX_U = (0.0, 1.4)          # gap between the chord's right edge and the dot
DOT_DY_U = 0.8                 # vertical distance from some notehead centre
ACC_DX_U = (0.7, 2.6)          # accidental centre this far left of the notehead centre
ACC_DSTEP = 1.0                # and within this many steps of its pitch
BAR_MAX_W_U = 0.6              # barline: thin ...
BAR_REACH_U = 0.6              # ... reaching the top and bottom lines within this
CHORD_STEM_DX_U = 0.35         # heads on stems this close (x) form one chord
HOLE_MIN = 0.05                # hole score above this = hollow head (grey-level test)
FILLED_MIN_RUN_U = 0.6         # a filled head is this tall at its centre (a beam end is ~0.5U)
HOLE_HALF_W_U = 0.15           # hole window half-width
HOLE_BAND_U = (0.04, 0.22)     # hole window rows beside a line through the head
LEDGER_MIN_SUPPORT = 0.5       # a head beyond the staff needs this share of its ledger lines
CLEF_G_STEPS = (2.5, 6.5)      # centre of a G clef (any size) lies between these steps
CLEF_F_STEPS = (1.0, 4.5)      # centre of an F clef
WEAK_HOLLOW = 0.40             # hollow matches from here up to the threshold need a stem and a closed rim
RIM_BOX_U = (0.9, 0.75)        # the hole of a weak hollow match must stay inside this half-size box ...
RIM_HOLE_MIN_U2, RIM_HOLE_MAX_U2 = 0.02, 0.45  # ... and have a hole-sized area (U^2) ...
RIM_HOLE_MAX_H_U = 0.75        # ... and height (a staff space between a stem and a flag is taller)
OTTAVA_BAND_U = (1.2, 8.0)     # octave lines lie this far above / below the staff (U)
OTTAVA_GAP_U = 1.6             # dashes of one line are at most this far apart
OTTAVA_MIN_DASHES = 4
BEAM_SIDE_U = 1.3              # a 'head' with beam ink this far out on both sides is part of a beam
BEAM_LINK_TOL_U = 0.45         # beam ink this close to the line between two stem tips ...
BEAM_LINK_FILL = 0.9           # ... along this share of it joins the two chords
BEAM_LINK_MAX_DX_U = 6.0       # neighbouring stems of one beam are at most this far apart
BEAM_LINK_MAX_DY_U = 6.0
BAR_SIDE_RUN_U = 3.5           # strokes this long on both sides of a 'head': a double barline
OCTAVE_ROWS_U = 0.4            # ink rows needed just below/above a G clef box for an "8"
TIE_MIN_W_U = 0.9              # tie arc: at least this wide ...
TIE_MAX_H_U = 1.3              # ... at most this high (+8 % of its width: long ties arch more) ...
TIE_MAX_FILL = 0.80            # ... and thin (ink / bounding box)
TIE_REACH_U = 1.6              # arc end at most this far from the notehead centre (x)
TIE_JOIN_GAP_U = 0.6           # pieces of one arc cut apart by staff removal
TIE_DY_U = 1.0                 # ... and this far vertically

TEMPLATE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "templates")


# --------------------------------------------------------------------------- #
#  Records
# --------------------------------------------------------------------------- #
@dataclass
class Template:
    name: str
    cls: str
    image: np.ndarray            # float32, ink = 1, at the page scale
    anchor_y: float              # pitch anchor row inside the template (px)


@dataclass
class Match:
    cls: str
    name: str
    score: float
    x: int
    y: int
    w: int
    h: int
    staff: int
    anchor_y: float              # absolute y of the pitch anchor
    u: float = 0.0               # staff space of the staff it was matched on (small staves differ)

    @property
    def cx(self) -> float:
        return self.x + self.w / 2

    @property
    def cy(self) -> float:
        return self.y + self.h / 2


@dataclass
class Head:
    match: Match
    step: int                    # half-spaces below the top line
    stem: tuple | None = None    # (x, top, bottom, "up" | "down")
    accidental: str | None = None
    hollow: bool = False


@dataclass
class Chord:
    """Noteheads sounding together (one stem, or one stemless whole note)."""
    heads: list
    x: float
    hollow: bool
    stem: tuple | None
    beams: int = 0
    dots: int = 0
    beam_group: int = -1          # chords joined by one beam (across staves too) share this id

    @property
    def base_ticks(self) -> int:
        if self.hollow:
            return 960 if self.stem else 1920
        return 480 >> min(self.beams, 3)

    @property
    def ticks(self) -> int:
        t = self.base_ticks
        return int(t * (1.5 if self.dots == 1 else 1.75 if self.dots >= 2 else 1))


@dataclass
class Rest:
    match: Match
    kind: str                    # whole | half | quarter | eighth | sixteenth
    x: float


@dataclass
class StaffSymbols:
    index: int
    start_x: float
    chords: list = field(default_factory=list)
    rests: list = field(default_factory=list)
    barlines: list = field(default_factory=list)
    accidentals: list = field(default_factory=list)     # Match, including unattached ones
    clef_changes: list = field(default_factory=list)    # (x, clef code, octave shift) inside the staff
    start_octave: int = 0                               # octave mark on the staff's opening clef
    ottavas: list = field(default_factory=list)         # (x from, x to, +1 for 8va / -1 for 8vb)


@dataclass
class Recognition:
    staves: list                 # StaffSymbols per staff
    matches: list                # every Match kept after NMS
    work: np.ndarray             # staff- and ledger-free image used for matching (ink = 255)
    u: float
    ties: list = field(default_factory=list)            # (Head, Head) pairs
    clefs: list = field(default_factory=list)           # clef Matches inside staves


# --------------------------------------------------------------------------- #
#  Templates
# --------------------------------------------------------------------------- #
def load_templates(u: float, folder: str = TEMPLATE_DIR) -> list[Template]:
    """Loads the bank from `folder` and rescales it from U_REF to `u`."""
    with open(os.path.join(folder, "templates.json")) as f:
        index = json.load(f)
    scale = u / float(index["u_ref"])
    out = []
    for t in index["templates"]:
        img = cv2.imread(os.path.join(folder, t["file"]), cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(os.path.join(folder, t["file"]))
        size = (max(3, int(round(img.shape[1] * scale))), max(3, int(round(img.shape[0] * scale))))
        small = cv2.resize(img, size, interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
        out.append(Template(t["name"], t["cls"], small, float(t["anchor_y"]) * scale))
    return out


# --------------------------------------------------------------------------- #
#  Ledger lines
# --------------------------------------------------------------------------- #
def _path_at(paths: np.ndarray, step: float) -> np.ndarray:
    """y of a (possibly extrapolated) staff step at every column."""
    spacing = (paths[-1] - paths[0]) / (paths.shape[0] - 1)
    return paths[0] + spacing * step / 2.0


def remove_ledgers(cleaned: np.ndarray, result: dict, removal: StaffRemoval) -> np.ndarray:
    """Erases thin horizontal ink at ledger-line positions, keeping anything
    whose vertical run is longer than a line (noteheads, stems, beams)."""
    ink = cleaned > 0
    height, width = ink.shape
    _, run_length, _ = vertical_run_maps(ink)
    t = float(result["metrics"].line_thickness)
    half = t / 2 + 1.0
    erase = np.zeros_like(ink)
    u = float(result["interline"])
    spans = [(float(np.median(p[0])) - 1.2 * u, float(np.median(p[-1])) + 1.2 * u) for p in removal.line_paths]
    for si, paths in enumerate(removal.line_paths):
        x_lo = int(removal.line_extents[si][:, 0].min())
        x_hi = int(removal.line_extents[si][:, 1].max())
        xs = np.arange(x_lo, x_hi + 1)
        for step in LEDGER_STEPS:
            ys = _path_at(paths, step)[xs]
            ym = float(np.median(ys))
            if any(lo <= ym <= hi for k, (lo, hi) in enumerate(spans) if k != si):
                continue                      # that position belongs to a neighbouring staff
            for off in np.arange(-np.ceil(half), np.ceil(half) + 1):
                rows = np.round(ys + off).astype(int)
                ok = (rows >= 0) & (rows < height) & (np.abs(rows - ys) <= half)
                r, c = rows[ok], xs[ok]
                hit = ink[r, c] & (run_length[r, c] <= removal.max_run)
                erase[r[hit], c[hit]] = True
    out = cleaned.copy()
    out[erase] = 0
    return out


# --------------------------------------------------------------------------- #
#  Matching
# --------------------------------------------------------------------------- #
def _peaks(score: np.ndarray, threshold: float, radius: int) -> tuple[np.ndarray, np.ndarray]:
    score = np.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0)
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (2 * radius + 1, 2 * radius + 1))
    local_max = cv2.dilate(score, k)
    ys, xs = np.nonzero((score >= threshold) & (score >= local_max - 1e-6))
    return ys, xs


def match_staff(work: np.ndarray, staff_index: int, paths: np.ndarray, x_lo: int, x_hi: int,
                templates: list[Template], u: float, thresholds: dict = THRESHOLDS) -> list[Match]:
    """All template peaks in one staff's band."""
    height = work.shape[0]
    top = int(max(0, np.floor(paths[0, x_lo:x_hi + 1].min() - BAND_ABOVE_U * u)))
    bottom = int(min(height, np.ceil(paths[-1, x_lo:x_hi + 1].max() + BAND_BELOW_U * u)))
    band = (work[top:bottom, x_lo:x_hi + 1] > 0).astype(np.float32)
    radius = max(1, int(round(PEAK_RADIUS_U * u)))
    found = []
    for t in templates:
        th, tw = t.image.shape
        if band.shape[0] < th or band.shape[1] < tw:
            continue
        score = cv2.matchTemplate(band, t.image, cv2.TM_CCOEFF_NORMED)
        ys, xs = _peaks(score, thresholds[t.cls], radius)
        for y, x in zip(ys, xs):
            found.append(Match(t.cls, t.name, float(score[y, x]), int(x_lo + x), int(top + y), tw, th,
                               staff_index, float(top + y + t.anchor_y), u))
    return found


def _overlap(a: Match, b: Match) -> tuple[float, float]:
    ix = max(0, min(a.x + a.w, b.x + b.w) - max(a.x, b.x))
    iy = max(0, min(a.y + a.h, b.y + b.h) - max(a.y, b.y))
    inter = ix * iy
    if inter == 0:
        return 0.0, 0.0
    union = a.w * a.h + b.w * b.h - inter
    return inter / union, inter / min(a.w * a.h, b.w * b.h)


def _suppresses(k: Match, m: Match, u: float) -> bool:
    """Whether kept match `k` suppresses candidate `m`. Two noteheads clash
    only when their centres nearly coincide, because chord notes a third
    apart touch and their boxes overlap; everything else uses box overlap."""
    if k.cls.startswith("notehead") and m.cls.startswith("notehead"):
        us = min(k.u or u, m.u or u)
        return abs(k.cx - m.cx) < NMS_HEAD_DX_U * us and abs(k.cy - m.cy) < NMS_HEAD_DY_U * us
    iou, ios = _overlap(m, k)
    return iou > NMS_IOU or ios > NMS_IOS


def non_max_suppression(matches: list[Match], u: float) -> list[Match]:
    """Greedy NMS: best score first; drop anything a kept match suppresses."""
    kept: list[Match] = []
    for m in sorted(matches, key=lambda m: -m.score):
        if not any(_suppresses(k, m, u) for k in kept):
            kept.append(m)
    return kept


# --------------------------------------------------------------------------- #
#  Notehead analysis
# --------------------------------------------------------------------------- #
def find_stem(run_length: np.ndarray, run_centre: np.ndarray, ink: np.ndarray,
              cx: float, cy: float, u: float) -> tuple | None:
    """Longest vertical ink run beside the head that passes near its centre row.
    Returns (x, top, bottom, direction) or None."""
    height, width = ink.shape
    r0 = int(max(0, round(cy - STEM_ROWS_U * u)))
    r1 = int(min(height - 1, round(cy + STEM_ROWS_U * u)))
    best = None
    for side in (+1, -1):
        c0 = int(round(cx + side * STEM_ZONE_U[0] * u))
        c1 = int(round(cx + side * STEM_ZONE_U[1] * u))
        for c in range(min(c0, c1), max(c0, c1) + 1):
            if not 0 <= c < width:
                continue
            rows = np.arange(r0, r1 + 1)
            rows = rows[ink[rows, c]]
            if rows.size == 0:
                continue
            k = rows[np.argmax(run_length[rows, c])]
            length = run_length[k, c]
            top = run_centre[k, c] - (length - 1) / 2
            bottom = run_centre[k, c] + (length - 1) / 2
            up, down = cy - top, bottom - cy
            reach = max(up, down)
            if reach < STEM_MIN_U * u:
                continue
            direction = "up" if up >= down else "down"
            if best is None or reach > best[0]:
                best = (reach, (float(c), float(top), float(bottom), direction))
    return best[1] if best else None


def count_beams(ink: np.ndarray, stem: tuple, head_ys: list[float], u: float) -> int:
    """Beams or flags crossing columns just beside the stem, near its tip."""
    x, top, bottom, direction = stem
    height, width = ink.shape
    if direction == "up":
        r0, r1 = top, min(head_ys) - BEAM_CLEAR_U * u
    else:
        r0, r1 = max(head_ys) + BEAM_CLEAR_U * u, bottom
    r0, r1 = int(max(0, np.floor(r0 - 0.1 * u))), int(min(height - 1, np.ceil(r1 + 0.1 * u)))
    if r1 <= r0:
        return 0
    lo, hi = BEAM_RUN_U[0] * u, BEAM_RUN_U[1] * u
    best = 0
    for side in (+1, -1):
        counts = []
        for off in BEAM_OFFSETS_U:
            c = int(round(x + side * off * u))
            if not 0 <= c < width:
                counts.append(0)
                continue
            col = ink[r0:r1 + 1, c].astype(np.int8)
            edges = np.diff(np.r_[0, col, 0])
            starts, ends = np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)
            lengths = ends - starts
            counts.append(int(np.count_nonzero((lengths >= lo) & (lengths <= hi))))
        best = max(best, min(counts))      # both columns must agree
    return min(best, 3)


def _count_holes(mask: np.ndarray, min_area: float) -> int:
    bg = np.pad(~mask, 1, constant_values=True).astype(np.uint8)
    n, lab, st, _ = cv2.connectedComponentsWithStats(bg, connectivity=4)
    outside = lab[0, 0]
    return sum(1 for k in range(1, n) if k != outside and st[k, cv2.CC_STAT_AREA] >= min_area)


def find_ottavas(labels: np.ndarray, stats: np.ndarray, paths: np.ndarray, u: float,
                 x_lo: float, x_hi: float) -> list[tuple[float, float, int]]:
    """Octave lines: an "8" (8va / 8vb) followed by a dashed line, above the
    staff (notes sound an octave higher) or below it (an octave lower).

    Dashes are short flat components in a row (gaps up to OTTAVA_GAP_U).
    The text before them must have the loops of "8va"/"8vb", or the line
    must end in a hook towards the staff; either tells it from "rit. - - -"
    or "cresc. - - -"."""
    top = float(np.median(paths[0]))
    bottom = float(np.median(paths[-1]))
    dashes = []
    for c in range(1, stats.shape[0]):
        x, y, w, h, area = (int(v) for v in stats[c])
        if not (x_lo - 2 * u <= x <= x_hi):
            continue
        cy = y + h / 2
        side = +1 if top - OTTAVA_BAND_U[1] * u <= cy <= top - OTTAVA_BAND_U[0] * u else \
            -1 if bottom + OTTAVA_BAND_U[0] * u <= cy <= bottom + OTTAVA_BAND_U[1] * u else 0
        if side and 0.15 * u <= w <= 1.2 * u and h <= 0.35 * u and w >= 1.5 * h:
            dashes.append((x, cy, w, side))
    dashes.sort()
    used, found = set(), []
    for i, (x, cy, w, side) in enumerate(dashes):
        if i in used:
            continue
        chain, end = [i], x + w
        for j in range(i + 1, len(dashes)):
            xj, cyj, wj, sj = dashes[j]
            if j in used or sj != side or abs(cyj - cy) > 0.3 * u:
                continue
            if xj - end > OTTAVA_GAP_U * u:
                break
            chain.append(j)
            end = xj + wj
        if len(chain) < OTTAVA_MIN_DASHES:
            continue
        used.update(chain)
        # the text before the dashes: "8va", "8vb" or "8" has an 8 (two
        # loops) and usually an "a" or "b": at least 3 loops in all, or 2 in
        # one glyph. "rit.", "dim.", "cresc.", "accel." have fewer.
        start, loops, best_one = None, 0, 0
        for c in range(1, stats.shape[0]):
            x, y, w, h, area = (int(v) for v in stats[c])
            if not (dashes[i][0] - 4 * u <= x + w <= dashes[i][0] + 0.2 * u):
                continue
            if not (0.3 * u <= h <= 2.4 * u and abs(y + h - cy) <= 1.6 * u):
                continue
            k = _count_holes(labels[y:y + h, x:x + w] == c, 1)
            loops += k
            best_one = max(best_one, k)
            start = x if start is None else min(start, x)
        # ... or the line ends in a hook towards the staff (8va/8vb lines do,
        # "rit. - - -" does not): small italic text loses its loops
        hook = False
        for c in range(1, stats.shape[0]):
            x, y, w, h, area = (int(v) for v in stats[c])
            if not (end - 0.3 * u <= x + w and x <= end + 1.2 * u and w <= 1.4 * u and 0.4 * u <= h <= 1.6 * u):
                continue
            if side > 0 and abs(y - cy) <= 0.4 * u or side < 0 and abs(y + h - cy) <= 0.4 * u:
                hook = True
                end = max(end, x + w)
        if start is not None and (loops >= 3 or best_one >= 2 or hook):
            found.append((float(start), float(end + 0.3 * u), side))
    return found


def link_beams(chords: list[Chord], ink: np.ndarray, u: float) -> None:
    """Gives chords whose stem tips are joined by beam ink the same beam_group.

    The straight line between two neighbouring stem tips runs along the beam
    that joins them; it is accepted when ink lies within BEAM_LINK_TOL_U of
    it at BEAM_LINK_FILL of the sampled columns. Works across the two staves
    of a piano system (cross-staff beams)."""
    stemmed = [c for c in chords if c.stem is not None and not c.hollow]
    # both ends of every stem: a stem shared by two voices (one head at each
    # end) reaches the beam at the end that is not its "direction"
    tips, owner = [], []
    for k, c in enumerate(stemmed):
        x, top, bottom, direction = c.stem
        tips += [(float(x), float(top)), (float(x), float(bottom))]
        owner += [k, k]
    order = sorted(range(len(tips)), key=lambda i: tips[i][0])
    parent = list(range(len(stemmed)))

    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    h, w = ink.shape
    tol = int(round(BEAM_LINK_TOL_U * u))
    for a_pos, i in enumerate(order):
        xi, yi = tips[i]
        for j in order[a_pos + 1:]:
            xj, yj = tips[j]
            if xj - xi > BEAM_LINK_MAX_DX_U * u:
                break
            if xj - xi < 0.3 * u or abs(yj - yi) > BEAM_LINK_MAX_DY_U * u:
                continue
            xs = np.arange(int(np.ceil(xi)), int(np.floor(xj)) + 1, 2)
            if xs.size < 3:
                continue
            ys = yi + (xs - xi) * (yj - yi) / (xj - xi)
            hit = 0
            for x_, y_ in zip(xs, ys):
                r0, r1 = int(max(0, y_ - tol)), int(min(h - 1, y_ + tol)) + 1
                hit += bool(0 <= x_ < w and ink[r0:r1, x_].any())
            if hit >= BEAM_LINK_FILL * xs.size and owner[i] != owner[j]:
                parent[root(owner[j])] = root(owner[i])
                break                      # nearest beamed neighbour only
    for i, c in enumerate(stemmed):
        c.beam_group = root(i)
    sizes: dict[int, int] = {}
    for c in stemmed:
        sizes[c.beam_group] = sizes.get(c.beam_group, 0) + 1
    for c in stemmed:
        if sizes[c.beam_group] < 2:
            c.beam_group = -1


def _dots_for(chord: Chord, heads: list[Head], dot_regions: list, u: float) -> int:
    # template boxes carry a ~0.3U empty margin, so the ink ends ~0.3U inside
    right = max(h.match.x + h.match.w for h in heads) - 0.3 * u
    n = 0
    for r in dot_regions:
        rcy = r.y + r.height / 2
        dx = r.x - right
        if not (DOT_DX_U[0] * u <= dx <= DOT_DX_U[1] * u):
            continue
        if any(abs(rcy - h.match.cy) <= DOT_DY_U * u for h in heads):
            n = max(n, 1)
    return n


def hole_score(gray: np.ndarray, levels: tuple[float, float], cx: float, cy: float, line_y: float | None,
               t: float, u: float) -> float:
    """How light the inside of a notehead is, on a 0 (ink) .. 1 (paper) scale.

    For a head in a space the window is its centre. For a head on a line
    (or ledger line) the centre is the line itself, so two thin windows just
    above and below the line are used. Grey levels survive blur that closes
    the hole in the binary image.
    """
    ink, paper = levels
    h, w = gray.shape
    x0, x1 = int(round(cx - HOLE_HALF_W_U * u)), int(round(cx + HOLE_HALF_W_U * u)) + 1
    x0, x1 = max(0, x0), min(w, x1)
    if line_y is None:
        y0, y1 = int(round(cy - 0.1 * u)), int(round(cy + 0.1 * u)) + 1
        win = gray[max(0, y0):min(h, y1), x0:x1].ravel()
    else:
        a0, a1 = line_y - t / 2 - HOLE_BAND_U[1] * u, line_y - t / 2 - HOLE_BAND_U[0] * u
        b0, b1 = line_y + t / 2 + HOLE_BAND_U[0] * u, line_y + t / 2 + HOLE_BAND_U[1] * u
        win = np.concatenate([gray[max(0, int(round(a0))):min(h, int(round(a1)) + 1), x0:x1].ravel(),
                              gray[max(0, int(round(b0))):min(h, int(round(b1)) + 1), x0:x1].ravel()])
    if win.size == 0:
        return 0.0
    return float((win.astype(np.float32).mean() - ink) / max(paper - ink, 1.0))


def ledger_support(cleaned: np.ndarray, paths: np.ndarray, m: Match, step: int, t: float, u: float) -> float:
    """Share of the ledger lines a head at `step` needs that are actually
    there, looked for just beyond the head's left and right edges."""
    if -1 <= step <= 9:
        return 1.0
    needed = range(-2, step - 1, -2) if step < 0 else range(10, step + 1, 2)
    needed = list(needed)
    if not needed:
        return 1.0
    height, width = cleaned.shape
    half_ink = max(0.3 * u, m.w / 2 - 0.3 * u)          # template margin is ~0.3U
    cols = [int(round(m.cx + s * (half_ink + 0.15 * u))) for s in (-1, 1)]
    reach = int(np.ceil(t / 2 + 1.5))
    found = 0
    for st in needed:
        hit = False
        for c in cols:
            if not 0 <= c < width:
                continue
            y = float(_path_at(paths, st)[c])
            r0, r1 = int(max(0, np.floor(y - reach))), int(min(height, np.ceil(y + reach) + 1))
            hit |= bool(cleaned[r0:r1, c].any())
        found += hit
    return found / len(needed)


def is_isolated(labels: np.ndarray, stats: np.ndarray, m: Match, u: float, pad_u: float = 0.35) -> bool:
    """True if the ink under the centre of `m` belongs to a component that
    fits inside the match box (plus a pad). Rejects beams, stems, noteheads."""
    x0, y0 = int(m.x + 0.3 * m.w), int(m.y + 0.3 * m.h)
    x1, y1 = int(m.x + 0.7 * m.w) + 1, int(m.y + 0.7 * m.h) + 1
    patch = labels[y0:y1, x0:x1]
    ids = patch[patch > 0]
    if ids.size == 0:
        return False
    c = int(np.bincount(ids).argmax())
    x, y, w, h = stats[c, :4]
    pad = pad_u * u
    return x >= m.x - pad and y >= m.y - pad and x + w <= m.x + m.w + pad and y + h <= m.y + m.h + pad


def find_barlines(ink: np.ndarray, run_length: np.ndarray, run_centre: np.ndarray, paths: np.ndarray,
                  x_lo: float, x_hi: float, head_xs: list[float], u: float) -> list[float]:
    """Columns whose ink run crosses the whole staff, away from noteheads,
    grouped into thin strokes."""
    height, width = ink.shape
    reach = BAR_REACH_U * u
    cols = []
    for x in range(int(max(0, x_lo)), int(min(width - 1, x_hi)) + 1):
        if any(abs(x - hx) <= 1.1 * u for hx in head_xs):
            continue
        r = int(round(paths[2, x]))
        if not (0 <= r < height and ink[r, x]):
            continue
        length = run_length[r, x]
        top = run_centre[r, x] - (length - 1) / 2
        bottom = run_centre[r, x] + (length - 1) / 2
        if top <= paths[0, x] + reach and bottom >= paths[-1, x] - reach:
            cols.append(x)
    bars, group = [], []
    for x in cols + [None]:
        if x is not None and group and x - group[-1] <= 1:
            group.append(x)
            continue
        if group and (group[-1] - group[0] + 1) <= BAR_MAX_W_U * u:
            xb = (group[0] + group[-1]) / 2
            if not bars or xb - bars[-1] > 1.5 * u:     # double bars count once
                bars.append(xb)
        group = [x] if x is not None else []
    return bars


def octave_mark(ink: np.ndarray, box: tuple, u: float) -> int:
    """-1 for a small 8 just below a G clef's box (treble 8 basso), +1 for one
    just above it, else 0. The 8 often touches the clef's tail, so ink rows
    are counted in a narrow window under (or over) the box instead of
    looking for a separate component."""
    x, y, w, h = box
    cx = x + w / 2
    c0, c1 = int(max(0, cx - 0.6 * u)), int(min(ink.shape[1], cx + 0.6 * u))

    def rows_with_ink(y0: float, y1: float) -> int:
        r0, r1 = int(max(0, y0)), int(min(ink.shape[0], y1))
        if r1 <= r0:
            return 0
        return int(np.count_nonzero(ink[r0:r1, c0:c1].sum(axis=1) >= 3))

    below = rows_with_ink(y + h - 0.2 * u, y + h + 0.6 * u)
    above = rows_with_ink(y - 0.6 * u, y + 0.2 * u)
    need = OCTAVE_ROWS_U * u
    if below >= need:
        return -1
    if above >= need:
        return 1
    return 0


def clef_code(m: Match, step: float) -> str:
    if m.cls == "clef_g":
        return "G2"
    line = int(np.clip(round(5 - (step - 0.5) / 2), 3, 5))   # F clef: dots straddle its line
    return f"F{line if line in (3, 4) else 4}"


def _vrun(mask: np.ndarray) -> int:
    best = 0
    for col in mask.T:
        d = np.diff(np.concatenate(([0], col.astype(np.int8), [0])))
        st, en = np.flatnonzero(d == 1), np.flatnonzero(d == -1)
        if st.size:
            best = max(best, int((en - st).max()))
    return best


def _square_corners(mask: np.ndarray) -> int:
    """How many of the four corners of a glyph's box are mostly ink (a bar
    fills them, an oval notehead leaves all four empty)."""
    h, w = mask.shape
    ch, cw = max(1, int(0.25 * h)), max(1, int(0.15 * w))
    corners = [mask[:ch, :cw], mask[:ch, -cw:], mask[-ch:, :cw], mask[-ch:, -cw:]]
    return sum(1 for c in corners if c.mean() >= 0.5)


def shape_rests(labels: np.ndarray, stats: np.ndarray, paths: np.ndarray, u: float, x_lo: float, x_hi: float,
                taken: list) -> list[tuple[str, tuple]]:
    """Rests recognised by shape alone (see step 4b of recognise)."""
    out = []
    for c in range(1, stats.shape[0]):
        x, y, w, h, area = (int(v) for v in stats[c])
        cx, cy = x + w / 2, y + h / 2
        if not (x_lo <= cx <= x_hi):
            continue
        if any(bx - 0.2 * u <= cx <= bx + bw + 0.2 * u and by - 0.2 * u <= cy <= by + bh + 0.2 * u
               for bx, by, bw, bh in taken):
            continue
        step = y_to_step(paths, cx, cy)
        if not -1 <= step <= 9:
            continue
        fill = area / max(w * h, 1)
        if 0.9 * u <= w <= 1.8 * u and 0.3 * u <= h <= 1.15 * u and fill >= 0.65 \
                and _square_corners(labels[y:y + h, x:x + w] == c) >= 2:   # a bar, not an oval notehead
            col = int(np.clip(round(cx), 0, paths.shape[1] - 1))
            ys = paths[:, col]
            d_top = float(np.min(np.abs(ys - y)))
            d_bottom = float(np.min(np.abs(ys - (y + h - 1))))
            if min(d_top, d_bottom) <= 0.35 * u:              # touches a line
                out.append(("whole" if d_top < d_bottom else "half", (x, y, w, h)))
        elif 2.2 * u <= h <= 3.3 * u and 0.5 * u <= w <= 1.3 * u and 3.0 <= step <= 5.5 and 0.25 <= fill <= 0.65:
            if _vrun(labels[y:y + h, x:x + w] == c) < 0.55 * h:  # no stem (a note, a sharp, a flat has one)
                out.append(("quarter", (x, y, w, h)))
    return out


def solid_block(labels: np.ndarray, stats: np.ndarray, m: Match, u: float) -> bool:
    """The component under a match is a filled, wide rectangle (a rest block)."""
    x0, y0 = int(m.cx - 0.2 * u), int(m.cy - 0.1 * u)
    patch = labels[max(0, y0):y0 + max(1, int(0.2 * u)) + 1, max(0, x0):x0 + max(1, int(0.4 * u)) + 1]
    ids = patch[patch > 0]
    if ids.size == 0:
        return False
    c = int(np.bincount(ids).argmax())
    x, y, w, h, area = (int(v) for v in stats[c])
    return area >= 0.85 * w * h and w >= 1.5 * h and h <= 0.75 * u


def block_rest_kind(labels: np.ndarray, stats: np.ndarray, paths: np.ndarray, m: Match, step: float) -> str:
    """A whole rest hangs from a staff line, a half rest sits on one. Rests
    of a second voice are moved up or down, so the side of the block that
    touches a line decides, not its height on the staff."""
    x0, y0 = int(m.x + 0.3 * m.w), int(m.y + 0.3 * m.h)
    patch = labels[y0:int(m.y + 0.7 * m.h) + 1, x0:int(m.x + 0.7 * m.w) + 1]
    ids = patch[patch > 0]
    if ids.size == 0:
        return "whole" if step < 3.0 else "half"
    c = int(np.bincount(ids).argmax())
    top, bottom = stats[c, 1], stats[c, 1] + stats[c, 3] - 1
    col = int(np.clip(round(m.cx), 0, paths.shape[1] - 1))
    ys = paths[:, col]
    d_top = float(np.min(np.abs(ys - top)))
    d_bottom = float(np.min(np.abs(ys - bottom)))
    return "whole" if d_top < d_bottom else "half"


def find_ties(labels: np.ndarray, stats: np.ndarray, run_length: np.ndarray, ink: np.ndarray,
              staves_heads: list[list], chords_x: list[list[float]], u: float) -> list[tuple]:
    """Thin arcs joining two noteheads at the same staff position.

    Staff removal cuts an arc where it runs along a staff line, so thin
    pieces that follow each other (gap <= TIE_JOIN_GAP_U) are joined first."""
    pieces = []
    for c in range(1, stats.shape[0]):
        x, y, w, h, area = (int(v) for v in stats[c])
        if h > TIE_MAX_H_U * u + 0.08 * max(w, 1) or w > 8 * u:
            continue
        mask = labels[y:y + h, x:x + w] == c
        if run_length[y:y + h, x:x + w][mask].max() > 0.45 * u:       # thin stroke only
            continue
        pieces.append((x, y, w, h, area, c))
    # join pieces into chains, left to right
    pieces.sort()
    parent = list(range(len(pieces)))

    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    for i, (xi, yi, wi, hi, _, _) in enumerate(pieces):
        for j in range(i + 1, len(pieces)):
            xj, yj, wj, hj, _, _ = pieces[j]
            if xj - (xi + wi) > TIE_JOIN_GAP_U * u:
                break
            if max(yi, yj) - min(yi + hi, yj + hj) <= 0.35 * u:
                parent[root(j)] = root(i)
    chains: dict[int, list] = {}
    for i in range(len(pieces)):
        chains.setdefault(root(i), []).append(pieces[i])

    ties = []
    for chain in chains.values():
        # specks of noise at either end would move the arc's end points
        def speck(p):
            return p[4] <= 3
        chain.sort()
        while chain and speck(chain[0]):
            chain.pop(0)
        while chain and speck(max(chain, key=lambda p: p[0] + p[2])):
            chain.remove(max(chain, key=lambda p: p[0] + p[2]))
        if not chain:
            continue
        x = min(p[0] for p in chain)
        y = min(p[1] for p in chain)
        w = max(p[0] + p[2] for p in chain) - x
        h = max(p[1] + p[3] for p in chain) - y
        area = sum(p[4] for p in chain)
        if w < TIE_MIN_W_U * u or h > TIE_MAX_H_U * u + 0.08 * w or h < 0.15 * u or area > TIE_MAX_FILL * w * h:
            continue
        first, last = chain[0], max(chain, key=lambda p: p[0] + p[2])
        lm = labels[first[1]:first[1] + first[3], first[0]:first[0] + first[2]] == first[5]
        rm = labels[last[1]:last[1] + last[3], last[0]:last[0] + last[2]] == last[5]
        lcols, rcols = np.flatnonzero(lm.any(axis=0)), np.flatnonzero(rm.any(axis=0))
        xl, yl = first[0] + lcols[0], first[1] + np.flatnonzero(lm[:, lcols[:2]].any(axis=1)).mean()
        xr, yr = last[0] + rcols[-1], last[1] + np.flatnonzero(rm[:, rcols[-2:]].any(axis=1)).mean()
        best = None
        for si, heads in enumerate(staves_heads):
            for a in heads:
                if not (xl - TIE_REACH_U * u <= a.match.cx <= xl + 0.5 * u and abs(a.match.cy - yl) <= TIE_DY_U * u):
                    continue
                for b in heads:
                    if b is a or b.step != a.step:
                        continue
                    if not (xr - 0.5 * u <= b.match.cx <= xr + TIE_REACH_U * u and abs(b.match.cy - yr) <= TIE_DY_U * u):
                        continue
                    if any(a.match.cx + 0.3 * u < cx < b.match.cx - 0.3 * u for cx in chords_x[si]):
                        continue          # a slur over other notes, not a tie
                    cost = abs(xl - a.match.cx) + abs(b.match.cx - xr)
                    if best is None or cost < best[0]:
                        best = (cost, a, b)
        if best:
            ties.append((best[1], best[2]))
    return ties


def rim_closed(ink: np.ndarray, m: Match, u: float) -> bool:
    """Whether the paper at the centre of a match is a closed hole, as inside
    the ring of a half note: flood-filled from the centre (in the page with
    its staff lines, where a rim cut by staff removal is whole), it stays
    inside the notehead and is no bigger than a notehead's hole. Paper between
    a stem and a flag, or between a flag and a barline, runs out."""
    h, w = ink.shape
    x0, x1 = int(max(0, m.cx - RIM_BOX_U[0] * u)), int(min(w, m.cx + RIM_BOX_U[0] * u + 1))
    y0, y1 = int(max(0, m.cy - RIM_BOX_U[1] * u)), int(min(h, m.cy + RIM_BOX_U[1] * u + 1))
    paper = (~ink[y0:y1, x0:x1]).astype(np.uint8)
    if paper.size == 0:
        return False
    cx, cy = int(round(m.cx)) - x0, int(round(m.cy)) - y0
    r = max(1, int(round(0.25 * u)))
    for dy in range(-r, r + 1):
        yy = cy + dy
        if not (0 <= yy < paper.shape[0] and 0 <= cx < paper.shape[1]) or not paper[yy, cx]:
            continue
        mask = np.zeros((paper.shape[0] + 2, paper.shape[1] + 2), np.uint8)
        area = cv2.floodFill(paper.copy(), mask, (cx, yy), 2, flags=4)[0]
        filled = mask[1:-1, 1:-1] > 0
        touches = filled[0].any() or filled[-1].any() or filled[:, 0].any() or filled[:, -1].any()
        ys = np.flatnonzero(filled.any(axis=1))
        tall = ys.size and (ys[-1] - ys[0] + 1) > RIM_HOLE_MAX_H_U * u     # a whole staff space
        if not touches and not tall and RIM_HOLE_MIN_U2 * u * u <= area <= RIM_HOLE_MAX_U2 * u * u:
            return True
    return False


def inside_beam(ink: np.ndarray, m: Match, u: float) -> bool:
    """A "notehead" in the middle of a beam: the ink through its centre runs
    on well past a notehead's width to both sides (a head, even with its
    stem or a neighbouring head of a second, stops on at least one side).
    Staff removal leaves holes in a beam where it crosses a line, and such a
    piece of beam can look like a half note on small or enlarged pages."""
    h, w = ink.shape
    reach = int(round(BEAM_SIDE_U * u))
    hits = 0
    for dy in (-0.15, 0.0, 0.15):
        y = int(round(m.cy + dy * u))
        if not 0 <= y < h:
            continue
        x0, x1 = int(round(m.cx - reach)), int(round(m.cx + reach))
        if x0 < 0 or x1 >= w:
            continue
        left = ink[y, x0:int(round(m.cx - 0.75 * u))]
        right = ink[y, int(round(m.cx + 0.75 * u)):x1 + 1]
        hits += bool(left.size and right.size and left.mean() > 0.9 and right.mean() > 0.9)
    return hits >= 2


def between_bars(run_length: np.ndarray, ink: np.ndarray, m: Match, u: float) -> bool:
    """The gap between the two strokes of a double or final barline looks like
    a hollow head; a real head has at most one long stroke (its stem) beside it."""
    r0, r1 = int(max(m.cy - 0.5 * u, 0)), int(min(m.cy + 0.5 * u, run_length.shape[0] - 1)) + 1
    def long_stroke(a: float, b: float) -> bool:
        c0, c1 = int(max(a, 0)), int(min(b, run_length.shape[1] - 1)) + 1
        return c1 > c0 and (run_length[r0:r1, c0:c1] * ink[r0:r1, c0:c1]).max(initial=0) >= BAR_SIDE_RUN_U * u
    # both strokes closer to the centre than a head's own stem (about 0.6U)
    return long_stroke(m.cx - 0.5 * u, m.cx - 0.15 * u) and long_stroke(m.cx + 0.15 * u, m.cx + 0.5 * u)


def staff_space(staff, page_u: float) -> float:
    """A staff's own line pitch (staff_detect.Staff.spacing), the page's when absent."""
    sp = float(getattr(staff, "spacing", 0.0) or 0.0)
    return sp if 0.5 * page_u <= sp <= 1.5 * page_u else page_u


def header_end(page_env: PageEnvironment, staff_index: int, x0: float, u: float) -> float:
    s = next((s for s in page_env.staves if s.index == staff_index), None)
    rights = []
    if s is not None:
        for found in (s.clef, s.key, s.time):
            # a low-confidence key reading is often the first note misread as
            # an accidental; it must not hide that note
            if found is s.key and not found.certain:
                continue
            rights += [x + w for (x, y, w, h) in found.boxes]
    return (max(rights) if rights else x0 + 4.0 * u) + HEADER_PAD_U * u


# --------------------------------------------------------------------------- #
#  Whole page
# --------------------------------------------------------------------------- #
def recognise(result: dict, removal: StaffRemoval, page_env: PageEnvironment, regions: list,
              templates: list[Template] | None = None, thresholds: dict = THRESHOLDS) -> Recognition:
    """Runs Stage 5 on one page.

    Args:
        result: Output of staff_detect.detect_staves.
        removal: Output of staff_remove.remove_from_detection.
        page_env: Output of env_detect.detect_environment (header boxes).
        regions: detect.SymbolDetection.regions (connected ink regions).
        templates: Pre-scaled bank; loaded from templates/ when omitted.
    """
    u = float(result["interline"])
    t = float(result["metrics"].line_thickness)
    templates = templates or load_templates(u)
    staves = result["staves"]
    work = remove_ledgers(removal.cleaned, result, removal)
    ink = work > 0
    _, run_length, run_centre = vertical_run_maps(ink)
    _, labels, stats, _ = cv2.connectedComponentsWithStats(ink.astype(np.uint8), connectivity=8)
    # matching only: a 3x3 median drops isolated noise pixels (which can fill a
    # half note's hole) but keeps every stroke at least 2 px wide
    match_img = cv2.medianBlur(work, MEDIAN_KSIZE) if MEDIAN_KSIZE > 1 else work

    gray = result.get("gray")
    levels = None
    if gray is not None:
        binary = result["binary"] > 0
        levels = (float(np.median(gray[binary])), float(np.median(gray[~binary])))

    # 1. template matching per staff band, then one NMS over the page.
    #    Hollow heads are also collected down to WEAK_HOLLOW: staff removal
    #    can cut the thin rim of a half note where it lies on a line, and such
    #    a head is kept only if it passes the checks of step 3.
    search_thresholds = dict(thresholds)
    search_thresholds["notehead_hollow"] = min(thresholds["notehead_hollow"], WEAK_HOLLOW)
    with_lines = result["binary"] > 0 if "binary" in result else None
    raw, starts, ends = [], [], []
    # every staff is matched at its own staff space: a solo part printed small
    # above a piano part has a smaller one than the page
    u_of = [staff_space(st, u) for st in staves]
    bank = {round(u, 1): templates}
    for si, paths in enumerate(removal.line_paths):
        us = u_of[si]
        if round(us, 1) not in bank:
            bank[round(us, 1)] = load_templates(us)
        x0 = float(removal.line_extents[si][:, 0].min())
        x1 = float(removal.line_extents[si][:, 1].max())
        start = header_end(page_env, si, x0, us)
        starts.append(start)
        ends.append(x1)
        x_hi = int(min(work.shape[1] - 1, x1 + 0.5 * us))
        if x_hi - int(start) < 2 * us:
            continue                      # nothing after the header (a cut-off or very short staff)
        raw += match_staff(match_img, si, paths, int(start), x_hi, bank[round(us, 1)], us, search_thresholds)
    # rests and accidentals must be isolated symbols: this rejects beams,
    # stems and flags, which otherwise resemble rest blocks and flats
    raw = [m for m in raw if not m.cls.startswith("rest") and m.cls not in ("sharp", "flat", "natural")
           or is_isolated(labels, stats, m, m.u or u)]

    # clefs inside staves: keep those at a clef's height on the staff, then
    # drop every other candidate inside a clef (and inside its octave 8)
    clefs = []
    for m in non_max_suppression([m for m in raw if m.cls.startswith("clef")], u):
        paths = removal.line_paths[m.staff]
        step = y_to_step(paths, m.cx, m.cy)
        lo, hi = CLEF_G_STEPS if m.cls == "clef_g" else CLEF_F_STEPS
        if lo <= step <= hi:
            clefs.append(m)
    zones = []
    for m in clefs:
        mu = m.u or u
        below = 1.8 * mu if m.cls == "clef_g" else 0.0
        zones.append((m.x - 0.1 * mu, m.y - 0.3 * mu, m.x + m.w + 0.1 * mu, m.y + m.h + below))
    raw = [m for m in raw if not m.cls.startswith("clef")
           and not any(z[0] <= m.cx <= z[2] and z[1] <= m.cy <= z[3] for z in zones)]
    kept = non_max_suppression(raw, u)

    # 2. staff of each match: nearest middle line by pitch anchor; a notehead
    #    between two staves goes to the one whose ledger lines it actually has
    mids = [np.median(p[2]) for p in removal.line_paths]
    unsupported = set()
    for m in kept:
        order = np.argsort([abs(m.anchor_y - y) for y in mids])
        m.staff = int(order[0])
        if m.cls.startswith("notehead"):
            best = None
            for si in order[:2]:
                step = int(round(y_to_step(removal.line_paths[si], m.cx, m.cy)))
                support = ledger_support(removal.cleaned > 0, removal.line_paths[si], m, step, t, u_of[si])
                if best is None or support > best[0] + 1e-9:
                    best = (support, int(si))
            m.staff = best[1]
            if best[0] < LEDGER_MIN_SUPPORT:      # beyond the staff without ledger lines:
                unsupported.add(id(m))            # tempo marks, text, dynamics
    kept = [m for m in kept if id(m) not in unsupported]

    out = [StaffSymbols(i, starts[i]) for i in range(len(staves))]

    for si, ss in enumerate(out):
        us = u_of[si]
        dot_regions = [r for r in regions
                       if r.width <= DOT_MAX_U * us and r.height <= DOT_MAX_U * us
                       and r.area >= DOT_MIN_AREA_U2 * us * us and 0.5 <= r.width / max(r.height, 1) <= 2.0]
        paths = removal.line_paths[si]
        mine = [m for m in kept if m.staff == si and starts[si] <= m.cx <= ends[si] + 0.5 * us]

        # 3. noteheads -> stems -> chords
        heads = []
        for m in mine:
            if not m.cls.startswith("notehead"):
                continue
            step = int(round(y_to_step(paths, m.cx, m.cy)))
            if not -12 <= step <= 20:
                continue
            if between_bars(run_length, ink, m, us) or inside_beam(ink, m, us):
                continue
            stem = find_stem(run_length, run_centre, ink, m.cx, m.cy, us)
            weak = m.cls == "notehead_hollow" and m.score < thresholds["notehead_hollow"]
            if weak:
                # weak hollow match: needs a stem and a closed rim in the page
                # with its staff lines (where the cut rim is still whole)
                if stem is None or with_lines is None or not rim_closed(with_lines, m, us):
                    continue
            heads.append(Head(m, step, stem))
            heads[-1].weak = weak
        # ... and a confirmed half-note head on the same stem next to it: the
        # rims that staff removal cuts are those of chord notes a second or a
        # third apart (a lone weak match is more often a flag or a digit)
        def chord_mate(h: Head, g: Head) -> bool:
            return (g is not h and g.stem is not None and abs(h.match.cx - g.match.cx) <= 1.4 * us
                    and abs(h.match.cy - g.match.cy) <= 1.1 * us and abs(g.stem[0] - h.stem[0]) <= CHORD_STEM_DX_U * us)
        heads = [h for h in heads if not getattr(h, "weak", False) or any(chord_mate(h, g) for g in heads)]
        stemmed = sorted([h for h in heads if h.stem], key=lambda h: h.stem[0])
        groups: list[list[Head]] = []
        for h in stemmed:
            g = groups[-1] if groups else None
            if g and abs(g[0].stem[0] - h.stem[0]) <= CHORD_STEM_DX_U * us and g[0].stem[3] == h.stem[3] \
                    and min(g[0].stem[2], h.stem[2]) >= max(g[0].stem[1], h.stem[1]):
                g.append(h)
            else:
                groups.append([h])
        # head type: grey-level hole test when the page image is available,
        # otherwise the class of the best template
        for h in heads:
            if levels is not None:
                line_y = float(_path_at(paths, h.step)[int(np.clip(round(h.match.cx), 0, paths.shape[1] - 1))]) \
                    if h.step % 2 == 0 else None
                h.hollow = hole_score(gray, levels, h.match.cx, h.match.cy, line_y, t, us) > HOLE_MIN
            else:
                h.hollow = h.match.cls == "notehead_hollow"
        # a filled head is about 1U tall through its centre; a beam end that
        # resembled one is only a beam's thickness
        def tall_enough(h: Head) -> bool:
            cols = range(int(round(h.match.cx - 0.1 * us)), int(round(h.match.cx + 0.1 * us)) + 1)
            r = int(round(h.match.cy))
            runs = [run_length[r, c] for c in cols if 0 <= c < ink.shape[1] and 0 <= r < ink.shape[0] and ink[r, c]]
            return bool(runs) and max(runs) >= FILLED_MIN_RUN_U * us
        dropped = {id(h) for h in heads if not h.hollow and not tall_enough(h)}
        if dropped:
            heads = [h for h in heads if id(h) not in dropped]
            groups = [[h for h in g if id(h) not in dropped] for g in groups]
            groups = [g for g in groups if g]
        for g in groups:
            hollow = sum(h.hollow for h in g) > len(g) / 2
            stem = (g[0].stem[0], min(h.stem[1] for h in g), max(h.stem[2] for h in g), g[0].stem[3])
            chord = Chord(g, float(np.mean([h.match.cx for h in g])), hollow, stem)
            if not hollow:
                chord.beams = count_beams(ink, stem, [h.match.cy for h in g], us)
            ss.chords.append(chord)
        for h in heads:
            if h.stem:
                continue
            # stemless: only a whole note. Filled stemless heads and hollow
            # "heads" inside a taller glyph (time-signature digits) are dropped.
            if not h.hollow:
                continue
            if not is_isolated(labels, stats, h.match, us, pad_u=0.2):
                continue
            block = solid_block(labels, stats, h.match, us)
            if block:
                # a solid rectangle is a whole/half rest that blur or a small
                # page made look like a whole note
                if -1 <= h.step <= 9 and not any(abs(r.x - h.match.cx) < us for r in ss.rests):
                    ss.rests.append(Rest(h.match, block_rest_kind(labels, stats, paths, h.match, h.step),
                                         h.match.cx))
                continue
            ss.chords.append(Chord([h], h.match.cx, h.hollow, None))
        for chord in ss.chords:
            chord.dots = _dots_for(chord, chord.heads, dot_regions, us)

        # 4. rests
        for m in mine:
            if not m.cls.startswith("rest"):
                continue
            step = y_to_step(paths, m.cx, m.cy)
            if not -1 <= step <= 9:
                continue
            if m.cls == "rest_block":
                kind = block_rest_kind(labels, stats, paths, m, step)
            else:
                kind = m.cls.split("_")[1]
            if any(abs(r.x - m.cx) < us and abs(r.match.cy - m.cy) < us for r in ss.rests):
                continue                      # already taken from a whole-note match (solid_block)
            ss.rests.append(Rest(m, kind, m.cx))

        # 4b. rests by shape, for pages too blurred for the rest templates
        #     (screenshots enlarged 2-4x): a filled bar hanging from / sitting
        #     on a line, or a quarter-rest-sized zigzag in the middle of the staff
        taken = [(h.match.x, h.match.y, h.match.w, h.match.h) for c in ss.chords for h in c.heads] + \
                [(r.match.x, r.match.y, r.match.w, r.match.h) for r in ss.rests] + \
                [(m.x, m.y, m.w, m.h) for m in kept if m.staff == si and m.cls in ("sharp", "flat", "natural")]
        for kind, box in shape_rests(labels, stats, paths, us, starts[si], ends[si], taken):
            m = Match("rest_block" if kind in ("whole", "half") else "rest_quarter", "shape", 0.0,
                      box[0], box[1], box[2], box[3], si, box[1] + box[3] / 2, us)
            ss.rests.append(Rest(m, kind, m.cx))
        ss.rests.sort(key=lambda r: r.x)

        # 5. barlines: thin strokes crossing the whole staff, away from noteheads
        ss.barlines = find_barlines(ink, run_length, run_centre, paths, starts[si] - 0.5 * us, ends[si] + 0.5 * us,
                                    [h.match.cx for h in heads], us)

    # 6. accidentals -> notehead just to their right at the same height (any staff)
    all_heads = [h for ss in out for c in ss.chords for h in c.heads]
    for m in kept:
        if m.cls not in ("sharp", "flat", "natural"):
            continue
        out[m.staff].accidentals.append(m)
        cands = [h for h in all_heads
                 if ACC_DX_U[0] * u <= h.match.cx - m.cx <= ACC_DX_U[1] * u
                 and abs(h.match.cy - m.anchor_y) <= ACC_DSTEP * u / 2 and h.accidental is None]
        if cands:
            best = min(cands, key=lambda h: (abs(h.match.cy - m.anchor_y), h.match.cx - m.cx))
            best.accidental = m.cls

    # 7. clef changes inside each staff, and an octave 8 on the opening clef
    for m in clefs:
        paths = removal.line_paths[m.staff]
        code = clef_code(m, y_to_step(paths, m.cx, m.cy))
        octave = octave_mark(ink, (m.x, m.y, m.w, m.h), u) if m.cls == "clef_g" else 0
        out[m.staff].clef_changes.append((float(m.cx), code, octave))
    for ss in out:
        ss.clef_changes.sort()
        s_env = next((s for s in page_env.staves if s.index == ss.index), None)
        if s_env is not None and s_env.clef.value is not None and s_env.clef.value.sign == "G" and s_env.clef.boxes:
            bx, by, bw, bh = s_env.clef.boxes[0]       # glyph box: add the template margin
            ss.start_octave = octave_mark(ink, (bx, by - 0.3 * u, bw, bh + 0.6 * u), u)

    for ss in out:
        ss.chords.sort(key=lambda c: c.x)
        ss.rests.sort(key=lambda r: r.x)
    link_beams([c for ss in out for c in ss.chords], ink, u)
    for ss in out:
        ss.ottavas = find_ottavas(labels, stats, removal.line_paths[ss.index], u_of[ss.index],
                                  starts[ss.index], ends[ss.index])

    # 8. ties
    staves_heads = [[h for c in ss.chords for h in c.heads] for ss in out]
    chords_x = [[c.x for c in ss.chords] for ss in out]
    # on the image before ledger removal: an arc above or below the staff
    # crosses ledger positions and would lose pieces there
    pre = removal.cleaned > 0
    _, pre_run, _ = vertical_run_maps(pre)
    _, pre_labels, pre_stats, _ = cv2.connectedComponentsWithStats(pre.astype(np.uint8), connectivity=8)
    ties = find_ties(pre_labels, pre_stats, pre_run, pre, staves_heads, chords_x, u)
    return Recognition(out, kept, work, u, ties, clefs)


# --------------------------------------------------------------------------- #
#  Visualisation
# --------------------------------------------------------------------------- #
COLOURS = {"notehead_filled": (220, 60, 30), "notehead_hollow": (30, 120, 220), "rest_block": (40, 160, 60),
           "rest_quarter": (40, 160, 60), "rest_eighth": (40, 160, 60), "rest_sixteenth": (40, 160, 60),
           "sharp": (170, 40, 170), "flat": (170, 40, 170), "natural": (170, 40, 170)}


def draw_recognition(gray: np.ndarray, rec: Recognition, labels: dict | None = None) -> np.ndarray:
    """RGB overlay: boxes per class, stems, barlines, optional text labels {id(Head): str}."""
    rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
    u = rec.u
    thick = max(1, int(u / 12))
    # symbols that the music is built from, coloured by what they were read as
    used = {}
    for ss in rec.staves:
        for c in ss.chords:
            for h in c.heads:
                used[id(h.match)] = "notehead_hollow" if c.hollow else "notehead_filled"
        for r in ss.rests:
            used[id(r.match)] = r.match.cls
        for a in ss.accidentals:
            used[id(a)] = a.cls
    # matches that were found but then rejected (digits, text, flags ...) in
    # light grey, so they are not mistaken for recognised notes
    for m in rec.matches:
        if m.cls in COLOURS and id(m) not in used:
            cv2.rectangle(rgb, (m.x, m.y), (m.x + m.w, m.y + m.h), (200, 200, 200), 1)
    for m in rec.matches:
        if id(m) in used:
            cv2.rectangle(rgb, (m.x, m.y), (m.x + m.w, m.y + m.h), COLOURS[used[id(m)]], thick)
    for ss in rec.staves:
        for c in ss.chords:
            if c.stem:
                x, t, b, _ = c.stem
                cv2.line(rgb, (int(x), int(t)), (int(x), int(b)), (240, 150, 0), max(1, int(u / 10)))
            for h in c.heads:
                if labels and id(h) in labels:
                    cv2.putText(rgb, labels[id(h)], (int(h.match.x), int(h.match.y - 0.15 * u)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.022 * u, (200, 0, 0), max(1, int(u / 14)), cv2.LINE_AA)
        for xb in ss.barlines:
            cv2.line(rgb, (int(xb), 0), (int(xb), 12), (0, 150, 0), 2)
    return rgb
