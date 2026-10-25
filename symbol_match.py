# =============================================================================
#  symbol_match.py  --  Stage 5: symbol recognition by template matching
#  Version: 1.0.0
#
#  Changelog
#  ---------
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
}
BAND_ABOVE_U = 4.5             # search this far above the top line ...
BAND_BELOW_U = 4.5             # ... and below the bottom line
PEAK_RADIUS_U = 0.35           # local-maximum neighbourhood
NMS_IOU = 0.30                 # suppress a box overlapping a better one by more than this ...
NMS_IOS = 0.60                 # ... or covering this share of the smaller box
NMS_HEAD_DX_U = 0.6            # two noteheads closer than this in x ...
NMS_HEAD_DY_U = 0.6            # ... and this in y are the same notehead
LEDGER_STEPS = (-2, -4, -6, 10, 12, 14)
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


@dataclass
class Recognition:
    staves: list                 # StaffSymbols per staff
    matches: list                # every Match kept after NMS
    work: np.ndarray             # staff- and ledger-free image used for matching (ink = 255)
    u: float


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
    for si, paths in enumerate(removal.line_paths):
        x_lo = int(removal.line_extents[si][:, 0].min())
        x_hi = int(removal.line_extents[si][:, 1].max())
        xs = np.arange(x_lo, x_hi + 1)
        for step in LEDGER_STEPS:
            ys = _path_at(paths, step)[xs]
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
                               staff_index, float(top + y + t.anchor_y)))
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
        return abs(k.cx - m.cx) < NMS_HEAD_DX_U * u and abs(k.cy - m.cy) < NMS_HEAD_DY_U * u
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

    # 1. template matching per staff band, then one NMS over the page
    raw, starts, ends = [], [], []
    for si, paths in enumerate(removal.line_paths):
        x0 = float(removal.line_extents[si][:, 0].min())
        x1 = float(removal.line_extents[si][:, 1].max())
        start = header_end(page_env, si, x0, u)
        starts.append(start)
        ends.append(x1)
        raw += match_staff(match_img, si, paths, int(start), int(min(work.shape[1] - 1, x1 + 0.5 * u)), templates, u,
                           thresholds)
    # rests and accidentals must be isolated symbols: this rejects beams,
    # stems and flags, which otherwise resemble rest blocks and flats
    raw = [m for m in raw if not m.cls.startswith("rest") and m.cls not in ("sharp", "flat", "natural")
           or is_isolated(labels, stats, m, u)]
    kept = non_max_suppression(raw, u)

    # 2. staff of each match: nearest middle line by pitch anchor; a notehead
    #    between two staves goes to the one whose ledger lines it actually has
    mids = [np.median(p[2]) for p in removal.line_paths]
    for m in kept:
        order = np.argsort([abs(m.anchor_y - y) for y in mids])
        m.staff = int(order[0])
        if m.cls.startswith("notehead") and len(order) > 1:
            best = None
            for si in order[:2]:
                step = int(round(y_to_step(removal.line_paths[si], m.cx, m.cy)))
                support = ledger_support(removal.cleaned > 0, removal.line_paths[si], m, step, t, u)
                if best is None or support > best[0] + 1e-9:
                    best = (support, int(si))
            m.staff = best[1]

    out = [StaffSymbols(i, starts[i]) for i in range(len(staves))]
    dot_regions = [r for r in regions
                   if r.width <= DOT_MAX_U * u and r.height <= DOT_MAX_U * u
                   and r.area >= DOT_MIN_AREA_U2 * u * u and 0.5 <= r.width / max(r.height, 1) <= 2.0]

    for si, ss in enumerate(out):
        paths = removal.line_paths[si]
        mine = [m for m in kept if m.staff == si and starts[si] <= m.cx <= ends[si] + 0.5 * u]

        # 3. noteheads -> stems -> chords
        heads = []
        for m in mine:
            if not m.cls.startswith("notehead"):
                continue
            step = int(round(y_to_step(paths, m.cx, m.cy)))
            if not -7 <= step <= 15:
                continue
            heads.append(Head(m, step, find_stem(run_length, run_centre, ink, m.cx, m.cy, u)))
        stemmed = sorted([h for h in heads if h.stem], key=lambda h: h.stem[0])
        groups: list[list[Head]] = []
        for h in stemmed:
            g = groups[-1] if groups else None
            if g and abs(g[0].stem[0] - h.stem[0]) <= CHORD_STEM_DX_U * u and g[0].stem[3] == h.stem[3] \
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
                h.hollow = hole_score(gray, levels, h.match.cx, h.match.cy, line_y, t, u) > HOLE_MIN
            else:
                h.hollow = h.match.cls == "notehead_hollow"
        # a filled head is about 1U tall through its centre; a beam end that
        # resembled one is only a beam's thickness
        def tall_enough(h: Head) -> bool:
            cols = range(int(round(h.match.cx - 0.1 * u)), int(round(h.match.cx + 0.1 * u)) + 1)
            r = int(round(h.match.cy))
            runs = [run_length[r, c] for c in cols if 0 <= c < ink.shape[1] and 0 <= r < ink.shape[0] and ink[r, c]]
            return bool(runs) and max(runs) >= FILLED_MIN_RUN_U * u
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
                chord.beams = count_beams(ink, stem, [h.match.cy for h in g], u)
            ss.chords.append(chord)
        for h in heads:
            if not h.stem:   # stemless: a whole note (a filled head without a stem is read as a quarter)
                ss.chords.append(Chord([h], h.match.cx, h.hollow, None))
        for chord in ss.chords:
            chord.dots = _dots_for(chord, chord.heads, dot_regions, u)

        # 4. rests
        for m in mine:
            if not m.cls.startswith("rest"):
                continue
            step = y_to_step(paths, m.cx, m.cy)
            if not -1 <= step <= 9:
                continue
            if m.cls == "rest_block":
                kind = "whole" if step < 3.0 else "half"
            else:
                kind = m.cls.split("_")[1]
            ss.rests.append(Rest(m, kind, m.cx))

        # 5. barlines: thin strokes crossing the whole staff, away from noteheads
        ss.barlines = find_barlines(ink, run_length, run_centre, paths, starts[si] - 0.5 * u, ends[si] + 0.5 * u,
                                    [h.match.cx for h in heads], u)

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

    for ss in out:
        ss.chords.sort(key=lambda c: c.x)
        ss.rests.sort(key=lambda r: r.x)
    return Recognition(out, kept, work, u)


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
    for m in rec.matches:
        if m.cls in COLOURS:
            cv2.rectangle(rgb, (m.x, m.y), (m.x + m.w, m.y + m.h), COLOURS[m.cls], max(1, int(u / 12)))
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
