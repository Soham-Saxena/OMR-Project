# =============================================================================
#  staff_detect.py  --  Stage 2: staff line detection and grouping
#  Version: 3.3.0
#
#  Changelog
#  ---------
#  3.3.0  (2026-10-23)
#    - Deskew (Stage 1). estimate_skew() finds the page angle that makes the
#      horizontal projection sharpest (maximum sum of squared row counts),
#      coarse +-3 deg then fine; deskew_page() rotates gray and binary.
#      detect_staves(..., deskew=True) does both first. On the real sample
#      s03 (lines drifting ~19 px across the page) a whole staff was lost
#      without it.
#    - Result dict gains `angle` (degrees) and `gray` (the deskewed page).
#      Everything in the result is in deskewed coordinates; plot_overlay now
#      draws on result["gray"] so lines always align.
#    - t/U sanity range widened to [0.03, 0.30]: printed staff paper scanned
#      at high resolution has lines ~0.05 U thick, which tripped a false
#      warning on 9 of 15 real pages.
#
#  3.2.0  (2026-10-23)
#    - FIX: polarity. detect_staves always used THRESH_BINARY_INV, so on the
#      CVC-MUSCIMA *_GT_BW images (white ink on black) it treated the
#      background as ink. New binarize() decides polarity once, from the ink
#      fraction, and every later step uses its output.
#    - vertical_runs() is now public (was _vertical_runs) so staff_remove.py
#      can build per-pixel run maps from it. The old name remains an alias.
#
#  3.1.0  (2026-10-23)
#    - The fitted-lines overlay now renders in its own figure
#      (plot_overlay), sized to the page's aspect ratio, with S0, S1, ...
#      labels in each staff's colour. plot_diagnostics is now 4 panels.
#
#  3.0.0  (2026-10-23)
#    - FIX: U was estimated from gaps between projection peaks. On real pages
#      a line carrying many noteheads produces a broad hump with several local
#      maxima a few px apart, which dragged the median gap to ~1/4 of the
#      true U and broke every downstream threshold. U now comes from vertical
#      run-length histograms (estimate_staff_metrics), which never look at the
#      projection, so micro-peaks cannot affect it.
#    - Peak finding is now scale-aware: light Gaussian smoothing, a minimum
#      peak distance of PEAK_DISTANCE_U * U, and a prominence threshold
#      relative to the strongest peaks. Replaces the plateau finder and the
#      mean-peak-height threshold.
#    - Sub-pixel peak positions via parabolic interpolation.
#    - Peak-gap U kept as a cross-check only; disagreement is reported.
#    - StaffMetrics record (t, s, U, dominance, warnings).
#
#  2.0.0  (2026-10-23)
#    - estimate_interline(): derives U (staff line pitch) from the median of
#      consecutive peak gaps, with one trimming pass. Replaces every
#      hard-coded pixel constant in the file.
#    - DBSCAN eps is now CLUSTER_EPS_U * U instead of a fixed 70 px.
#    - filter_clusters() replaced by extract_staves(): per cluster, searches
#      for the best equally-spaced N_STAFF_LINES configuration, returns a
#      least-squares (offset, spacing) model with sub-pixel line positions.
#    - Rejects configurations whose fitted spacing disagrees with U, which is
#      what stops the "every other line" degenerate fit from winning.
#    - Handles clusters with missing lines (3-4 peaks) by searching the
#      alignment offset and scoring predicted positions against the profile.
#    - Iterative extraction, so one merged cluster can yield several staves.
#    - 5-panel diagnostic figure: binary -> peaks -> clusters -> staves ->
#      overlay.
#    - Dropped unused imports (time, scipy.maximum_filter1d, MatLike).
#
#  1.0.0
#    - Initial version: horizontal projection, plateau peak finder, mean peak
#      height threshold, DBSCAN clustering only.
# =============================================================================

from __future__ import annotations

import random
from dataclasses import dataclass
from glob import glob
from itertools import combinations

import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks as _scipy_find_peaks
from sklearn.cluster import DBSCAN

# --------------------------------------------------------------------------- #
#  Tunables. Everything that is a length is expressed in units of U (the staff
#  line pitch, centre-to-centre), never in pixels.
# --------------------------------------------------------------------------- #
N_STAFF_LINES   = 5
CLUSTER_EPS_U   = 1.8    # DBSCAN neighbourhood, in units of U
SPACING_TOL     = 0.25   # |fitted spacing / U - 1| must be below this
MAX_RMS_U       = 0.15   # fit residual ceiling, as a fraction of U
STRENGTH_WEIGHT = 0.10   # how much profile strength offsets fit error
MAX_COMBO_PEAKS = 14     # cap on brute-force subset search  -> C(14,5) = 2002
MIN_SPARSE_PEAKS = 3     # fewest peaks from which a staff may be extrapolated
MAX_OVERLAP     = 0.30   # two accepted staves may not overlap more than this

PEAK_DISTANCE_U   = 0.5   # two line peaks can't be closer than this (in U)
PEAK_SMOOTH_U     = 0.08  # Gaussian sigma for the profile, in U
PEAK_PROMINENCE   = 0.25  # fraction of the 90th-percentile prominence
U_DISAGREE_WARN   = 0.10  # warn if run-length U and peak-gap U differ by more


@dataclass
class Staff:
    """One detected staff, as a fitted arithmetic progression of line positions.

    Attributes:
        lines (np.ndarray): Sub-pixel y positions of all `N_STAFF_LINES` lines,
            top to bottom. Taken from the fitted model, not from the raw peaks,
            so these are more accurate than the peaks they were derived from.
        offset (float): Model intercept `a` -- the fitted y of the top line.
        spacing (float): Model slope `b` -- this staff's local U. Prefer this
            over the page-global U for anything downstream.
        rms (float): RMS residual of the fit, in pixels.
        score (float): Selection score; lower is better.
        detected (np.ndarray): The raw peak positions the fit was built from.
            Fewer than `N_STAFF_LINES` means some lines were extrapolated.
    """
    lines: np.ndarray
    offset: float
    spacing: float
    rms: float
    score: float
    detected: np.ndarray

    @property
    def top(self) -> float:
        return float(self.lines[0])

    @property
    def bottom(self) -> float:
        return float(self.lines[-1])

    @property
    def height(self) -> float:
        """Staff height in pixels (4 * U for a 5 line staff)."""
        return self.bottom - self.top

    def step_to_y(self, step: float) -> float:
        """y position of a diatonic step, counted downward from the top line.
        - `0` is the top line, `1` the space below it, `2` the next line, ...
        - Negative values and values above `2*(N-1)` extrapolate past the staff,
          which is what ledger line notes need.
        Args:
            step: The step index, in half-space units.
        """
        return self.offset + self.spacing * (step / 2.0)

    def y_to_step(self, y: float) -> float:
        """Continuous diatonic step index for a y position (inverse of
        `step_to_y`). Round the result to snap to the nearest line or space.
        Args:
            y: The y position, in pixels.
        """
        return 2.0 * (y - self.offset) / self.spacing


# --------------------------------------------------------------------------- #
#  Profile and peaks
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class StaffMetrics:
    """Page-level staff scale, measured from vertical run-lengths.

    Attributes:
        line_thickness (float): `t`, the most common vertical black run.
        space_height (float): `s`, the most common interior vertical white run.
        interline (float): `U = s + t`, the staff line pitch (centre to centre).
        dominance_black (float): Mode count / median nonzero count in the search
            window. Well above ~3 means a sharp, trustworthy mode.
        dominance_white (float): Same, for the white histogram.
        warnings (tuple[str, ...]): Anything that looked off.
    """
    line_thickness: float
    space_height: float
    interline: float
    dominance_black: float
    dominance_white: float
    warnings: tuple[str, ...]


def vertical_runs(ink: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Column-wise run-length decomposition of a boolean ink mask.

    Every column is wrapped in a sentinel value (2) and the array is flattened
    column-major, so each column is contiguous and no run of 0s or 1s can
    continue from the bottom of one column into the top of the next.

    Returns:
        `(lengths, values, prev_values, next_values)` for every run.
    """
    sentinel = 2
    padded = np.full((ink.shape[0] + 2, ink.shape[1]), sentinel, dtype=np.uint8)
    padded[1:-1] = ink
    flat = padded.ravel(order="F")

    change = np.flatnonzero(flat[1:] != flat[:-1]) + 1
    starts = np.r_[0, change]
    lengths = np.diff(np.r_[starts, flat.size])
    values = flat[starts]
    prev_values = np.r_[sentinel, values[:-1]]
    next_values = np.r_[values[1:], sentinel]

    return lengths, values, prev_values, next_values


_vertical_runs = vertical_runs   # pre-3.2.0 name


def _histogram_mode(hist: np.ndarray, lo: int, hi: int) -> tuple[float, float]:
    """Sub-pixel mode of `hist` within `[lo, hi]`, plus its dominance."""
    hi = min(hi, hist.size - 1)
    lo = max(lo, 1)
    if hi < lo or hist[lo:hi + 1].sum() == 0:
        return float("nan"), 0.0

    window = hist[lo:hi + 1]
    m = lo + int(np.argmax(window))

    # count-weighted centroid over the three bins around the mode
    k = np.arange(max(1, m - 1), min(hist.size, m + 2))
    w = hist[k].astype(float)
    refined = float((k * w).sum() / w.sum())

    nonzero = window[window > 0]
    dominance = float(hist[m] / max(np.median(nonzero), 1.0))

    return refined, dominance


def estimate_staff_metrics(binary: np.ndarray) -> StaffMetrics:
    """Estimates staff line thickness and pitch from vertical run-lengths.

    Staff lines span the page width while music is horizontally sparse, so
    the most common short black run is the line thickness and the most common
    interior white run is the space between lines. Only white runs bounded by
    ink on both sides count, which drops the page margins.

    Args:
        binary: Image with ink as nonzero.
    Returns:
        A `StaffMetrics`.
    """
    ink = binary > 0
    warnings: list[str] = []

    fraction = ink.mean()
    if fraction > 0.5:
        ink = ~ink
        warnings.append(f"ink fraction {fraction:.2f} > 0.5, image was inverted")

    lengths, values, prev_v, next_v = vertical_runs(ink)

    black = lengths[(values == 1) & (prev_v == 0) & (next_v == 0)]
    white = lengths[(values == 0) & (prev_v == 1) & (next_v == 1)]

    black_hist = np.bincount(np.minimum(black, 255), minlength=256)
    white_hist = np.bincount(np.minimum(white, 1023), minlength=1024)

    t, dom_b = _histogram_mode(black_hist, 1, 40)
    if not np.isfinite(t):
        raise ValueError("No vertical ink runs found -- is the page blank?")

    s, dom_w = _histogram_mode(white_hist, max(3, int(2 * t)), int(25 * t))
    if not np.isfinite(s):
        raise ValueError("No interior white runs found in the staff-space range.")

    u = s + t

    ratio = t / u
    if not 0.03 <= ratio <= 0.30:
        warnings.append(f"t/U = {ratio:.3f} outside [0.03, 0.30]")
    if u < 12:
        warnings.append(f"U = {u:.1f} px is below the reliable range (~15+)")
    if dom_b < 3:
        warnings.append(f"weak black-run mode (dominance {dom_b:.1f})")
    if dom_w < 3:
        warnings.append(f"weak white-run mode (dominance {dom_w:.1f})")

    return StaffMetrics(t, s, u, dom_b, dom_w, tuple(warnings))


def smooth_profile(projection: np.ndarray, u: float) -> np.ndarray:
    """Light Gaussian smoothing of the projection, scaled to U.
    - Merges the jitter on top of a line's hump into one maximum without
      blurring neighbouring lines (sigma is ~8% of the line pitch).
    """
    sigma = max(0.8, PEAK_SMOOTH_U * u)
    radius = int(np.ceil(3 * sigma))
    x = np.arange(-radius, radius + 1)
    kernel = np.exp(-0.5 * (x / sigma) ** 2)
    kernel /= kernel.sum()

    return np.convolve(projection.astype(float), kernel, mode="same")


def find_line_peaks(profile: np.ndarray, u: float) -> tuple[np.ndarray, np.ndarray, float]:
    """Finds candidate staff line positions in a smoothed profile.

    Three things separate a staff line from the clutter around it:
    - Distance: two lines are never closer than ~U, so peaks within
      `PEAK_DISTANCE_U * U` of a stronger one are suppressed.
    - Prominence: a bump sitting in the dip between two tall line spikes has
      tiny prominence even if its absolute height is large, so the threshold
      is on prominence, not height.
    - Relativity: the threshold is a fraction of the strongest peaks'
      prominence, so it adapts to page width and ink density.

    Args:
        profile: Smoothed horizontal projection.
        u: Staff line pitch.
    Returns:
        `(peaks, prominences, threshold)` with sub-pixel peak positions.
    """
    distance = max(2, int(PEAK_DISTANCE_U * u))
    wlen = max(5, int(3 * u)) | 1

    peaks, props = _scipy_find_peaks(profile, distance=distance,
                                     prominence=0, wlen=wlen)
    if peaks.size == 0:
        return np.array([]), np.array([]), 0.0

    prominence = props["prominences"]
    threshold = PEAK_PROMINENCE * float(np.percentile(prominence, 90))
    keep = prominence >= threshold
    peaks, prominence = peaks[keep], prominence[keep]

    # parabolic interpolation for a sub-pixel position
    refined = peaks.astype(float)
    inner = (peaks > 0) & (peaks < profile.size - 1)
    y0 = profile[peaks[inner] - 1]
    y1 = profile[peaks[inner]]
    y2 = profile[peaks[inner] + 1]
    denom = y0 - 2 * y1 + y2
    with np.errstate(divide="ignore", invalid="ignore"):
        delta = np.where(denom != 0, 0.5 * (y0 - y2) / denom, 0.0)
    refined[inner] += np.clip(delta, -0.5, 0.5)

    return refined, prominence, threshold


def estimate_interline(peaks: np.ndarray) -> float | None:
    """Estimates U (staff line pitch, centre to centre) from peak spacing.

    A page of S staves contributes 4*S within-staff gaps of ~U against only
    S-1 much larger between-staff gaps, so the median of all consecutive gaps
    already sits inside the within-staff population. One trimming pass then
    discards the outliers and re-medians, which removes the pull of spurious
    peaks inside dense passages.

    Args:
        peaks: Peak positions, in any order.
    Returns:
        The estimated pitch in pixels, or `None` if there are too few peaks.
    """
    peaks = np.sort(np.asarray(peaks, dtype=float))
    if peaks.size < 2:
        return None

    gaps = np.diff(peaks)
    gaps = gaps[gaps > 0]
    if gaps.size == 0:
        return None

    u = float(np.median(gaps))
    kept = gaps[np.abs(gaps - u) <= 0.4 * u]
    if kept.size:
        u = float(np.median(kept))

    return u if u > 0 else None


def cluster_peaks(peaks: np.ndarray, u: float) -> list[np.ndarray]:
    """Groups peaks into rough staff neighbourhoods.
    - With `min_samples=1` this is equivalent to splitting wherever the gap
      between consecutive peaks exceeds `eps`; DBSCAN just saves writing it.
    Args:
        peaks: Peak positions.
        u: Estimated staff line pitch, used to scale the neighbourhood.
    Returns:
        Clusters as sorted arrays, ordered top to bottom down the page.
    """
    if peaks.size == 0:
        return []

    eps = CLUSTER_EPS_U * u
    labels = DBSCAN(eps=eps, min_samples=1).fit_predict(peaks.reshape(-1, 1))
    clusters = [np.sort(peaks[labels == label]) for label in np.unique(labels)]
    clusters.sort(key=lambda c: c[0])

    return clusters


# --------------------------------------------------------------------------- #
#  Fitting one staff
# --------------------------------------------------------------------------- #
def _fit_progression(ks: np.ndarray, ys: np.ndarray) -> tuple[float, float, float]:
    """Least-squares fit of `y = a + b*k`.
    Args:
        ks: Line indices.
        ys: Observed y positions.
    Returns:
        `(a, b, rms)` -- intercept, spacing, and RMS residual in pixels.
    """
    b, a = np.polyfit(ks, ys, 1)
    residual = ys - (a + b * ks)
    rms = float(np.sqrt(np.mean(residual ** 2)))

    return float(a), float(b), rms


def _model_strength(a: float, b: float, projection: np.ndarray,
                    n_lines: int = N_STAFF_LINES, window: int = 1) -> float:
    """How much ink actually sits at the positions a model predicts.
    - Lets a model that explains the profile beat one that merely fits the
      peaks it was given, which is what picks the right alignment when lines
      are missing.
    Args:
        a: Model intercept.
        b: Model spacing.
        projection: The horizontal projection profile.
        n_lines: Number of staff lines the model predicts.
        window: Half-width of the sampling window, in pixels.
    Returns:
        Mean predicted-position strength, normalised to roughly `[0, 1]`.
    """
    height = projection.size
    peak = max(float(projection.max()), 1.0)

    total = 0.0
    for k in range(n_lines):
        y = int(round(a + b * k))
        lo, hi = max(0, y - window), min(height, y + window + 1)
        if hi > lo:
            total += float(projection[lo:hi].max())

    return total / (n_lines * peak)


def _build_candidate(ks: np.ndarray, ys: np.ndarray, projection: np.ndarray,
                     u: float, n_lines: int) -> Staff | None:
    """Fits, gates and scores one candidate staff.
    - The spacing gate is the important one: without it, a uniformly spaced
      subset that skips every other line fits perfectly and wins.
    Args:
        ks: Line indices assigned to `ys`.
        ys: Observed peak positions.
        projection: The horizontal projection profile.
        u: Page-global staff line pitch, used as the spacing prior.
        n_lines: Number of staff lines to predict.
    Returns:
        A `Staff`, or `None` if the candidate fails a gate.
    """
    a, b, rms = _fit_progression(ks, ys)

    if b <= 0:
        return None
    if abs(b / u - 1.0) > SPACING_TOL:
        return None
    if rms / b > MAX_RMS_U:
        return None

    strength = _model_strength(a, b, projection, n_lines)
    score = rms / b - STRENGTH_WEIGHT * strength
    lines = a + b * np.arange(n_lines, dtype=float)

    return Staff(lines, a, b, rms, score, ys)


def _candidate_from_subset(ys: np.ndarray, projection: np.ndarray,
                           u: float, n_lines: int) -> Staff | None:
    """Candidate from exactly `n_lines` peaks, assumed to be lines 0..n-1."""
    ys = np.sort(np.asarray(ys, dtype=float))
    ks = np.arange(ys.size, dtype=float)

    return _build_candidate(ks, ys, projection, u, n_lines)


def _candidate_from_sparse(ys: np.ndarray, projection: np.ndarray,
                           u: float, n_lines: int) -> Staff | None:
    """Candidate from fewer than `n_lines` peaks, extrapolating the rest.

    The peaks are assigned relative line indices by rounding their gaps to
    multiples of U. Which absolute lines those are is ambiguous -- three peaks
    could be lines 0-2, 1-3 or 2-4 -- so every feasible alignment is scored
    against the profile and the best-supported one wins.
    """
    ys = np.sort(np.asarray(ys, dtype=float))
    if ys.size < MIN_SPARSE_PEAKS:
        return None

    relative = np.round((ys - ys[0]) / u).astype(int)
    if np.unique(relative).size != relative.size:
        return None

    span = int(relative.max())
    if span > n_lines - 1:
        return None

    best: Staff | None = None
    for shift in range(0, n_lines - span):
        candidate = _build_candidate(
            (relative + shift).astype(float), ys, projection, u, n_lines
        )
        if candidate is None:
            continue
        if best is None or candidate.score < best.score:
            best = candidate

    return best


def best_in_cluster(cluster: np.ndarray, projection: np.ndarray, u: float,
                    n_lines: int = N_STAFF_LINES) -> Staff | None:
    """Finds the best equally-spaced `n_lines` configuration within a cluster.
    - With enough peaks, every subset is tried, which rejects outlying peaks
      by construction.
    - With too few, the missing lines are extrapolated instead.
    Args:
        cluster: Peak positions belonging to one rough staff neighbourhood.
        projection: The horizontal projection profile.
        u: Page-global staff line pitch.
        n_lines: Number of staff lines per staff.
    Returns:
        The best scoring `Staff`, or `None` if nothing passed the gates.
    """
    cluster = np.sort(np.asarray(cluster, dtype=float))

    if cluster.size < n_lines:
        return _candidate_from_sparse(cluster, projection, u, n_lines)

    pool = cluster
    if pool.size > MAX_COMBO_PEAKS:
        strengths = projection[pool.astype(int)]
        keep = np.sort(np.argsort(strengths)[-MAX_COMBO_PEAKS:])
        pool = pool[keep]

    best: Staff | None = None
    for subset in combinations(pool, n_lines):
        candidate = _candidate_from_subset(
            np.asarray(subset), projection, u, n_lines
        )
        if candidate is None:
            continue
        if best is None or candidate.score < best.score:
            best = candidate

    return best


def extract_staves(clusters: list[np.ndarray], projection: np.ndarray, u: float,
                   n_lines: int = N_STAFF_LINES,
                   max_per_cluster: int = 4) -> list[Staff]:
    """Extracts every staff from every cluster.
    - Extraction repeats within a cluster, so a cluster that merged two nearby
      staves still yields both: the first fit consumes its own peaks and the
      remainder is searched again.
    Args:
        clusters: Peak clusters from `cluster_peaks`.
        projection: The horizontal projection profile.
        u: Page-global staff line pitch.
        n_lines: Number of staff lines per staff.
        max_per_cluster: Safety cap on repeated extraction.
    Returns:
        Detected staves, ordered top to bottom down the page.
    """
    candidates: list[Staff] = []

    for cluster in clusters:
        remaining = np.sort(np.asarray(cluster, dtype=float))

        for _ in range(max_per_cluster):
            candidate = best_in_cluster(remaining, projection, u, n_lines)
            if candidate is None:
                break

            candidates.append(candidate)
            remaining = remaining[~np.isin(remaining, candidate.detected)]
            if remaining.size < MIN_SPARSE_PEAKS:
                break

    # Real staves never overlap vertically, so accept greedily by score and
    # drop anything that lands on top of an already accepted staff. Without
    # this, an interleaved set of lines yields a second, half-space-offset
    # "staff" that fits just as well as the real one.
    staves: list[Staff] = []
    for candidate in sorted(candidates, key=lambda s: s.score):
        span = candidate.height
        clash = False
        for accepted in staves:
            overlap = (min(candidate.bottom, accepted.bottom)
                       - max(candidate.top, accepted.top))
            if overlap > MAX_OVERLAP * min(span, accepted.height):
                clash = True
                break
        if not clash:
            staves.append(candidate)

    staves.sort(key=lambda s: s.top)

    return staves


# --------------------------------------------------------------------------- #
#  Pipeline
# --------------------------------------------------------------------------- #
def binarize(gray: np.ndarray) -> np.ndarray:
    """Otsu binarisation with ink as 255, whatever the input polarity.

    Handles both black-on-white scans and white-on-black ground-truth images.
    A music page is mostly paper (typically 5-15% ink), so whichever side of
    the threshold holds the majority of pixels is the background.

    Args:
        gray: Grayscale or already-binary page image.
    Returns:
        uint8 image, ink = 255, background = 0.
    """
    _, binary = cv2.threshold(
        gray, 0, 255,
        cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU
    )
    if np.count_nonzero(binary) > binary.size / 2:
        binary = cv2.bitwise_not(binary)

    return binary


SKEW_RANGE_DEG  = 3.0     # search +-this many degrees
SKEW_COARSE_DEG = 0.1
SKEW_FINE_DEG   = 0.01
SKEW_MIN_DEG    = 0.02    # don't bother rotating below this
SKEW_MAX_INK    = 400_000 # subsample ink pixels above this, for speed


def estimate_skew(binary: np.ndarray, seed: int = 0) -> float:
    """Estimates page rotation, in degrees (positive = counter-clockwise).

    Staff lines are long horizontal runs of ink. When the page is level they
    pile into a few rows and the horizontal projection is sharply peaked;
    when it is rotated they smear across neighbouring rows. So the angle that
    maximises the sum of squared row counts is the one that levels the page.
    For angles this small, rotation is well approximated by a shear: shift
    each pixel's row by x * tan(angle).

    Args:
        binary: Binary page, ink = 255.
        seed: Seed for subsampling ink pixels on very dense pages.
    Returns:
        The correction angle in degrees.
    """
    ys, xs = np.nonzero(binary)
    if ys.size == 0:
        return 0.0
    if ys.size > SKEW_MAX_INK:
        pick = np.random.default_rng(seed).choice(ys.size, SKEW_MAX_INK, replace=False)
        ys, xs = ys[pick], xs[pick]

    xs = xs.astype(float) - binary.shape[1] / 2.0
    ys = ys.astype(float)
    offset = abs(np.tan(np.radians(SKEW_RANGE_DEG))) * binary.shape[1] + 2

    def sharpness(deg: float) -> float:
        rows = np.round(ys - xs * np.tan(np.radians(deg)) + offset).astype(int)
        counts = np.bincount(rows)
        return float(np.dot(counts, counts))

    coarse = np.arange(-SKEW_RANGE_DEG, SKEW_RANGE_DEG + 1e-9, SKEW_COARSE_DEG)
    best = coarse[int(np.argmax([sharpness(a) for a in coarse]))]
    fine = np.arange(best - SKEW_COARSE_DEG, best + SKEW_COARSE_DEG + 1e-9, SKEW_FINE_DEG)
    return float(fine[int(np.argmax([sharpness(a) for a in fine]))])


def deskew_page(gray: np.ndarray, binary: np.ndarray,
                angle: float) -> tuple[np.ndarray, np.ndarray]:
    """Rotates the page by `angle` degrees about its centre.
    - Gray uses bilinear interpolation, binary uses nearest neighbour so it
      stays binary. Exposed corners are filled with background.
    Returns:
        `(gray, binary)` deskewed.
    """
    if abs(angle) < SKEW_MIN_DEG:
        return gray, binary

    height, width = binary.shape
    matrix = cv2.getRotationMatrix2D((width / 2.0, height / 2.0), angle, 1.0)

    ink_is_dark = np.mean(gray[binary > 0]) < np.mean(gray[binary == 0])
    paper = int(np.median(gray[binary == 0])) if np.any(binary == 0) else (255 if ink_is_dark else 0)

    gray_out = cv2.warpAffine(gray, matrix, (width, height), flags=cv2.INTER_LINEAR,
                              borderMode=cv2.BORDER_CONSTANT, borderValue=paper)
    binary_out = cv2.warpAffine(binary, matrix, (width, height), flags=cv2.INTER_NEAREST,
                                borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return gray_out, binary_out


def detect_staves(gray: np.ndarray, u_override: float | None = None,
                  deskew: bool = True) -> dict:
    """Runs the whole Stage 2 chain on one grayscale page.

    Order matters: U is measured first from run-lengths, and only then used
    to configure peak finding, clustering and fitting. U must never be
    derived from the same peaks it is used to clean up.

    Args:
        gray: Grayscale page image.
        u_override: Staff line pitch to use instead of the run-length estimate.
        deskew: Level the page first (recommended). All returned coordinates
            are in the deskewed frame; use `result["gray"]` for display.
    Returns:
        Dict holding every intermediate, for diagnostics as well as output:
        `binary`, `metrics`, `projection`, `profile`, `strong_peaks`,
        `prominence_threshold`, `interline`, `interline_peak_gap`,
        `clusters`, `staves`, `warnings`.
    """
    binary = binarize(gray)
    angle = estimate_skew(binary) if deskew else 0.0
    gray, binary = deskew_page(gray, binary, angle)

    metrics = estimate_staff_metrics(binary)
    interline = float(u_override or metrics.interline)
    warnings = list(metrics.warnings)

    projection = np.count_nonzero(binary, axis=1)
    profile = smooth_profile(projection, interline)

    strong_peaks, _, prominence_threshold = find_line_peaks(profile, interline)
    if strong_peaks.size == 0:
        raise ValueError("No line peaks found in the horizontal projection.")

    # Independent cross-check. Never used as the prior: it is computed from
    # the very peaks that clutter can corrupt.
    interline_peak_gap = estimate_interline(strong_peaks)
    if interline_peak_gap is not None:
        disagreement = abs(interline_peak_gap / interline - 1.0)
        if disagreement > U_DISAGREE_WARN:
            warnings.append(
                f"peak-gap U {interline_peak_gap:.1f} vs run-length U "
                f"{interline:.1f} ({100 * disagreement:.0f}% apart)"
            )

    clusters = cluster_peaks(strong_peaks, interline)
    staves = extract_staves(clusters, profile, interline)

    return {
        "binary": binary,
        "gray": gray,
        "angle": angle,
        "metrics": metrics,
        "projection": projection,
        "profile": profile,
        "strong_peaks": strong_peaks,
        "prominence_threshold": prominence_threshold,
        "interline": interline,
        "interline_peak_gap": interline_peak_gap,
        "clusters": clusters,
        "staves": staves,
        "warnings": warnings,
    }


def draw_staves(gray: np.ndarray, staves: list[Staff], u: float) -> np.ndarray:
    """Draws detected staff lines over the page, one colour per staff.
    Args:
        gray: Grayscale page image.
        staves: Detected staves.
        u: Staff line pitch, used to scale the stroke width.
    Returns:
        BGR overlay image.
    """
    overlay = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    thickness = max(1, int(round(0.10 * u)))
    cmap = plt.colormaps["tab10"]
    width = overlay.shape[1]

    for i, staff in enumerate(staves):
        rgb = cmap(i % 10)[:3]
        bgr = tuple(int(255 * c) for c in reversed(rgb))
        for y in staff.lines:
            cv2.line(overlay, (0, int(round(y))), (width - 1, int(round(y))),
                     color=bgr, thickness=thickness)

    return overlay


def plot_diagnostics(gray: np.ndarray, result: dict):
    """Renders the 4-panel diagnostic figure.
    - binary -> line peaks -> clustered peaks -> identified staves
    - The fitted-lines overlay is a separate figure; see `plot_overlay`.
    Args:
        gray: Grayscale page image.
        result: The dict returned by `detect_staves`.
    """
    projection = result["projection"]
    staves = result["staves"]
    top = float(projection.max())
    cmap = plt.colormaps["tab10"]

    fig, axes = plt.subplots(nrows=4, figsize=(11, 16))

    axes[0].imshow(result["binary"], cmap="gray")
    axes[0].axis("off")
    axes[0].set_title("1. Binarised (Otsu, ink = white)")

    axes[1].plot(projection, linewidth=0.5, color="0.75", label="raw projection")
    axes[1].plot(result["profile"], linewidth=0.8, label="smoothed")
    axes[1].vlines(result["strong_peaks"], 0, top, colors="red", alpha=0.7)
    axes[1].legend(loc="upper right", fontsize=8)
    gap_u = result["interline_peak_gap"]
    gap_txt = f", peak-gap check {gap_u:.2f} px" if gap_u else ""
    axes[1].set_title(
        f"2. Line peaks ({result['strong_peaks'].size}) | "
        f"U = {result['interline']:.2f} px (run-length{gap_txt}) | "
        f"t = {result['metrics'].line_thickness:.2f} px"
    )

    axes[2].plot(result["profile"], linewidth=0.8)
    for i, cluster in enumerate(result["clusters"]):
        axes[2].vlines(cluster, 0, top, colors=cmap(i % 10), alpha=0.8)
    axes[2].set_title(f"3. Clustered peaks ({len(result['clusters'])} clusters, "
                      f"eps = {CLUSTER_EPS_U * result['interline']:.0f} px)")

    axes[3].plot(result["profile"], linewidth=0.8)
    for i, staff in enumerate(staves):
        axes[3].vlines(staff.lines, 0, top, colors=cmap(i % 10), alpha=0.9)
        axes[3].annotate(f"S{i}", (staff.top, top), fontsize=7,
                         ha="center", va="bottom", color=cmap(i % 10))
    axes[3].set_title(f"4. Identified staves ({len(staves)})")

    plt.tight_layout()

    return fig


def plot_overlay(gray: np.ndarray, result: dict, width_in: float = 16.0):
    """Renders the fitted staff lines over the page, in a figure of its own.
    - The figure takes the page's aspect ratio, so nothing is squashed.
    - Each staff is labelled S0, S1, ... in its line colour, matching the
      labels in panel 4 of `plot_diagnostics` and the summary table.
    Args:
        gray: Grayscale page image.
        result: The dict returned by `detect_staves`.
        width_in: Figure width in inches.
    """
    gray = result.get("gray", gray)   # deskewed page, aligned with the staves
    staves = result["staves"]
    cmap = plt.colormaps["tab10"]
    height_px, width_px = gray.shape[:2]

    fig, ax = plt.subplots(figsize=(width_in, width_in * height_px / width_px))
    ax.imshow(cv2.cvtColor(
        draw_staves(gray, staves, result["interline"]), cv2.COLOR_BGR2RGB
    ))

    for i, staff in enumerate(staves):
        ax.text(4, (staff.top + staff.bottom) / 2, f"S{i}",
                color=cmap(i % 10), fontsize=10, fontweight="bold",
                ha="left", va="center",
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.7, pad=1))

    ax.set_title(f"Fitted staff lines: {len(staves)} staves, "
                 f"U = {result['interline']:.2f} px")
    ax.axis("off")
    fig.tight_layout()

    return fig


def summarise(result: dict):
    """Prints a per-staff summary table.
    Args:
        result: The dict returned by `detect_staves`.
    """
    m = result["metrics"]
    print(f"line thickness t : {m.line_thickness:.2f} px  (dominance {m.dominance_black:.1f})")
    print(f"space height   s : {m.space_height:.2f} px  (dominance {m.dominance_white:.1f})")
    print(f"U (run-length)   : {result['interline']:.2f} px")
    if result["interline_peak_gap"] is not None:
        print(f"U (peak-gap)     : {result['interline_peak_gap']:.2f} px  (cross-check)")
    for w in result["warnings"]:
        print(f"WARNING          : {w}")
    if "angle" in result:
        print(f"deskew angle     : {result['angle']:+.2f} deg")
    print(f"line peaks       : {result['strong_peaks'].size}")
    print(f"clusters         : {len(result['clusters'])}")
    print(f"staves           : {len(result['staves'])}")
    print()
    print(f"{'#':>2}  {'top':>8}  {'spacing':>8}  {'rms':>6}  "
          f"{'rms/U':>6}  {'peaks':>5}  lines")
    for i, staff in enumerate(result["staves"]):
        lines = "  ".join(f"{y:7.2f}" for y in staff.lines)
        print(f"{i:>2}  {staff.top:8.2f}  {staff.spacing:8.2f}  "
              f"{staff.rms:6.2f}  {staff.rms / staff.spacing:6.3f}  "
              f"{staff.detected.size:>5}  {lines}")


if __name__ == "__main__":
    random.seed(35)

    files = glob("./Separate_Set/*.png")
    if not files:
        raise SystemExit("No images found in ./Separate_Set/")

    path = random.sample(files, 1)[0]
    print(f"page: {path}\n")

    gray = cv2.imread(path, 0)
    result = detect_staves(gray)

    summarise(result)
    plot_diagnostics(gray, result)
    plot_overlay(gray, result)
    plt.show()