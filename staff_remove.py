# =============================================================================
#  staff_remove.py  --  Stage 3: staff line removal
#  Version: 1.2.0
#
#  Changelog
#  ---------
#  1.2.0  (2026-10-28)
#    - MAX_RUN_T 2.0 -> 1.5: a vertical run counts as pure staff line only up
#      to max(1.5t, t+2). With 2t, beams of a small staff (or of an enlarged
#      screenshot, where lines are thick) lying on a line were erased.
#  1.1.0  (2026-10-23)
#    - plot_cleaned() takes an optional `before` image and then shows
#      before (top) and after (bottom) in one figure, with linked axes:
#      zooming or panning one panel moves the other to the same spot.
#      The script passes the binarised page, so both panels are black on
#      white whatever the input polarity.
#
#  1.0.0  (2026-10-23)
#    - Line tracking: follows each detected line column by column, seeded by
#      the straight fit from staff_detect, so wavy/skewed lines are removed
#      along their real path. The tracked paths are returned for later stages.
#    - Run-length removal: in each column, the vertical ink run crossing the
#      line is erased only if it is about as thick as a staff line. Longer
#      runs mean a symbol crosses there, and are kept untouched.
#    - Constrained regrowth: vertical closing that may only restore pixels the
#      removal step erased (repairs slurs/ties crossing a line).
#    - Residue cleanup: drops thin fragments lying inside a line's band.
#    - Pixel-level evaluation against a staff-free ground-truth image.
#    - Separate figure windows for the cleaned page and the change map.
#    - Optional one-sided trim (TRIM_ONE_SIDED, off by default): erases the
#      band part of long runs that leave the band on one side only. Halves
#      the staff ink left behind but triples symbol ink lost on synthetic
#      pages (F1 0.971 vs 0.975), so it stays off until real ground truth
#      says otherwise.
#
#  Depends on staff_detect.py >= 3.2.0
# =============================================================================

from __future__ import annotations

import argparse
import random
from dataclasses import dataclass
from glob import glob

import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import median_filter, uniform_filter1d

from staff_detect import Staff, binarize, detect_staves, vertical_runs

# --------------------------------------------------------------------------- #
#  Tunables. Lengths are in units of U (staff line pitch) or t (staff line
#  thickness), never in pixels.
# --------------------------------------------------------------------------- #
TRACK_WINDOW_U     = 0.30  # search +-this around the fitted line while tracking
TRACK_SMOOTH_U     = 2.0   # median filter width along the line, in U
MIN_TRACK_FRACTION = 0.05  # below this share of columns, fall back to the fit
EXTENT_PERCENTILE  = 0.5   # trims stray hits when estimating a line's x-extent

MAX_RUN_T          = 1.5   # a run is "pure staff" if <= max(MAX_RUN_T*t, t+2)
BAND_TOL_PX        = 1.0   # extra half-width of the removal band, in pixels

RESIDUE_IN_BAND    = 0.80  # share of a fragment's pixels that must be in-band
TRIM_ONE_SIDED     = False # erase the band part of runs touching from one side


@dataclass
class StaffRemoval:
    """Everything staff removal produced, for downstream use and debugging.

    Attributes:
        cleaned (np.ndarray): Binary page with staff lines removed (ink = 255).
        removed (np.ndarray): Bool mask of pixels erased as pure staff.
        restored (np.ndarray): Bool mask of erased pixels the closing put back.
        residue (np.ndarray): Bool mask of leftover fragments dropped at the end.
        line_paths (list[np.ndarray]): Per staff, a `(n_lines, width)` array of
            the tracked y position of each line at every column. Use these
            instead of `Staff.lines` wherever a line's local position matters
            (pitch on curved staves, ledger line spacing).
        line_extents (list[np.ndarray]): Per staff, `(n_lines, 2)` first/last
            column each line actually spans.
        max_run (float): The pure-staff run length threshold used, in pixels.
    """
    cleaned: np.ndarray
    removed: np.ndarray
    restored: np.ndarray
    residue: np.ndarray
    line_paths: list[np.ndarray]
    line_extents: list[np.ndarray]
    max_run: float


# --------------------------------------------------------------------------- #
#  Per-pixel run maps
# --------------------------------------------------------------------------- #
def vertical_run_maps(ink: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Labels every pixel with the vertical run it belongs to.

    Built from staff_detect.vertical_runs, which decomposes the sentinel-padded,
    column-major flattened image into runs. Repeating each run's attributes
    `length` times and reshaping back gives per-pixel maps.

    Args:
        ink: Boolean ink mask.
    Returns:
        `(run_id, run_length, run_centre)` maps with the image's shape.
        `run_centre` is the run's centre row (float).
    """
    height, width = ink.shape
    lengths, _, _, _ = vertical_runs(ink)

    starts = np.r_[0, np.cumsum(lengths)[:-1]]
    padded_rows = height + 2
    centre = (starts % padded_rows) - 1 + (lengths - 1) / 2.0  # -1: top sentinel

    def to_map(per_run, dtype):
        flat = np.repeat(per_run.astype(dtype), lengths)
        return flat.reshape((padded_rows, width), order="F")[1:-1]

    run_id = to_map(np.arange(lengths.size), np.int32)
    run_length = to_map(lengths, np.int32)
    run_centre = to_map(centre, np.float32)

    return run_id, run_length, run_centre


# --------------------------------------------------------------------------- #
#  Tracking
# --------------------------------------------------------------------------- #
def track_line(ink: np.ndarray, run_length: np.ndarray, run_centre: np.ndarray,
               y_fit: float, u: float, max_run: float) -> tuple[np.ndarray, tuple[int, int]]:
    """Follows one staff line across the page.

    In every column, looks within +-TRACK_WINDOW_U*U of the straight fit for
    ink runs short enough to be pure staff, and takes the one whose centre is
    nearest the fit. Columns where a symbol covers the line yield nothing and
    are interpolated over; a median filter then removes stray hits from dots
    or text strokes that happened to fall in the window.

    Args:
        ink: Boolean ink mask.
        run_length: Per-pixel vertical run length.
        run_centre: Per-pixel vertical run centre row.
        y_fit: The line's position from the straight-line fit.
        u: Staff line pitch.
        max_run: Pure-staff run length threshold.
    Returns:
        `(path, (x_lo, x_hi))` -- y for every column, and the line's x-extent.
    """
    height, width = ink.shape
    radius = max(2, int(round(TRACK_WINDOW_U * u)))
    rows = np.clip(int(round(y_fit)) + np.arange(-radius, radius + 1), 0, height - 1)

    window_ink = ink[rows]
    window_len = run_length[rows]
    window_centre = run_centre[rows]

    candidate = window_ink & (window_len <= max_run)
    distance = np.where(candidate, np.abs(window_centre - y_fit), np.inf)
    best = np.argmin(distance, axis=0)
    columns = np.arange(width)
    found = np.isfinite(distance[best, columns])

    if found.sum() < MIN_TRACK_FRACTION * width:
        return np.full(width, float(y_fit)), (0, width - 1)

    xs = columns[found]
    ys = window_centre[best, columns][found].astype(float)

    x_lo = int(np.percentile(xs, EXTENT_PERCENTILE))
    x_hi = int(np.percentile(xs, 100 - EXTENT_PERCENTILE))

    path = np.interp(columns, xs, ys)
    size = max(5, int(TRACK_SMOOTH_U * u)) | 1
    path = median_filter(path, size=size, mode="nearest")
    path = uniform_filter1d(path, size=max(3, int(u)), mode="nearest")

    return path, (x_lo, x_hi)


# --------------------------------------------------------------------------- #
#  Removal
# --------------------------------------------------------------------------- #
def _band_coordinates(path: np.ndarray, extent: tuple[int, int], half_width: float,
                      height: int) -> tuple[np.ndarray, np.ndarray]:
    """Row/column coordinates of every pixel within `half_width` of a path."""
    x_lo, x_hi = extent
    xs = np.arange(x_lo, x_hi + 1)
    reach = int(np.ceil(half_width)) + 1
    offsets = np.arange(-reach, reach + 1)

    rows = np.round(path[xs]).astype(int)[None, :] + offsets[:, None]
    cols = np.broadcast_to(xs, rows.shape)
    inside = (np.abs(rows - path[xs][None, :]) <= half_width) & (rows >= 0) & (rows < height)

    return rows[inside], cols[inside]


def remove_staves(binary: np.ndarray, staves: list[Staff], line_thickness: float,
                  interline: float) -> StaffRemoval:
    """Removes staff lines while preserving the symbols that cross them.

    Steps:
      1. Track every line's real path (seeded by the straight fit).
      2. In each column, erase the run crossing the line only if it is no
         longer than a staff line can be. Longer runs are symbols: kept.
      3. Vertical closing, allowed to restore only pixels step 2 erased.
         Repairs strokes that crossed a line at a shallow angle, where line
         plus stroke was still short enough to look like pure staff.
      4. Drop thin fragments lying almost entirely inside a line's band
         (thick spots step 2 left behind).

    Args:
        binary: Binary page, ink = 255 (from staff_detect.binarize).
        staves: Detected staves.
        line_thickness: Staff line thickness t, in pixels.
        interline: Staff line pitch U, in pixels.
    Returns:
        A `StaffRemoval`.
    """
    ink = binary > 0
    height, width = ink.shape
    t, u = float(line_thickness), float(interline)
    max_run = max(MAX_RUN_T * t, t + 2)
    half_width = t / 2 + BAND_TOL_PX

    run_id, run_length, run_centre = vertical_run_maps(ink)

    # 1 + 2: track, then collect the ids of pure-staff runs crossing each band
    line_paths, line_extents = [], []
    band = np.zeros_like(ink)
    staff_run_ids = []
    trim_rows, trim_cols = [], []
    run_top = run_centre - (run_length - 1) / 2.0
    run_bottom = run_centre + (run_length - 1) / 2.0

    for staff in staves:
        paths, extents = [], []
        for y_fit in staff.lines:
            path, extent = track_line(ink, run_length, run_centre, y_fit, u, max_run)
            paths.append(path)
            extents.append(extent)

            rows, cols = _band_coordinates(path, extent, half_width, height)
            band[rows, cols] = True

            hit = ink[rows, cols] & (run_length[rows, cols] <= max_run)
            staff_run_ids.append(run_id[rows[hit], cols[hit]])

            if TRIM_ONE_SIDED:
                # Long runs: a symbol is involved. If it leaves the band on one
                # side only, the symbol merely touches the line and the band
                # part is staff. If it leaves on both sides, a symbol crosses
                # the line there and everything is kept.
                long_run = ink[rows, cols] & ~hit
                centre = path[cols]
                above = run_top[rows, cols] < centre - half_width
                below = run_bottom[rows, cols] > centre + half_width
                one_sided = long_run & (above ^ below)
                trim_rows.append(rows[one_sided])
                trim_cols.append(cols[one_sided])

        line_paths.append(np.vstack(paths))
        line_extents.append(np.array(extents))

    ids = np.unique(np.concatenate(staff_run_ids)) if staff_run_ids else np.array([], int)
    removed = ink & np.isin(run_id, ids)
    if trim_rows:
        removed[np.concatenate(trim_rows), np.concatenate(trim_cols)] = True
        removed &= ink
    cleaned = ink & ~removed

    # 3: constrained regrowth
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, int(np.ceil(max_run)) + 2))
    closed = cv2.morphologyEx(cleaned.astype(np.uint8), cv2.MORPH_CLOSE, kernel) > 0
    restored = closed & removed
    cleaned |= restored

    # 4: residue -- thin components lying (almost) entirely in a line's band
    n, labels, stats, _ = cv2.connectedComponentsWithStats(
        cleaned.astype(np.uint8), connectivity=8
    )
    area = stats[:, cv2.CC_STAT_AREA]
    comp_height = stats[:, cv2.CC_STAT_HEIGHT]
    in_band = np.bincount(labels[cleaned & band], minlength=n)
    is_residue = (comp_height <= max_run) & (in_band >= RESIDUE_IN_BAND * area)
    is_residue[0] = False
    residue = is_residue[labels] & cleaned
    cleaned &= ~residue

    return StaffRemoval(
        cleaned=cleaned.astype(np.uint8) * 255,
        removed=removed & ~restored,
        restored=restored,
        residue=residue,
        line_paths=line_paths,
        line_extents=line_extents,
        max_run=max_run,
    )


def remove_from_detection(result: dict) -> StaffRemoval:
    """Convenience wrapper: run removal on the output of `detect_staves`."""
    return remove_staves(
        result["binary"], result["staves"],
        result["metrics"].line_thickness, result["interline"],
    )


# --------------------------------------------------------------------------- #
#  Evaluation
# --------------------------------------------------------------------------- #
def evaluate_removal(binary: np.ndarray, cleaned: np.ndarray,
                     gt_no_staff: np.ndarray) -> dict:
    """Pixel-level scores against a staff-free ground-truth image.

    "Staff pixels" are ink in the original but not in the ground truth. This
    is the measure used by the ICDAR/GREC staff removal competitions on
    CVC-MUSCIMA.

    Args:
        binary: Original binary page, ink = 255.
        cleaned: Output of removal, ink = 255.
        gt_no_staff: Ground-truth page without staff lines (any polarity;
            it is binarised the same way).
    Returns:
        Dict with `precision`, `recall`, `f1` for staff-pixel detection, plus
        `symbol_loss` (share of symbol ink wrongly erased) and `staff_left`
        (share of staff ink left behind).
    """
    original = binary > 0
    kept = cleaned > 0
    gt_symbols = binarize(gt_no_staff) > 0

    true_staff = original & ~gt_symbols
    pred_staff = original & ~kept

    tp = np.count_nonzero(true_staff & pred_staff)
    fp = np.count_nonzero(~true_staff & pred_staff)
    fn = np.count_nonzero(true_staff & ~pred_staff)

    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "symbol_loss": fp / max(np.count_nonzero(gt_symbols & original), 1),
        "staff_left": fn / max(np.count_nonzero(true_staff), 1),
    }


# --------------------------------------------------------------------------- #
#  Visualisation
# --------------------------------------------------------------------------- #
def change_map(removal: StaffRemoval) -> np.ndarray:
    """RGB image showing what removal did.
    - black: ink kept           - red:   erased as staff
    - green: restored by regrowth  - blue:  dropped as residue
    """
    rgb = np.full(removal.cleaned.shape + (3,), 255, dtype=np.uint8)
    rgb[removal.cleaned > 0] = (0, 0, 0)
    rgb[removal.removed] = (230, 40, 40)
    rgb[removal.residue] = (40, 90, 230)
    rgb[removal.restored] = (20, 170, 60)

    return rgb


def _page_figure(image: np.ndarray, title: str, width_in: float = 16.0, cmap=None):
    height_px, width_px = image.shape[:2]
    fig, ax = plt.subplots(figsize=(width_in, width_in * height_px / width_px))
    ax.imshow(image, cmap=cmap, interpolation="nearest")
    ax.set_title(title)
    ax.axis("off")
    fig.tight_layout()
    return fig


def plot_cleaned(removal: StaffRemoval, before: np.ndarray | None = None,
                 width_in: float = 16.0):
    """The cleaned page, black on white, in its own figure.
    - With `before`, shows before (top) and after (bottom) with linked axes,
      so zooming into one panel shows the same region in the other.
    Args:
        removal: Output of `remove_staves`.
        before: Page before removal. Pass the binary from `detect_staves`
            (ink = 255) so both panels share the same polarity.
        width_in: Figure width in inches.
    """
    after = 255 - removal.cleaned
    if before is None:
        return _page_figure(after, "Staff lines removed", width_in, cmap="gray")

    if before.shape != after.shape:
        raise ValueError(f"before {before.shape} and after {after.shape} differ in size")

    height_px, width_px = after.shape[:2]
    panel_h = width_in * height_px / width_px
    fig, axes = plt.subplots(ncols=2, sharex=True, sharey=True,
                             figsize=(width_in, 2 * panel_h + 0.6))

    for ax, image, title in ((axes[0], 255 - before, "Before"),
                             (axes[1], after, "After: staff lines removed")):
        ax.imshow(image, cmap="gray", vmin=0, vmax=255, interpolation="nearest")
        ax.set_title(title)
        ax.axis("off")

    fig.tight_layout()
    return fig


def plot_changes(removal: StaffRemoval):
    """The change map, in its own figure."""
    counts = (f"erased {np.count_nonzero(removal.removed):,} px (red) | "
              f"restored {np.count_nonzero(removal.restored):,} px (green) | "
              f"residue {np.count_nonzero(removal.residue):,} px (blue)")
    return _page_figure(change_map(removal), f"What changed -- {counts}")


def summarise(result: dict, removal: StaffRemoval, scores: dict | None = None):
    """Prints a short report."""
    print(f"staves           : {len(result['staves'])}")
    print(f"pure-staff run   : <= {removal.max_run:.1f} px")
    print(f"erased           : {np.count_nonzero(removal.removed):,} px")
    print(f"restored         : {np.count_nonzero(removal.restored):,} px")
    print(f"residue dropped  : {np.count_nonzero(removal.residue):,} px")

    drift = [float(np.ptp(p[k, e[k, 0]:e[k, 1] + 1]))
             for p, e in zip(removal.line_paths, removal.line_extents)
             for k in range(p.shape[0])]
    if drift:
        print(f"line drift       : median {np.median(drift):.1f} px, "
              f"max {np.max(drift):.1f} px  (vertical wander across the page)")

    if scores:
        print()
        print(f"precision        : {scores['precision']:.4f}")
        print(f"recall           : {scores['recall']:.4f}")
        print(f"F1               : {scores['f1']:.4f}")
        print(f"symbol ink lost  : {100 * scores['symbol_loss']:.2f} %")
        print(f"staff ink left   : {100 * scores['staff_left']:.2f} %")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stage 3: staff line removal")
    parser.add_argument("image", nargs="?", help="page image (default: random from ./Separate_Set)")
    parser.add_argument("--gt", help="staff-free ground truth for scoring (e.g. PNG_GT_NoStaff)")
    args = parser.parse_args()

    path = args.image
    if path is None:
        random.seed(40)
        files = glob("./Separate_Set/*.png")
        if not files:
            raise SystemExit("No images found in ./Separate_Set/")
        path = random.sample(files, 1)[0]
    print(f"page: {path}\n")

    gray = cv2.imread(path, 0)
    result = detect_staves(gray)
    removal = remove_from_detection(result)

    scores = None
    if args.gt:
        scores = evaluate_removal(result["binary"], removal.cleaned, cv2.imread(args.gt, 0))

    summarise(result, removal, scores)
    plot_cleaned(removal, before=result["binary"])
    plot_changes(removal)
    plt.show()