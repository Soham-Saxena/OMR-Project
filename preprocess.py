# =============================================================================
#  preprocess.py  --  Stage 1: page preprocessing (grey scale, deskew)
#  Version: 1.1.0
#
#  Changelog
#  ---------
#  1.1.0  (2026-10-28)
#    - clean_page(): screenshots often show more than the page (a grey
#      side panel, the taskbar, a mouse cursor). The page is the largest
#      bright region; everything outside it (not enclosed by it) is painted
#      paper white, so the grey panel is not binarised as one huge block of ink.
#      Pages that fill the image are left untouched.
#    - to_gray(): colour images with an alpha channel are composed on white
#      first (transparent PNG exports came out black).
#  1.0.0  (2026-10-25)
#    - to_gray(): any image (BGR, BGRA, grey) to 8-bit grey.
#    - estimate_skew(): projection-profile method. Staff lines are long and
#      horizontal, so the horizontal projection of the binarised page is
#      sharpest (largest sum of squares) when the page is level. Coarse search
#      over +-MAX_SKEW_DEG, then a fine search around the best angle.
#    - deskew(): rotates by the estimated angle (white border), only when the
#      angle is at least MIN_SKEW_DEG, so level pages are left untouched.
#
#  Usage:  python preprocess.py page.png  (prints the estimated angle)
# =============================================================================

from __future__ import annotations

import argparse

import cv2
import numpy as np

MAX_SKEW_DEG = 5.0     # search range
COARSE_STEP = 0.25     # degrees
FINE_STEP = 0.02       # degrees
MIN_SKEW_DEG = 0.1     # below this the page is left as it is
WORK_WIDTH = 1200      # the search runs on a copy scaled to this width


def to_gray(image: np.ndarray) -> np.ndarray:
    if image.dtype != np.uint8:                         # 16-bit PNG/TIFF
        image = cv2.convertScaleAbs(image, alpha=255.0 / max(float(image.max()), 1.0))
    if image.ndim == 2:
        return image
    if image.shape[2] == 4:
        # transparent background (MuseScore PNG export): compose on white
        alpha = image[:, :, 3:4].astype(np.float32) / 255.0
        rgb = image[:, :, :3].astype(np.float32) * alpha + 255.0 * (1.0 - alpha)
        return cv2.cvtColor(rgb.astype(np.uint8), cv2.COLOR_BGR2GRAY)
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


PAPER_MARGIN = 30          # grey levels below the paper level that still count as paper
PAGE_MIN_SHARE = 0.15      # the page must be at least this share of the image ...
PAGE_MAX_SHARE = 0.92      # ... and below this share for anything to be cleaned


def clean_page(gray: np.ndarray) -> tuple[np.ndarray, tuple | None]:
    """Paints everything outside the page white. Returns the image and the
    page box (x, y, w, h), or None when the page already fills the image."""
    h, w = gray.shape
    paper_level = float(np.percentile(gray, 95))
    if paper_level < 128:                                # white ink on a dark page: leave it
        return gray, None
    paper = (gray >= paper_level - PAPER_MARGIN).astype(np.uint8)
    # close the gaps that the music itself cuts into the paper
    k = max(3, int(round(min(h, w) / 60)))
    paper = cv2.morphologyEx(paper, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (k, k)))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(paper, connectivity=4)
    if n <= 1:
        return gray, None
    best = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    page = (labels == best).astype(np.uint8)
    # the music inside the page is enclosed by paper: fill those holes
    border = np.pad(1 - page, 1, constant_values=1)
    cv2.floodFill(border, np.zeros((h + 4, w + 4), np.uint8), (0, 0), 2, flags=4)
    reached = border[1:-1, 1:-1] == 2                    # non-page pixels connected to the image border
    share = 1.0 - reached.mean()
    if not PAGE_MIN_SHARE <= share <= PAGE_MAX_SHARE:
        return gray, None
    out = gray.copy()
    out[reached] = 255
    ys, xs = np.nonzero(~reached)
    return out, (int(xs.min()), int(ys.min()), int(xs.max() - xs.min() + 1), int(ys.max() - ys.min() + 1))


def _ink(gray: np.ndarray) -> np.ndarray:
    """Otsu binarisation, ink = 1, whichever polarity the page has."""
    _, b = cv2.threshold(gray, 0, 1, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    if b.mean() > 0.5:
        b = 1 - b
    return b.astype(np.float32)


def _sharpness(ink: np.ndarray, angle: float) -> float:
    h, w = ink.shape
    m = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    rotated = cv2.warpAffine(ink, m, (w, h), flags=cv2.INTER_NEAREST, borderValue=0)
    profile = rotated.sum(axis=1)
    return float((profile ** 2).sum())


def estimate_skew(gray: np.ndarray) -> float:
    """Angle in degrees (counter-clockwise positive, as cv2) that levels the page."""
    scale = min(1.0, WORK_WIDTH / gray.shape[1])
    small = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) if scale < 1 else gray
    ink = _ink(small)
    coarse = np.arange(-MAX_SKEW_DEG, MAX_SKEW_DEG + 1e-9, COARSE_STEP)
    best = max(coarse, key=lambda a: _sharpness(ink, a))
    fine = np.arange(best - COARSE_STEP, best + COARSE_STEP + 1e-9, FINE_STEP)
    return float(max(fine, key=lambda a: _sharpness(ink, a)))


def deskew(gray: np.ndarray) -> tuple[np.ndarray, float]:
    """Returns (levelled page, angle applied in degrees)."""
    angle = estimate_skew(gray)
    if abs(angle) < MIN_SKEW_DEG:
        return gray, 0.0
    h, w = gray.shape
    m = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    paper = int(np.median(gray))
    out = cv2.warpAffine(gray, m, (w, h), flags=cv2.INTER_LINEAR, borderValue=paper)
    return out, angle


def main(argv=None):
    p = argparse.ArgumentParser(description="Estimate page skew")
    p.add_argument("image")
    args = p.parse_args(argv)
    gray = to_gray(cv2.imread(args.image, cv2.IMREAD_UNCHANGED))
    print(f"estimated skew: {estimate_skew(gray):+.2f} degrees")


if __name__ == "__main__":
    main()
