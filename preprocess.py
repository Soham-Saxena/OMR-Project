# =============================================================================
#  preprocess.py  --  Stage 1: page preprocessing (grey scale, deskew)
#  Version: 1.0.0
#
#  Changelog
#  ---------
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
    if image.ndim == 2:
        return image
    if image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


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
