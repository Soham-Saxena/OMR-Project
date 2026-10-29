# =============================================================================
#  make_templates.py  --  builds the template bank used by symbol_match.py
#  Version: 1.1.0
#
#  Changelog
#  ---------
#  1.1.0  (2026-10-25)
#    - Clef templates (FreeSerif G and F clefs) at full size and at the
#      ~80 % size used for clef changes inside a staff, for symbol_match's
#      mid-staff clef detection.
#  1.0.0  (2026-10-25)
#    - Notehead templates keep a 0.3U border at the sides but only 0.12U
#      above and below, so touching chord notes still score well.
#    - Noteheads are drawn geometrically (filled ellipse, ring with a tilted
#      hole, wide whole-note ring), in a few proportions, so no single font's
#      notehead is baked in.
#    - Accidentals and the quarter/eighth/sixteenth rests are rasterised from
#      the FreeSerif font (GNU FreeFont, Unicode "Musical Symbols" block).
#    - Whole/half rests are a plain filled rectangle; position on the staff
#      tells them apart later.
#    - Everything is drawn at a reference staff space U_REF and saved as PNG
#      (ink = white) plus templates.json, so symbol_match.py needs neither the
#      font nor this script at run time. Replace or add PNGs (e.g. crops from
#      real scans) and re-list them in templates.json to change the bank.
#
#  Usage:  python make_templates.py [--out templates] [--font path/to/FreeSerif.ttf]
# =============================================================================

from __future__ import annotations

import argparse
import json
import os

import cv2
import numpy as np

U_REF = 40          # staff space (line pitch) the templates are drawn at, in px
MARGIN_U = 0.3      # empty border around every template, in U
HEAD_VMARGIN_U = 0.12   # smaller border above/below noteheads (chords in thirds)

FREESERIF = "/usr/share/fonts/truetype/freefont/FreeSerif.ttf"
FONT_GLYPHS = {     # class, code point, height in U, anchor
    "sharp":          ("sharp",          0x266F, 2.7, "centre"),
    "flat":           ("flat",           0x266D, 2.3, "flat"),
    "natural":        ("natural",        0x266E, 2.7, "centre"),
    "rest_quarter":   ("rest_quarter",   0x1D13D, 2.9, "centre"),
    "rest_eighth":    ("rest_eighth",    0x1D13E, 1.9, "centre"),
    "rest_sixteenth": ("rest_sixteenth", 0x1D13F, 2.8, "centre"),
    "clef_g":         ("clef_g",         0x1D11E, 7.0, "centre"),
    "clef_g_change":  ("clef_g",         0x1D11E, 5.6, "centre"),
    "clef_f":         ("clef_f",         0x1D122, 3.3, "centre"),
    "clef_f_change":  ("clef_f",         0x1D122, 2.65, "centre"),
}


def _canvas(w_u: float, h_u: float) -> np.ndarray:
    return np.zeros((int(round((h_u + 2 * MARGIN_U) * U_REF)), int(round((w_u + 2 * MARGIN_U) * U_REF))), np.uint8)


def _ellipse(img: np.ndarray, a_u: float, b_u: float, angle: float, value: int):
    h, w = img.shape
    centre = (int(round(w / 2 * 16)), int(round(h / 2 * 16)))
    axes = (int(round(a_u * U_REF * 16)), int(round(b_u * U_REF * 16)))
    cv2.ellipse(img, centre, axes, angle, 0, 360, value, -1, lineType=cv2.LINE_AA, shift=4)


def notehead(a_u: float, b_u: float, angle: float, hole: tuple | None) -> np.ndarray:
    """Filled ellipse, optionally with an elliptical hole (a, b, angle).
    The border is MARGIN_U at the sides but only HEAD_VMARGIN_U above and
    below, so stacked chord notes a third apart do not eat into the score."""
    img = _canvas(2 * a_u, 2 * a_u)
    _ellipse(img, a_u, b_u, angle, 255)
    if hole:
        _ellipse(img, *hole, 0)
    ys, xs = np.nonzero(img)
    m, mv = int(MARGIN_U * U_REF), int(HEAD_VMARGIN_U * U_REF)
    return img[max(0, ys.min() - mv):ys.max() + mv + 1, max(0, xs.min() - m):xs.max() + m + 1]


def font_glyph(font_path: str, code_point: int, height_u: float) -> tuple[np.ndarray, float]:
    """Glyph scaled to height_u, with margin. Returns (image, flat-loop anchor row fraction)."""
    from PIL import Image, ImageDraw, ImageFont
    font = ImageFont.truetype(font_path, 400)
    im = Image.new("L", (900, 900), 0)
    ImageDraw.Draw(im).text((200, 100), chr(code_point), font=font, fill=255)
    a = np.asarray(im)
    ys, xs = np.nonzero(a)
    a = a[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    h = int(round(height_u * U_REF))
    w = max(1, int(round(a.shape[1] * h / a.shape[0])))
    a = cv2.resize(a, (w, h), interpolation=cv2.INTER_AREA)
    m = int(MARGIN_U * U_REF)
    return np.pad(a, m), m


def build(font_path: str) -> list[dict]:
    bank = []

    def add(name, cls, img, anchor_y=None):
        bank.append({"name": name, "cls": cls, "image": img,
                     "anchor_y": float(anchor_y if anchor_y is not None else img.shape[0] / 2)})

    # filled noteheads: (a, b, tilt) in U
    for i, (a, b, tilt) in enumerate([(0.64, 0.45, -20), (0.76, 0.48, -18), (0.55, 0.46, -12)]):
        add(f"filled_{i}", "notehead_filled", notehead(a, b, tilt, None))
    # half noteheads: ring with a thin tilted hole
    for i, (a, b, tilt, hole) in enumerate([(0.64, 0.45, -20, (0.46, 0.18, -35)),
                                            (0.76, 0.48, -18, (0.56, 0.20, -30))]):
        add(f"hollow_{i}", "notehead_hollow", notehead(a, b, tilt, hole))
    # whole notes: wide ring with a hole tilted the other way
    for i, (a, b, hole) in enumerate([(0.80, 0.48, (0.30, 0.25, 55)), (0.92, 0.50, (0.34, 0.27, 55))]):
        add(f"whole_{i}", "notehead_hollow", notehead(a, b, 0, hole))
    # whole / half rest block
    block = _canvas(1.15, 0.5)
    m = int(MARGIN_U * U_REF)
    block[m:m + int(0.5 * U_REF), m:m + int(1.15 * U_REF)] = 255
    add("rest_block", "rest_block", block)
    # font glyphs
    for name, (cls, cp, height_u, anchor) in FONT_GLYPHS.items():
        img, m = font_glyph(font_path, cp, height_u)
        anchor_y = None
        if anchor == "flat":
            # the pitch a flat marks is the centre of its loop, ~0.5U above its bottom
            anchor_y = img.shape[0] - m - 0.5 * U_REF
        add(name, cls, img, anchor_y)
    return bank


def main(argv=None):
    p = argparse.ArgumentParser(description="Build the template bank")
    p.add_argument("--out", default="templates")
    p.add_argument("--font", default=FREESERIF)
    args = p.parse_args(argv)
    os.makedirs(args.out, exist_ok=True)
    bank = build(args.font)
    index = {"u_ref": U_REF, "templates": []}
    for t in bank:
        fn = f"{t['name']}.png"
        cv2.imwrite(os.path.join(args.out, fn), t["image"])
        index["templates"].append({"name": t["name"], "cls": t["cls"], "file": fn, "anchor_y": t["anchor_y"]})
        print(f"{fn:24s} {t['cls']:16s} {t['image'].shape[1]}x{t['image'].shape[0]}")
    with open(os.path.join(args.out, "templates.json"), "w") as f:
        json.dump(index, f, indent=2)


if __name__ == "__main__":
    main()
