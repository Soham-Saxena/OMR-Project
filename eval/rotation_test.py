# =============================================================================
#  rotation_test.py  --  effect of page rotation, with and without deskew
#  Version: 1.0.0
#
#  Changelog
#  ---------
#  1.0.0  (2026-10-25)
#    - Engraves clean pages (degrade 0) rotated by each angle, runs the
#      pipeline with and without skew correction and reports, per angle and
#      mode: pages whose staff count is exact, pitch-sequence accuracy, MIDI
#      onset+pitch F1 and the estimated skew.
#
#  Usage:  python eval/rotation_test.py --work /tmp/rot --seeds 0-5 --json rotation.json
# =============================================================================

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import cv2  # noqa: E402

from evaluate import evaluate_page  # noqa: E402
from synth_score import engrave_page, parse_seeds  # noqa: E402

ANGLES = [0.0, 0.2, 0.5, 1.0, 2.0]


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--work", required=True)
    p.add_argument("--seeds", default="0-5")
    p.add_argument("--json")
    args = p.parse_args(argv)
    out = {}
    for angle in ANGLES:
        d = os.path.join(args.work, f"r{angle}")
        os.makedirs(d, exist_ok=True)
        pngs = []
        for seed in parse_seeds(args.seeds):
            img, nostaff, gt = engrave_page(seed, degrade=0, rotation=angle or None)
            stem = os.path.join(d, f"synth_{seed:03d}")
            cv2.imwrite(stem + ".png", img)
            cv2.imwrite(stem + "_nostaff.png", nostaff)
            json.dump(gt, open(stem + ".json", "w"), indent=1, default=float)
            pngs.append(stem + ".png")
        for mode, straighten in (("raw", False), ("deskew", True)):
            rows, failed, n_failed_notes = [], 0, 0
            for png in pngs:
                try:
                    rows.append(evaluate_page(png, d, straighten=straighten))
                except Exception:                      # staff detection collapsed: scores 0
                    failed += 1
                    n_failed_notes += len(json.load(open(png.replace(".png", ".json")))["heads"])
            n = max(1, sum(r["e2e_onset_pitch"]["gt"] for r in rows) + n_failed_notes)
            out[f"{angle}|{mode}"] = {
                "pages": len(pngs), "failed": failed,
                "staves_exact": sum(r["staves"]["det"] == r["staves"]["gt"] for r in rows),
                "seq_acc": sum(r["pitch_sequence_acc"] * r["e2e_onset_pitch"]["gt"] for r in rows) / n,
                "e2e_f1": sum(r["e2e_onset_pitch"]["f1"] for r in rows) / len(pngs),
                "skew": [round(r["skew_deg"], 2) for r in rows]}
            print(angle, mode, out[f"{angle}|{mode}"])
    if args.json:
        json.dump(out, open(args.json, "w"), indent=1)


if __name__ == "__main__":
    main()
