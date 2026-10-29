# =============================================================================
#  evaluate_cvc.py  --  staff / clef / key / time / system checks on the
#                       handwritten CVC-MUSCIMA pages against a manual key
#  Version: 1.0.0
#
#  Changelog
#  ---------
#  1.0.0  (2026-10-25)
#    - Runs preprocessing (optional deskew), staff detection, staff removal
#      and env_detect on each page in Separate_Set and compares with
#      eval/cvc_manual_gt.json: staff count, clef sign, key signature,
#      time-signature location and system grouping.
#
#  Usage:  python eval/evaluate_cvc.py [--no-deskew]
# =============================================================================

from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import cv2

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from env_detect import detect_environment  # noqa: E402
from preprocess import deskew, to_gray  # noqa: E402
from staff_detect import detect_staves  # noqa: E402
from staff_remove import remove_from_detection  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--pages", default=os.path.join(HERE, "..", "Separate_Set"))
    p.add_argument("--no-deskew", action="store_true")
    p.add_argument("--json")
    args = p.parse_args(argv)
    gt_all = json.load(open(os.path.join(HERE, "cvc_manual_gt.json")))
    tot = {"staves_true": 0, "staves_found": 0, "pages_staves_exact": 0, "clef_ok": 0, "key_ok": 0, "scored": 0,
           "time_true": 0, "time_found": 0, "time_hit": 0, "systems_ok": 0, "pages": 0}
    rows = []
    for path in sorted(glob.glob(os.path.join(args.pages, "*.png"))):
        name = os.path.basename(path)[:-4]
        gt = gt_all[name]
        gray = to_gray(cv2.imread(path, cv2.IMREAD_UNCHANGED))
        angle = 0.0
        if not args.no_deskew:
            gray, angle = deskew(gray)
        result = detect_staves(gray)
        env = detect_environment(result, remove_from_detection(result))
        n_true, n_found = len(gt["clef"]), len(result["staves"])
        row = {"page": name, "skew": round(angle, 2), "staves": f"{n_found}/{n_true}"}
        tot["pages"] += 1
        tot["staves_true"] += n_true
        tot["staves_found"] += n_found
        if n_found == n_true:
            tot["pages_staves_exact"] += 1
            clef = [s.clef.value.sign if s.clef.value else "?" for s in env.staves]
            key = [s.key.value for s in env.staves]
            time = [int(s.time_present) for s in env.staves]
            row["clef_ok"] = sum(a == b for a, b in zip(clef, gt["clef"]))
            row["key_ok"] = sum(a == b for a, b in zip(key, gt["key"]))
            row["time"] = f'{sum(a and b for a, b in zip(time, gt["time"]))}/{sum(gt["time"])} (+{sum(a and not b for a, b in zip(time, gt["time"]))} false)'
            row["systems_ok"] = [sorted(s) for s in env.systems] == gt["systems"]
            tot["clef_ok"] += row["clef_ok"]
            tot["key_ok"] += row["key_ok"]
            tot["scored"] += n_true
            tot["time_true"] += sum(gt["time"])
            tot["time_found"] += sum(time)
            tot["time_hit"] += sum(a and b for a, b in zip(time, gt["time"]))
            tot["systems_ok"] += row["systems_ok"]
        rows.append(row)
        print(row)
    print(tot)
    if args.json:
        json.dump({"pages": rows, "total": tot}, open(args.json, "w"), indent=1)


if __name__ == "__main__":
    main()
