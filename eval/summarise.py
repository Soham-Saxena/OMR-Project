# =============================================================================
#  summarise.py  --  pooled results over a set of evaluated pages
#  Version: 1.0.0
#
#  Changelog
#  ---------
#  1.0.0  (2026-10-25)
#    - Measure-level note F1 row.
#    - Micro-averaged (pooled counts) precision/recall/F1 and accuracies from
#      the all.json written by evaluate.py, overall and split by degradation
#      level and by page type. Prints Markdown tables; --json saves numbers.
#
#  Usage:  python eval/summarise.py eval/results/all.json [--json out.json]
# =============================================================================

from __future__ import annotations

import argparse
import json

import numpy as np


def pooled(rows: list[dict]) -> dict:
    def prf(key):
        tp = sum(r[key]["tp"] for r in rows)
        gt = sum(r[key]["gt"] for r in rows)
        det = sum(r[key]["det"] for r in rows)
        p, rc = (tp / det if det else 0.0), (tp / gt if gt else 0.0)
        return {"precision": p, "recall": rc, "f1": 2 * p * rc / (p + rc) if p + rc else 0.0,
                "tp": tp, "gt": gt, "det": det}

    heads_tp = sum(r["heads"]["tp"] for r in rows) or 1
    out = {
        "pages": len(rows),
        "staves_exact": sum(r["staves"]["det"] == r["staves"]["gt"] for r in rows),
        "staves_gt": sum(r["staves"]["gt"] for r in rows),
        "staves_det": sum(r["staves"]["det"] for r in rows),
        "line_err_px": float(np.mean([r["staves"]["line_err_px"] for r in rows])),
        "removal_f1": float(np.mean([r["removal"]["f1"] for r in rows])),
        "removal_precision": float(np.mean([r["removal"]["precision"] for r in rows])),
        "removal_recall": float(np.mean([r["removal"]["recall"] for r in rows])),
        "symbol_loss": float(np.mean([r["removal"]["symbol_loss"] for r in rows])),
        "clef_acc": float(np.mean([r["clef_acc"] for r in rows])),
        "key_acc": float(np.mean([r["key_acc"] for r in rows])),
        "heads": prf("heads"),
        "head_type_acc": sum(r["head_type_acc"] * r["heads"]["tp"] for r in rows) / heads_tp,
        "pitch_acc": sum(r["pitch_acc"] * r["heads"]["tp"] for r in rows) / heads_tp,
        "duration_acc": sum(r["duration_acc"] * r["heads"]["tp"] for r in rows) / heads_tp,
        "accidental_acc": (sum((r["accidental_acc"] or 0) * r["accidental_n"] for r in rows)
                           / max(1, sum(r["accidental_n"] for r in rows))),
        "accidental_n": sum(r["accidental_n"] for r in rows),
        "rests": prf("rests"),
        "measures_ok": sum(r["measures"]["ok"] for r in rows),
        "measures_total": sum(r["measures"]["total"] for r in rows),
        "e2e_onset_pitch": prf("e2e_onset_pitch"),
        "measure_onset_pitch": prf("measure_onset_pitch"),
        "e2e_with_duration": prf("e2e_with_duration"),
        "pitch_sequence_acc": float(np.mean([r["pitch_sequence_acc"] for r in rows])),
        "pages_e2e_perfect": sum(r["e2e_onset_pitch"]["f1"] >= 0.999 for r in rows),
        "seconds_per_page": float(np.mean([sum(r["timings"].values()) for r in rows])),
        "timings": {k: float(np.mean([r["timings"][k] for r in rows])) for k in rows[0]["timings"]},
    }
    return out


def table(groups: dict[str, dict]) -> str:
    rows = [
        ("Pages", lambda s: f'{s["pages"]}'),
        ("Staves found / true", lambda s: f'{s["staves_det"]} / {s["staves_gt"]}'),
        ("Staff removal F1 (pixels)", lambda s: f'{s["removal_f1"]:.3f}'),
        ("Clef accuracy (per staff)", lambda s: f'{s["clef_acc"]:.3f}'),
        ("Key accuracy (per staff)", lambda s: f'{s["key_acc"]:.3f}'),
        ("Notehead precision", lambda s: f'{s["heads"]["precision"]:.3f}'),
        ("Notehead recall", lambda s: f'{s["heads"]["recall"]:.3f}'),
        ("Head type accuracy", lambda s: f'{s["head_type_acc"]:.3f}'),
        ("Pitch accuracy", lambda s: f'{s["pitch_acc"]:.3f}'),
        ("Duration accuracy", lambda s: f'{s["duration_acc"]:.3f}'),
        ("Accidental accuracy", lambda s: f'{s["accidental_acc"]:.3f} (n={s["accidental_n"]})'),
        ("Rest F1", lambda s: f'{s["rests"]["f1"]:.3f}'),
        ("Measures matching time signature", lambda s: f'{s["measures_ok"]}/{s["measures_total"]}'),
        ("Measure-level note F1", lambda s: f'{s["measure_onset_pitch"]["f1"]:.3f}'),
        ("MIDI onset+pitch F1", lambda s: f'{s["e2e_onset_pitch"]["f1"]:.3f}'),
        ("MIDI onset+pitch+duration F1", lambda s: f'{s["e2e_with_duration"]["f1"]:.3f}'),
        ("Pitch-sequence accuracy", lambda s: f'{s["pitch_sequence_acc"]:.3f}'),
        ("Pages with perfect MIDI", lambda s: f'{s["pages_e2e_perfect"]}/{s["pages"]}'),
        ("Seconds per page", lambda s: f'{s["seconds_per_page"]:.2f}'),
    ]
    names = list(groups)
    out = ["| Metric | " + " | ".join(names) + " |", "|---" * (len(names) + 1) + "|"]
    for label, fn in rows:
        out.append(f"| {label} | " + " | ".join(fn(groups[n]) for n in names) + " |")
    return "\n".join(out)


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("all_json")
    p.add_argument("--json")
    args = p.parse_args(argv)
    rows = json.load(open(args.all_json))
    groups = {"All": pooled(rows)}
    for d in (0, 1, 2):
        sub = [r for r in rows if r["degrade"] == d]
        if sub:
            groups[f"Degrade {d}"] = pooled(sub)
    for mode in ("single", "piano"):
        sub = [r for r in rows if r["mode"] == mode]
        if sub:
            groups[mode.capitalize()] = pooled(sub)
    print(table(groups))
    if args.json:
        json.dump(groups, open(args.json, "w"), indent=1)


if __name__ == "__main__":
    main()
