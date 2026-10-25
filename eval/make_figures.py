# =============================================================================
#  make_figures.py  --  figures for the project report, from real runs
#  Version: 1.0.0
#
#  Changelog
#  ---------
#  1.0.0  (2026-10-25)
#    - Pipeline stages on one handwritten page (staff diagnostics, removal,
#      change map, environment), template bank, NCC response map, recognition
#      overlay with pitch labels, piano roll vs ground truth, score and hole
#      distributions on the dev pages, results by degradation, rotation test,
#      stage timings, handwritten failure case.
#
#  Usage:  python eval/make_figures.py --test eval/pages --dev eval/dev
#          --auto eval/results/all.json --rotation rotation.json --out figures
# =============================================================================

from __future__ import annotations

import argparse
import json
import os
import sys

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": INK2,
                     "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
                     "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
                     "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
                     "legend.frameon": False, "figure.dpi": 150})


def save(fig, out, name):
    fig.savefig(os.path.join(out, name), bbox_inches="tight", dpi=200, facecolor="white")
    plt.close(fig)
    print("wrote", name)


def crop_save(img, out, name, box=None, width=None):
    if box:
        x0, y0, x1, y1 = box
        img = img[y0:y1, x0:x1]
    if width and img.shape[1] > width:
        s = width / img.shape[1]
        img = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    cv2.imwrite(os.path.join(out, name), img)
    print("wrote", name)


def stage_figures(out):
    from env_detect import detect_environment, plot_environment
    from preprocess import deskew
    from staff_detect import detect_staves, plot_diagnostics, plot_overlay
    from staff_remove import change_map, remove_from_detection
    path = os.path.join(ROOT, "Separate_Set", "PNG_GT_Gray_p008.png")
    gray, _ = deskew(cv2.imread(path, cv2.IMREAD_GRAYSCALE))
    res = detect_staves(gray)
    res["gray"] = gray
    u = res["interline"]
    fig = plot_diagnostics(gray, res)
    save(fig, out, "fig_staff_diagnostics.png")
    fig = plot_overlay(gray, res, width_in=10)
    save(fig, out, "fig_staff_overlay.png")
    rem = remove_from_detection(res)
    s0, s1 = res["staves"][0], res["staves"][1]
    y0, y1 = int(s0.top - 3 * u), int(s1.bottom + 3 * u)
    x1 = int(min(gray.shape[1], 52 * u))
    before = 255 - res["binary"][y0:y1, :x1]
    after = 255 - rem.cleaned[y0:y1, :x1]
    sep = np.full((12, x1), 160, np.uint8)
    crop_save(np.vstack([before, sep, after]), out, "fig_removal_before_after.png", width=1800)
    cm = cv2.cvtColor(change_map(rem), cv2.COLOR_RGB2BGR)
    crop_save(cm, out, "fig_removal_changes.png", (0, y0, x1, y1), width=1800)
    env = detect_environment(res, rem)
    fig = plot_environment(res, env, width_in=10)
    save(fig, out, "fig_environment.png")
    cv2.imwrite(os.path.join(out, "fig_input_handwritten.png"),
                cv2.resize(gray, None, fx=0.35, fy=0.35, interpolation=cv2.INTER_AREA))


def template_figure(out):
    idx = json.load(open(os.path.join(ROOT, "templates", "templates.json")))
    fig, axes = plt.subplots(1, len(idx["templates"]), figsize=(12, 1.9))
    for ax, t in zip(axes, idx["templates"]):
        img = cv2.imread(os.path.join(ROOT, "templates", t["file"]), 0)
        ax.imshow(255 - img, cmap="gray", vmin=0, vmax=255)
        ax.set_title(t["name"].replace("_", "\n", 1), fontsize=7, color=INK)
        ax.set_xticks([]), ax.set_yticks([])
        ax.grid(False)
        for s in ax.spines.values():
            s.set_visible(True)
            s.set_color(GRID)
    save(fig, out, "fig_templates.png")


def matching_figures(test_dir, out):
    from omr import overlay, run_pipeline
    from symbol_match import load_templates
    page = os.path.join(test_dir, "synth_013.png")
    gt = json.load(open(page.replace(".png", ".json")))
    gray = cv2.imread(page, 0)
    res = run_pipeline(gray, time_signature=tuple(gt["time_signature"]))
    u = res.staff["interline"]
    st = res.staff["staves"][0]
    y0, y1 = int(st.top - 4.5 * u), int(st.bottom + 4.5 * u)
    x0, x1 = int(res.recognition.staves[0].start_x), int(res.recognition.staves[0].start_x + 40 * u)
    band = (res.recognition.work[y0:y1, x0:x1] > 0).astype(np.float32)
    t = [t for t in load_templates(u) if t.name == "filled_0"][0]
    score = cv2.matchTemplate(band, t.image, cv2.TM_CCOEFF_NORMED)
    fig, axes = plt.subplots(2, 1, figsize=(10, 5.0), sharex=True)
    axes[0].imshow(1 - band, cmap="gray", aspect="equal")
    axes[0].set_title("Staff-free image (one staff band)", color=INK, loc="left")
    pad = np.full(band.shape, np.nan, np.float32)
    pad[t.image.shape[0] // 2:t.image.shape[0] // 2 + score.shape[0],
        t.image.shape[1] // 2:t.image.shape[1] // 2 + score.shape[1]] = score
    im = axes[1].imshow(pad, cmap="viridis", vmin=-0.3, vmax=1.0, aspect="equal")
    axes[1].set_title("NCC response of one filled-notehead template", color=INK, loc="left")
    for ax in axes:
        ax.set_xticks([]), ax.set_yticks([])
        ax.grid(False)
    cb = fig.colorbar(im, ax=list(axes), orientation="horizontal", fraction=0.05, pad=0.04, aspect=50)
    cb.set_label("NCC score")
    save(fig, out, "fig_ncc_response.png")
    ov = cv2.cvtColor(overlay(res), cv2.COLOR_RGB2BGR)
    s0, s1 = res.staff["staves"][0], res.staff["staves"][1]
    x_end = int(min(gray.shape[1], max(res.removal.line_extents[1][:, 1]) + 2 * u))
    crop_save(ov, out, "fig_overlay_printed.png",
              (0, int(s1.top - 4.5 * u), x_end, int(s1.bottom + 4.5 * u)), width=2000)
    cv2.imwrite(os.path.join(out, "fig_input_printed.png"),
                cv2.resize(gray, None, fx=0.45, fy=0.45, interpolation=cv2.INTER_AREA))

    # piano roll: part 0, written MIDI vs ground truth
    from evaluate import midi_notes
    from omr import write_midi
    stem = os.path.join(out, "_roll")
    write_midi(res, stem)
    det = [n for n in midi_notes(stem + ".mid") if n[0] == 0]
    os.remove(stem + ".mid")
    gtn = [(h["onset"], h["pitch"], h["ticks"]) for h in gt["heads"] if h["part"] == 0]
    limit = 16 * 480
    fig, ax = plt.subplots(figsize=(10, 3.2))
    for o, p, d in gtn:
        if o < limit:
            ax.add_patch(plt.Rectangle((o / 480, p - 0.45), d / 480, 0.9, color=BLUE, alpha=0.35, lw=0))
    for _, o, p, d in det:
        if o < limit:
            ax.add_patch(plt.Rectangle((o / 480, p - 0.18), d / 480 - 0.04, 0.36, color=ORANGE, lw=0))
    ps = [p for o, p, d in gtn if o < limit] + [p for _, o, p, d in det if o < limit]
    ax.set_xlim(0, limit / 480)
    ax.set_ylim(min(ps) - 2, max(ps) + 2)
    ax.set_xlabel("Time (quarter notes)")
    ax.set_ylabel("MIDI note number")
    ax.plot([], [], color=BLUE, alpha=0.35, lw=8, label="Ground truth")
    ax.plot([], [], color=ORANGE, lw=3, label="Recognised (written MIDI file)")
    ax.legend(loc="upper right", ncol=2)
    save(fig, out, "fig_piano_roll.png")


def dev_distributions(dev_dir, out):
    import symbol_match
    from evaluate import greedy_match
    from omr import run_pipeline
    from symbol_match import _path_at
    saved = dict(symbol_match.THRESHOLDS)
    symbol_match.THRESHOLDS["notehead_filled"] = 0.45
    symbol_match.THRESHOLDS["notehead_hollow"] = 0.45
    tp, fp, holes = [], [], {"filled": [], "hollow on a line": [], "hollow in a space": []}
    for s in range(100, 112):
        png = os.path.join(dev_dir, f"synth_{s}.png")
        gt = json.load(open(png.replace(".png", ".json")))
        u = gt["u"]
        g8 = cv2.imread(png, 0)
        res = run_pipeline(g8, time_signature=tuple(gt["time_signature"]))
        gray = g8.astype(np.float32)
        b = res.staff["binary"] > 0
        ink, paper = np.median(gray[b]), np.median(gray[~b])
        t = res.staff["metrics"].line_thickness
        dh = [(h, si) for si, ss in enumerate(res.recognition.staves) for c in ss.chords for h in c.heads]
        pairs = greedy_match([(h["cx"], h["cy"]) for h in gt["heads"]],
                             [(h.match.cx, h.match.cy) for h, _ in dh], 0.5 * u)
        matched = {j: i for i, j in pairs}
        for j, (h, si) in enumerate(dh):
            if h.match.cls == "notehead_filled":
                (tp if j in matched else fp).append(h.match.score)
            if j in matched:
                g = gt["heads"][matched[j]]
                if g["kind"] == "whole":
                    continue
                paths = res.removal.line_paths[si]
                line_y = float(_path_at(paths, h.step)[int(round(h.match.cx))]) if h.step % 2 == 0 else None
                sc = symbol_match.hole_score(g8, (ink, paper), h.match.cx, h.match.cy, line_y, t, u)
                if g["kind"] == "filled":
                    holes["filled"].append(sc)
                elif h.step % 2 == 0:
                    holes["hollow on a line"].append(sc)
                else:
                    holes["hollow in a space"].append(sc)
    symbol_match.THRESHOLDS.update(saved)
    fig, ax = plt.subplots(figsize=(6.4, 2.8))
    bins = np.linspace(0.45, 1.0, 45)
    ax.hist(tp, bins=bins, color=BLUE, alpha=0.85, label=f"true noteheads (n={len(tp)})")
    ax.hist(fp, bins=bins, color=ORANGE, alpha=0.85, label=f"false matches (n={len(fp)})")
    ax.axvline(saved["notehead_filled"], color=INK, lw=1, ls="--")
    ax.text(saved["notehead_filled"] + 0.005, ax.get_ylim()[1] * 0.85, f"threshold {saved['notehead_filled']}",
            color=INK, fontsize=8)
    ax.set_yscale("log")
    ax.set_xlabel("NCC score of filled-notehead matches (development pages)")
    ax.set_ylabel("Count (log)")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, fontsize=8)
    save(fig, out, "fig_score_threshold.png")
    fig, ax = plt.subplots(figsize=(6.4, 2.6))
    bins = np.linspace(-0.45, 1.0, 59)
    for (name, vals), col in zip(holes.items(), (BLUE, ORANGE, AQUA)):
        ax.hist(vals, bins=bins, color=col, alpha=0.85, label=f"{name} (n={len(vals)})")
    ax.axvline(symbol_match.HOLE_MIN, color=INK, lw=1, ls="--")
    ax.set_yscale("log")
    ax.set_xlabel("Hole score: 0 = ink level, 1 = paper level (development pages)")
    ax.set_ylabel("Count (log)")
    ax.text(symbol_match.HOLE_MIN + 0.01, 0.9, f"threshold {symbol_match.HOLE_MIN}", color=INK, fontsize=8,
            transform=ax.get_xaxis_transform())
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3, fontsize=8)
    save(fig, out, "fig_hole_score.png")
    json.dump({"tp_min": float(min(tp)), "fp_max": float(max(fp)), "n_tp": len(tp), "n_fp": len(fp),
               "fp_above": int(sum(x >= saved["notehead_filled"] for x in fp)),
               "tp_below": int(sum(x < saved["notehead_filled"] for x in tp)),
               "holes": {k: [float(min(v)), float(max(v)), len(v)] for k, v in holes.items()}},
              open(os.path.join(out, "dev_distributions.json"), "w"), indent=1)


def results_figures(sum_auto, rotation, out):
    s = json.load(open(sum_auto))
    groups = ["Degrade 0", "Degrade 1", "Degrade 2"]
    metrics = [("Notehead recall", lambda g: g["heads"]["recall"]),
               ("Pitch", lambda g: g["pitch_acc"]),
               ("Duration", lambda g: g["duration_acc"]),
               ("Rest F1", lambda g: g["rests"]["f1"]),
               ("Measure-\nlevel F1", lambda g: g["measure_onset_pitch"]["f1"]),
               ("MIDI onset\n+pitch F1", lambda g: g["e2e_onset_pitch"]["f1"])]
    fig, ax = plt.subplots(figsize=(8, 3.2))
    w = 0.26
    x = np.arange(len(metrics))
    labels = {"Degrade 0": "clean", "Degrade 1": "blur + noise", "Degrade 2": "strong blur + noise + specks"}
    for k, (gname, col) in enumerate(zip(groups, (BLUE, ORANGE, AQUA))):
        vals = [fn(s[gname]) for _, fn in metrics]
        ax.bar(x + (k - 1) * w, vals, w - 0.03, color=col, label=f"{labels[gname]} ({s[gname]['pages']} pages)")
    ax.set_xticks(x, [m for m, _ in metrics])
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3, fontsize=8)
    ax.grid(axis="x", visible=False)
    save(fig, out, "fig_results_degradation.png")
    rot = json.load(open(rotation))
    angles = ["0", "0.2", "0.5", "1.0", "2.0"]
    fig, ax = plt.subplots(figsize=(6.4, 2.6))
    for mode, col, lab in (("raw", ORANGE, "without deskew"), ("deskew", BLUE, "with deskew (preprocess.py)")):
        ys = [rot[f"{a}|{mode}"]["seq_acc"] for a in angles]
        ax.plot([float(a) for a in angles], ys, color=col, lw=2, marker="o", ms=5, label=lab)
    ax.set_xlabel("Page rotation (degrees)")
    ax.set_ylabel("Pitch-sequence accuracy")
    ax.set_ylim(0, 1.05)
    ax.legend(loc="lower left")
    save(fig, out, "fig_rotation.png")
    tm = s["All"]["timings"]
    fig, ax = plt.subplots(figsize=(6.4, 2.4))
    names = list(tm)
    ax.barh(names[::-1], [tm[n] for n in names][::-1], color=BLUE, height=0.6)
    for i, n in enumerate(names[::-1]):
        ax.text(tm[n] + 0.01, i, f"{tm[n]:.2f} s", va="center", color=INK2, fontsize=8)
    ax.set_xlabel("Mean time per page (s), 30 test pages")
    ax.grid(axis="y", visible=False)
    save(fig, out, "fig_timings.png")


def handwritten_failure(out):
    from omr import overlay, run_pipeline
    path = os.path.join(ROOT, "Separate_Set", "PNG_GT_Gray_p002.png")
    res = run_pipeline(cv2.imread(path, cv2.IMREAD_GRAYSCALE))
    ov = cv2.cvtColor(overlay(res), cv2.COLOR_RGB2BGR)
    st = res.staff["staves"][0]
    u = res.staff["interline"]
    crop_save(ov, out, "fig_overlay_handwritten.png",
              (100, int(st.top - 4 * u), int(min(ov.shape[1], 1850)), int(st.bottom + 2.5 * u)), width=1800)


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--test", required=True)
    p.add_argument("--dev", required=True)
    p.add_argument("--auto", required=True, help="summarise.py --json output for the automatic run")
    p.add_argument("--rotation", required=True)
    p.add_argument("--out", default="figures")
    args = p.parse_args(argv)
    os.makedirs(args.out, exist_ok=True)
    sys.path.insert(0, os.path.join(ROOT, "eval"))
    stage_figures(args.out)
    template_figure(args.out)
    matching_figures(args.test, args.out)
    dev_distributions(args.dev, args.out)
    results_figures(args.auto, args.rotation, args.out)
    handwritten_failure(args.out)


if __name__ == "__main__":
    main()
