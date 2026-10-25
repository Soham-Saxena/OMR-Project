# Optical Music Recognition: sheet-music image to MIDI

CSE 3176 Foundations of Computer Vision Lab mini project. Classical computer
vision (no neural networks): staff detection and removal, template matching
with normalised cross-correlation, rule-based pitch and rhythm, MIDI output.

## Run

```
pip install -r requirements.txt
python omr.py page.png --yes --time 4/4 --bpm 100      # writes page.mid, page.overlay.png, page.notes.json
python omr_gui.py                                       # desktop interface (Tkinter)
python env_config.py page.png                           # set clef/key/instrument/tempo interactively, saves page.env.json
python omr.py page.png --env page.env.json
```

## Pipeline and modules

| Stage | Module | What it does |
|---|---|---|
| 1 | `preprocess.py` | grey conversion, skew estimation and correction (projection profile) |
| 2 | `staff_detect.py` | Otsu binarisation, staff space from run lengths, projection peaks, DBSCAN grouping, 5-line fit |
| 3 | `staff_remove.py` | line tracking and run-length staff removal, constrained regrowth |
| 4 | `env_detect.py`, `env_config.py` | systems, clefs, key signatures, time-signature location; user settings |
| 5 | `detect.py` | connected ink regions and dense proposals on the staff-free page |
| 6 | `symbol_match.py` | template matching (NCC), NMS, stems, beams/flags, dots, accidentals, rests, barlines |
| 7 | `interpret.py` | pitch from staff position, clef, key and accidentals; durations; note sequences |
| 8 | `note.py`, `events.py`, `sequences.py`, `writer.py` | note/event/sequence model and MIDI writer (mido) |
| - | `omr.py`, `omr_gui.py` | command-line pipeline and GUI |
| - | `make_templates.py`, `templates/` | template bank (drawn noteheads, FreeSerif accidentals and rests) |

## Evaluation

```
python eval/synth_score.py --out eval/pages --seeds 0-29        # printed test pages + ground truth
python eval/evaluate.py eval/pages --out eval/results           # per-page scores
python eval/evaluate.py eval/pages --out eval/results_confirmed --confirmed
python eval/summarise.py eval/results/all.json                  # pooled tables
python eval/evaluate_cvc.py                                      # handwritten pages vs eval/cvc_manual_gt.json
```

Thresholds in `symbol_match.py` were tuned on pages from seeds 100-111 only;
seeds 0-29 are the test set.
