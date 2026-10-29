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

For a printed piano score in compound time, give the time signature and the
beat unit of the tempo mark, e.g. for "quarter = 95" in 12/8:

```
python omr.py page.png --yes --time 12/8 --beat-unit quarter --bpm 95
```

## Input tips

- Best input: a **PDF exported from MuseScore** (or any notation program):
  `python omr.py score.pdf --yes` reads every page into one MIDI file.
  PDF pages are rendered at 300 dpi with PyMuPDF (`pip install pymupdf`) or
  poppler's `pdftoppm`.
- Next best: PNG export at 300 dpi. Several images: `python omr.py p1.png p2.png`.
- Screenshots work but are the weakest input: at normal zoom a staff space is
  only 6-10 pixels (a small solo staff even less), so the program has to
  enlarge the page 2-4x and thin details (beams on staff lines, the holes of
  half notes, rests) start to blur together.

## Fixes in this version (2026-10-25 / 2026-10-28)

Found on a real printed piano score (5 flats, 12/8, treble-8 clef changes in
the left hand), where the MIDI file came out wrong:

| Problem | Module (version) | Fix |
|---|---|---|
| Key signature stopped after 1-2 flats (staff removal cut a flat's stem from its loop; split sharps too) | `env_detect.py` 1.0.1 | pieces of one accidental are joined again before the key is read |
| 12/8 not located (the digits touch), so they were read as notes | `env_detect.py` 1.0.1 | a staff-high joined glyph right after the key is accepted as a time signature (two wide halves, digit loops, no stem) |
| Bass clef with one dot lost read as F on line 3 | `env_detect.py` 1.0.1 | one dot means the usual F4 |
| Mid-staff clef changes ignored (left hand in treble / treble-8) | `make_templates.py` 1.1.0, `symbol_match.py` 1.1.0, `interpret.py` 1.1.0 | G and F clef templates matched inside the staff, octave 8 detected, notes after a change use that clef |
| Tempo mark note, text and dynamics read as noteheads | `symbol_match.py` 1.1.0 | a head outside the staff needs its ledger lines; stemless filled heads dropped |
| Arpeggio line taken as a barline | `interpret.py` 1.1.0 | a barline must be found in every staff of the system |
| Displaced half rest read as a whole rest | `symbol_match.py` 1.1.0 | whole/half from which edge touches a staff line |
| Gap of a final double barline read as a half note | `symbol_match.py` 1.1.0 | rejected when staff-high strokes lie on both sides |
| Ties played as two notes | `symbol_match.py` 1.1.0, `interpret.py` 1.1.0 | tie arcs found (pieces cut by staff removal joined) and the notes merged |
| (2026-10-28) Around 19 s in Drowning Love (measure 6) the hands went out of step: a second voice's chords sit on the cross-staff sixteenth beam, so beam counts were misread | `symbol_match.py` 1.2.0, `interpret.py` 1.2.0 | chords joined by one beam are grouped (also across staves); a measure that does not add up is re-read with each beam group as equal notes filling whole beats, kept only if the measure then fits |
| (2026-10-28) Run with the 4/4 default on a 12/8 piece: every measure was the wrong length | `omr.py` 1.1.0, `omr_gui.py` 1.0.1 | if most measures agree on another length, that time signature is used (printed as a note); `--time` still recommended |
| (2026-10-28, Happy Birthday screenshot) Half notes and half-note chords not found, most of all on small pages (staff space 10-12 px: 44-57 % of noteheads found) | `omr.py` 1.2.0 | pages with a staff space under 16 px are enlarged to 20 px before recognition |
| Half-note chords in a MuseScore (Leland) font: the thin rims of heads a second or third apart are cut by staff removal | `symbol_match.py` 1.2.0 | weaker hollow matches (NCC 0.40-0.50) are kept when the head has a stem, an enclosed hole in the page with staff lines, and another head beside it on the same stem |
| Time signature 3/4 or 4/4 in the treble staff boxed as a note, "7" of a chord symbol boxed as a rest | `env_detect.py` 1.0.2, `symbol_match.py` 1.2.0 | touching digits (4 over 4) are located as a time signature; the overlay now shows rejected matches as thin grey boxes, only the symbols actually used are in colour |
| Key of flats read as sharps in Leland | `env_detect.py` 1.0.2 | a flat's loop piece may start up to 0.35U from its stem |
| (Butterfly Waltz screenshot) Violin notes not read at all: the violin staff is printed at 75 % size, and the staff detector only accepted staves within 25 % of the page's staff space | `staff_detect.py` 3.3.0, `symbol_match.py` 1.2.0, `env_detect.py` 1.0.2 | small staves (55-100 %) are detected; every staff is read with templates and distances scaled to its own staff space |
| Screenshot: grey side panel and taskbar | `preprocess.py` 1.1.0 | everything outside the page is painted white first |
| Tiny pages: staff space estimate failed below about 7 px | `staff_detect.py` 3.3.0, `omr.py` 1.3.0 | white-run search starts at 1.2t; enlargement uses the smallest staff (up to 4x) |
| Violin staff and piano split into separate systems when the clef touches the opening line | `env_detect.py` 1.0.2 | a straight vertical stroke from the top of one staff to the bottom of the next also joins them |
| Beams lying on a staff line erased with it (small staves) | `staff_remove.py` 1.2.0 | a run counts as pure staff line up to 1.5t (was 2t) |
| Whole-measure rests read as whole notes, or missed, on blurred pages | `symbol_match.py` 1.2.0 | a solid bar touching a line is a rest; rests are also found by shape when templates fail |
| Bits of a beam crossing lines read as half notes | `symbol_match.py` 1.2.0 | a "head" with beam ink 1.3U to both sides is dropped |
| Thin system line read as a C clef | `env_detect.py` 1.0.2 | a clef must be at least 0.6U wide |
| Not yet seen, now handled: 8va / 8vb lines | `symbol_match.py`, `interpret.py` | dashed octave lines found, notes under them shifted an octave |
| Not yet seen, now handled: staves hidden on later systems (MuseScore hides empty staves) | `env_detect.py` 1.0.2 | systems with fewer staves are matched to parts by clef and size |
| Not yet seen, now handled: triplets | `interpret.py` 1.2.0 | beam-group repair also tries triplet values when a staff's notes do not fill the measure |
| Not yet seen, now handled: PDF and multi-page input, transparent PNGs, 16-bit images, 600 dpi scans | `omr.py` 1.3.0, `preprocess.py` 1.1.0, `omr_gui.py` 1.1.0 | PDF pages rendered at 300 dpi; pages joined in one MIDI file; transparency composed on white; huge pages reduced |
| (2026-10-29, Megalovania) Alto (C) clef read as a bass clef, so every viola pitch was wrong | `env_detect.py` 1.0.3 | the clef's bars, cut off by staff removal, are joined back to its body |
| Hands drifted apart after one misread note | `interpret.py` 1.1.0 | timing per system: a staff whose notes add up to the time signature sets the onsets; each measure restarts on the time-signature grid |

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
python eval/synth_score.py --out eval/pages_x --seeds 200-229 --extended   # 6/8, 12/8, 5-accidental keys, ties, clef changes, tempo marks
python eval/rotation_test.py --work eval/rot --json rotation.json          # rotation with/without deskew
# MuseScore-style pages: get Leland.otf from https://github.com/MuseScoreFonts/Leland (SIL OFL)
python eval/synth_score.py --out eval/pages_leland --seeds 400-411 --extended --music-font Leland.otf
python eval/synth_score.py --out eval/hb --seeds 0 --u 12 --degrade 0 --music-font Leland.otf --score eval/scores/happy_birthday.json
python eval/synth_score.py --out eval/bw --seeds 0 --u 10 --degrade 0 --music-font Leland.otf --score eval/scores/butterfly_waltz.json
# also: butterfly_waltz_hidden.json (hidden staff), triplets.json
```

Known limits: repeats and voltas are not expanded, grace notes and
multi-measure rests are not read, time signatures are located but their
digits are not read (give `--time`; a wrong guess is corrected when most
measures agree on another length), and mid-piece key or time changes are
not followed.

Results with this version (time signature given, clef/key detected automatically):

| Test set | Notehead P / R | Pitch | Duration | Measure-level F1 | MIDI onset+pitch F1 |
|---|---|---|---|---|---|
| seeds 0-29 (original features) | 1.000 / 0.999 | 0.996 | 0.993 | 0.985 | 0.978 |
| seeds 200-229 (`--extended`) | 1.000 / 0.999 | 0.977 | 0.962 | 0.978 | 0.969 |
| seeds 400-411, Leland font, `--extended` | 0.998 / 0.998 | 0.962 | 0.969 | 0.963 | 0.963 |
| seeds 500-505, Leland font, staff space 12 px | 0.991 / 0.993 | 0.973 | 0.939 | 0.906 | 0.674 |
| Happy Birthday (Leland), staff space 10 / 12 / 20 px | 1.00 / 1.00 | 1.00 | 1.00 | - | 1.00 / 1.00 / 1.00 |
| Butterfly Waltz (violin at 75 % + piano, 8va), staff space 20 / 10 / 8 px | 1.00 / 1.00, 1.00 / 1.00, 0.94 / 1.00 | 1.00 / 1.00 / 0.87 | 1.00 / 0.95 / 0.79 | - | 1.00 / 0.98 / 0.78 |
| Same with the violin staff hidden on one system | 1.00 / 1.00 | 1.00 | 1.00 | - | 1.00 |
| Triplet study | 1.00 / 1.00 | 1.00 | 1.00 | - | 1.00 |

On small pages with many sixteenths (seeds 500-505) beam counts are still the
weak point: scan or export at 150 dpi or more where possible.

Handwritten CVC-MUSCIMA pages (environment only): 81/81 staves, 79/81 clefs,
75/81 keys, 22 of 25 time signatures located with no false ones.

Thresholds in `symbol_match.py` were tuned on pages from seeds 100-111 only;
seeds 0-29 are the test set.
