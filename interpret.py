# =============================================================================
#  interpret.py  --  Stage 6: pitch and rhythm, symbols -> note sequences
#  Version: 1.0.0
#
#  Changelog
#  ---------
#  1.0.0  (2026-10-25)
#    - Pitch: notehead step (from the tracked staff-line paths) -> letter and
#      octave through the staff's clef (env_detect.Clef.step_to_pitch), then
#      the alteration: an accidental on the note, else an earlier accidental
#      on the same letter+octave in the measure, else the key signature.
#      Accidentals reset at every detected barline.
#    - Rhythm: whole/half from hollow heads with/without stem, quarter and
#      shorter from filled heads and the beam/flag count, x1.5 for a dot.
#      Rests by type; a whole rest fills its measure.
#    - Timeline: each part (staff slot of env_config) is one
#      sequences.RelativeNotes; staves of that part are read in page order,
#      events left to right. Chords add several events at one node.
#    - Transposition from env_config is applied to the sounding pitch.
#    - Every note records its measure number in the part and its offset
#      inside that measure (for reports and measure-level scoring).
#    - Measure check: compares each measure's total duration with the time
#      signature, as an internal consistency measure. Barlines with no
#      events before them (e.g. a time signature read as a stroke) are
#      not counted as measures.
#
#  Depends on symbol_match.py >= 1.0.0, env_config.py >= 1.0.1,
#  sequences.py >= 1.1.0, events.py >= 1.1.0
# =============================================================================

from __future__ import annotations

from dataclasses import dataclass, field

from env_config import Environment
from env_detect import Clef, y_to_step
from events import NoteEvent, TICKS_PER_BEAT
from note import Note
from sequences import RelativeNotes, Sequences
from symbol_match import Chord, Recognition, Rest

SHARP_ORDER, FLAT_ORDER = "FCGDAEB", "BEADGCF"
PITCH_CLASS = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}
SHARP_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
ALTER = {"sharp": 1, "flat": -1, "natural": 0}


@dataclass
class NoteOut:
    """One recognised note, for reports and evaluation."""
    part: int
    staff: int
    onset: int          # ticks from the start of the part
    ticks: int
    midi: int           # sounding MIDI number
    name: str           # written pitch, e.g. "F#4"
    x: float
    y: float
    head: object = None
    measure: int = 0    # measure number within the part (from detected barlines)
    offset: int = 0     # ticks from the start of that measure


@dataclass
class Interpretation:
    sequences: list                  # RelativeNotes per part
    notes: list                      # NoteOut
    rests: list                      # (part, staff, onset, ticks, kind)
    measures: list                   # (staff, measure index, ticks found, ticks expected)
    labels: dict = field(default_factory=dict)   # id(Head) -> label text

    @property
    def measures_ok(self) -> tuple[int, int]:
        good = sum(1 for _, _, found, want in self.measures if found == want)
        return good, len(self.measures)


def key_alterations(fifths: int) -> dict[str, int]:
    if fifths > 0:
        return {letter: 1 for letter in SHARP_ORDER[:fifths]}
    return {letter: -1 for letter in FLAT_ORDER[:-fifths]}


def midi_to_note(midi: int, velocity: int) -> Note:
    """Note object for a MIDI number. note.py numbers 12*octave + pitch class,
    so middle C (scientific C4, MIDI 60) is octave 5 there."""
    return Note(SHARP_NAMES[midi % 12], velocity, midi // 12)


def rest_ticks(rest: Rest, measure_ticks: int) -> int:
    if rest.kind == "whole":
        return measure_ticks
    return {"half": 960, "quarter": 480, "eighth": 240, "sixteenth": 120}[rest.kind]


def interpret(rec: Recognition, removal, env: Environment) -> Interpretation:
    beats, beat_type = env.time_signature
    measure_ticks = int(round(beats * 4 / beat_type * TICKS_PER_BEAT))
    velocity = env.default_velocity
    parts = {p.slot: p for p in env.parts}
    sequences = {slot: Sequences.relative().start() for slot in parts}
    clocks = {slot: 0 for slot in parts}
    part_measure = {slot: 0 for slot in parts}     # running measure number per part
    notes, rests, measures, labels = [], [], [], {}

    settings = {s.index: s for s in env.staves}
    for ss in rec.staves:
        setting = settings.get(ss.index)
        if setting is None:
            continue
        slot = setting.slot
        part = parts[slot]
        clef = Clef.parse(setting.clef)
        key = key_alterations(setting.key)
        paths = removal.line_paths[ss.index]
        seq: RelativeNotes = sequences[slot]

        events = [(c.x, c) for c in ss.chords] + [(r.x, r) for r in ss.rests]
        events.sort(key=lambda e: e[0])
        bars = list(ss.barlines)
        measure_acc: dict[tuple[str, int], int] = {}
        measure_index, measure_sum, measure_events = 0, 0, 0

        for x, ev in events:
            while bars and x > bars[0]:
                if measure_events:          # a "measure" with nothing in it is a header stroke
                    measures.append((ss.index, measure_index, measure_sum, measure_ticks))
                    measure_index += 1
                    part_measure[slot] += 1
                bars.pop(0)
                measure_acc.clear()
                measure_sum, measure_events = 0, 0
            if isinstance(ev, Rest):
                d = rest_ticks(ev, measure_ticks)
                rests.append((slot, ss.index, clocks[slot], d, ev.kind))
                seq.advance(duration_ticks=d)
            else:
                d = ev.ticks
                for h in ev.heads:
                    step = int(round(y_to_step(paths, h.match.cx, h.match.cy)))
                    letter, octave = clef.step_to_pitch(step)
                    if h.accidental:
                        alter = ALTER[h.accidental]
                        measure_acc[(letter, octave)] = alter
                    else:
                        alter = measure_acc.get((letter, octave), key.get(letter, 0))
                    midi = 12 * (octave + 1) + PITCH_CLASS[letter] + alter + part.transpose
                    if not 0 <= midi <= 127:
                        continue
                    seq.add(NoteEvent(midi_to_note(midi, velocity), duration_ticks=d))
                    written = letter + {1: "#", -1: "b", 0: ""}[alter] + str(octave)
                    notes.append(NoteOut(slot, ss.index, clocks[slot], d, midi, written,
                                         h.match.cx, h.match.cy, h, part_measure[slot], measure_sum))
                    labels[id(h)] = written
                seq.advance(duration_ticks=d)
            clocks[slot] += d
            measure_sum += d
            measure_events += 1
        if measure_events:                  # the staff's last measure
            measures.append((ss.index, measure_index, measure_sum, measure_ticks))
            part_measure[slot] += 1

    return Interpretation([sequences[s] for s in sorted(sequences)], notes, rests, measures, labels)
