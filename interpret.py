# =============================================================================
#  interpret.py  --  Stage 6: pitch and rhythm, symbols -> note sequences
#  Version: 1.2.0
#
#  Changelog
#  ---------
#  1.2.0  (2026-10-28)
#    - Measure repair with beam groups: when a measure does not add up to
#      the time signature, every beam group (symbol_match 1.2.0) is read as
#      n equal notes filling whole beats (dotted quarter in 6/8, 9/8, 12/8),
#      using the value the correctly read chords vote for; dotted chords of
#      another voice keep theirs. The repair is kept only if the measure then
#      adds up. Fixes runs of sixteenths beamed across the staves with a
#      second voice's chords on the same beam (wrong beam counts made the
#      hands drift apart in the MIDI file). A dotted chord of the other
#      voice on that beam lasts until that voice's next chord on the beam.
#    - Interpretation.end_ticks: where the page ends, for multi-page input.
#    - Beam-group repair also tries triplet values (160 and 80 ticks), so a
#      measure with triplets adds up. It is tried whenever a staff's own
#      notes do not fill the measure (before: only when the whole system's
#      measure did not), and kept when fewer staves are then off.
#    - Notes under an octave line (symbol_match.find_ottavas) sound an
#      octave higher (8va) or lower (8vb).
#  Depends on symbol_match.py >= 1.2.0 (Chord.beam_group).
#  1.1.0  (2026-10-25)
#    - Timing per SYSTEM instead of per staff. All chords and rests of the
#      staves of a system are sorted by x; events within COLUMN_TOL_U of each
#      other form a column and start together; the next column starts when
#      the earliest still-sounding event ends. This handles beams that cross
#      between the two staves of a piano part, several voices on one staff,
#      and keeps the parts of a system in step. Each measure starts where
#      the previous one ended; barlines of all staves of the system are
#      merged.
#    - Mid-staff clef changes from symbol_match (with the octave 8) decide
#      the pitch of every note after them.
#    - Tied notes are merged into one longer note.
#    - Parts are written as sequences.AbsoluteNotes with the computed onsets.
#    - The measure check is per system measure.
#    - A barline must be found in every staff of its system.
#    - A whole (measure) rest starts at its measure's start, whatever its x.
#    - Each staff of a measure is also timed on its own; a staff whose
#      events add up to the time signature is trusted for the columns it
#      plays in, so a duration misread in one hand no longer moves the
#      other hand (column_onsets). Without such a staff the system-wide
#      timing is used.
#    - The next measure starts one time-signature length after the previous
#      one when the measured length is within SNAP_FRACTION of it, so a
#      single misread duration (now shared by every staff of the system)
#      does not shift every later note.
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
from sequences import Sequences
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
    sequences: list                  # AbsoluteNotes per part
    notes: list                      # NoteOut
    rests: list                      # (part, staff, onset, ticks, kind)
    measures: list                   # (first staff of the system, measure number, ticks found, ticks expected)
    labels: dict = field(default_factory=dict)   # id(Head) -> label text
    end_ticks: int = 0                           # where the page's music ends (next page starts here)

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


COLUMN_TOL_U = 0.5      # events this close in x (within a system) start together
SNAP_FRACTION = 0.30    # measures within this fraction of the time signature advance the clock by it


def _merge_close(xs: list[float], tol: float) -> list[float]:
    out = []
    for x in sorted(xs):
        if not out or x - out[-1] > tol:
            out.append(x)
    return out


@dataclass
class Repair:
    """Duration changes for one measure: new ticks per chord, and chords of
    another voice whose end must not decide when the next column starts."""
    ticks: dict = field(default_factory=dict)
    other_voice: set = field(default_factory=set)
    group_of: dict = field(default_factory=dict)     # id(chord) -> beam group
    group_step: dict = field(default_factory=dict)   # beam group -> note value d

    def __bool__(self) -> bool:
        return bool(self.ticks or self.other_voice)


def _duration(ev, measure_ticks: int, override: Repair | None = None) -> int:
    if isinstance(ev, Rest):
        return rest_ticks(ev, measure_ticks)
    if override and id(ev) in override.ticks:
        return override.ticks[id(ev)]
    return ev.ticks


def beat_ticks(beats: int, beat_type: int) -> int:
    """Length of one beat: a dotted quarter in 6/8, 9/8, 12/8, else 1/beat_type."""
    unit = 4 * TICKS_PER_BEAT // beat_type
    return 3 * unit if beat_type == 8 and beats % 3 == 0 and beats > 3 else unit


def repair_beam_groups(columns: list[list], beat: int, u: float) -> Repair:
    """New durations for the chords of beam groups that do not fill whole beats.

    A beamed group of n columns of equal notes lasts n*d, a whole number of
    beats. When a second voice's beam lies on top of the group, or a stem
    is shared, some of its chords get a wrong beam count; the group's value
    d is then the note value that (a) the correctly read chords vote for and
    (b) makes n*d a whole number of beats. Undotted chords of the group get
    d; dotted chords (another voice) keep their value, and their ends do
    not decide when the group's next column starts."""
    groups: dict[int, list] = {}
    for col in columns:
        for x, ss, ev in col:
            if isinstance(ev, Chord) and ev.beam_group >= 0:
                groups.setdefault(ev.beam_group, []).append((x, ev))
    override = Repair()
    for members in groups.values():
        xs = _merge_close([x for x, _ in members], COLUMN_TOL_U * u)
        n = len(xs)
        if n < 2:
            continue
        votes = [ev.base_ticks for _, ev in members if not ev.dots and 0 < ev.beams]
        # 160 / 80: triplet eighths / sixteenths (a "3" over the beam), tried
        # only because the measure did not add up with the printed values
        options = [d for d in (60, 80, 120, 160, 240) if (n * d) % beat == 0]
        if not options:
            continue
        d = max(options, key=lambda d: (votes.count(d), -abs(n * d - beat)))
        override.group_step[members[0][1].beam_group] = d
        for _, ev in members:
            override.group_of[id(ev)] = ev.beam_group
            if ev.dots:
                override.other_voice.add(id(ev))
            elif ev.ticks != d:
                override.ticks[id(ev)] = d
    return override


def _chord_of(note, columns):
    for col in columns:
        for _, _, ev in col:
            if isinstance(ev, Chord) and note.head in ev.heads:
                return ev
    return None


def _earliest_end_timeline(columns: list[list], measure_ticks: int, keep=lambda e: True, override=None):
    """Onset (ticks from the measure start) of every column holding a kept
    event: the next column starts when the earliest sounding event ends."""
    t, ends, onsets = 0, [], {}
    for ci, col in enumerate(columns):
        evs = [e for e in col if keep(e)]
        if not evs:
            continue
        onsets[ci] = t
        ends += [t + _duration(e[2], measure_ticks, override) for e in evs
                 if not (override and id(e[2]) in override.other_voice)]
        if not ends:
            ends = [t + _duration(e[2], measure_ticks, override) for e in evs]
        later = [e for e in ends if e > t]
        if later:
            t = min(later)
    return onsets, (max(ends) if ends else 0)


def staves_off(columns: list[list], staves: list, measure_ticks: int, override=None) -> list[int]:
    """Staves whose own events in this measure do not add up to it (whole
    rests excluded, as they fill any measure)."""
    off = []
    for ss in staves:
        on, length = _earliest_end_timeline(columns, measure_ticks, lambda e, ss=ss: e[1] is ss, override)
        if on and length != measure_ticks:
            off.append(ss.index)
    return off


def column_onsets(columns: list[list], staves: list, measure_ticks: int, override=None) -> tuple[dict, int]:
    """Onsets of a measure's columns, and the measure's length.

    Each staff is first timed on its own. A staff whose events add up to the
    time signature is trusted: its onsets are used for every column it has an
    event in, and the other staves' columns keep their own spacing from the
    last trusted column. One misread duration then only moves notes of its
    own staff. When no staff adds up, the whole system is timed together
    (earliest end over all staves), which also covers several voices and
    beams that cross between staves."""
    own, own_len = {}, {}
    for ss in staves:
        on, length = _earliest_end_timeline(columns, measure_ticks, lambda e, ss=ss: e[1] is ss, override)
        if on:
            own[ss.index], own_len[ss.index] = on, length
    good = [i for i in own if own_len[i] == measure_ticks]
    if not good:
        return _earliest_end_timeline(columns, measure_ticks, override=override)
    onset = {}
    for ci in range(len(columns)):
        votes = [own[i][ci] for i in good if ci in own[i]]
        if votes:
            onset[ci] = max(set(votes), key=votes.count)
    for ci, col in enumerate(columns):
        if ci in onset:
            continue
        si = col[0][1].index
        anchors = [cj for cj in own[si] if cj < ci and cj in onset]
        if anchors:
            cj = max(anchors)
            onset[ci] = onset[cj] + own[si][ci] - own[si][cj]
        else:
            onset[ci] = own[si][ci]
    length = max([measure_ticks] + [onset[ci] + _duration(e[2], measure_ticks, override)
                                    for ci, col in enumerate(columns) for e in col
                                    if e[1].index in good])
    return onset, length


def interpret(rec: Recognition, removal, env: Environment) -> Interpretation:
    """Pitch and timing for a whole page (see the changelog for the rules)."""
    beats, beat_type = env.time_signature
    measure_ticks = int(round(beats * 4 / beat_type * TICKS_PER_BEAT))
    beat = beat_ticks(beats, beat_type)
    velocity = env.default_velocity
    parts = {p.slot: p for p in env.parts}
    settings = {s.index: s for s in env.staves}
    u = rec.u
    notes, rests, measures, labels = [], [], [], {}

    systems: dict[int, list] = {}
    for ss in rec.staves:
        if ss.index in settings:
            systems.setdefault(settings[ss.index].system, []).append(ss)

    clock, measure_no = 0, 0
    for sys_index in sorted(systems):
        staves = systems[sys_index]
        # a barline crosses every staff of its system; a stroke found in only
        # one staff (an arpeggio line, a long stem) is not a barline
        bars = [b for b in _merge_close([b for ss in staves for b in ss.barlines], 1.0 * u)
                if all(any(abs(b - x) <= 0.75 * u for x in ss.barlines) for ss in staves)]
        events = sorted([(c.x, ss, c) for ss in staves for c in ss.chords] +
                        [(r.x, ss, r) for ss in staves for r in ss.rests], key=lambda e: e[0])
        # split the system into measures at its barlines
        segments: list[list] = [[] for _ in range(len(bars) + 1)]
        for e in events:
            segments[sum(1 for b in bars if b < e[0])].append(e)
        for seg in segments:
            if not seg:
                continue                      # a "measure" with nothing in it is a header stroke
            measure_acc = {ss.index: {} for ss in staves}
            # a whole rest is drawn centred in its measure but fills it from
            # the start: take it out of the column order
            measure_rests = [e for e in seg if isinstance(e[2], Rest) and e[2].kind == "whole"]
            seg = [e for e in seg if not (isinstance(e[2], Rest) and e[2].kind == "whole")]
            # columns: events lined up vertically sound together
            columns: list[list] = []
            for e in seg:
                if columns and e[0] - columns[-1][0][0] <= COLUMN_TOL_U * u:
                    columns[-1].append(e)
                else:
                    columns.append([e])
            t0 = clock
            onset, length = column_onsets(columns, staves, measure_ticks)
            override = Repair()
            bad = staves_off(columns, staves, measure_ticks)
            if length != measure_ticks or bad:
                # try reading the beam groups as whole beats; keep it only if
                # the measure then adds up, in every staff that was off
                trial = repair_beam_groups(columns, beat, u)
                if trial:
                    on2, len2 = column_onsets(columns, staves, measure_ticks, trial)
                    if len2 == measure_ticks and len(staves_off(columns, staves, measure_ticks, trial)) < len(bad) \
                            or len2 == measure_ticks and length != measure_ticks:
                        onset, length, override = on2, len2, trial
            first_note = len(notes)
            chord_onset = {}
            for ci, col in enumerate(columns):
                t = t0 + onset[ci]
                for x, ss, ev in col:
                    chord_onset[id(ev)] = t
                    setting = settings[ss.index]
                    part = parts[setting.slot]
                    if isinstance(ev, Rest):
                        d = rest_ticks(ev, measure_ticks)
                        rests.append((setting.slot, ss.index, t, d, ev.kind))
                        continue
                    d = override.ticks.get(id(ev), ev.ticks)
                    # clef in force at this x: the last change before it, else the opening clef
                    code, octave = setting.clef, ss.start_octave
                    for cx, ccode, coct in ss.clef_changes:
                        if cx < x:
                            code, octave = ccode, coct
                    clef = Clef.parse(code)
                    key = key_alterations(setting.key)
                    paths = removal.line_paths[ss.index]
                    shift = sum(sh for x0, x1, sh in getattr(ss, "ottavas", []) if x0 <= x <= x1)
                    for h in ev.heads:
                        step = int(round(y_to_step(paths, h.match.cx, h.match.cy)))
                        letter, oct_ = clef.step_to_pitch(step)
                        acc = measure_acc[ss.index]
                        if h.accidental:
                            alter = ALTER[h.accidental]
                            acc[(letter, oct_)] = alter
                        else:
                            alter = acc.get((letter, oct_), key.get(letter, 0))
                        midi = 12 * (oct_ + 1 + octave + shift) + PITCH_CLASS[letter] + alter + part.transpose
                        if not 0 <= midi <= 127:
                            continue
                        written = letter + {1: "#", -1: "b", 0: ""}[alter] + str(oct_ + octave)
                        notes.append(NoteOut(setting.slot, ss.index, t, d, midi, written,
                                             h.match.cx, h.match.cy, h, measure_no, t - t0))
                        labels[id(h)] = written
            if override.other_voice:
                # a chord of the other voice on a repaired beam lasts until that
                # voice's next chord on the beam, or the end of the group
                for gid, d in override.group_step.items():
                    ids = [i for i, g in override.group_of.items() if g == gid and i in chord_onset]
                    end = max(chord_onset[i] for i in ids) + d
                    voice = sorted((chord_onset[i], i) for i in ids if i in override.other_voice)
                    for k, (t_, i) in enumerate(voice):
                        nxt = voice[k + 1][0] if k + 1 < len(voice) else end
                        for n in notes[first_note:]:
                            if n.onset == t_ and id(_chord_of(n, columns)) == i:
                                n.ticks = nxt - t_
            for x, ss, ev in measure_rests:
                rests.append((settings[ss.index].slot, ss.index, t0, measure_ticks, ev.kind))
                length = max(length, measure_ticks)
            measures.append((staves[0].index, measure_no, length, measure_ticks))
            # a measure a little too long or short is a misread duration inside
            # it; the next measure still starts on the time signature's grid,
            # so one error does not shift the rest of the piece. A large
            # difference (pickup bar, unread change of time signature) is kept.
            if abs(length - measure_ticks) <= SNAP_FRACTION * measure_ticks:
                length = measure_ticks
            clock = t0 + length
            measure_no += 1

    # ties: the second note of a tie extends the first instead of sounding again
    by_head = {id(n.head): n for n in notes}
    absorbed = {}
    for a, b in sorted(rec.ties, key=lambda ab: ab[0].match.cx):
        na, nb = by_head.get(id(a)), by_head.get(id(b))
        while na is not None and id(na) in absorbed:
            na = absorbed[id(na)]
        if na is None or nb is None or na is nb or nb.onset <= na.onset:
            continue
        na.ticks = max(na.ticks, nb.onset + nb.ticks - na.onset)
        absorbed[id(nb)] = na
        labels[id(b)] = labels.get(id(b), "") + "~"
    notes = [n for n in notes if id(n) not in absorbed]

    sequences = {slot: Sequences.absolute() for slot in parts}
    for n in sorted(notes, key=lambda n: (n.onset, n.midi)):
        sequences[n.part].add_event(NoteEvent(midi_to_note(n.midi, velocity), duration_ticks=n.ticks,
                                              timestamp_ticks=n.onset))
    return Interpretation([sequences[s] for s in sorted(sequences)], notes, rests, measures, labels, clock)
