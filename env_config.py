# =============================================================================
#  env_config.py  --  musical environment from the user (terminal)
#  Version: 1.0.1
#
#  Changelog
#  ---------
#  1.0.1  (2026-10-23)
#    - When a part has no confident key reading, the suggested default is the
#      key the rest of the page reads confidently (parts usually share one),
#      not "no sharps or flats". Same idea for clefs, from the part's own
#      uncertain readings.
#  1.0.0  (2026-10-23)
#    - Asks only for what the page cannot say, or says uncertainly:
#        always   : tempo (BPM + beat unit), instrument per part, time
#                   signature digits (located by env_detect, not read)
#        if needed: clef / key per part when detection is missing, low
#                   confidence, or inconsistent
#        defaults : transposition for transposing instruments, default dynamic
#    - Detection results are the default answers: Enter accepts them.
#    - General MIDI program table (0-127), searchable by name.
#    - MIDI channels assigned per instrument, skipping channel 10 (drums).
#    - Saves/loads JSON, so a page is only configured once.
#    - --yes accepts every default without asking (scripting, tests).
#
#  Depends on env_detect.py >= 1.0.0
# =============================================================================

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict, dataclass, field

from env_detect import Clef, PageEnvironment, key_name

CONFIG_VERSION = 1

# --------------------------------------------------------------------------- #
#  Standards
# --------------------------------------------------------------------------- #
GENERAL_MIDI = [
    "Acoustic Grand Piano", "Bright Acoustic Piano", "Electric Grand Piano", "Honky-tonk Piano",
    "Electric Piano 1", "Electric Piano 2", "Harpsichord", "Clavinet",
    "Celesta", "Glockenspiel", "Music Box", "Vibraphone", "Marimba", "Xylophone", "Tubular Bells", "Dulcimer",
    "Drawbar Organ", "Percussive Organ", "Rock Organ", "Church Organ", "Reed Organ", "Accordion", "Harmonica",
    "Tango Accordion",
    "Acoustic Guitar (nylon)", "Acoustic Guitar (steel)", "Electric Guitar (jazz)", "Electric Guitar (clean)",
    "Electric Guitar (muted)", "Overdriven Guitar", "Distortion Guitar", "Guitar Harmonics",
    "Acoustic Bass", "Electric Bass (finger)", "Electric Bass (pick)", "Fretless Bass", "Slap Bass 1",
    "Slap Bass 2", "Synth Bass 1", "Synth Bass 2",
    "Violin", "Viola", "Cello", "Contrabass", "Tremolo Strings", "Pizzicato Strings", "Orchestral Harp", "Timpani",
    "String Ensemble 1", "String Ensemble 2", "Synth Strings 1", "Synth Strings 2", "Choir Aahs", "Voice Oohs",
    "Synth Choir", "Orchestra Hit",
    "Trumpet", "Trombone", "Tuba", "Muted Trumpet", "French Horn", "Brass Section", "Synth Brass 1", "Synth Brass 2",
    "Soprano Sax", "Alto Sax", "Tenor Sax", "Baritone Sax", "Oboe", "English Horn", "Bassoon", "Clarinet",
    "Piccolo", "Flute", "Recorder", "Pan Flute", "Blown Bottle", "Shakuhachi", "Whistle", "Ocarina",
    "Lead 1 (square)", "Lead 2 (sawtooth)", "Lead 3 (calliope)", "Lead 4 (chiff)", "Lead 5 (charang)",
    "Lead 6 (voice)", "Lead 7 (fifths)", "Lead 8 (bass + lead)",
    "Pad 1 (new age)", "Pad 2 (warm)", "Pad 3 (polysynth)", "Pad 4 (choir)", "Pad 5 (bowed)", "Pad 6 (metallic)",
    "Pad 7 (halo)", "Pad 8 (sweep)",
    "FX 1 (rain)", "FX 2 (soundtrack)", "FX 3 (crystal)", "FX 4 (atmosphere)", "FX 5 (brightness)",
    "FX 6 (goblins)", "FX 7 (echoes)", "FX 8 (sci-fi)",
    "Sitar", "Banjo", "Shamisen", "Koto", "Kalimba", "Bagpipe", "Fiddle", "Shanai",
    "Tinkle Bell", "Agogo", "Steel Drums", "Woodblock", "Taiko Drum", "Melodic Tom", "Synth Drum", "Reverse Cymbal",
    "Guitar Fret Noise", "Breath Noise", "Seashore", "Bird Tweet", "Telephone Ring", "Helicopter", "Applause",
    "Gunshot",
]
GM_FAMILIES = ["Piano", "Chromatic Percussion", "Organ", "Guitar", "Bass", "Strings", "Ensemble", "Brass",
               "Reed", "Pipe", "Synth Lead", "Synth Pad", "Synth Effects", "Ethnic", "Percussive", "Sound Effects"]
DRUM_CHANNEL = 9   # channel 10 in 1-based numbering, reserved for drums by General MIDI

# Written -> sounding pitch, in semitones, for common transposing instruments.
TRANSPOSITION = {
    "Clarinet": -2, "Trumpet": -2, "Muted Trumpet": -2, "Soprano Sax": -2, "French Horn": -7,
    "English Horn": -7, "Alto Sax": -9, "Tenor Sax": -14, "Baritone Sax": -21, "Piccolo": 12,
    "Contrabass": -12, "Acoustic Bass": -12, "Electric Bass (finger)": -12, "Electric Bass (pick)": -12,
    "Fretless Bass": -12, "Acoustic Guitar (nylon)": -12, "Acoustic Guitar (steel)": -12,
    "Electric Guitar (jazz)": -12, "Electric Guitar (clean)": -12, "Celesta": 12, "Glockenspiel": 24,
    "Xylophone": 12,
}

# Velocity for each dynamic (MuseScore's defaults).
DYNAMICS = {"ppp": 16, "pp": 33, "p": 49, "mp": 64, "mf": 80, "f": 96, "ff": 112, "fff": 126}

# Beat units, in quarter notes. MIDI tempo is always quarter notes per minute.
BEAT_UNITS = {"eighth": 0.5, "quarter": 1.0, "dotted quarter": 1.5, "half": 2.0, "dotted half": 3.0}


# --------------------------------------------------------------------------- #
#  Configuration
# --------------------------------------------------------------------------- #
@dataclass
class Part:
    """One part: the staves in one slot of every system (one MIDI track).

    Attributes:
        name (str): Display name, e.g. "Piano RH", "Violin I".
        slot (int): Staff position within each system (0 = top).
        instrument (str): General MIDI instrument name.
        program (int): General MIDI program number, 0-127.
        channel (int): MIDI channel, 0-15 (never 9, the drum channel).
        transpose (int): Semitones from written to sounding pitch.
        clef (str): Opening clef, MusicXML sign+line (e.g. "G2").
        key (int): Opening key signature, in fifths (+sharps / -flats).
    """
    name: str
    slot: int
    instrument: str
    program: int
    channel: int
    transpose: int
    clef: str
    key: int


@dataclass
class StaffSetting:
    """Clef and key at the start of one staff, and where each came from.

    Attributes:
        index (int): Staff index on the page.
        system (int): System index.
        slot (int): Slot within the system (= part).
        clef (str): MusicXML sign+line.
        key (int): Fifths.
        source (dict): `{"clef": ..., "key": ...}` -- "detected (high)",
            "part default", "user", ...
    """
    index: int
    system: int
    slot: int
    clef: str
    key: int
    source: dict = field(default_factory=dict)


@dataclass
class Environment:
    """Everything the later stages need to turn symbols into MIDI.

    Attributes:
        bpm (float): Tempo in `beat_unit`s per minute, as the score marks it.
        beat_unit (str): One of BEAT_UNITS.
        time_signature (tuple[int, int]): (beats, beat type).
        default_dynamic (str): Dynamic used before any marking, e.g. "mf".
        parts (list[Part]): One per slot.
        staves (list[StaffSetting]): Clef and key at each staff's start.
        dynamics (dict[str, int]): Dynamic -> MIDI velocity.
        page (str): Source image, if any.
        version (int): Config format version.
    """
    bpm: float
    beat_unit: str
    time_signature: tuple
    default_dynamic: str
    parts: list
    staves: list
    dynamics: dict = field(default_factory=lambda: dict(DYNAMICS))
    page: str = ""
    version: int = CONFIG_VERSION

    @property
    def quarter_bpm(self) -> float:
        """Tempo in quarter notes per minute -- what MIDIWriter(bpm=...) wants."""
        return self.bpm * BEAT_UNITS[self.beat_unit]

    @property
    def default_velocity(self) -> int:
        return self.dynamics[self.default_dynamic]

    def to_json(self, path: str):
        data = asdict(self)
        data["quarter_bpm"] = self.quarter_bpm   # convenience, ignored on load
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def from_json(cls, path: str) -> "Environment":
        with open(path) as f:
            data = json.load(f)
        data.pop("quarter_bpm", None)
        data["time_signature"] = tuple(data["time_signature"])
        data["parts"] = [Part(**p) for p in data["parts"]]
        data["staves"] = [StaffSetting(**s) for s in data["staves"]]
        return cls(**data)


# --------------------------------------------------------------------------- #
#  Parsing answers
# --------------------------------------------------------------------------- #
def parse_bpm(text: str) -> float:
    value = float(text)
    if not 10 <= value <= 400:
        raise ValueError("BPM should be between 10 and 400.")
    return value


def parse_beat_unit(text: str) -> str:
    text = text.strip().lower().replace("-", " ")
    aliases = {"q": "quarter", "crotchet": "quarter", "e": "eighth", "quaver": "eighth",
               "h": "half", "minim": "half", "dq": "dotted quarter", "dh": "dotted half"}
    text = aliases.get(text, text)
    if text not in BEAT_UNITS:
        raise ValueError(f"Choose one of: {', '.join(BEAT_UNITS)}.")
    return text


def parse_time(text: str) -> tuple[int, int]:
    text = text.strip().lower()
    if text in ("c", "common"):
        return (4, 4)
    if text in ("c|", "cut", "alla breve"):
        return (2, 2)
    beats, sep, kind = text.partition("/")
    if not sep:
        raise ValueError("Write it as beats/beat-type, e.g. 3/4, or C for common time.")
    beats, kind = int(beats), int(kind)
    if beats <= 0 or kind not in (1, 2, 4, 8, 16, 32):
        raise ValueError("Beat type must be 1, 2, 4, 8, 16 or 32.")
    return (beats, kind)


def parse_key(text: str) -> int:
    """`2` / `+2` / `2#` / `2 sharps` / `-3` / `3b` / `3 flats` / `0` / `none`."""
    t = text.strip().lower().replace(" ", "")
    if t in ("0", "none", "c", "nosharpsorflats"):
        return 0
    sign = 1
    for suffix, s in (("sharps", 1), ("sharp", 1), ("#", 1), ("flats", -1), ("flat", -1), ("b", -1)):
        if t.endswith(suffix):
            t, sign = t[:-len(suffix)], s
            break
    value = sign * int(t)
    if not -7 <= value <= 7:
        raise ValueError("Between 7 flats (-7) and 7 sharps (+7).")
    return value


def parse_clef(text: str) -> str:
    return Clef.parse(text).code


def parse_dynamic(text: str) -> str:
    text = text.strip().lower()
    if text not in DYNAMICS:
        raise ValueError(f"One of: {', '.join(DYNAMICS)}.")
    return text


def parse_int(lo: int, hi: int):
    def parse(text: str) -> int:
        value = int(text)
        if not lo <= value <= hi:
            raise ValueError(f"Between {lo} and {hi}.")
        return value
    return parse


def find_instruments(query: str) -> list[int]:
    query = query.strip().lower()
    return [i for i, name in enumerate(GENERAL_MIDI) if query in name.lower()]


def instrument_table() -> str:
    lines = []
    for family_index, family in enumerate(GM_FAMILIES):
        names = [f"{i:3d} {GENERAL_MIDI[i]}" for i in range(family_index * 8, family_index * 8 + 8)]
        lines.append(f"  {family}: " + " | ".join(names))
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
#  Prompting
# --------------------------------------------------------------------------- #
class Prompter:
    """Terminal questions with defaults.
    - Enter accepts the default shown in brackets.
    - `?` shows help for that question, when there is any.
    - With `auto=True`, every question takes its default without waiting.
    """

    def __init__(self, auto: bool = False, input_fn=input, out=print):
        self.auto = auto
        self.input = input_fn
        self.out = out

    def ask(self, question: str, default, parse, show=None, help_text: str | None = None):
        shown = show(default) if (show and default is not None) else default
        prompt = f"{question} [{shown}]: " if default is not None else f"{question}: "
        if self.auto:
            self.out(f"{prompt}{shown}  (default)")
            return default
        while True:
            try:
                answer = self.input(prompt).strip()
            except EOFError:
                self.out("")
                if default is None:
                    raise SystemExit("No answer and no default; stopping.")
                return default
            if answer == "" and default is not None:
                return default
            if answer == "?" and help_text:
                self.out(help_text)
                continue
            try:
                return parse(answer)
            except (ValueError, KeyError) as error:
                message = str(error)
                if not message or message.startswith(("invalid literal", "could not convert")):
                    message = "Not understood, try again" + (" ('?' for help)." if help_text else ".")
                self.out(f"  {message}")

    def confirm(self, question: str, default: bool = True) -> bool:
        def parse(text: str) -> bool:
            t = text.lower()
            if t in ("y", "yes"):
                return True
            if t in ("n", "no"):
                return False
            raise ValueError("y or n.")
        return self.ask(question, default, parse, show=lambda d: "Y/n" if d else "y/N")

    def instrument(self, question: str, default: int) -> int:
        def parse(text: str) -> int:
            if text.isdigit():
                return parse_int(0, 127)(text)
            matches = find_instruments(text)
            if len(matches) == 1:
                return matches[0]
            if not matches:
                raise ValueError("No General MIDI instrument matches that. '?' lists them all.")
            listing = ", ".join(f"{i} {GENERAL_MIDI[i]}" for i in matches[:12])
            raise ValueError(f"Several match -- type the number: {listing}")
        return self.ask(question, default, parse, show=lambda p: f"{p} {GENERAL_MIDI[p]}",
                        help_text="Type a number, or part of a name (e.g. 'cello').\n" + instrument_table())


# --------------------------------------------------------------------------- #
#  Building the environment
# --------------------------------------------------------------------------- #
def _confidence_tag(found) -> str:
    return f"detected ({found.confidence})"


def _majority(values: list):
    return max(set(values), key=values.count) if values else None


def _suggest_part_names(slots: int, clefs: list[str | None]) -> list[str]:
    if slots == 2 and clefs == ["G2", "F4"]:
        return ["Piano RH", "Piano LH"]
    if slots == 1:
        return ["Part 1"]
    return [f"Part {i + 1}" for i in range(slots)]


def _suggest_program(name: str, clef: str | None) -> int:
    lowered = name.lower()
    for i, instrument in enumerate(GENERAL_MIDI):
        if instrument.lower() in lowered or lowered in instrument.lower():
            return i
    if "piano" in lowered:
        return 0
    return {"G2": 0, "F4": 42, "C3": 41, "C4": 42}.get(clef or "", 0)


def _default_beat_unit(time_signature: tuple[int, int]) -> str:
    beats, kind = time_signature
    if kind == 8 and beats in (6, 9, 12):
        return "dotted quarter"
    return {2: "half", 8: "eighth"}.get(kind, "quarter")


def build_environment(detected: PageEnvironment | None, prompt: Prompter,
                      staff_count: int | None = None, page: str = "") -> Environment:
    """Asks the user for the environment, using detection as defaults.

    Args:
        detected: Output of `env_detect.detect_environment`, or None to set
            everything up by hand.
        prompt: The `Prompter` to ask through.
        staff_count: Without detection: staves on the page.
        page: Source image path, stored in the result.
    Returns:
        A complete `Environment`.
    """
    say = prompt.out

    # ---- layout ----------------------------------------------------------
    if detected is not None:
        slots = detected.slots
        staff_rows = [(s.index, s.system, s.slot) for s in detected.staves]
        if not detected.consistent:
            say(f"! Systems have different numbers of staves "
                f"({[len(s) for s in detected.systems]}); parts follow staff position.")
    else:
        staff_count = staff_count or prompt.ask("How many staves on the page", 1, parse_int(1, 64))
        slots = prompt.ask("Staves per system (e.g. 2 for piano)", 1, parse_int(1, staff_count))
        staff_rows = [(i, i // slots, i % slots) for i in range(staff_count)]

    by_slot = {slot: [] for slot in range(slots)}
    for index, _, slot in staff_rows:
        by_slot[slot].append(index)
    staff_by_index = {s.index: s for s in detected.staves} if detected else {}

    # ---- clefs and keys per part -----------------------------------------
    part_clefs, part_keys = [], []
    for slot in range(slots):
        members = [staff_by_index[i] for i in by_slot[slot]] if detected else []
        sure_clefs = [s.clef.value.code for s in members if s.clef.certain]
        sure_keys = [s.key.value for s in members if s.key.certain]
        part_clefs.append(_majority(sure_clefs))
        part_keys.append(_majority(sure_keys))

    # fallbacks when a part has nothing certain: the rest of the page, then the
    # part's own uncertain readings (parts usually share a key signature)
    if detected is not None:
        page_keys = [s.key.value for s in detected.staves if s.key.certain]
        for slot in range(slots):
            if part_keys[slot] is None:
                own = [staff_by_index[i].key.value for i in by_slot[slot]
                       if staff_by_index[i].key.value is not None]
                part_keys[slot] = _majority(page_keys) if page_keys else _majority(own)
            if part_clefs[slot] is None:
                own = [staff_by_index[i].clef.value.code for i in by_slot[slot]
                       if staff_by_index[i].clef.value is not None]
                part_clefs[slot] = _majority(own)
        fallback_from_page = {slot for slot in range(slots)
                              if not any(staff_by_index[i].key.certain for i in by_slot[slot])}
    else:
        fallback_from_page = set()

    names = _suggest_part_names(slots, part_clefs)

    # ---- time signature (never read by detection yet) --------------------
    located = [s.index for s in detected.staves if s.time_present] if detected else []
    say("")
    if located:
        say(f"A time signature was found at the start of staff {located[0]}, but its digits "
            f"can't be read yet (that needs the glyph classifier).")
    else:
        say("No time signature found at the start of any staff.")
    time_signature = prompt.ask("Time signature", (4, 4), parse_time,
                                show=lambda t: f"{t[0]}/{t[1]}",
                                help_text="e.g. 3/4, 6/8, C (= 4/4), cut (= 2/2)")

    # ---- tempo -----------------------------------------------------------
    say("")
    beat_unit = prompt.ask("Beat unit for the tempo", _default_beat_unit(time_signature), parse_beat_unit,
                           help_text=f"One of: {', '.join(BEAT_UNITS)}. In 6/8 the beat is "
                                     "usually the dotted quarter.")
    bpm = prompt.ask(f"Tempo ({beat_unit} beats per minute)", 100.0, parse_bpm, show=lambda b: f"{b:g}")
    default_dynamic = prompt.ask("Starting dynamic", "mf", parse_dynamic,
                                 help_text=", ".join(f"{k}={v}" for k, v in DYNAMICS.items()))

    # ---- parts -----------------------------------------------------------
    parts: list[Part] = []
    channels: dict[str, int] = {}
    for slot in range(slots):
        say("")
        where = f"staff {slot + 1} of {slots} in each system" if slots > 1 else "every staff"
        say(f"Part {slot + 1} ({where}):")
        members = [staff_by_index[i] for i in by_slot[slot]] if detected else []

        name = prompt.ask("  Name", names[slot], str)
        program = prompt.instrument("  Instrument", _suggest_program(name, part_clefs[slot]))
        instrument = GENERAL_MIDI[program]
        transpose = TRANSPOSITION.get(instrument, 0)
        if transpose:
            transpose = prompt.ask(f"  {instrument} sounds this many semitones from written pitch",
                                   transpose, parse_int(-36, 36), show=lambda t: f"{t:+d}")

        # clef: ask when nothing certain was detected, or staves disagree
        clef = part_clefs[slot]
        uncertain = [s.index for s in members if not s.clef.certain]
        if clef is None or uncertain or not members:
            detail = (f"detected {Clef.parse(clef).name}" if clef else "not detected") + \
                     (f"; unsure on staves {uncertain}" if uncertain and clef else "")
            say(f"  Clef: {detail}.")
            clef = prompt.ask("  Opening clef", clef or "G2", parse_clef,
                              show=lambda c: f"{Clef.parse(c).name} ({c})",
                              help_text="treble, bass, alto, tenor, soprano, or sign+line like G2/F4/C3")
        else:
            say(f"  Clef: {Clef.parse(clef).name} ({clef}) -- detected on all {len(members)} staves.")

        key = part_keys[slot]
        if slot in fallback_from_page and key is not None:
            say(f"  (no confident key on this part's staves; suggesting the rest of the page's)")
        unsure_keys = [s for s in members if not s.key.certain]
        reads = sorted({s.key.value for s in members if s.key.value is not None})
        if key is None or unsure_keys or len(reads) > 1:
            if reads:
                say(f"  Key: read as {', '.join(key_name(v) for v in reads)}"
                    + (f"; unsure on staves {[s.index for s in unsure_keys]}" if unsure_keys else "") + ".")
            else:
                say("  Key: not detected.")
            key = prompt.ask("  Key signature", key if key is not None else 0, parse_key, show=key_name,
                             help_text="Number of sharps (2 or 2#) or flats (-3 or 3b); 0 for none.")
        else:
            say(f"  Key: {key_name(key)} -- detected on all {len(members)} staves.")

        if instrument not in channels:
            free = [c for c in range(16) if c != DRUM_CHANNEL and c not in channels.values()]
            channels[instrument] = free[0] if free else 0
        parts.append(Part(name, slot, instrument, program, channels[instrument], transpose, clef, key))

    # ---- per-staff settings ----------------------------------------------
    staves: list[StaffSetting] = []
    for index, system, slot in staff_rows:
        part = parts[slot]
        clef, key = part.clef, part.key
        source = {"clef": "part", "key": "part"}
        found = staff_by_index.get(index)
        if found is not None:
            # a confident reading that differs from the part is kept: clefs change mid-piece
            if found.clef.certain and found.clef.value.code != clef:
                clef, source["clef"] = found.clef.value.code, _confidence_tag(found.clef)
            elif found.clef.certain:
                source["clef"] = _confidence_tag(found.clef)
            if found.key.certain and found.key.value != key:
                key, source["key"] = found.key.value, _confidence_tag(found.key)
            elif found.key.certain:
                source["key"] = _confidence_tag(found.key)
        staves.append(StaffSetting(index, system, slot, clef, key, source))

    return Environment(bpm, beat_unit, time_signature, default_dynamic, parts, staves, page=page)


def summarise(env: Environment, out=print):
    out(f"tempo        : {env.bpm:g} per {env.beat_unit}  (= {env.quarter_bpm:g} quarter notes per minute)")
    out(f"time         : {env.time_signature[0]}/{env.time_signature[1]}")
    out(f"dynamic      : {env.default_dynamic} (velocity {env.default_velocity})")
    for p in env.parts:
        t = f", transposes {p.transpose:+d}" if p.transpose else ""
        out(f"part {p.slot}       : {p.name} -- {p.instrument} (program {p.program}, channel {p.channel + 1}{t}), "
            f"{Clef.parse(p.clef).name}, {key_name(p.key)}")
    changed = [s for s in env.staves if s.clef != env.parts[s.slot].clef or s.key != env.parts[s.slot].key]
    for s in changed:
        out(f"staff {s.index:<6} : differs from its part -- {Clef.parse(s.clef).name}, {key_name(s.key)} "
            f"({s.source})")


# --------------------------------------------------------------------------- #
#  Command line
# --------------------------------------------------------------------------- #
def main(argv=None):
    parser = argparse.ArgumentParser(description="Set up the musical environment for a page.")
    parser.add_argument("image", nargs="?", help="page image; detection supplies the defaults")
    parser.add_argument("--config", help="load this environment JSON instead of asking")
    parser.add_argument("--out", help="where to save (default: <image>.env.json or environment.json)")
    parser.add_argument("-y", "--yes", action="store_true", help="accept every default without asking")
    args = parser.parse_args(argv)

    prompt = Prompter(auto=args.yes)
    out_path = args.out or (os.path.splitext(args.image)[0] + ".env.json" if args.image else "environment.json")

    if args.config and os.path.exists(args.config):
        env = Environment.from_json(args.config)
        print(f"Loaded {args.config}\n")
        summarise(env)
        if prompt.confirm("\nUse this as it is", True):
            return env

    detected = None
    if args.image:
        import cv2
        from staff_detect import detect_staves
        from staff_remove import remove_from_detection
        from env_detect import detect_environment, summarise as summarise_detected

        print(f"Reading {args.image} ...\n")
        result = detect_staves(cv2.imread(args.image, 0))
        detected = detect_environment(result, remove_from_detection(result))
        summarise_detected(detected)

    print("\nPress Enter to accept the value in [brackets]; '?' for help where offered.")
    env = build_environment(detected, prompt, page=args.image or "")

    print("\n" + "-" * 60)
    summarise(env)
    if prompt.confirm("\nSave", True):
        env.to_json(out_path)
        print(f"Saved {out_path}")
    return env


if __name__ == "__main__":
    main()
