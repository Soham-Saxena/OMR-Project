# =============================================================================
#  events.py  --  note events
#  Version: 1.1.0
#
#  Changelog
#  ---------
#  1.1.0  (2026-10-23)
#    - Integer tick timeline stored ALONGSIDE the existing ms values. Nothing
#      that existed is removed: `duration` and `timestamp` (ms) are still set
#      exactly as before. New fields:
#        NoteEvent.duration_ticks, NoteEvent.timestamp_ticks,
#        MidoEvent.timestamp_ticks
#    - Ticks are the authoritative timeline: sequences match nodes on ticks
#      and the writer reads ticks. ms are kept for compatibility and display.
#    - How ticks are derived:
#        beats + bpm  -> beats * TICKS_PER_BEAT        (exact, tempo-free)
#        ms only      -> converted at REFERENCE_BPM     (1 ms == 1 tick at the
#                                                       defaults 125 BPM, 480)
#        duration_ticks= / timestamp_ticks=  -> used as given  (new keywords)
#    - Module constants TICKS_PER_BEAT, REFERENCE_BPM and helpers
#      beats_to_ticks, ticks_to_beats, ms_to_ticks, ticks_to_ms.
#    - NoteEvent.beats property (duration in beats, from ticks).
#    - MidoEvent.sort_key(): orders by tick, note_off before note_on at the
#      same tick, so a repeated pitch is never cut off by its own release.
#    - __copy__ preserves ticks exactly (previously re-derived from ms).
#
#  1.0.0
#    - Original ms-based version.
# =============================================================================

from __future__ import annotations
from note import Note
from typing import overload, Literal
import copy
from dataclasses import dataclass

TICKS_PER_BEAT = 480
"""Timeline resolution in ticks per beat (one beat = one quarter note).
480 divides by 2, 3, 4, 5, 6, 8, 10, 12, 15 and 16, so every common note value
-- triplets and quintuplets included -- is a whole number of ticks."""

REFERENCE_BPM = 125
"""Tempo at which durations given in ms are converted into ticks.
At 125 BPM and 480 ticks per beat, 1 ms is exactly 1 tick, so ms-based code
produces the same MIDI it always did with the writer's default tempo. If you
write ms durations and use a different writer BPM, set this to that BPM before
creating events."""


def beats_to_ticks(beats: float) -> int:
    """Converts beats (quarter notes) into ticks."""
    return int(round(beats * TICKS_PER_BEAT))


def ticks_to_beats(ticks: int) -> float:
    """Converts ticks into beats (quarter notes)."""
    return ticks / TICKS_PER_BEAT


def ms_to_ticks(ms: float, bpm: float | None = None) -> int:
    """Converts ms into ticks at `bpm` (defaults to `REFERENCE_BPM`)."""
    bpm = REFERENCE_BPM if bpm is None else bpm
    return int(round(ms * bpm * TICKS_PER_BEAT / 60000))


def ticks_to_ms(ticks: int, bpm: float | None = None) -> int:
    """Converts ticks into ms at `bpm` (defaults to `REFERENCE_BPM`)."""
    bpm = REFERENCE_BPM if bpm is None else bpm
    return int(round(ticks * 60000 / (bpm * TICKS_PER_BEAT)))


class NoteEvent:
    """Describes a note event, containing the Note, its timestamp and duration.

    Attributes:
        note (Note): The note contained by the event.
        timestamp (int): The timestamp of when the note plays, in ms.
        duration (int): The duration of the note, in ms.
        timestamp_ticks (int): The timestamp in ticks (`-1` until resolved).
            This is the authoritative timeline.
        duration_ticks (int): The duration in ticks. Authoritative.
        DEFAULT_DURATION (int): The default duration to consider if no duration provided.
    """
    DEFAULT_DURATION = 400
    DEFAULT_BEATS = 4
    @overload
    def __init__(self, note: Note, duration: int|None, *, timestamp: int = -1):
        """Creates a Note Event based on duration, in `ms`.
        Args:
            note: The note contained by the event.
            duration: Duration of the note. Defaults to `DEFAULT_DURATION`.
            timestamp: Timestamp of the note. Defaults to `-1`. Must be specified in the argument with the name.
                - `-1` for Relative Events, will get resolved when the sequence is compiled.
        """
        ...
    @overload
    def __init__(self, note: Note, beats: float|None = None, bpm: int|None = None, *, timestamp: int = -1):
        """Creates a Note Event based on beats and bpm.
        Args:
            note: The note contained by the event.
            beats: The number of beats taken up by the note.
            bpm: The BPM the event is played at.
            timestamp: Timestamp of the note. Defaults to `-1`. Must be specified in the argument with the name.
                - `-1` for Relative Events, will get resolved when the sequence is compiled.
        """
        ...
    @overload
    def __init__(self, note: Note, *, duration_ticks: int, timestamp_ticks: int = -1):
        """Creates a Note Event directly in ticks.
        Args:
            note: The note contained by the event.
            duration_ticks: Duration in ticks (`TICKS_PER_BEAT` per quarter note).
            timestamp_ticks: Timestamp in ticks. Defaults to `-1` (relative).
        """
        ...
    def __init__(self, note: Note, beatsOrDuration: int|float|None = None, bpm: int|None = None, *,
                 timestamp: int = -1, duration_ticks: int|None = None, timestamp_ticks: int = -1):
        duration = 0
        if duration_ticks is not None:
            if beatsOrDuration is not None:
                raise ValueError("Provide either beats/duration or duration_ticks, not both.")
            if bpm is not None and bpm <= 0:
                raise ValueError("BPM must be greater than 0.")
            ticks = int(duration_ticks)
            if ticks <= 0:
                raise ValueError("duration_ticks must be greater than 0")
            duration = max(1, ticks_to_ms(ticks, bpm))
        elif bpm is not None:
            if bpm <= 0:
                raise ValueError("BPM must be greater than 0.")
            beats = float(beatsOrDuration if beatsOrDuration is not None else NoteEvent.DEFAULT_BEATS)
            duration = int((beats/bpm) * 60000)
            ticks = beats_to_ticks(beats)
        else:
            duration = int(beatsOrDuration if beatsOrDuration is not None else NoteEvent.DEFAULT_DURATION)
            ticks = ms_to_ticks(duration)
        if duration <= 0:
            raise ValueError("Beats or Duration must be greater than 0")
        if ticks <= 0:
            raise ValueError(f"Duration is shorter than one tick at {TICKS_PER_BEAT} ticks per beat.")
        self.note = copy.copy(note)
        self.timestamp = timestamp
        self.duration = duration
        self.duration_ticks = ticks

        if timestamp_ticks >= 0:
            self.timestamp_ticks = int(timestamp_ticks)
            if timestamp < 0:
                self.timestamp = ticks_to_ms(self.timestamp_ticks, bpm)
        elif timestamp >= 0:
            self.timestamp_ticks = ms_to_ticks(timestamp, bpm)
        else:
            self.timestamp_ticks = -1
    @property
    def beats(self) -> float:
        """Duration in beats (quarter notes), from the tick duration."""
        return ticks_to_beats(self.duration_ticks)
    def __copy__(self):
        new = type(self)(
            self.note,
            duration_ticks = self.duration_ticks,
            timestamp = self.timestamp,
            timestamp_ticks = self.timestamp_ticks
        )
        new.duration = self.duration
        return new
    @overload
    @classmethod
    def from_info(cls, name: str, velocity: int = 64, octave: int|None = None, duration: int|None = None, *, timestamp: int = -1) -> NoteEvent:
        """Creates a `NoteEvent` from provided information.
        Args:
            name: The name of the note.
            velocity: The loudness of the note.
            octave: The octave of the note.
            timestamp: Timestamp of the note. Defaults to `-1`. Must be specified in the argument with the name.
                - `-1` for Relative Events, will get resolved when the sequence is compiled.
            duration: Duration of the note.
        """
        ...
    @overload
    @classmethod
    def from_info(cls, name: str, velocity: int = 64, octave: int|None = None, beats: float|None = None, bpm: int|None = None, *, timestamp: int = -1) -> NoteEvent:
        """Creates a `NoteEvent` from provided information.
        Args:
            name: The name of the note.
            velocity: The loudness of the note.
            octave: The octave of the note.
            timestamp: Timestamp of the note. Defaults to `-1`. Must be specified in the argument with the name.
                - `-1` for Relative Events, will get resolved when the sequence is compiled.
            beats: The number of beats taken up by the note.
            bpm: The BPM the event is played at.
        """
        ...
    @overload
    @classmethod
    def from_info(cls, name: str, velocity: int = 64, octave: int|None = None, *, duration_ticks: int, timestamp_ticks: int = -1) -> NoteEvent:
        """Creates a `NoteEvent` from provided information, timed in ticks.
        Args:
            name: The name of the note.
            velocity: The loudness of the note.
            octave: The octave of the note.
            duration_ticks: Duration in ticks.
            timestamp_ticks: Timestamp in ticks. Defaults to `-1` (relative).
        """
        ...
    @classmethod
    def from_info(cls, name: str, velocity: int = 64, octave: int|None = None, beatsOrDuration: float|int|None = None, bpm: int|None = None, *,
                  timestamp: int = -1, duration_ticks: int|None = None, timestamp_ticks: int = -1) -> NoteEvent:
        return cls(
            Note(name, velocity, octave),
            beatsOrDuration, bpm, timestamp = timestamp,
            duration_ticks = duration_ticks, timestamp_ticks = timestamp_ticks
        )
    @classmethod
    def default_duration(cls, defDuration: int):
        """The default duration of the note event to consider if no duration provided.
        Args:
            defDuration: The default duration (must be between greater than 0).
        """
        if defDuration <= 0:
            raise ValueError("Provided duration cannot be lesser than or equal to 0.")
        NoteEvent.DEFAULT_DURATION = defDuration
    @classmethod
    def default_beats(cls, defBeats: float):
        """The default beat of the note event to consider if no beat provided.
        Args:
            defBeats: The default number of beats (must be greater than 0).
        """
        if defBeats <= 0:
            raise ValueError("Provided beats cannot be lesser than or equal to 0.")
        NoteEvent.DEFAULT_BEATS = defBeats

@overload
def note(note: Literal["C", "C#", "Db", "D", "D#", "Eb", "E", "F", "F#", "Gb", "G", "G#", "Ab", "A", "A#", "Bb", "B"],
         duration: float|None = None,
         octave: int|None = None,
         velocity: int|None = None,
         *, timestamp: int = -1) -> NoteEvent:
    """Creates a `NoteEvent` from provided information.
    Args:
        note: The name of the note.
        velocity: The loudness of the note.
        octave: The octave of the note.
        timestamp: Timestamp of the note. Defaults to `-1`. Must be specified in the argument with the name.
            - `-1` for Relative Events, will get resolved when the sequence is compiled.
        duration: Duration of the note.
    """
    ...
@overload
def note(note: Literal["C", "C#", "Db", "D", "D#", "Eb", "E", "F", "F#", "Gb", "G", "G#", "Ab", "A", "A#", "Bb", "B"],
         beats: float|None = None,
         bpm: int|None = None,
         octave: int|None = None,
         velocity: int|None = None,
         *, timestamp: int = -1) -> NoteEvent:
    """Creates a `NoteEvent` from provided information.
    Args:
        note: The name of the note.
        velocity: The loudness of the note.
        octave: The octave of the note.
        timestamp: Timestamp of the note. Defaults to `-1`. Must be specified in the argument with the name.
            - `-1` for Relative Events, will get resolved when the sequence is compiled.
        beats: The number of beats taken up by the note.
        bpm: The BPM the event is played at.
    """
    ...
@overload
def note(note: Literal["C", "C#", "Db", "D", "D#", "Eb", "E", "F", "F#", "Gb", "G", "G#", "Ab", "A", "A#", "Bb", "B"],
         *,
         octave: int|None = None,
         velocity: int|None = None,
         duration_ticks: int,
         timestamp_ticks: int = -1) -> NoteEvent:
    """Creates a `NoteEvent` timed in ticks.
    Args:
        note: The name of the note.
        octave: The octave of the note.
        velocity: The loudness of the note.
        duration_ticks: Duration in ticks (`TICKS_PER_BEAT` per quarter note).
        timestamp_ticks: Timestamp in ticks. Defaults to `-1` (relative).
    """
    ...

def note(note: Literal["C", "C#", "Db", "D", "D#", "Eb", "E", "F", "F#", "Gb", "G", "G#", "Ab", "A", "A#", "Bb", "B"],
         beats: float|None = None,
         bpm: int|None = None,
         octave: int|None = None,
         velocity: int = 64,
         *, timestamp: int = -1, duration_ticks: int|None = None, timestamp_ticks: int = -1) -> NoteEvent:
    return NoteEvent.from_info(note, velocity, octave, beats, bpm, timestamp=timestamp,
                               duration_ticks=duration_ticks, timestamp_ticks=timestamp_ticks)

@dataclass
class MidoEvent:
    """Data class used to effectively convert `NoteEvent` into something more Mido focused.
    Attributes:
        timestamp: The absolute timestamp of the event, in ms
        event_type: Describes the event type.
            - `note_on`: Beginning of the note
            - `note_off`: Ending of the note
        note: The note being referenced.
        timestamp_ticks: The absolute timestamp of the event, in ticks. Authoritative.
    """
    note: Note
    event_type: Literal["note_on", "note_off"]
    timestamp: int
    timestamp_ticks: int = -1

    def __post_init__(self):
        self.note = copy.copy(self.note)

    def sort_key(self) -> tuple[int, int]:
        """Orders by tick, with `note_off` before `note_on` at the same tick."""
        return (self.timestamp_ticks, 0 if self.event_type == "note_off" else 1)

    @classmethod
    def from_note_event(cls, event: NoteEvent) -> tuple[MidoEvent, MidoEvent]:
        """Returns two `MidoEvent`s (beginning event, ending event) From provided `NoteEvent`.
        Args:
            event: The note event to be converted. Expects event to have a valid timestamp.
        """
        if event.timestamp < 0 or event.timestamp_ticks < 0:
            raise ValueError("Please provide note event with a valid timestamp.")
        return (cls(event.note, "note_on", event.timestamp, event.timestamp_ticks),
                cls(event.note, "note_off", event.timestamp + event.duration,
                    event.timestamp_ticks + event.duration_ticks))