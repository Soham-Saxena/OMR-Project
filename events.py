from __future__ import annotations
from note import Note
from typing import overload, Literal
import copy
from dataclasses import dataclass

class NoteEvent:
    """Describes a note event, containing the Note, its timestamp and duration.

    Attributes:
        note (Note): The note contained by the event.
        timestamp (int): The timestamp of when the note plays.
        duration (int): The duration of the note.
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
    def __init__(self, note: Note, beatsOrDuration: int|float|None = None, bpm: int|None = None,*, timestamp: int = -1):
        duration = 0
        if bpm is not None:
            if bpm <= 0:
                raise ValueError("BPM must be greater than 0.")
            beats = float(beatsOrDuration if beatsOrDuration is not None else NoteEvent.DEFAULT_BEATS)
            duration = int((beats/bpm) * 60000)
        else:
            duration = int(beatsOrDuration if beatsOrDuration is not None else NoteEvent.DEFAULT_DURATION)
        if duration <= 0:
            raise ValueError("Beats or Duration must be greater than 0")
        self.note = copy.copy(note)
        self.timestamp = timestamp
        self.duration = duration
    def __copy__(self):
        new = type(self)(
            self.note,
            self.duration,
            timestamp = self.timestamp
        )
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
    @classmethod
    def from_info(cls, name: str, velocity: int = 64, octave: int|None = None, beatsOrDuration: float|int|None = None, bpm: int|None = None, *, timestamp: int = -1) -> NoteEvent:
        return cls(
            Note(name, velocity, octave),
            beatsOrDuration, bpm, timestamp = timestamp
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

@dataclass
class MidoEvent:
    """Data class used to effectively convert `NoteEvent` into something more Mido focused.
    Attributes:
        timestamp: The absolute timestamp of the event
        event_type: Describes the event type.
            - `note_on`: Beginning of the note
            - `note_off`: Ending of the note
        note: The note being referenced.
    """
    note: Note
    event_type: Literal["note_on", "note_off"]
    timestamp: int

    def __post_init__(self):
        self.note = copy.copy(self.note)

    @classmethod
    def from_note_event(cls, event: NoteEvent) -> tuple[MidoEvent, MidoEvent]:
        """Returns two `MidoEvent`s (beginning event, ending event) From provided `NoteEvent`.
        Args:
            event: The note event to be converted. Expects event to have a valid timestamp.
        """
        if event.timestamp < 0:
            raise ValueError("Please provide note event with a valid timestamp.")
        return (cls(event.note, "note_on", event.timestamp), cls(event.note, "note_off", event.timestamp + event.duration))