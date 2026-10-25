# =============================================================================
#  writer.py  --  MIDI file writer
#  Version: 1.2.0
#
#  Changelog
#  ---------
#  1.2.0  (2026-10-25)
#    - addSequence() takes optional program, channel and name, so each part
#      can be its own General MIDI instrument (env_config.Part). compile()
#      writes track_name and program_change at tick 0 and sends the part's
#      notes on its channel.
#    - Optional time_signature attribute, written as a meta message next to
#      the tempo.
#    - Defaults reproduce 1.1.0 output byte for byte (channel 0, no program
#      change, no time signature).
#
#  1.1.0  (2026-10-23)
#    - compile() now reads tick timestamps (MidoEvent.timestamp_ticks) and
#      writes them directly. Tempo is applied in exactly one place: the
#      set_tempo message. No ms -> ticks conversion, so no rounding drift.
#    - Event ticks are rescaled if this writer's ticks_per_beat differs from
#      events.TICKS_PER_BEAT.
#    - The original ms-based compile is kept unchanged as compile_legacy_ms(),
#      for comparison or rollback. Call one of the two per writer, not both.
#
#  1.0.0
#    - Original version (ms -> ticks at write time).
# =============================================================================

from mido import MidiFile, MidiTrack, Message, MetaMessage, bpm2tempo
import events as _events
from events import MidoEvent
from sequences import BaseNoteSequence

class MIDIWriter:
    """Writes a given `BaseNoteSequence` into a MIDI file.
    Attributes:
        filename (str): The name of the output file.
        midFile (MidiFile): MIDI File object.
        track (dict[int, MidiTrack]): Mapping between track number and MIDI Track object.
        trackNo (int): Current track number.
        seq (dict[int, BaseNoteSequence]): Mapping between track number and note sequence.
        BPM (int): The default BPM of a new MIDI track (defaults to 125)
        TICKS_PER_BEAT (int): The ticks per beat used to write into the midi file.
    """
    def __init__(self, filename: str, bpm: float = 125.00, ticks_per_beat: int = 480):
        self.filename: str = filename
        self.BPM: float = bpm
        self.TICKS_PER_BEAT: int = ticks_per_beat
        self.midFile: MidiFile = MidiFile(ticks_per_beat = self.TICKS_PER_BEAT)
        self.tracks: dict[int, MidiTrack] = {0: MidiTrack()}
        self.midFile.tracks.append(self.tracks[0])
        self.trackNo: int = 0
        self.seq: dict[int, BaseNoteSequence] = {}
        self.trackInfo: dict[int, dict] = {}
        self.time_signature: tuple[int, int] | None = None

    def addSequence(self, seq : BaseNoteSequence, program: int|None = None, channel: int = 0,
                    name: str|None = None):
        """Adds a sequence to a the current track and initializes the next track.
        Args:
            seq: The BaseNoteSequence to append.
            program: General MIDI program (0-127) for this track, or None to leave it unset.
            channel: MIDI channel (0-15) for this track's notes.
            name: Optional track name.
        """
        if not 0 <= channel <= 15:
            raise ValueError("channel must be within [0, 15]")
        if program is not None and not 0 <= program <= 127:
            raise ValueError("program must be within [0, 127]")
        self.trackInfo[self.trackNo] = {"program": program, "channel": channel, "name": name}
        self.seq[self.trackNo] = seq
        self.trackNo += 1
        self.tracks[self.trackNo] = MidiTrack()
        self.midFile.tracks.append(self.tracks[self.trackNo])
    def compile(self):
        """Writes the file using the tick timeline.
        - Every event's position comes from `timestamp_ticks`; the BPM only
          sets the tempo message, so changing it never moves any note.
        """
        if not self.tracks: return
        self.tracks[0].insert(0, MetaMessage('set_tempo', tempo=bpm2tempo(self.BPM), time=0))
        if self.time_signature is not None:
            num, den = self.time_signature
            self.tracks[0].insert(1, MetaMessage('time_signature', numerator=num, denominator=den, time=0))
        scale = self.TICKS_PER_BEAT / _events.TICKS_PER_BEAT
        for i in range(len(self.seq)):
            seq = self.seq[i]
            track = self.tracks[i]
            info = self.trackInfo.get(i, {"program": None, "channel": 0, "name": None})
            channel = info["channel"]
            if info["name"]:
                track.append(MetaMessage('track_name', name=info["name"], time=0))
            if info["program"] is not None:
                track.append(Message('program_change', program=info["program"], channel=channel, time=0))
            previous = 0 #absolute tick of the previous message
            midoEvents: list[MidoEvent] = seq.resolve(True)
            for event in midoEvents:
                if event.timestamp_ticks < 0:
                    raise ValueError("Event has no tick timestamp; was the sequence resolved?")
                absolute = int(round(event.timestamp_ticks * scale))
                note = event.note
                track.append(Message(event.event_type,
                                     note = note.number,
                                     velocity = note.velocity,
                                     channel = channel,
                                     time = absolute - previous))
                previous = absolute

        self.midFile.save(self.filename + ".mid")

    def compile_legacy_ms(self):
        """Version 1.0.0 compile, unchanged: converts ms timestamps into ticks
        at the writer's BPM. Kept for comparison and rollback."""
        if not self.tracks: return
        self.tracks[0].insert(0, MetaMessage('set_tempo', tempo=bpm2tempo(self.BPM), time=0))
        for i in range(len(self.seq)):
            seq = self.seq[i]
            track = self.tracks[i]
            time = 0 #time in ms
            midoEvents: list[MidoEvent] = seq.resolve(True)
            midoEvents.sort(key= lambda x: x.timestamp) # 1.0.0 ordering (resolve now orders by tick)
            bpm = self.BPM
            for event in midoEvents:
                deltaTime = event.timestamp - time
                note = event.note
                track.append(Message(event.event_type,
                                     note = note.number,
                                     velocity = note.velocity,
                                     time = int(round((deltaTime * bpm * self.TICKS_PER_BEAT)/60000))))
                time += deltaTime

        self.midFile.save(self.filename + ".mid")