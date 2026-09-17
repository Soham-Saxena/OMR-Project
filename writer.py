from mido import MidiFile, MidiTrack, Message, MetaMessage, bpm2tempo
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

    def addSequence(self, seq : BaseNoteSequence):
        """Adds a sequence to a the current track and initializes the next track.
        Args:
            seq: The BaseNoteSequence to append.
        """
        self.seq[self.trackNo] = seq
        self.trackNo += 1
        self.tracks[self.trackNo] = MidiTrack()
        self.midFile.tracks.append(self.tracks[self.trackNo])
    def compile(self):
        if not self.tracks: return
        self.tracks[0].insert(0, MetaMessage('set_tempo', tempo=bpm2tempo(self.BPM), time=0))
        for i in range(len(self.seq)):
            seq = self.seq[i]
            track = self.tracks[i]
            time = 0 #time in ms
            midoEvents: list[MidoEvent] = seq.resolve(True)
            bpm = self.BPM
            for event in midoEvents:
                deltaTime = event.timestamp - time
                note = event.note
                track.append(Message(event.event_type,
                                     note = note.number,
                                     velocity = note.velocity,
                                     time = round((deltaTime * bpm * self.TICKS_PER_BEAT)/60000)))
                time += deltaTime

        self.midFile.save(self.filename + ".mid")