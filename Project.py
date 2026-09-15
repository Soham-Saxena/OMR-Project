from mido import Message, MidiFile, MidiTrack

class Note:
    noteDictionary = {
        'C' : 0,
        'D' : 2,
        'E' : 4,
        'F' : 5,
        'G' : 7,
        'A' : 9,
        'B' : 11 
    }
    sharpFlat = {
        '#' : 1,
        'b' : -1
    }
    def __init__(self, name, duration, octave=5):
        self.name = name
        self.number = 12 * octave + Note.noteDictionary[name[0]]
        if len(name) == 2:
            self.number += Note.sharpFlat[name[1]]
        self.octave = octave
        self.duration = duration

class NoteSequence:
    def __init__(self):
        self.notes : list[Note] = list()
    def appendNote(self, note: Note):
        self.notes.append(note)
        
class MIDIWriter:
    def __init__(self, filename):
        self.filename = filename
        self.midFile = MidiFile()
        self.track = MidiTrack()
        self.midFile.tracks.append(self.track)
    def addSequence(self, seq : NoteSequence):
        self.seq = seq
    def compile(self):
        for note in self.seq.notes:
            self.track.append(Message('note_on', note=note.number, velocity=64, time=0))
            self.track.append(Message('note_off', note=note.number, velocity=64, time=note.duration))
        
        self.midFile.save(self.filename + ".mid")
        print("Compiled and saved.")
    
notes = NoteSequence()
noteNames = ['C', 'D', 'E', 'F', 'G', 'A', 'B']
for name in noteNames:
    notes.appendNote(Note(name, 400))
notes.appendNote(Note('C', 400, 6))

NoteCompiler = MIDIWriter("TestFile")
NoteCompiler.addSequence(notes)
NoteCompiler.compile()
    
# # 1. Initialize the file and track
# mid = MidiFile()
# track = MidiTrack()
# mid.tracks.append(track)

# # 2. Add notes to the track
# # MIDI notes are numbers: 60 is Middle C, 62 is D, 64 is E.
# # 'time' is the delta time in ticks. 480 is typically one beat.

# # -- First Note: Middle C --
# # time=0 means this happens immediately.
# track.append(Message('note_on', note=60, velocity=64, time=0))
# # time=480 means wait 1 beat, then turn the note off.
# track.append(Message('note_off', note=60, velocity=64, time=480))

# # -- Second Note: D --
# # time=0 means this happens immediately AFTER the previous message.
# track.append(Message('note_on', note=62, velocity=64, time=0))
# track.append(Message('note_off', note=62, velocity=64, time=480))

# # -- Third Note: E --
# track.append(Message('note_on', note=64, velocity=64, time=0))
# track.append(Message('note_off', note=64, velocity=64, time=480))

# # 3. Save the file
# mid.save('basic_melody.mid')
# print("File 'basic_melody.mid' has been created successfully!")