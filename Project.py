from events import *
from note import *
from sequences import *
from writer import *

#Other code stuff
NoteEvent.DEFAULT_DURATION = 500
seq = (Sequences.relative().start()
       .add(note("C")).add(note("E")).add(note("G")).advance_to_end()
       .add(note("G")).add(note("B")).add(note("D", octave=6)).advance_to_end().default_octave(6)
       .add(note("A", octave=5)).add(note("C")).add(note("E")).advance_to_end().default_octave(5)
       .add(note("F")).add(note("A")).add(note("C", octave=6)))
writer = MIDIWriter("chord_progression")
writer.addSequence(seq)
writer.compile()
