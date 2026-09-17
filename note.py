class Note:
    """Represents a Musical Note.

    Attributes:
        name (str): The name of the note.
        number (int): MIDI note number.
        octave (int): Octave of the note.
        velocity (int): MIDI Velocity (How hard the key is pressed).
        DEFAULT_OCTAVE (int): The default octave if no octave provided.
    """
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
    DEFAULT_OCTAVE = 5
    def __init__(self, name: str, velocity: int = 64, octave: int|None = None):
        """
        Args:
            name: The name of the note.
            velocity: The loudness of the note.
            octave: The octave of the note.
        """
        if octave is None: octave = self.DEFAULT_OCTAVE
        Note.assertNote(name)
        self.name = name
        self.number = 12 * octave + Note.noteDictionary[name[0]]
        if len(name) == 2:
            self.number += Note.sharpFlat[name[1]]
        self.octave = octave
        self.velocity = velocity
    def __copy__(self):
        return type(self)(self.name, self.velocity, self.octave)
    @classmethod
    def assertNote(cls, note: str):
        """Ensures provided note is a valid musical note, raises `ValueError` if not.
        Args:
            note: The note to validate
        """
        failed = False
        if not note: raise ValueError("Please provide a valid note.")
        if len(note) > 2: failed = True
        if note[0] not in cls.noteDictionary: failed = True
        if len(note) == 2 and note[1] not in cls.sharpFlat: failed = True
        if failed:
            raise ValueError("Please provide a valid note.")
    @classmethod
    def getNum(cls, note: str, octave: int):
        """Returns MIDI number of provided note + octave.
        """
        Note.assertNote(note)
        number = 12 * octave + Note.noteDictionary[note[0]]
        if len(note) == 2:
            number += Note.sharpFlat[note[1]]

        return number
    @classmethod
    def default_octave(cls, defOctave: int):
        """The default octave for the note sequence to consider if no octave provided where needed.
        Args:
            defOctave: The default octave (must be between 1 to 10)
        """
        if not(1 <= defOctave <= 10):
            raise ValueError("Octave must be within the range [1, 10]")
        Note.DEFAULT_OCTAVE = defOctave