from __future__ import annotations

import copy
from note import Note
from events import NoteEvent, MidoEvent
from abc import ABC, abstractmethod
from typing import overload

class BaseNoteSequence(ABC):
    @abstractmethod
    def resolve(self, Mido_events: bool = False) -> list[NoteEvent]|list[MidoEvent]:
        """Returns the resolved version of the internal timeline, either as a list of `NoteEvent`s or a list of `MidoEvent`s.
        - Returned list is ordered based on timestamp.
        Args:
            Mido_events: if `true` -> returns a list of `MidoEvent`s, else returns a list of `NoteEvent`s.
        """
        ...

class RelativeNotes(BaseNoteSequence):
    """Creates and handles **Relative** Note Sequences.
    Attributes:
        head (_Node|None): Starting of the relative sequence
        tail (_Node|None): Ending of the relative sequence
        cursor (_Node|None): Current active node for relative operations.
        noteCatalogue (dict[str, _Node]): Catalogue of named events to refer/jump the cursor to.
        DEFAULT_OCTAVE (int): The default octave for the sequence to consider.
    """
    class _Node:
        def __init__(self):
            self.data: dict[int, NoteEvent] = {} 
            self.maxDuration = 0
            self.timestamp = -1
            self.next: RelativeNotes._Node|None = None
            self.prev: RelativeNotes._Node|None = None
    def __init__(self):
        self.head: RelativeNotes._Node|None = None
        self.tail: RelativeNotes._Node|None = self.head
        self.cursor: RelativeNotes._Node|None = self.head
        self.noteCatalogue: dict[str, RelativeNotes._Node]= {}
        self.nodeCount = 1
        self.DEFAULT_OCTAVE = 5
    def _insert(self, reference: RelativeNotes._Node|None, toInsert: RelativeNotes._Node):
        """Helper function that inserts before/after provided reference. Assumes provided reference and insertion node fits in the timeline.
        Args:
            reference: The reference node to insert after. None if theres no nodes present yet.
            toInsert: The node to insert.
        """
        if reference is None and self.head is None:
            self.head = toInsert
            self.tail = self.head
        elif reference is not None:
            temp = reference.next
            reference.next = toInsert
            toInsert.prev = reference
            toInsert.next = temp
            if temp: 
                temp.prev = toInsert
            else:
                self.tail = toInsert
        else:
            raise ValueError("Reference cannot be None for a non-empty sequence.")
    def _update_cursor(self, timestamp: int) -> bool:
        """Helper function that updates cursor to the nearest node with `node.timestamp` <= `timestamp`
        Args:
            timestamp: The timestamp to update cursor with respect to.
        Returns:
            Whether existing node found or not.
        """
        if self.cursor.timestamp == timestamp:
            return True
        nextNode = self.cursor.next
        if nextNode:
            if nextNode.timestamp == timestamp:
                self.cursor = nextNode
                return True
            else:
                while(nextNode and nextNode.timestamp < timestamp):
                    self.cursor = nextNode
                    nextNode = nextNode.next
                if nextNode and nextNode.timestamp == timestamp:
                    self.cursor = nextNode
                    return True
        return False
    def start(self, nodeName: str|None = None) -> RelativeNotes:
        """Initializes the sequence and creates the starting node.
        - Can be chained.
        Args:
            nodeName: The name to register node under.  If not provided, defaults to `node 1`.
        """
        if self.head is not None or self.tail is not None:
            raise RuntimeError("Cannot use start once sequence is already initialized.")
        newNode = self._Node()
        newNode.timestamp = 0
        self._insert(self.tail, newNode)
        self.cursor = newNode
        if nodeName is None:
            nodeName = f"node {self.nodeCount}"
            self.nodeCount += 1
        self.noteCatalogue[nodeName] = newNode
        return self
    def advance(self, duration: int, nodeName: str|None = None) -> RelativeNotes:
        """Advances the cursor by given duration, creates a new node if no pre-existing node.
        - Can be chained.
        Args:
            duration: The duration to advance by, in **ms**.
            nodeName: The name to register node under, discarded if node already exists. If not provided, defaults to `node {count}`.
        """
        if self.cursor is None:
            raise RuntimeError("Cannot advance an empty sequence.")
        if duration < 0:
            raise ValueError("Advance duration cannot be negative.")
        nextTimeStamp = self.cursor.timestamp + duration
        found = self._update_cursor(nextTimeStamp)
        if not found:
            if nodeName is None:
                nodeName = f"node {self.nodeCount}"
                self.nodeCount += 1
            if nodeName in self.noteCatalogue:
                raise RuntimeError("Name cannot already exist in catalogue, please choose something else.")
            newNode = self._Node()
            newNode.timestamp = nextTimeStamp
            self._insert(self.cursor, newNode)
            self.cursor = newNode
            
            self.noteCatalogue[nodeName] = newNode
        return self
    @overload
    def advance_to_end(self, extra: int = 0, nodeName: str|None = None) -> RelativeNotes:
        """Advances to the end of the maximum duration of the cursor node. Creates a node if one doesnt exist already.
        - Can be chained.
        Args:
            extra: Any extra duration to add after end of last node. Defaults to `0`.
            nodeName: The name to register node under, discarded if node already exists. If not provided, defaults to `node {count}`.
        """
        ...
    @overload
    def advance_to_end(self, extra: int = 0, nodeName: str|None = None, noteName: str|None = None, octave: int|None = None) -> RelativeNotes:
        """Advances to the end of the specified notes/maximum duration of the previous node. Creates a node if one doesnt exist already.
        - Can be chained.
        Args:
            extra: Any extra duration to add after end of last node. Defaults to `0`.
            nodeName: The name to register node under, discarded if node already exists. If not provided, defaults to `node {count}`.
            noteName: Specifies end of which notes duration.
            octave: Octave of the specified note, defaults to `DEFAULT_OCTAVE`.
        """
        ...   
    def advance_to_end(self, extra: int = 0, nodeName: str|None = None, noteName: str|None = None, octave: int|None = None) -> RelativeNotes:
        if self.cursor is None:
            raise RuntimeError("Cannot advance to the end of an empty sequence.")
        if extra < 0:
            raise ValueError("Extra duration cannot be negative.")
        if octave is None: octave = self.DEFAULT_OCTAVE
        if noteName is None:
            duration = self.cursor.maxDuration
        else:
            noteNum = Note.getNum(noteName, octave)
            duration = (
                self.cursor.data[noteNum].duration
                if noteNum in self.cursor.data
                else self.cursor.maxDuration
            )
        nextTimestamp = self.cursor.timestamp + duration + extra
        found = self._update_cursor(nextTimestamp)
        if not found:
            if nodeName is None:
                nodeName = f"node {self.nodeCount}"
                self.nodeCount += 1
            if nodeName in self.noteCatalogue:
                raise RuntimeError("Name cannot already exist in catalogue, please choose something else.")
            newNode = self._Node()
            newNode.timestamp = nextTimestamp
            self._insert(self.cursor, newNode)
            self.cursor = newNode
            
            self.noteCatalogue[nodeName] = newNode
        return self
    def add(self, event: NoteEvent) -> RelativeNotes:
        """Adds another event to the node cursor is currently at.
        - Note: merges notes if they already exist. (mearge logic -> set duration as maximum).
        - Can be chained.
        Args:
            event: The event to add to the cursors node.
        """
        if self.cursor is None:
            raise RuntimeError("Cannot add to an empty sequence.")
        noteNum = event.note.number
        currentNode = self.cursor
        if noteNum in currentNode.data:
            existingDuration = currentNode.data[noteNum].duration
            currentNode.data[noteNum].duration = max(existingDuration, event.duration)
            currentNode.maxDuration = max(currentNode.maxDuration, event.duration)
        else:
            currentNode.data[noteNum] = copy.copy(event)
            currentNode.maxDuration = max(currentNode.maxDuration, event.duration)
        return self
    def set(self, event: NoteEvent) -> RelativeNotes:
        """Sets an event at the node the cursor is currently at.
        - Replaces an existing event with the same note.
        - Adds the event if that note does not already exist.
        - Can be chained.

        Args:
            event: The event to set at the cursor's node.
        """
        if self.cursor is None:
            raise RuntimeError("Cannot set event in an empty sequence.")
        noteNum = event.note.number
        currentNode = self.cursor
        old = currentNode.data.get(noteNum)
        currentNode.data[noteNum] = copy.copy(event)

        if old is None:
            currentNode.maxDuration = max(currentNode.maxDuration, event.duration)
        elif old.duration == currentNode.maxDuration and event.duration < old.duration:
            currentNode.maxDuration = max(
                (e.duration for e in currentNode.data.values()),
                default=0
            )
        else:
            currentNode.maxDuration = max(currentNode.maxDuration, event.duration)
        return self
    def jump_to(self, nodeName: str) -> RelativeNotes:
        """Moves cursor to provided node name.
        - Can be chained.
        Args:
            nodeName: The node to jump cursor to.
        """
        if nodeName not in self.noteCatalogue:
            raise ValueError("Provided node does not exist in catalogue.")
        self.cursor = self.noteCatalogue[nodeName]

        return self
    def jump_to_end(self) -> RelativeNotes:
        """Moves cursor to the end of the timeline list.
        - Can be chained.
        """
        self.cursor = self.tail

        return self
    def remove(self, noteName: str, octave: int|None = None) -> RelativeNotes:
        """Removes the provided note from the cursor's node.
        - Can be chained.
        Args:
            noteName: The name of the note to remove.
            octave: The octave of the note, defaults to `DEFAULT_OCTAVE`.
        """
        if self.cursor is None:
            raise RuntimeError("Cannot remove from an empty sequence.")
        if octave is None: octave = self.DEFAULT_OCTAVE
        noteNum = Note.getNum(noteName, octave)
        if noteNum not in self.cursor.data: 
            raise ValueError("Provided note does not exist in current node.")
        note = self.cursor.data[noteNum]
        del self.cursor.data[noteNum]
        if note.duration == self.cursor.maxDuration:
            self.cursor.maxDuration = max(
                (e.duration for e in self.cursor.data.values()),
                default=0
            )
        
        return self
    def default_octave(self, defOctave: int) -> RelativeNotes:
        """The default octave for the note sequence to consider if no octave provided where needed.
        - Can be chained.
        Args:
            defOctave: The default octave (must be between 1 to 10)
        """
        if not(1 <= defOctave <= 10):
            raise ValueError("Octave must be within the range [1, 10]")
        self.DEFAULT_OCTAVE = defOctave

        return self

    def resolve(self, Mido_events: bool = False) -> list[NoteEvent]|list[MidoEvent]:
        """Resolves the internal timeline, and returns either a list of `NoteEvent`s or a list of `MidoEvent`s based on provided flag. 
        - Returned list is ordered based on timestamp.
        Args:
            Mido_events: if `true` -> returns a list of `MidoEvent`s, else returns a list of `NoteEvent`s.
        """
        if Mido_events:
            eventList : list[MidoEvent] = []
            itr = self.head
            while itr:
                for noteEvent in itr.data.values():
                    noteEvent.timestamp = itr.timestamp
                    eventList.extend(MidoEvent.from_note_event(noteEvent))
                itr = itr.next
            eventList.sort(key= lambda x: x.timestamp)
        else:
            eventList: list[NoteEvent] = []
            itr = self.head
            while itr:
                for noteEvent in itr.data.values():
                    noteEvent.timestamp = itr.timestamp
                    eventList.append(copy.copy(noteEvent))
                itr = itr.next

        return eventList

class AbsoluteNotes(BaseNoteSequence):
    """Creates and handles **Absolute** Note Sequences.
    Attributes:
        events (list[NoteEvent]): List of absolute note events.
        noteCatalogue (dict[str, NoteEvent]): Catalogue of named note events.
        eventCount (int): Counter used for automatically generated event names.
    """
    def __init__(self):
        self.events: list[NoteEvent] = []
        self.noteCatalogue: dict[str, NoteEvent] = {}
        self.eventCount = 1

    def add_event(self, event: NoteEvent, eventName: str|None = None) -> AbsoluteNotes:
        """Adds a note event to the sequence.
        - Can be chained.
        Args:
            event: The event to add to the sequence.
            eventName: The name to register event under. If not provided, defaults to `event {count}`.
        """
        if event.timestamp < 0:
            raise ValueError("Absolute NoteEvent must have a valid timestamp.")
        if eventName is None:
            eventName = f"event {self.eventCount}"
            self.eventCount += 1
        if eventName in self.noteCatalogue:
            raise RuntimeError("Name cannot already exist in catalogue, please choose something else.")
        self.events.append(event)
        self.noteCatalogue[eventName] = event
        return self

    def remove_event(self, eventName: str) -> AbsoluteNotes:
        """Marks the provided note event for removal.
        - Event is removed during resolution.
        - Can be chained.
        Args:
            eventName: The name of the event to remove.
        """
        if eventName not in self.noteCatalogue:
            raise ValueError("Provided event does not exist in catalogue.")
        self.noteCatalogue[eventName].duration = 0

        return self

    def resolve(self, Mido_events: bool = False) -> list[NoteEvent]|list[MidoEvent]:
        """Resolves the internal timeline, and returns either a list of `NoteEvent`s or a list of `MidoEvent`s based on provided flag.
        - Returned list is ordered based on timestamp.
        - Events with duration `0` are discarded.
        Args:
            Mido_events: if `true` -> returns a list of `MidoEvent`s, else returns a list of `NoteEvent`s.
        """
        activeEvents = [
            event for event in self.events
            if event.duration > 0
        ]

        if Mido_events:
            eventList: list[MidoEvent] = []
            for event in activeEvents:
                eventList.extend(MidoEvent.from_note_event(event))
            eventList.sort(key= lambda x: x.timestamp)
        else:
            eventList: list[NoteEvent] = []
            for event in activeEvents:
                eventList.append(copy.copy(event))
            eventList.sort(key= lambda x: x.timestamp)

        return eventList

class Sequences:
    """Sequence Factory that can be used to create both Relative sequencing and Absolute sequencing."""
    @classmethod
    def relative(cls) -> RelativeNotes:
        """Returns an instance of a relative note sequence builder."""
        return RelativeNotes()
    @classmethod
    def absolute(cls) -> AbsoluteNotes:
        """Returns an instance of an absolute note sequence builder."""
        return AbsoluteNotes()