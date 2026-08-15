import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum


class EventType(Enum):
    WIN = "win"
    MOVE = "move"
    RESET = "reset"


@dataclass
class Event:
    type: EventType
    listeners: list[Callable] = field(default_factory=list)

    def subscribe(self, callback: Callable):
        if callback not in self.listeners:
            self.listeners.append(callback)

    def unsubscribe(self, callback: Callable):
        if callback in self.listeners:
            self.listeners.remove(callback)

    def emit(self, *args, **kwargs):
        for callback in self.listeners.copy():
            callback(*args, **kwargs)


class EventEmitter:
    """
    Uses the Observer/Dispatch pattern to allow event-subscription
    """

    def __init__(self, logger: logging.Logger):
        self._events: dict[EventType, Event] = {}
        self.logger: logging.Logger = logger

    def subscribe(self, event_type: EventType, callback: Callable):
        if event_type not in self._events:
            self._events[event_type] = Event(event_type)
            self.logger.info("created event: %s", event_type)

        self._events[event_type].subscribe(callback)
        self.logger.info("%s subscribed to event: %s", callback, event_type)

    def unsubscribe(self, event_type: EventType, callback: Callable):
        if event_type not in self._events:
            raise ValueError("cannot unsubscribe from event that does not exist")
        self._events[event_type].unsubscribe(callback)

    def emit(self, event_type: EventType, *args, **kwargs):
        if event_type not in self._events:
            raise ValueError("emitted event that does not exist")

        self._events[event_type].emit(*args, **kwargs)
