import logging
from ast import Constant, Dict
from enum import IntEnum
from multiprocessing import Value

from typing_extensions import Any, Callable, Literal, Required, TypedDict


class LogLevel(IntEnum):
    DEBUG = logging.DEBUG
    INFO = logging.INFO
    WARNING = logging.WARNING
    ERROR = logging.ERROR
    CRITICAL = logging.CRITICAL


class Tile(IntEnum):
    AGENT = 0
    EMPTY = 1
    WIN = 2


class Direction(IntEnum):
    LEFT = 0
    RIGHT = 1


class EnvParams(TypedDict, total=False):
    logger: Required[logging.Logger]
    level: Required[LogLevel]
    map_size: Required[int]
    agent_idx: Required[int]
    win_idx: Required[int]


class Event(TypedDict):
    name: str
    callback: Callable


class EventEmitter:
    """
    Uses the Observer/Dispatch pattern to allow event-subscription"""

    def __init__(self):
        self._listeners = {}

    def on(self, event: Event):
        self._listeners.setdefault(event["name"], []).append(event["callback"])


class EnvBuilder:
    """
    Constructs Env() using the Builder Pattern
    """

    def __init__(self):
        self._params: EnvParams = {}  # pyright: ignore[reportAttributeAccessIssue]

    def logger(self, ext_logger: logging.Logger, level: LogLevel):
        # Configure global logging settings
        logging.basicConfig(
            level=level,  # Capture INFO and above
            format="%(asctime)s - %(levelname)s - %(message)s",  # Log message structure
        )
        self._params["logger"] = ext_logger
        self._params["level"] = level
        return self

    def map_size(self, size: int):
        if size <= 1:
            raise ValueError("map size should be greater than 1")
        self._params["map_size"] = size
        return self

    def agent_index(self, idx: int):
        self._params["agent_idx"] = idx
        return self

    def win_index(self, idx: int):
        self._params["win_idx"] = idx
        return self

    def build(self):

        if not 0 < self._params["agent_idx"] >= self._params["map_size"]:
            raise ValueError("agent_idx is out of bounds")

        if self._params["agent_idx"] == self._params["win_idx"]:
            raise ValueError("agent_idx and win_idx should not be the same")

        if not 0 < self._params["win_idx"] >= self._params["map_size"]:
            raise ValueError("win_idx is out of bounds")

        return Env(**self._params)


class Env:
    """
    Construct using EnvBuilder()

    The environment which the agent operates in.
    Holds the state of the map including agent position and map parameters.
    """

    def __init__(
        self,
        *,
        logger: logging.Logger,
        level: LogLevel,
        map_size: int,
        agent_idx: int,
        win_idx: int,
    ) -> None:
        self.LOGGER: logging.Logger = logger
        self.LEVEL: LogLevel = level
        self.INITIAL_MAP_SIZE: int = map_size
        self.INITIAL_AGENT_IDX: int = agent_idx
        self.INITIAL_WIN_IDX: int = win_idx

        self.agent_idx: int = self.INITIAL_AGENT_IDX
        self.win_idx: int = self.INITIAL_WIN_IDX
        self.map_size: int = self.INITIAL_MAP_SIZE

        self.map: list[Tile] = [Tile.EMPTY for _ in range(self.map_size)]
        self.map[self.agent_idx] = Tile.AGENT
        self.map[self.win_idx] = Tile.WIN

        self.emitter = EventEmitter()

    def show(self):
        print(self.map)

    def move(self, direction: Direction):
        """
        Moves the AGENT tile one index left or right
        """
        AGENT_IDX = self.map.index(Tile.AGENT)
        MAX_INDEX = len(self.map)

        match direction:
            case direction.LEFT:
                LEFT_IDX = AGENT_IDX - 1
                if 0 <= LEFT_IDX < MAX_INDEX:
                    self.map[AGENT_IDX], self.map[LEFT_IDX] = (
                        self.map[LEFT_IDX],
                        self.map[AGENT_IDX],
                    )
                    self.LOGGER.info("moved left")

            case direction.RIGHT:
                RIGHT_IDX = AGENT_IDX + 1
                if 0 <= RIGHT_IDX < MAX_INDEX:
                    self.map[AGENT_IDX], self.map[RIGHT_IDX] = (
                        self.map[RIGHT_IDX],
                        self.map[AGENT_IDX],
                    )
                    self.LOGGER.info("moved right")

    def win(self):
        pass

    def reset(self):
        """
        Returns a new environment with the same initial state used for its creation
        """
        return (
            EnvBuilder()
            .logger(self.LOGGER, self.LEVEL)
            .map_size(self.INITIAL_MAP_SIZE)
            .agent_index(self.INITIAL_AGENT_IDX)
            .win_index(self.INITIAL_WIN_IDX)
            .build()
        )
