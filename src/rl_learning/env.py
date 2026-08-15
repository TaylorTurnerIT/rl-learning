import logging
from enum import Enum, IntEnum
from typing import Required, TypedDict

from .events import EventEmitter, EventType


class LogLevel(IntEnum):
    DEBUG = logging.DEBUG
    INFO = logging.INFO
    WARNING = logging.WARNING
    ERROR = logging.ERROR
    CRITICAL = logging.CRITICAL


class Tile(Enum):
    AGENT = "Agent"
    EMPTY = "Empty"
    WIN = "Win"


class Direction(Enum):
    LEFT = "Left"
    RIGHT = "Right"


class EnvBuilder:
    """
    Constructs Env() using the Builder Pattern
    """

    class EnvBuilder:
        def __init__(self):
            self._logger: logging.Logger | None = None
            self._map_size: int | None = None
            self._agent_idx: int | None = None
            self._win_idx: int | None = None

    def logger(self, ext_logger: logging.Logger):
        self._logger = ext_logger
        return self

    def map_size(self, size: int):
        if size <= 1:
            raise ValueError("map size should be greater than 1")
        self._map_size = size
        return self

    def agent_index(self, idx: int):
        self._agent_idx = idx
        return self

    def win_index(self, idx: int):
        self._win_idx = idx
        return self

    def build(self):

        if not (0 <= self._agent_idx < self._map_size):
            raise ValueError(
                "agent_idx is out of bounds. agent_idx:",
                self._win_idx,
                ". map_size:",
                self._map_size,
            )

        if not (0 <= self._win_idx < self._map_size):
            raise ValueError(
                "win_idx is out of bounds. win_idx:",
                self._win_idx,
                ". map_size:",
                self._map_size,
            )

        if self._agent_idx == self._win_idx:
            raise ValueError("agent_idx and win_idx should not be the same")

        return Env(self._logger, self._map_size, self._agent_idx, self._win_idx)


class Env:
    """
    Construct using EnvBuilder()

    The environment which the agent operates in.
    Holds the state of the map including agent position and map parameters.
    """

    def __init__(
        self,
        logger: logging.Logger,
        map_size: int,
        agent_idx: int,
        win_idx: int,
    ) -> None:
        self.LOGGER: logging.Logger = logger
        self.INITIAL_MAP_SIZE: int = map_size
        self.INITIAL_AGENT_IDX: int = agent_idx
        self.INITIAL_WIN_IDX: int = win_idx

        self.agent_idx: int = self.INITIAL_AGENT_IDX
        self.win_idx: int = self.INITIAL_WIN_IDX
        self.map_size: int = self.INITIAL_MAP_SIZE

        self.map: list[Tile] = [Tile.EMPTY for _ in range(self.map_size)]
        self.map[self.agent_idx] = Tile.AGENT
        self.map[self.win_idx] = Tile.WIN

        self.emitter = EventEmitter(self.LOGGER)
        self.emitter.subscribe(EventType.WIN, self.win)
        self.emitter.subscribe(EventType.MOVE, self.move)
        self.emitter.subscribe(EventType.RESET, self.reset)

    def show(self):
        print(self.map)

    def swap_tiles(self, OLD_IDX: int, NEW_IDX: int):
        self.map[OLD_IDX], self.map[NEW_IDX] = (
            self.map[NEW_IDX],
            self.map[OLD_IDX],
        )

    def move(self, direction: Direction):
        """
        Moves the AGENT tile one index left or right
        """
        AGENT_IDX = self.map.index(Tile.AGENT)
        MAX_INDEX = len(self.map)

        match direction:
            case direction.LEFT:
                NEW_IDX = AGENT_IDX - 1

            case direction.RIGHT:
                NEW_IDX = AGENT_IDX + 1

        if 0 <= NEW_IDX < MAX_INDEX:
            self.swap_tiles(OLD_IDX=AGENT_IDX, NEW_IDX=NEW_IDX)
            self.LOGGER.info("moved %s", direction)

    def win(self):
        self.LOGGER.info("you win!")

    def reset(self):
        """
        Returns a new environment with the same initial state used for its creation
        """
        return (
            EnvBuilder()
            .logger(self.LOGGER)
            .map_size(self.INITIAL_MAP_SIZE)
            .agent_index(self.INITIAL_AGENT_IDX)
            .win_index(self.INITIAL_WIN_IDX)
            .build()
        )
