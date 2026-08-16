import logging
from enum import Enum
from typing import Final

from rl_learning.agent import agent

from .events import EventEmitter, EventType


class Direction(Enum):
    LEFT = "Left"
    RIGHT = "Right"


class EnvParams(Enum):
    map_size: int
    agent_idx: int
    win_idx: int
    logger: logging.Logger


class EnvBuilder:
    """
    Constructs Env() using the Builder Pattern
    """

    def __init__(self, params: EnvParams | None = None):
        self.env: Env

        self._map_size: int
        self._agent_idx: int
        self._win_idx: int
        self._logger: logging.Logger

        if params is not None:
            self._map_size = params.map_size
            self._logger = params.logger
            self._agent_idx = params.agent_idx
            self._win_idx = params.win_idx
            self.build()

    def map_size(self, size: int):
        self._map_size = size
        return self

    def agent_index(self, idx: int):
        self._agent_idx = idx
        return self

    def win_index(self, idx: int):
        self._win_idx = idx
        return self

    def logger(self, logger: logging.Logger):
        self._logger = logger
        return self

    def build(self):
        if self._map_size <= 1:
            raise ValueError(
                "map size should be greater than 1. map_size:", self._map_size
            )

        if not (0 <= self._agent_idx < self._map_size):
            raise ValueError(
                "agent_idx is out of bounds. agent_idx:",
                self._agent_idx,
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

        self.env = Env(
            map_size=self._map_size,
            agent_idx=self._agent_idx,
            win_idx=self._win_idx,
            logger=self._logger,
        )

        return Env


class Env:
    """
    Construct using EnvBuilder()

    The environment which the agent operates in.
    Holds the state of the map including agent position and map parameters.
    """

    def __init__(
        self,
        map_size: int,
        agent_idx: int,
        win_idx: int,
        logger: logging.Logger,
    ) -> None:
        self.LOGGER: Final[logging.Logger] = logger
        self.INITIAL_AGENT_IDX: Final[int] = agent_idx
        self.INITIAL_WIN_IDX: Final[int] = win_idx

        self.agent_idx: int = self.INITIAL_AGENT_IDX
        self.win_idx: int = self.INITIAL_WIN_IDX

        self.map_size: int = map_size

        self.emitter = EventEmitter(self.LOGGER)
        self.emitter.subscribe(EventType.WIN, self.win)
        self.emitter.subscribe(EventType.MOVE, self.move)
        self.emitter.subscribe(EventType.RESET, self.reset)

    def show(self):
        map_visual = ["empty" for _ in range(self.map_size)]
        map_visual[self.agent_idx] = "agent"
        map_visual[self.win_idx] = "win"
        print(map_visual)

    def move(self, direction: Direction):
        """
        Moves the AGENT index left or right
        """

        match direction:
            case direction.LEFT:
                NEW_IDX = self.agent_idx - 1

            case direction.RIGHT:
                NEW_IDX = self.agent_idx + 1

        if not (0 <= NEW_IDX < self.map_size):
            return

        if self.win_idx == NEW_IDX:
            self.emitter.emit(EventType.WIN)

        self.agent_idx = NEW_IDX

        self.LOGGER.info("moved %s", direction)

    def win(self):
        self.LOGGER.info("you win!")

    def reset(self):
        """
        Returns a new environment with the same initial state used for its creation
        """
        return (
            EnvBuilder()
            .map_size(self.map_size)
            .agent_index(self.INITIAL_AGENT_IDX)
            .win_index(self.INITIAL_WIN_IDX)
            .build()
        )
