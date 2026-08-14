import logging
from enum import Enum

# Configure global logging settings
logging.basicConfig(
    level=logging.DEBUG,  # Capture INFO and above
    format="%(asctime)s - %(levelname)s - %(message)s",  # Log message structure
)


class Tile(Enum):
    AGENT = 0
    EMPTY = 1
    WIN = 2


class Direction(Enum):
    LEFT = 0
    RIGHT = 1


class Env:
    def __init__(
        self,
        logger: logging.Logger,
        map_size: int = 4,
        agent_idx: int = 0,
        win_idx: int = 3,
    ):
        self.logger: logging.Logger = logger

        self.map: list[Tile] = [Tile.EMPTY for _ in range(map_size)]
        self.map[agent_idx] = Tile.AGENT
        self.map[win_idx] = Tile.WIN

    def show(self):
        print(self.map)

    def move(self, direction: Direction):
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
                    self.logger.info("moved left")
                return

            case direction.RIGHT:
                RIGHT_IDX = AGENT_IDX + 1
                if 0 <= RIGHT_IDX < MAX_INDEX:
                    self.map[AGENT_IDX], self.map[RIGHT_IDX] = (
                        self.map[RIGHT_IDX],
                        self.map[AGENT_IDX],
                    )
                    self.logger.info("moved right")
                return

    def reset(self):
        # trying to reinit the game, we will see
        return self.__class__(self.logger)
