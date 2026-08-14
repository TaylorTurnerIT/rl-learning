import logging
from enum import Enum

# Configure global logging settings
logging.basicConfig(
    level=logging.INFO,  # Capture INFO and above
    format="%(asctime)s - %(levelname)s - %(message)s",  # Log message structure
    filename="%(asctime)-game-session.log",  # Write to file instead of console
    filemode="w",  # 'w' to overwrite, 'a' to append
)


class Tile(Enum):
    AGENT = 0
    EMPTY = 1
    WIN = 2


class Direction(Enum):
    LEFT = 0
    RIGHT = 1


class MoveResult(Enum):
    MOVED_LEFT = 0
    MOVED_RIGHT = 1
    INVALID_MOVE = 2


class Env:
    def __init__(self, logger: logging.Logger):
        self.logger: logging.Logger = logger
        self.map: list[Tile] = [Tile.AGENT, Tile.EMPTY, Tile.EMPTY, Tile.WIN]

    def show(self):
        print(self.map)

    def move(self, direction: Direction) -> MoveResult:
        AGENT_IDX = self.map.index(Tile.AGENT)
        valid_moves = self._getValidMoves()
        if not direction in valid_moves:
            return MoveResult.INVALID_MOVE

        match direction:
            case direction.LEFT:
                LEFT_IDX = AGENT_IDX - 1
                self.map[AGENT_IDX], self.map[LEFT_IDX] = (
                    self.map[LEFT_IDX],
                    self.map[AGENT_IDX],
                )
                self.logger.info("moved left")
                return MoveResult.MOVED_LEFT

            case direction.RIGHT:
                RIGHT_IDX = AGENT_IDX + 1
                self.map[AGENT_IDX], self.map[RIGHT_IDX] = (
                    self.map[RIGHT_IDX],
                    self.map[AGENT_IDX],
                )
                self.logger.info("moved right")
                return MoveResult.MOVED_RIGHT

    def _getValidMoves(self) -> list[Direction]:
        MAX_INDEX = len(self.map)
        AGENT_IDX = self.map.index(Tile.AGENT)
        valid_moves: list[Direction] = []

        LEFT_IDX = AGENT_IDX - 1
        if 0 <= LEFT_IDX < MAX_INDEX:
            valid_moves.append(Direction.LEFT)

        RIGHT_IDX = AGENT_IDX + 1
        if 0 <= RIGHT_IDX < MAX_INDEX:
            valid_moves.append(Direction.RIGHT)

        return valid_moves

    def reset(self):
        # trying to reinit the game, we will see
        return self.__class__(self.logger)
