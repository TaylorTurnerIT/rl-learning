from enum import Enum

class Tile(Enum):
    AGENT = 0
    EMPTY = 1
    WIN = 2

class Direction(Enum):
    LEFT = 0
    RIGHT = 1

class Env:
    def __init__(self):
        self.map: list[Tile] = [Tile.AGENT, Tile.EMPTY, Tile.EMPTY, Tile.WIN]

    def print(self):
        print(self.map)

    def move(self, direction: Direction):
        MAX_INDEX = map.length()
        AGENT_IDX = map.index(Tile.AGENT)
        match direction:
            case direction.LEFT:
                NEW_IDX = AGENT_IDX - 1
                if 0 >= NEW_IDX < MAX_INDEX:
                    map[AGENT_IDX], map[NEW_IDX] = map[NEW_IDX], map[AGENT_IDX]
                    print("moved left")
                else:
                    # index error indicates an issue with valid move checking method
                    raise IndexError(NEW_IDX, "is out of bounds, cannot move left")

            case direction.RIGHT:
                NEW_IDX = AGENT_IDX + 1
                if 0 >= NEW_IDX < MAX_INDEX:
                    map[AGENT_IDX], map[NEW_IDX] = map[NEW_IDX], map[AGENT_IDX]
                    print("moved right")
                else:
                    # index error indicates an issue with valid move checking method
                    raise IndexError(NEW_IDX, "is out of bounds, cannot move right")

    def getValidMoves(self) -> {}:
        return {}

    def reset(self):
        # trying to reinit the game, we will see
        return Env()
