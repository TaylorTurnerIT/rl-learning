from dataclasses import dataclass
from enum import Enum


class Direction(Enum):
    UP = "up"
    DOWN = "down"
    LEFT = "left"
    RIGHT = "right"


class State(Enum):
    ALIVE = 0
    DEAD = 1
    WON = 2


@dataclass(
    eq=True,
    slots=False,
)
class Position:
    x: int
    y: int

    def __init__(self, x, y):
        self.x = x
        self.y = y


class Game:
    def __init__(
        self,
        x_size: int,
        y_size: int,
        pit_pos: Position,
        wumpus_pos: Position,
        win_pos: Position,
    ):
        self.x_size: int = x_size
        self.y_size: int = y_size
        self.pit_pos: Position = pit_pos
        self.wumpus_pos: Position = wumpus_pos
        self.win_pos: Position = win_pos

    def get_valid_moves(self, current_pos: Position) -> list[Direction]:
        """
        Check adjacent squares for being out of bounds
        """
        valid_moves: list[Direction] = []

        move_left = current_pos.x - 1
        if 0 < move_left < self.x_size:
            valid_moves.append(Direction.LEFT)

        move_right = current_pos.x + 1
        if 0 < move_right < self.x_size:
            valid_moves.append(Direction.RIGHT)

        move_down = current_pos.y - 1
        if 0 < move_down < self.y_size:
            valid_moves.append(Direction.DOWN)

        move_up = current_pos.y + 1
        if 0 < move_up < self.y_size:
            valid_moves.append(Direction.UP)

        return valid_moves

    def check_pos(self, pos: Position) -> State:
        match pos:
            case self.pit_pos:
                return State.DEAD
            case self.wumpus_pos:
                return State.DEAD
            case self.win_pos:
                return State.WON
        return State.ALIVE
