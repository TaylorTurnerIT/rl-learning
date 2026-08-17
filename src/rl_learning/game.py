from dataclasses import dataclass
from enum import Enum


class Direction(Enum):
    UP = "up"
    DOWN = "down"
    LEFT = "left"
    RIGHT = "right"


@dataclass(
    eq=True,
    slots=False,
)
class Position:
    x: int
    y: int
    def __init__(self, pos: tuple[int,int]):
        self.x = pos[0]
        self.y = pos[1]

class Game:
    def __init__(self, pit_pos: Position, wumpus_pos: Position):
        self.x_size: int
        self.y_size: int
        self.pit_pos: Position = pit_pos
        self.wumpus_pos: Position = wumpus_pos

    def get_valid_moves(self, current_pos: Position) -> list[Direction]:
        """
        Check adjacent squares for being out of bounds
        """
        valid_moves: list[Direction] = []

        move_left = current_pos.x - 1
        if move_left > 0:
            valid_moves.append(Direction.LEFT)

        move_right = current_pos.x + 1
        if move_right > 0:
            valid_moves.append(Direction.RIGHT)

        move_down = current_pos.y - 1
        if move_down > 0:
            valid_moves.append(Direction.DOWN)

        move_up = current_pos.y + 1
        if move_up > 0:
            valid_moves.append(Direction.UP)

        return valid_moves

    # TODO: move to agent, let them control their state, env tells only valid moves
    def move(self, pos: Position, direction: Direction):
        valid_directions = self.get_valid_moves(pos)
        if direction in valid_directions:
            match direction:
                case Direction.LEFT:
                    pos.x - 1
                case Direction.RIGHT:
                    pos.x + 1
                case Direction.UP:
                    pos.y + 1
                case Direction.DOWN:
                    pos.y - 1

    def check_pos(self, pos: Position):
        # check win and lose
        for x in range(self.x_size):
            for y in range(self.y_size):
                if pos == self.pit_pos or pos == self.wumpus_pos
