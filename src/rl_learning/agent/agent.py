from rl_learning.agent.brain import Brain
from rl_learning.game import Direction, Game, Position


class Agent:
    def __init__(self, starting_pos: Position, brain: Brain, game: Game):
        self.pos: Position = starting_pos
        self.brain: Brain = brain
        self.game: Game = game

    def run_actions(self, actions: list[Direction]):
        for direction in actions:
            self.move(self.pos, direction)

    def move(self, pos: Position, direction: Direction):
        valid_directions = self.game.get_valid_moves(pos)
        if direction in valid_directions:
            match direction:
                case Direction.LEFT:
                    pos.x -= 1
                case Direction.RIGHT:
                    pos.x += 1
                case Direction.UP:
                    pos.y += 1
                case Direction.DOWN:
                    pos.y -= 1
