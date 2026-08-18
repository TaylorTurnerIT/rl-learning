from copy import deepcopy

from rl_learning.agent.brain import Brain
from rl_learning.game import Direction, Game, Position, State


class Agent:
    def __init__(self, starting_pos: Position, brain: Brain, game: Game):
        self.starting_pos: Position = deepcopy(starting_pos)
        self.pos: Position = starting_pos
        self.brain: Brain = brain
        self.game: Game = game
        self.state: State = State.ALIVE

    def run_actions(self):
        for direction in self.brain.actions:
            self.move(self.pos, direction)
            match self.game.check_pos(self.pos):
                case State.DEAD:
                    self.state = State.DEAD
                    break
                case State.WON:
                    self.state = State.WON
                    break

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

    def calculate_fitness(
        self,
    ) -> int:
        score: int = 0
        match self.state:
            case State.DEAD:
                score -= 10000
        score -= len(self.brain.actions) * 10

        distance = self.game.calculate_distance(self.pos, self.game.win_pos)
        starting_distance = self.game.calculate_distance(
            self.starting_pos, self.game.win_pos
        )
        progress = starting_distance - distance
        score += progress**2 * 100
        return score

    def reset(self):
        self.pos = deepcopy(self.starting_pos)
        self.state = State.ALIVE
