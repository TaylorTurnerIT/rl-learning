from rl_learning.agent.brain import Brain
from rl_learning.game import Game, Position


class Agent:
    def __init__(self, starting_pos: Position, brain: Brain, game: Game):
        self.pos: Position = starting_pos
        self.brain: Brain = brain
        self.game: Game = game
