from logging import Logger

from ..env import Env
from ..events import EventEmitter, EventType
from .brain import Brain


class Agent:
    def __init__(self, id:int, brain: Brain, env: Env, logger: Logger):
        self.id = id
        self.brain = brain
        self.env = env
        self.events: EventEmitter = EventEmitter(logger)
        self.win: bool = False

        self.events.subscribe(EventType.WIN, self.set_win)

    def run_actions(self):
        for direction in self.brain.actions:
            self.env.move(direction)

    def set_win(self):
        self.win = True

    def get_win_distance(self):
        return abs(self.env.win_idx - self.env.agent_idx)

    def calc_fitness(self) -> [Bool, ]
