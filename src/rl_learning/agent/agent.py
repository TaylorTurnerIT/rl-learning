from logging import Logger

from ..env import Env
from ..events import EventEmitter, EventType
from .brain import Brain
from .handler import Fitness


class Agent:
    def __init__(self, id: int, brain: Brain, logger: Logger, env: Env):
        self.id = id
        self.brain = brain
        self.events: EventEmitter = EventEmitter(logger)
        self.win: bool = False
        self.env: Env = env
        self.move_count = 0

        self.events.subscribe(EventType.WIN, self.set_win)

    def run_actions(self):
        for direction in self.brain.actions:
            self.env.move(direction)

    def set_win(self):
        self.win = True

    def get_win_distance(self):
        return abs(self.env.win_idx - self.env.agent_idx)

    def calculate_fitness(self) -> Fitness:
        return Fitness(
            move_count=self.env.map_size,
            win=self.win,
            distance_to_win=self.get_win_distance(),
        )

    def reset(self, new_brain: Brain | None = None):
        self.env.reset()
        self.move_count = 0

        if new_brain is not None:
            self.brain = new_brain
            self.win = False
