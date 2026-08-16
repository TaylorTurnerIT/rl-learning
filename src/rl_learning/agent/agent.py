from ..env import Env
from ..events import EventEmitter, EventType
from .brain import Brain


class Agent:
    def __init__(self, brain: Brain, env: Env, events: EventEmitter):
        self.brain = brain
        self.env = env
        self.events = events
        self.win: bool = False

        self.events.subscribe(EventType.WIN, self.set_win)

    def run_actions(self):
        for direction in self.brain.actions:
            self.env.move(direction)

    def set_win(self):
        self.win = True

    def get_win_distance(self):
        self.env