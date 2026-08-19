from copy import deepcopy
from enum import Enum

from dodge.agent.brain import Brain
from dodge.control import MovementCommand, parse_commands


class State(Enum):
    ALIVE = 0
    DEAD = 1


class Agent:
    def __init__(self, brain: Brain):
        self.brain: Brain = brain
        self.state: State = State.ALIVE

    def run_actions(self):
        for _ in self.brain.actions:
            pass

    def calculate_fitness(self) -> int:
        score: int = 0
        match self.state:
            case State.DEAD:
                score -= 100000
        return score

    def reset(self, new_brain: Brain | None = None):
        self.state = State.ALIVE
        if new_brain is not None:
            self.brain = deepcopy(new_brain)
