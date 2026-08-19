from copy import deepcopy
from enum import Enum

from dodge.agent.brain import Brain
from dodge.headless import HeadlessResult, run_headless


class State(Enum):
    ALIVE = 0
    DEAD = 1


class Agent:
    def __init__(self, brain: Brain):
        self.brain: Brain = brain
        self.state: State = State.ALIVE

    def run_actions(self) -> HeadlessResult:
        result = run_headless(commands=self.brain.parse_actions(), seed=42)
        self.state = State.DEAD if result["died"] else State.ALIVE
        return result

    def calculate_fitness(self, result: HeadlessResult) -> int:
        return int(result["survival_frames"])

    def reset(self, new_brain: Brain | None = None):
        self.state = State.ALIVE
        if new_brain is not None:
            self.brain = deepcopy(new_brain)
