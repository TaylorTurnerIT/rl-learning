from copy import deepcopy
from enum import Enum

from dodge.agent.brain import Brain
from dodge.control import MovementCommand, execute_commands, parse_commands
from dodge.headless import HeadlessResult, run_headless


class State(Enum):
    ALIVE = 0
    DEAD = 1


class Agent:
    def __init__(self, brain: Brain):
        self.brain: Brain = brain
        self.state: State = State.ALIVE

    def run_actions(self) -> HeadlessResult:
        return run_headless(commands=self.brain.parse_actions(), seed=42)

    def calculate_fitness(self, result: HeadlessResult) -> int:
        score: int = 0
        score += int(result["score"]) * 100
        return score

    def reset(self, new_brain: Brain | None = None):
        self.state = State.ALIVE
        if new_brain is not None:
            self.brain = deepcopy(new_brain)
