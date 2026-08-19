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
        parse_commands()
        actions_to_run: list[MovementCommand] = []
        actions_to_run.append(MovementCommand("x", 10))
        actions_to_run.append(MovementCommand("neutral", 10))
        for direction in self.brain.actions:
            actions_to_run.append(MovementCommand(move=direction, duration_ms=100))

    def step(self, direction: MovementCommand):
        pass

    def calculate_fitness(self) -> int:
        score: int = 0
        match self.state:
            case State.DEAD:
                score -= 100000
        return score

    def reset(self):
        self.state = State.ALIVE
