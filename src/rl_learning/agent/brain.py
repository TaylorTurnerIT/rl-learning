import random

import numpy as np

from ..env import Direction


class Brain:
    def __init__(self, population: int = 100):
        self.actions: list[Direction] = []

    def randomize_actions(self, length: int, mutation_chance: float):
        if mutation_chance > 1:
            raise ValueError("mutation_chance should not exceed 1: ", mutation_chance)
        if length <= 0:
            raise ValueError("length should be gt 0: ", length)

        # resize the array if needed
        difference = len(self.actions) - length
        if difference < 0:
            self.actions[:difference]

        for _ in range(difference):
            self.actions.append(Direction.LEFT)

        # generate random values
        for index in range(len(self.actions)):
            if random.random() < mutation_chance:
                self.actions[index] = random.choice([Direction.LEFT, Direction.RIGHT])
