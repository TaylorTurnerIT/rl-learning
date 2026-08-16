# Genetic Algorithm
# Permutation encoding
#
# For each agent:
#   Create a random list of instructions with either Tile.Left or Tile.Right
import numpy as np

from ..env import Direction


class Brain:
    def __init__(self, population: int = 100):
        self.actions: list[Direction] = []

    def randomize_actions(self, length: int, mutation_chance: float):
        pass
