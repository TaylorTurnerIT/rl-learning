import random

from rl_learning.game import Direction


class Brain:
    def __init__(self, mutation_chance: float):
        starting_action_count = 1
        self.actions: list[Direction] = [
            random.choice(list(Direction)) for _ in range(starting_action_count)
        ]
        self.mutation_chance: float = mutation_chance

    def mutate_actions(self):

        self.add_move()

        for action in range(len(self.actions)):
            if self.mutation_chance < random.random():
                continue
            self.actions[action] = random.choice(list(Direction))

    def add_move(self):
        self.actions.append(Direction.UP)
