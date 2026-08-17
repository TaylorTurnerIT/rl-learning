import random

from rl_learning.game import Direction


class Brain:
    def __init__(self, mutation_chance: float):
        starting_action_count = 3
        self.actions: list[Direction] = [
            random.choice(list(Direction)) for _ in range(starting_action_count)
        ]
        self.mutation_chance: float = mutation_chance

    def mutate_actions(self):
        if self.mutation_chance > random.random():
            return

        self.actions.append(Direction.UP)

        for action in range(len(self.actions)):
            self.actions[action] = random.choice(list(Direction))
