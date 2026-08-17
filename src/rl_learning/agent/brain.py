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

        for action in range(len(self.actions)):
            if self.mutation_chance < random.random():
                continue
            self.actions[action] = self.get_random_move()
        if self.mutation_chance > random.random():
            self.add_random_move()

    def add_random_move(self):
        self.actions.append(self.get_random_move())

    def get_random_move(self) -> Direction:
        return random.choice(list(Direction))
