import random

from ..env import Direction


class Brain:
    def __init__(self, action_count: int):
        self.actions: list[Direction] = []
        self.action_count: int
        self.mutation_chance: float = 0.2

    def mutate(self):
        self.mutate_action_count()
        self.mutate_actions()

    def mutate_action_count(self):
        if self.mutation_chance > 1:
            raise ValueError(
                "mutation_chance should not exceed 1: ", self.mutation_chance
            )
        if self.action_count <= 0:
            raise ValueError("length should be gt 0: ", self.action_count)
        if random.random() > self.mutation_chance:
            return

        self.action_count -= random.randrange(-3, 3)

    def mutate_actions(self):
        if self.mutation_chance > 1:
            raise ValueError(
                "mutation_chance should not exceed 1: ", self.mutation_chance
            )

        # resize the array if needed
        difference = len(self.actions) - self.action_count
        if difference < 0:
            self.actions[:difference]

        for _ in range(difference):
            self.actions.append(Direction.LEFT)

        # generate random values
        for index in range(len(self.actions)):
            if random.random() < self.mutation_chance:
                self.actions[index] = random.choice([Direction.LEFT, Direction.RIGHT])
