import random

from dodge.control import DIRECTION_KEYS, MovementCommand, parse_commands

# MovementCommand("x", 50)


class Brain:
    def __init__(self, mutation_chance: float):
        starting_action_count = 3
        self.actions: list[str] = [
            random.choice(list(DIRECTION_KEYS)) for _ in range(starting_action_count)
        ]
        self.mutation_chance: float = mutation_chance
        parse_commands(self.actions)

    def mutate_actions(self):
        for action in range(len(self.actions)):
            if self.mutation_chance < random.random():
                continue
            self.actions[action] = self.get_random_move()
        if self.mutation_chance > random.random():
            self.add_random_move()

    def add_random_move(self):
        self.actions.append(self.get_random_move())

    def get_random_move(self) -> str:
        return random.choice(list(DIRECTION_KEYS))
