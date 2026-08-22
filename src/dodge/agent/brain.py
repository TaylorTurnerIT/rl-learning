import random

from dodge.control import DIRECTION_KEYS, MovementCommand, parse_commands

# MovementCommand("x", 50)


class Brain:
    def __init__(self, mutation_chance: float):
        starting_action_count = 30
        self.options: list[str] = list(DIRECTION_KEYS)
        self.options.remove("x")
        self.options.remove("neutral")
        self.actions: list[str] = ["x", "neutral"]
        self.actions = [self.get_random_move() for _ in range(starting_action_count)]

        self.mutation_chance: float = mutation_chance

    def mutate_actions(self):
        for action in range(len(self.actions)):
            if self.mutation_chance < random.random():
                continue
            self.actions[action] = self.get_random_move()
        # if self.mutation_chance > random.random():
        self.add_random_move()
        if self.mutation_chance < random.random():
            self.add_random_move()
        if self.mutation_chance < random.random():
            self.add_random_move()
        if self.mutation_chance < random.random():
            self.add_random_move()

    def parse_actions(self) -> list[MovementCommand]:
        raw_commands = [
            {"move": "x", "duration_ms": 50},
            {"move": "neutral", "duration_ms": 300},
            {"move": "up", "duration_ms": 100},
            {"move": "down", "duration_ms": 100},
            {"move": "neutral", "duration_ms": 3000},
        ]

        for direction in self.actions:
            raw_commands.append(
                {
                    "move": direction,
                    "duration_ms": 300,
                }
            )

        return parse_commands(raw_commands)

    def add_random_move(self):
        self.actions.append(self.get_random_move())

    def get_random_move(self) -> str:
        return random.choice(self.options)
