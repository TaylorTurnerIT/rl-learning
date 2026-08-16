from dataclasses import dataclass


@dataclass
class Fitness:
    win: bool
    distance_to_win: int
    move_count: int
