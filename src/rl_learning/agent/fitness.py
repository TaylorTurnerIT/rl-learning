from dataclasses import dataclass
from typing import Self


@dataclass
class Fitness:
    win: bool
    distance_to_win: int
    move_count: int

    def __gt__(self, other: Self) -> bool:
        if self.win and not other.win:
            return True
        return self.distance_to_win > other.distance_to_win
