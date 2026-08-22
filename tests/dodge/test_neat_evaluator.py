from __future__ import annotations

from dataclasses import dataclass

import pytest

from dodge.neat.environment import (
    EpisodeResult,
    EpisodeTrace,
    Observation,
    Transition,
)
from dodge.neat.evaluator import DodgeEvaluator, action_from_outputs
from dodge.neat.state import ProjectedObservation, parse_raw_state


def _observation() -> Observation:
    raw = parse_raw_state("__x__0|0,0,0,0,4||", prefix="__x__")
    return Observation(raw, ProjectedObservation((0.0,) * 197, False, False))


class FakeEnvironment:
    seeds: list[int] = []
    actions: list[str] = []

    def __init__(self, **_: object) -> None:
        self.seed = 0

    def reset(self, seed: int) -> Observation:
        self.seed = seed
        self.seeds.append(seed)
        return _observation()

    def step(self, action: str) -> Transition:
        self.actions.append(action)
        result = EpisodeResult(0, 10, self.seed, self.seed)
        self.episode_trace = EpisodeTrace(
            seed=self.seed,
            step_frames=4,
            enemy_slots=16,
            aoe_slots=8,
            actions=(action,),
            result=result,
            max_visible_enemies=1,
            max_visible_aoes=0,
            enemy_overflow_frames=0,
            aoe_overflow_frames=0,
        )
        return Transition(_observation(), float(self.seed), True, result)

    def close(self) -> None:
        pass


class FakeNetwork:
    def activate(self, _: tuple[float, ...]) -> tuple[float, ...]:
        return (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0)


@dataclass
class Genome:
    fitness: float | None = None


def test_evaluator_gives_all_genomes_same_three_seed_bank_then_rotates() -> None:
    FakeEnvironment.seeds.clear()
    FakeEnvironment.actions.clear()
    banks = iter(((11, 12, 13), (21, 22, 23)))
    evaluator = DodgeEvaluator(
        environment_factory=FakeEnvironment,  # type: ignore[arg-type]
        network_factory=lambda _genome, _config: FakeNetwork(),
        seed_bank_factory=lambda: next(banks),
    )
    first = [Genome(), Genome()]
    second = [Genome(), Genome()]

    evaluator(enumerate(first), object())
    first_generation = evaluator.last_generation
    evaluator(enumerate(second), object())

    assert FakeEnvironment.seeds == [11, 12, 13, 11, 12, 13, 21, 22, 23, 21, 22, 23]
    assert [genome.fitness for genome in first] == [12, 12]
    assert [genome.fitness for genome in second] == [22, 22]
    assert first_generation is not None
    assert first_generation.seeds == (11, 12, 13)
    assert evaluator.last_generation is not None
    assert evaluator.last_generation.seeds == (21, 22, 23)
    assert FakeEnvironment.actions == ["up_right"] * 12


def test_action_from_outputs_requires_nine_actions() -> None:
    assert action_from_outputs((0.0,) * 8 + (1.0,)) == "down_right"
    with pytest.raises(ValueError, match="exactly 9"):
        action_from_outputs((0.0,))
