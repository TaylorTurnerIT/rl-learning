from __future__ import annotations

from concurrent.futures import Future, ProcessPoolExecutor
from dataclasses import dataclass
from multiprocessing import get_context
from pathlib import Path

import pytest

from dodge.control import PROJECT_ROOT, ControlRuntimeError
from dodge.neat.bridge import InputAcknowledgementTimeout
from dodge.neat.environment import (
    EpisodeResult,
    EpisodeTrace,
    Observation,
    Transition,
    load_episode,
)
from dodge.neat.evaluator import (
    BENCHMARK_GENERATIONS,
    DodgeEvaluator,
    GenomeEvaluationTask,
    SeedBankSchedule,
    _evaluate_genome_with_retries,
    action_from_outputs,
    compact_network_summary,
)
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
        self.episode_trace = _trace(self.seed, action)
        result = self.episode_trace.result
        return Transition(_observation(), float(self.seed), True, result)

    def close(self) -> None:
        pass


class FakeNetwork:
    def activate(self, _: tuple[float, ...]) -> tuple[float, ...]:
        return (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0)


class FlakyEnvironment(FakeEnvironment):
    failures_remaining = 1

    def step(self, action: str) -> Transition:
        if self.failures_remaining:
            type(self).failures_remaining -= 1
            raise ControlRuntimeError("transient bridge input timeout")
        return super().step(action)


@dataclass
class Genome:
    fitness: float | None = None


def _trace(seed: int, action: str = "up_right") -> EpisodeTrace:
    result = EpisodeResult(0, 10, seed, seed)
    return EpisodeTrace(
        seed=seed,
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


class ImmediateExecutor:
    def __init__(self, _: int) -> None:
        pass

    def __enter__(self) -> ImmediateExecutor:
        return self

    def __exit__(self, *_: object) -> None:
        pass

    def submit(
        self, _: object, task: GenomeEvaluationTask
    ) -> Future[tuple[int, tuple[EpisodeTrace, ...]]]:
        genome_id = task.genome_id
        seeds = task.seeds
        future: Future[tuple[int, tuple[EpisodeTrace, ...]]] = Future()
        future.set_result((genome_id, tuple(_trace(seed) for seed in seeds)))
        return future


def _worker_task_details(task: GenomeEvaluationTask) -> tuple[int, int, int]:
    return (
        task.genome_id,
        len(task.genome.connections),
        task.config.genome_config.num_inputs,
    )


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


def test_v7_seed_schedule_holds_training_bank_and_reports_unselected_validation() -> (
    None
):
    schedule = SeedBankSchedule(123)
    assert schedule.training_bank(1) == schedule.training_bank(5)
    assert schedule.training_bank(5) != schedule.training_bank(6)
    assert schedule.training_bank(1) != schedule.validation_bank(1)
    assert SeedBankSchedule(123).validation_bank(1) == schedule.validation_bank(1)

    FakeEnvironment.seeds.clear()
    evaluator = DodgeEvaluator(
        environment_factory=FakeEnvironment,  # type: ignore[arg-type]
        network_factory=lambda _genome, _config: FakeNetwork(),
        seed_bank_schedule=schedule,
    )
    genomes = [Genome(), Genome()]
    evaluator(enumerate(genomes), object())

    training = schedule.training_bank(1)
    validation = schedule.validation_bank(1)
    assert FakeEnvironment.seeds == [*training, *training, *validation]
    assert [genome.fitness for genome in genomes] == [sum(training) / 3] * 2
    assert evaluator.last_generation is not None
    assert evaluator.last_generation.validation_seeds == validation
    assert evaluator.last_generation.validation_fitness == sum(validation) / 3


def test_v29_fixed_benchmark_reports_same_held_out_bank_every_five_generations() -> (
    None
):
    FakeEnvironment.seeds.clear()
    schedule = SeedBankSchedule(123)
    evaluator = DodgeEvaluator(
        environment_factory=FakeEnvironment,  # type: ignore[arg-type]
        network_factory=lambda _genome, _config: FakeNetwork(),
        seed_bank_schedule=schedule,
    )
    genome = Genome()

    for _ in range(BENCHMARK_GENERATIONS):
        evaluator([(1, genome)], object())

    assert evaluator.last_generation is not None
    assert evaluator.last_generation.benchmark_seeds == schedule.benchmark_bank()
    assert evaluator.last_generation.benchmark_fitness == (
        sum(schedule.benchmark_bank()) / len(schedule.benchmark_bank())
    )
    assert set(schedule.benchmark_bank()).isdisjoint(evaluator.last_generation.seeds)


def test_action_from_outputs_requires_nine_actions() -> None:
    assert action_from_outputs((0.0,) * 8 + (1.0,)) == "down_right"
    with pytest.raises(ValueError, match="exactly 9"):
        action_from_outputs((0.0,))


def test_evaluator_reports_genome_progress_and_generation_best() -> None:
    messages: list[str] = []
    evaluator = DodgeEvaluator(
        environment_factory=FakeEnvironment,  # type: ignore[arg-type]
        network_factory=lambda _genome, _config: FakeNetwork(),
        seed_bank_factory=lambda: (11, 12, 13),
        progress=messages.append,
    )

    evaluator(enumerate([Genome(), Genome()]), object())

    assert (
        messages[0]
        == "generation 1: evaluating 2 genomes on seeds [11, 12, 13] with 1 worker(s)"
    )
    assert "genome 1/2" in messages[1]
    assert "genome 2/2" in messages[2]
    assert (
        "generation 1 complete: best id=0 mean=12.0 network=unavailable" in messages[3]
    )


def test_v21_retries_a_transient_episode_with_the_same_seed() -> None:
    FakeEnvironment.seeds.clear()
    FakeEnvironment.actions.clear()
    FlakyEnvironment.failures_remaining = 1
    evaluator = DodgeEvaluator(
        environment_factory=FlakyEnvironment,  # type: ignore[arg-type]
        network_factory=lambda _genome, _config: FakeNetwork(),
        seed_bank_factory=lambda: (11, 12, 13),
    )
    genome = Genome()

    evaluator([(0, genome)], object())

    assert FakeEnvironment.seeds == [11, 11, 12, 13]
    assert genome.fitness == 12


def test_v31_retries_three_transient_timeouts_with_same_seed() -> None:
    FakeEnvironment.seeds.clear()
    FakeEnvironment.actions.clear()
    FlakyEnvironment.failures_remaining = 2
    evaluator = DodgeEvaluator(
        environment_factory=FlakyEnvironment,  # type: ignore[arg-type]
        network_factory=lambda _genome, _config: FakeNetwork(),
        seed_bank_factory=lambda: (11, 12, 13),
    )
    genome = Genome()

    evaluator([(0, genome)], object())

    assert FakeEnvironment.seeds == [11, 11, 11, 12, 13]
    assert genome.fitness == 12


def test_v31_retries_whole_genome_after_worker_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = 0

    def evaluate(_network: object, seed: int, **_: object) -> EpisodeTrace:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise InputAcknowledgementTimeout("lost Pemsa acknowledgement")
        return _trace(seed)

    monkeypatch.setattr("dodge.neat.evaluator._evaluate_default_episode", evaluate)

    traces = _evaluate_genome_with_retries(
        FakeNetwork(),
        (11, 12, 13),
        step_frames=4,
        enemy_slots=16,
        aoe_slots=8,
        include_time_to_intersection=False,
    )

    assert [trace.seed for trace in traces] == [11, 12, 13]
    assert calls == 5


@dataclass
class Connection:
    weight: float
    enabled: bool = True


@dataclass
class Node:
    bias: float = 0


class SummaryGenome:
    nodes = {0: Node(), 1: Node(), 4: Node()}
    connections = {(-1, 0): Connection(2.5), (-2, 1): Connection(-1.25)}


class SummaryConfig:
    class genome_config:
        input_keys = (-1, -2)
        output_keys = (0, 1)


def test_compact_network_summary_describes_topology_and_strongest_edges() -> None:
    summary = compact_network_summary(SummaryGenome(), SummaryConfig())

    assert summary == (
        "network=2→1→2 edges=2/2 top=[player.x→neutral:+2.50, player.y→left:-1.25]"
    )


def test_parallel_evaluator_preserves_three_seed_fitness_and_parent_history(
    tmp_path: Path,
) -> None:
    evaluator = DodgeEvaluator(
        workers=2,
        history_directory=tmp_path,
        seed_bank_factory=lambda: (11, 12, 13),
        parallel_executor_factory=ImmediateExecutor,  # type: ignore[arg-type]
    )
    genomes = [SummaryGenome(), SummaryGenome()]

    evaluator(enumerate(genomes), SummaryConfig())

    assert [genome.fitness for genome in genomes] == [12, 12]
    assert evaluator.last_generation is not None
    assert evaluator.last_generation.seeds == (11, 12, 13)
    assert evaluator.last_generation.network_visualization == (
        tmp_path / "generation-0001/network.html"
    )
    paths = sorted((tmp_path / "generation-0001").glob("*.json"))
    assert [path.name for path in paths] == [
        "genome-0000-seed-11.json",
        "genome-0000-seed-12.json",
        "genome-0000-seed-13.json",
        "genome-0001-seed-11.json",
        "genome-0001-seed-12.json",
        "genome-0001-seed-13.json",
    ]
    assert load_episode(paths[0]).seed == 11
    assert (tmp_path / "generation-0001/network.html").is_file()


def test_parallel_evaluator_rejects_custom_unpicklable_factories() -> None:
    with pytest.raises(ValueError, match="workers must be positive"):
        DodgeEvaluator(workers=0)
    with pytest.raises(ValueError, match="default Dodge environment"):
        DodgeEvaluator(workers=2, environment_factory=FakeEnvironment)  # type: ignore[arg-type]


def test_parallel_real_neat_task_is_spawn_serializable() -> None:
    import neat

    config = neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        PROJECT_ROOT / "src/dodge/neat/config-dodge",
    )
    genome = next(iter(neat.Population(config).population.values()))
    task = GenomeEvaluationTask(
        genome_id=7,
        genome=genome,
        config=config,
        seeds=(11, 12, 13),
        step_frames=4,
        enemy_slots=16,
        aoe_slots=8,
    )
    with ProcessPoolExecutor(max_workers=1, mp_context=get_context("spawn")) as pool:
        assert pool.submit(_worker_task_details, task).result(timeout=10) == (
            7,
            1773,
            197,
        )
