from __future__ import annotations

import secrets
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from dodge.neat.bridge import Direction
from dodge.neat.environment import DodgeEnv, EpisodeTrace

ACTIONS: tuple[Direction, ...] = (
    "neutral",
    "left",
    "right",
    "up",
    "down",
    "up_left",
    "up_right",
    "down_left",
    "down_right",
)
SEED_BANK_SIZE = 3


class Network(Protocol):
    def activate(self, inputs: Sequence[float]) -> Sequence[float]: ...


NetworkFactory = Callable[[object, object], Network]
EnvironmentFactory = Callable[..., DodgeEnv]
SeedBankFactory = Callable[[], tuple[int, int, int]]


@dataclass(frozen=True, slots=True)
class GenerationEvaluation:
    generation: int
    seeds: tuple[int, int, int]
    mean_survival_frames: dict[int, float]
    traces: dict[int, tuple[EpisodeTrace, ...]]


class DodgeEvaluator:
    """NEAT callback which evaluates every genome against the same fresh seeds."""

    def __init__(
        self,
        *,
        step_frames: int = 4,
        enemy_slots: int = 16,
        aoe_slots: int = 8,
        history_directory: Path | None = None,
        environment_factory: EnvironmentFactory = DodgeEnv,
        network_factory: NetworkFactory | None = None,
        seed_bank_factory: SeedBankFactory | None = None,
    ) -> None:
        self.step_frames = step_frames
        self.enemy_slots = enemy_slots
        self.aoe_slots = aoe_slots
        self.history_directory = history_directory
        self._environment_factory = environment_factory
        self._network_factory = network_factory or _neat_network
        self._seed_bank_factory = seed_bank_factory or fresh_seed_bank
        self.generation = 0
        self.last_generation: GenerationEvaluation | None = None

    def __call__(self, genomes: Iterable[tuple[int, object]], config: object) -> None:
        seeds = self._seed_bank_factory()
        results: dict[int, float] = {}
        traces: dict[int, tuple[EpisodeTrace, ...]] = {}
        for genome_id, genome in genomes:
            network = self._network_factory(genome, config)
            genome_traces = tuple(
                self._evaluate_episode(network, seed) for seed in seeds
            )
            fitness = (
                sum(trace.result.survival_frames for trace in genome_traces)
                / SEED_BANK_SIZE
            )
            genome.fitness = fitness
            results[genome_id] = fitness
            traces[genome_id] = genome_traces
        self.generation += 1
        self.last_generation = GenerationEvaluation(
            generation=self.generation,
            seeds=seeds,
            mean_survival_frames=results,
            traces=traces,
        )

    def _evaluate_episode(self, network: Network, seed: int) -> EpisodeTrace:
        environment = self._environment_factory(
            step_frames=self.step_frames,
            enemy_slots=self.enemy_slots,
            aoe_slots=self.aoe_slots,
        )
        try:
            observation = environment.reset(seed=seed)
            while True:
                action = action_from_outputs(
                    network.activate(observation.projected.values)
                )
                transition = environment.step(action)
                observation = transition.observation
                if transition.done:
                    break
            trace = environment.episode_trace
            if self.history_directory is not None:
                generation_directory = self.history_directory / (
                    f"generation-{self.generation + 1:04d}"
                )
                environment.save_episode(generation_directory)
            return trace
        finally:
            environment.close()


def action_from_outputs(outputs: Sequence[float]) -> Direction:
    if len(outputs) != len(ACTIONS):
        raise ValueError(f"Dodge network must have exactly {len(ACTIONS)} outputs")
    return ACTIONS[max(range(len(outputs)), key=outputs.__getitem__)]


def fresh_seed_bank() -> tuple[int, int, int]:
    seeds: set[int] = set()
    while len(seeds) < SEED_BANK_SIZE:
        seeds.add(secrets.randbelow(32_768))
    return tuple(sorted(seeds))  # stable saved ordering, fresh entropy every generation


def _neat_network(genome: object, config: object) -> Network:
    import neat

    return neat.nn.FeedForwardNetwork.create(genome, config)
