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
ProgressReporter = Callable[[str], None]


@dataclass(frozen=True, slots=True)
class GenerationEvaluation:
    generation: int
    seeds: tuple[int, int, int]
    mean_survival_frames: dict[int, float]
    traces: dict[int, tuple[EpisodeTrace, ...]]
    best_genome_id: int | None
    best_fitness: float | None
    network_summary: str


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
        progress: ProgressReporter | None = None,
    ) -> None:
        self.step_frames = step_frames
        self.enemy_slots = enemy_slots
        self.aoe_slots = aoe_slots
        self.history_directory = history_directory
        self._environment_factory = environment_factory
        self._network_factory = network_factory or _neat_network
        self._seed_bank_factory = seed_bank_factory or fresh_seed_bank
        self._progress = progress
        self.generation = 0
        self.last_generation: GenerationEvaluation | None = None

    def __call__(self, genomes: Iterable[tuple[int, object]], config: object) -> None:
        genome_items = tuple(genomes)
        generation = self.generation + 1
        seeds = self._seed_bank_factory()
        self._report(
            f"generation {generation}: evaluating {len(genome_items)} genomes "
            f"on seeds {list(seeds)}"
        )
        results: dict[int, float] = {}
        traces: dict[int, tuple[EpisodeTrace, ...]] = {}
        for completed, (genome_id, genome) in enumerate(genome_items, start=1):
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
            survival = ", ".join(
                str(trace.result.survival_frames) for trace in genome_traces
            )
            self._report(
                f"generation {generation}: genome {completed}/{len(genome_items)} "
                f"id={genome_id} mean={fitness:.1f} survival=[{survival}]"
            )
        self.generation += 1
        best_genome_id, best_genome = max(
            genome_items,
            key=lambda item: results[item[0]],
            default=(None, None),
        )
        best_fitness = (
            results.get(best_genome_id) if best_genome_id is not None else None
        )
        summary = compact_network_summary(
            best_genome,
            config,
            enemy_slots=self.enemy_slots,
            aoe_slots=self.aoe_slots,
        )
        if best_fitness is None:
            self._report(f"generation {generation} complete: no genomes")
        else:
            self._report(
                f"generation {generation} complete: best id={best_genome_id} "
                f"mean={best_fitness:.1f} {summary}"
            )
        self.last_generation = GenerationEvaluation(
            generation=generation,
            seeds=seeds,
            mean_survival_frames=results,
            traces=traces,
            best_genome_id=best_genome_id,
            best_fitness=best_fitness,
            network_summary=summary,
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

    def _report(self, message: str) -> None:
        if self._progress is not None:
            self._progress(message)


def action_from_outputs(outputs: Sequence[float]) -> Direction:
    if len(outputs) != len(ACTIONS):
        raise ValueError(f"Dodge network must have exactly {len(ACTIONS)} outputs")
    return ACTIONS[max(range(len(outputs)), key=outputs.__getitem__)]


def fresh_seed_bank() -> tuple[int, int, int]:
    seeds: set[int] = set()
    while len(seeds) < SEED_BANK_SIZE:
        seeds.add(secrets.randbelow(32_768))
    return tuple(sorted(seeds))  # stable saved ordering, fresh entropy every generation


def compact_network_summary(
    genome: object | None,
    config: object,
    *,
    enemy_slots: int = 16,
    aoe_slots: int = 8,
    edge_limit: int = 6,
) -> str:
    genome_config = getattr(config, "genome_config", None)
    nodes = getattr(genome, "nodes", None)
    connections = getattr(genome, "connections", None)
    input_keys = tuple(getattr(genome_config, "input_keys", ()))
    output_keys = tuple(getattr(genome_config, "output_keys", ()))
    if not isinstance(nodes, dict) or not isinstance(connections, dict):
        return "network=unavailable"

    hidden_count = len(set(nodes).difference(output_keys))
    edge_rows = [
        (key, connection, float(getattr(connection, "weight", 0.0)))
        for key, connection in connections.items()
        if getattr(connection, "enabled", False)
        and isinstance(key, tuple)
        and len(key) == 2
    ]
    strongest = sorted(edge_rows, key=lambda row: abs(row[2]), reverse=True)[
        :edge_limit
    ]
    edges = ", ".join(
        f"{_node_label(source, input_keys, output_keys, enemy_slots, aoe_slots)}"
        f"→{_node_label(target, input_keys, output_keys, enemy_slots, aoe_slots)}"
        f":{weight:+.2f}"
        for (source, target), _connection, weight in strongest
    )
    return (
        f"network={len(input_keys)}→{hidden_count}→{len(output_keys)} "
        f"edges={len(edge_rows)}/{len(connections)} top=[{edges}]"
    )


def _node_label(
    key: object,
    input_keys: tuple[object, ...],
    output_keys: tuple[object, ...],
    enemy_slots: int,
    aoe_slots: int,
) -> str:
    if key in output_keys:
        index = output_keys.index(key)
        return ACTIONS[index] if index < len(ACTIONS) else f"out{index}"
    if key not in input_keys:
        return f"h{key}"
    index = input_keys.index(key)
    player_features = ("player.x", "player.y", "player.vx", "player.vy", "player.size")
    entity_features = ("present", "dx", "dy", "vx", "vy", "width", "height", "stage")
    if index < len(player_features):
        return player_features[index]
    entity_index = index - len(player_features)
    enemy_features = enemy_slots * len(entity_features)
    if entity_index < enemy_features:
        slot, feature = divmod(entity_index, len(entity_features))
        return f"enemy{slot + 1}.{entity_features[feature]}"
    aoe_index = entity_index - enemy_features
    if aoe_index < aoe_slots * len(entity_features):
        slot, feature = divmod(aoe_index, len(entity_features))
        return f"aoe{slot + 1}.{entity_features[feature]}"
    return f"input{index}"


def _neat_network(genome: object, config: object) -> Network:
    import neat

    return neat.nn.FeedForwardNetwork.create(genome, config)
