from __future__ import annotations

from dataclasses import asdict, dataclass

from neat.reporting import BaseReporter

TARGET_SPECIES = 6
SPECIES_TOLERANCE = 2
COMPATIBILITY_ADJUSTMENT = 0.9
MIN_COMPATIBILITY_THRESHOLD = 0.1
MAX_COMPATIBILITY_THRESHOLD = 5.0


@dataclass(frozen=True, slots=True)
class SpeciesMetrics:
    generation: int
    count: int
    sizes: tuple[int, ...]
    compatibility_threshold: float
    next_compatibility_threshold: float
    mean_hidden_nodes: float
    max_hidden_nodes: int

    def as_json(self) -> dict[str, object]:
        return asdict(self)


class SpeciesMonitor(BaseReporter):
    """Persist species health and steer compatibility toward viable niches."""

    def __init__(self) -> None:
        self._generation = 0
        self.history: list[SpeciesMetrics] = []

    @property
    def latest(self) -> SpeciesMetrics | None:
        return self.history[-1] if self.history else None

    def start_generation(self, generation: int) -> None:
        self._generation = generation + 1

    def post_evaluate(
        self,
        config: object,
        population: dict[int, object],
        species_set: object,
        best_genome: object,
    ) -> None:
        del best_genome
        species = getattr(species_set, "species", {})
        sizes = tuple(
            sorted((len(group.members) for group in species.values()), reverse=True)
        )
        output_keys = set(
            getattr(getattr(config, "genome_config", None), "output_keys", ())
        )
        hidden_counts = [
            len(set(getattr(genome, "nodes", {})).difference(output_keys))
            for genome in population.values()
        ]
        threshold = _compatibility_threshold(config)
        self.history.append(
            SpeciesMetrics(
                generation=self._generation,
                count=len(sizes),
                sizes=sizes,
                compatibility_threshold=threshold,
                next_compatibility_threshold=threshold,
                mean_hidden_nodes=(sum(hidden_counts) / len(hidden_counts))
                if hidden_counts
                else 0.0,
                max_hidden_nodes=max(hidden_counts, default=0),
            )
        )

    def end_generation(
        self,
        config: object,
        population: dict[int, object],
        species_set: object,
    ) -> None:
        del population
        metrics = self.latest
        if metrics is None or metrics.generation != self._generation:
            return
        species_count = len(getattr(species_set, "species", {}))
        next_threshold = _adjust_threshold(
            _compatibility_threshold(config), species_count
        )
        _set_compatibility_threshold(config, next_threshold)
        self.history[-1] = SpeciesMetrics(
            generation=metrics.generation,
            count=metrics.count,
            sizes=metrics.sizes,
            compatibility_threshold=metrics.compatibility_threshold,
            next_compatibility_threshold=next_threshold,
            mean_hidden_nodes=metrics.mean_hidden_nodes,
            max_hidden_nodes=metrics.max_hidden_nodes,
        )


def ensure_species_monitor(population: object) -> SpeciesMonitor:
    reporter_set = getattr(population, "reporters", None)
    reporters = getattr(
        reporter_set, "reporters", getattr(population, "_reporters", ())
    )
    for reporter in reporters:
        if isinstance(reporter, SpeciesMonitor):
            return reporter
    monitor = SpeciesMonitor()
    add_reporter = getattr(population, "add_reporter", None)
    if not callable(add_reporter):
        raise TypeError("NEAT population does not expose add_reporter")
    add_reporter(monitor)
    return monitor


def format_species_metrics(metrics: SpeciesMetrics | None) -> str:
    if metrics is None:
        return "species=unavailable"
    sizes = ",".join(str(size) for size in metrics.sizes)
    return (
        f"species={metrics.count} sizes=[{sizes}] "
        f"threshold={metrics.compatibility_threshold:.3f}"
        f"→{metrics.next_compatibility_threshold:.3f} "
        f"hidden={metrics.mean_hidden_nodes:.2f}/{metrics.max_hidden_nodes}"
    )


def _compatibility_threshold(config: object) -> float:
    species_config = getattr(config, "species_set_config", None)
    threshold = getattr(species_config, "compatibility_threshold", None)
    if isinstance(threshold, bool) or not isinstance(threshold, (int, float)):
        raise TypeError("NEAT config has no numeric compatibility threshold")
    return float(threshold)


def _set_compatibility_threshold(config: object, value: float) -> None:
    species_config = getattr(config, "species_set_config", None)
    if species_config is None:
        raise TypeError("NEAT config has no species configuration")
    species_config.compatibility_threshold = value


def _adjust_threshold(threshold: float, species_count: int) -> float:
    if species_count < TARGET_SPECIES - SPECIES_TOLERANCE:
        return max(MIN_COMPATIBILITY_THRESHOLD, threshold * COMPATIBILITY_ADJUSTMENT)
    if species_count > TARGET_SPECIES + SPECIES_TOLERANCE:
        return min(MAX_COMPATIBILITY_THRESHOLD, threshold / COMPATIBILITY_ADJUSTMENT)
    return threshold
