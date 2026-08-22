from __future__ import annotations

from types import SimpleNamespace

import neat

from dodge.control import PROJECT_ROOT
from dodge.neat.speciation import SpeciesMonitor, format_species_metrics


def test_v28_records_species_health_and_adapts_sparse_population() -> None:
    config = SimpleNamespace(
        genome_config=SimpleNamespace(output_keys=(0,)),
        species_set_config=SimpleNamespace(compatibility_threshold=0.6),
    )
    population = {
        1: SimpleNamespace(nodes={0: object()}),
        2: SimpleNamespace(nodes={0: object(), 4: object()}),
    }
    species_set = SimpleNamespace(species={1: SimpleNamespace(members=population)})
    monitor = SpeciesMonitor()

    monitor.start_generation(0)
    monitor.post_evaluate(config, population, species_set, population[1])
    monitor.end_generation(config, population, species_set)

    assert monitor.latest is not None
    assert monitor.latest.generation == 1
    assert monitor.latest.count == 1
    assert monitor.latest.sizes == (2,)
    assert monitor.latest.mean_hidden_nodes == 0.5
    assert monitor.latest.max_hidden_nodes == 1
    assert monitor.latest.compatibility_threshold == 0.6
    assert monitor.latest.next_compatibility_threshold == 0.54
    assert config.species_set_config.compatibility_threshold == 0.54
    assert format_species_metrics(monitor.latest) == (
        "species=1 sizes=[2] threshold=0.600→0.540 hidden=0.50/1"
    )


def test_v28_v3_initial_population_starts_with_viable_species_count() -> None:
    config = neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        PROJECT_ROOT / "src/dodge/neat/config-dodge-v3",
    )
    population = neat.Population(config, seed=42)

    sizes = sorted(
        (len(species.members) for species in population.species.species.values()),
        reverse=True,
    )

    assert 4 <= len(sizes) <= 8
    assert sum(sizes) == config.pop_size
