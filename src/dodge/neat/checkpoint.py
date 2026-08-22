from __future__ import annotations

import gzip
import os
import pickle
import random
import tempfile
from collections.abc import Callable
from pathlib import Path

from neat.reporting import BaseReporter

CHECKPOINT_RETENTION = 5
CHECKPOINT_PREFIX = "checkpoint-"
CheckpointSaved = Callable[[int, Path], None]


def _discard_checkpoint_callback(_generation: int, _path: Path) -> None:
    """Placeholder used only inside a restored NEAT checkpoint."""


class RunCheckpointer(BaseReporter):
    """Save the next NEAT generation state after every completed generation."""

    def __init__(
        self,
        directory: Path,
        *,
        on_saved: CheckpointSaved,
        retention: int = CHECKPOINT_RETENTION,
    ) -> None:
        if retention < 1:
            raise ValueError("checkpoint retention must be positive")
        self.directory = directory
        self.on_saved = on_saved
        self.retention = retention
        self._generation: int | None = None

    def __getstate__(self) -> dict[str, object]:
        state = self.__dict__.copy()
        state["on_saved"] = _discard_checkpoint_callback
        return state

    def start_generation(self, generation: int) -> None:
        self._generation = generation

    def end_generation(
        self,
        config: object,
        population: object,
        species_set: object,
    ) -> None:
        if self._generation is None:
            raise RuntimeError("checkpoint reporter did not receive a generation")
        generation = self._generation + 1
        path = self._save(config, population, species_set, generation)
        self._prune()
        self.on_saved(generation, path)

    def _save(
        self,
        config: object,
        population: object,
        species_set: object,
        generation: int,
    ) -> Path:
        self.directory.mkdir(parents=True, exist_ok=True)
        path = checkpoint_path(self.directory, generation)
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{path.name}.", suffix=".tmp", dir=self.directory
        )
        os.close(descriptor)
        temporary = Path(temporary_name)
        try:
            with gzip.open(temporary, "wb", compresslevel=5) as output:
                pickle.dump(
                    (generation, config, population, species_set, random.getstate()),
                    output,
                    protocol=pickle.HIGHEST_PROTOCOL,
                )
            os.replace(temporary, path)
        except Exception:
            temporary.unlink(missing_ok=True)
            raise
        return path

    def _prune(self) -> None:
        paths = checkpoint_paths(self.directory)
        for path in paths[: -self.retention]:
            path.unlink()


def checkpoint_path(directory: Path, generation: int) -> Path:
    return directory / f"{CHECKPOINT_PREFIX}{generation:06d}.gz"


def checkpoint_paths(directory: Path) -> list[Path]:
    return sorted(directory.glob(f"{CHECKPOINT_PREFIX}*.gz"))


def latest_checkpoint(directory: Path) -> Path:
    paths = checkpoint_paths(directory)
    if not paths:
        raise FileNotFoundError(f"no NEAT checkpoint in {directory}")
    return paths[-1]
