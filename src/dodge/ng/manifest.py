from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from dodge.control import PROJECT_ROOT

NG_MANIFEST_SCHEMA_VERSION = 1
NG_MANIFEST_ID = "dodge-ng-v1"
NATIVE_SEED_MIN = 0
NATIVE_SEED_MAX = 32_767
LEGACY_SEED_MAX = 30_010
DEFAULT_SEED_START = 30_100
DEFAULT_SEED_COUNT = 100
DEFAULT_SPLIT_SEED = 0xD06E
DEFAULT_MANIFEST_PATH = PROJECT_ROOT / "context" / "kits" / "dodge-ng" / "ng-v1.json"


@dataclass(frozen=True, slots=True)
class SeedManifest:
    """An immutable, hashed sample-space and train/holdout partition."""

    manifest_id: str
    schema_version: int
    split_seed: int
    sample_space: tuple[int, ...]
    training_seeds: tuple[int, ...]
    holdout_seeds: tuple[int, ...]
    legacy_seed_max: int = LEGACY_SEED_MAX

    @classmethod
    def fresh_default(
        cls,
        *,
        seed_start: int = DEFAULT_SEED_START,
        seed_count: int = DEFAULT_SEED_COUNT,
        split_seed: int = DEFAULT_SPLIT_SEED,
    ) -> SeedManifest:
        if seed_count < 1:
            raise ValueError("seed count must be positive")
        sample_space = list(range(seed_start, seed_start + seed_count))
        random_source = random.Random(split_seed)
        random_source.shuffle(sample_space)
        training_count = seed_count * 7 // 10
        return cls(
            manifest_id=NG_MANIFEST_ID,
            schema_version=NG_MANIFEST_SCHEMA_VERSION,
            split_seed=split_seed,
            sample_space=tuple(sample_space),
            training_seeds=tuple(sample_space[:training_count]),
            holdout_seeds=tuple(sample_space[training_count:]),
        )

    @property
    def sample_count(self) -> int:
        return len(self.sample_space)

    @property
    def training_fraction(self) -> float:
        return len(self.training_seeds) / self.sample_count

    @property
    def holdout_fraction(self) -> float:
        return len(self.holdout_seeds) / self.sample_count

    @property
    def sha256(self) -> str:
        return hashlib.sha256(_canonical_json(self._body()).encode()).hexdigest()

    def validate(self) -> None:
        if self.manifest_id != NG_MANIFEST_ID:
            raise ValueError(f"unsupported manifest id: {self.manifest_id}")
        if self.schema_version != NG_MANIFEST_SCHEMA_VERSION:
            raise ValueError(f"unsupported manifest schema: {self.schema_version}")
        if self.legacy_seed_max != LEGACY_SEED_MAX:
            raise ValueError("manifest legacy boundary does not match this build")
        if not self.sample_space or len(self.sample_space) % 10:
            raise ValueError("sample space must be nonempty and divisible by ten")
        if not isinstance(self.split_seed, int) or isinstance(self.split_seed, bool):
            raise ValueError("split seed must be an integer")
        _validate_seed_values(self.sample_space, "sample space")
        _validate_seed_values(self.training_seeds, "training seeds")
        _validate_seed_values(self.holdout_seeds, "holdout seeds")
        sample_set = set(self.sample_space)
        training_set = set(self.training_seeds)
        holdout_set = set(self.holdout_seeds)
        if len(training_set) != len(self.training_seeds):
            raise ValueError("training seeds must be unique")
        if len(holdout_set) != len(self.holdout_seeds):
            raise ValueError("holdout seeds must be unique")
        if training_set & holdout_set:
            raise ValueError("training and holdout seeds must be disjoint")
        if training_set | holdout_set != sample_set:
            raise ValueError("training and holdout seeds must cover the sample space")
        expected_training_count = len(self.sample_space) * 7 // 10
        if len(self.training_seeds) != expected_training_count:
            raise ValueError("training partition must contain exactly 70% of seeds")
        if len(self.holdout_seeds) != len(self.sample_space) * 3 // 10:
            raise ValueError("holdout partition must contain exactly 30% of seeds")
        if any(seed <= self.legacy_seed_max for seed in self.sample_space):
            raise ValueError(
                "NG sample space overlaps the legacy seed boundary "
                f"0..{self.legacy_seed_max}"
            )

    def _body(self) -> dict[str, object]:
        return {
            "manifest_id": self.manifest_id,
            "schema_version": self.schema_version,
            "split_seed": self.split_seed,
            "legacy_seed_max": self.legacy_seed_max,
            "sample_space": list(self.sample_space),
            "training_seeds": list(self.training_seeds),
            "holdout_seeds": list(self.holdout_seeds),
        }

    def to_json(self) -> dict[str, object]:
        self.validate()
        return {**self._body(), "manifest_sha256": self.sha256}

    @classmethod
    def from_json(cls, value: Mapping[str, object]) -> SeedManifest:
        try:
            manifest = cls(
                manifest_id=_string_field(value, "manifest_id"),
                schema_version=_int_field(value, "schema_version"),
                split_seed=_int_field(value, "split_seed"),
                legacy_seed_max=_int_field(value, "legacy_seed_max"),
                sample_space=_integer_tuple(value, "sample_space"),
                training_seeds=_integer_tuple(value, "training_seeds"),
                holdout_seeds=_integer_tuple(value, "holdout_seeds"),
            )
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"invalid NG seed manifest: {error}") from error
        manifest.validate()
        expected_hash = value.get("manifest_sha256")
        if not isinstance(expected_hash, str) or expected_hash != manifest.sha256:
            raise ValueError("NG seed manifest hash is missing or invalid")
        return manifest


def load_manifest(path: Path) -> SeedManifest:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"could not read NG seed manifest {path}: {error}") from error
    if not isinstance(value, Mapping):
        raise ValueError("NG seed manifest must contain a JSON object")
    return SeedManifest.from_json(value)


def save_manifest(path: Path, manifest: SeedManifest, *, replace: bool = False) -> None:
    manifest.validate()
    if path.exists() and not replace:
        raise ValueError(f"manifest already exists; pass --replace: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        temporary.write_text(
            json.dumps(manifest.to_json(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temporary.replace(path)
    except OSError as error:
        raise ValueError(f"could not write NG seed manifest {path}: {error}") from error
    finally:
        temporary.unlink(missing_ok=True)


def _validate_seed_values(values: Sequence[int], label: str) -> None:
    for seed in values:
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise ValueError(f"{label} must contain only integers")
        if not NATIVE_SEED_MIN <= seed <= NATIVE_SEED_MAX:
            raise ValueError(
                f"{label} seed {seed} is outside native range "
                f"{NATIVE_SEED_MIN}..{NATIVE_SEED_MAX}"
            )


def _string_field(value: Mapping[str, object], key: str) -> str:
    field = value[key]
    if not isinstance(field, str):
        raise TypeError(f"{key} must be a string")
    return field


def _int_field(value: Mapping[str, object], key: str) -> int:
    field = value[key]
    if isinstance(field, bool) or not isinstance(field, int):
        raise TypeError(f"{key} must be an integer")
    return field


def _integer_tuple(value: Mapping[str, object], key: str) -> tuple[int, ...]:
    field = value[key]
    if isinstance(field, (str, bytes)) or not isinstance(field, Sequence):
        raise TypeError(f"{key} must be an integer list")
    result = tuple(field)
    if any(isinstance(item, bool) or not isinstance(item, int) for item in result):
        raise TypeError(f"{key} must be an integer list")
    return result


def _canonical_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="dodge-ng-manifest")
    parser.add_argument("--output", type=Path, default=DEFAULT_MANIFEST_PATH)
    parser.add_argument("--seed-start", type=int, default=DEFAULT_SEED_START)
    parser.add_argument("--seed-count", type=int, default=DEFAULT_SEED_COUNT)
    parser.add_argument("--split-seed", type=int, default=DEFAULT_SPLIT_SEED)
    parser.add_argument("--replace", action="store_true")
    arguments = parser.parse_args(argv)
    try:
        manifest = SeedManifest.fresh_default(
            seed_start=arguments.seed_start,
            seed_count=arguments.seed_count,
            split_seed=arguments.split_seed,
        )
        save_manifest(arguments.output, manifest, replace=arguments.replace)
    except (OSError, ValueError) as error:
        print(f"dodge-ng-manifest: {error}", file=sys.stderr)
        return 1
    print(
        json.dumps(
            {
                "manifest": str(arguments.output),
                "manifest_sha256": manifest.sha256,
                "sample_count": manifest.sample_count,
                "training_count": len(manifest.training_seeds),
                "holdout_count": len(manifest.holdout_seeds),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
