from __future__ import annotations

import json
import secrets
from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import cast

from dodge.control import PROJECT_ROOT, ControlInputError, parse_seed
from dodge.neat.bridge import ACTION_KEYS, BridgeResult, Direction, PemsaStepBridge
from dodge.neat.state import ProjectedObservation, RawState, project_state

NEAT_HISTORY_DIRECTORY = PROJECT_ROOT / "history" / "dodge" / "neat"
EPISODE_VERSION = 1


@dataclass(frozen=True, slots=True)
class Observation:
    raw_state: RawState
    projected: ProjectedObservation

    def to_json(self) -> dict[str, object]:
        return {
            "raw_state": self.raw_state.to_json(),
            "projection": list(self.projected.values),
            "enemy_overflow": self.projected.enemy_overflow,
            "aoe_overflow": self.projected.aoe_overflow,
        }


@dataclass(frozen=True, slots=True)
class EpisodeResult:
    score: float
    frames: int
    survival_frames: int
    seed: int
    died: bool = True


@dataclass(frozen=True, slots=True)
class Transition:
    observation: Observation
    reward: float
    done: bool
    result: EpisodeResult | None


@dataclass(frozen=True, slots=True)
class EpisodeTrace:
    seed: int
    step_frames: int
    enemy_slots: int
    aoe_slots: int
    actions: tuple[Direction, ...]
    result: EpisodeResult
    max_visible_enemies: int
    max_visible_aoes: int
    enemy_overflow_frames: int
    aoe_overflow_frames: int

    def to_json(self) -> dict[str, object]:
        return {
            "version": EPISODE_VERSION,
            "kind": "neat_episode",
            "seed": self.seed,
            "config": {
                "step_frames": self.step_frames,
                "enemy_slots": self.enemy_slots,
                "aoe_slots": self.aoe_slots,
            },
            "actions": list(self.actions),
            "result": asdict(self.result),
            "telemetry": {
                "max_visible_enemies": self.max_visible_enemies,
                "max_visible_aoes": self.max_visible_aoes,
                "enemy_overflow_frames": self.enemy_overflow_frames,
                "aoe_overflow_frames": self.aoe_overflow_frames,
            },
        }


BridgeFactory = Callable[..., PemsaStepBridge]


class DodgeEnv:
    """One disposable, deterministic Dodge episode per `reset()` call."""

    def __init__(
        self,
        *,
        step_frames: int = 4,
        enemy_slots: int = 16,
        aoe_slots: int = 8,
        bridge_factory: BridgeFactory = PemsaStepBridge,
    ) -> None:
        if not 3 <= step_frames <= 5:
            raise ValueError("step_frames must be between 3 and 5")
        if enemy_slots < 1 or aoe_slots < 1:
            raise ValueError("observation slot counts must be positive")
        self.step_frames = step_frames
        self.enemy_slots = enemy_slots
        self.aoe_slots = aoe_slots
        self._bridge_factory = bridge_factory
        self._bridge: PemsaStepBridge | None = None
        self._seed: int | None = None
        self._actions: list[Direction] = []
        self._terminated = False
        self._last_frame = 0
        self._last_survival_frames = 0
        self._trace: EpisodeTrace | None = None

    def reset(self, seed: int | None = None) -> Observation:
        self.close()
        selected_seed = secrets.randbelow(32_768) if seed is None else _seed(seed)
        bridge = self._bridge_factory(
            seed=selected_seed,
            step_frames=self.step_frames,
            enemy_slots=self.enemy_slots,
            aoe_slots=self.aoe_slots,
        )
        try:
            raw_state = bridge.start()
        except Exception:
            bridge.close()
            raise
        self._bridge = bridge
        self._seed = selected_seed
        self._actions = []
        self._terminated = False
        self._last_frame = raw_state.frame
        self._last_survival_frames = 0
        self._trace = None
        return self._observe(raw_state)

    def step(self, action: Direction) -> Transition:
        if self._bridge is None:
            raise RuntimeError("call reset() before step()")
        if self._terminated:
            raise RuntimeError("episode is complete; call reset() before step()")
        if action not in ACTION_KEYS:
            raise ControlInputError(f"unsupported Dodge action: {action!r}")

        update = self._bridge.step(action)
        self._actions.append(action)
        if isinstance(update, BridgeResult):
            observation = self._observe(update.state)
            result = EpisodeResult(
                score=update.score,
                frames=update.frames,
                survival_frames=update.survival_frames,
                seed=update.seed,
            )
            self._terminated = True
            self._trace = EpisodeTrace(
                seed=update.seed,
                step_frames=self.step_frames,
                enemy_slots=self.enemy_slots,
                aoe_slots=self.aoe_slots,
                actions=tuple(self._actions),
                result=result,
                max_visible_enemies=update.max_visible_enemies,
                max_visible_aoes=update.max_visible_aoes,
                enemy_overflow_frames=update.enemy_overflow_frames,
                aoe_overflow_frames=update.aoe_overflow_frames,
            )
            reward = float(update.survival_frames - self._last_survival_frames)
            self._last_survival_frames = update.survival_frames
            self._last_frame = update.frames
            return Transition(observation, reward, True, result)

        observation = self._observe(update)
        reward = float(update.frame - self._last_frame)
        self._last_frame = update.frame
        self._last_survival_frames += int(reward)
        return Transition(observation, reward, False, None)

    @property
    def episode_trace(self) -> EpisodeTrace:
        if self._trace is None:
            raise RuntimeError("episode history is available only after death")
        return self._trace

    def save_episode(
        self,
        directory: Path = NEAT_HISTORY_DIRECTORY,
        *,
        created_at: datetime | None = None,
        filename: str | None = None,
    ) -> Path:
        return save_episode_trace(
            self.episode_trace,
            directory,
            created_at=created_at,
            filename=filename,
        )

    def close(self) -> None:
        if self._bridge is not None:
            self._bridge.close()
            self._bridge = None

    def __enter__(self) -> DodgeEnv:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def _observe(self, raw_state: RawState) -> Observation:
        return Observation(
            raw_state=raw_state,
            projected=project_state(
                raw_state,
                enemy_slots=self.enemy_slots,
                aoe_slots=self.aoe_slots,
            ),
        )


def load_episode(path: Path) -> EpisodeTrace:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ControlInputError(f"could not read NEAT episode: {error}") from error
    if not isinstance(value, dict):
        raise ControlInputError("NEAT episode must be an object")
    try:
        if value["version"] != EPISODE_VERSION or value["kind"] != "neat_episode":
            raise ControlInputError("unsupported NEAT episode format")
        config = _object(value["config"], "config")
        result = _object(value["result"], "result")
        telemetry = _object(value["telemetry"], "telemetry")
        actions_value = value["actions"]
        if not isinstance(actions_value, list):
            raise ControlInputError("actions must be a list")
        actions = tuple(_action(action) for action in actions_value)
        trace = EpisodeTrace(
            seed=_seed(value["seed"]),
            step_frames=_step_frames(config["step_frames"]),
            enemy_slots=_positive(config["enemy_slots"], "enemy_slots"),
            aoe_slots=_positive(config["aoe_slots"], "aoe_slots"),
            actions=actions,
            result=EpisodeResult(
                score=_number(result["score"], "result.score"),
                frames=_nonnegative(result["frames"], "result.frames"),
                survival_frames=_nonnegative(
                    result["survival_frames"], "result.survival_frames"
                ),
                seed=_seed(result["seed"]),
                died=result["died"] is True,
            ),
            max_visible_enemies=_nonnegative(
                telemetry["max_visible_enemies"], "telemetry.max_visible_enemies"
            ),
            max_visible_aoes=_nonnegative(
                telemetry["max_visible_aoes"], "telemetry.max_visible_aoes"
            ),
            enemy_overflow_frames=_nonnegative(
                telemetry["enemy_overflow_frames"],
                "telemetry.enemy_overflow_frames",
            ),
            aoe_overflow_frames=_nonnegative(
                telemetry["aoe_overflow_frames"], "telemetry.aoe_overflow_frames"
            ),
        )
    except (KeyError, ControlInputError) as error:
        raise ControlInputError(f"invalid NEAT episode: {error}") from error
    if trace.result.seed != trace.seed or not trace.result.died:
        raise ControlInputError("NEAT episode has inconsistent terminal result")
    return trace


def save_episode_trace(
    trace: EpisodeTrace,
    directory: Path = NEAT_HISTORY_DIRECTORY,
    *,
    created_at: datetime | None = None,
    filename: str | None = None,
) -> Path:
    timestamp = created_at or datetime.now(UTC)
    directory.mkdir(parents=True, exist_ok=True)
    name = filename or f"episode-{timestamp.strftime('%Y%m%dT%H%M%S.%fZ')}.json"
    path = directory / name
    if path.exists():
        raise FileExistsError(f"NEAT episode history already exists: {path}")
    _write_json(path, trace.to_json())
    return path


def _write_json(path: Path, value: dict[str, object]) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _object(value: object, name: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise ControlInputError(f"{name} must be an object")
    return cast(dict[str, object], value)


def _action(value: object) -> Direction:
    if not isinstance(value, str) or value not in ACTION_KEYS:
        raise ControlInputError("actions contain an unsupported direction")
    return cast(Direction, value)


def _seed(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ControlInputError("seed must be an integer from 0 to 32767")
    return parse_seed(str(value))


def _step_frames(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not 3 <= value <= 5:
        raise ControlInputError("step_frames must be between 3 and 5")
    return value


def _positive(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ControlInputError(f"{name} must be a positive integer")
    return value


def _nonnegative(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ControlInputError(f"{name} must be a non-negative integer")
    return value


def _number(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ControlInputError(f"{name} must be numeric")
    return float(value)
