from __future__ import annotations

import hashlib
import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path

from dodge.control import ControlInputError, MovementCommand
from dodge.dataset import ACTION_CHOICES, BOOTSTRAP, STEP_FRAMES

DEFAULT_DATABASE = Path("history/dodge/dataset.sqlite3")
_REQUIRED_TABLES = {"episodes", "steps"}
_BOOTSTRAP_ACTIONS = tuple(action for action, _ in BOOTSTRAP[1:])


@dataclass(frozen=True, slots=True)
class DatasetEpisode:
    """A validated GA episode and the complete schedule needed to replay it."""

    episode_id: int
    seed: int
    action_hash: str
    stored_result: dict[str, object]
    commands: tuple[MovementCommand, ...]
    genome_actions: tuple[str, ...]
    recorded_steps: int
    skipped_episode_ids: tuple[int, ...] = ()

    @property
    def survival_frames(self) -> int:
        value = self.stored_result.get("survival_frames")
        if not isinstance(value, int) or isinstance(value, bool):
            raise ControlInputError("dataset episode survival_frames is invalid")
        return value

    @property
    def total_frames(self) -> int:
        value = self.stored_result.get("frames")
        if not isinstance(value, int) or isinstance(value, bool):
            raise ControlInputError("dataset episode frames is invalid")
        return value


def load_longest_episode(
    database: Path = DEFAULT_DATABASE,
    *,
    episode_id: int | None = None,
) -> DatasetEpisode:
    """Load one GA episode from SQLite without changing the database.

    The default selection is the episode with the greatest recorded survival,
    then greatest total frame count, then lowest database id. The steps table
    stores the four post-start bootstrap actions followed by the genome; the
    explicit ``x`` bootstrap action is reconstructed from the game contract.
    """
    connection = _open_read_only(database)
    try:
        _validate_schema(connection)
        if episode_id is not None:
            row = _select_episode(connection, episode_id=episode_id)
            steps = _load_steps(connection, row[0])
            return _build_episode(row, steps)

        skipped: list[int] = []
        for row in _ordered_episode_rows(connection):
            try:
                episode = _build_episode(row, _load_steps(connection, row[0]))
            except ControlInputError:
                skipped.append(_int_value(row[0], "episode id"))
                continue
            return DatasetEpisode(
                episode_id=episode.episode_id,
                seed=episode.seed,
                action_hash=episode.action_hash,
                stored_result=episode.stored_result,
                commands=episode.commands,
                genome_actions=episode.genome_actions,
                recorded_steps=episode.recorded_steps,
                skipped_episode_ids=tuple(skipped),
            )
    finally:
        connection.close()

    raise ControlInputError("GA dataset contains no replayable episodes")


def _open_read_only(database: Path) -> sqlite3.Connection:
    if not database.is_file():
        raise ControlInputError(f"GA dataset database does not exist: {database}")
    try:
        connection = sqlite3.connect(
            f"{database.resolve().as_uri()}?mode=ro",
            uri=True,
        )
        connection.execute("PRAGMA query_only=ON")
        return connection
    except sqlite3.Error as error:
        raise ControlInputError(
            f"could not open GA dataset database read-only: {error}"
        ) from error


def _validate_schema(connection: sqlite3.Connection) -> None:
    tables = {
        row[0]
        for row in connection.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        )
    }
    missing = _REQUIRED_TABLES - tables
    if missing:
        names = ", ".join(sorted(missing))
        raise ControlInputError(f"GA dataset database is missing tables: {names}")


def _select_episode(
    connection: sqlite3.Connection,
    *,
    episode_id: int | None,
) -> tuple[object, ...]:
    if episode_id is not None:
        row = connection.execute(
            "SELECT id, seed, action_hash, result_json FROM episodes WHERE id=?",
            (episode_id,),
        ).fetchone()
        if row is None:
            raise ControlInputError(f"GA dataset has no episode id {episode_id}")
        return row

    return _ordered_episode_rows(connection)[0]


def _ordered_episode_rows(
    connection: sqlite3.Connection,
) -> list[tuple[object, ...]]:
    rows = connection.execute(
        "SELECT id, seed, action_hash, result_json FROM episodes"
    ).fetchall()
    if not rows:
        raise ControlInputError("GA dataset contains no episodes")

    parsed: list[tuple[tuple[int, int, int], tuple[object, ...]]] = []
    for row in rows:
        result = _parse_result(row[3], row[1], row[0])
        parsed.append(
            (
                (
                    _result_int(result, "survival_frames", row[0]),
                    _result_int(result, "frames", row[0]),
                    -_int_value(row[0], "episode id"),
                ),
                row,
            )
        )
    return [row for _, row in sorted(parsed, key=lambda value: value[0], reverse=True)]


def _load_steps(
    connection: sqlite3.Connection,
    episode_id: object,
) -> list[tuple[object, ...]]:
    return connection.execute(
        "SELECT action_index, frame, action, bootstrap "
        "FROM steps WHERE episode_id=? ORDER BY action_index",
        (episode_id,),
    ).fetchall()


def _build_episode(
    row: tuple[object, ...],
    steps: list[tuple[object, ...]],
) -> DatasetEpisode:
    episode_id = _int_value(row[0], "episode id")
    seed = _int_value(row[1], "episode seed")
    action_hash = row[2]
    if not isinstance(action_hash, str) or not action_hash:
        raise ControlInputError(f"GA episode {episode_id} action_hash is invalid")
    result = _parse_result(row[3], seed, episode_id)

    if len(steps) <= len(_BOOTSTRAP_ACTIONS):
        raise ControlInputError(
            f"GA episode {episode_id} does not contain a genome action"
        )

    actions: list[str] = []
    previous_frame = -1
    for expected_index, step in enumerate(steps):
        action_index = _int_value(step[0], f"episode {episode_id} action index")
        frame = _int_value(step[1], f"episode {episode_id} step frame")
        action = step[2]
        bootstrap = step[3]
        if action_index != expected_index:
            raise ControlInputError(
                f"GA episode {episode_id} action indexes are not contiguous"
            )
        if frame < 0 or frame < previous_frame:
            raise ControlInputError(
                f"GA episode {episode_id} step frames are not monotonic"
            )
        previous_frame = frame
        if not isinstance(action, str) or action not in ACTION_CHOICES:
            raise ControlInputError(
                f"GA episode {episode_id} contains invalid action {action!r}"
            )
        expected_bootstrap = int(expected_index < len(_BOOTSTRAP_ACTIONS))
        if bootstrap != expected_bootstrap:
            raise ControlInputError(
                f"GA episode {episode_id} has an invalid bootstrap boundary"
            )
        actions.append(action)

    if tuple(actions[: len(_BOOTSTRAP_ACTIONS)]) != _BOOTSTRAP_ACTIONS:
        raise ControlInputError(
            f"GA episode {episode_id} bootstrap actions do not match the game contract"
        )
    genome_actions = tuple(actions[len(_BOOTSTRAP_ACTIONS) :])
    expected_hash = hashlib.sha256(
        json.dumps(genome_actions, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    if action_hash != expected_hash:
        raise ControlInputError(
            f"GA episode {episode_id} action_hash does not match its genome"
        )

    schedule = [*BOOTSTRAP, *((action, STEP_FRAMES) for action in genome_actions)]
    commands = tuple(
        MovementCommand(action, _frames_to_duration_ms(frames))
        for action, frames in schedule
    )
    return DatasetEpisode(
        episode_id=episode_id,
        seed=seed,
        action_hash=action_hash,
        stored_result=result,
        commands=commands,
        genome_actions=genome_actions,
        recorded_steps=len(steps),
    )


def _parse_result(raw: object, seed: object, episode_id: object) -> dict[str, object]:
    if not isinstance(raw, str):
        raise ControlInputError(f"GA episode {episode_id} result_json is invalid")
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as error:
        raise ControlInputError(
            f"GA episode {episode_id} result_json is not valid JSON"
        ) from error
    if not isinstance(value, dict):
        raise ControlInputError(f"GA episode {episode_id} result_json is not an object")
    if value.get("seed") != seed:
        raise ControlInputError(f"GA episode {episode_id} result seed is inconsistent")
    for name in ("frames", "survival_frames"):
        _result_int(value, name, episode_id)
    for name in ("started", "died"):
        if not isinstance(value.get(name), bool):
            raise ControlInputError(f"GA episode {episode_id} result {name} is invalid")
    return value


def _result_int(value: dict[str, object], name: str, episode_id: object) -> int:
    result = value.get(name)
    if not isinstance(result, int) or isinstance(result, bool) or result < 0:
        raise ControlInputError(f"GA episode {episode_id} result {name} is invalid")
    return result


def _int_value(value: object, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise ControlInputError(f"{name} is invalid")
    return value


def _frames_to_duration_ms(frames: int) -> int:
    if frames < 1:
        raise ControlInputError("GA action duration must contain at least one frame")
    return (frames * 1_000) // 60
