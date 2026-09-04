from __future__ import annotations

import argparse
import json
import math
import struct
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from dodge.control import CARTRIDGE_PATH
from dodge.native.manifest import manifest_for_path

FRAME_WIDTH = 128
FRAME_HEIGHT = 128
FRAME_SIZE = FRAME_WIDTH * FRAME_HEIGHT
FIXED_SCALE = 1 << 16
SNAPSHOT_MAGIC = b"DGSN"
SNAPSHOT_WIRE_VERSION = 7
SNAPSHOT_PLAYER_OFFSET = 191
SNAPSHOT_PLAYER_END = SNAPSHOT_PLAYER_OFFSET + (2 * 4)
MAX_ENEMIES = 4_096

NATIVE_MODES = {
    0: "menu",
    1: "transition_to_game",
    2: "game",
    3: "terminal",
    4: "settings",
    5: "transition_to_settings",
    6: "transition_to_menu",
}
NATIVE_TRANSITION_MODES = {
    0: "transition_to_menu",
    1: "transition_to_game",
    2: "transition_to_settings",
}
ORACLE_MODES = {
    1: "menu",
    2: "game",
    3: "settings",
    4: "transition_to_game",
}


class NativeDifferentialError(ValueError):
    """A native trace cannot be decoded or compared."""


@dataclass(frozen=True, slots=True)
class NativeEnemy:
    x: int
    y: int
    vx: int
    vy: int
    size: int
    max_size: int
    personality: int
    speed: int
    inside: bool
    is_dying: bool
    isizing: bool
    life: int | None


@dataclass(frozen=True, slots=True)
class NativeParticle:
    x: int
    y: int
    dx: int
    dy: int
    radius: int
    kind: int
    max_age: int
    age: int
    color: int
    colors: tuple[int, int, int]
    color_count: int


@dataclass(frozen=True, slots=True)
class NativePatternTarget:
    kind: str
    move: tuple[int, int, int, int] | None = None
    wait: int | None = None
    set_fyou: bool | None = None
    spawns: tuple[tuple[int, int], ...] = ()


@dataclass(frozen=True, slots=True)
class NativePatternRect:
    x: int
    y: int
    width: int
    height: int
    speed: int
    dx: int
    dy: int
    targets: tuple[NativePatternTarget, ...]
    target_index: int
    wait: int
    shown: bool
    sh: int
    warnings: tuple[tuple[int, int, int, int], ...]
    collision_done: bool
    finished: bool


@dataclass(frozen=True, slots=True)
class NativePattern:
    id: int
    mins: int
    maxs: int
    probability: int
    variants: tuple[int, ...]
    smooth: bool
    pattern_type: int
    bounce_cap: bool
    spawn_enabled: bool
    automatic_variant: int | None
    special: int
    counter: int
    timer: int
    rects: tuple[NativePatternRect, ...]


@dataclass(frozen=True, slots=True)
class NativeRng:
    seed: int
    state: tuple[int, ...]
    front: int
    rear: int


@dataclass(frozen=True, slots=True)
class NativeSettings:
    theme_index: int
    theme_background: int
    theme_shadow: int
    difficulty: int
    patterns_enabled: bool
    powerups_enabled: bool
    cursor: int
    message_timer: int
    message_sprite: int
    message_x: int
    message_y: int


@dataclass(frozen=True, slots=True)
class NativeRenderState:
    draw_color: int
    fill_pattern: int
    draw_palette: bytes
    screen_palette: bytes
    transparent: bytes
    camera_x: int
    camera_y: int
    clip_x: int
    clip_y: int
    clip_width: int
    clip_height: int
    transition_y: int


@dataclass(frozen=True, slots=True)
class NativeSnapshot:
    source_sha256: str
    core_schema_version: int
    seed: int
    frame: int
    mode: str
    transition_y: int
    started: bool
    game_ready: bool
    dead: bool
    input_mask: int
    previous_input_mask: int
    input_source_mode: bool
    rng: NativeRng
    player: tuple[int, int, int, int, int]
    enemies: tuple[NativeEnemy, ...]
    particles: tuple[NativeParticle, ...]
    patterns: tuple[NativePattern, ...]
    active_pattern: int | None
    spawns: tuple[tuple[int, int], ...]
    physical_screen: bytes
    enemy_timer: int
    enemy_est: int
    enemy_stats: tuple[int, int, int, int, int]
    friendly_timer: int
    friendly_enabled: bool
    enemy_max_size: int
    speed: int
    freeze_rate: int
    freeze_active: bool
    freeze_timer: int
    size_timer: int
    patterns_enabled: bool
    powerups_enabled: bool
    pattern_timer: int
    pattern_delay_frames: int
    pattern_active: bool
    new_highscore: bool
    can_click: bool
    has_played: bool
    should_collide: bool
    enemy_should_collide: bool
    bounce_cap_static: int
    bounce_cap_moving: int
    bounce_cap: int
    score: int
    survival_frames: int
    shake: int
    camera_x: int
    camera_y: int
    transition_render_y: int
    transition_from: str
    settings: NativeSettings
    highscores: tuple[int, ...]
    render_state: NativeRenderState
    pixels: bytes


@dataclass(frozen=True, slots=True)
class DifferentialMismatch:
    frame: int
    path: str
    expected: object
    actual: object
    source: dict[str, object]

    def to_json(self) -> dict[str, object]:
        return {
            "frame": self.frame,
            "path": self.path,
            "expected": self.expected,
            "actual": self.actual,
            "source": self.source,
        }


def decode_native_snapshot(snapshot_hex: str) -> NativeSnapshot:
    if not isinstance(snapshot_hex, str):
        raise NativeDifferentialError("native snapshot_hex must be a string")
    try:
        data = bytes.fromhex(snapshot_hex)
    except ValueError as error:
        raise NativeDifferentialError(
            "native snapshot_hex is not hexadecimal"
        ) from error
    reader = _Reader(data)
    if reader.take(4) != SNAPSHOT_MAGIC:
        raise NativeDifferentialError("native snapshot magic mismatch")
    if reader.u32() != SNAPSHOT_WIRE_VERSION:
        raise NativeDifferentialError("native snapshot wire version mismatch")
    core_schema = reader.u32()
    if core_schema < 1:
        raise NativeDifferentialError("native snapshot core schema is invalid")
    source_sha256 = reader.take(32).hex()
    seed = reader.u32()
    frame = reader.u32()
    mode_tag = reader.u8()
    if mode_tag not in NATIVE_MODES:
        raise NativeDifferentialError("native snapshot mode is invalid")
    transition_y = reader.i16()
    started = reader.boolean()
    game_ready = reader.boolean()
    dead = reader.boolean()
    input_mask = reader.u8()
    previous_input_mask = reader.u8()
    input_source_mode = reader.boolean()
    if input_mask > 63 or previous_input_mask > 63:
        raise NativeDifferentialError("native snapshot input mask is invalid")

    rng_seed = reader.u32()
    rng_state = tuple(reader.u32() for _ in range(31))
    front = reader.u8()
    rear = reader.u8()
    if front >= 31 or rear >= 31:
        raise NativeDifferentialError("native snapshot RNG checkpoint is invalid")
    rng = NativeRng(seed=rng_seed, state=rng_state, front=front, rear=rear)

    player = tuple(reader.i32() for _ in range(5))
    enemy_count = reader.u32()
    if enemy_count > MAX_ENEMIES:
        raise NativeDifferentialError("native snapshot enemy count is invalid")
    enemies = tuple(_read_enemy(reader) for _ in range(enemy_count))
    particle_count = reader.u32()
    if particle_count > 16_384:
        raise NativeDifferentialError("native snapshot particle count is invalid")
    particles = tuple(_read_particle(reader) for _ in range(particle_count))
    pattern_count = reader.u32()
    if pattern_count > 128:
        raise NativeDifferentialError("native snapshot pattern count is invalid")
    patterns = tuple(_read_pattern(reader) for _ in range(pattern_count))
    active_pattern: int | None = None
    if reader.boolean():
        active_pattern_index = reader.u32()
        if active_pattern_index >= pattern_count:
            raise NativeDifferentialError("native active pattern index is invalid")
        active_pattern = active_pattern_index

    spawn_count = reader.u32()
    if spawn_count > 256:
        raise NativeDifferentialError("native spawn count is invalid")
    spawns = tuple((reader.i32(), reader.i32()) for _ in range(spawn_count))
    enemy_timer = reader.i32()
    enemy_est = reader.i32()
    enemy_stats = tuple(reader.i32() for _ in range(5))
    friendly_timer = reader.u32()
    friendly_enabled = reader.boolean()
    enemy_max_size = reader.i32()
    speed = reader.i32()
    freeze_rate = reader.i32()
    freeze_active = reader.boolean()
    freeze_timer = reader.u32()
    size_timer = reader.i32()
    patterns_enabled = reader.boolean()
    powerups_enabled = reader.boolean()
    pattern_timer = reader.u32()
    pattern_delay_frames = reader.u32()
    pattern_active = reader.boolean()
    new_highscore = reader.boolean()
    can_click = reader.boolean()
    has_played = reader.boolean()
    should_collide = reader.boolean()
    enemy_should_collide = reader.boolean()
    bounce_cap_static = reader.i32()
    bounce_cap_moving = reader.i32()
    bounce_cap = reader.i32()
    score = reader.i32()
    survival_frames = reader.u32()
    shake = reader.i32()
    camera_x = reader.i32()
    camera_y = reader.i32()
    transition_render_y = reader.i16()
    transition_from_tag = reader.u8()
    if transition_from_tag not in NATIVE_MODES:
        raise NativeDifferentialError("native transition source mode is invalid")
    transition_from = NATIVE_MODES[transition_from_tag]
    settings = NativeSettings(
        theme_index=reader.u8(),
        theme_background=reader.u8(),
        theme_shadow=reader.u8(),
        difficulty=reader.u8(),
        patterns_enabled=reader.boolean(),
        powerups_enabled=reader.boolean(),
        cursor=reader.u8(),
        message_timer=reader.u8(),
        message_sprite=reader.u8(),
        message_x=reader.i16(),
        message_y=reader.i16(),
    )
    highscores = tuple(reader.i32() for _ in range(12))
    physical_screen = reader.take(FRAME_SIZE)
    if any(pixel >= 16 for pixel in physical_screen):
        raise NativeDifferentialError("native physical screen is not palette indexes")

    draw_color = reader.u8()
    fill_pattern = reader.u16()
    draw_palette = reader.take(16)
    screen_palette = reader.take(16)
    transparent = reader.take(16)
    if draw_color >= 16 or any(color >= 16 for color in draw_palette + screen_palette):
        raise NativeDifferentialError("native snapshot palette is invalid")
    if any(value not in {0, 1} for value in transparent):
        raise NativeDifferentialError("native snapshot transparency is invalid")
    render_camera_x = reader.i32()
    render_camera_y = reader.i32()
    clip_x = reader.i16()
    clip_y = reader.i16()
    clip_width = reader.u16()
    clip_height = reader.u16()
    if clip_width > FRAME_WIDTH or clip_height > FRAME_HEIGHT:
        raise NativeDifferentialError("native snapshot clip is invalid")
    render_transition_y = reader.i16()
    render_state = NativeRenderState(
        draw_color=draw_color,
        fill_pattern=fill_pattern,
        draw_palette=draw_palette,
        screen_palette=screen_palette,
        transparent=transparent,
        camera_x=render_camera_x,
        camera_y=render_camera_y,
        clip_x=clip_x,
        clip_y=clip_y,
        clip_width=clip_width,
        clip_height=clip_height,
        transition_y=render_transition_y,
    )

    pixels = reader.take(FRAME_SIZE)
    if any(pixel >= 16 for pixel in pixels):
        raise NativeDifferentialError("native snapshot pixels are not palette indexes")
    reader.ensure_finished()
    return NativeSnapshot(
        source_sha256=source_sha256,
        core_schema_version=core_schema,
        seed=seed,
        frame=frame,
        mode=NATIVE_MODES[mode_tag],
        transition_y=transition_y,
        started=started,
        game_ready=game_ready,
        dead=dead,
        input_mask=input_mask,
        previous_input_mask=previous_input_mask,
        input_source_mode=input_source_mode,
        rng=rng,
        player=player,  # type: ignore[arg-type]
        enemies=enemies,
        particles=particles,
        patterns=patterns,
        active_pattern=active_pattern,
        spawns=spawns,
        physical_screen=physical_screen,
        enemy_timer=enemy_timer,
        enemy_est=enemy_est,
        enemy_stats=enemy_stats,  # type: ignore[arg-type]
        friendly_timer=friendly_timer,
        friendly_enabled=friendly_enabled,
        enemy_max_size=enemy_max_size,
        speed=speed,
        freeze_rate=freeze_rate,
        freeze_active=freeze_active,
        freeze_timer=freeze_timer,
        size_timer=size_timer,
        patterns_enabled=patterns_enabled,
        powerups_enabled=powerups_enabled,
        pattern_timer=pattern_timer,
        pattern_delay_frames=pattern_delay_frames,
        pattern_active=pattern_active,
        new_highscore=new_highscore,
        can_click=can_click,
        has_played=has_played,
        should_collide=should_collide,
        enemy_should_collide=enemy_should_collide,
        bounce_cap_static=bounce_cap_static,
        bounce_cap_moving=bounce_cap_moving,
        bounce_cap=bounce_cap,
        score=score,
        survival_frames=survival_frames,
        shake=shake,
        camera_x=camera_x,
        camera_y=camera_y,
        transition_render_y=transition_render_y,
        transition_from=transition_from,
        settings=settings,
        highscores=highscores,  # type: ignore[arg-type]
        render_state=render_state,
        pixels=pixels,
    )


def decode_native_player_position(snapshot: bytes) -> tuple[float, float]:
    """Read player coordinates without materializing the full snapshot graph."""
    if not isinstance(snapshot, bytes):
        raise NativeDifferentialError("native snapshot must be bytes")
    if len(snapshot) < SNAPSHOT_PLAYER_END:
        raise NativeDifferentialError("native snapshot is too short for player state")
    if snapshot[:4] != SNAPSHOT_MAGIC:
        raise NativeDifferentialError("native snapshot magic mismatch")
    if struct.unpack_from("<I", snapshot, 4)[0] != SNAPSHOT_WIRE_VERSION:
        raise NativeDifferentialError("native snapshot wire version mismatch")
    x, y = struct.unpack_from("<ii", snapshot, SNAPSHOT_PLAYER_OFFSET)
    return x / FIXED_SCALE, y / FIXED_SCALE


def compare_native_to_oracle(
    native: Mapping[str, object],
    oracle: Mapping[str, object],
    *,
    source_map: Mapping[str, object] | None = None,
    compare_pixels: bool = True,
) -> dict[str, object]:
    native_frames = _list_value(native, "frames")
    oracle_frames = _list_value(oracle, "frames")
    for index, expected_frame_value in enumerate(oracle_frames):
        frame = _mapping_value(expected_frame_value, f"oracle frame {index}")
        if index >= len(native_frames):
            frame_number = _int_value(frame, "frame", index)
            mismatch = _mismatch(
                frame_number,
                "frames.present",
                frame_number,
                None,
                "lifecycle",
                source_map,
            )
            return _report(native, oracle, index, mismatch)
        actual_frame = _mapping_value(native_frames[index], f"native frame {index}")
        mismatch = _compare_frame(
            actual_frame,
            frame,
            source_map=source_map,
            compare_pixels=compare_pixels,
        )
        if mismatch is not None:
            return _report(native, oracle, index + 1, mismatch)

    if len(native_frames) > len(oracle_frames):
        actual_frame = _mapping_value(
            native_frames[len(oracle_frames)], "extra native frame"
        )
        frame_number = _int_value(actual_frame, "frame", len(oracle_frames) + 1)
        mismatch = _mismatch(
            frame_number,
            "frames.present",
            None,
            frame_number,
            "lifecycle",
            source_map,
        )
        return _report(native, oracle, len(oracle_frames), mismatch)

    mismatch = _compare_results(
        native,
        oracle,
        native_frames,
        source_map=source_map,
    )
    if mismatch is not None:
        return _report(native, oracle, len(native_frames), mismatch)
    return _report(native, oracle, len(native_frames), None)


def load_source_map(
    path: Path | None = None,
    *,
    source: Path = CARTRIDGE_PATH,
) -> dict[str, object]:
    if path is not None:
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise NativeDifferentialError(
                f"could not read source map: {error}"
            ) from error
        if not isinstance(value, dict):
            raise NativeDifferentialError("source map must be a JSON object")
        return value

    generated = Path("src/dodge/runtime/.native-assets-check-final/source_map.json")
    if generated.exists():
        try:
            value = json.loads(generated.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            value = None
        if isinstance(value, dict):
            return value

    manifest = manifest_for_path(source)
    functions = list(manifest.functions)
    mapped: list[dict[str, object]] = []
    for index, function in enumerate(functions):
        next_line = (
            functions[index + 1].line
            if index + 1 < len(functions)
            else max(section.end_line for section in manifest.sections) + 1
        )
        mapped.append(
            {
                "pico8_name": function.name,
                "source": {"section": "lua", "span": [function.line, next_line - 1]},
                "rust_target": f"dodge_core::{function.name}",
            }
        )
    return {"functions": mapped}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="dodge-native-diff",
        description="Compare a native runner trace with a canonical Pemsa trace.",
    )
    parser.add_argument("--native", required=True, type=Path)
    parser.add_argument("--oracle", required=True, type=Path)
    parser.add_argument("--source-map", type=Path)
    parser.add_argument("--source", type=Path, default=CARTRIDGE_PATH)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--no-pixels",
        action="store_true",
        help="compare logical slice fields only",
    )
    arguments = parser.parse_args(argv)
    try:
        native = _load_json(arguments.native, "native trace")
        oracle = _load_json(arguments.oracle, "oracle trace")
        source_map = load_source_map(arguments.source_map, source=arguments.source)
        report = compare_native_to_oracle(
            native,
            oracle,
            source_map=source_map,
            compare_pixels=not arguments.no_pixels,
        )
        payload = (
            json.dumps(report, sort_keys=True, separators=(",", ":")) + "\n"
        ).encode("utf-8")
        if arguments.output is not None:
            arguments.output.parent.mkdir(parents=True, exist_ok=True)
            arguments.output.write_bytes(payload)
        else:
            sys.stdout.buffer.write(payload)
    except (NativeDifferentialError, OSError, ValueError) as error:
        print(f"dodge-native-diff: {error}", file=sys.stderr)
        return 1
    return 0 if report["status"] == "match" else 2


def _compare_frame(
    actual: Mapping[str, object],
    expected: Mapping[str, object],
    *,
    source_map: Mapping[str, object] | None,
    compare_pixels: bool,
) -> DifferentialMismatch | None:
    frame_number = _int_value(expected, "frame", 0)
    actual_frame = _int_value(actual, "frame", frame_number)
    if actual_frame != frame_number:
        return _mismatch(
            frame_number,
            "frame",
            frame_number,
            actual_frame,
            "lifecycle",
            source_map,
        )

    expected_input = _mapping_value(expected.get("input"), "oracle input")
    expected_mode = ORACLE_MODES.get(
        _int_value(expected_input, "mode", frame_number), "invalid"
    )
    transition_target = expected_input.get("transition_target")
    if transition_target is not None and not isinstance(transition_target, int):
        raise NativeDifferentialError(
            f"oracle frame {frame_number} transition target is invalid"
        )
    actual_mode = actual.get("mode")
    if transition_target is not None:
        expected_mode = NATIVE_TRANSITION_MODES.get(transition_target, "invalid")
    elif expected_mode == "transition_to_game":
        expected_mode = "transition_to_game"
    comparisons = (
        (
            "input.mask",
            _int_value(expected_input, "mask", frame_number),
            _int_value(actual, "input_mask", frame_number),
            "input",
        ),
        (
            "input.previous_mask",
            _int_value(expected_input, "previous_mask", frame_number),
            _int_value(actual, "previous_input_mask", frame_number),
            "input",
        ),
        (
            "lifecycle.mode",
            expected_mode,
            actual_mode,
            "lifecycle",
        ),
        (
            "lifecycle.dead",
            bool(expected_input.get("dead")),
            bool(actual.get("dead")),
            "lifecycle",
        ),
        (
            "terminal.done",
            bool(expected.get("done")),
            bool(actual.get("done")),
            "lifecycle",
        ),
    )
    for path, expected_value, actual_value, source_name in comparisons:
        if expected_value != actual_value:
            return _mismatch(
                frame_number,
                path,
                expected_value,
                actual_value,
                source_name,
                source_map,
            )

    expected_reward = expected.get("reward", 0.0)
    actual_reward_raw = actual.get("reward_raw", 0)
    if not isinstance(expected_reward, int | float) or isinstance(
        expected_reward, bool
    ):
        raise NativeDifferentialError(f"oracle frame {frame_number} reward is invalid")
    if not isinstance(actual_reward_raw, int) or isinstance(actual_reward_raw, bool):
        raise NativeDifferentialError(
            f"native frame {frame_number} reward_raw is invalid"
        )
    actual_reward = actual_reward_raw / FIXED_SCALE
    if not math.isclose(float(expected_reward), actual_reward, abs_tol=0.0002):
        return _mismatch(
            frame_number,
            "reward",
            expected_reward,
            actual_reward,
            "updategame",
            source_map,
        )

    expected_events = expected.get("events", [])
    actual_events = actual.get("events", [])
    if not isinstance(expected_events, list) or not all(
        isinstance(event, str) for event in expected_events
    ):
        raise NativeDifferentialError(f"oracle frame {frame_number} events are invalid")
    if not isinstance(actual_events, list) or not all(
        isinstance(event, str) for event in actual_events
    ):
        raise NativeDifferentialError(f"native frame {frame_number} events are invalid")
    if expected_events != actual_events:
        return _mismatch(
            frame_number,
            "events",
            expected_events,
            actual_events,
            "updategame",
            source_map,
        )

    snapshot = decode_native_snapshot(
        _string_value(actual, "snapshot_hex", frame_number)
    )
    if snapshot.frame != frame_number:
        return _mismatch(
            frame_number,
            "snapshot.lifecycle.frame",
            frame_number,
            snapshot.frame,
            "lifecycle",
            source_map,
        )
    if snapshot.mode != actual.get("mode"):
        return _mismatch(
            frame_number,
            "snapshot.lifecycle.mode",
            actual.get("mode"),
            snapshot.mode,
            "lifecycle",
            source_map,
        )
    state = _mapping_value(expected.get("state"), "oracle state")
    expected_player = _mapping_value(state.get("player"), "oracle player")
    for index, name in enumerate(("x", "y", "vx", "vy", "size")):
        expected_value = _number_value(expected_player, name, frame_number)
        actual_value = _fixed_float(snapshot.player[index])
        if not math.isclose(expected_value, actual_value, abs_tol=0.0002):
            return _mismatch(
                frame_number,
                f"player.{name}",
                expected_value,
                actual_value,
                "player",
                source_map,
            )

    expected_enemies = _list_value(state, "enemies")
    if len(expected_enemies) != len(snapshot.enemies):
        source_name = "updatefyou" if len(expected_enemies) > 0 else "updateenemies"
        return _mismatch(
            frame_number,
            "enemies.count",
            len(expected_enemies),
            len(snapshot.enemies),
            source_name,
            source_map,
        )
    for index, expected_enemy_value in enumerate(expected_enemies):
        expected_enemy = _mapping_value(
            expected_enemy_value,
            f"oracle enemy {index}",
        )
        actual_enemy = snapshot.enemies[index]
        fields = (
            ("x", expected_enemy.get("x"), actual_enemy.x),
            ("y", expected_enemy.get("y"), actual_enemy.y),
            ("vx", expected_enemy.get("vx"), actual_enemy.vx),
            ("vy", expected_enemy.get("vy"), actual_enemy.vy),
            (
                "width",
                expected_enemy.get("width"),
                8 * FIXED_SCALE if actual_enemy.personality >= 2 else actual_enemy.size,
            ),
            (
                "height",
                expected_enemy.get("height"),
                8 * FIXED_SCALE if actual_enemy.personality >= 2 else actual_enemy.size,
            ),
        )
        for name, expected_value, actual_raw in fields:
            if not isinstance(expected_value, int | float):
                raise NativeDifferentialError(
                    f"oracle enemy {index} field {name} is not numeric"
                )
            actual_value = _fixed_float(actual_raw)
            if not math.isclose(float(expected_value), actual_value, abs_tol=0.0002):
                return _mismatch(
                    frame_number,
                    f"enemies[{index}].{name}",
                    expected_value,
                    actual_value,
                    "updateenemies",
                    source_map,
                )

    expected_particles = state.get("particles", [])
    if not isinstance(expected_particles, list):
        raise NativeDifferentialError(
            f"oracle frame {frame_number} particles are invalid"
        )
    if len(expected_particles) != len(snapshot.particles):
        return _mismatch(
            frame_number,
            "particles.count",
            len(expected_particles),
            len(snapshot.particles),
            "updateparts",
            source_map,
        )
    for index, expected_particle_value in enumerate(expected_particles):
        expected_particle = _mapping_value(
            expected_particle_value,
            f"oracle particle {index}",
        )
        actual_particle = snapshot.particles[index]
        particle_fields = (
            ("x", expected_particle.get("x"), actual_particle.x),
            ("y", expected_particle.get("y"), actual_particle.y),
            ("dx", expected_particle.get("dx"), actual_particle.dx),
            ("dy", expected_particle.get("dy"), actual_particle.dy),
            ("radius", expected_particle.get("radius"), actual_particle.radius),
            ("max_age", expected_particle.get("max_age"), actual_particle.max_age),
            ("age", expected_particle.get("age"), actual_particle.age),
            ("kind", expected_particle.get("kind"), actual_particle.kind),
            ("color", expected_particle.get("color"), actual_particle.color),
        )
        for name, expected_value, actual_value in particle_fields:
            if not isinstance(expected_value, int | float) or isinstance(
                expected_value, bool
            ):
                raise NativeDifferentialError(
                    f"oracle particle {index} field {name} is not numeric"
                )
            normalized_actual = (
                _fixed_float(actual_value)
                if name in {"x", "y", "dx", "dy", "radius", "max_age"}
                else actual_value
            )
            if not math.isclose(
                float(expected_value), float(normalized_actual), abs_tol=0.0002
            ):
                return _mismatch(
                    frame_number,
                    f"particles[{index}].{name}",
                    expected_value,
                    normalized_actual,
                    "updateparts",
                    source_map,
                )
        expected_colors = expected_particle.get("colors")
        if not isinstance(expected_colors, list) or expected_colors != list(
            actual_particle.colors[: actual_particle.color_count]
        ):
            return _mismatch(
                frame_number,
                f"particles[{index}].colors",
                expected_colors,
                list(actual_particle.colors[: actual_particle.color_count]),
                "addpart",
                source_map,
            )

    if compare_pixels:
        expected_pixels = _oracle_pixels(expected, frame_number)
        actual_pixels = snapshot.pixels
        if len(expected_pixels) != len(actual_pixels):
            raise NativeDifferentialError(
                "native and oracle framebuffer lengths differ"
            )
        for pixel_index, (expected_value, actual_value) in enumerate(
            zip(expected_pixels, actual_pixels, strict=True)
        ):
            if expected_value != actual_value:
                y, x = divmod(pixel_index, FRAME_WIDTH)
                mode = _int_value(expected_input, "mode", frame_number)
                source_name = "drawtransition" if mode == 4 else "drawgame"
                return _mismatch(
                    frame_number,
                    f"pixels[{y},{x}]",
                    expected_value,
                    actual_value,
                    source_name,
                    source_map,
                )
    return None


def _compare_results(
    native: Mapping[str, object],
    oracle: Mapping[str, object],
    native_frames: Sequence[object],
    *,
    source_map: Mapping[str, object] | None,
) -> DifferentialMismatch | None:
    if not native_frames:
        raise NativeDifferentialError("trace contains no frames")
    native_result = _mapping_value(native.get("result"), "native result")
    oracle_result = _mapping_value(oracle.get("result"), "oracle result")
    frame = _int_value(native_frames[-1], "frame", len(native_frames))
    for name in ("frames",):
        if name in oracle_result and name in native_result:
            expected = oracle_result[name]
            actual = native_result[name]
            if expected != actual:
                return _mismatch(
                    frame,
                    f"result.{name}",
                    expected,
                    actual,
                    "lifecycle",
                    source_map,
                )
    last_snapshot = decode_native_snapshot(
        _string_value(native_frames[-1], "snapshot_hex", frame)
    )
    if "score" in oracle_result:
        expected_score = oracle_result["score"]
        actual_score = _fixed_float(last_snapshot.score)
        if isinstance(expected_score, int | float) and not math.isclose(
            float(expected_score), actual_score, abs_tol=0.0002
        ):
            return _mismatch(
                frame,
                "result.score",
                expected_score,
                actual_score,
                "collide",
                source_map,
            )
    if "survival_frames" in oracle_result:
        expected_survival = oracle_result["survival_frames"]
        if expected_survival != last_snapshot.survival_frames:
            return _mismatch(
                frame,
                "result.survival_frames",
                expected_survival,
                last_snapshot.survival_frames,
                "updategame",
                source_map,
            )
    return None


def _report(
    native: Mapping[str, object],
    oracle: Mapping[str, object],
    frames_compared: int,
    mismatch: DifferentialMismatch | None,
) -> dict[str, object]:
    return {
        "schema_version": 1,
        "status": "match" if mismatch is None else "mismatch",
        "frames_compared": frames_compared,
        "first_mismatch": None if mismatch is None else mismatch.to_json(),
        "native_result": native.get("result"),
        "oracle_result": oracle.get("result"),
    }


def _mismatch(
    frame: int,
    path: str,
    expected: object,
    actual: object,
    source_name: str,
    source_map: Mapping[str, object] | None,
) -> DifferentialMismatch:
    return DifferentialMismatch(
        frame=frame,
        path=path,
        expected=expected,
        actual=actual,
        source=_source_for(source_name, source_map),
    )


def _source_for(
    function_name: str,
    source_map: Mapping[str, object] | None,
) -> dict[str, object]:
    function_name = {
        "input": "_update60",
        "lifecycle": "_update60",
        "player": "updategame",
        "updatefyou": "updatefyou",
        "updateenemies": "updateenemies",
        "collide": "collide",
    }.get(function_name, function_name)
    if source_map is not None:
        functions = source_map.get("functions")
        if isinstance(functions, list):
            for function in functions:
                if not isinstance(function, dict):
                    continue
                if function.get("pico8_name") != function_name:
                    continue
                source = function.get("source")
                if isinstance(source, dict):
                    result: dict[str, object] = {
                        "pico8_name": function_name,
                        "section": source.get("section"),
                        "span": source.get("span"),
                    }
                    if "rust_target" in function:
                        result["rust_target"] = function["rust_target"]
                    return result
    return {
        "pico8_name": function_name,
        "section": "lua",
        "span": [1, 2_257],
    }


def _oracle_pixels(frame: Mapping[str, object], frame_number: int) -> bytes:
    pixels = _mapping_value(frame.get("pixels"), f"oracle pixels {frame_number}")
    if pixels.get("encoding") != "palette_index_u8_row_major":
        raise NativeDifferentialError("oracle framebuffer encoding is not canonical")
    if pixels.get("width") != FRAME_WIDTH or pixels.get("height") != FRAME_HEIGHT:
        raise NativeDifferentialError("oracle framebuffer dimensions are invalid")
    data_hex = pixels.get("data_hex")
    if not isinstance(data_hex, str):
        raise NativeDifferentialError("oracle framebuffer data_hex is invalid")
    try:
        data = bytes.fromhex(data_hex)
    except ValueError as error:
        raise NativeDifferentialError(
            "oracle framebuffer is not hexadecimal"
        ) from error
    if len(data) != FRAME_SIZE:
        raise NativeDifferentialError("oracle framebuffer length is invalid")
    return data


def _read_enemy(reader: _Reader) -> NativeEnemy:
    return NativeEnemy(
        x=reader.i32(),
        y=reader.i32(),
        vx=reader.i32(),
        vy=reader.i32(),
        size=reader.i32(),
        max_size=reader.i32(),
        personality=reader.i8(),
        speed=reader.i32(),
        inside=reader.boolean(),
        is_dying=reader.boolean(),
        isizing=reader.boolean(),
        life=reader.i32() if reader.boolean() else None,
    )


def _read_particle(reader: _Reader) -> NativeParticle:
    x = reader.i32()
    y = reader.i32()
    dx = reader.i32()
    dy = reader.i32()
    radius = reader.i32()
    kind = reader.i8()
    max_age = reader.i32()
    age = reader.u32()
    color = reader.u8()
    colors = tuple(reader.u8() for _ in range(3))
    color_count = reader.u8()
    if color_count == 0 or color_count > 3 or any(color >= 16 for color in colors):
        raise NativeDifferentialError("native snapshot particle color is invalid")
    return NativeParticle(
        x=x,
        y=y,
        dx=dx,
        dy=dy,
        radius=radius,
        kind=kind,
        max_age=max_age,
        age=age,
        color=color,
        colors=colors,  # type: ignore[arg-type]
        color_count=color_count,
    )


def _read_pattern(reader: _Reader) -> NativePattern:
    pattern_id = reader.u8()
    mins = reader.i32()
    maxs = reader.i32()
    probability = reader.i32()
    variant_count = reader.u32()
    if variant_count > 256:
        raise NativeDifferentialError("native pattern variant count is invalid")
    variants = tuple(reader.u8() for _ in range(variant_count))
    smooth = reader.boolean()
    pattern_type = reader.u8()
    bounce_cap = reader.boolean()
    spawn_enabled = reader.boolean()
    automatic_variant = reader.u8() if reader.boolean() else None
    special = reader.i32()
    counter = reader.u32()
    timer = reader.i32()
    rect_count = reader.u32()
    if rect_count > 4096:
        raise NativeDifferentialError("native pattern rectangle count is invalid")
    rects = tuple(_read_pattern_rect(reader) for _ in range(rect_count))
    return NativePattern(
        id=pattern_id,
        mins=mins,
        maxs=maxs,
        probability=probability,
        variants=variants,
        smooth=smooth,
        pattern_type=pattern_type,
        bounce_cap=bounce_cap,
        spawn_enabled=spawn_enabled,
        automatic_variant=automatic_variant,
        special=special,
        counter=counter,
        timer=timer,
        rects=rects,
    )


def _read_pattern_rect(reader: _Reader) -> NativePatternRect:
    x = reader.i32()
    y = reader.i32()
    width = reader.i32()
    height = reader.i32()
    speed = reader.i32()
    dx = reader.i32()
    dy = reader.i32()
    target_count = reader.u32()
    if target_count > 256:
        raise NativeDifferentialError("native pattern target count is invalid")
    targets: list[NativePatternTarget] = []
    for _ in range(target_count):
        tag = reader.u8()
        if tag == 0:
            targets.append(
                NativePatternTarget(
                    kind="move",
                    move=tuple(reader.i32() for _ in range(4)),  # type: ignore[arg-type]
                )
            )
        elif tag == 1:
            targets.append(NativePatternTarget(kind="wait", wait=reader.i32()))
        elif tag == 2:
            targets.append(
                NativePatternTarget(kind="set_fyou", set_fyou=reader.boolean())
            )
        elif tag == 3:
            point_count = reader.u32()
            if point_count > 256:
                raise NativeDifferentialError("native pattern point count is invalid")
            targets.append(
                NativePatternTarget(
                    kind="set_spawns",
                    spawns=tuple(
                        (reader.i32(), reader.i32()) for _ in range(point_count)
                    ),
                )
            )
        else:
            raise NativeDifferentialError("native pattern target tag is invalid")
    target_index = reader.u32()
    if target_index > target_count:
        raise NativeDifferentialError("native pattern target index is invalid")
    wait = reader.i32()
    shown = reader.boolean()
    sh = reader.i32()
    warning_count = reader.u32()
    if warning_count > 256:
        raise NativeDifferentialError("native pattern warning count is invalid")
    warnings = tuple(
        tuple(reader.i32() for _ in range(4))  # type: ignore[arg-type]
        for _ in range(warning_count)
    )
    collision_done = reader.boolean()
    finished = reader.boolean()
    return NativePatternRect(
        x=x,
        y=y,
        width=width,
        height=height,
        speed=speed,
        dx=dx,
        dy=dy,
        targets=tuple(targets),
        target_index=target_index,
        wait=wait,
        shown=shown,
        sh=sh,
        warnings=warnings,
        collision_done=collision_done,
        finished=finished,
    )


def _fixed_float(raw: int) -> float:
    return raw / FIXED_SCALE


def _load_json(path: Path, label: str) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise NativeDifferentialError(f"could not read {label}: {error}") from error
    if not isinstance(value, dict):
        raise NativeDifferentialError(f"{label} must be a JSON object")
    return value


def _mapping_value(value: object, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise NativeDifferentialError(f"{label} must be a JSON object")
    return value


def _list_value(value: Mapping[str, object], name: str) -> list[object]:
    result = value.get(name)
    if not isinstance(result, list):
        raise NativeDifferentialError(f"{name} must be a JSON list")
    return result


def _int_value(value: Mapping[str, object], name: str, frame: int) -> int:
    result = value.get(name)
    if isinstance(result, bool) or not isinstance(result, int):
        raise NativeDifferentialError(f"frame {frame} {name} must be an integer")
    return result


def _number_value(value: Mapping[str, object], name: str, frame: int) -> float:
    result = value.get(name)
    if isinstance(result, bool) or not isinstance(result, int | float):
        raise NativeDifferentialError(f"frame {frame} {name} must be numeric")
    return float(result)


def _string_value(value: Mapping[str, object], name: str, frame: int) -> str:
    result = value.get(name)
    if not isinstance(result, str):
        raise NativeDifferentialError(f"frame {frame} {name} must be a string")
    return result


class _Reader:
    def __init__(self, data: bytes) -> None:
        self._data = data
        self._offset = 0

    def take(self, length: int) -> bytes:
        end = self._offset + length
        if length < 0 or end > len(self._data):
            raise NativeDifferentialError("native snapshot is truncated")
        value = self._data[self._offset : end]
        self._offset = end
        return value

    def skip(self, length: int) -> None:
        self.take(length)

    def u8(self) -> int:
        return self.take(1)[0]

    def i8(self) -> int:
        value = self.u8()
        return value - 256 if value >= 128 else value

    def boolean(self) -> bool:
        value = self.u8()
        if value not in {0, 1}:
            raise NativeDifferentialError("native snapshot boolean is invalid")
        return bool(value)

    def u16(self) -> int:
        return struct.unpack("<H", self.take(2))[0]

    def i16(self) -> int:
        return struct.unpack("<h", self.take(2))[0]

    def u32(self) -> int:
        return struct.unpack("<I", self.take(4))[0]

    def i32(self) -> int:
        return struct.unpack("<i", self.take(4))[0]

    def ensure_finished(self) -> None:
        if self._offset != len(self._data):
            raise NativeDifferentialError("native snapshot has trailing bytes")
