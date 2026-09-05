"""Mojo collection/replay with a batched Python/PyTorch learner.

The native Mojo DQN remains in ``dqn.mojo``.  This executable reuses its
native batch and replay contracts, but delegates dense inference and learning
to a persistent Python object so NumPy and PyTorch can use their optimized
kernels.  The Python boundary is crossed once for action inference per macro
step and once for each learner update; individual tensor operations never
cross it.
"""

import dqn

from std.collections import List
from std.python import Python, PythonObject
from std.python.numpy import copy_to_numpy_array, from_numpy_array
from std.sys import argv
from std.time import perf_counter_ns


struct TorchLearner:
    var learner: PythonObject

    def __init__(
        out self, seed: UInt64, lanes: Int, threads: Int, validate_inputs: Bool
    ) raises:
        Python.add_to_path("mojo/python")
        var module = Python.import_module("torch_waypoint_learner")
        self.learner = module.TorchWaypointLearner(
            Int(seed), lanes, threads, 256, validate_inputs
        )

    def choose_actions(
        self,
        observations: List[Float32],
        epsilon: Float32,
        mut actions: List[UInt8],
    ) raises:
        var python_observations = copy_to_numpy_array(observations)
        var python_actions = self.learner.choose_actions(
            python_observations, Float64(epsilon)
        )
        var action_span = from_numpy_array[DType.uint8](python_actions)
        if len(action_span) != len(actions):
            raise Error("Python learner returned the wrong action count")
        for index in range(len(actions)):
            actions[index] = action_span[index]

    def learn(
        self,
        observations: List[Float32],
        actions: List[UInt8],
        rewards: List[Float32],
        next_observations: List[Float32],
        discounts: List[Float32],
    ) raises -> dqn.LearningStats:
        var python_observations = copy_to_numpy_array(observations)
        var python_actions = copy_to_numpy_array(actions)
        var python_rewards = copy_to_numpy_array(rewards)
        var python_next_observations = copy_to_numpy_array(next_observations)
        var python_discounts = copy_to_numpy_array(discounts)
        var python_metrics = self.learner.learn(
            python_observations,
            python_actions,
            python_rewards,
            python_next_observations,
            python_discounts,
        )
        var metrics = from_numpy_array[DType.float32](python_metrics)
        if len(metrics) != 5:
            raise Error("Python learner returned invalid learning metrics")
        return dqn.LearningStats(
            metrics[0],
            metrics[1],
            metrics[2],
            metrics[3],
            metrics[4],
        )

    def sync_target(self) raises:
        _ = self.learner.sync_target()

    def save_checkpoint(self, path: String) raises:
        _ = self.learner.save_checkpoint(path)


def sample_replay_batch(
    replay: dqn.ReplayBuffer,
    mut rng: dqn.FastRng,
    batch_size: Int,
    mut observations: List[Float32],
    mut actions: List[UInt8],
    mut rewards: List[Float32],
    mut next_observations: List[Float32],
    mut discounts: List[Float32],
):
    for row in range(batch_size):
        var index = rng.below(replay.size)
        var source_offset = index * dqn.OBSERVATION_SIZE
        var destination_offset = row * dqn.OBSERVATION_SIZE
        for feature in range(dqn.OBSERVATION_SIZE):
            observations[destination_offset + feature] = replay.observations[
                source_offset + feature
            ]
            next_observations[
                destination_offset + feature
            ] = replay.next_observations[source_offset + feature]
        actions[row] = replay.actions[index]
        rewards[row] = replay.rewards[index]
        discounts[row] = replay.discounts[index]


def fill_manifest_training_seeds() -> List[UInt32]:
    var values = List[Int](
        [
            30181,
            30169,
            30124,
            30157,
            30129,
            30171,
            30146,
            30113,
            30155,
            30187,
            30182,
            30122,
            30105,
            30193,
            30120,
            30190,
            30178,
            30125,
            30154,
            30132,
            30191,
            30117,
            30158,
            30121,
            30135,
            30118,
            30192,
            30150,
            30141,
            30152,
            30156,
            30151,
            30100,
            30166,
            30106,
            30137,
            30161,
            30109,
            30180,
            30168,
            30197,
            30165,
            30128,
            30194,
            30153,
            30175,
            30145,
            30101,
            30185,
            30136,
            30104,
            30116,
            30130,
            30162,
            30138,
            30183,
            30170,
            30134,
            30189,
            30179,
            30167,
            30177,
            30103,
            30111,
            30115,
            30199,
            30195,
            30108,
            30160,
            30112,
        ],
        __list_literal__=None,
    )
    var seeds = List[UInt32](length=len(values), fill=UInt32(0))
    for index in range(len(values)):
        seeds[index] = UInt32(values[index])
    return seeds.copy()


def collect_macro_transition_with_actions(
    mut environment: dqn.NativeBatch,
    mut current_observations: List[Float32],
    mut current_positions: List[Float32],
    mut episode_steps: List[Int],
    waypoint_actions: List[UInt8],
    mut accumulator: dqn.NStepAccumulator,
    mut replay: dqn.ReplayBuffer,
    training_seeds: List[UInt32],
    mut seed_cursor: Int,
    hold_decisions: Int,
    max_episode_steps: Int,
    grid_spacing: Int,
    mut decision_observations: List[Float32],
    mut next_observations: List[Float32],
    mut target_columns: List[UInt16],
    mut target_rows: List[UInt16],
    mut macro_rewards: List[Float32],
    mut macro_terminated: List[UInt8],
    mut macro_truncated: List[UInt8],
    mut boundary: List[UInt8],
    mut native_actions: List[UInt8],
    mut reset_lanes: List[UInt32],
    mut reset_seeds: List[UInt32],
) raises -> Tuple[Int, Int]:
    for lane in range(environment.lane_count):
        var observation_offset = lane * dqn.OBSERVATION_SIZE
        for feature in range(dqn.OBSERVATION_SIZE):
            decision_observations[
                observation_offset + feature
            ] = current_observations[observation_offset + feature]
        var action = Int(waypoint_actions[lane])
        var cells = dqn.target_cell_for_action(
            current_positions[lane * 2],
            current_positions[lane * 2 + 1],
            action,
            grid_spacing,
        )
        target_columns[lane] = UInt16(cells[0])
        target_rows[lane] = UInt16(cells[1])
        macro_rewards[lane] = Float32(0.0)
        macro_terminated[lane] = UInt8(0)
        macro_truncated[lane] = UInt8(0)
        boundary[lane] = UInt8(0)

    var native_steps = 0
    for _ in range(hold_decisions):
        for lane in range(environment.lane_count):
            native_actions[lane] = UInt8(0)
            if boundary[lane] == UInt8(0):
                native_actions[lane] = dqn.native_action_for_position(
                    current_positions[lane * 2],
                    current_positions[lane * 2 + 1],
                    Int(target_columns[lane]),
                    Int(target_rows[lane]),
                    grid_spacing,
                )
        environment.step(native_actions)
        native_steps += environment.lane_count
        var reset_count = 0
        for lane in range(environment.lane_count):
            var actual_terminal = environment.done[lane] != UInt8(0)
            if boundary[lane] != UInt8(0):
                if actual_terminal:
                    reset_lanes[reset_count] = UInt32(lane)
                    reset_seeds[reset_count] = training_seeds[
                        seed_cursor % len(training_seeds)
                    ]
                    seed_cursor += 1
                    reset_count += 1
                else:
                    dqn.copy_lane(
                        environment.observations,
                        environment.positions,
                        lane,
                        current_observations,
                        current_positions,
                        lane,
                    )
                    episode_steps[lane] += 1
                continue
            macro_rewards[lane] += environment.rewards[lane]
            episode_steps[lane] += 1
            var truncated = (
                not actual_terminal and episode_steps[lane] >= max_episode_steps
            )
            if actual_terminal or truncated:
                boundary[lane] = UInt8(1)
                macro_terminated[lane] = UInt8(1) if actual_terminal else UInt8(
                    0
                )
                macro_truncated[lane] = UInt8(1) if truncated else UInt8(0)
                var observation_offset = lane * dqn.OBSERVATION_SIZE
                for feature in range(dqn.OBSERVATION_SIZE):
                    next_observations[
                        observation_offset + feature
                    ] = environment.observations[observation_offset + feature]
                reset_lanes[reset_count] = UInt32(lane)
                reset_seeds[reset_count] = training_seeds[
                    seed_cursor % len(training_seeds)
                ]
                seed_cursor += 1
                reset_count += 1
            else:
                dqn.copy_lane(
                    environment.observations,
                    environment.positions,
                    lane,
                    current_observations,
                    current_positions,
                    lane,
                )
        if reset_count > 0:
            environment.reset_lanes(reset_lanes, reset_seeds, reset_count)
            for reset_index in range(reset_count):
                var lane = Int(reset_lanes[reset_index])
                dqn.copy_lane(
                    environment.observations,
                    environment.positions,
                    lane,
                    current_observations,
                    current_positions,
                    lane,
                )
                episode_steps[lane] = 0

    for lane in range(environment.lane_count):
        var observation_offset = lane * dqn.OBSERVATION_SIZE
        var next_offset = observation_offset
        if boundary[lane] == UInt8(0):
            for feature in range(dqn.OBSERVATION_SIZE):
                next_observations[
                    observation_offset + feature
                ] = current_observations[observation_offset + feature]
        accumulator.append(
            lane,
            decision_observations,
            observation_offset,
            Int(waypoint_actions[lane]),
            Int(target_columns[lane]),
            Int(target_rows[lane]),
            macro_rewards[lane],
            next_observations,
            next_offset,
            macro_terminated[lane],
            macro_truncated[lane],
            replay,
        )
    return native_steps, seed_cursor


def parse_int_argument(
    args: Span[StringSpan[ImmStaticOrigin], ...], name: String, default: Int
) raises -> Int:
    for index in range(1, len(args) - 1):
        if args[index] == name:
            return Int(args[index + 1])
    return default


def has_argument(
    args: Span[StringSpan[ImmStaticOrigin], ...], name: String
) -> Bool:
    for index in range(1, len(args)):
        if args[index] == name:
            return True
    return False


def run_training(
    library_path: String,
    lanes: Int,
    total_steps: Int,
    batch_size: Int,
    warmup_steps: Int,
    hold_decisions: Int,
    step_frames: Int,
    grid_spacing: Int,
    max_episode_steps: Int,
    seed: UInt64,
    learn_enabled: Bool,
    parallel: Bool,
    torch_threads: Int,
    validate_inputs: Bool,
    checkpoint_path: String,
) raises:
    var environment = dqn.NativeBatch(
        library_path, lanes, step_frames, grid_spacing, parallel
    )
    var learner = TorchLearner(seed, lanes, torch_threads, validate_inputs)
    var replay = dqn.ReplayBuffer(dqn.DEFAULT_REPLAY_CAPACITY)
    var accumulator = dqn.NStepAccumulator(lanes)
    var rng = dqn.FastRng(seed ^ UInt64(0x9E3779B97F4A7C15))
    var training_seeds = fill_manifest_training_seeds()
    var initial_seeds = List[UInt32](length=lanes, fill=UInt32(0))
    for lane in range(lanes):
        initial_seeds[lane] = training_seeds[lane % len(training_seeds)]
    environment.reset(initial_seeds)

    var current_observations = List[Float32](
        length=lanes * dqn.OBSERVATION_SIZE, fill=Float32(0.0)
    )
    var current_positions = List[Float32](length=lanes * 2, fill=Float32(0.0))
    for lane in range(lanes):
        dqn.copy_lane(
            environment.observations,
            environment.positions,
            lane,
            current_observations,
            current_positions,
            lane,
        )
    var episode_steps = List[Int](length=lanes, fill=0)
    var decision_observations = List[Float32](
        length=lanes * dqn.OBSERVATION_SIZE, fill=Float32(0.0)
    )
    var next_observations = List[Float32](
        length=lanes * dqn.OBSERVATION_SIZE, fill=Float32(0.0)
    )
    var waypoint_actions = List[UInt8](length=lanes, fill=UInt8(0))
    var target_columns = List[UInt16](length=lanes, fill=UInt16(0))
    var target_rows = List[UInt16](length=lanes, fill=UInt16(0))
    var macro_rewards = List[Float32](length=lanes, fill=Float32(0.0))
    var macro_terminated = List[UInt8](length=lanes, fill=UInt8(0))
    var macro_truncated = List[UInt8](length=lanes, fill=UInt8(0))
    var boundary = List[UInt8](length=lanes, fill=UInt8(0))
    var native_actions = List[UInt8](length=lanes, fill=UInt8(0))
    var reset_lanes = List[UInt32](length=lanes, fill=UInt32(0))
    var reset_seeds = List[UInt32](length=lanes, fill=UInt32(0))
    var batch_observations = List[Float32](
        length=batch_size * dqn.OBSERVATION_SIZE, fill=Float32(0.0)
    )
    var batch_actions = List[UInt8](length=batch_size, fill=UInt8(0))
    var batch_rewards = List[Float32](length=batch_size, fill=Float32(0.0))
    var batch_next_observations = List[Float32](
        length=batch_size * dqn.OBSERVATION_SIZE, fill=Float32(0.0)
    )
    var batch_discounts = List[Float32](length=batch_size, fill=Float32(0.0))
    var seed_cursor = 0
    var native_steps = 0
    var learner_updates = 0
    var last_stats = dqn.LearningStats(
        Float32(0.0), Float32(0.0), Float32(0.0), Float32(0.0), Float32(0.0)
    )
    var started = perf_counter_ns()
    for step in range(total_steps):
        var epsilon = dqn.epsilon_for_step(step, total_steps)
        learner.choose_actions(current_observations, epsilon, waypoint_actions)
        var result = collect_macro_transition_with_actions(
            environment,
            current_observations,
            current_positions,
            episode_steps,
            waypoint_actions,
            accumulator,
            replay,
            training_seeds,
            seed_cursor,
            hold_decisions,
            max_episode_steps,
            grid_spacing,
            decision_observations,
            next_observations,
            target_columns,
            target_rows,
            macro_rewards,
            macro_terminated,
            macro_truncated,
            boundary,
            native_actions,
            reset_lanes,
            reset_seeds,
        )
        seed_cursor = result[1]
        native_steps += result[0]
        var completed_step = step + 1
        if (
            learn_enabled
            and completed_step >= warmup_steps
            and replay.size >= batch_size
        ):
            sample_replay_batch(
                replay,
                rng,
                batch_size,
                batch_observations,
                batch_actions,
                batch_rewards,
                batch_next_observations,
                batch_discounts,
            )
            last_stats = learner.learn(
                batch_observations,
                batch_actions,
                batch_rewards,
                batch_next_observations,
                batch_discounts,
            )
            learner_updates += 1
        if completed_step % 1000 == 0:
            learner.sync_target()
        if (
            completed_step == 1
            or completed_step == warmup_steps
            or completed_step == total_steps
        ):
            var elapsed = Float64(perf_counter_ns() - started) / Float64(
                1_000_000_000
            )
            print(
                "progress step=",
                completed_step,
                " replay=",
                replay.size,
                " learner_updates=",
                learner_updates,
                " elapsed_s=",
                elapsed,
            )
    var elapsed = Float64(perf_counter_ns() - started) / Float64(1_000_000_000)
    var checksum = Float64(0.0)
    for index in range(len(current_observations)):
        checksum += Float64(current_observations[index]) * Float64(index + 1)
    print(
        "result backend=mojo-collection-python-pytorch lanes=",
        lanes,
        " collection_steps=",
        total_steps,
        " learner_updates=",
        learner_updates,
        " native_steps=",
        native_steps,
        " replay=",
        replay.size,
        " elapsed_s=",
        elapsed,
        " steps_per_s=",
        Float64(total_steps) / max(elapsed, 1e-9),
        " native_steps_per_s=",
        Float64(native_steps) / max(elapsed, 1e-9),
        " checksum=",
        checksum,
        " loss=",
        last_stats.loss,
        " q_mean=",
        last_stats.q_mean,
        " target_mean=",
        last_stats.target_mean,
        " td_error=",
        last_stats.td_error,
        " grad_norm=",
        last_stats.gradient_norm,
    )
    if checkpoint_path.byte_length() > 0:
        learner.save_checkpoint(checkpoint_path)
    environment.close()


def main() raises:
    var args = argv()
    var library_path = "mojo/ffi/target/release/libdodge_mojo_ffi.so"
    for index in range(1, len(args) - 1):
        if args[index] == "--ffi":
            library_path = args[index + 1]
    var lanes = parse_int_argument(args, "--lanes", dqn.DEFAULT_LANES)
    var total_steps = parse_int_argument(
        args, "--steps", dqn.DEFAULT_TOTAL_STEPS
    )
    var batch_size = parse_int_argument(
        args, "--batch-size", dqn.DEFAULT_BATCH_SIZE
    )
    var warmup_steps = parse_int_argument(
        args, "--warmup", dqn.DEFAULT_WARMUP_STEPS
    )
    var hold_decisions = parse_int_argument(
        args, "--hold-decisions", dqn.DEFAULT_HOLD_DECISIONS
    )
    var step_frames = parse_int_argument(
        args, "--step-frames", dqn.DEFAULT_STEP_FRAMES
    )
    var grid_spacing = parse_int_argument(
        args, "--grid-spacing", dqn.DEFAULT_GRID_SPACING
    )
    var max_episode_steps = parse_int_argument(
        args, "--max-episode-steps", dqn.DEFAULT_MAX_EPISODE_STEPS
    )
    var seed = UInt64(parse_int_argument(args, "--seed", 2_026_0903))
    var torch_threads = parse_int_argument(args, "--torch-threads", 2)
    var checkpoint_path = ""
    for index in range(1, len(args) - 1):
        if args[index] == "--checkpoint":
            checkpoint_path = args[index + 1]
    var learn_enabled = not has_argument(args, "--no-learning")
    var parallel = not has_argument(args, "--serial")
    var validate_inputs = has_argument(args, "--validate-inputs")
    if (
        lanes < 1
        or total_steps < 1
        or batch_size < 1
        or warmup_steps < 1
        or torch_threads < 1
    ):
        raise Error(
            "lane, step, batch, warmup, and thread counts must be positive"
        )
    print(
        "hybrid-waypoint-dqn lanes=",
        lanes,
        " steps=",
        total_steps,
        " batch=",
        batch_size,
        " warmup=",
        warmup_steps,
        " learning=",
        learn_enabled,
        " execution=",
        "parallel" if parallel else "serial",
        " torch_threads=",
        torch_threads,
    )
    run_training(
        library_path,
        lanes,
        total_steps,
        batch_size,
        warmup_steps,
        hold_decisions,
        step_frames,
        grid_spacing,
        max_episode_steps,
        seed,
        learn_enabled,
        parallel,
        torch_threads,
        validate_inputs,
        checkpoint_path,
    )
