"""Independent Mojo port of the waypoint DQN hot path.

The Rust native batch engine supplies the same 225-float observation contract
as the Python learner through ``mojo/ffi``.  The learner, replay storage,
three-step accumulator, waypoint controller, and benchmark loop are native
Mojo code.  Checkpointing, dashboard telemetry, and report generation are
intentionally outside this first performance investigation.
"""

from std.collections import List
from std.ffi import OwnedDLHandle, c_int, c_ulong_long
from std.math import sqrt
from std.sys import argv
from std.time import perf_counter_ns


comptime OBSERVATION_SIZE = 225
comptime ACTION_COUNT = 9
comptime HIDDEN_SIZE = 256
comptime SIMD_WIDTH = 8
comptime N_STEP = 3
comptime DEFAULT_LANES = 32
comptime DEFAULT_TOTAL_STEPS = 20_000
comptime DEFAULT_BATCH_SIZE = 256
comptime DEFAULT_REPLAY_CAPACITY = 100_000
comptime DEFAULT_WARMUP_STEPS = 2_000
comptime DEFAULT_HOLD_DECISIONS = 8
comptime DEFAULT_STEP_FRAMES = 4
comptime DEFAULT_GRID_SPACING = 32
comptime DEFAULT_MAX_EPISODE_STEPS = 2_000
comptime GAMMA: Float32 = 0.99
comptime LEARNING_RATE: Float32 = 0.0001
comptime WEIGHT_DECAY: Float32 = 0.01


struct FastRng:
    var state: UInt64

    def __init__(out self, seed: UInt64):
        self.state = seed if seed != UInt64(0) else UInt64(1)

    def next_u64(mut self) -> UInt64:
        var value = self.state
        value ^= value << 13
        value ^= value >> 7
        value ^= value << 17
        self.state = value
        return value

    def uniform(mut self) -> Float32:
        var value = self.next_u64() >> 40
        return Float32(value) / Float32(16_777_216.0)

    def below(mut self, upper: Int) -> Int:
        return Int(self.next_u64() % UInt64(upper))


struct NativeBatch:
    var library: OwnedDLHandle
    var handle: c_ulong_long
    var lane_count: Int
    var observations: List[Float32]
    var positions: List[Float32]
    var rewards: List[Float32]
    var done: List[UInt8]

    def __init__(
        out self,
        library_path: String,
        lanes: Int,
        step_frames: Int,
        grid_spacing: Int,
        parallel: Bool,
    ) raises:
        self.library = OwnedDLHandle(library_path)
        var abi_version = self.library.call["mojo_batch_abi_version", UInt32]()
        var observation_size = self.library.call[
            "mojo_batch_observation_size", UInt32
        ]()
        if abi_version != UInt32(1) or observation_size != UInt32(
            OBSERVATION_SIZE
        ):
            raise Error("unsupported Mojo native batch ABI")
        self.handle = self.library.call["mojo_batch_new", c_ulong_long](
            UInt32(lanes),
            UInt32(step_frames),
            UInt32(grid_spacing),
            UInt8(1) if parallel else UInt8(0),
        )
        if self.handle == c_ulong_long(0):
            raise Error("could not create the Mojo native batch")
        self.lane_count = lanes
        self.observations = List[Float32](
            length=lanes * OBSERVATION_SIZE, fill=Float32(0.0)
        )
        self.positions = List[Float32](length=lanes * 2, fill=Float32(0.0))
        self.rewards = List[Float32](length=lanes, fill=Float32(0.0))
        self.done = List[UInt8](length=lanes, fill=UInt8(0))

    def reset(mut self, seeds: List[UInt32]) raises:
        var status = self.library.call["mojo_batch_reset", c_int](
            self.handle,
            seeds.unsafe_ptr(),
            UInt32(self.lane_count),
        )
        if status != 0:
            raise Error("native batch reset failed")
        self.copy_result()

    def step(mut self, actions: List[UInt8]) raises:
        var status = self.library.call["mojo_batch_step", c_int](
            self.handle,
            actions.unsafe_ptr(),
            UInt32(self.lane_count),
        )
        if status != 0:
            raise Error("native batch step failed")
        self.copy_result()

    def reset_lanes(
        mut self,
        lanes: List[UInt32],
        seeds: List[UInt32],
        count: Int,
    ) raises:
        if count == 0:
            return
        var status = self.library.call["mojo_batch_reset_lanes", c_int](
            self.handle,
            lanes.unsafe_ptr(),
            seeds.unsafe_ptr(),
            UInt32(count),
        )
        if status != 0:
            raise Error("native batch lane reset failed")
        var reset_observations = List[Float32](
            length=count * OBSERVATION_SIZE, fill=Float32(0.0)
        )
        var reset_positions = List[Float32](length=count * 2, fill=Float32(0.0))
        var reset_rewards = List[Float32](length=count, fill=Float32(0.0))
        var reset_done = List[UInt8](length=count, fill=UInt8(0))
        copy_native_result(
            self.library,
            self.handle,
            reset_observations,
            reset_positions,
            reset_rewards,
            reset_done,
            count,
        )
        for reset_index in range(count):
            var lane = Int(lanes[reset_index])
            for feature in range(OBSERVATION_SIZE):
                self.observations[
                    lane * OBSERVATION_SIZE + feature
                ] = reset_observations[reset_index * OBSERVATION_SIZE + feature]
            self.positions[lane * 2] = reset_positions[reset_index * 2]
            self.positions[lane * 2 + 1] = reset_positions[reset_index * 2 + 1]
            self.rewards[lane] = reset_rewards[reset_index]
            self.done[lane] = reset_done[reset_index]

    def close(mut self):
        if self.handle != c_ulong_long(0):
            self.library.call["mojo_batch_free", NoneType](self.handle)
            self.handle = c_ulong_long(0)

    def copy_result(mut self) raises:
        copy_native_result(
            self.library,
            self.handle,
            self.observations,
            self.positions,
            self.rewards,
            self.done,
            self.lane_count,
        )


def copy_native_result(
    mut library: OwnedDLHandle,
    handle: c_ulong_long,
    mut destination_observations: List[Float32],
    mut destination_positions: List[Float32],
    mut destination_rewards: List[Float32],
    mut destination_done: List[UInt8],
    count: Int,
) raises:
    var status = library.call["mojo_batch_copy_all", c_int](
        handle,
        destination_observations.unsafe_ptr(),
        UInt32(count * OBSERVATION_SIZE),
        destination_positions.unsafe_ptr(),
        UInt32(count * 2),
        destination_rewards.unsafe_ptr(),
        UInt32(count),
        destination_done.unsafe_ptr(),
        UInt32(count),
    )
    if status != 0:
        raise Error("native batch result copy failed")


struct ReplayBuffer:
    """Fixed-size uniform replay storage matching the Python fields."""

    var capacity: Int
    var position: Int
    var size: Int
    var observations: List[Float32]
    var next_observations: List[Float32]
    var actions: List[UInt8]
    var target_columns: List[UInt16]
    var target_rows: List[UInt16]
    var rewards: List[Float32]
    var discounts: List[Float32]
    var terminated: List[UInt8]
    var truncated: List[UInt8]
    var n_steps: List[UInt8]

    def __init__(out self, capacity: Int):
        self.capacity = capacity
        self.position = 0
        self.size = 0
        self.observations = List[Float32](
            length=capacity * OBSERVATION_SIZE, fill=Float32(0.0)
        )
        self.next_observations = List[Float32](
            length=capacity * OBSERVATION_SIZE, fill=Float32(0.0)
        )
        self.actions = List[UInt8](length=capacity, fill=UInt8(0))
        self.target_columns = List[UInt16](length=capacity, fill=UInt16(0))
        self.target_rows = List[UInt16](length=capacity, fill=UInt16(0))
        self.rewards = List[Float32](length=capacity, fill=Float32(0.0))
        self.discounts = List[Float32](length=capacity, fill=Float32(0.0))
        self.terminated = List[UInt8](length=capacity, fill=UInt8(0))
        self.truncated = List[UInt8](length=capacity, fill=UInt8(0))
        self.n_steps = List[UInt8](length=capacity, fill=UInt8(0))

    def add(
        mut self,
        observation: List[Float32],
        observation_offset: Int,
        action: Int,
        target_column: Int,
        target_row: Int,
        reward: Float32,
        next_observation: List[Float32],
        next_observation_offset: Int,
        discount: Float32,
        terminated: UInt8,
        truncated: UInt8,
        horizon: Int,
    ):
        var index = self.position
        var source_offset = 0
        var destination_offset = index * OBSERVATION_SIZE
        for feature in range(OBSERVATION_SIZE):
            self.observations[destination_offset + feature] = observation[
                observation_offset + source_offset + feature
            ]
            self.next_observations[
                destination_offset + feature
            ] = next_observation[
                next_observation_offset + source_offset + feature
            ]
        self.actions[index] = UInt8(action)
        self.target_columns[index] = UInt16(target_column)
        self.target_rows[index] = UInt16(target_row)
        self.rewards[index] = reward
        self.discounts[index] = discount
        self.terminated[index] = terminated
        self.truncated[index] = truncated
        self.n_steps[index] = UInt8(horizon)
        self.position = (index + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample_index(self, mut rng: FastRng) -> Int:
        return rng.below(self.size)

    def copy_observation(self, index: Int, mut destination: List[Float32]):
        var offset = index * OBSERVATION_SIZE
        for feature in range(OBSERVATION_SIZE):
            destination[feature] = self.observations[offset + feature]

    def copy_next_observation(self, index: Int, mut destination: List[Float32]):
        var offset = index * OBSERVATION_SIZE
        for feature in range(OBSERVATION_SIZE):
            destination[feature] = self.next_observations[offset + feature]


struct NStepAccumulator:
    """Fixed three-step queues, one queue per native lane."""

    var lane_count: Int
    var queue_length: List[Int]
    var observations: List[Float32]
    var next_observations: List[Float32]
    var actions: List[UInt8]
    var target_columns: List[UInt16]
    var target_rows: List[UInt16]
    var rewards: List[Float32]
    var terminated: List[UInt8]
    var truncated: List[UInt8]

    def __init__(out self, lane_count: Int):
        self.lane_count = lane_count
        self.queue_length = List[Int](length=lane_count, fill=0)
        self.observations = List[Float32](
            length=lane_count * N_STEP * OBSERVATION_SIZE, fill=Float32(0.0)
        )
        self.next_observations = List[Float32](
            length=lane_count * N_STEP * OBSERVATION_SIZE, fill=Float32(0.0)
        )
        self.actions = List[UInt8](length=lane_count * N_STEP, fill=UInt8(0))
        self.target_columns = List[UInt16](
            length=lane_count * N_STEP, fill=UInt16(0)
        )
        self.target_rows = List[UInt16](
            length=lane_count * N_STEP, fill=UInt16(0)
        )
        self.rewards = List[Float32](
            length=lane_count * N_STEP, fill=Float32(0.0)
        )
        self.terminated = List[UInt8](length=lane_count * N_STEP, fill=UInt8(0))
        self.truncated = List[UInt8](length=lane_count * N_STEP, fill=UInt8(0))

    def append(
        mut self,
        lane: Int,
        observation: List[Float32],
        observation_offset: Int,
        action: Int,
        target_column: Int,
        target_row: Int,
        reward: Float32,
        next_observation: List[Float32],
        next_observation_offset: Int,
        is_terminated: UInt8,
        is_truncated: UInt8,
        mut replay: ReplayBuffer,
    ):
        var length = self.queue_length[lane]
        var slot = lane * N_STEP + length
        var obs_offset = slot * OBSERVATION_SIZE
        for feature in range(OBSERVATION_SIZE):
            self.observations[obs_offset + feature] = observation[
                observation_offset + feature
            ]
            self.next_observations[obs_offset + feature] = next_observation[
                next_observation_offset + feature
            ]
        self.actions[slot] = UInt8(action)
        self.target_columns[slot] = UInt16(target_column)
        self.target_rows[slot] = UInt16(target_row)
        self.rewards[slot] = reward
        self.terminated[slot] = is_terminated
        self.truncated[slot] = is_truncated
        self.queue_length[lane] = length + 1
        if self.queue_length[lane] >= N_STEP:
            self.emit(lane, replay)
        if is_terminated != UInt8(0) or is_truncated != UInt8(0):
            while self.queue_length[lane] > 0:
                self.emit(lane, replay)

    def emit(mut self, lane: Int, mut replay: ReplayBuffer):
        var length = self.queue_length[lane]
        if length == 0:
            return
        var reward = Float32(0.0)
        var discount = Float32(1.0)
        var horizon = 0
        var last_slot = lane * N_STEP
        for offset in range(length):
            var slot = lane * N_STEP + offset
            reward += discount * self.rewards[slot]
            horizon += 1
            last_slot = slot
            discount *= GAMMA
            if (
                self.terminated[slot] != UInt8(0)
                or self.truncated[slot] != UInt8(0)
                or horizon >= N_STEP
            ):
                break
        var first_slot = lane * N_STEP
        var boundary = self.terminated[last_slot] != UInt8(0) or self.truncated[
            last_slot
        ] != UInt8(0)
        replay.add(
            self.observations,
            first_slot * OBSERVATION_SIZE,
            Int(self.actions[first_slot]),
            Int(self.target_columns[first_slot]),
            Int(self.target_rows[first_slot]),
            reward,
            self.next_observations,
            last_slot * OBSERVATION_SIZE,
            Float32(0.0) if boundary else discount,
            self.terminated[last_slot],
            self.truncated[last_slot],
            horizon,
        )
        for offset in range(1, length):
            var source_slot = lane * N_STEP + offset
            var destination_slot = source_slot - 1
            for feature in range(OBSERVATION_SIZE):
                self.observations[
                    destination_slot * OBSERVATION_SIZE + feature
                ] = self.observations[source_slot * OBSERVATION_SIZE + feature]
                self.next_observations[
                    destination_slot * OBSERVATION_SIZE + feature
                ] = self.next_observations[
                    source_slot * OBSERVATION_SIZE + feature
                ]
            self.actions[destination_slot] = self.actions[source_slot]
            self.target_columns[destination_slot] = self.target_columns[
                source_slot
            ]
            self.target_rows[destination_slot] = self.target_rows[source_slot]
            self.rewards[destination_slot] = self.rewards[source_slot]
            self.terminated[destination_slot] = self.terminated[source_slot]
            self.truncated[destination_slot] = self.truncated[source_slot]
        self.queue_length[lane] = length - 1


def nearest_axis(value: Float32, spacing: Int) -> Int:
    var point_count = ((125 - 2) // spacing) + 1
    var last_point = 2 + (point_count - 1) * spacing
    if last_point != 125:
        point_count += 1
    var best = 0
    var best_distance = Float32(1_000_000.0)
    for index in range(point_count):
        var point = 2 + index * spacing
        if index == point_count - 1:
            point = 125
        var distance = abs(value - Float32(point))
        if distance < best_distance:
            best = index
            best_distance = distance
    return best


def target_cell_for_action(
    x: Float32, y: Float32, action: Int, spacing: Int
) -> Tuple[Int, Int]:
    var column = nearest_axis(x, spacing)
    var row = nearest_axis(y, spacing)
    var point_count = ((125 - 2) // spacing) + 1
    var last_point = 2 + (point_count - 1) * spacing
    if last_point != 125:
        point_count += 1
    var horizontal = 0
    var vertical = 0
    if action == 1 or action == 5 or action == 7:
        horizontal = -1
    elif action == 2 or action == 6 or action == 8:
        horizontal = 1
    if action == 3 or action == 5 or action == 6:
        vertical = -1
    elif action == 4 or action == 7 or action == 8:
        vertical = 1
    column = min(point_count - 1, max(0, column + horizontal))
    row = min(point_count - 1, max(0, row + vertical))
    return column, row


def native_action_for_position(
    x: Float32, y: Float32, target_column: Int, target_row: Int, spacing: Int
) -> UInt8:
    var point_count = ((125 - 2) // spacing) + 1
    var last_point = 2 + (point_count - 1) * spacing
    if last_point != 125:
        point_count += 1
    var target_x = Float32(2 + target_column * spacing)
    var target_y = Float32(2 + target_row * spacing)
    if target_column == point_count - 1:
        target_x = Float32(125.0)
    if target_row == point_count - 1:
        target_y = Float32(125.0)
    var horizontal = 0
    if target_x - x > Float32(2.0):
        horizontal = 1
    elif target_x - x < Float32(-2.0):
        horizontal = -1
    var vertical = 0
    if target_y - y > Float32(2.0):
        vertical = 1
    elif target_y - y < Float32(-2.0):
        vertical = -1
    if horizontal == 0 and vertical == 0:
        return UInt8(0)
    if horizontal == -1 and vertical == 0:
        return UInt8(1)
    if horizontal == 1 and vertical == 0:
        return UInt8(2)
    if horizontal == 0 and vertical == -1:
        return UInt8(3)
    if horizontal == 0 and vertical == 1:
        return UInt8(4)
    if horizontal == -1 and vertical == -1:
        return UInt8(5)
    if horizontal == 1 and vertical == -1:
        return UInt8(6)
    if horizontal == -1 and vertical == 1:
        return UInt8(7)
    return UInt8(8)


struct ForwardScratch:
    var z1: List[Float32]
    var normalized: List[Float32]
    var h1: List[Float32]
    var z2: List[Float32]
    var h2: List[Float32]
    var q: List[Float32]
    var inverse_std: Float32

    def __init__(out self):
        self.z1 = List[Float32](length=HIDDEN_SIZE, fill=Float32(0.0))
        self.normalized = List[Float32](length=HIDDEN_SIZE, fill=Float32(0.0))
        self.h1 = List[Float32](length=HIDDEN_SIZE, fill=Float32(0.0))
        self.z2 = List[Float32](length=HIDDEN_SIZE, fill=Float32(0.0))
        self.h2 = List[Float32](length=HIDDEN_SIZE, fill=Float32(0.0))
        self.q = List[Float32](length=ACTION_COUNT, fill=Float32(0.0))
        self.inverse_std = Float32(1.0)


struct BackwardScratch:
    var d_h2: List[Float32]
    var d_z2: List[Float32]
    var d_h1: List[Float32]
    var d_affine1: List[Float32]
    var d_z1: List[Float32]
    var d_advantage: List[Float32]

    def __init__(out self):
        self.d_h2 = List[Float32](length=HIDDEN_SIZE, fill=Float32(0.0))
        self.d_z2 = List[Float32](length=HIDDEN_SIZE, fill=Float32(0.0))
        self.d_h1 = List[Float32](length=HIDDEN_SIZE, fill=Float32(0.0))
        self.d_affine1 = List[Float32](length=HIDDEN_SIZE, fill=Float32(0.0))
        self.d_z1 = List[Float32](length=HIDDEN_SIZE, fill=Float32(0.0))
        self.d_advantage = List[Float32](length=ACTION_COUNT, fill=Float32(0.0))


struct LearningWorkspace:
    var current: List[Float32]
    var next: List[Float32]
    var online_scratch: ForwardScratch
    var next_scratch: ForwardScratch
    var target_scratch: ForwardScratch
    var backward: BackwardScratch

    def __init__(out self):
        self.current = List[Float32](length=OBSERVATION_SIZE, fill=Float32(0.0))
        self.next = List[Float32](length=OBSERVATION_SIZE, fill=Float32(0.0))
        self.online_scratch = ForwardScratch()
        self.next_scratch = ForwardScratch()
        self.target_scratch = ForwardScratch()
        self.backward = BackwardScratch()


struct LearningStats:
    var loss: Float32
    var q_mean: Float32
    var target_mean: Float32
    var td_error: Float32
    var gradient_norm: Float32

    def __init__(
        out self,
        loss: Float32,
        q_mean: Float32,
        target_mean: Float32,
        td_error: Float32,
        gradient_norm: Float32,
    ):
        self.loss = loss
        self.q_mean = q_mean
        self.target_mean = target_mean
        self.td_error = td_error
        self.gradient_norm = gradient_norm


struct DuelingNetwork:
    """Dueling MLP with the Python model's LayerNorm/ReLU topology."""

    var w1: List[Float32]
    var b1: List[Float32]
    var layer_norm_weight: List[Float32]
    var layer_norm_bias: List[Float32]
    var w2: List[Float32]
    var b2: List[Float32]
    var value_weight: List[Float32]
    var value_bias: Float32
    var advantage_weight: List[Float32]
    var advantage_bias: List[Float32]

    var gw1: List[Float32]
    var gb1: List[Float32]
    var glayer_norm_weight: List[Float32]
    var glayer_norm_bias: List[Float32]
    var gw2: List[Float32]
    var gb2: List[Float32]
    var gvalue_weight: List[Float32]
    var gvalue_bias: Float32
    var gadvantage_weight: List[Float32]
    var gadvantage_bias: List[Float32]

    var mw1: List[Float32]
    var mb1: List[Float32]
    var mlayer_norm_weight: List[Float32]
    var mlayer_norm_bias: List[Float32]
    var mw2: List[Float32]
    var mb2: List[Float32]
    var mvalue_weight: List[Float32]
    var mvalue_bias: Float32
    var madvantage_weight: List[Float32]
    var madvantage_bias: List[Float32]

    var vw1: List[Float32]
    var vb1: List[Float32]
    var vlayer_norm_weight: List[Float32]
    var vlayer_norm_bias: List[Float32]
    var vw2: List[Float32]
    var vb2: List[Float32]
    var vvalue_weight: List[Float32]
    var vvalue_bias: Float32
    var vadvantage_weight: List[Float32]
    var vadvantage_bias: List[Float32]

    var optimizer_step: Int
    var beta1_power: Float32
    var beta2_power: Float32

    def __init__(out self, seed: UInt64):
        self.w1 = List[Float32](
            length=HIDDEN_SIZE * OBSERVATION_SIZE, fill=Float32(0.0)
        )
        self.b1 = List[Float32](length=HIDDEN_SIZE, fill=Float32(0.0))
        self.layer_norm_weight = List[Float32](
            length=HIDDEN_SIZE, fill=Float32(1.0)
        )
        self.layer_norm_bias = List[Float32](
            length=HIDDEN_SIZE, fill=Float32(0.0)
        )
        self.w2 = List[Float32](
            length=HIDDEN_SIZE * HIDDEN_SIZE, fill=Float32(0.0)
        )
        self.b2 = List[Float32](length=HIDDEN_SIZE, fill=Float32(0.0))
        self.value_weight = List[Float32](length=HIDDEN_SIZE, fill=Float32(0.0))
        self.value_bias = Float32(0.0)
        self.advantage_weight = List[Float32](
            length=ACTION_COUNT * HIDDEN_SIZE, fill=Float32(0.0)
        )
        self.advantage_bias = List[Float32](
            length=ACTION_COUNT, fill=Float32(0.0)
        )

        self.gw1 = List[Float32](length=len(self.w1), fill=Float32(0.0))
        self.gb1 = List[Float32](length=HIDDEN_SIZE, fill=Float32(0.0))
        self.glayer_norm_weight = List[Float32](
            length=HIDDEN_SIZE, fill=Float32(0.0)
        )
        self.glayer_norm_bias = List[Float32](
            length=HIDDEN_SIZE, fill=Float32(0.0)
        )
        self.gw2 = List[Float32](length=len(self.w2), fill=Float32(0.0))
        self.gb2 = List[Float32](length=HIDDEN_SIZE, fill=Float32(0.0))
        self.gvalue_weight = List[Float32](
            length=HIDDEN_SIZE, fill=Float32(0.0)
        )
        self.gvalue_bias = Float32(0.0)
        self.gadvantage_weight = List[Float32](
            length=len(self.advantage_weight), fill=Float32(0.0)
        )
        self.gadvantage_bias = List[Float32](
            length=ACTION_COUNT, fill=Float32(0.0)
        )

        self.mw1 = List[Float32](length=len(self.w1), fill=Float32(0.0))
        self.mb1 = List[Float32](length=HIDDEN_SIZE, fill=Float32(0.0))
        self.mlayer_norm_weight = List[Float32](
            length=HIDDEN_SIZE, fill=Float32(0.0)
        )
        self.mlayer_norm_bias = List[Float32](
            length=HIDDEN_SIZE, fill=Float32(0.0)
        )
        self.mw2 = List[Float32](length=len(self.w2), fill=Float32(0.0))
        self.mb2 = List[Float32](length=HIDDEN_SIZE, fill=Float32(0.0))
        self.mvalue_weight = List[Float32](
            length=HIDDEN_SIZE, fill=Float32(0.0)
        )
        self.mvalue_bias = Float32(0.0)
        self.madvantage_weight = List[Float32](
            length=len(self.advantage_weight), fill=Float32(0.0)
        )
        self.madvantage_bias = List[Float32](
            length=ACTION_COUNT, fill=Float32(0.0)
        )

        self.vw1 = List[Float32](length=len(self.w1), fill=Float32(0.0))
        self.vb1 = List[Float32](length=HIDDEN_SIZE, fill=Float32(0.0))
        self.vlayer_norm_weight = List[Float32](
            length=HIDDEN_SIZE, fill=Float32(0.0)
        )
        self.vlayer_norm_bias = List[Float32](
            length=HIDDEN_SIZE, fill=Float32(0.0)
        )
        self.vw2 = List[Float32](length=len(self.w2), fill=Float32(0.0))
        self.vb2 = List[Float32](length=HIDDEN_SIZE, fill=Float32(0.0))
        self.vvalue_weight = List[Float32](
            length=HIDDEN_SIZE, fill=Float32(0.0)
        )
        self.vvalue_bias = Float32(0.0)
        self.vadvantage_weight = List[Float32](
            length=len(self.advantage_weight), fill=Float32(0.0)
        )
        self.vadvantage_bias = List[Float32](
            length=ACTION_COUNT, fill=Float32(0.0)
        )

        self.optimizer_step = 0
        self.beta1_power = Float32(1.0)
        self.beta2_power = Float32(1.0)

        var rng = FastRng(seed)
        var first_scale = sqrt(
            Float32(6.0) / Float32(OBSERVATION_SIZE + HIDDEN_SIZE)
        )
        var second_scale = sqrt(Float32(6.0) / Float32(2 * HIDDEN_SIZE))
        var head_scale = sqrt(
            Float32(6.0) / Float32(HIDDEN_SIZE + ACTION_COUNT)
        )
        for index in range(len(self.w1)):
            self.w1[index] = (
                rng.uniform() * Float32(2.0) - Float32(1.0)
            ) * first_scale
        for index in range(len(self.w2)):
            self.w2[index] = (
                rng.uniform() * Float32(2.0) - Float32(1.0)
            ) * second_scale
        for index in range(len(self.value_weight)):
            self.value_weight[index] = (
                rng.uniform() * Float32(2.0) - Float32(1.0)
            ) * head_scale
        for index in range(len(self.advantage_weight)):
            self.advantage_weight[index] = (
                rng.uniform() * Float32(2.0) - Float32(1.0)
            ) * head_scale

    def copy_parameters_from(mut self, source: DuelingNetwork):
        for index in range(len(self.w1)):
            self.w1[index] = source.w1[index]
        for index in range(len(self.w2)):
            self.w2[index] = source.w2[index]
        for index in range(len(self.b1)):
            self.b1[index] = source.b1[index]
            self.layer_norm_weight[index] = source.layer_norm_weight[index]
            self.layer_norm_bias[index] = source.layer_norm_bias[index]
            self.b2[index] = source.b2[index]
            self.value_weight[index] = source.value_weight[index]
        for index in range(len(self.advantage_weight)):
            self.advantage_weight[index] = source.advantage_weight[index]
        for index in range(ACTION_COUNT):
            self.advantage_bias[index] = source.advantage_bias[index]
        self.value_bias = source.value_bias

    def forward(
        self,
        observations: List[Float32],
        input_offset: Int,
        mut scratch: ForwardScratch,
    ):
        var mean = Float32(0.0)
        for hidden in range(HIDDEN_SIZE):
            var total = self.b1[hidden]
            var weight_offset = hidden * OBSERVATION_SIZE
            total += simd_dot(
                self.w1,
                weight_offset,
                observations,
                input_offset,
                OBSERVATION_SIZE,
            )
            scratch.z1[hidden] = total
            mean += total
        mean /= Float32(HIDDEN_SIZE)
        var variance = Float32(0.0)
        for hidden in range(HIDDEN_SIZE):
            var difference = scratch.z1[hidden] - mean
            variance += difference * difference
        variance /= Float32(HIDDEN_SIZE)
        scratch.inverse_std = Float32(1.0) / sqrt(variance + Float32(1e-5))
        for hidden in range(HIDDEN_SIZE):
            var normalized = (scratch.z1[hidden] - mean) * scratch.inverse_std
            scratch.normalized[hidden] = normalized
            var activated = (
                self.layer_norm_weight[hidden] * normalized
                + self.layer_norm_bias[hidden]
            )
            scratch.h1[hidden] = max(activated, Float32(0.0))
        for hidden in range(HIDDEN_SIZE):
            var total = self.b2[hidden]
            var weight_offset = hidden * HIDDEN_SIZE
            total += simd_dot(
                self.w2,
                weight_offset,
                scratch.h1,
                0,
                HIDDEN_SIZE,
            )
            scratch.z2[hidden] = total
            scratch.h2[hidden] = max(total, Float32(0.0))
        var value = self.value_bias
        var advantage_mean = Float32(0.0)
        value += simd_dot(self.value_weight, 0, scratch.h2, 0, HIDDEN_SIZE)
        for action in range(ACTION_COUNT):
            var advantage = self.advantage_bias[action]
            var weight_offset = action * HIDDEN_SIZE
            advantage += simd_dot(
                self.advantage_weight,
                weight_offset,
                scratch.h2,
                0,
                HIDDEN_SIZE,
            )
            scratch.q[action] = advantage
            advantage_mean += advantage
        advantage_mean /= Float32(ACTION_COUNT)
        for action in range(ACTION_COUNT):
            scratch.q[action] = value + scratch.q[action] - advantage_mean

    def zero_grad(mut self):
        for index in range(len(self.gw1)):
            self.gw1[index] = Float32(0.0)
        for index in range(len(self.gb1)):
            self.gb1[index] = Float32(0.0)
            self.glayer_norm_weight[index] = Float32(0.0)
            self.glayer_norm_bias[index] = Float32(0.0)
            self.gb2[index] = Float32(0.0)
            self.gvalue_weight[index] = Float32(0.0)
        for index in range(len(self.gw2)):
            self.gw2[index] = Float32(0.0)
        for index in range(len(self.gadvantage_weight)):
            self.gadvantage_weight[index] = Float32(0.0)
        for index in range(ACTION_COUNT):
            self.gadvantage_bias[index] = Float32(0.0)
        self.gvalue_bias = Float32(0.0)

    def accumulate_gradient(
        mut self,
        observations: List[Float32],
        input_offset: Int,
        action: Int,
        td_gradient: Float32,
        scratch: ForwardScratch,
        mut backward: BackwardScratch,
    ):
        var action_scale = Float32(1.0) / Float32(ACTION_COUNT)
        for hidden in range(HIDDEN_SIZE):
            backward.d_h1[hidden] = Float32(0.0)
        for candidate in range(ACTION_COUNT):
            var multiplier = -action_scale
            if candidate == action:
                multiplier += Float32(1.0)
            backward.d_advantage[candidate] = td_gradient * multiplier
            self.gadvantage_bias[candidate] += backward.d_advantage[candidate]
        self.gvalue_bias += td_gradient
        for hidden in range(HIDDEN_SIZE):
            self.gvalue_weight[hidden] += td_gradient * scratch.h2[hidden]
        for candidate in range(ACTION_COUNT):
            var gradient = backward.d_advantage[candidate]
            var weight_offset = candidate * HIDDEN_SIZE
            for hidden in range(HIDDEN_SIZE):
                self.gadvantage_weight[weight_offset + hidden] += (
                    gradient * scratch.h2[hidden]
                )
        for hidden in range(HIDDEN_SIZE):
            var gradient = td_gradient * self.value_weight[hidden]
            for candidate in range(ACTION_COUNT):
                gradient += (
                    backward.d_advantage[candidate]
                    * self.advantage_weight[candidate * HIDDEN_SIZE + hidden]
                )
            backward.d_h2[hidden] = gradient
            backward.d_z2[hidden] = gradient if scratch.z2[
                hidden
            ] > 0 else Float32(0.0)
        for hidden in range(HIDDEN_SIZE):
            var gradient = backward.d_z2[hidden]
            self.gb2[hidden] += gradient
            var weight_offset = hidden * HIDDEN_SIZE
            for previous in range(HIDDEN_SIZE):
                self.gw2[weight_offset + previous] += (
                    gradient * scratch.h1[previous]
                )
                backward.d_h1[previous] += (
                    gradient * self.w2[weight_offset + previous]
                )
        for hidden in range(HIDDEN_SIZE):
            backward.d_affine1[hidden] = backward.d_h1[hidden] if scratch.h1[
                hidden
            ] > 0 else Float32(0.0)
        var mean_u = Float32(0.0)
        var mean_u_normalized = Float32(0.0)
        for hidden in range(HIDDEN_SIZE):
            var weighted = (
                backward.d_affine1[hidden] * self.layer_norm_weight[hidden]
            )
            mean_u += weighted
            mean_u_normalized += weighted * scratch.normalized[hidden]
        mean_u /= Float32(HIDDEN_SIZE)
        mean_u_normalized /= Float32(HIDDEN_SIZE)
        for hidden in range(HIDDEN_SIZE):
            var affine_gradient = backward.d_affine1[hidden]
            self.glayer_norm_weight[hidden] += (
                affine_gradient * scratch.normalized[hidden]
            )
            self.glayer_norm_bias[hidden] += affine_gradient
            var weighted = affine_gradient * self.layer_norm_weight[hidden]
            backward.d_z1[hidden] = scratch.inverse_std * (
                weighted
                - mean_u
                - scratch.normalized[hidden] * mean_u_normalized
            )
            self.gb1[hidden] += backward.d_z1[hidden]
            var weight_offset = hidden * OBSERVATION_SIZE
            for feature in range(OBSERVATION_SIZE):
                self.gw1[weight_offset + feature] += (
                    backward.d_z1[hidden] * observations[input_offset + feature]
                )

    def apply_adamw(mut self, batch_size: Int) -> Float32:
        var scale = Float32(1.0) / Float32(batch_size)
        var squared_norm = Float32(0.0)
        for index in range(len(self.gw1)):
            self.gw1[index] *= scale
            squared_norm += self.gw1[index] * self.gw1[index]
        for index in range(len(self.gb1)):
            self.gb1[index] *= scale
            self.glayer_norm_weight[index] *= scale
            self.glayer_norm_bias[index] *= scale
            self.gb2[index] *= scale
            self.gvalue_weight[index] *= scale
            squared_norm += self.gb1[index] * self.gb1[index]
            squared_norm += (
                self.glayer_norm_weight[index] * self.glayer_norm_weight[index]
            )
            squared_norm += (
                self.glayer_norm_bias[index] * self.glayer_norm_bias[index]
            )
            squared_norm += self.gb2[index] * self.gb2[index]
            squared_norm += (
                self.gvalue_weight[index] * self.gvalue_weight[index]
            )
        for index in range(len(self.gw2)):
            self.gw2[index] *= scale
            squared_norm += self.gw2[index] * self.gw2[index]
        for index in range(len(self.gadvantage_weight)):
            self.gadvantage_weight[index] *= scale
            squared_norm += (
                self.gadvantage_weight[index] * self.gadvantage_weight[index]
            )
        for index in range(ACTION_COUNT):
            self.gadvantage_bias[index] *= scale
            squared_norm += (
                self.gadvantage_bias[index] * self.gadvantage_bias[index]
            )
        self.gvalue_bias *= scale
        squared_norm += self.gvalue_bias * self.gvalue_bias
        var gradient_norm = sqrt(squared_norm)
        var clip_scale = min(
            Float32(1.0), Float32(10.0) / max(gradient_norm, Float32(1e-12))
        )
        self.optimizer_step += 1
        self.beta1_power *= Float32(0.9)
        self.beta2_power *= Float32(0.999)
        var step_scale = LEARNING_RATE
        var first_correction = Float32(1.0) - self.beta1_power
        var second_correction = Float32(1.0) - self.beta2_power
        adamw_update(
            self.w1,
            self.gw1,
            self.mw1,
            self.vw1,
            clip_scale,
            step_scale,
            first_correction,
            second_correction,
        )
        adamw_update(
            self.b1,
            self.gb1,
            self.mb1,
            self.vb1,
            clip_scale,
            step_scale,
            first_correction,
            second_correction,
        )
        adamw_update(
            self.layer_norm_weight,
            self.glayer_norm_weight,
            self.mlayer_norm_weight,
            self.vlayer_norm_weight,
            clip_scale,
            step_scale,
            first_correction,
            second_correction,
        )
        adamw_update(
            self.layer_norm_bias,
            self.glayer_norm_bias,
            self.mlayer_norm_bias,
            self.vlayer_norm_bias,
            clip_scale,
            step_scale,
            first_correction,
            second_correction,
        )
        adamw_update(
            self.w2,
            self.gw2,
            self.mw2,
            self.vw2,
            clip_scale,
            step_scale,
            first_correction,
            second_correction,
        )
        adamw_update(
            self.b2,
            self.gb2,
            self.mb2,
            self.vb2,
            clip_scale,
            step_scale,
            first_correction,
            second_correction,
        )
        adamw_update(
            self.value_weight,
            self.gvalue_weight,
            self.mvalue_weight,
            self.vvalue_weight,
            clip_scale,
            step_scale,
            first_correction,
            second_correction,
        )
        adamw_update(
            self.advantage_weight,
            self.gadvantage_weight,
            self.madvantage_weight,
            self.vadvantage_weight,
            clip_scale,
            step_scale,
            first_correction,
            second_correction,
        )
        adamw_update(
            self.advantage_bias,
            self.gadvantage_bias,
            self.madvantage_bias,
            self.vadvantage_bias,
            clip_scale,
            step_scale,
            first_correction,
            second_correction,
        )
        var value_gradient = self.gvalue_bias * clip_scale
        self.mvalue_bias = (
            Float32(0.9) * self.mvalue_bias + Float32(0.1) * value_gradient
        )
        self.vvalue_bias = (
            Float32(0.999) * self.vvalue_bias
            + Float32(0.001) * value_gradient * value_gradient
        )
        var corrected_mean = self.mvalue_bias / first_correction
        var corrected_variance = self.vvalue_bias / second_correction
        self.value_bias = self.value_bias * (
            Float32(1.0) - LEARNING_RATE * WEIGHT_DECAY
        ) - LEARNING_RATE * corrected_mean / (
            sqrt(corrected_variance) + Float32(1e-8)
        )
        return gradient_norm

    def learn(
        mut self,
        target: DuelingNetwork,
        mut replay: ReplayBuffer,
        mut rng: FastRng,
        batch_size: Int,
        mut workspace: LearningWorkspace,
    ) -> LearningStats:
        self.zero_grad()
        var loss = Float32(0.0)
        var q_mean = Float32(0.0)
        var target_mean = Float32(0.0)
        var td_error = Float32(0.0)
        for _ in range(batch_size):
            var index = rng.below(replay.size)
            replay.copy_observation(index, workspace.current)
            replay.copy_next_observation(index, workspace.next)
            self.forward(workspace.current, 0, workspace.online_scratch)
            var action = Int(replay.actions[index])
            var q_value = workspace.online_scratch.q[action]
            self.forward(workspace.next, 0, workspace.next_scratch)
            var next_action = argmax(workspace.next_scratch.q)
            target.forward(workspace.next, 0, workspace.target_scratch)
            var target_value = (
                replay.rewards[index]
                + replay.discounts[index]
                * workspace.target_scratch.q[next_action]
            )
            var difference = q_value - target_value
            var smooth_gradient = difference if abs(difference) <= Float32(
                1.0
            ) else (Float32(1.0) if difference > 0 else Float32(-1.0))
            self.accumulate_gradient(
                workspace.current,
                0,
                action,
                smooth_gradient,
                workspace.online_scratch,
                workspace.backward,
            )
            if abs(difference) <= Float32(1.0):
                loss += Float32(0.5) * difference * difference
            else:
                loss += abs(difference) - Float32(0.5)
            q_mean += q_value
            target_mean += target_value
            td_error += abs(difference)
        var gradient_norm = self.apply_adamw(batch_size)
        var divisor = Float32(batch_size)
        return LearningStats(
            loss / divisor,
            q_mean / divisor,
            target_mean / divisor,
            td_error / divisor,
            gradient_norm,
        )


def adamw_update(
    mut parameter: List[Float32],
    gradient: List[Float32],
    mut first_moment: List[Float32],
    mut second_moment: List[Float32],
    clip_scale: Float32,
    learning_rate: Float32,
    first_correction: Float32,
    second_correction: Float32,
):
    for index in range(len(parameter)):
        var gradient_value = gradient[index] * clip_scale
        first_moment[index] = (
            Float32(0.9) * first_moment[index] + Float32(0.1) * gradient_value
        )
        second_moment[index] = (
            Float32(0.999) * second_moment[index]
            + Float32(0.001) * gradient_value * gradient_value
        )
        var corrected_mean = first_moment[index] / first_correction
        var corrected_variance = second_moment[index] / second_correction
        parameter[index] = parameter[index] * (
            Float32(1.0) - learning_rate * WEIGHT_DECAY
        ) - learning_rate * corrected_mean / (
            sqrt(corrected_variance) + Float32(1e-8)
        )


def argmax(values: List[Float32]) -> Int:
    var best = 0
    var best_value = values[0]
    for index in range(1, len(values)):
        if values[index] > best_value:
            best = index
            best_value = values[index]
    return best


def simd_dot(
    left: List[Float32],
    left_offset: Int,
    right: List[Float32],
    right_offset: Int,
    length: Int,
) -> Float32:
    var total = Float32(0.0)
    var index = 0
    while index + SIMD_WIDTH <= length:
        var left_vector = SIMD[DType.float32, SIMD_WIDTH]()
        var right_vector = SIMD[DType.float32, SIMD_WIDTH]()
        for lane in range(SIMD_WIDTH):
            left_vector[lane] = left[left_offset + index + lane]
            right_vector[lane] = right[right_offset + index + lane]
        total += (left_vector * right_vector).reduce_add()
        index += SIMD_WIDTH
    while index < length:
        total += left[left_offset + index] * right[right_offset + index]
        index += 1
    return total


def copy_lane(
    source_observations: List[Float32],
    source_positions: List[Float32],
    lane: Int,
    mut observations: List[Float32],
    mut positions: List[Float32],
    destination_lane: Int,
):
    var source_observation_offset = lane * OBSERVATION_SIZE
    var destination_observation_offset = destination_lane * OBSERVATION_SIZE
    for feature in range(OBSERVATION_SIZE):
        observations[
            destination_observation_offset + feature
        ] = source_observations[source_observation_offset + feature]
    positions[destination_lane * 2] = source_positions[lane * 2]
    positions[destination_lane * 2 + 1] = source_positions[lane * 2 + 1]


def fill_training_seeds() -> List[UInt32]:
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


def epsilon_for_step(step: Int, total_steps: Int) -> Float32:
    var decay_steps = max(total_steps // 2, 1)
    var progress = min(Float32(1.0), Float32(step) / Float32(decay_steps))
    return Float32(1.0) + progress * Float32(-0.95)


def collect_macro_transition(
    mut environment: NativeBatch,
    mut current_observations: List[Float32],
    mut current_positions: List[Float32],
    mut episode_steps: List[Int],
    mut online: DuelingNetwork,
    mut action_scratch: ForwardScratch,
    mut accumulator: NStepAccumulator,
    mut replay: ReplayBuffer,
    mut rng: FastRng,
    training_seeds: List[UInt32],
    mut seed_cursor: Int,
    total_steps: Int,
    global_step: Int,
    hold_decisions: Int,
    max_episode_steps: Int,
    grid_spacing: Int,
    mut decision_observations: List[Float32],
    mut next_observations: List[Float32],
    mut waypoint_actions: List[UInt8],
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
    var epsilon = epsilon_for_step(global_step, total_steps)
    for lane in range(environment.lane_count):
        var observation_offset = lane * OBSERVATION_SIZE
        for feature in range(OBSERVATION_SIZE):
            decision_observations[
                observation_offset + feature
            ] = current_observations[observation_offset + feature]
        online.forward(
            decision_observations, observation_offset, action_scratch
        )
        var action = argmax(action_scratch.q)
        if rng.uniform() < epsilon:
            action = rng.below(ACTION_COUNT)
        waypoint_actions[lane] = UInt8(action)
        var cells = target_cell_for_action(
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
                native_actions[lane] = native_action_for_position(
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
                    copy_lane(
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
                var observation_offset = lane * OBSERVATION_SIZE
                for feature in range(OBSERVATION_SIZE):
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
                copy_lane(
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
                copy_lane(
                    environment.observations,
                    environment.positions,
                    lane,
                    current_observations,
                    current_positions,
                    lane,
                )
                episode_steps[lane] = 0

    for lane in range(environment.lane_count):
        var observation_offset = lane * OBSERVATION_SIZE
        var next_offset = observation_offset
        if boundary[lane] != UInt8(0):
            next_offset = lane * OBSERVATION_SIZE
            for feature in range(OBSERVATION_SIZE):
                next_observations[next_offset + feature] = next_observations[
                    observation_offset + feature
                ]
        else:
            for feature in range(OBSERVATION_SIZE):
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


def parse_string_argument(
    args: Span[StringSpan[ImmStaticOrigin], ...], name: String, default: String
) -> String:
    for index in range(1, len(args) - 1):
        if args[index] == name:
            return args[index + 1]
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
) raises:
    var environment = NativeBatch(
        library_path, lanes, step_frames, grid_spacing, parallel
    )
    var online = DuelingNetwork(seed)
    var target = DuelingNetwork(seed)
    target.copy_parameters_from(online)
    var replay = ReplayBuffer(DEFAULT_REPLAY_CAPACITY)
    var accumulator = NStepAccumulator(lanes)
    var rng = FastRng(seed ^ UInt64(0x9E3779B97F4A7C15))
    var training_seeds = fill_training_seeds()
    var initial_seeds = List[UInt32](length=lanes, fill=UInt32(0))
    for lane in range(lanes):
        initial_seeds[lane] = training_seeds[lane % len(training_seeds)]
    environment.reset(initial_seeds)
    var current_observations = List[Float32](
        length=lanes * OBSERVATION_SIZE, fill=Float32(0.0)
    )
    var current_positions = List[Float32](length=lanes * 2, fill=Float32(0.0))
    for lane in range(lanes):
        copy_lane(
            environment.observations,
            environment.positions,
            lane,
            current_observations,
            current_positions,
            lane,
        )
    var episode_steps = List[Int](length=lanes, fill=0)
    var decision_observations = List[Float32](
        length=lanes * OBSERVATION_SIZE, fill=Float32(0.0)
    )
    var next_observations = List[Float32](
        length=lanes * OBSERVATION_SIZE, fill=Float32(0.0)
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
    var seed_cursor = 0
    var native_steps = 0
    var learner_updates = 0
    var action_scratch = ForwardScratch()
    var learning_workspace = LearningWorkspace()
    var last_stats = LearningStats(
        Float32(0.0), Float32(0.0), Float32(0.0), Float32(0.0), Float32(0.0)
    )
    var started = perf_counter_ns()
    for step in range(total_steps):
        var result = collect_macro_transition(
            environment,
            current_observations,
            current_positions,
            episode_steps,
            online,
            action_scratch,
            accumulator,
            replay,
            rng,
            training_seeds,
            seed_cursor,
            total_steps,
            step,
            hold_decisions,
            max_episode_steps,
            grid_spacing,
            decision_observations,
            next_observations,
            waypoint_actions,
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
            last_stats = online.learn(
                target, replay, rng, batch_size, learning_workspace
            )
            learner_updates += 1
        if completed_step % 1000 == 0:
            target.copy_parameters_from(online)
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
        "result lanes=",
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
    environment.close()


def main() raises:
    var args = argv()
    var library_path = parse_string_argument(
        args,
        "--ffi",
        "mojo/ffi/target/release/libdodge_mojo_ffi.so",
    )
    var lanes = parse_int_argument(args, "--lanes", DEFAULT_LANES)
    var total_steps = parse_int_argument(args, "--steps", DEFAULT_TOTAL_STEPS)
    var batch_size = parse_int_argument(
        args, "--batch-size", DEFAULT_BATCH_SIZE
    )
    var warmup_steps = parse_int_argument(
        args, "--warmup", DEFAULT_WARMUP_STEPS
    )
    var hold_decisions = parse_int_argument(
        args, "--hold-decisions", DEFAULT_HOLD_DECISIONS
    )
    var step_frames = parse_int_argument(
        args, "--step-frames", DEFAULT_STEP_FRAMES
    )
    var grid_spacing = parse_int_argument(
        args, "--grid-spacing", DEFAULT_GRID_SPACING
    )
    var max_episode_steps = parse_int_argument(
        args, "--max-episode-steps", DEFAULT_MAX_EPISODE_STEPS
    )
    var seed = UInt64(parse_int_argument(args, "--seed", 2_026_0903))
    var learn_enabled = not has_argument(args, "--no-learning")
    var parallel = not has_argument(args, "--serial")
    if lanes < 1 or total_steps < 1 or batch_size < 1 or warmup_steps < 1:
        raise Error("lane, step, batch, and warmup counts must be positive")
    print(
        "mojo-waypoint-dqn lanes=",
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
    )
