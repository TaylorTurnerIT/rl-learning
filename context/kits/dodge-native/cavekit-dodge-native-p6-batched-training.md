---
name: dodge-native-p6-batched-training
description: Replace per-decision Pemsa IPC with a batched Rust environment and preserve the current Python PPO/NEAT training contract
metadata:
  type: project-phase
  phase: P6
  created: "2026-09-01"
  last_edited: "2026-09-01"
---

# Cavekit: P6 Batched Training

## Scope

Expose P4's native core to the existing Python training stack through a
versioned binding and batch API. Keep the Python policy/trainer initially
compatible, but move environment stepping, state assembly, pixel generation,
and independent-environment parallelism into Rust. The first optimization
target is removing xdotool/X11/Pemsa/process/stdout/JSON overhead; the second
is making policy inference and environment stepping genuinely batched.

Depends on accepted P4. P5 is not a prerequisite.

## Phase contract

### §G GOAL

Native Rust batch environment → N independent Dodge lanes reset and stepped in
one binding call, returning full state, indexed pixels, reward, done, and
metadata with current PPO/NEAT action and seed semantics preserved.

### §C CONSTRAINTS

- No Pemsa, Xvfb, xdotool, window, subprocess, stdout protocol, or per-step
  JSON in the native training path.
- Macroquad is never initialized by training workers.
- Current action order, nine actions, step_frames, reward definition, seed
  partitions, held-out evaluation, and checkpoint/resume behavior remain
  compatible.
- Serial native and parallel native lanes must be differential-equivalent
  before parallelism becomes default.
- Full-state and pixel observations remain available; semantic 19 × 16 × 16
  board input is a derived compatibility view, not the canonical state.
- Buffer ownership and lifetime are explicit. Do not claim zero-copy unless
  measured and tested at the Python boundary.
- Rayon or another parallel iterator may parallelize independent lanes only;
  shared mutable RNG or hidden global game state is forbidden.
- Python fallback to existing DodgeEnv remains available for oracle and
  migration comparison.

### §I INTERFACES

native batch config → lane count, seed/config array, action horizon, observation
flags, and deterministic/parallel mode.

batch reset → contiguous lanes containing FullState, palette-index pixels,
reward zero, done false, seed, and lane metadata.

batch step → ordered Action array → one result per lane containing exact frames
advanced, FullState or view, pixels, reward, done, event flags, and hashes.

Rust batch observations own optional `FullState`/`RenderState`, indexed pixel
`[u8;128*128]`, canonical binary snapshot, and `Board19x16`; the Python
binding copies these into NumPy arrays with shapes `(N,128,128)` and
`(N,19,16,16)`.

Python environment → reset(seed), step(action), reset_batch(seeds),
reset_lanes(lanes, seeds), step_batch(actions), observe_full_state(),
observe_pixels(), and observe_board_19x16().

PPO backend → python fallback or native batch backend; backend selection and
observation mode recorded in run configuration/checkpoint.

benchmark record → workload, lane count, step_frames, observation payload,
repetitions, host, toolchain, statistic, throughput, and comparison baseline.

### §R RESEARCH

| id | topic | finding | source |
|----|-------|---------|--------|
| R1 | Rust parallelism | Namtao's suggested Rust library set includes Rayon for parallel iterators; use it only after serial lane equivalence is proven | https://www.namtao.com/rust/ |
| R2 | Serialization/config | Namtao suggests Serde for serialization; use it for config/traces, never for per-decision hot-path transport | https://www.namtao.com/rust/ |
| R3 | Benchmarks/tests | Namtao suggests Criterion and cargo-nextest; use workload-defined benchmarks plus focused tests | https://www.namtao.com/rust/ |
| R4 | Macroquad boundary | Macroquad's rendering features belong to a separate viewer process/crate; its API is not a training environment contract | https://docs.rs/macroquad/latest/macroquad/ |
| R5 | PyO3 binding | PyO3 0.29.2 supports a stable extension-module boundary and an ABI3 floor suitable for the project's Python `>=3.11` contract; the binding remains outside `dodge-core` | https://pyo3.rs/main/building-and-distribution |
| R6 | NumPy ownership | NumPy 0.29.0 provides PyO3-backed arrays and accepts owned `ndarray` buffers; the first binding copies native results into Python-owned arrays and therefore makes no zero-copy claim | https://docs.rs/numpy/0.29.0/numpy/ |
| R7 | Build workflow | Maturin 1.14.1 is available from the current Nixpkgs environment and will build/install the `dodge-python` `cdylib` into the project uv environment | https://www.maturin.rs/ |

### §V INVARIANTS

- V1: serial native batch lane result equals P4 serial result for the same
  seed, initial data, action schedule, and frame horizon.
- V2: parallel native batch result equals serial native batch result lane by
  lane, including full state, pixels, reward, done, events, and hashes.
- V3: batch output preserves input lane order and action order; no lane can
  consume another lane's RNG or input.
- V4: Python native backend and existing Python/Pemsa backend agree on the
  accepted trace corpus before backend replacement.
- V5: full-state, pixels, and derived 19 × 16 × 16 board observations have
  documented shapes, dtypes, ordering, and finite values.
- V6: native PPO preserves action ordering, step_frames, survival-frame reward,
  neutral bonus, seed exclusions, held-out evaluation, and checkpoint config.
- V7: native training path opens no display and spawns no per-decision child
  process; tests fail if forbidden boundary calls occur.
- V8: no per-decision JSON or text serialization occurs between Python and
  native batch environment.
- V9: observation buffers remain valid for their documented lifetime and cannot
  expose freed or concurrently mutated memory.
- V10: benchmark includes full-state + pixels and a pixels-off comparison; a
  fast benchmark that omits required observations cannot pass the phase.
- V11: target performance is median native batch throughput ≥10× median current
  Python/Pemsa interactive throughput for the same fixed workload, with five
  repetitions, on one named host and pinned toolchain. Failure opens an
  architecture decision record; it does not weaken parity.
- V12: training smoke run produces the same deterministic short-horizon
  trajectory and reward sequence under fallback and native backends.

### §T TASKS

| id | status | task | cites |
|----|--------|------|-------|
| P6-T1 | x | Select and research Python binding boundary; record PyO3/NumPy/maturin decision before coding | R1-R7 |
| P6-T2 | x | Add native batch crate/API over accepted P4 core with serial lane mode | V1,V3 |
| P6-T3 | x | Add typed full-state, pixel, and derived-board buffer views with shape/dtype tests | V5,V9 |
| P6-T4 | x | Add Python binding and native DodgeEnv compatibility adapter | V4,V6,V8 |
| P6-T5 | x | Add reset_batch/step_batch/selective reset and compare serial vs parallel lanes | V1,V2,V3 |
| P6-T6 | x | Add PPO native backend, preserve fallback path, and record backend/observation config | V6,V12 |
| P6-T7 | x | Add forbidden-IPC tests and short deterministic backend comparison | V4,V7,V8,V12 |
| P6-T8 | x | Run fixed throughput benchmark with full-state and pixels, then pixels-off control | V10,V11 |
| P6-T9 | x | Produce training acceptance report and hand benchmark/provenance data to P7 | V1-V12 |

### §B BUGS

| id | date | cause | fix |
|----|------|-------|-----|
| B1 | - | - | - |

## Gate

### Automated

- Serial and parallel native lanes match exactly.
- Native and fallback Python backends match on accepted traces and deterministic
  short-horizon PPO rollouts.
- Shapes, dtypes, ordering, and buffer-lifetime tests pass.
- Native training executes with no display, emulator, child process, or
  per-step JSON.
- Benchmark report contains the required workload/provenance and meets V11.
- Pixels-on and pixels-off results are both recorded; no hidden observation
  downgrade.

### Owner

Inspect a native batch frame and full state in the P5 viewer or a lossless
artifact. Review one PPO run configuration and confirm backend/observation
choice is visible and reproducible. Approve the measured speedup only after
confirming it includes the requested information exposure.

### Handoff

P6 is accepted when the training stack can use native batching without
semantic changes and the measured bottleneck is documented for P7 hardening.

## Cavekit routing

- grill: settle batch size, observation defaults, and Python ownership choices.
- research: PyO3/NumPy/maturin and buffer API facts before binding work.
- spec: add native backend, batch, buffer, and performance invariants.
- review: required for public ABI, memory ownership, and training semantics.
- build: execute P6 tasks only after P4 gate.
- check: audit backend parity, forbidden dependencies, and benchmark claims.
- backprop: classify mismatches separately from ABI/lifetime or benchmark
  harness defects.
- deepen: optimize one measured bottleneck after behavior and throughput gates.
