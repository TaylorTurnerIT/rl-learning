# Independent Mojo waypoint-DQN investigation

This directory is a separate experiment. The Python learner and its existing
Rust/PyO3 boundary are unchanged.

## What is ported

`src/dqn.mojo` reproduces the hot training loop used by the waypoint DQN:

- 225-float native ML observations and player positions;
- nine relative waypoint actions and sign-based native steering;
- 32-lane native collection with eight decisions held per waypoint action;
- fixed uniform replay storage;
- per-lane three-step returns with zero bootstrap at episode boundaries;
- dueling Double DQN;
- `Linear(225, 256)`, `LayerNorm(256)`, ReLU, `Linear(256, 256)`, ReLU;
- Smooth L1 loss, max-norm-10 gradient clipping, and AdamW.

The current benchmark path leaves checkpointing, dashboard telemetry, final
evaluation, and report generation out of the loop. It measures the learner
core, not a drop-in replacement for the Python campaign command.

This is logic parity, not bit-for-bit parity: Mojo uses its own xorshift RNG and
weight initialization, so action trajectories and learned weights are not
expected to match PyTorch exactly.

`ffi/` is a narrow C ABI over the existing Rust `dodge-batch` simulator. Mojo
does not import Python, call CPython, or link the PyO3 extension. The bridge
copies the native result buffers into Mojo lists, so this is an independent
learner investigation rather than a claim that the simulator itself has been
ported to Mojo.

## Hybrid path

`src/hybrid_dqn.mojo` keeps the native Mojo collection loop, waypoint
controller, replay buffer, and n-step accumulator, while
`python/torch_waypoint_learner.py` owns the dueling network, batched inference,
autograd, and AdamW. Mojo crosses into Python once per macro action batch and
once per learner update. Each update hands NumPy arrays to `torch.from_numpy`,
so the dense work stays in PyTorch's optimized kernels instead of the scalar
Mojo reference implementation. The hybrid defaults to two PyTorch CPU threads
for this small MLP; the count remains configurable for machine-specific
benchmarks.

The hybrid executable has a runtime dependency on the project Python
environment. `mojo build` does not bundle CPython, NumPy, or PyTorch; run it
with `MOJO_PYTHON` pointed at `.venv/bin/python`:

```bash
devenv -q shell -- /tmp/rl-learning-mojo-venv/bin/mojo build \
  -I mojo/src mojo/src/hybrid_dqn.mojo -o /tmp/rl-learning-mojo-hybrid

devenv -q shell -- env \
  MOJO_PYTHON="$PWD/.venv/bin/python" \
  /tmp/rl-learning-mojo-hybrid \
  --steps 128 --lanes 32 --batch-size 32 --warmup 16
```

The hybrid accepts the same training arguments plus `--torch-threads`,
`--validate-inputs`, and `--checkpoint PATH`. The Mojo boundary defaults to the
fast path because the typed collector owns shape and finite-value guarantees;
`--validate-inputs` enables redundant Python-side checks for diagnostics.

After a checkpointed run, evaluate it against the frozen manifest without
touching the Python campaign:

```bash
devenv -q shell -- env PYTHONPATH="$PWD/src" \
  uv run python mojo/python/evaluate_hybrid.py \
  --checkpoint history/dodge/ng/mojo-hybrid/checkpoint-final.pt \
  --split all
```

## Build and run

Mojo is kept outside the repository in this experiment. The current toolchain
was installed into `/tmp/rl-learning-mojo-venv`; use the official [Mojo
installation guide](https://docs.modular.com/mojo/manual/install) if that
environment is not present.

From the repository root:

```bash
devenv -q shell -- cargo build \
  --manifest-path mojo/ffi/Cargo.toml --release

devenv -q shell -- /tmp/rl-learning-mojo-venv/bin/mojo build \
  --target-cpu skylake mojo/src/dqn.mojo -o /tmp/rl-learning-mojo-dqn

/tmp/rl-learning-mojo-dqn \
  --steps 128 --lanes 32 --batch-size 32 --warmup 16
```

The executable accepts `--steps`, `--lanes`, `--batch-size`, `--warmup`,
`--hold-decisions`, `--step-frames`, `--grid-spacing`,
`--max-episode-steps`, `--seed`, `--ffi`, `--serial`, and `--no-learning`.
Defaults follow the current Python DQN hot-loop configuration, including
20,000 collection steps, 256-sample batches, 2,000 warmup steps, 32 lanes,
eight held decisions, four native frames, and a 32-pixel grid.

The small FFI probe can be built and run after the Rust library:

```bash
devenv -q shell -- /tmp/rl-learning-mojo-venv/bin/mojo build \
  mojo/src/ffi_probe.mojo -o /tmp/rl-learning-mojo-ffi-probe
/tmp/rl-learning-mojo-ffi-probe
```

## Initial timing

These are single-run measurements from 2026-09-04 on an Intel i7-10610U. The
Python process used four PyTorch CPU threads. Both paths used the same Rust
native ML simulator, 32 lanes, four native frames, eight held decisions, a
32-pixel grid, and a 200-step episode cap.

| Workload | Python | Mojo | Result |
|---|---:|---:|---:|
| 128 collection steps, no learner updates | 2.401 s | 2.130 s | Mojo 1.13x faster |
| 128 steps, batch 32, 113 learner updates | 2.842 s | 7.807 s | Mojo 2.75x slower |
| 32 steps, batch 256, 17 learner updates | 0.932 s | 6.848 s | Mojo 7.35x slower |

The first hybrid measurements, using the same native simulator and four
PyTorch CPU threads, were:

| Workload | Hybrid elapsed | Result |
|---|---:|---:|
| 128 collection steps, no learner updates | 0.284 s | 7.5x faster than Python |
| 128 steps, batch 32, 113 learner updates | 0.884 s | 3.2x faster than Python |
| 32 steps, batch 256, 17 learner updates | 0.166 s | 5.6x faster than Python |

Those are in-process loop timings and exclude roughly 2.5 seconds of Python
runtime startup. The startup cost is negligible for a full campaign. The
hybrid result is the useful comparison: Mojo removes the Python/native
environment-loop overhead, while PyTorch retains optimized batched dense
operations.

The pure-Mojo learner-inclusive result is still useful as a native-kernel
baseline: its network uses explicit CPU loops and a manual backward pass. The
hybrid path is not a bit-for-bit training replica because its PyTorch model
uses PyTorch initialization and RNG, but it preserves the same topology,
loss, target logic, optimizer settings, and interaction contract.

## Controlled side-by-side timing

`python/benchmark_python_hot_loop.py` invokes the current Python
`_collect_macro_transition` and `_learn_step` functions directly. This keeps
the comparison independent of checkpointing, dashboard telemetry, evaluation,
and report generation on both sides.

The final matched run used 20,000 collection steps, 32 lanes, batch size 256,
2,000 warmup steps, eight held decisions, four native frames, a 32-pixel grid,
a 2,000-step episode cap, and 18,001 learner updates. Both runs used manifest
`c75e2c8327888d3f2b31a3bb4681c0a537c8603161f8ffbf4a2539f32d2e01f6` and were
pinned to CPUs 4-6 under the same unrelated background load.

| Path | Loop time | Wall time |
|---|---:|---:|
| Current Python + PyTorch | 519.36 s | 527.43 s |
| Mojo collection + Python/PyTorch learner | 426.59 s | 430.87 s |
| Hybrid speedup | - | 1.22x |

The hybrid path was 96.56 seconds shorter wall-clock, or about 18.3% faster.
This is a runtime comparison, not a model-quality or bit-for-bit trajectory
comparison: the two implementations have different RNG, sampling, and
floating-point/control paths, so their checksums and training statistics differ.
The absolute times are also not idle-host estimates because the unrelated
20,000-update baseline process was left running untouched.

The pure Mojo implementation remains the fully native reference. The hybrid
path is the practical best-of-both-worlds candidate, but the current measured
gain for this matched hot loop is about 1.22x rather than the earlier
exploratory estimate.
