---
name: dodge-native-overview
description: Master target and dependency map for a source-faithful Rust Dodge runtime with exact indexed pixels, full state exposure, Macroquad replay, and batched reinforcement-learning execution
metadata:
  type: project
  created: "2026-09-01"
  last_edited: "2026-09-01"
---

# Cavekit Overview: Native Dodge Training Runtime

## Target

Build a side-by-side Rust implementation of the checked-in PICO-8 Dodge
cartridge. The cartridge remains the authority during migration. The native
runtime becomes acceptable only when it matches a recorded Pemsa oracle
frame-by-frame for the same initial state, seed, input masks, rewards, terminal
events, mutable game state, and 128 × 128 indexed framebuffer.

Macroquad supplies the human-facing replay window. It does not own simulation,
game rules, RNG, or pixel semantics. Training uses the Rust core directly,
without Pemsa, Xvfb, xdotool, subprocesses, stdout protocols, or JSON in its
per-decision hot path.

The LLM may assist with source mapping, Rust conversion, test generation, and
failure diagnosis. It is never the correctness authority. Oracle traces,
differential tests, property checks, and accepted performance evidence are.

This directory contains planning and implementation kits only. No existing
Dodge spec or runtime path is replaced by this bundle.

## Non-negotiable target contract

- Checked-in src/dodge/game/dodge.p8 remains byte-for-byte unchanged and
  hash-addressed.
- Native source maps every converted subsystem and important function back to
  a PICO-8 source span.
- Native state has no opaque catch-all for mutable behavior-affecting fields.
- State snapshots expose logical state and pixels together. Pixels use PICO-8
  palette indices in row-major 128 × 128 order; GPU color conversion happens
  only at the viewer boundary.
- Same seed, initial persistent state, action masks, and frame schedule produce
  the same canonical trace.
- Draw-side effects observed in the cartridge are part of the parity contract.
  A renderless training mode may skip GPU work, never behavior-affecting draw
  work.
- Existing Python control, NEAT, and PPO paths remain runnable until native
  parity and a measured replacement are accepted.
- Audio events are extracted and represented, but visual and gameplay parity
  gates do not silently claim audio waveform parity.

## Why this is phased

This crosses five independent risk boundaries: source semantics, asset and
numeric compatibility, mutable game state, pixel rendering, and RL throughput.
A mistake in an early boundary can make every later result look plausible while
being wrong. Each phase therefore ships one testable capability and blocks
dependent work until its evidence is accepted.

## Kits

| Kit | Goal | Depends on | Gate |
|-----|------|------------|------|
| cavekit-dodge-native-p1-source-oracle.md | Freeze source identity and capture full reference traces | - | Repeatable Pemsa state and pixel traces |
| cavekit-dodge-native-p2-extraction-compat.md | Extract cartridge assets and lock PICO-8 compatibility primitives | P1 accepted | Asset hashes and primitive probes match |
| cavekit-dodge-native-p3-native-slice.md | Prove the Rust state/snapshot/frame API on a vertical slice | P2 accepted | Slice traces match frame-by-frame |
| cavekit-dodge-native-p4-full-port.md | Port all gameplay, transitions, render side effects, and state | P3 accepted | Full corpus state and pixel parity |
| cavekit-dodge-native-p5-macroquad-viewer.md | Display and replay native indexed frames through Macroquad | P4 accepted | Native frames replay identically and owner approves visuals |
| cavekit-dodge-native-p6-batched-training.md | Replace per-step emulator IPC with a batched Rust/Python environment | P4 accepted | Behavioral parity plus measured throughput target |
| cavekit-dodge-native-p7-proof-performance.md | Add Kani properties, differential fuzzing, CI, provenance, and regression budgets | P5 and P6 accepted | Proof, fuzz, regression, and benchmark gates |

## Dependency graph

    P1 source + oracle
       |
    P2 extraction + compatibility
       |
    P3 native vertical slice
       |
    P4 complete native core + indexed renderer
      / \
     /   \
    P5    P6
    Macro  batch ABI + training
     \     /
      \   /
       P7 proof + performance hardening

P2 must precede P3 because native code cannot be compared reliably until
numeric, RNG, input, and asset semantics are locked. P3 must precede P4 because
the public state and frame interfaces need one accepted end-to-end slice before
all game systems are ported. P4 must precede P5 because Macroquad must consume a
stable framebuffer rather than become a second simulation. P4 must precede P6
because a batch ABI around wrong state is only a faster wrong environment. P5
and P6 do not depend on each other. P7 waits for both so it can test the
actual viewer and training path together.

## Current source and runtime map

| Authority or surface | Current fact | Migration meaning |
|----------------------|--------------|-------------------|
| src/dodge/game/dodge.p8 | 2,461-line PICO-8 cartridge with Lua, gfx, sfx, and music sections | Immutable source and extraction input |
| src/dodge/game/SPEC.md | Existing control contract; current headless mode replaces _draw with a no-op; full visible state remains deferred | Preserve legacy contract; amend before native parity work |
| src/dodge/neat/bridge.py | Hidden Pemsa/X11 bridge injects keys through xdotool and parses stdout markers | P1 oracle only; never training hot loop |
| src/dodge/neat/state.py | RawState currently exposes player plus enemy/AOE subsets | Compatibility adapter, not full native state |
| src/dodge/imitation/board.py | Current CNN input is a semantic 19 × 16 × 16 tensor | Retain as a derived observation; add canonical pixels and full state |
| src/dodge/rl/ppo.py | Direct PPO steps one DodgeEnv sequentially and encodes each state in Python | P6 replaces environment ownership and batches policy observations |
| src/dodge/neat/SPEC.md | Current NEAT and PPO contracts preserve step-frame, seed, reward, and checkpoint behavior | Native backend must provide an explicit compatibility path |

The current no-op _draw headless mode is not automatically equivalent to the
visible cartridge. The cartridge updates gameplay state from _update60, but
drawgame also creates trails and consumes random values. P1 must record whether
the canonical oracle executes the full draw path, a side-effect-preserving
software rasterizer, or a formally factored draw side effect. Native training
must not mix these modes silently.

## Canonical state model

The implementation should expose a typed snapshot equivalent to the following
conceptual groups:

- lifecycle: frame, mode, transition target/source, started, dead, input mode;
- player: position, displacement, speed, size, collision flags;
- enemies: every active record, including position, velocity, size, type,
  growth/shrink state, life, spawn/inside flags, and death state;
- particles: every particle record and age/type/color/radius fields;
- patterns: selected pattern, counters, timers, rectangles, warnings, targets,
  visibility, interpolation, completion, and spawn effects;
- progression: score, difficulty, freeze and size timers, spawn parameters,
  settings, and all flags affecting future behavior;
- input: current mask, previous mask, pressed mask, mouse/stat inputs, and
  scheduled action boundary;
- randomness: exact RNG state or a compatible serializable representation;
- side effects: camera/shake, palette/fill state, sound events, and other
  mutable renderer-visible state;
- framebuffer: 16,384 palette indices plus palette state and camera state.

This list is a completeness checklist, not permission to hide fields in an
untyped map. The final inventory comes from P1 source analysis and P4 parity
failures.

Conceptual Rust surfaces:

    NativeGame::reset(seed, initial_state) -> Snapshot
    NativeGame::advance_frame(input_mask) -> FrameResult
    NativeGame::step(action, step_frames) -> StepResult
    Snapshot::logical_state() -> FullState
    Snapshot::pixels() -> &[PaletteIndex; 128 * 128]

Names may change during spec review, but the boundaries may not: one pure-ish
core, one canonical snapshot, one indexed framebuffer, and no viewer dependency
in the core.

## Exactness tiers

| Tier | Required equality | First owning kit |
|------|-------------------|------------------|
| A | action schedule, reward, done, score, survival frames | P1 |
| B | every mutable gameplay and input field at every frame | P3/P4 |
| C | palette-index framebuffer at every captured frame | P1/P2/P4 |
| D | sound event identity/timing and persistent data effects | P2/P4, visual gate may defer waveform parity |

No later tier can compensate for a failed earlier tier. A pixel match with a
different hidden RNG state is a failure.

## Throughput strategy

The speedup comes primarily from removing process and display boundaries and
from batching independent environments:

1. Keep one Rust core instance per environment.
2. Advance N environments from one Rust call with an action array.
3. Return contiguous pixel and structured-state buffers; do not serialize JSON
   for each decision.
4. Batch policy inference in the existing PyTorch trainer.
5. Keep Macroquad and all GPU/window work outside training.
6. Use Rayon only across independent lanes after serial and parallel traces
   match exactly.
7. Benchmark both pixels-on and pixels-off observation modes so a copied
   framebuffer cannot hide the real bottleneck.

The performance gate names workload, machine, toolchain, repetitions, statistic,
and comparison path. “Rust is faster” is not evidence.

## Rust workspace and configuration direction

Namtao's configuration advice is useful as a guardrail, not as a dependency
shopping list. The proposed workspace keeps hot code small and puts ergonomic
tools at the edges:

| Crate or area | Responsibility | Dependency direction |
|---------------|----------------|----------------------|
| dodge-core | deterministic state transition, compatibility layer, indexed rasterizer | std-only unless P2 evidence requires a narrowly scoped crate |
| dodge-assets | generated palette, sprites, sfx/music, source manifest | consumes generated assets; no Macroquad |
| dodge-oracle | Pemsa scenario/capture/diff tooling | CLI/process dependencies allowed; never linked by core |
| dodge-viewer | Macroquad window, input, texture presentation, debug overlays | depends on core/assets; no game rules |
| dodge-python | PyO3/NumPy binding and batch adapter, if P6 research accepts it | depends on core; no viewer |
| dodge-proof | Kani harnesses and bounded proof fixtures | depends on core; proof-only configuration |
| benches/tests | Criterion workloads, nextest integration, differential fixtures | dev dependencies only |

Configuration rules:

- Pin Rust, Macroquad, binding, and test-tool versions in Cargo.lock and the
  project devenv; never copy a wildcard dependency version into production.
- Use stable Rust for application/release builds. Isolate any Kani-compatible
  nightly in a proof shell or documented command.
- Put clap/color-eyre/chrono at CLI and report edges; keep them out of
  dodge-core.
- Use Serde for config, traces, provenance, and offline artifacts; never use
  text serialization between Python and a native batch step.
- Use Rayon only for independent environment lanes after P6 serial parity.
- Use Criterion for workload-defined benchmarks and cargo-nextest for test
  execution. Bacon/watchexec are developer conveniences, not runtime
  dependencies.
- Start with strict Clippy/rustfmt/rust-analyzer settings. Deny panics,
  unwrap/expect, unchecked conversions, slicing/indexing, and arithmetic
  side-effects in production code where the selected Rust version supports
  those lints. Allow a rule only with a local rationale and test coverage.
- Use typestate only for lifecycle/config boundaries such as an initialized
  versus running scenario. Do not encode every per-frame game transition in
  generic types; the canonical FullState must remain inspectable and cheap.

Macroquad is intentionally the simpler viewer choice here. It does not decide
the core architecture, and any API/version detail that changes build or buffer
ownership remains a P2/P5 research item.

## Verification model

There are three different claims:

1. Differential conformance: native output equals Pemsa for a finite corpus or
   fuzzed trace.
2. Local formal properties: Kani proves bounded Rust properties such as safe
   framebuffer indexing, action-mask constraints, and snapshot invariants.
3. Mathematical refinement: a separately formalized model proves an abstract
   property of the implementation.

Only the first two are in the initial target. The AWS Automated Reasoning
announcement concerns structured policy validation in Amazon Bedrock
Guardrails; it does not establish a Dafny- or Lean-based Lua-to-Rust compiler.
Dafny's current documentation describes Rust support as partial and growing,
and its generated-runtime model does not make arbitrary generated Rust a
verified drop-in. Therefore this effort uses strict Rust plus oracle tests and
Kani; Dafny or Lean remains an optional later modeling tool, not the production
transpiler.

## Cavekit routing

| Phase action | Route |
|--------------|-------|
| Unresolved parity or scope choice | grill, one question at a time |
| Macroquad, PyO3, Kani, or toolchain API fact | research, source-backed row before spec |
| Behavior, interface, invariant, or task change | spec; nested Dodge specs remain owners |
| New core, renderer, or public ABI boundary | review before build |
| Accepted phase implementation | build only that phase's open tasks |
| Automated gate or drift check | check against controlling spec and kit |
| Failed test or parity mismatch | backprop; add recurrence-catching invariant when justified |
| Accepted phase with spare budget | deepen only with behavior held constant |

The kits describe implementation order and evidence. They do not silently
rewrite src/dodge/game/SPEC.md or src/dodge/neat/SPEC.md.

## Required spec handoff before P1

The repository has nested Dodge specs, not a root SPEC.md. Before implementation
starts, route these amendments through Cavekit spec:

- src/dodge/game/SPEC.md: permit an additive native implementation while
  retaining the immutable PICO-8 oracle; define canonical full-frame capture,
  snapshot identity, and draw-side-effect mode;
- src/dodge/game/SPEC.md: replace the current “full visible world-state JSON
  deferred” boundary with a native snapshot/trace interface and retain JSON
  only as an oracle/export format;
- src/dodge/neat/SPEC.md: add native serial and batched environment interfaces,
  pixel/full-state observation modes, backend selection, and old Pemsa
  compatibility;
- src/dodge/neat/SPEC.md: preserve action order, step_frames, reward shaping,
  seed partitions, checkpoint semantics, and held-out evaluation behavior;
- both specs: add invariants for source hash, deterministic traces, state
  completeness, pixel equality, and no emulator IPC in the native training
  path;
- record open numeric, RNG, draw-side-effect, and audio parity decisions as
  explicit unknowns until P1 evidence resolves them.

## Overall acceptance

- [ ] Every phase gate accepted in dependency order.
- [ ] Native full-state and indexed-pixel trace equals Pemsa on the accepted
      corpus and on a separately generated held-out corpus.
- [ ] Macroquad replay is visibly approved and automated frame hashes match.
- [ ] PPO can select the native backend without changing action order, reward
      definition, seed roles, checkpoint/resume behavior, or evaluation
      semantics.
- [ ] Native batched throughput meets the P6 budget with pixels and full-state
      exposure enabled.
- [ ] Kani, differential fuzzing, nextest, clippy, and benchmark gates pass.
- [ ] Original Python/Pemsa replay remains available as an oracle and fallback.
