---
name: dodge-native-p3-native-slice
description: Establish the engine-free Rust game/state/frame API on a small end-to-end Dodge slice before porting every subsystem
metadata:
  type: project-phase
  phase: P3
  created: "2026-09-01"
  last_edited: "2026-09-01"
---

# Cavekit: P3 Native Vertical Slice

## Scope

Create the first executable native path: initialize the game, cross the menu
start boundary, accept the nine-action input model, advance exact game frames,
update the player and a representative hazard, and emit a canonical full-state
snapshot plus indexed framebuffer. This phase proves the architecture on a
small slice; it does not claim the complete game is ported.

Depends on accepted P2.

## Phase contract

### §G GOAL

Native serial game → deterministic reset, frame advance, action step, full
snapshot, reward/done result, and pixel buffer; slice trace equals Pemsa.

### §C CONSTRAINTS

- Core crate has no Macroquad, window, Python, Pemsa, xdotool, or subprocess
  dependency.
- Core state uses typed Rust structures; no untyped Lua table or opaque
  “remaining state” field.
- Frame advancement and action stepping are separate so validation can inspect
  every underlying game frame while training can use step_frames.
- State and framebuffer are snapshots/views of one simulation, not separately
  simulated models.
- Core does not optimize away draw behavior until P1/P2 classify its side
  effects.
- Existing Python/Pemsa paths remain untouched.

### §I INTERFACES

native config → seed, initial persistent state, settings, action horizon, and
observation mode.

native reset → initial Snapshot with frame zero and canonical hashes.

native frame → input mask → FrameResult containing Snapshot, reward delta,
  done, side-effect events, and optional diff metadata.

native action → Action + exact step_frames → StepResult containing final
  Snapshot, cumulative reward, done, and frames advanced.

snapshot → typed FullState, indexed palette framebuffer, palette/camera state,
  audio events, and provenance/version.

trace diff → field path, frame, expected value, actual value, and source-map
  reference.

### §R RESEARCH

| id | topic | finding | source |
|----|-------|---------|--------|
| R1 | Macroquad boundary | Macroquad supplies a small 2D API and automatic geometry batching; keeping it outside core preserves a headless path independent of its frame loop | https://docs.rs/macroquad/latest/macroquad/ |
| R2 | Rust verification | Kani checks Rust safety and custom assertions, and reports proof, counterexample, or resource exhaustion; it is suitable for local core properties, not a black-box equivalence proof | https://model-checking.github.io/kani/ |

### §V INVARIANTS

- V1: reset with same seed, initial data, and config → byte-identical snapshot.
- V2: advance_frame(input) advances exactly one game frame; step(action,n)
  advances exactly n frames unless terminal behavior is explicitly recorded.
- V3: nine actions map to the same button masks and order used by the oracle;
  neutral clears prior input.
- V4: every slice frame exposes the complete mutable state for that slice and
  a 16,384-entry palette-index framebuffer.
- V5: native state hash, reward, done, score, survival frame count, and pixels
  equal Pemsa on the accepted slice corpus.
- V6: snapshot serialization/deserialization preserves all fields and hashes.
- V7: renderer reads canonical state and writes the framebuffer; it cannot
  advance gameplay RNG, input, timers, or entity state.
- V8: state transitions do not index outside bounded asset, framebuffer, or
  fixed-slot storage; failures return typed errors.
- V9: a native mismatch reports first frame, first field path, source span, and
  expected/actual values; aggregate hashes alone are insufficient.

### §T TASKS

| id | status | task | cites |
|----|--------|------|-------|
| P3-T1 | . | Scaffold Rust workspace and engine-free core crate with selected toolchain/config | §C,V8 |
| P3-T2 | . | Port reset, lifecycle, menu transition, action masks, numeric helpers, and RNG state into typed modules | V1,V2,V3 |
| P3-T3 | . | Port player movement, bounds, representative enemy/hazard, collision, reward, and terminal behavior | V2,V5 |
| P3-T4 | . | Implement FullState, Snapshot, canonical serialization, and indexed framebuffer ownership | V4,V6 |
| P3-T5 | . | Add serial native runner that consumes P1 scenarios without emulator IPC | V2,V5 |
| P3-T6 | . | Add field-level differential comparison and source-map diagnostics | V5,V9 |
| P3-T7 | . | Run slice corpus at every frame and fix mismatches without weakening the oracle | V5,V7 |
| P3-T8 | . | Produce vertical-slice acceptance report and hand stable APIs to P4 | V1-V9 |

### §B BUGS

| id | date | cause | fix |
|----|------|-------|-----|
| B1 | - | - | - |

## Gate

### Automated

- Native and Pemsa traces match for the defined slice at every captured frame.
- First-mismatch diagnostics identify a logical field or pixel coordinate.
- Same scenario repeated across separate native instances is byte-identical.
- Snapshot roundtrip preserves logical and pixel hashes.
- Core builds and tests without Macroquad or Python imports.
- Invalid action, frame count, or state input fails before partial mutation.

### Owner

Review the native state schema and confirm every field needed to explain the
slice is named and typed. Inspect an indexed framebuffer dump from menu and
gameplay. Reject any opaque state escape hatch.

### Handoff

P3 is accepted only when P4 can add the remaining cartridge systems without
changing the public snapshot, frame, or action contracts except through an
explicit reviewed amendment.

## Cavekit routing

- grill: settle slice boundary or state ownership questions not answered by
  P1/P2 evidence.
- research: validate selected Rust crate/toolchain APIs.
- spec: add native serial interfaces and slice invariants to controlling specs.
- review: adversarially review typed state completeness and draw boundary.
- build: execute P3 tasks only.
- check: ensure no future Macroquad or batch behavior leaked into core.
- backprop: every slice mismatch gets a root-cause record and new invariant
  when it represents a reusable failure class.
- deepen: improve one accepted core abstraction only after the slice gate.
