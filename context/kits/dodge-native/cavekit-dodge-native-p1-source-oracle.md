---
name: dodge-native-p1-source-oracle
description: Freeze Dodge cartridge identity and capture a deterministic Pemsa oracle with full logical state, side effects, and indexed pixels
metadata:
  type: project-phase
  phase: P1
  created: "2026-09-01"
  last_edited: "2026-09-01"
---

# Cavekit: P1 Source and Oracle

## Scope

Make the original cartridge an executable, repeatable reference before writing
native gameplay code. Inventory source sections and mutable symbols, define a
versioned trace, and capture enough state and pixels to distinguish a correct
port from a plausible imitation.

Related target: cavekit-dodge-native-overview.md.

## Phase contract

### §G GOAL

Same cartridge bytes, seed, initial persistent state, and action schedule →
repeatable per-frame oracle trace containing logical observations, rewards,
terminal state, side effects, and 128 × 128 palette-index pixels.

### §C CONSTRAINTS

- src/dodge/game/dodge.p8 immutable; record SHA-256 and section hashes.
- Existing Pemsa binary and bridge remain reference infrastructure.
- Do not replace current control/headless behavior during P1.
- Canonical capture must distinguish full _draw from current no-op headless
  mode; no silent equivalence assumption.
- Every process, temporary cartridge, display, and cartdata directory owned and
  cleaned by the probe.
- Capture source line, cartridge hash, Pemsa revision, host/toolchain, seed,
  action schedule, and capture mode with every trace.
- PICO-8 numeric, RNG, list mutation, input, and draw semantics remain unknown
  until probes resolve them.

### §I INTERFACES

oracle scenario → cartridge hash, Pemsa identity, seed, initial state,
input-mask schedule, step-frame schedule, capture mode.

oracle frame → frame index, pre/post input masks, mode, full raw state,
reward, done, side-effect events, pixel buffer, state hash, pixel hash.

oracle trace → schema version, scenario metadata, ordered frames, terminal
result, aggregate hashes, and provenance.

source manifest → cartridge section ranges, function/source spans, mutable
global inventory, asset hashes, and unresolved symbols.

probe command → deterministic trace artifact or nonzero diagnostic; no partial
trace accepted as a golden fixture.

### §R RESEARCH

| id | topic | finding | source |
|----|-------|---------|--------|
| R1 | AWS Automated Reasoning | Announcement describes structured policy validation in Amazon Bedrock Guardrails; not a Lua-to-Rust proof compiler | https://aws.amazon.com/blogs/aws/prevent-factual-errors-from-llm-hallucinations-with-mathematically-sound-automated-reasoning-checks-preview/ |
| R2 | Pemsa output | Existing cartridge instrumentation uses printh/stdout as its observation channel | https://github.com/egordorichev/pemsa/blob/6c13c5879c800af33543f702a353285cfa9e6fb0/src/pemsa/util/pemsa_system_api.cpp#L35-L44 |
| R3 | Pemsa cartdata | Relative cartdata storage makes per-process working-directory isolation part of deterministic capture | https://github.com/egordorichev/pemsa/blob/6c13c5879c800af33543f702a353285cfa9e6fb0/src/pemsa/cart/pemsa_cartridge_module.cpp#L621-L637 |

### §V INVARIANTS

- V1: source bytes + section hashes unchanged before and after every probe.
- V2: same scenario repeated ≥3 times → byte-identical canonical trace,
  frame hashes, terminal result, and provenance fields except run timestamp.
- V3: every accepted frame has exactly one monotonically increasing game-frame
  index; capture boundary documented as before/after update and draw.
- V4: accepted full-draw trace records pixels and every mutable field needed to
  reproduce the next frame; unknown mutable state fails capture validation.
- V5: full-draw and current no-op-headless captures are labeled distinct modes;
  no fixture compares them as equivalent.
- V6: action schedule maps to exact PICO-8 button masks and previous-button
  semantics; no host mouse or key state enters canonical input.
- V7: seed, initial persistent data, settings, and RNG initialization recorded;
  replay consumes no host entropy after scenario creation.
- V8: probe error, timeout, signal, or parse failure reaps owned Pemsa/Xvfb
  processes and rejects incomplete output.
- V9: golden fixtures include menu start, movement, neutral, diagonal input,
  enemy spawn, pattern activation, collision, death, and transition boundaries.
- V10: source inventory marks each behavior-affecting mutable symbol as
  represented, derived, persistent input, or unresolved.

### §T TASKS

| id | status | task | cites |
|----|--------|------|-------|
| P1-T1 | . | Hash cartridge bytes and sections; inventory functions, globals, assets, and source spans | V1,V10 |
| P1-T2 | . | Define versioned scenario/frame/trace schema with canonical field ordering | V3,V4,V7 |
| P1-T3 | . | Extend instrumentation to capture full draw-side-effect mode and indexed framebuffer | V4,V5 |
| P1-T4 | . | Add deterministic probes for RNG, numeric operations, input masks, list mutation, and camera/palette state | V6,V7,V10 |
| P1-T5 | . | Capture the required golden scenario corpus, including death and transition boundaries | V9 |
| P1-T6 | . | Repeat corpus runs and compare canonical bytes, logical hashes, and pixel hashes | V2 |
| P1-T7 | . | Record current Python/Pemsa interactive and headless throughput baselines with workload metadata | V2,V8 |
| P1-T8 | . | Produce acceptance report and hand source manifest plus oracle fixtures to P2 | V1-V10 |

### §B BUGS

| id | date | cause | fix |
|----|------|-------|-----|
| B1 | - | - | - |

## Gate

### Automated

- Repeated corpus traces compare equal after removing only run timestamps.
- Every accepted frame has a valid state hash and 16,384-entry indexed pixel
  buffer.
- The source hash before and after capture is equal.
- Owned emulator and Xvfb processes are absent after success and failure.
- Both full-draw and legacy no-op traces are reproducibly labeled.

### Owner

Open representative frame pairs from menu, active gameplay, pattern warning,
collision, and death. Confirm that the captured pixels are the intended
original-game images and that a draw-side-effect decision is explicit.

### Handoff

P1 is accepted only when P2 can consume a stable source manifest and replay
fixtures without guessing frame boundaries or hidden state.

## Cavekit routing

- research: only unresolved Pemsa/PICO-8 behavior or capture API facts.
- spec: add source-hash, trace, capture-mode, and full-state invariants.
- review: review the oracle boundary before any native conversion.
- build: execute P1 tasks only.
- check: audit the updated game spec against the trace schema.
- backprop: every mismatch or flaky capture gets a cause and recurrence guard.
- deepen: optional trace compression or better diff tooling after gate only.
