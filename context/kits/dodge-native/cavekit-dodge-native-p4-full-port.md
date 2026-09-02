---
name: dodge-native-p4-full-port
description: Complete source-faithful Rust conversion of Dodge gameplay, transitions, render side effects, and full observable state
metadata:
  type: project-phase
  phase: P4
  created: "2026-09-01"
  last_edited: "2026-09-01"
---

# Cavekit: P4 Full Native Port

## Scope

Port every behavior-affecting subsystem found in the P1 source inventory into
the typed native core. Preserve source order, PICO-8 numeric/RNG semantics,
list mutation behavior, input timing, dynamic pattern construction, draw-side
effects, palette/camera state, and terminal transitions. Finish the canonical
full state and software renderer before adding a viewer or high-throughput
binding.

Depends on accepted P3.

## Phase contract

### §G GOAL

Full native Dodge core → same frame-by-frame logical state, reward, terminal
events, side effects, and palette-index pixels as the canonical Pemsa oracle.

### §C CONSTRAINTS

- Conversion follows the P1 source manifest and P2 compatibility layer;
  source-faithful Rust is required.
- No mechanical “equivalent” rewrite is accepted without differential
  evidence for affected scenarios and a reviewed source-map note.
- Preserve update order, iteration order, deletion behavior, floating/fixed
  numeric behavior, RNG consumption, and draw-driven mutation.
- Dynamic patterns, generated variants, particle trails, collision rules,
  transitions, settings, and persistent data effects remain in scope.
- Macroquad remains out of the core. Software raster output is authoritative.
- Training and optimization work stays deferred to P6.
- Existing dodge.p8 and existing Python/Pemsa commands remain available as
  oracle/fallback.

### §I INTERFACES

full native game → reset, advance_frame, step, snapshot, restore, and close
without host process or display dependencies.

FullState → lifecycle, player, every enemy, every particle, patterns and
rectangles, settings, progression, input, RNG, camera/palette/fill state,
sound events, persistent-state inputs, and all other inventory entries.

FrameResult → pre/post frame identity, state snapshot, indexed pixels, reward,
done, audio/event records, and state/pixel hashes.

restore(snapshot) → next-frame-equivalent native instance; unsupported
persistent external state fails explicitly.

parity report → scenario, first mismatch, field/pixel coordinate, source map,
RNG delta, and trace artifacts.

### §R RESEARCH

| id | topic | finding | source |
|----|-------|---------|--------|
| R1 | Dafny target limits | Current Dafny documentation describes Rust support as partial and growing; generated code also relies on a Dafny-specific runtime and verified scope | https://dafny.org/latest/Installation |
| R2 | Dafny verification boundary | Dafny verifies Dafny-originated code through Boogie/Z3; generated target code is not a general proof that arbitrary translated Rust is equivalent | https://dafny.org/dafny/DafnyRef/DafnyRef |
| R3 | Kani role | Kani can prove local Rust safety/correctness properties or produce counterexamples, but resource exhaustion is a possible result | https://model-checking.github.io/kani/ |

### §V INVARIANTS

- V1: every behavior-affecting mutable source-manifest entry has exactly one
  typed native representation or an explicit reviewed derived field.
- V2: same seed, persistent initialization, settings, action masks, and frame
  schedule → identical full canonical trace through terminal state.
- V3: native frame update invokes systems in the source-observed order; no
  subsystem consumes RNG, input, or time implicitly outside the trace contract.
- V4: full state restore followed by the same next input produces the same next
  state, pixels, reward, events, and hashes.
- V5: every active enemy, particle, pattern, rectangle, warning, timer, flag,
  and side-effect state appears in FullState or is proven derived.
- V6: native indexed framebuffer equals oracle pixels at every accepted capture
  frame; RGB conversion and GPU output are not used as proof of equality.
- V7: draw-side effects and RNG use match the canonical capture mode; a
  renderless path preserves behavior-affecting draw work.
- V8: camera, palette remapping, fill patterns, sprite/text rasterization, and
  clipping use the extracted indexed asset/compatibility data.
- V9: input and stat/mouse mode are replay-controlled; host state cannot alter
  canonical traces.
- V10: reward, death, survival frames, score, transition frames, and terminal
  events match existing Dodge/PPO contracts.
- V11: a mismatch fails the phase even if final score or final image matches;
  first divergent frame and field/pixel are recorded.
- V12: existing Python control, headless, NEAT, and PPO paths pass their
  pre-port regression suite unchanged.

### §T TASKS

| id | status | task | cites |
|----|--------|------|-------|
| P4-T1 | . | Port lifecycle, initialization, settings, menu, transitions, persistent state, and high-score boundaries | V1,V2,V10 |
| P4-T2 | . | Port player movement, collision, death, progression, freeze, sizing, and difficulty behavior | V2,V3,V10 |
| P4-T3 | . | Port particles, trails, enemy families, growth/shrink/death states, and spawn logic | V1,V3,V5,V7 |
| P4-T4 | ~ | Port pattern tables, dynamic variants, interpolation, warnings, visibility, and completion | V1,V2,V3,V5 |
| P4-T5 | . | Port complete indexed draw path, palette/camera/fill state, sprite/text primitives, and sound-event emission | V6,V7,V8 |
| P4-T6 | . | Expand FullState inventory, restore, canonical serializer, and source-map coverage | V1,V4,V5 |
| P4-T7 | . | Differential-run full corpus frame-by-frame; add targeted fixtures for every first mismatch | V2,V6,V11 |
| P4-T8 | . | Run held-out randomized seed/action traces and legacy regression suite | V2,V12 |
| P4-T9 | . | Produce full-port acceptance report and stable core handoff to P5/P6 | V1-V12 |

### §B BUGS

| id | date | cause | fix |
|----|------|-------|-----|
| B1 | - | - | - |

## Gate

### Automated

- Full accepted corpus matches at every frame for logical state, reward, done,
  events, RNG checkpoint, and indexed pixels.
- Held-out corpus contains no mismatch after all targeted fixtures pass.
- Snapshot restore/replay is deterministic.
- Source manifest has no unresolved behavior-affecting symbols.
- Existing Python/Pemsa tests remain green.
- Legacy no-op headless and canonical full-draw modes are explicit and never
  compared as if they were one semantics.

### Owner

Play or replay representative menu, settings, active game, high-score/death,
pattern, freeze, sizing, and transition sequences in the original runtime and
native frame dump. Confirm the native images are visually identical at native
128 × 128 resolution.

### Handoff

P4 is the major architecture gate. P5 and P6 may consume only the accepted
native APIs and must not add game rules to their viewer or binding layers.

## Cavekit routing

- grill: decide any source-semantic ambiguity before coding around it.
- research: verify only external API/toolchain questions; cartridge behavior
  remains oracle-derived.
- spec: add full-state, pixel, restore, and draw-side-effect invariants.
- review: required before this high-blast-radius conversion.
- build: execute P4 subsystem tasks in source-order groups.
- check: compare source inventory, native state, and accepted task status.
- backprop: classify every parity failure as conversion defect, oracle defect,
  or phase-boundary defect before retrying.
- deepen: defer until full parity; performance refactors before parity are
  forbidden.
