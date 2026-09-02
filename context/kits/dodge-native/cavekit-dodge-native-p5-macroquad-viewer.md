---
name: dodge-native-p5-macroquad-viewer
description: Provide a simple Macroquad replay viewer that displays the native indexed framebuffer without becoming a second game implementation
metadata:
  type: project-phase
  phase: P5
  created: "2026-09-01"
  last_edited: "2026-09-01"
---

# Cavekit: P5 Macroquad Viewer

## Scope

Build the human-facing native replay application after P4 parity. Macroquad
owns windowing, input collection, texture presentation, and optional debug
overlays. The native core owns every simulation and pixel decision. The viewer
must be able to display a captured frame without advancing the game, then
replay a stored action trace at visible cadence.

Depends on accepted P4. P6 does not depend on P5.

## Phase contract

### §G GOAL

Macroquad viewer → exact native 128 × 128 indexed frames, integer-scaled with
nearest-neighbor presentation, trace replay, and optional full-state inspection.

### §C CONSTRAINTS

- Macroquad is a viewer dependency only; core crates compile without it.
- Do not reproduce Dodge rules with Macroquad draw calls. Upload or present the
  canonical indexed framebuffer produced by the core.
- Preserve palette indices until the final presentation conversion.
- Default display scale is an integer; no smoothing, filtering, or aspect-ratio
  distortion.
- Viewer input maps to the same Action/InputMask contract as replay files.
- Viewer rendering cannot mutate game state, RNG, timers, particles, or
  persistent data.
- Screenshot comparisons supplement, but do not replace, source framebuffer
  hash comparisons.
- Audio playback remains optional and cannot gate visual parity.

### §I INTERFACES

viewer command → trace or seed/config + optional window/debug options →
Macroquad window and clean exit.

frame presenter → indexed 128 × 128 pixels + palette state → one displayed
native frame.

trace replay → stored seed, initial state, action schedule, and frame cadence →
visible replay without physical keyboard injection.

debug panel → selected frame + FullState → read-only field/entity/timer/RNG
inspection and state/pixel hashes.

capture command → selected native frame → lossless indexed or RGBA artifact
with source hash and frame metadata.

### §R RESEARCH

| id | topic | finding | source |
|----|-------|---------|--------|
| R1 | Macroquad API | Macroquad 0.4.16 exposes `Texture2D::from_rgba8` and `FilterMode::Nearest`, which fit the final indexed-to-RGBA viewer boundary | https://docs.rs/macroquad/0.4.16/macroquad/texture/struct.Texture2D.html |
| R2 | Macroquad getting started | Official examples use a Macroquad main loop and next_frame; viewer lifecycle must remain outside core stepping | https://macroquad.rs/ |
| R3 | Namtao Rust devenv | Suggested setup includes rustfmt, clippy, rust-analyzer, bacon, cargo-nextest, and a pinned Rust devenv; use those practices without putting viewer tools in core | https://www.namtao.com/rust/ |

### §V INVARIANTS

- V1: presented source texture bytes derive only from the native Snapshot
  framebuffer and palette; no duplicate game draw logic exists in viewer.
- V2: same captured frame presented twice → identical source texture and
  lossless capture bytes.
- V3: integer scaling uses nearest-neighbor sampling and preserves the 128 ×
  128 aspect ratio.
- V4: rendering a frame does not change native state, RNG, event queues, or
  subsequent frame hashes.
- V5: replayed trace action order, seed, frame boundaries, and terminal result
  equal the captured trace.
- V6: viewer input is disabled during trace replay; no host input can alter a
  replay.
- V7: debug overlays are opt-in and excluded from canonical framebuffer/hash
  comparisons.
- V8: a displayed frame's source hash, frame index, state hash, and pixel hash
  remain available for owner inspection.

### §T TASKS

| id | status | task | cites |
|----|--------|------|-------|
| P5-T1 | x | Add Macroquad viewer crate and pin selected version/toolchain in Rust workspace | §C,R1 |
| P5-T2 | x | Upload canonical indexed framebuffer through palette-preserving texture path | V1,V2,V3 |
| P5-T3 | x | Add seed/config launch, native frame loop, and visible cadence without core coupling | V4,V5 |
| P5-T4 | x | Add trace replay with controls disabled and exact action/frame scheduling | V5,V6 |
| P5-T5 | x | Add optional read-only state/hash/debug overlay and lossless frame capture | V7,V8 |
| P5-T6 | x | Add viewer screenshot/source-buffer comparison fixtures | V1,V2,V3 |
| P5-T7 | ~ | Perform original-vs-native visual review across representative scenarios | V8 |
| P5-T8 | x | Produce viewer acceptance report and hand stable presentation contract to P7 | V1-V8 |

### §B BUGS

| id | date | cause | fix |
|----|------|-------|-----|
| B1 | - | - | - |

## Gate

### Automated

- Native source framebuffer hash equals the frame loaded by Macroquad before
  and after presentation.
- Presenting a frame leaves subsequent core state/hash unchanged.
- Replay action and frame hashes equal the captured trace.
- Lossless captures at native resolution compare equal to native framebuffer
  conversion; overlay pixels are excluded.
- Core crate still builds without Macroquad.

### Owner

Use Macroquad to inspect menu, gameplay, pattern warnings, particles, death,
settings, and palette changes. Compare native-resolution captures side by side
with Pemsa. Approve only after any differences are explained as overlay,
window-scaling, or an actual parity defect.

### Handoff

P5 is accepted when a human can inspect the same full state and pixels that P6
will send to a training process, without introducing a second source of truth.

## Cavekit routing

- grill: resolve display/audio/debug scope questions.
- research: validate selected Macroquad texture/window APIs.
- spec: add viewer and trace-replay interfaces plus no-render-mutation rules.
- review: challenge any viewer code that draws game entities independently.
- build: execute P5 tasks only.
- check: compare viewer scope to P4 core boundary.
- backprop: image or replay mismatch → source frame/texture/input root cause.
- deepen: optional inspector ergonomics after visual gate.
