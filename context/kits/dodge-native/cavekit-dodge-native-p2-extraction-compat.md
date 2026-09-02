---
name: dodge-native-p2-extraction-compat
description: Deterministically extract PICO-8 assets and establish Rust compatibility primitives from oracle probes
metadata:
  type: project-phase
  phase: P2
  created: "2026-09-01"
  last_edited: "2026-09-01"
---

# Cavekit: P2 Extraction and Compatibility

## Scope

Turn the immutable cartridge into deterministic Rust inputs and lock the small
PICO-8 semantic layer that every later port depends on. Extract graphics,
sprites, sound/music records, static tables, and source mappings. Convert Lua
subsystems deliberately; do not pretend that a generic Lua-to-Rust transpiler
preserves PICO-8 APIs, number behavior, implicit globals, or draw side effects.

Depends on accepted P1.

## Phase contract

### §G GOAL

One extractor invocation over one hash-addressed dodge.p8 → reproducible assets,
source map, and compatibility test suite; every locked primitive matches the
P1 oracle probes.

### §C CONSTRAINTS

- Read only the checked-in cartridge; generated output is disposable and
  reproducible.
- Static data may be generated only when extraction proves its source span and
  ordering. Dynamic pattern construction remains ported logic.
- No gameplay simplification, RNG substitution, or renderer shortcut hidden in
  compatibility helpers.
- Numeric representation remains an evidence-backed decision. If native Rust
  arithmetic cannot match oracle boundaries, isolate a PICO-8-compatible numeric
  type rather than accepting drift.
- Asset decode preserves palette indices, sprite coordinates, dimensions,
  transparency, fill patterns, ordering, and audio record identity.
- Generated files include source hash and generator version; stale generated
  output fails validation.

### §I INTERFACES

extract command → input cartridge + output directory → manifest, source map,
graphics, palette, sprites, sfx, music, static tables, and hashes.

source map → Rust target symbol + PICO-8 function/global/section + source span +
conversion note + parity status.

PicoCompat → numeric helpers, RNG, input/stat access, palette/camera/fill
state, primitive drawing sink, sprite/text lookup, and sound-event sink.

asset package → indexed graphics/palette/sfx/music data with source and content
hashes.

primitive probe → input fixture + expected indexed pixels/state/RNG delta.

### §R RESEARCH

| id | topic | finding | source |
|----|-------|---------|--------|
| R1 | Macroquad scope | Macroquad presents itself as a simple Rust game library with efficient 2D rendering; use it later as a viewer, not as PICO-8 raster authority | https://docs.rs/macroquad/latest/macroquad/ |
| R2 | Macroquad version | docs.rs currently documents macroquad 0.4.16; implementation must pin and validate the selected version in this repo's devenv | https://docs.rs/macroquad/latest/macroquad/ |
| R3 | Namtao configuration | Suggested Rust setup includes strict Clippy lints, rustfmt/rust-analyzer, cargo-nextest, Criterion, Rayon, and Serde; adopt selectively per crate | https://www.namtao.com/rust/ |

### §V INVARIANTS

- V1: extractor output for same cartridge bytes and generator version is
  byte-identical; changed source hash invalidates output.
- V2: graphics decode reproduces source palette indices and sprite rectangles;
  no implicit RGB conversion in core assets.
- V3: static pattern/table output preserves source order and numeric values;
  dynamic table construction remains represented by executable logic.
- V4: seeded RNG probe sequence and state checkpoint/restart match Pemsa.
- V5: floor, ceil, round, midpoint/clamp, division, modulo, comparisons,
  coercions, and boundary values match PICO-8 probes.
- V6: btn, btnp, stat, mouse/input mode, camera, palette, fill pattern,
  sprite, primitive drawing, text, and sound-event probes match their
  canonical oracle effects.
- V7: every generated asset and compatibility symbol has source mapping and
  content hash.
- V8: unresolved primitive or source symbol blocks P3; no “best guess” entry
  may be marked accepted.
- V9: isolated primitive raster tests compare indexed pixels, not screenshots
  after GPU scaling or color conversion.

### §T TASKS

| id | status | task | cites |
|----|--------|------|-------|
| P2-T1 | . | Implement hash-checked p8 section parser and source manifest generator | V1,V7 |
| P2-T2 | . | Decode graphics, palette, sprite metadata, sfx, music, and proven static tables | V2,V3,V7 |
| P2-T3 | . | Build source-span map from PICO-8 functions/globals to planned Rust modules | V7,V8 |
| P2-T4 | . | Implement compatibility numeric/RNG/input/persistent-state probes | V4,V5,V6 |
| P2-T5 | . | Implement indexed software raster primitives and palette/fill/camera state | V2,V6,V9 |
| P2-T6 | . | Add primitive differential fixtures against P1 oracle frames | V5,V6,V9 |
| P2-T7 | . | Add stale-output and unresolved-symbol failures | V1,V7,V8 |
| P2-T8 | . | Produce conversion map and accepted compatibility report for P3 | V1-V9 |

### §B BUGS

| id | date | cause | fix |
|----|------|-------|-----|
| B1 | - | - | - |

## Gate

### Automated

- Clean extraction twice produces identical manifests and asset hashes.
- Corrupt or stale input hash fails before generated assets are consumed.
- Primitive, numeric, RNG, input, palette, and isolated raster fixtures match
  the P1 oracle.
- No unresolved behavior-affecting symbol remains unclassified.
- Generated graphics can be reassembled into the same indexed source image.

### Owner

Inspect extracted title sprites, gameplay sprites, palette remaps, and at least
one fill-pattern frame. Confirm that the asset output is visually the original
cartridge data before Macroquad exists.

### Handoff

P2 is accepted only when P3 can build a native frame without importing
Macroquad, Python, Pemsa, or an untested semantic helper.

## Cavekit routing

- grill: decide unresolved numeric, persistent-state, or audio scope questions.
- research: verify current crate APIs only; local cartridge behavior comes from P1.
- spec: add source-map, asset-hash, compatibility, and unresolved-symbol rules.
- review: challenge the semantic boundary before broad porting.
- build: execute P2 tasks only.
- check: compare generated interfaces and source map against P1 inventory.
- backprop: promote each conversion mismatch into a primitive invariant when
  recurrence is possible.
- deepen: improve one compatibility module only after all P2 tests are green.
