# P4 review gate

Review basis: controlling native spec, P2 acceptance artifacts, P3 code and
corpus, checked-in `dodge.p8`, and P4 Cavekit contract.

## Findings

- HARDEN: P4 must port source update order, not only output fields. The source
  `updategame` calls `updateparts`, freeze handling, sizing, spawn, enemy
  update, and pattern update in order (`src/dodge/game/dodge.p8:287-355`),
  while the current slice stops after collision, player movement, spawn,
  enemies, and pattern scheduling (`native/crates/dodge-core/src/game.rs:258-268`).
  P4-T2/T3/T4 must add boundary fixtures for each stage.
- HARDEN: P4's FullState interface is broader than the current slice. The
  contract requires particles, patterns, settings, progression, input, RNG,
  camera/palette/fill, sound events, and persistent inputs
  (`context/kits/dodge-native/cavekit-dodge-native-p4-full-port.md:48-62`),
  while the current state ends at transition and a representative enemy
  (`native/crates/dodge-core/src/snapshot.rs:38-58`). P4-T6 must not hide any
  remaining source state behind a catch-all field.
- HARDEN: exact pixels require PICO-8 built-in text compatibility in addition
  to the extracted cartridge gfx. `print2` invokes the built-in print primitive
  (`src/dodge/game/dodge.p8:1075-1078`), and the current P3 pixel mismatch is
  already in the transition/menu boundary (`p3-acceptance-report.json`). P4-T5
  needs a glyph fixture, sprite transparency, palette remap, clipping, and
  transition-order tests.
- HARDEN: draw-side RNG must remain explicit. The source `drawgame` calls
  `spawntrail` for every power-up enemy (`src/dodge/game/dodge.p8:407-501` and
  `src/dodge/game/dodge.p8:845-850`), while P3 accounts for the known slice
  draws in a dedicated compatibility step (`native/crates/dodge-core/src/game.rs:175-176`).
  P4-T3/T5 must replace this slice accounting with state-driven draw side
  effects and test renderless/full-draw equivalence.
- NOTE: P4's deterministic contract must document any host-independent numeric
  choice. The current nearest-corner helper uses `f64`
  (`native/crates/dodge-core/src/game.rs:397-415`) while the source distance
  helper uses square root (`src/dodge/game/dodge.p8:1080-1083`). Keep the
  source-compatible choice or replace it only with differential evidence.

## Verdict

BLOCK: 0.

HARDEN: 4. All are already owned by P4 tasks and invariants; they must remain
acceptance-gated.

NOTE: 1.

gate: GO for P4-T1 after the accepted P3 handoff. P5/P6 remain blocked until
P4-T9 accepts full logical and indexed-pixel parity.
