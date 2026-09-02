# P1 live evidence

Captured 2026-09-01 with the checked-in cartridge and local Pemsa runtime.
The source and runtime identities below are embedded in every canonical trace.

## Immutable identities

- cartridge `src/dodge/game/dodge.p8`: SHA-256
  `7453a9658fd32577385ad72672a54ad84ff70567fadbde75ba6634aa5cc684a3`
- Pemsa `src/dodge/runtime/pemsa`: SHA-256
  `19823433dd0fa75568207408d09646960113858a43c159930388b7bf62b259e4`
- cartridge size: 61,699 bytes; 2,461 lines
- sections: `lua`, `gfx`, `sfx`, `music`; each section has its own byte length
  and SHA-256 in the trace provenance

## Compatibility probe

`dodge-native-probe --seed 42` completed successfully. The tracked result is
`p1-probe-seed-42.json`.

- RNG sample: `0.0334`, then `rnd(10) = 3.2996`
- numeric sample: `flr(-1.2) = -2`, `ceil(-1.2) = -1`, `mid(0,9,4) = 4`,
  `7 % 4 = 3`
- list sample: deleting `2` and appending `4` yields `1,3,4`
- camera/pixel sample: camera offset followed by `pset`/reset reads palette
  index `5` through `pget`
- input sample: x, left, neutral, up-right, down produce the expected held
  and pressed masks as explicit `0|1` values

## Canonical corpus

The canonical artifact is JSON with a 128 × 128 row-major palette-index buffer
per frame. `trace_sha256` is the complete canonical file hash; `frames_sha256`
is the hash over its ordered frame records.

| scenario | frames | result | trace SHA-256 | frames SHA-256 |
|---|---:|---|---|---|
| seed-42 movement | 501 | score 3, survival 487 | `4c8188fa6316d45f128fad011f3ee1f23f34ad75fea24048f7416b307365d7a5` | `b596f906a838606b544bfc48ec71449c114b6bbe094833215b216f4c4ec8635a` |
| seed-7 neutral | 204 | score 1, survival 190 | `d3727a21306b0a06ba9239abdbb12c6ca332dbed76e1a3cdabb4edd043c5d11e` | `9d69f016f84866d4974d412d1d7ba7397af51eeca82bc9a93146ab0e1d82cfb3` |
| seed-99 diagonal | 217 | score 0, survival 203 | `5b0664edf2bc01850a5546d86d3abba78a870a23d8a4f24bdaaf3d0e998585c1` | `4151e9e8aa1e2779ead4301bfd3710ff9800009876ae9c7c5fc3dfc3c731d1f6` |

The seed-42 scenario was captured three times with the finalized harness. All
three complete files were byte-identical with SHA-256
`4c8188fa6316d45f128fad011f3ee1f23f34ad75fea24048f7416b307365d7a5`.

Representative seed-42 event coverage includes 12 enemy spawns, a collision,
active-pattern frames, and terminal death. The first captured frame is a
transition/menu visual; the terminal frame is the game-over visual. A local
four-frame nearest-neighbor montage was inspected during the gate.

## Mode distinction and baseline

The same schedule and seed intentionally produce different results in the
existing no-op headless mode because `drawgame` consumes gameplay RNG through
`spawntrail`:

- full-draw oracle: score 3, frame 501, survival 487
- legacy no-op headless, default startup: score 4, frame 608, survival 581
- legacy no-op headless, game-start wait: score 6, frame 533, survival 506

The distinction is now an explicit contract. A future renderless Rust core must
retain behavior-affecting draw-side effects even when it omits GPU presentation.

Measured with the same seed-42 movement workload on 2026-09-01, including
devenv/uv startup and canonical JSON output:

- legacy no-op headless: 1.05 s wall time
- full-draw oracle with 501 × 16,384 pixel indices: 22.86 s wall time

The second number is an oracle/capture cost, not the desired training target.
The native training target is to remove Pemsa/Xvfb/stdout/JSON from the hot
path, keep the deterministic draw-side-effect model, and batch independent
environments.

## Open before native parity

This evidence closes the source-identity and capture boundary; it does not yet
claim complete mutable-game-state parity. The frame `state` currently uses the
existing `RawState` observation, augmented by lifecycle/input metadata,
side-effect event names, and pixels. Exact internal RNG state, persistent
cartdata/settings, particle records, pattern internals, palette/camera state,
and every other behavior-affecting mutable field remain to be extracted and
represented before the native Rust core can be accepted.

## Reproduction

```text
DODGE_HEADLESS=1 devenv -q shell -- dodge-native-probe --seed 42
DODGE_HEADLESS=1 devenv -q shell -- dodge-native-oracle --commands context/kits/dodge-native/corpus/seed-42-movement.json --seed 42 --output src/dodge/runtime/.native-oracle-check/seed-42-movement.json --timeout 30
```
