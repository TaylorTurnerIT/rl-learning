# P3 acceptance: native vertical slice

Status: `accepted_slice_visual_deferred`

P3 proves the engine-free Rust boundary on the defined slice. It does not
claim complete-game parity.

## Identity

- Cartridge: `src/dodge/game/dodge.p8`
- Cartridge SHA-256: `7453a9658fd32577385ad72672a54ad84ff70567fadbde75ba6634aa5cc684a3`
- Pemsa SHA-256: `19823433dd0fa75568207408d09646960113858a43c159930388b7bf62b259e4`
- Rust toolchain: `rustc 1.97.1 (8bab26f4f 2026-07-14)`
- Cargo lock SHA-256: `2c3794e25512263d8ceb11c0b4075b7eb61556f61c9d2a99209622c87fe25ef0`
- Capture: canonical full-draw, post-update/post-draw, indexed 128 × 128

## Corpus

| fixture | seed | frames | source result | logical parity | source frame hash |
|---|---:|---:|---|---|---|
| `seed-42-movement` | 42 | 501 | score 3, survival 487, died | match | `c20c557f6e38eaab5fcf872c20827979e931eb9865b65c79d4e4f8459d40bfca` |
| `seed-7-neutral` | 7 | 204 | score 1, survival 190, died | match | `82e80d870f4fb735a2d23312f2bd8c1fa137e88db6b6fa189412ca62957b75e7` |
| `seed-99-diagonal` | 99 | 217 | score 0, survival 203, died | match | `8e0dbaeaa702178815683ad112c89f3967a41aff8f9f0afb63d03497635f94c8` |

Logical comparison checks every captured frame's frame number, input masks,
lifecycle mode, dead/done flags, reward, ordered events, player fields, enemy
count/order/fields, final score, and survival count. Native traces are stored
in the ignored `src/dodge/runtime/.native-p3-oracle/` directory and regenerated
from the tracked corpus files.

## Pixel boundary

All three full-draw fixtures reach the same first pixel mismatch:

| frame | pixel | expected | native | source span | classification |
|---:|---|---:|---:|---|---|
| 1 | `[20,17]` | 7 | 12 | `drawtransition`, lines 502–527 | missing transition/menu sprite and text raster |

The native framebuffer already owns the canonical 128 × 128 palette-index
buffer, but P3's renderer contains only the representative primitive slice.
P4 must port the extracted sprite sheet, PICO-8 built-in font behavior,
menu/settings/death screens, transition fill ordering, particles, patterns,
palette state, and all remaining draw-side effects before pixel parity can be
accepted.

## Accepted boundary

- Typed reset, lifecycle, menu start, transition, game-ready frame, nine action
  masks, one-frame stepping, and exact frame-horizon stepping.
- Fixed-point movement and bounds, representative normal enemies, enemy
  growth, source RNG offset, personality/size selection, ordered overlap
  deletion, player collision/death, power-up size behavior, score and survival
  accounting.
- Deterministic initial pattern RNG consumption and first pattern-selection RNG
  boundary; pattern geometry remains deferred.
- Canonical snapshot serialization roundtrip, provenance, indexed framebuffer
  ownership, render purity, typed side-effect events, and first-mismatch
  diagnostics.

## Deferred to P4

- Complete mutable state: particles, all enemy families, dynamic patterns,
  settings, progression, persistent data, camera/palette/fill state, and sound
  events.
- Full source-faithful update/draw ordering and visual parity.
- Full restore API and complete source-map closure.
- Held-out full-game corpus and owner visual inspection.

Handoff: `p4_full_port_review_go`
