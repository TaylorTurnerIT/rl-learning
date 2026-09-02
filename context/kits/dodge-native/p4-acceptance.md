# P4 acceptance: full native core

Status: `accepted_automated_owner_visual_review_pending`

The engine-free Rust core now covers the complete extracted update and indexed
draw boundary. Eight canonical full-draw Pemsa traces—four golden and four
held-out—match the native trace at every frame, including decoded FullState,
reward, terminal state, ordered audio-event metadata, RNG checkpoints, and all
16,384 palette-index pixels. The raw traces remain ignored because they are
large; their paths and SHA-256 identities are recorded in
`p4-acceptance-report.json`.

## Identity

- Cartridge: `src/dodge/game/dodge.p8`
- Cartridge SHA-256: `7453a9658fd32577385ad72672a54ad84ff70567fadbde75ba6634aa5cc684a3`
- Pemsa SHA-256: `19823433dd0fa75568207408d09646960113858a43c159930388b7bf62b259e4`
- Rust: `rustc 1.97.1`, Cargo `1.97.0`
- Cargo.lock SHA-256: `2c3794e25512263d8ceb11c0b4075b7eb61556f61c9d2a99209622c87fe25ef0`
- Capture boundary: canonical full-draw, post-update/post-draw, indexed 128 × 128

## Evidence

| group | fixtures | frames | parity |
|---|---:|---:|---|
| golden | 4 | 1,155 | match |
| held-out | 4 | 1,004 | match |

The only first mismatch found during the gate was seed 7 frame 135 in
`particles[0].x`. The targeted fixture records the source span and values;
V159 corrected the Pemsa `f32`-input/double-math/`f32`-output trig bridge, and
the rerun is now exact.

Restore/replay, full-state decoder coverage, source-map closure, strict
Clippy, formatting, native tests, and the legacy Python suite are green. The
legacy suite was run through the verified devenv boundary with the SDL2 and
libXext runtime paths available to X11 subprocesses.

## Handoff

P5 may present `Snapshot` pixels without adding game rules. P6 may build its
batch and Python boundaries over `NativeGame`, `FullState`, `Snapshot`, and
canonical restore; it must preserve the accepted serial semantics.

Human visual signoff remains explicitly pending because automated indexed
frame equality is evidence of pixel parity, not a substitute for the owner's
side-by-side review.
