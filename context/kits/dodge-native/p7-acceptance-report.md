# P7 acceptance: proof and performance hardening

Status: `accepted_automated_owner_review_pending`

The native Dodge implementation passes the defined automated P7 gates. The
report keeps local formal properties, finite native/Pemsa conformance,
reproducibility, performance, and human visual review as separate claims.

## Proof

Kani verified six bounded properties with zero failures and zero resource
exhaustion: framebuffer coordinates, palette indexes, the nine-action domain,
board indexes, fixed high-score slots, and board-buffer slots.

This is a real proof of those Rust contracts. It is not a proof that the whole
Rust game is mathematically equivalent to the PICO-8/Pemsa implementation.
The retained machine-readable result is
`p7-kani-results.json`.

## Finite conformance

The accepted P4 corpus has eight fixtures and 2,159 compared frames. The P7
retained fuzz corpus has four full-draw fixtures (seeds 3, 17, 41, and 89) and
1,042 compared frames. Every compared frame matched decoded state, reward,
termination, events, hashes, and the 128 × 128 palette-index framebuffer.

The P4 and P7 fuzz corpora cover 3,201 frames. The fuzz report is
`p7-fuzz-report.json`; the P4 report is `p4-acceptance-report.json`.

The read-only GA SQLite gate selected episode 4 as the longest replayable
record by stored headless survival. Its 2,205-frame no-draw replay reproduced
the database result, and its 371-frame full-draw native/Pemsa run matched every
state field and indexed pixel. An explicit episode 113 replay reached the
static pattern and terminal path: 562 full-draw frames matched exactly. The
headless and full-draw runs are intentionally separate because the cartridge's
draw-transition path performs source-side updates. Evidence is retained in
`p7-ga-full-run-report.json` and `p7-ga-rendered-pattern-report.json`.

## Reproducibility

Two identical eight-lane native batch runs produced 140 canonical records per
run with byte-identical result records, including full state, board, pixels,
events, and reset behavior. The shared run digest is
`b2fe14c327a7dbc4b81eef0a5c3d9ca7c2573db39dec5cb4dd1c0f20a79c12e1`.

## Performance

The P7 full-observation regression workload uses 32 lanes, 1,024 lane steps,
four-frame decisions, full state, and pixels. Its median was 0.6306 seconds
(1,624 lane-steps/second), within the 0.7860-second P7 limit. Population
standard deviation was 0.02675 seconds, within the 0.05-second limit. The
accepted P6 median was 0.6288 seconds; this sample is slightly slower but
remains well inside the accepted regression budget.

Criterion retained a 570.675 ms median point estimate and an 18.976 ms
standard-deviation point estimate. Raw results and the accepted budget are in
`p7-benchmark-report.json`.

## Regression and visual boundary

The legacy suite passed 211 tests. The focused native/differential suite passed
18 tests, native GA loader tests passed 4 tests, native PPO smoke passed 2
tests, the strict native suite passed 62 tests, and all 6 Kani harnesses
passed. These results are retained in
`p7-legacy-regression.json`.

Macroquad’s lossless 128 × 128 capture also matched the native indexed pixels.
The retained visual packet `p7-visual-review.json` compares 21 representative
menu, settings, transition, gameplay, pattern, particle, and death frames;
all 21 have zero differing indexed pixels. Owner review is still required for
menu, settings, gameplay, patterns, particles, death, and transition
presentation. Automated pixel equality is evidence for the renderer boundary,
not owner approval.

Deferred claims remain audio waveform parity, cross-platform floating-point
identity, infinite-input proof, and full-game mathematical equivalence.
