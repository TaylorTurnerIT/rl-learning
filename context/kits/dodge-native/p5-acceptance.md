# P5 acceptance: Macroquad viewer

Status: `accepted_automated_owner_visual_review_pending`

`dodge-viewer` is a separate Macroquad 0.4.16 crate. It consumes only the
accepted native runner trace and `dodge_core::Snapshot`; it does not implement
Dodge rules or draw entities independently. The source remains a 128 × 128
palette-index buffer until the final RGBA texture upload, which uses nearest
filtering and an integer centered viewport.

The viewer has a live native mode, 60 Hz trace replay, Escape-only replay exit,
read-only hash/state debug text, and lossless indexed PGM or RGB PPM capture.
Six viewer tests, the full native workspace test/lint/format gate, and a live
Xvfb smoke passed. The smoke discovered the window by title, exited with
Escape, and verified all 16,384 captured index bytes against the native trace.

P5 is technically ready for P6 and P7. Human side-by-side visual approval is
still recorded as pending rather than inferred from automated equality.
