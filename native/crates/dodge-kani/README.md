# Kani properties

This crate contains bounded properties for the public native contracts. It is
not a second game implementation and it does not claim to prove equivalence
with Pemsa. The application workspace remains on Rust 1.97.1; Kani uses the
separate nightly selected by its installed release bundle.

Run the named properties from `native/` after `cargo kani setup`:

```text
cargo kani -p dodge-kani --harness framebuffer_coordinates_are_bounded
cargo kani -p dodge-kani --harness palette_lookup_returns_a_valid_palette_index
cargo kani -p dodge-kani --harness action_masks_stay_inside_the_pico8_button_domain
cargo kani -p dodge-kani --harness board_coordinate_index_is_bounded
cargo kani -p dodge-kani --harness fixed_highscore_slot_is_bounded
cargo kani -p dodge-kani --harness board_buffer_slot_is_bounded
```

Each harness constrains its symbolic inputs before calling the corresponding
checked production boundary. A successful Kani result is a local property
proof. Differential Pemsa traces, visual review, and benchmark evidence remain
separate gates.
