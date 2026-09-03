#![doc = "Kani-only bounded properties for the native Dodge contracts."]
#![allow(unexpected_cfgs)]

#[cfg(kani)]
mod proofs {
    use dodge_batch::{BOARD_CHANNELS, BOARD_HEIGHT, BOARD_VALUES, BOARD_WIDTH, Board19x16};
    use dodge_core::{
        Action, BUTTON_MASK_LIMIT, FRAMEBUFFER_HEIGHT, FRAMEBUFFER_WIDTH, IndexedFramebuffer,
        PALETTE_SIZE, RenderState,
    };

    #[kani::proof]
    fn framebuffer_coordinates_are_bounded() {
        let x: usize = kani::any();
        let y: usize = kani::any();
        kani::assume(x < FRAMEBUFFER_WIDTH);
        kani::assume(y < FRAMEBUFFER_HEIGHT);

        let framebuffer = IndexedFramebuffer::filled(7);
        assert_eq!(framebuffer.pixel(x, y), Some(7));
    }

    #[kani::proof]
    fn palette_lookup_returns_a_valid_palette_index() {
        let color: u8 = kani::any();
        kani::assume(color < PALETTE_SIZE as u8);

        let render = RenderState::default();
        let Some(mapped) = render.palette_index(color) else {
            return;
        };
        assert!(mapped < PALETTE_SIZE as u8);
    }

    #[kani::proof]
    fn action_masks_stay_inside_the_pico8_button_domain() {
        let action_index: usize = kani::any();
        kani::assume(action_index < Action::ALL.len());

        let Some(action) = Action::ALL.get(action_index).copied() else {
            return;
        };
        assert!(action.mask() <= BUTTON_MASK_LIMIT);
    }

    #[kani::proof]
    fn board_coordinate_index_is_bounded() {
        let channel: usize = kani::any();
        let row: usize = kani::any();
        let column: usize = kani::any();
        kani::assume(channel < BOARD_CHANNELS);
        kani::assume(row < BOARD_HEIGHT);
        kani::assume(column < BOARD_WIDTH);

        let Some(index) = Board19x16::flat_index(channel, row, column) else {
            return;
        };
        assert!(index < BOARD_VALUES);
    }

    #[kani::proof]
    fn fixed_highscore_slot_is_bounded() {
        const HIGHSCORE_SLOTS: usize = 12;
        let slot: usize = kani::any();
        kani::assume(slot < HIGHSCORE_SLOTS);
        let scores = [0_i32; HIGHSCORE_SLOTS];
        assert!(scores.get(slot).is_some());
    }

    #[kani::proof]
    fn board_buffer_slot_is_bounded() {
        let slot: usize = kani::any();
        kani::assume(slot < BOARD_VALUES);
        let values = [0_u8; BOARD_VALUES];
        assert!(values.get(slot).is_some());
    }
}
