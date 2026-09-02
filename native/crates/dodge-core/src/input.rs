use crate::CoreError;

pub const BUTTON_MASK_LIMIT: u8 = 0b11_1111;

/// PICO-8 button identities used by the cartridge.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Button {
    Left = 0,
    Right = 1,
    Up = 2,
    Down = 3,
    O = 4,
    X = 5,
}

impl Button {
    pub const fn index(self) -> u8 {
        self as u8
    }

    pub const fn mask(self) -> u8 {
        1 << self.index()
    }
}

/// Current/previous input masks and mouse/stat compatibility values.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct InputState {
    current_mask: u8,
    previous_mask: u8,
}

impl InputState {
    pub const fn new() -> Self {
        Self {
            current_mask: 0,
            previous_mask: 0,
        }
    }

    pub const fn current_mask(self) -> u8 {
        self.current_mask
    }

    pub const fn previous_mask(self) -> u8 {
        self.previous_mask
    }

    pub fn validate_mask(mask: u8) -> Result<(), CoreError> {
        if mask > BUTTON_MASK_LIMIT {
            Err(CoreError::InvalidButtonMask(mask))
        } else {
            Ok(())
        }
    }

    pub fn advance(&mut self, mask: u8) -> Result<(), CoreError> {
        Self::validate_mask(mask)?;
        self.previous_mask = self.current_mask;
        self.current_mask = mask;
        Ok(())
    }

    pub const fn btn(self, button: Button) -> bool {
        self.current_mask & button.mask() != 0
    }

    pub const fn btnp(self, button: Button) -> bool {
        self.btn(button) && self.previous_mask & button.mask() == 0
    }
}

impl Default for InputState {
    fn default() -> Self {
        Self::new()
    }
}
