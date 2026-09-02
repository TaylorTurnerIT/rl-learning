use std::fmt::{Display, Formatter};

/// Errors returned before or during a native state transition.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CoreError {
    InvalidButtonMask(u8),
    InvalidButtonIndex(u8),
    InvalidActionName,
    InvalidFrameCount(u32),
    DivisionByZero,
    InvalidRngCheckpoint,
}

impl Display for CoreError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidButtonMask(mask) => {
                write!(formatter, "invalid PICO-8 button mask: {mask}")
            }
            Self::InvalidButtonIndex(index) => {
                write!(formatter, "invalid PICO-8 button index: {index}")
            }
            Self::InvalidActionName => formatter.write_str("invalid Dodge action name"),
            Self::InvalidFrameCount(frames) => {
                write!(formatter, "invalid native frame count: {frames}")
            }
            Self::DivisionByZero => formatter.write_str("PICO-8 fixed division by zero"),
            Self::InvalidRngCheckpoint => formatter.write_str("invalid PICO-8 RNG checkpoint"),
        }
    }
}

impl std::error::Error for CoreError {}
