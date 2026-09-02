use crate::PicoFixed;

/// Mutable player fields used by the P3 movement slice.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PlayerState {
    pub x: PicoFixed,
    pub y: PicoFixed,
    pub vx: PicoFixed,
    pub vy: PicoFixed,
    pub size: PicoFixed,
}

impl PlayerState {
    pub const fn new() -> Self {
        Self {
            x: PicoFixed::from_int(64),
            y: PicoFixed::from_int(64),
            vx: PicoFixed::ZERO,
            vy: PicoFixed::ZERO,
            size: PicoFixed::from_int(4),
        }
    }
}

impl Default for PlayerState {
    fn default() -> Self {
        Self::new()
    }
}

/// Typed normal-enemy fields used by the P3 representative hazard slice.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct EnemyState {
    pub x: PicoFixed,
    pub y: PicoFixed,
    pub vx: PicoFixed,
    pub vy: PicoFixed,
    pub size: PicoFixed,
    pub max_size: PicoFixed,
    pub personality: i8,
    pub speed: PicoFixed,
    pub inside: bool,
    pub is_dying: bool,
}

impl EnemyState {
    pub const fn normal(x: PicoFixed, y: PicoFixed, max_size: PicoFixed) -> Self {
        Self {
            x,
            y,
            vx: PicoFixed::ZERO,
            vy: PicoFixed::ZERO,
            size: PicoFixed::ONE,
            max_size,
            personality: 0,
            speed: PicoFixed::ONE,
            inside: false,
            is_dying: false,
        }
    }
}
