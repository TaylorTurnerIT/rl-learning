use crate::{Button, InputState};

/// Native mode corresponding to the cartridge's update callback.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Mode {
    Menu,
    TransitionToGame,
    Game,
    Terminal,
}

/// Explicit lifecycle state for the P3 menu/start boundary.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct LifecycleState {
    pub frame: u32,
    pub mode: Mode,
    pub transition_y: i16,
    pub started: bool,
    pub game_ready: bool,
    pub dead: bool,
}

impl LifecycleState {
    pub const fn new() -> Self {
        Self {
            frame: 0,
            mode: Mode::Menu,
            transition_y: -128,
            started: false,
            game_ready: false,
            dead: false,
        }
    }

    pub fn advance(&mut self, input: InputState) {
        match self.mode {
            Mode::Menu => {
                if input.btnp(Button::X) {
                    self.mode = Mode::TransitionToGame;
                    self.started = true;
                    self.transition_y = -128;
                    self.advance_transition();
                }
            }
            Mode::TransitionToGame => {
                self.advance_transition();
            }
            Mode::Game | Mode::Terminal => {}
        }
        self.frame += 1;
    }

    fn advance_transition(&mut self) {
        self.transition_y += 10;
        if self.transition_y >= 0 {
            self.mode = Mode::Game;
            self.game_ready = true;
        }
    }
}

impl Default for LifecycleState {
    fn default() -> Self {
        Self::new()
    }
}
