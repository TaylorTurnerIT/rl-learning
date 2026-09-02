use crate::{
    Action, Button, CoreError, InputState, LifecycleState, Mode, NativeConfig, PicoFixed, PicoRng,
};

/// Result of one native simulation frame before Snapshot is added in T57.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct FrameResult {
    pub frame: u32,
    pub mode: Mode,
    pub input_mask: u8,
    pub previous_input_mask: u8,
    pub game_ready: bool,
    pub started: bool,
    pub dead: bool,
    pub done: bool,
    pub reward: PicoFixed,
}

/// Deterministic, engine-free native game boundary.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct NativeGame {
    config: NativeConfig,
    lifecycle: LifecycleState,
    input: InputState,
    rng: PicoRng,
}

impl NativeGame {
    pub fn new(config: NativeConfig) -> Self {
        Self {
            rng: PicoRng::new(config.seed),
            config,
            lifecycle: LifecycleState::new(),
            input: InputState::new(),
        }
    }

    pub fn reset(&mut self) {
        self.lifecycle = LifecycleState::new();
        self.input = InputState::new();
        self.rng.seed(self.config.seed);
    }

    pub fn advance_frame(&mut self, input_mask: u8) -> Result<FrameResult, CoreError> {
        InputState::validate_mask(input_mask)?;
        self.input.advance(input_mask)?;
        let _ = self.rng.rnd(PicoFixed::from_int(32));
        let _ = self.rng.rnd(PicoFixed::from_int(32));
        self.lifecycle.advance(self.input);
        Ok(self.result())
    }

    pub fn step(&mut self, action: Action, frames: u32) -> Result<FrameResult, CoreError> {
        if frames == 0 {
            return Err(CoreError::InvalidFrameCount(frames));
        }
        let mut result = self.result();
        for _ in 0..frames {
            result = self.advance_frame(action.mask())?;
            if result.done {
                break;
            }
        }
        Ok(result)
    }

    pub const fn lifecycle(&self) -> LifecycleState {
        self.lifecycle
    }

    pub const fn input(&self) -> InputState {
        self.input
    }

    pub const fn rng_checkpoint(&self) -> crate::RngCheckpoint {
        self.rng.checkpoint()
    }

    fn result(&self) -> FrameResult {
        FrameResult {
            frame: self.lifecycle.frame,
            mode: self.lifecycle.mode,
            input_mask: self.input.current_mask(),
            previous_input_mask: self.input.previous_mask(),
            game_ready: self.lifecycle.game_ready,
            started: self.lifecycle.started,
            dead: self.lifecycle.dead,
            done: self.lifecycle.mode == Mode::Terminal,
            reward: PicoFixed::ZERO,
        }
    }

    #[allow(dead_code)]
    const fn start_button_is_pressed(&self) -> bool {
        self.input.btnp(Button::X)
    }
}

#[cfg(test)]
mod tests {
    use super::NativeGame;
    use crate::{Action, BUTTON_X_MASK, CoreError, Mode, NativeConfig};

    #[test]
    fn reset_starts_at_menu_frame_zero() {
        let game = NativeGame::new(NativeConfig::new(42));
        assert_eq!(game.lifecycle().frame, 0);
        assert_eq!(game.lifecycle().mode, Mode::Menu);
        assert!(!game.lifecycle().started);
    }

    #[test]
    fn v102_start_crosses_transition_on_observed_thirteenth_frame() {
        let mut game = NativeGame::new(NativeConfig::new(42));
        let first = game.advance_frame(BUTTON_X_MASK);
        assert!(first.is_ok());
        assert_eq!(first.map(|value| value.mode), Ok(Mode::TransitionToGame));
        for _ in 0..11 {
            assert!(game.advance_frame(BUTTON_X_MASK).is_ok());
        }
        assert_eq!(game.lifecycle().frame, 12);
        assert_eq!(game.lifecycle().mode, Mode::TransitionToGame);
        let ready = game.advance_frame(0);
        assert!(ready.is_ok());
        assert_eq!(ready.map(|value| value.frame), Ok(13));
        assert_eq!(ready.map(|value| value.mode), Ok(Mode::Game));
        assert!(game.lifecycle().game_ready);
    }

    #[test]
    fn action_step_advances_exact_frames_and_neutral_is_zero_mask() {
        let mut game = NativeGame::new(NativeConfig::default());
        let result = game.step(Action::Neutral, 4);
        assert!(result.is_ok());
        assert_eq!(result.map(|value| value.frame), Ok(4));
        assert_eq!(result.map(|value| value.input_mask), Ok(0));
        assert_eq!(result.map(|value| value.previous_input_mask), Ok(0));
    }

    #[test]
    fn invalid_input_and_zero_step_fail_without_mutation() {
        let mut game = NativeGame::new(NativeConfig::default());
        assert_eq!(
            game.advance_frame(64),
            Err(CoreError::InvalidButtonMask(64))
        );
        assert_eq!(game.lifecycle().frame, 0);
        assert_eq!(
            game.step(Action::Neutral, 0),
            Err(CoreError::InvalidFrameCount(0))
        );
        assert_eq!(game.lifecycle().frame, 0);
    }
}
