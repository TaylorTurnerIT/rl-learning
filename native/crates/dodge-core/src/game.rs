use crate::{
    Action, Button, CoreError, EnemyState, FullState, InputState, LifecycleState, Mode,
    NativeConfig, PicoFixed, PicoRng, PlayerState, Snapshot, pico_mid,
};

const PLAYER_MIN: PicoFixed = PicoFixed::from_int(2);
const PLAYER_MAX: PicoFixed = PicoFixed::from_int(125);
const PLAYER_SPEED: PicoFixed = PicoFixed::from_raw(32_768);
const PLAYER_FRICTION: PicoFixed = PicoFixed::from_raw(52_428);
const ENEMY_FRICTION: PicoFixed = PicoFixed::from_raw(64_880);
const ENEMY_ACCELERATION: PicoFixed = PicoFixed::from_raw(655);
const ENEMY_INITIAL_MAX_SIZE: PicoFixed = PicoFixed::from_int(3);
const SPAWN_EDGE: PicoFixed = PicoFixed::from_int(-10);
const SPAWN_EDGE_FAR: PicoFixed = PicoFixed::from_int(138);
const SPAWN_INTERVAL: PicoFixed = PicoFixed::from_int(60);

/// Result of one native simulation frame and its canonical observation.
#[derive(Clone, Debug, Eq, PartialEq)]
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
    pub snapshot: Snapshot,
}

/// Deterministic, engine-free native game boundary.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct NativeGame {
    config: NativeConfig,
    lifecycle: LifecycleState,
    input: InputState,
    rng: PicoRng,
    player: PlayerState,
    enemies: Vec<EnemyState>,
    enemy_timer: PicoFixed,
    friendly_timer: u32,
    enemy_max_size: PicoFixed,
    speed: PicoFixed,
    freeze_rate: PicoFixed,
    score: PicoFixed,
    survival_frames: u32,
    transition_render_y: i16,
}

impl NativeGame {
    pub fn new(config: NativeConfig) -> Self {
        Self {
            rng: PicoRng::new(config.seed),
            config,
            lifecycle: LifecycleState::new(),
            input: InputState::new(),
            player: PlayerState::new(),
            enemies: Vec::new(),
            enemy_timer: PicoFixed::ZERO,
            friendly_timer: 0,
            enemy_max_size: ENEMY_INITIAL_MAX_SIZE,
            speed: PicoFixed::ONE,
            freeze_rate: PicoFixed::ONE,
            score: PicoFixed::ZERO,
            survival_frames: 0,
            transition_render_y: -128,
        }
    }

    pub fn reset(&mut self) -> Snapshot {
        self.lifecycle = LifecycleState::new();
        self.input = InputState::new();
        self.rng.seed(self.config.seed);
        self.player = PlayerState::new();
        self.enemies.clear();
        self.enemy_timer = PicoFixed::ZERO;
        self.friendly_timer = 0;
        self.enemy_max_size = ENEMY_INITIAL_MAX_SIZE;
        self.speed = PicoFixed::ONE;
        self.freeze_rate = PicoFixed::ONE;
        self.score = PicoFixed::ZERO;
        self.survival_frames = 0;
        self.transition_render_y = -128;
        self.snapshot()
    }

    pub fn advance_frame(&mut self, input_mask: u8) -> Result<FrameResult, CoreError> {
        self.advance_frame_with_post_mask(input_mask, input_mask)
    }

    pub fn advance_frame_with_post_mask(
        &mut self,
        input_mask: u8,
        post_frame_mask: u8,
    ) -> Result<FrameResult, CoreError> {
        InputState::validate_mask(input_mask)?;
        InputState::validate_mask(post_frame_mask)?;
        self.input.advance(input_mask)?;
        let mode_before = self.lifecycle.mode;
        let start_pressed = mode_before == Mode::Menu && self.input.btnp(Button::X);
        self.lifecycle.advance(self.input);
        if start_pressed {
            self.transition_render_y = -108;
        }

        let game_update_from_draw = match mode_before {
            Mode::Menu => false,
            Mode::TransitionToGame => {
                self.transition_render_y += 20;
                self.transition_render_y >= 0
            }
            Mode::Game => true,
            Mode::Terminal => false,
        };
        if game_update_from_draw {
            self.update_game_frame();
        }
        let _ = self.rng.rnd(PicoFixed::from_int(32));
        let _ = self.rng.rnd(PicoFixed::from_int(32));
        let reward = if mode_before == Mode::Game && !self.lifecycle.dead {
            self.survival_frames += 1;
            PicoFixed::ONE
        } else {
            PicoFixed::ZERO
        };
        self.input.finalize_frame(post_frame_mask);
        Ok(self.result(reward))
    }

    pub fn step(&mut self, action: Action, frames: u32) -> Result<FrameResult, CoreError> {
        if frames == 0 {
            return Err(CoreError::InvalidFrameCount(frames));
        }
        let mut result = self.result(PicoFixed::ZERO);
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

    pub const fn player(&self) -> PlayerState {
        self.player
    }

    pub fn enemies(&self) -> &[EnemyState] {
        self.enemies.as_slice()
    }

    pub const fn score(&self) -> PicoFixed {
        self.score
    }

    pub const fn survival_frames(&self) -> u32 {
        self.survival_frames
    }

    pub const fn transition_render_y(&self) -> i16 {
        self.transition_render_y
    }

    pub(crate) fn full_state(&self) -> FullState {
        FullState {
            seed: self.config.seed,
            lifecycle: self.lifecycle,
            input: self.input,
            rng: self.rng.checkpoint(),
            player: self.player,
            enemies: self.enemies.clone(),
            enemy_timer: self.enemy_timer,
            friendly_timer: self.friendly_timer,
            enemy_max_size: self.enemy_max_size,
            speed: self.speed,
            freeze_rate: self.freeze_rate,
            score: self.score,
            survival_frames: self.survival_frames,
            transition_render_y: self.transition_render_y,
        }
    }

    pub fn snapshot(&self) -> Snapshot {
        Snapshot::from_game(self)
    }

    fn update_game_frame(&mut self) {
        self.update_fyou();
        self.collision_check();
        if self.lifecycle.dead {
            return;
        }
        self.update_player();
        self.spawn_enemies();
        self.update_enemies();
    }

    fn update_fyou(&mut self) {
        let in_center = self
            .player
            .x
            .add(self.player.size)
            .sub(PicoFixed::from_int(2))
            > PicoFixed::from_int(60)
            && self
                .player
                .y
                .add(self.player.size)
                .sub(PicoFixed::from_int(2))
                > PicoFixed::from_int(60)
            && self
                .player
                .x
                .sub(self.player.size)
                .add(PicoFixed::from_int(2))
                < PicoFixed::from_int(68)
            && self
                .player
                .y
                .sub(self.player.size)
                .add(PicoFixed::from_int(2))
                < PicoFixed::from_int(68);
        if !in_center {
            self.friendly_timer = 0;
            return;
        }
        self.friendly_timer += 1;
        if self.friendly_timer > 50 {
            self.friendly_timer = 0;
            self.add_corner_enemies();
        }
    }

    fn add_corner_enemies(&mut self) {
        for (x, y) in [
            (SPAWN_EDGE, SPAWN_EDGE),
            (SPAWN_EDGE_FAR, SPAWN_EDGE),
            (SPAWN_EDGE, SPAWN_EDGE_FAR),
            (SPAWN_EDGE_FAR, SPAWN_EDGE_FAR),
        ] {
            self.enemies
                .push(EnemyState::normal(x, y, self.enemy_max_size));
        }
    }

    fn collision_check(&mut self) {
        if self.lifecycle.dead {
            return;
        }
        let player = self.player;
        let collision = self.enemies.iter().any(|enemy| {
            player.x.add(player.size).sub(PicoFixed::ONE) > enemy.x
                && player.y.add(player.size).sub(PicoFixed::ONE) > enemy.y
                && player.x.sub(player.size).add(PicoFixed::ONE) < enemy.x.add(enemy.size)
                && player.y.sub(player.size).add(PicoFixed::ONE) < enemy.y.add(enemy.size)
        });
        if collision {
            self.rng.consume(11, PicoFixed::ONE);
            self.lifecycle.mark_dead();
        }
    }

    fn update_player(&mut self) {
        self.player.vx = self.player.vx.mul_fixed(PLAYER_FRICTION);
        self.player.vy = self.player.vy.mul_fixed(PLAYER_FRICTION);
        if self.input.btn(Button::Left) {
            self.player.vx = self.player.vx.sub(PLAYER_SPEED);
        }
        if self.input.btn(Button::Right) {
            self.player.vx = self.player.vx.add(PLAYER_SPEED);
        }
        if self.input.btn(Button::Up) {
            self.player.vy = self.player.vy.sub(PLAYER_SPEED);
        }
        if self.input.btn(Button::Down) {
            self.player.vy = self.player.vy.add(PLAYER_SPEED);
        }
        self.player.x = pico_mid(PLAYER_MIN, PLAYER_MAX, self.player.x.add(self.player.vx));
        self.player.y = pico_mid(PLAYER_MIN, PLAYER_MAX, self.player.y.add(self.player.vy));
    }

    fn spawn_enemies(&mut self) {
        self.enemy_timer = self.enemy_timer.add(PicoFixed::ONE);
        if self.enemy_timer <= SPAWN_INTERVAL {
            return;
        }
        self.enemy_timer = PicoFixed::ZERO;
        let _size_roll = self.rng.rnd(PicoFixed::from_int(100));
        let _personality_roll = self.rng.rnd(PicoFixed::from_int(100));
        let (x, y) = self.nearest_corner();
        self.enemies
            .push(EnemyState::normal(x, y, self.enemy_max_size));
    }

    fn nearest_corner(&self) -> (PicoFixed, PicoFixed) {
        let corners = [
            (SPAWN_EDGE, SPAWN_EDGE),
            (SPAWN_EDGE_FAR, SPAWN_EDGE),
            (SPAWN_EDGE, SPAWN_EDGE_FAR),
            (SPAWN_EDGE_FAR, SPAWN_EDGE_FAR),
        ];
        let mut best = (SPAWN_EDGE, SPAWN_EDGE);
        let mut best_distance = f64::INFINITY;
        for (x, y) in corners {
            let dx = x.to_double() - self.player.x.to_double();
            let dy = y.to_double() - self.player.y.to_double();
            let distance = dx * dx + dy * dy;
            if distance < best_distance {
                best = (x, y);
                best_distance = distance;
            }
        }
        best
    }

    fn update_enemies(&mut self) {
        let player = self.player;
        for enemy in &mut self.enemies {
            if enemy.size < enemy.max_size && enemy.personality != -1 {
                enemy.size = enemy.size.add(self.freeze_rate);
            }
            if enemy.x < player.x {
                enemy.vx = enemy.vx.mul_fixed(ENEMY_FRICTION).add(ENEMY_ACCELERATION);
            } else if enemy.x > player.x {
                enemy.vx = enemy.vx.mul_fixed(ENEMY_FRICTION).sub(ENEMY_ACCELERATION);
            } else {
                enemy.vx = enemy.vx.mul_fixed(ENEMY_FRICTION);
            }
            if enemy.y < player.y {
                enemy.vy = enemy.vy.mul_fixed(ENEMY_FRICTION).add(ENEMY_ACCELERATION);
            } else if enemy.y > player.y {
                enemy.vy = enemy.vy.mul_fixed(ENEMY_FRICTION).sub(ENEMY_ACCELERATION);
            } else {
                enemy.vy = enemy.vy.mul_fixed(ENEMY_FRICTION);
            }
            enemy.x = enemy.x.add(
                enemy
                    .vx
                    .mul_fixed(self.speed)
                    .mul_fixed(self.freeze_rate)
                    .mul_fixed(enemy.speed),
            );
            enemy.y = enemy.y.add(
                enemy
                    .vy
                    .mul_fixed(self.speed)
                    .mul_fixed(self.freeze_rate)
                    .mul_fixed(enemy.speed),
            );
            enemy.inside = enemy.x >= PicoFixed::from_int(1)
                && enemy.x <= PicoFixed::from_int(128)
                && enemy.y >= PicoFixed::from_int(0)
                && enemy.y <= PicoFixed::from_int(128);
        }
    }

    fn result(&self, reward: PicoFixed) -> FrameResult {
        FrameResult {
            frame: self.lifecycle.frame,
            mode: self.lifecycle.mode,
            input_mask: self.input.current_mask(),
            previous_input_mask: self.input.previous_mask(),
            game_ready: self.lifecycle.game_ready,
            started: self.lifecycle.started,
            dead: self.lifecycle.dead,
            done: self.lifecycle.dead,
            reward,
            snapshot: self.snapshot(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::NativeGame;
    use crate::{Action, BUTTON_X_MASK, CoreError, EnemyState, Mode, NativeConfig, PicoFixed};

    fn start_game(game: &mut NativeGame) {
        assert!(game.advance_frame(BUTTON_X_MASK).is_ok());
        for _ in 1..13 {
            assert!(game.advance_frame(0).is_ok());
        }
        assert_eq!(game.lifecycle().mode, Mode::Game);
    }

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
        assert_eq!(ready.as_ref().map(|value| value.frame), Ok(13));
        assert_eq!(ready.as_ref().map(|value| value.mode), Ok(Mode::Game));
        assert!(game.lifecycle().game_ready);
    }

    #[test]
    fn action_step_advances_exact_frames_and_neutral_is_zero_mask() {
        let mut game = NativeGame::new(NativeConfig::default());
        let result = game.step(Action::Neutral, 4);
        assert!(result.is_ok());
        assert_eq!(result.as_ref().map(|value| value.frame), Ok(4));
        assert_eq!(result.as_ref().map(|value| value.input_mask), Ok(0));
        assert_eq!(
            result.as_ref().map(|value| value.previous_input_mask),
            Ok(0)
        );
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

    #[test]
    fn player_movement_uses_fixed_friction_and_clamped_bounds() {
        let mut game = NativeGame::new(NativeConfig::default());
        start_game(&mut game);
        let first = game.advance_frame(1);
        assert!(first.is_ok());
        assert_eq!(game.player().x, PicoFixed::from_f32(63.5));
        assert_eq!(game.player().vx, PicoFixed::from_f32(-0.5));
        let second = game.advance_frame(1);
        assert!(second.is_ok());
        assert_eq!(game.player().x, PicoFixed::from_raw(4_102_554));
        assert_eq!(game.player().vx, PicoFixed::from_f32(-0.9));
    }

    #[test]
    fn first_friendly_spawn_and_normal_enemy_match_seed_42_slice() {
        let mut game = NativeGame::new(NativeConfig::new(42));
        start_game(&mut game);
        for _ in 0..44 {
            assert!(game.advance_frame(0).is_ok());
        }
        assert_eq!(game.lifecycle().frame, 57);
        assert_eq!(game.enemies().len(), 4);
        let first = game.enemies().first().copied();
        assert_eq!(
            first.map(|enemy| enemy.x),
            Some(PicoFixed::from_raw(-654_705))
        );
        assert_eq!(
            first.map(|enemy| enemy.y),
            Some(PicoFixed::from_raw(-654_705))
        );
        assert_eq!(first.map(|enemy| enemy.size), Some(PicoFixed::from_int(2)));
        assert_eq!(game.score(), PicoFixed::ZERO);
    }

    #[test]
    fn normal_enemy_collision_marks_terminal_without_survival_reward() {
        let mut game = NativeGame::new(NativeConfig::default());
        start_game(&mut game);
        game.enemies.push(EnemyState::normal(
            PicoFixed::from_int(61),
            PicoFixed::from_int(64),
            PicoFixed::from_int(3),
        ));
        let result = game.advance_frame(0);
        assert_eq!(result.as_ref().map(|value| value.done), Ok(true));
        assert!(game.lifecycle().dead);
        assert_eq!(game.survival_frames(), 0);
        assert_eq!(
            result.as_ref().map(|value| value.reward),
            Ok(PicoFixed::ZERO)
        );
    }
}
