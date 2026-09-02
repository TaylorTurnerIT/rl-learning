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
const ENEMY_INITIAL_EST: PicoFixed = PicoFixed::ONE;
const SPAWN_EDGE: PicoFixed = PicoFixed::from_int(-10);
const SPAWN_EDGE_FAR: PicoFixed = PicoFixed::from_int(138);
const SPAWN_INTERVAL: PicoFixed = PicoFixed::from_int(60);
const DIFFICULTY_SPEED_TARGET: PicoFixed = PicoFixed::from_int(3);
const DIFFICULTY_EST_TARGET: PicoFixed = PicoFixed::from_raw(14_417);
const DIFFICULTY_SPEED_STEP: PicoFixed = PicoFixed::from_raw(226);
const DIFFICULTY_EST_STEP: PicoFixed = PicoFixed::from_raw(327);
const DIFFICULTY_SPEED_FULL_STEP: PicoFixed = PicoFixed::from_raw(452);
const DIFFICULTY_EST_FULL_STEP: PicoFixed = PicoFixed::from_raw(655);
const SCORE_PER_SHATTER: PicoFixed = PicoFixed::from_raw(32_768);
const ENEMY_SPEED_STEP: PicoFixed = PicoFixed::from_raw(655);
const KAMIKAZE_RADIUS_SQUARED: PicoFixed = PicoFixed::from_int(625);
const LOW_SCORE_PERSONALITY_ROLL_LIMIT: PicoFixed = PicoFixed::from_int(97);
const FULL_PERSONALITY_ROLL_LIMIT: PicoFixed = PicoFixed::from_int(99);
const INITIAL_PATTERN_DELAY_FRAMES: u32 = 420;
const INITIAL_PATTERN_RND_PREFIX_LIMITS: [PicoFixed; 20] = [
    PicoFixed::from_int(16),
    PicoFixed::from_int(16),
    PicoFixed::from_int(10),
    PicoFixed::from_int(10),
    PicoFixed::from_int(23),
    PicoFixed::from_int(20),
    PicoFixed::from_int(10),
    PicoFixed::from_int(10),
    PicoFixed::from_int(60),
    PicoFixed::from_int(10),
    PicoFixed::from_int(5),
    PicoFixed::from_int(8),
    PicoFixed::from_int(18),
    PicoFixed::from_int(12),
    PicoFixed::from_int(5),
    PicoFixed::from_int(5),
    PicoFixed::from_int(40),
    PicoFixed::from_int(14),
    PicoFixed::from_int(10),
    PicoFixed::from_int(10),
];
const INITIAL_PATTERN_BRANCH_LIMITS: [PicoFixed; 4] = [
    PicoFixed::from_int(24),
    PicoFixed::from_int(40),
    PicoFixed::from_int(10),
    PicoFixed::from_int(10),
];
const INITIAL_PATTERN_BRANCH_THRESHOLD: PicoFixed = PicoFixed::from_raw(32_768);

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
    pub events: Vec<FrameEvent>,
    pub snapshot: Snapshot,
}

/// Source-visible side effects emitted at one completed frame boundary.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum FrameEvent {
    EnemySpawn,
    Collision,
    Death,
    PatternActive,
    Terminal,
}

impl FrameEvent {
    pub const fn name(self) -> &'static str {
        match self {
            Self::EnemySpawn => "enemy_spawn",
            Self::Collision => "collision",
            Self::Death => "death",
            Self::PatternActive => "pattern_active",
            Self::Terminal => "terminal",
        }
    }
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
    enemy_est: PicoFixed,
    friendly_timer: u32,
    enemy_max_size: PicoFixed,
    speed: PicoFixed,
    freeze_rate: PicoFixed,
    pattern_timer: u32,
    pattern_active: bool,
    score: PicoFixed,
    survival_frames: u32,
    transition_render_y: i16,
}

impl NativeGame {
    pub fn new(config: NativeConfig) -> Self {
        let mut rng = PicoRng::new(config.seed);
        consume_initial_pattern_random_state(&mut rng);
        Self {
            rng,
            config,
            lifecycle: LifecycleState::new(),
            input: InputState::new(),
            player: PlayerState::new(),
            enemies: Vec::new(),
            enemy_timer: PicoFixed::ZERO,
            enemy_est: ENEMY_INITIAL_EST,
            friendly_timer: 0,
            enemy_max_size: ENEMY_INITIAL_MAX_SIZE,
            speed: PicoFixed::ONE,
            freeze_rate: PicoFixed::ONE,
            pattern_timer: 0,
            pattern_active: false,
            score: PicoFixed::ZERO,
            survival_frames: 0,
            transition_render_y: -128,
        }
    }

    pub fn reset(&mut self) -> Snapshot {
        self.lifecycle = LifecycleState::new();
        self.input = InputState::new();
        self.rng.seed(self.config.seed);
        consume_initial_pattern_random_state(&mut self.rng);
        self.player = PlayerState::new();
        self.enemies.clear();
        self.enemy_timer = PicoFixed::ZERO;
        self.enemy_est = ENEMY_INITIAL_EST;
        self.friendly_timer = 0;
        self.enemy_max_size = ENEMY_INITIAL_MAX_SIZE;
        self.speed = PicoFixed::ONE;
        self.freeze_rate = PicoFixed::ONE;
        self.pattern_timer = 0;
        self.pattern_active = false;
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
        let mut events = Vec::new();
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
            self.update_game_frame(&mut events);
        }
        let _ = self.rng.rnd(PicoFixed::from_int(32));
        let _ = self.rng.rnd(PicoFixed::from_int(32));
        self.consume_draw_random_state(mode_before);
        if self.pattern_active && mode_before == Mode::Game {
            events.push(FrameEvent::PatternActive);
        }
        if self.lifecycle.dead {
            events.push(FrameEvent::Terminal);
        }
        let reward = if mode_before == Mode::Game && !self.lifecycle.dead {
            self.survival_frames += 1;
            PicoFixed::ONE
        } else {
            PicoFixed::ZERO
        };
        self.input.finalize_frame(post_frame_mask);
        Ok(self.result(reward, events))
    }

    pub fn step(&mut self, action: Action, frames: u32) -> Result<FrameResult, CoreError> {
        if frames == 0 {
            return Err(CoreError::InvalidFrameCount(frames));
        }
        let mut result = self.result(PicoFixed::ZERO, Vec::new());
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
            enemy_est: self.enemy_est,
            friendly_timer: self.friendly_timer,
            enemy_max_size: self.enemy_max_size,
            speed: self.speed,
            freeze_rate: self.freeze_rate,
            pattern_timer: self.pattern_timer,
            pattern_active: self.pattern_active,
            score: self.score,
            survival_frames: self.survival_frames,
            transition_render_y: self.transition_render_y,
        }
    }

    pub fn snapshot(&self) -> Snapshot {
        Snapshot::from_game(self)
    }

    fn update_game_frame(&mut self, events: &mut Vec<FrameEvent>) {
        self.update_fyou(events);
        self.collision_check(events);
        if self.lifecycle.dead {
            return;
        }
        self.update_player();
        self.spawn_enemies(events);
        self.update_enemies();
        self.update_pattern_schedule();
    }

    fn update_fyou(&mut self, events: &mut Vec<FrameEvent>) {
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
            self.add_corner_enemies(events);
        }
    }

    fn add_corner_enemies(&mut self, events: &mut Vec<FrameEvent>) {
        for (x, y) in [
            (SPAWN_EDGE, SPAWN_EDGE),
            (SPAWN_EDGE_FAR, SPAWN_EDGE),
            (SPAWN_EDGE, SPAWN_EDGE_FAR),
            (SPAWN_EDGE_FAR, SPAWN_EDGE_FAR),
        ] {
            self.enemies
                .push(EnemyState::normal(x, y, self.enemy_max_size));
            events.push(FrameEvent::EnemySpawn);
        }
    }

    fn collision_check(&mut self, events: &mut Vec<FrameEvent>) {
        if self.lifecycle.dead {
            return;
        }
        let mut index = 0;
        while index < self.enemies.len() {
            let Some(enemy) = self.enemies.get(index).copied() else {
                break;
            };
            let player = self.player;
            let collision = player.x.add(player.size).sub(PicoFixed::ONE) > enemy.x
                && player.y.add(player.size).sub(PicoFixed::ONE) > enemy.y
                && player.x.sub(player.size).add(PicoFixed::ONE) < enemy.x.add(enemy.size)
                && player.y.sub(player.size).add(PicoFixed::ONE) < enemy.y.add(enemy.size);
            let powerup_collision = player.x.add(player.size) > enemy.x.sub(PicoFixed::from_int(4))
                && player.y.add(player.size) > enemy.y.sub(PicoFixed::from_int(4))
                && player.x.sub(player.size) < enemy.x.add(PicoFixed::from_int(4))
                && player.y.sub(player.size) < enemy.y.add(PicoFixed::from_int(4));
            let collision = if enemy.personality >= 2 {
                powerup_collision
            } else {
                collision
            };
            if collision {
                events.push(FrameEvent::Collision);
                self.enemies.remove(index);
                self.rng.consume(11, PicoFixed::ONE);
                match enemy.personality {
                    2 => self.resolve_explosion_powerup(),
                    3 => self.resolve_freeze_powerup(),
                    4 => self.resolve_size_powerup(),
                    _ => {
                        self.lifecycle.mark_dead();
                        events.push(FrameEvent::Death);
                    }
                }
            } else {
                index += 1;
            }
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

    fn spawn_enemies(&mut self, events: &mut Vec<FrameEvent>) {
        self.enemy_timer = self.enemy_timer.add(PicoFixed::ONE);
        if self.enemy_timer <= self.enemy_est.mul_fixed(SPAWN_INTERVAL) {
            return;
        }
        self.enemy_timer = PicoFixed::ZERO;
        let size_roll = self.rng.rnd(PicoFixed::from_int(100));
        let personality_roll_limit = if self.score <= PicoFixed::from_int(10) {
            LOW_SCORE_PERSONALITY_ROLL_LIMIT
        } else {
            FULL_PERSONALITY_ROLL_LIMIT
        };
        let personality_roll = self.rng.rnd(personality_roll_limit);
        let (x, y) = self.nearest_corner();
        let max_size = enemy_max_size_from_roll(size_roll);
        self.enemy_max_size = max_size;
        self.enemies.push(EnemyState {
            personality: normal_personality_from_roll(personality_roll),
            ..EnemyState::normal(x, y, max_size)
        });
        events.push(FrameEvent::EnemySpawn);
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
        let mut index = 0;
        while index < self.enemies.len() {
            let Some(original) = self.enemies.get(index).copied() else {
                break;
            };
            let x = original.x;
            let y = original.y;
            let local_size = original.size;
            let was_dying = original.is_dying;
            if was_dying {
                self.enemies.remove(index);
                self.resolve_enemy_death(original);
            } else {
                let mut current = original;
                if current.size < current.max_size && current.personality != -1 {
                    current.size = current.size.add(self.freeze_rate);
                }
                if current.personality >= 2 {
                    current.size = PicoFixed::from_int(4);
                }
                current.inside = current.inside || enemy_is_inside(x);
                let Some(slot) = self.enemies.get_mut(index) else {
                    break;
                };
                *slot = current;
            }

            let mut vx = original.vx.mul_fixed(ENEMY_FRICTION);
            let mut vy = original.vy.mul_fixed(ENEMY_FRICTION);
            if original.personality >= 0 {
                if x < player.x {
                    vx = vx.add(ENEMY_ACCELERATION);
                } else if x > player.x {
                    vx = vx.sub(ENEMY_ACCELERATION);
                }
                if y < player.y {
                    vy = vy.add(ENEMY_ACCELERATION);
                } else if y > player.y {
                    vy = vy.sub(ENEMY_ACCELERATION);
                }
            }

            let current_inside = original.inside || enemy_is_inside(x);
            if current_inside && original.personality != -1 {
                if y >= PicoFixed::from_int(129).sub(local_size) {
                    vy = fixed_abs(vy).neg();
                } else if y <= PicoFixed::from_int(-2) {
                    vy = fixed_abs(vy);
                }
                if x >= PicoFixed::from_int(129).sub(local_size) {
                    vx = fixed_abs(vx).neg();
                } else if x <= PicoFixed::from_int(-2) {
                    vx = fixed_abs(vx);
                }
            }

            let current_offset = if original.personality >= 2 {
                PicoFixed::from_int(4)
            } else {
                PicoFixed::ZERO
            };
            for other_index in 0..self.enemies.len() {
                if !was_dying && other_index == index {
                    continue;
                }
                let Some(other) = self.enemies.get(other_index).copied() else {
                    continue;
                };
                let other_offset = if other.personality >= 2 {
                    PicoFixed::from_int(4)
                } else {
                    PicoFixed::ZERO
                };
                if current_inside
                    && other.inside
                    && x.add(local_size) > other.x.sub(other_offset)
                    && y.add(local_size) > other.y.sub(other_offset)
                    && x.sub(current_offset) < other.x.add(other.size)
                    && y.sub(current_offset) < other.y.add(other.size)
                {
                    if !was_dying && let Some(current) = self.enemies.get_mut(index) {
                        current.is_dying = true;
                    }
                    if let Some(other) = self.enemies.get_mut(other_index) {
                        other.is_dying = true;
                    }
                }
            }

            if !was_dying {
                let Some(mut current) = self.enemies.get(index).copied() else {
                    break;
                };
                current.speed = update_enemy_speed(current, player);
                current.x = x.add(
                    vx.mul_fixed(self.speed)
                        .mul_fixed(self.freeze_rate)
                        .mul_fixed(current.speed),
                );
                current.y = y.add(
                    vy.mul_fixed(self.speed)
                        .mul_fixed(self.freeze_rate)
                        .mul_fixed(current.speed),
                );
                current.vx = vx;
                current.vy = vy;
                let Some(slot) = self.enemies.get_mut(index) else {
                    break;
                };
                *slot = current;
                index += 1;
            }
        }
    }

    fn resolve_enemy_death(&mut self, enemy: EnemyState) {
        if enemy.personality != 1 {
            self.rng.consume(11, PicoFixed::ONE);
        }
        self.score = self.score.add(SCORE_PER_SHATTER);
        self.apply_difficulty(true);
    }

    fn resolve_explosion_powerup(&mut self) {
        self.score = self.score.add(PicoFixed::ONE);
        self.apply_difficulty(false);
    }

    fn resolve_freeze_powerup(&mut self) {
        self.score = self.score.add(PicoFixed::ONE);
        self.freeze_rate = PicoFixed::from_f32(0.4);
        self.apply_difficulty(false);
    }

    fn resolve_size_powerup(&mut self) {
        self.player.size = PicoFixed::from_int(2);
        self.score = self.score.add(PicoFixed::ONE);
        self.apply_difficulty(false);
    }

    fn apply_difficulty(&mut self, half: bool) {
        let speed_step = if half {
            DIFFICULTY_SPEED_STEP
        } else {
            DIFFICULTY_SPEED_FULL_STEP
        };
        let enemy_step = if half {
            DIFFICULTY_EST_STEP
        } else {
            DIFFICULTY_EST_FULL_STEP
        };
        self.speed = pico_lerp(self.speed, DIFFICULTY_SPEED_TARGET, speed_step);
        self.enemy_est = pico_lerp(self.enemy_est, DIFFICULTY_EST_TARGET, enemy_step);
    }

    fn update_pattern_schedule(&mut self) {
        if self.pattern_active {
            return;
        }
        self.pattern_timer += 1;
        if self.pattern_timer >= INITIAL_PATTERN_DELAY_FRAMES {
            self.pattern_timer = 0;
            self.pattern_active = true;
            let _ = self.rng.rnd(PicoFixed::ONE);
        }
    }

    fn consume_draw_random_state(&mut self, mode_before: Mode) {
        if self.lifecycle.dead || !matches!(mode_before, Mode::Game | Mode::TransitionToGame) {
            return;
        }
        for enemy in &self.enemies {
            if enemy.personality >= 2 {
                let _ = self.rng.rnd(PicoFixed::ONE);
                let _ = self.rng.rnd(PicoFixed::from_int(15));
            }
        }
    }

    fn result(&self, reward: PicoFixed, events: Vec<FrameEvent>) -> FrameResult {
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
            events,
            snapshot: self.snapshot(),
        }
    }
}

fn enemy_max_size_from_roll(roll: PicoFixed) -> PicoFixed {
    if roll <= PicoFixed::from_int(20) {
        PicoFixed::from_int(3)
    } else if roll <= PicoFixed::from_int(70) {
        PicoFixed::from_int(4)
    } else if roll <= PicoFixed::from_int(90) {
        PicoFixed::from_int(5)
    } else {
        PicoFixed::from_int(6)
    }
}

fn normal_personality_from_roll(roll: PicoFixed) -> i8 {
    let index = roll.floor().raw() >> 16;
    if index < 76 {
        0
    } else if index < 93 {
        1
    } else if index < 95 {
        2
    } else {
        4
    }
}

fn update_enemy_speed(enemy: EnemyState, player: PlayerState) -> PicoFixed {
    if enemy.personality != 1 {
        return enemy.speed;
    }
    let dx = enemy.x.sub(player.x);
    let dy = enemy.y.sub(player.y);
    let distance_squared = dx.mul_fixed(dx).add(dy.mul_fixed(dy));
    if distance_squared <= KAMIKAZE_RADIUS_SQUARED {
        enemy.speed.sub(ENEMY_SPEED_STEP)
    } else if enemy.speed < PicoFixed::ONE {
        enemy.speed.add(ENEMY_SPEED_STEP).min(PicoFixed::ONE)
    } else {
        PicoFixed::ONE
    }
}

fn consume_initial_pattern_random_state(rng: &mut PicoRng) {
    for limit in INITIAL_PATTERN_RND_PREFIX_LIMITS {
        let _ = rng.rnd(limit);
    }
    let branch = rng.rnd(PicoFixed::ONE);
    if branch > INITIAL_PATTERN_BRANCH_THRESHOLD {
        for limit in INITIAL_PATTERN_BRANCH_LIMITS {
            let _ = rng.rnd(limit);
        }
    }
}

fn enemy_is_inside(x: PicoFixed) -> bool {
    x >= PicoFixed::from_int(1) && x <= PicoFixed::from_int(128)
}

fn fixed_abs(value: PicoFixed) -> PicoFixed {
    if value < PicoFixed::ZERO {
        value.neg()
    } else {
        value
    }
}

fn pico_lerp(position: PicoFixed, target: PicoFixed, percentage: PicoFixed) -> PicoFixed {
    PicoFixed::ONE
        .sub(percentage)
        .mul_fixed(position)
        .add(percentage.mul_fixed(target))
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
    fn first_normal_spawn_preserves_source_init_rng_offset() {
        let mut game = NativeGame::new(NativeConfig::new(42));
        start_game(&mut game);
        for _ in 0..54 {
            assert!(game.advance_frame(0).is_ok());
        }
        assert_eq!(game.lifecycle().frame, 67);
        let first_normal = game.enemies().get(4).copied();
        assert_eq!(
            first_normal.map(|enemy| enemy.max_size),
            Some(PicoFixed::from_int(3))
        );
        assert_eq!(
            first_normal.map(|enemy| enemy.size),
            Some(PicoFixed::from_int(2))
        );
    }

    #[test]
    fn initial_pattern_rng_branch_is_seed_dependent() {
        let mut game = NativeGame::new(NativeConfig::new(7));
        start_game(&mut game);
        for _ in 0..57 {
            assert!(game.advance_frame(0).is_ok());
        }
        assert_eq!(game.lifecycle().frame, 70);
        assert_eq!(
            game.enemies().get(4).map(|enemy| enemy.max_size),
            Some(PicoFixed::from_int(5))
        );
    }

    #[test]
    fn normal_spawn_size_is_reused_by_later_friendly_spawn() {
        let mut game = NativeGame::new(NativeConfig::new(7));
        start_game(&mut game);
        for _ in 0..95 {
            assert!(game.advance_frame(0).is_ok());
        }
        assert_eq!(game.lifecycle().frame, 108);
        assert_eq!(game.enemies().len(), 9);
        assert!(
            game.enemies()
                .iter()
                .skip(5)
                .all(|enemy| { enemy.max_size == PicoFixed::from_int(5) })
        );
    }

    #[test]
    fn normal_enemy_size_roll_uses_source_thresholds() {
        assert_eq!(
            super::enemy_max_size_from_roll(PicoFixed::from_int(20)),
            PicoFixed::from_int(3)
        );
        assert_eq!(
            super::enemy_max_size_from_roll(PicoFixed::from_raw(20 * 65_536 + 1)),
            PicoFixed::from_int(4)
        );
        assert_eq!(
            super::enemy_max_size_from_roll(PicoFixed::from_int(70)),
            PicoFixed::from_int(4)
        );
        assert_eq!(
            super::enemy_max_size_from_roll(PicoFixed::from_raw(70 * 65_536 + 1)),
            PicoFixed::from_int(5)
        );
        assert_eq!(
            super::enemy_max_size_from_roll(PicoFixed::from_int(90)),
            PicoFixed::from_int(5)
        );
        assert_eq!(
            super::enemy_max_size_from_roll(PicoFixed::from_raw(90 * 65_536 + 1)),
            PicoFixed::from_int(6)
        );
    }

    #[test]
    fn normal_personality_roll_matches_low_score_source_distribution() {
        assert_eq!(super::normal_personality_from_roll(PicoFixed::ZERO), 0);
        assert_eq!(
            super::normal_personality_from_roll(PicoFixed::from_int(75)),
            0
        );
        assert_eq!(
            super::normal_personality_from_roll(PicoFixed::from_int(76)),
            1
        );
        assert_eq!(
            super::normal_personality_from_roll(PicoFixed::from_int(92)),
            1
        );
        assert_eq!(
            super::normal_personality_from_roll(PicoFixed::from_int(93)),
            2
        );
        assert_eq!(
            super::normal_personality_from_roll(PicoFixed::from_int(95)),
            4
        );
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
        assert!(game.enemies().is_empty());
        assert_eq!(game.survival_frames(), 0);
        assert_eq!(
            result.as_ref().map(|value| value.reward),
            Ok(PicoFixed::ZERO)
        );
    }

    #[test]
    fn powerup_collision_applies_size_reward_and_difficulty_without_terminal() {
        let mut game = NativeGame::new(NativeConfig::default());
        start_game(&mut game);
        game.enemies.push(EnemyState {
            personality: 4,
            ..EnemyState::normal(
                PicoFixed::from_int(61),
                PicoFixed::from_int(64),
                PicoFixed::from_int(3),
            )
        });
        let result = game.advance_frame(0);
        assert_eq!(result.as_ref().map(|value| value.done), Ok(false));
        let snapshot = game.snapshot();
        assert_eq!(snapshot.logical_state().player.size, PicoFixed::from_int(2));
        assert_eq!(game.score(), PicoFixed::ONE);
        assert!(game.enemies().is_empty());
    }

    #[test]
    fn v109_ordered_enemy_overlap_removes_marked_entries_and_applies_side_effects() {
        let mut game = NativeGame::new(NativeConfig::new(42));
        start_game(&mut game);
        game.enemies.push(EnemyState::normal(
            PicoFixed::from_int(10),
            PicoFixed::from_int(10),
            PicoFixed::from_int(3),
        ));
        game.enemies.push(EnemyState::normal(
            PicoFixed::from_int(11),
            PicoFixed::from_int(10),
            PicoFixed::from_int(3),
        ));

        assert!(game.advance_frame(0).is_ok());
        assert_eq!(game.enemies().len(), 2);
        assert!(game.enemies().iter().all(|enemy| enemy.is_dying));

        assert!(game.advance_frame(0).is_ok());
        assert!(game.enemies().is_empty());
        let snapshot = game.snapshot();
        let state = snapshot.logical_state();
        assert_eq!(state.score, PicoFixed::ONE);
        assert!(state.speed > PicoFixed::ONE);
        assert!(state.enemy_est < PicoFixed::ONE);
    }

    #[test]
    fn draw_side_effect_rng_is_consumed_for_powerup_enemy_trails() {
        let mut game = NativeGame::new(NativeConfig::new(42));
        start_game(&mut game);
        game.enemies.push(EnemyState {
            personality: 4,
            ..EnemyState::normal(
                PicoFixed::from_int(10),
                PicoFixed::from_int(10),
                PicoFixed::from_int(3),
            )
        });
        let before = game.rng_checkpoint();
        assert!(game.advance_frame(0).is_ok());
        let after = game.rng_checkpoint();

        let mut expected = crate::PicoRng::new(42);
        assert!(expected.restore(before).is_ok());
        let _ = expected.rnd(PicoFixed::from_int(32));
        let _ = expected.rnd(PicoFixed::from_int(32));
        let _ = expected.rnd(PicoFixed::ONE);
        let _ = expected.rnd(PicoFixed::from_int(15));
        assert_eq!(after, expected.checkpoint());
    }
}
