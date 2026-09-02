use crate::{
    Action, Button, CoreError, EnemyState, FullState, IndexedFramebuffer, InputState,
    LifecycleState, Mode, NativeConfig, ParticleState, PatternState, PicoFixed, PicoRng,
    PlayerState, SettingsState, Snapshot, SpawnPoint, pico_mid,
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
const ENEMY_HALF_STEP: PicoFixed = PicoFixed::from_raw(32_768);
const DIFFICULTY_SPEED_TARGET: PicoFixed = PicoFixed::from_int(3);
const DIFFICULTY_EST_TARGET: PicoFixed = PicoFixed::from_raw(14_417);
const SCORE_PER_SHATTER: PicoFixed = PicoFixed::from_raw(32_768);
const ENEMY_SPEED_STEP: PicoFixed = PicoFixed::from_raw(655);
const KAMIKAZE_RADIUS_SQUARED: PicoFixed = PicoFixed::from_int(625);
const INITIAL_PATTERN_DELAY_FRAMES: u32 = 420;
const ACTIVE_PATTERN_DELAY_FRAMES: u32 = 600;

#[derive(Clone, Copy)]
struct DifficultyCurve {
    speed_increment: PicoFixed,
    enemy_increment: PicoFixed,
    static_increment: PicoFixed,
    static_target: PicoFixed,
    moving_increment: PicoFixed,
    moving_target: PicoFixed,
}

const DIFFICULTY_CURVES: [DifficultyCurve; 3] = [
    DifficultyCurve {
        speed_increment: PicoFixed::from_raw(131),
        enemy_increment: PicoFixed::from_raw(327),
        static_increment: PicoFixed::from_raw(32),
        static_target: PicoFixed::from_raw(29_491),
        moving_increment: PicoFixed::from_raw(85),
        moving_target: PicoFixed::from_raw(29_491),
    },
    DifficultyCurve {
        speed_increment: PicoFixed::from_raw(452),
        enemy_increment: PicoFixed::from_raw(655),
        static_increment: PicoFixed::from_raw(1_245),
        static_target: PicoFixed::from_raw(29_491),
        moving_increment: PicoFixed::from_raw(1_245),
        moving_target: PicoFixed::from_raw(29_491),
    },
    DifficultyCurve {
        speed_increment: PicoFixed::from_raw(1_835),
        enemy_increment: PicoFixed::from_raw(1_310),
        static_increment: PicoFixed::from_raw(1_572),
        static_target: PicoFixed::from_raw(26_214),
        moving_increment: PicoFixed::from_raw(1_572),
        moving_target: PicoFixed::from_raw(26_214),
    },
];

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
    particles: Vec<ParticleState>,
    patterns: Vec<PatternState>,
    active_pattern: Option<usize>,
    spawns: Vec<SpawnPoint>,
    screen: IndexedFramebuffer,
    enemy_timer: PicoFixed,
    enemy_est: PicoFixed,
    enemy_stats: [PicoFixed; 5],
    friendly_timer: u32,
    friendly_enabled: bool,
    enemy_max_size: PicoFixed,
    speed: PicoFixed,
    freeze_rate: PicoFixed,
    freeze_active: bool,
    freeze_timer: u32,
    size_timer: PicoFixed,
    patterns_enabled: bool,
    powerups_enabled: bool,
    pattern_timer: u32,
    pattern_delay_frames: u32,
    pattern_active: bool,
    new_highscore: bool,
    can_click: bool,
    has_played: bool,
    should_collide: bool,
    enemy_should_collide: bool,
    bounce_cap_static: PicoFixed,
    bounce_cap_moving: PicoFixed,
    bounce_cap: PicoFixed,
    score: PicoFixed,
    survival_frames: u32,
    shake: PicoFixed,
    camera_x: PicoFixed,
    camera_y: PicoFixed,
    transition_render_y: i16,
    transition_from: Mode,
    settings: SettingsState,
    highscores: [PicoFixed; 12],
}

impl NativeGame {
    pub fn new(config: NativeConfig) -> Self {
        let mut rng = PicoRng::new(config.seed);
        let patterns = crate::patterns::init_patterns(&mut rng);
        Self {
            rng,
            config,
            lifecycle: LifecycleState::new(),
            input: InputState::new(),
            player: PlayerState::new(),
            enemies: Vec::new(),
            particles: Vec::new(),
            patterns,
            active_pattern: None,
            spawns: initial_spawns(),
            screen: IndexedFramebuffer::filled(12),
            enemy_timer: PicoFixed::ZERO,
            enemy_est: initial_enemy_est(config.difficulty),
            enemy_stats: initial_enemy_stats(config.powerups_enabled),
            friendly_timer: 0,
            friendly_enabled: true,
            enemy_max_size: ENEMY_INITIAL_MAX_SIZE,
            speed: initial_speed(config.difficulty),
            freeze_rate: PicoFixed::ONE,
            freeze_active: false,
            freeze_timer: 0,
            size_timer: PicoFixed::ZERO,
            patterns_enabled: config.patterns_enabled,
            powerups_enabled: config.powerups_enabled,
            pattern_timer: 0,
            pattern_delay_frames: INITIAL_PATTERN_DELAY_FRAMES,
            pattern_active: false,
            new_highscore: false,
            can_click: true,
            has_played: false,
            should_collide: true,
            enemy_should_collide: true,
            bounce_cap_static: initial_bounce_static(config.difficulty),
            bounce_cap_moving: initial_bounce_moving(config.difficulty),
            bounce_cap: PicoFixed::from_f32(1.45),
            score: PicoFixed::ZERO,
            survival_frames: 0,
            shake: PicoFixed::ZERO,
            camera_x: PicoFixed::ZERO,
            camera_y: PicoFixed::ZERO,
            transition_render_y: -128,
            transition_from: Mode::Menu,
            settings: SettingsState::new(config),
            highscores: config.highscores,
        }
    }

    pub fn reset(&mut self) -> Snapshot {
        self.lifecycle = LifecycleState::new();
        self.input = InputState::new();
        self.rng.seed(self.config.seed);
        self.patterns = crate::patterns::init_patterns(&mut self.rng);
        self.player = PlayerState::new();
        self.enemies.clear();
        self.particles.clear();
        self.active_pattern = None;
        self.spawns = initial_spawns();
        self.screen = IndexedFramebuffer::filled(12);
        self.enemy_timer = PicoFixed::ZERO;
        self.enemy_est = initial_enemy_est(self.config.difficulty);
        self.enemy_stats = initial_enemy_stats(self.config.powerups_enabled);
        self.friendly_timer = 0;
        self.friendly_enabled = true;
        self.enemy_max_size = ENEMY_INITIAL_MAX_SIZE;
        self.speed = initial_speed(self.config.difficulty);
        self.freeze_rate = PicoFixed::ONE;
        self.freeze_active = false;
        self.freeze_timer = 0;
        self.size_timer = PicoFixed::ZERO;
        self.patterns_enabled = self.config.patterns_enabled;
        self.powerups_enabled = self.config.powerups_enabled;
        self.pattern_timer = 0;
        self.pattern_delay_frames = INITIAL_PATTERN_DELAY_FRAMES;
        self.pattern_active = false;
        self.new_highscore = false;
        self.can_click = true;
        self.should_collide = true;
        self.enemy_should_collide = true;
        self.bounce_cap_static = initial_bounce_static(self.config.difficulty);
        self.bounce_cap_moving = initial_bounce_moving(self.config.difficulty);
        self.bounce_cap = PicoFixed::from_f32(1.45);
        self.score = PicoFixed::ZERO;
        self.survival_frames = 0;
        self.shake = PicoFixed::ZERO;
        self.camera_x = PicoFixed::ZERO;
        self.camera_y = PicoFixed::ZERO;
        self.transition_render_y = -128;
        self.transition_from = Mode::Menu;
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
        self.refresh_enemy_stats();
        let mode_before = self.lifecycle.mode;
        let mut events = Vec::new();
        let start_pressed = mode_before == Mode::Menu && self.input.btnp(Button::X);
        if start_pressed {
            self.has_played = true;
            self.transition_from = Mode::Menu;
        }
        if mode_before == Mode::Menu && self.input.btnp(Button::O) {
            self.begin_transition(Mode::TransitionToSettings);
        }
        if matches!(mode_before, Mode::Game | Mode::Terminal) && self.input.btnp(Button::O) {
            self.begin_transition(Mode::TransitionToSettings);
        }
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
            Mode::TransitionToSettings | Mode::TransitionToMenu => {
                self.transition_render_y -= 20;
                false
            }
            Mode::Game => true,
            Mode::Settings | Mode::Terminal => false,
        };
        match mode_before {
            Mode::TransitionToGame | Mode::TransitionToMenu if self.transition_render_y >= 128 => {
                self.lifecycle.mode = if mode_before == Mode::TransitionToGame {
                    Mode::Game
                } else {
                    Mode::Menu
                };
                self.lifecycle.transition_y = 0;
                self.lifecycle.game_ready = mode_before == Mode::TransitionToGame;
            }
            Mode::TransitionToSettings if self.transition_render_y <= -128 => {
                self.lifecycle.mode = Mode::Settings;
                self.lifecycle.transition_y = 0;
            }
            _ => {}
        }
        if game_update_from_draw {
            self.update_game_frame(&mut events);
        }
        let game_update_from_transition_draw = self.lifecycle.mode == Mode::TransitionToSettings
            && self.transition_render_y > 0
            && matches!(self.transition_from, Mode::Game | Mode::Terminal);
        if game_update_from_transition_draw {
            self.update_game_frame(&mut events);
        }
        if mode_before == Mode::Settings {
            self.update_settings();
        }
        if matches!(mode_before, Mode::Settings) && self.input.btnp(Button::X) {
            let target = if self.has_played {
                Mode::TransitionToGame
            } else {
                Mode::TransitionToMenu
            };
            self.begin_transition(target);
        }
        if self.active_pattern.is_some() && mode_before == Mode::Game {
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
        self.update_camera();
        self.render_current_frame();
        let draw_snapshot = self.snapshot();
        self.apply_draw_side_effects(mode_before);
        self.input.finalize_frame(post_frame_mask);
        let result = self.result_with_snapshot(
            reward,
            events,
            Snapshot::with_framebuffer_from_game(
                self,
                draw_snapshot.framebuffer().clone(),
                draw_snapshot.render_state().clone(),
            ),
        );
        // Pemsa can schedule a second visible draw on the transition boundary
        // after the canonical capture callback has observed the first draw.
        // Preserve that source-side particle mutation for the next frame while
        // keeping this frame's snapshot at the capture boundary.
        if mode_before == Mode::TransitionToGame
            && self.lifecycle.mode == Mode::Game
            && self.transition_from == Mode::Menu
        {
            self.add_player_trail();
        }
        Ok(result)
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
            particles: self.particles.clone(),
            patterns: self.patterns.clone(),
            active_pattern: self.active_pattern,
            spawns: self.spawns.clone(),
            physical_screen: self.screen.clone(),
            enemy_timer: self.enemy_timer,
            enemy_est: self.enemy_est,
            enemy_stats: self.enemy_stats,
            friendly_timer: self.friendly_timer,
            friendly_enabled: self.friendly_enabled,
            enemy_max_size: self.enemy_max_size,
            speed: self.speed,
            freeze_rate: self.freeze_rate,
            freeze_active: self.freeze_active,
            freeze_timer: self.freeze_timer,
            size_timer: self.size_timer,
            patterns_enabled: self.patterns_enabled,
            powerups_enabled: self.powerups_enabled,
            pattern_timer: self.pattern_timer,
            pattern_delay_frames: self.pattern_delay_frames,
            pattern_active: self.pattern_active,
            new_highscore: self.new_highscore,
            can_click: self.can_click,
            has_played: self.has_played,
            should_collide: self.should_collide,
            enemy_should_collide: self.enemy_should_collide,
            bounce_cap_static: self.bounce_cap_static,
            bounce_cap_moving: self.bounce_cap_moving,
            bounce_cap: self.bounce_cap,
            score: self.score,
            survival_frames: self.survival_frames,
            shake: self.shake,
            camera_x: self.camera_x,
            camera_y: self.camera_y,
            transition_render_y: self.transition_render_y,
            transition_from: self.transition_from,
            settings: self.settings,
            highscores: self.highscores,
        }
    }

    pub fn snapshot(&self) -> Snapshot {
        Snapshot::from_game(self)
    }

    fn update_game_frame(&mut self, events: &mut Vec<FrameEvent>) {
        if self.friendly_enabled {
            self.update_fyou(events);
        }
        self.collision_check(events);
        self.update_player();
        self.update_particles();
        self.update_freeze();
        self.update_size();
        if !self.lifecycle.dead {
            self.spawn_enemies(events);
            self.update_enemies();
            self.update_pattern_schedule();
        }
    }

    fn begin_transition(&mut self, mode: Mode) {
        let from = self.lifecycle.mode;
        self.transition_from = match from {
            Mode::Menu | Mode::Game | Mode::Terminal | Mode::Settings => from,
            _ => self.transition_from,
        };
        match mode {
            Mode::TransitionToGame => {
                self.lifecycle.mode = Mode::TransitionToGame;
                self.lifecycle.transition_y = -128;
                self.transition_render_y = -108;
                self.lifecycle.game_ready = false;
            }
            Mode::TransitionToSettings => {
                self.lifecycle.mode = Mode::TransitionToSettings;
                self.lifecycle.transition_y = 128;
                self.transition_render_y = 108;
            }
            Mode::TransitionToMenu => {
                self.lifecycle.mode = Mode::TransitionToMenu;
                self.lifecycle.transition_y = -128;
                self.transition_render_y = -108;
                self.lifecycle.game_ready = false;
            }
            Mode::Menu | Mode::Game | Mode::Terminal | Mode::Settings => {}
        }
    }

    fn update_settings(&mut self) {
        let gameplay_editable = self.lifecycle.dead || !self.has_played;
        let cursor = self.settings.cursor;
        if self.input.btnp(Button::Left) || self.input.btnp(Button::Right) {
            let increase = self.input.btnp(Button::Right);
            if cursor == 1 {
                self.settings.theme_index = cycle_index(self.settings.theme_index, 13, increase);
                let (background, shadow) = theme_values(self.settings.theme_index);
                self.config.theme_background = background;
                self.config.theme_shadow = shadow;
                self.settings.theme_background = background;
                self.settings.theme_shadow = shadow;
            } else if gameplay_editable {
                match cursor {
                    2 => {
                        self.settings.difficulty =
                            cycle_index(self.settings.difficulty, 3, increase);
                        self.config.difficulty = self.settings.difficulty;
                        self.apply_initial_difficulty();
                    }
                    3 => {
                        self.settings.patterns_enabled = !self.settings.patterns_enabled;
                        self.config.patterns_enabled = self.settings.patterns_enabled;
                        self.patterns_enabled = self.settings.patterns_enabled;
                    }
                    4 => {
                        self.settings.powerups_enabled = !self.settings.powerups_enabled;
                        self.config.powerups_enabled = self.settings.powerups_enabled;
                        self.powerups_enabled = self.settings.powerups_enabled;
                    }
                    _ => {}
                }
            }
            self.settings.message_sprite = if increase { 193 } else { 192 };
            self.settings.message_x = if increase { 80 } else { 96 };
            self.settings.message_y = settings_row(cursor) - 1;
            self.settings.message_timer = 6;
        }
        let max_cursor = if gameplay_editable { 4 } else { 1 };
        if self.input.btnp(Button::Down) {
            self.settings.cursor = self.settings.cursor.saturating_add(1).min(max_cursor);
        }
        if self.input.btnp(Button::Up) {
            self.settings.cursor = self.settings.cursor.saturating_sub(1).max(1);
        }
        if self.settings.message_timer > 0 {
            self.settings.message_timer -= 1;
        }
    }

    fn apply_initial_difficulty(&mut self) {
        self.enemy_est = initial_enemy_est(self.settings.difficulty);
        self.speed = initial_speed(self.settings.difficulty);
        self.bounce_cap_static = initial_bounce_static(self.settings.difficulty);
        self.bounce_cap_moving = initial_bounce_moving(self.settings.difficulty);
    }

    fn refresh_enemy_stats(&mut self) {
        let powerup_value = if self.powerups_enabled {
            PicoFixed::from_int(2)
        } else {
            PicoFixed::ZERO
        };
        let mut stats = [
            PicoFixed::from_f32(76.5),
            PicoFixed::from_f32(17.5),
            powerup_value,
            powerup_value,
            powerup_value,
        ];
        for enemy in &self.enemies {
            if enemy.personality >= 2 {
                let personality_index = usize::try_from(enemy.personality).unwrap_or(usize::MAX);
                if let Some(value) = stats.get_mut(personality_index) {
                    *value = value.sub(PicoFixed::from_int(2));
                }
                for (index, value) in stats.iter_mut().enumerate().skip(2) {
                    if *value != PicoFixed::ZERO && enemy.personality != index as i8 {
                        *value = value.add(PicoFixed::ONE);
                    }
                    *value = pico_mid(PicoFixed::ZERO, PicoFixed::from_int(3), *value);
                }
            }
        }
        if self.score <= PicoFixed::from_int(10) {
            stats[3] = PicoFixed::ZERO;
        }
        self.enemy_stats = stats;
    }

    fn update_freeze(&mut self) {
        if !self.freeze_active {
            return;
        }
        self.freeze_timer += 1;
        if self.freeze_timer <= 420 {
            self.freeze_rate = PicoFixed::from_f32(0.4);
        } else {
            self.freeze_rate = self.freeze_rate.add(PicoFixed::from_f32(0.6 / 60.0));
        }
        if self.freeze_timer > 480 {
            self.freeze_active = false;
            self.freeze_rate = PicoFixed::ONE;
        }
    }

    fn update_size(&mut self) {
        if self.player.size == PicoFixed::from_int(4) {
            return;
        }
        self.size_timer = self.size_timer.add(self.freeze_rate);
        if self.size_timer <= PicoFixed::from_int(600) {
            return;
        }
        if self.player.size < PicoFixed::from_int(4) {
            self.player.size = self
                .player
                .size
                .add(PicoFixed::from_f32(0.5).mul_fixed(self.freeze_rate));
        } else {
            self.player.size = PicoFixed::from_int(4);
        }
    }

    fn render_current_frame(&mut self) {
        let state = self.full_state();
        let mut render = crate::RenderState::new(state.transition_render_y);
        render.camera_x = state.camera_x;
        render.camera_y = state.camera_y;
        render.screen_palette[12] = state.settings.theme_background;
        render.screen_palette[1] = state.settings.theme_shadow;
        crate::snapshot::render_full_state_into(&state, &render, &mut self.screen);
    }

    fn update_particles(&mut self) {
        let mut index = 0;
        while index < self.particles.len() {
            let Some(mut particle) = self.particles.get(index).copied() else {
                break;
            };
            particle.age += 1;
            if PicoFixed::from_int(particle.age as i32) > particle.max_age {
                self.particles.remove(index);
                continue;
            }
            particle.color = particle_color(particle);
            particle.x = particle.x.add(particle.dx);
            particle.y = particle.y.add(particle.dy);
            if particle.kind != 2 {
                particle.radius = particle.radius.mul_fixed(PicoFixed::from_f32(0.9));
            }
            if particle.radius < PicoFixed::ZERO && particle.kind == 0 {
                self.particles.remove(index);
                continue;
            }
            if let Some(slot) = self.particles.get_mut(index) {
                *slot = particle;
            }
            index += 1;
        }
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
        for spawn in self.spawns.clone() {
            self.enemies
                .push(EnemyState::normal(spawn.x, spawn.y, self.enemy_max_size));
            events.push(FrameEvent::EnemySpawn);
        }
    }

    fn collision_check(&mut self, events: &mut Vec<FrameEvent>) {
        if self.lifecycle.dead || !self.should_collide {
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
                self.collide_enemy(index, enemy, events);
            } else {
                index += 1;
            }
        }
        if let Some(pattern_index) = self.active_pattern {
            let Some(pattern) = self.patterns.get(pattern_index) else {
                return;
            };
            if pattern.pattern_type == 1 {
                for rect in &pattern.rects {
                    if self
                        .player
                        .x
                        .sub(PicoFixed::from_int(2))
                        .add(self.player.size)
                        > rect.x
                        && self
                            .player
                            .y
                            .sub(PicoFixed::from_int(2))
                            .add(self.player.size)
                            > rect.y
                        && self
                            .player
                            .x
                            .add(PicoFixed::from_int(2))
                            .sub(self.player.size)
                            < rect.x.add(rect.width).sub(PicoFixed::ONE)
                        && self
                            .player
                            .y
                            .add(PicoFixed::from_int(2))
                            .sub(self.player.size)
                            < rect.y.add(rect.height).sub(PicoFixed::ONE)
                    {
                        self.die();
                        events.push(FrameEvent::Death);
                        break;
                    }
                }
            }
        }
    }

    fn collide_enemy(&mut self, index: usize, enemy: EnemyState, events: &mut Vec<FrameEvent>) {
        self.shatter(enemy.x, enemy.y);
        self.shake = self.shake.add(PicoFixed::from_f32(0.07));
        match enemy.personality {
            personality if personality < 2 => {
                self.remove_enemy_at(index);
                self.die();
                events.push(FrameEvent::Death);
            }
            2 => {
                self.explode_powerup_enemy(index);
            }
            3 => {
                self.remove_enemy_at(index);
                self.freeze_active = true;
                self.freeze_timer = 0;
                self.score = self.score.add(PicoFixed::ONE);
                self.apply_difficulty(false);
            }
            4 => {
                self.remove_enemy_at(index);
                self.player.size = PicoFixed::from_int(2);
                self.size_timer = PicoFixed::ZERO;
                self.score = self.score.add(PicoFixed::ONE);
                self.apply_difficulty(false);
            }
            _ => {
                self.remove_enemy_at(index);
            }
        }
    }

    fn explode_powerup_enemy(&mut self, index: usize) {
        let _ = self.remove_enemy_at(index);
        let mut other_index = 0;
        while other_index < self.enemies.len() {
            let Some(other) = self.enemies.get(other_index).copied() else {
                break;
            };
            if other.personality == 1 {
                self.add_kamikaze(other);
            } else if other.personality != -1 {
                self.shatter(other.x, other.y);
            }
            if other.personality != -1 {
                self.remove_enemy_at(other_index);
            } else {
                other_index += 1;
            }
            self.score = self.score.add(PicoFixed::ONE);
            self.apply_difficulty(false);
        }
    }

    fn remove_enemy_at(&mut self, index: usize) -> Option<EnemyState> {
        if index < self.enemies.len() {
            Some(self.enemies.remove(index))
        } else {
            None
        }
    }

    fn die(&mut self) {
        self.lifecycle.mark_dead();
        self.shake = self.shake.add(PicoFixed::from_f32(0.07));
        let slot = self.current_highscore_slot();
        let previous = self
            .highscores
            .get(slot)
            .copied()
            .unwrap_or(PicoFixed::ZERO);
        if self.score > previous {
            if let Some(highscore) = self.highscores.get_mut(slot) {
                *highscore = self.score.ceil();
            }
            self.new_highscore = true;
        }
    }

    fn current_highscore_slot(&self) -> usize {
        let difficulty = self.settings.difficulty;
        let patterns = self.settings.patterns_enabled;
        let powerups = self.settings.powerups_enabled;
        (0..12)
            .find(|index| {
                let category = index / 4 + 1;
                let slot = index % 4;
                category == usize::from(difficulty)
                    && patterns == (slot == 0 || slot == 1)
                    && powerups == (slot == 0 || slot == 3)
            })
            .unwrap_or(0)
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
        let spawn_multiplier = self
            .active_pattern
            .and_then(|index| self.patterns.get(index))
            .map_or(PicoFixed::ONE, |pattern| {
                if matches!(pattern.special, 2 | 3) {
                    PicoFixed::from_f32(1.75)
                } else {
                    PicoFixed::ONE
                }
            });
        let threshold = self
            .enemy_est
            .mul_fixed(SPAWN_INTERVAL)
            .mul_fixed(spawn_multiplier)
            .mul_fixed(PicoFixed::from_int(2).sub(self.freeze_rate));
        if self.enemy_timer <= threshold || self.spawns.is_empty() {
            return;
        }
        self.enemy_timer = PicoFixed::ZERO;
        let size_roll = self.rng.rnd(PicoFixed::from_int(100));
        let (x, y) = self.nearest_corner();
        let max_size = enemy_max_size_from_roll(size_roll);
        self.enemy_max_size = max_size;
        let personality = self.random_personality();
        self.enemies.push(EnemyState {
            personality,
            ..EnemyState::normal(x, y, max_size)
        });
        events.push(FrameEvent::EnemySpawn);
    }

    fn random_personality(&mut self) -> i8 {
        let total = self
            .enemy_stats
            .iter()
            .map(|value| value.floor().raw().max(0) >> 16)
            .sum::<i32>();
        if total <= 0 {
            return 0;
        }
        let roll = (self.rng.rnd(PicoFixed::from_int(total)).floor().raw() >> 16).max(0);
        let mut cursor = 0_i32;
        for (index, value) in self.enemy_stats.iter().enumerate() {
            cursor += value.floor().raw().max(0) >> 16;
            if roll < cursor {
                return index as i8;
            }
        }
        0
    }

    fn nearest_corner(&self) -> (PicoFixed, PicoFixed) {
        let Some(first) = self.spawns.first().copied() else {
            return (SPAWN_EDGE, SPAWN_EDGE);
        };
        let mut best = (first.x, first.y);
        let mut best_distance = f64::INFINITY;
        for spawn in &self.spawns {
            let dx = spawn.x.to_double() - self.player.x.to_double();
            let dy = spawn.y.to_double() - self.player.y.to_double();
            let distance = dx * dx + dy * dy;
            if distance < best_distance {
                best = (spawn.x, spawn.y);
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
            let mut vx = original.vx.mul_fixed(ENEMY_FRICTION);
            let mut vy = original.vy.mul_fixed(ENEMY_FRICTION);
            let mut size = original.size;
            let mut next_x = original.x;
            let mut next_y = original.y;
            let mut isizing = original.isizing;
            let mut life = original.life;

            if size < original.max_size && original.personality != -1 {
                size = size.add(self.freeze_rate);
            }
            if size < original.max_size && original.personality == -1 && isizing {
                size = size.add(self.freeze_rate);
                next_x = next_x.sub(ENEMY_HALF_STEP.mul_fixed(self.freeze_rate));
                next_y = next_y.sub(ENEMY_HALF_STEP.mul_fixed(self.freeze_rate));
            }
            if size >= original.max_size && original.personality == -1 {
                isizing = false;
            }
            if original.personality == -1 && !isizing {
                size = size.sub(self.freeze_rate);
                next_x = next_x.add(ENEMY_HALF_STEP.mul_fixed(self.freeze_rate));
                next_y = next_y.add(ENEMY_HALF_STEP.mul_fixed(self.freeze_rate));
            }
            if original.personality >= 2 {
                size = PicoFixed::from_int(4);
            }
            if let Some(current_life) = life {
                let remaining = current_life.sub(self.freeze_rate);
                life = Some(remaining);
                if remaining <= PicoFixed::ZERO {
                    self.remove_enemy_at(index);
                    continue;
                }
            }

            let inside = original.inside || enemy_is_inside(x);
            let mut current = original;
            current.x = next_x;
            current.y = next_y;
            current.size = size;
            current.inside = inside;
            current.isizing = isizing;
            current.life = life;

            if original.is_dying {
                self.enemies.remove(index);
                self.resolve_enemy_death(current);
                continue;
            }
            if let Some(slot) = self.enemies.get_mut(index) {
                *slot = current;
            } else {
                break;
            }

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

            self.apply_pattern_enemy_collision(index, &mut vx, &mut vy, x, y, size);
            self.apply_pattern_crush(index, x, y, size);

            let current_offset = if original.personality >= 2 {
                PicoFixed::from_int(4)
            } else {
                PicoFixed::ZERO
            };
            for other_index in 0..self.enemies.len() {
                if other_index == index {
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
                if self.enemy_should_collide
                    && inside
                    && other.inside
                    && x.add(size) > other.x.sub(other_offset)
                    && y.add(size) > other.y.sub(other_offset)
                    && x.sub(current_offset) < other.x.add(other.size)
                    && y.sub(current_offset) < other.y.add(other.size)
                {
                    if let Some(current_enemy) = self.enemies.get_mut(index) {
                        current_enemy.is_dying = true;
                    }
                    if let Some(other_enemy) = self.enemies.get_mut(other_index) {
                        other_enemy.is_dying = true;
                    }
                }
            }

            if inside && original.personality != -1 {
                if y >= PicoFixed::from_int(129).sub(size) {
                    vy = fixed_abs(vy).neg();
                } else if y <= PicoFixed::from_int(-2) {
                    vy = fixed_abs(vy);
                }
                if x >= PicoFixed::from_int(129).sub(size) {
                    vx = fixed_abs(vx).neg();
                } else if x <= PicoFixed::from_int(-2) {
                    vx = fixed_abs(vx);
                }
            }

            let Some(mut updated) = self.enemies.get(index).copied() else {
                break;
            };
            updated.speed = update_enemy_speed(updated, player);
            updated.x = next_x.add(
                vx.mul_fixed(self.speed)
                    .mul_fixed(self.freeze_rate)
                    .mul_fixed(updated.speed),
            );
            updated.y = next_y.add(
                vy.mul_fixed(self.speed)
                    .mul_fixed(self.freeze_rate)
                    .mul_fixed(updated.speed),
            );
            updated.vx = vx;
            updated.vy = vy;
            if let Some(slot) = self.enemies.get_mut(index) {
                *slot = updated;
            }
            index += 1;
        }
    }

    fn apply_pattern_enemy_collision(
        &self,
        _index: usize,
        vx: &mut PicoFixed,
        vy: &mut PicoFixed,
        x: PicoFixed,
        y: PicoFixed,
        size: PicoFixed,
    ) {
        let Some(pattern_index) = self.active_pattern else {
            return;
        };
        let Some(pattern) = self.patterns.get(pattern_index) else {
            return;
        };
        for rect in &pattern.rects {
            if rect.sh != PicoFixed::from_int(2) {
                continue;
            }
            let powerup = self
                .enemies
                .get(_index)
                .is_some_and(|enemy| enemy.personality > 1);
            let powerup_offset = if powerup {
                PicoFixed::from_int(3)
            } else {
                PicoFixed::ZERO
            };
            if x.add(*vx).sub(powerup_offset) < rect.x.add(rect.width)
                && x.add(size) > rect.x.add(rect.dx)
                && y.add(size) > rect.y.add(rect.dy)
                && y.add(*vy).sub(powerup_offset) < rect.y.add(rect.height)
            {
                let minimum = if pattern.pattern_type == 0 {
                    self.bounce_cap_static
                } else {
                    self.bounce_cap_moving
                };
                if x.add(*vx) < rect.x {
                    *vx = fixed_abs(*vx).max(minimum).neg();
                } else if x.add(*vx).add(size) > rect.x.add(rect.width) {
                    *vx = fixed_abs(*vx).max(minimum);
                }
                if y.add(*vy) < rect.y {
                    *vy = fixed_abs(*vy).max(minimum).neg();
                } else if y.add(*vy).add(size) > rect.y.add(rect.height) {
                    *vy = fixed_abs(*vy).max(minimum);
                }
                let mut cap = if pattern.pattern_type == 0 {
                    self.bounce_cap_static
                } else {
                    self.bounce_cap_moving
                };
                if pattern.bounce_cap {
                    cap = cap.sub(PicoFixed::from_f32(0.15));
                }
                *vx = with_sign(fixed_abs(*vx).min(cap), *vx);
                *vy = with_sign(fixed_abs(*vy).min(cap), *vy);
            }
        }
    }

    fn apply_pattern_crush(&mut self, index: usize, x: PicoFixed, y: PicoFixed, size: PicoFixed) {
        let Some(pattern_index) = self.active_pattern else {
            return;
        };
        let Some(pattern) = self.patterns.get(pattern_index) else {
            return;
        };
        let Some(enemy) = self.enemies.get(index) else {
            return;
        };
        if enemy.personality == -1 || !enemy.inside {
            return;
        }
        let mut dying = false;
        for rect in &pattern.rects {
            if rect.sh != PicoFixed::from_int(2) {
                continue;
            }
            let mut should_check = false;
            let mut add_x = PicoFixed::ZERO;
            let mut add_y = PicoFixed::ZERO;
            if x.add(size) > PicoFixed::from_int(127) {
                should_check = true;
                add_x = PicoFixed::from_int(-4);
            } else if x.sub(size) < PicoFixed::ZERO {
                should_check = true;
                add_x = PicoFixed::from_int(4);
            }
            if y.add(size) > PicoFixed::from_int(127) {
                should_check = true;
                add_y = PicoFixed::from_int(-4);
            } else if y.sub(size) < PicoFixed::ZERO {
                should_check = true;
                add_y = PicoFixed::from_int(4);
            }
            if should_check
                && x.add(add_x) > rect.x
                && y.add(add_y) > rect.y
                && x.add(add_x) < rect.x.add(rect.width)
                && y.add(add_y) < rect.y.add(rect.height)
            {
                dying = true;
                break;
            }
        }
        if dying && let Some(enemy) = self.enemies.get_mut(index) {
            enemy.is_dying = true;
        }
    }

    fn resolve_enemy_death(&mut self, enemy: EnemyState) {
        if enemy.personality == 1 {
            self.add_kamikaze(enemy);
        } else {
            self.shatter(enemy.x, enemy.y);
        }
        self.score = self.score.add(SCORE_PER_SHATTER);
        self.shake = self.shake.add(PicoFixed::from_f32(0.07));
        self.apply_difficulty(true);
    }

    fn shatter(&mut self, x: PicoFixed, y: PicoFixed) {
        for _ in 0..11 {
            let angle = self.rng.rnd(PicoFixed::ONE);
            self.particles.push(ParticleState {
                x,
                y,
                dx: pico_sine(angle),
                dy: pico_cosine(angle),
                radius: PicoFixed::ZERO,
                kind: 1,
                max_age: PicoFixed::from_int(60),
                age: 0,
                color: 0,
                colors: [7, 0, 0],
                color_count: 1,
            });
        }
    }

    fn add_kamikaze(&mut self, enemy: EnemyState) {
        self.enemies.push(EnemyState {
            x: enemy.x.add(enemy.size.mul_fixed(PicoFixed::from_f32(0.5))),
            y: enemy.y.add(enemy.size.mul_fixed(PicoFixed::from_f32(0.5))),
            vx: PicoFixed::ZERO,
            vy: PicoFixed::ZERO,
            size: PicoFixed::ZERO,
            max_size: PicoFixed::from_int(30),
            personality: -1,
            speed: PicoFixed::ONE,
            inside: false,
            is_dying: false,
            isizing: true,
            life: Some(PicoFixed::from_int(60)),
        });
    }

    fn apply_difficulty(&mut self, half: bool) {
        let index = difficulty_index(self.settings.difficulty);
        let Some(curve) = DIFFICULTY_CURVES.get(index).copied() else {
            return;
        };
        let speed_step = difficulty_increment(curve.speed_increment, half);
        let enemy_step = difficulty_increment(curve.enemy_increment, half);
        self.speed = pico_lerp(self.speed, DIFFICULTY_SPEED_TARGET, speed_step);
        self.enemy_est = pico_lerp(self.enemy_est, DIFFICULTY_EST_TARGET, enemy_step);
        let static_step = difficulty_increment(curve.static_increment, half);
        let moving_step = difficulty_increment(curve.moving_increment, half);
        self.bounce_cap_static =
            pico_lerp(self.bounce_cap_static, curve.static_target, static_step);
        self.bounce_cap_moving =
            pico_lerp(self.bounce_cap_moving, curve.moving_target, moving_step);
    }

    fn update_pattern_schedule(&mut self) {
        if self.active_pattern.is_some() {
            self.update_active_pattern();
            return;
        }
        self.pattern_timer += 1;
        if self.pattern_timer < self.pattern_delay_frames {
            return;
        }
        self.pattern_timer = 0;
        self.pattern_delay_frames = ACTIVE_PATTERN_DELAY_FRAMES;
        let mut weighted = Vec::new();
        for (index, pattern) in self.patterns.iter().enumerate() {
            if self.score >= pattern.mins && self.score <= pattern.maxs {
                let repetitions = (pattern.probability.floor().raw() >> 16).max(0);
                for _ in 0..repetitions {
                    weighted.push(index);
                }
            }
        }
        let Some(&selected) = weighted.get(
            self.rng
                .rnd(PicoFixed::from_int(weighted.len() as i32))
                .floor()
                .raw() as usize,
        ) else {
            return;
        };
        self.active_pattern = Some(selected);
        self.pattern_active = true;
        let variants = self
            .patterns
            .get(selected)
            .map(|pattern| pattern.variants.clone())
            .unwrap_or_default();
        if let Some(pattern) = self.patterns.get_mut(selected) {
            pattern.counter += 1;
            pattern.probability = PicoFixed::ONE;
            if pattern.spawn_enabled {
                self.spawns.clear();
            }
        }
        for variant in variants {
            if let Some(variant_pattern) = self.patterns.get_mut(usize::from(variant - 1)) {
                variant_pattern.probability = PicoFixed::ONE;
            }
        }
    }

    fn update_active_pattern(&mut self) {
        let Some(pattern_index) = self.active_pattern else {
            return;
        };
        let increment = PicoFixed::from_f32(0.02).mul_fixed(self.freeze_rate);
        if let Some(pattern) = self.patterns.get_mut(pattern_index) {
            pattern.timer = pattern.timer.add(self.freeze_rate);
            for rect in &mut pattern.rects {
                if rect.shown {
                    if rect.sh < PicoFixed::from_int(2) {
                        rect.sh = rect.sh.add(increment);
                        if rect.sh > PicoFixed::from_f32(1.99) {
                            rect.sh = PicoFixed::from_int(2);
                        }
                    }
                } else if rect.sh > PicoFixed::ZERO {
                    rect.sh = rect.sh.sub(increment);
                    if rect.sh < PicoFixed::from_f32(0.05) {
                        rect.sh = PicoFixed::ZERO;
                    }
                }
            }
        }
    }

    fn apply_draw_side_effects(&mut self, mode_before: Mode) {
        let renders_game = match mode_before {
            Mode::Game => {
                self.lifecycle.mode == Mode::Game
                    || (self.lifecycle.mode == Mode::TransitionToSettings
                        && self.transition_render_y > 0
                        && matches!(self.transition_from, Mode::Game | Mode::Terminal))
            }
            Mode::TransitionToGame => self.transition_render_y > 0,
            Mode::TransitionToSettings => {
                self.transition_render_y > 0
                    && matches!(self.transition_from, Mode::Game | Mode::Terminal)
            }
            _ => false,
        };
        if self.lifecycle.dead || !renders_game {
            return;
        }
        for enemy in &self.enemies {
            if enemy.personality >= 2 {
                let angle = self.rng.rnd(PicoFixed::ONE);
                let x = enemy.x.floor();
                let y = enemy.y.floor();
                let offset_x = pico_sine(angle).mul_fixed(PicoFixed::from_f32(2.4));
                let offset_y = pico_cosine(angle).mul_fixed(PicoFixed::from_f32(2.4));
                let max_age = PicoFixed::from_int(20).add(self.rng.rnd(PicoFixed::from_int(15)));
                let colors = match enemy.personality {
                    2 => [8, 9, 10],
                    3 => [1, 13, 7],
                    _ => [7, 0, 0],
                };
                let color_count = if enemy.personality == 4 { 1 } else { 3 };
                self.particles.push(ParticleState {
                    x: x.add(offset_x),
                    y: y.add(offset_y),
                    dx: PicoFixed::ZERO,
                    dy: PicoFixed::ZERO,
                    radius: PicoFixed::ZERO,
                    kind: 1,
                    max_age,
                    age: 0,
                    color: 0,
                    colors,
                    color_count,
                });
            }
        }
        self.add_player_trail();
    }

    fn add_player_trail(&mut self) {
        self.particles.push(ParticleState::player_trail(
            self.player.x,
            self.player.y,
            self.player.size,
        ));
    }

    fn update_camera(&mut self) {
        let shake_limit = PicoFixed::from_f32(0.1);
        self.shake = self.shake.min(shake_limit);
        let horizontal = PicoFixed::from_int(16).sub(self.rng.rnd(PicoFixed::from_int(32)));
        let vertical = PicoFixed::from_int(16).sub(self.rng.rnd(PicoFixed::from_int(32)));
        self.camera_x = horizontal.mul_fixed(self.shake);
        self.camera_y = vertical.mul_fixed(self.shake);
        self.shake = self.shake.mul_fixed(PicoFixed::from_f32(0.95));
        if self.shake < PicoFixed::from_f32(0.05) {
            self.shake = PicoFixed::ZERO;
        }
    }

    fn result(&self, reward: PicoFixed, events: Vec<FrameEvent>) -> FrameResult {
        let snapshot = self.snapshot();
        self.result_with_snapshot(reward, events, snapshot)
    }

    fn result_with_snapshot(
        &self,
        reward: PicoFixed,
        events: Vec<FrameEvent>,
        snapshot: Snapshot,
    ) -> FrameResult {
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
            snapshot,
        }
    }
}

fn particle_color(particle: ParticleState) -> u8 {
    if particle.color_count <= 1 {
        return particle.colors[0];
    }
    let age = PicoFixed::from_int(particle.age as i32);
    let fraction = age.div_fixed(particle.max_age).unwrap_or(PicoFixed::ZERO);
    let scaled = fraction.mul_fixed(PicoFixed::from_int(i32::from(particle.color_count)));
    let index = (scaled.floor().raw() >> 16)
        .max(0)
        .min(i32::from(particle.color_count - 1));
    particle
        .colors
        .get(index as usize)
        .copied()
        .or_else(|| particle.colors.first().copied())
        .unwrap_or(0)
}

fn pico_sine(angle: PicoFixed) -> PicoFixed {
    let radians = angle.to_f32() * std::f32::consts::TAU;
    PicoFixed::from_f32(-radians.sin())
}

fn pico_cosine(angle: PicoFixed) -> PicoFixed {
    let radians = angle.to_f32() * std::f32::consts::TAU;
    PicoFixed::from_f32(radians.cos())
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

#[cfg(test)]
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

fn with_sign(magnitude: PicoFixed, signed_value: PicoFixed) -> PicoFixed {
    if signed_value < PicoFixed::ZERO {
        magnitude.neg()
    } else {
        magnitude
    }
}

fn pico_lerp(position: PicoFixed, target: PicoFixed, percentage: PicoFixed) -> PicoFixed {
    PicoFixed::ONE
        .sub(percentage)
        .mul_fixed(position)
        .add(percentage.mul_fixed(target))
}

fn initial_spawns() -> Vec<SpawnPoint> {
    vec![
        SpawnPoint {
            x: SPAWN_EDGE,
            y: SPAWN_EDGE,
        },
        SpawnPoint {
            x: SPAWN_EDGE_FAR,
            y: SPAWN_EDGE,
        },
        SpawnPoint {
            x: SPAWN_EDGE,
            y: SPAWN_EDGE_FAR,
        },
        SpawnPoint {
            x: SPAWN_EDGE_FAR,
            y: SPAWN_EDGE_FAR,
        },
    ]
}

fn difficulty_index(difficulty: u8) -> usize {
    usize::from(difficulty.clamp(1, 3) - 1)
}

// These are the cartridge's `difspd`, `difest`, `incbs`, `tarbs`, `incbm`, and
// `tarbm` tables in the accepted Q16.16 boundary. Keeping table values here,
// rather than converting through host floats at each update, makes the source
// difficulty curve part of the portable deterministic contract.

fn difficulty_increment(value: PicoFixed, half: bool) -> PicoFixed {
    if half {
        PicoFixed::from_raw(value.raw() / 2)
    } else {
        value
    }
}

fn initial_speed(difficulty: u8) -> PicoFixed {
    match difficulty_index(difficulty) {
        0 => PicoFixed::from_f32(0.8),
        1 => PicoFixed::ONE,
        _ => PicoFixed::from_f32(1.6),
    }
}

fn initial_enemy_est(difficulty: u8) -> PicoFixed {
    match difficulty_index(difficulty) {
        0 => PicoFixed::from_f32(1.4),
        1 => PicoFixed::ONE,
        _ => PicoFixed::from_f32(0.8),
    }
}

fn initial_enemy_stats(powerups_enabled: bool) -> [PicoFixed; 5] {
    let powerup_value = if powerups_enabled {
        PicoFixed::from_int(2)
    } else {
        PicoFixed::ZERO
    };
    [
        PicoFixed::from_f32(76.5),
        PicoFixed::from_f32(17.5),
        powerup_value,
        PicoFixed::ZERO,
        powerup_value,
    ]
}

fn initial_bounce_static(difficulty: u8) -> PicoFixed {
    match difficulty_index(difficulty) {
        0 => PicoFixed::from_f32(0.8),
        1 => PicoFixed::from_f32(0.65),
        _ => PicoFixed::from_f32(0.6),
    }
}

fn initial_bounce_moving(difficulty: u8) -> PicoFixed {
    match difficulty_index(difficulty) {
        0 => PicoFixed::from_f32(1.05),
        1 => PicoFixed::from_f32(0.9),
        _ => PicoFixed::from_f32(0.6),
    }
}

fn cycle_index(current: u8, count: u8, increase: bool) -> u8 {
    if increase {
        if current >= count { 1 } else { current + 1 }
    } else if current <= 1 {
        count
    } else {
        current - 1
    }
}

fn theme_values(index: u8) -> (u8, u8) {
    [
        (12, 1),
        (1, 0),
        (3, 1),
        (13, 1),
        (2, 1),
        (9, 8),
        (14, 2),
        (6, 5),
        (5, 0),
        (0, 0),
        (0, 8),
        (0, 12),
        (0, 11),
    ]
    .get(usize::from(index.saturating_sub(1)))
    .copied()
    .unwrap_or((12, 1))
}

fn settings_row(cursor: u8) -> i16 {
    match cursor {
        1 => 29,
        2 => 59,
        3 => 69,
        4 => 79,
        _ => 29,
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
    fn v149_collision_flags_gate_player_and_enemy_collision() {
        let mut player_game = NativeGame::new(NativeConfig::default());
        start_game(&mut player_game);
        player_game.enemies.clear();
        player_game.should_collide = false;
        player_game.enemies.push(EnemyState::normal(
            PicoFixed::from_int(61),
            PicoFixed::from_int(64),
            PicoFixed::from_int(3),
        ));
        assert!(player_game.advance_frame(0).is_ok());
        assert!(!player_game.lifecycle.dead);

        let mut enemy_game = NativeGame::new(NativeConfig::default());
        start_game(&mut enemy_game);
        enemy_game.enemies.clear();
        enemy_game.enemy_should_collide = false;
        let mut first = EnemyState::normal(
            PicoFixed::from_int(10),
            PicoFixed::from_int(10),
            PicoFixed::from_int(3),
        );
        first.inside = true;
        let mut second = first;
        second.x = PicoFixed::from_int(11);
        enemy_game.enemies.push(first);
        enemy_game.enemies.push(second);
        assert!(enemy_game.advance_frame(0).is_ok());
        assert!(enemy_game.enemies().iter().all(|enemy| !enemy.is_dying));
    }

    #[test]
    fn v150_difficulty_curve_uses_all_source_tables_and_half_steps() {
        let expected = [
            (131, 327, 32, 29_491, 85, 29_491),
            (452, 655, 1_245, 29_491, 1_245, 29_491),
            (1_835, 1_310, 1_572, 26_214, 1_572, 26_214),
        ];
        for (difficulty_index, values) in expected.iter().enumerate() {
            let difficulty = u8::try_from(difficulty_index + 1).unwrap_or(1);
            let curve = super::DIFFICULTY_CURVES.get(difficulty_index).copied();
            assert!(curve.is_some());
            let Some(curve) = curve else {
                continue;
            };
            assert_eq!(curve.speed_increment, PicoFixed::from_raw(values.0));
            assert_eq!(curve.enemy_increment, PicoFixed::from_raw(values.1));
            assert_eq!(curve.static_increment, PicoFixed::from_raw(values.2));
            assert_eq!(curve.static_target, PicoFixed::from_raw(values.3));
            assert_eq!(curve.moving_increment, PicoFixed::from_raw(values.4));
            assert_eq!(curve.moving_target, PicoFixed::from_raw(values.5));

            let mut config = NativeConfig::new(42);
            config.difficulty = difficulty;
            let mut full = NativeGame::new(config);
            full.apply_difficulty(false);
            assert_eq!(
                full.speed,
                super::pico_lerp(
                    super::initial_speed(difficulty),
                    super::DIFFICULTY_SPEED_TARGET,
                    curve.speed_increment,
                )
            );
            assert_eq!(
                full.enemy_est,
                super::pico_lerp(
                    super::initial_enemy_est(difficulty),
                    super::DIFFICULTY_EST_TARGET,
                    curve.enemy_increment,
                )
            );
            assert_eq!(
                full.bounce_cap_static,
                super::pico_lerp(
                    super::initial_bounce_static(difficulty),
                    curve.static_target,
                    curve.static_increment,
                )
            );
            assert_eq!(
                full.bounce_cap_moving,
                super::pico_lerp(
                    super::initial_bounce_moving(difficulty),
                    curve.moving_target,
                    curve.moving_increment,
                )
            );

            let mut half = NativeGame::new(config);
            half.apply_difficulty(true);
            assert_eq!(
                half.speed,
                super::pico_lerp(
                    super::initial_speed(difficulty),
                    super::DIFFICULTY_SPEED_TARGET,
                    super::difficulty_increment(curve.speed_increment, true),
                )
            );
            assert_eq!(
                half.enemy_est,
                super::pico_lerp(
                    super::initial_enemy_est(difficulty),
                    super::DIFFICULTY_EST_TARGET,
                    super::difficulty_increment(curve.enemy_increment, true),
                )
            );
            assert_eq!(
                half.bounce_cap_static,
                super::pico_lerp(
                    super::initial_bounce_static(difficulty),
                    curve.static_target,
                    super::difficulty_increment(curve.static_increment, true),
                )
            );
            assert_eq!(
                half.bounce_cap_moving,
                super::pico_lerp(
                    super::initial_bounce_moving(difficulty),
                    curve.moving_target,
                    super::difficulty_increment(curve.moving_increment, true),
                )
            );
        }
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

    #[test]
    fn v140_kamikaze_speed_updates_once_before_movement() {
        let mut game = NativeGame::new(NativeConfig::new(42));
        start_game(&mut game);
        game.enemies.push(EnemyState {
            x: PicoFixed::from_int(80),
            y: PicoFixed::from_int(64),
            personality: 1,
            ..EnemyState::normal(
                PicoFixed::from_int(80),
                PicoFixed::from_int(64),
                PicoFixed::from_int(3),
            )
        });
        assert!(game.advance_frame(0).is_ok());
        assert_eq!(
            game.enemies().last().map(|enemy| enemy.speed),
            Some(PicoFixed::from_raw(64_881))
        );
    }
}
