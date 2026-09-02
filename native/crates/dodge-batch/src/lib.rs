#![doc = "Deterministic serial and parallel native Dodge training lanes."]

use std::fmt::{Display, Formatter};

use dodge_core::{
    Action, AudioEvent, BUTTON_X_MASK, CoreError, FRAMEBUFFER_SIZE, FrameEvent, FullState, Mode,
    NativeConfig, NativeGame, RenderState, Snapshot,
};
use rayon::prelude::*;

mod board;

pub use board::{BOARD_CHANNELS, BOARD_HEIGHT, BOARD_SIZE, BOARD_VALUES, BOARD_WIDTH, Board19x16};

const START_HOLD_FRAMES: usize = 13;

/// Selects whether independent environment lanes execute serially or with Rayon.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum ExecutionMode {
    #[default]
    Serial,
    Parallel,
}

/// Chooses which observation buffers a batch call materializes.
///
/// The core always computes a canonical snapshot internally so the game and
/// render boundaries remain the same. Flags control which owned views are
/// retained in the batch result.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ObservationFlags {
    pub full_state: bool,
    pub pixels: bool,
    pub board: bool,
}

impl ObservationFlags {
    pub const fn all() -> Self {
        Self {
            full_state: true,
            pixels: true,
            board: true,
        }
    }

    pub const fn training_board() -> Self {
        Self {
            full_state: false,
            pixels: false,
            board: true,
        }
    }

    pub const fn pixels_and_state() -> Self {
        Self {
            full_state: true,
            pixels: true,
            board: false,
        }
    }
}

impl Default for ObservationFlags {
    fn default() -> Self {
        Self::all()
    }
}

/// Configuration shared by every lane in one batch.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct BatchConfig {
    pub native: NativeConfig,
    pub step_frames: u32,
    pub observations: ObservationFlags,
    pub execution: ExecutionMode,
}

impl BatchConfig {
    pub const fn new(step_frames: u32) -> Self {
        Self {
            native: NativeConfig::new(42),
            step_frames,
            observations: ObservationFlags::all(),
            execution: ExecutionMode::Serial,
        }
    }

    fn validate(self) -> Result<Self, BatchError> {
        if !(3..=5).contains(&self.step_frames) {
            return Err(BatchError::InvalidStepFrames(self.step_frames));
        }
        Ok(self)
    }
}

impl Default for BatchConfig {
    fn default() -> Self {
        Self::new(4)
    }
}

/// One ordered result from reset or step.
#[derive(Clone, Debug, PartialEq)]
pub struct BatchObservation {
    pub lane: usize,
    pub seed: u32,
    pub frame: u32,
    pub frames_advanced: u32,
    pub reward: u32,
    pub done: bool,
    pub mode: Mode,
    pub events: Vec<FrameEvent>,
    pub audio: Vec<AudioEvent>,
    pub state_hash: u64,
    pub pixel_hash: u64,
    pub full_state: Option<FullState>,
    pub render_state: Option<RenderState>,
    pub pixels: Option<[u8; FRAMEBUFFER_SIZE]>,
    pub board: Option<Board19x16>,
    pub canonical_snapshot: Option<Vec<u8>>,
}

impl BatchObservation {
    pub fn is_game_ready(&self) -> bool {
        self.full_state
            .as_ref()
            .is_some_and(|state| state.lifecycle.game_ready)
    }

    pub fn is_dead(&self) -> bool {
        self.full_state
            .as_ref()
            .is_some_and(|state| state.lifecycle.dead)
    }
}

/// Errors raised at the batch boundary before a result can be returned.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum BatchError {
    InvalidStepFrames(u32),
    EmptyBatch,
    LaneCountMismatch { expected: usize, actual: usize },
    LaneAlreadyDone(usize),
    Core(CoreError),
}

impl Display for BatchError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidStepFrames(frames) => {
                write!(
                    formatter,
                    "batch step_frames must be between 3 and 5: {frames}"
                )
            }
            Self::EmptyBatch => formatter.write_str("batch must contain at least one lane"),
            Self::LaneCountMismatch { expected, actual } => write!(
                formatter,
                "batch lane count mismatch: expected {expected}, got {actual}"
            ),
            Self::LaneAlreadyDone(lane) => write!(formatter, "batch lane {lane} is complete"),
            Self::Core(error) => Display::fmt(error, formatter),
        }
    }
}

impl std::error::Error for BatchError {}

impl From<CoreError> for BatchError {
    fn from(error: CoreError) -> Self {
        Self::Core(error)
    }
}

/// Persistent independent native environments with stable lane ordering.
#[derive(Clone, Debug)]
pub struct BatchEnvironment {
    config: BatchConfig,
    games: Vec<NativeGame>,
    seeds: Vec<u32>,
    last_frames: Vec<u32>,
    last_survival_frames: Vec<u32>,
    done: Vec<bool>,
}

impl BatchEnvironment {
    pub fn new(config: BatchConfig) -> Result<Self, BatchError> {
        Ok(Self {
            config: config.validate()?,
            games: Vec::new(),
            seeds: Vec::new(),
            last_frames: Vec::new(),
            last_survival_frames: Vec::new(),
            done: Vec::new(),
        })
    }

    pub const fn config(&self) -> BatchConfig {
        self.config
    }

    pub fn lane_count(&self) -> usize {
        self.games.len()
    }

    pub fn reset(&mut self, seeds: &[u32]) -> Result<Vec<BatchObservation>, BatchError> {
        if seeds.is_empty() {
            return Err(BatchError::EmptyBatch);
        }
        self.seeds = seeds.to_vec();
        self.games = seeds
            .iter()
            .copied()
            .map(|seed| {
                let mut config = self.config.native;
                config.seed = seed;
                NativeGame::new(config)
            })
            .collect();
        self.last_frames = vec![0; seeds.len()];
        self.last_survival_frames = vec![0; seeds.len()];
        self.done = vec![false; seeds.len()];

        let mut observations = Vec::with_capacity(seeds.len());
        let game_count = self.games.len();
        for lane in 0..seeds.len() {
            let game = self
                .games
                .get_mut(lane)
                .ok_or(BatchError::LaneCountMismatch {
                    expected: seeds.len(),
                    actual: game_count,
                })?;
            let mut snapshot = game.reset();
            for _ in 0..START_HOLD_FRAMES {
                snapshot = game.advance_frame(BUTTON_X_MASK)?.snapshot;
            }
            let state = snapshot.logical_state();
            if let Some(last_frame) = self.last_frames.get_mut(lane) {
                *last_frame = state.lifecycle.frame;
            }
            if let Some(last_survival) = self.last_survival_frames.get_mut(lane) {
                *last_survival = state.survival_frames;
            }
            observations.push(observation_from_snapshot(
                lane,
                0,
                false,
                Vec::new(),
                Vec::new(),
                &snapshot,
                self.config.observations,
            ));
        }
        Ok(observations)
    }

    pub fn step(&mut self, actions: &[Action]) -> Result<Vec<BatchObservation>, BatchError> {
        self.validate_actions(actions)?;
        match self.config.execution {
            ExecutionMode::Serial => self.step_serial(actions),
            ExecutionMode::Parallel => self.step_parallel(actions),
        }
    }

    fn validate_actions(&self, actions: &[Action]) -> Result<(), BatchError> {
        if actions.is_empty() {
            return Err(BatchError::EmptyBatch);
        }
        if actions.len() != self.games.len() {
            return Err(BatchError::LaneCountMismatch {
                expected: self.games.len(),
                actual: actions.len(),
            });
        }
        for (lane, complete) in self.done.iter().copied().enumerate() {
            if complete {
                return Err(BatchError::LaneAlreadyDone(lane));
            }
        }
        Ok(())
    }

    fn step_serial(&mut self, actions: &[Action]) -> Result<Vec<BatchObservation>, BatchError> {
        let mut observations = Vec::with_capacity(actions.len());
        let game_count = self.games.len();
        for (lane, action) in actions.iter().copied().enumerate() {
            let game = self
                .games
                .get_mut(lane)
                .ok_or(BatchError::LaneCountMismatch {
                    expected: actions.len(),
                    actual: game_count,
                })?;
            let result = game.step(action, self.config.step_frames)?;
            observations.push(self.record_step(lane, result)?);
        }
        Ok(observations)
    }

    fn step_parallel(&mut self, actions: &[Action]) -> Result<Vec<BatchObservation>, BatchError> {
        let previous_frames = self.last_frames.clone();
        let previous_survival_frames = self.last_survival_frames.clone();
        let flags = self.config.observations;
        let step_frames = self.config.step_frames;
        let results: Vec<Result<IndexedStep, BatchError>> =
            self.games
                .par_iter_mut()
                .enumerate()
                .zip(actions.par_iter().copied())
                .map(|((lane, game), action)| {
                    let previous_frame = previous_frames.get(lane).copied().ok_or(
                        BatchError::LaneCountMismatch {
                            expected: actions.len(),
                            actual: previous_frames.len(),
                        },
                    )?;
                    let previous_survival = previous_survival_frames.get(lane).copied().ok_or(
                        BatchError::LaneCountMismatch {
                            expected: actions.len(),
                            actual: previous_survival_frames.len(),
                        },
                    )?;
                    let result = game.step(action, step_frames)?;
                    Ok(IndexedStep {
                        lane,
                        previous_frame,
                        previous_survival,
                        result,
                        flags,
                    })
                })
                .collect();

        let mut observations = Vec::with_capacity(results.len());
        for indexed in results {
            let indexed = indexed?;
            let lane = indexed.lane;
            let frame_result = indexed.result;
            let observation = observation_from_frame_result(
                lane,
                indexed.previous_frame,
                indexed.previous_survival,
                &frame_result,
                indexed.flags,
            );
            if let Some(last_frame) = self.last_frames.get_mut(lane) {
                *last_frame = frame_result.frame;
            }
            if let Some(last_survival) = self.last_survival_frames.get_mut(lane) {
                *last_survival = frame_result.snapshot.logical_state().survival_frames;
            }
            if let Some(done) = self.done.get_mut(lane) {
                *done = frame_result.done;
            }
            observations.push(observation);
        }
        Ok(observations)
    }

    fn record_step(
        &mut self,
        lane: usize,
        result: dodge_core::FrameResult,
    ) -> Result<BatchObservation, BatchError> {
        let previous_frame =
            self.last_frames
                .get(lane)
                .copied()
                .ok_or(BatchError::LaneCountMismatch {
                    expected: self.games.len(),
                    actual: lane + 1,
                })?;
        let previous_survival =
            self.last_survival_frames
                .get(lane)
                .copied()
                .ok_or(BatchError::LaneCountMismatch {
                    expected: self.games.len(),
                    actual: lane + 1,
                })?;
        let observation = observation_from_frame_result(
            lane,
            previous_frame,
            previous_survival,
            &result,
            self.config.observations,
        );
        if let Some(last_frame) = self.last_frames.get_mut(lane) {
            *last_frame = result.frame;
        }
        if let Some(last_survival) = self.last_survival_frames.get_mut(lane) {
            *last_survival = result.snapshot.logical_state().survival_frames;
        }
        if let Some(done) = self.done.get_mut(lane) {
            *done = result.done;
        }
        Ok(observation)
    }
}

struct IndexedStep {
    lane: usize,
    previous_frame: u32,
    previous_survival: u32,
    result: dodge_core::FrameResult,
    flags: ObservationFlags,
}

fn observation_from_frame_result(
    lane: usize,
    previous_frame: u32,
    previous_survival: u32,
    result: &dodge_core::FrameResult,
    flags: ObservationFlags,
) -> BatchObservation {
    let frames_advanced = result.frame.saturating_sub(previous_frame);
    let survival_now = result.snapshot.logical_state().survival_frames;
    let reward = survival_now.saturating_sub(previous_survival);
    observation_from_snapshot(
        lane,
        frames_advanced,
        result.done,
        result.events.clone(),
        result.audio.clone(),
        &result.snapshot,
        flags,
    )
    .with_reward(reward)
}

impl BatchObservation {
    fn with_reward(mut self, reward: u32) -> Self {
        self.reward = reward;
        self
    }
}

fn observation_from_snapshot(
    lane: usize,
    frames_advanced: u32,
    done: bool,
    events: Vec<FrameEvent>,
    audio: Vec<AudioEvent>,
    snapshot: &Snapshot,
    flags: ObservationFlags,
) -> BatchObservation {
    let logical_state = snapshot.logical_state();
    BatchObservation {
        lane,
        seed: logical_state.seed,
        frame: logical_state.lifecycle.frame,
        frames_advanced,
        reward: 0,
        done,
        mode: logical_state.lifecycle.mode,
        events,
        audio,
        state_hash: snapshot.state_hash(),
        pixel_hash: snapshot.pixel_hash(),
        full_state: flags.full_state.then(|| logical_state.clone()),
        render_state: flags.full_state.then(|| snapshot.render_state().clone()),
        pixels: flags.pixels.then(|| *snapshot.pixels()),
        board: flags
            .board
            .then(|| Board19x16::from_full_state(logical_state)),
        canonical_snapshot: flags.full_state.then(|| snapshot.canonical_bytes()),
    }
}

#[cfg(test)]
mod tests {
    use super::{
        Action, BOARD_VALUES, BatchConfig, BatchEnvironment, ExecutionMode, ObservationFlags,
    };

    fn configured(execution: ExecutionMode) -> BatchConfig {
        BatchConfig {
            native: dodge_core::NativeConfig::new(42),
            step_frames: 4,
            observations: ObservationFlags::all(),
            execution,
        }
    }

    #[test]
    fn reset_starts_every_lane_at_the_game_ready_boundary() {
        let mut environment = BatchEnvironment::new(configured(ExecutionMode::Serial))
            .unwrap_or_else(|_| unreachable!("valid batch config"));
        let observations = environment
            .reset(&[13, 27])
            .unwrap_or_else(|_| unreachable!("reset should succeed"));
        assert_eq!(observations.len(), 2);
        assert!(
            observations
                .iter()
                .all(|observation| observation.frame == 13)
        );
        assert!(
            observations
                .iter()
                .all(|observation| observation.is_game_ready())
        );
        assert!(
            observations
                .iter()
                .all(|observation| observation.reward == 0)
        );
    }

    #[test]
    fn serial_and_parallel_lanes_match_in_order() {
        let seeds = [13, 27, 58, 101];
        let actions = [
            Action::Neutral,
            Action::Left,
            Action::UpRight,
            Action::DownLeft,
        ];
        let mut serial = BatchEnvironment::new(configured(ExecutionMode::Serial))
            .unwrap_or_else(|_| unreachable!("valid batch config"));
        let mut parallel = BatchEnvironment::new(configured(ExecutionMode::Parallel))
            .unwrap_or_else(|_| unreachable!("valid batch config"));
        let serial_reset = serial
            .reset(&seeds)
            .unwrap_or_else(|_| unreachable!("reset should succeed"));
        let parallel_reset = parallel
            .reset(&seeds)
            .unwrap_or_else(|_| unreachable!("reset should succeed"));
        assert_eq!(serial_reset, parallel_reset);

        for _ in 0..90 {
            let serial_step = serial
                .step(&actions)
                .unwrap_or_else(|_| unreachable!("serial step should succeed"));
            let parallel_step = parallel
                .step(&actions)
                .unwrap_or_else(|_| unreachable!("parallel step should succeed"));
            assert_eq!(serial_step, parallel_step);
            if serial_step.iter().any(|observation| observation.done) {
                break;
            }
        }
    }

    #[test]
    fn board_buffer_has_the_documented_flat_length() {
        let mut environment = BatchEnvironment::new(configured(ExecutionMode::Serial))
            .unwrap_or_else(|_| unreachable!("valid batch config"));
        let observations = environment
            .reset(&[42])
            .unwrap_or_else(|_| unreachable!("reset should succeed"));
        let board = observations
            .first()
            .and_then(|observation| observation.board.as_ref())
            .unwrap_or_else(|| unreachable!("board requested"));
        assert_eq!(board.as_slice().len(), BOARD_VALUES);
        assert!(board.as_slice().iter().all(|value| value.is_finite()));
    }
}
