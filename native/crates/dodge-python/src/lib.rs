#![doc = "PyO3/NumPy boundary for batched native Dodge training."]

use dodge_batch::{
    ACTION_COUNT, BOARD_CHANNELS, BOARD_HEIGHT, BOARD_WIDTH, BatchConfig, BatchEnvironment,
    BatchError, BatchObservation, ExecutionMode, ML_OBSERVATION_SIZE, ObservationFlags,
    PIXEL_HEIGHT, PIXEL_WIDTH,
};
use dodge_core::{Action, FrameEvent, Mode};
use ndarray::{Array1, Array2, Array3, Array4};
use numpy::{IntoPyArray, PyReadonlyArray1};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict, PyList};

const BATCH_SCHEMA_VERSION: u32 = 1;

/// A persistent native batch environment exposed to Python.
#[pyclass(name = "NativeBatchEnv")]
pub struct NativeBatchEnv {
    inner: BatchEnvironment,
    flags: ObservationFlags,
}

#[pymethods]
impl NativeBatchEnv {
    #[new]
    #[pyo3(signature = (step_frames=4, execution="serial", full_state=false, pixels=false, board=true, difficulty=2, patterns_enabled=true, powerups_enabled=true, ml=false, ml_grid_spacing=32, include_offscreen_board=false))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        step_frames: u32,
        execution: &str,
        full_state: bool,
        pixels: bool,
        board: bool,
        difficulty: u8,
        patterns_enabled: bool,
        powerups_enabled: bool,
        ml: bool,
        ml_grid_spacing: u32,
        include_offscreen_board: bool,
    ) -> PyResult<Self> {
        let execution = parse_execution(execution)?;
        let mut config = BatchConfig::new(step_frames);
        config.native.difficulty = difficulty;
        config.native.patterns_enabled = patterns_enabled;
        config.native.powerups_enabled = powerups_enabled;
        config.observations = ObservationFlags {
            full_state,
            pixels,
            ml,
            ml_grid_spacing,
            board,
            include_offscreen_board,
        };
        config.execution = execution;
        let inner = BatchEnvironment::new(config).map_err(batch_error)?;
        Ok(Self {
            inner,
            flags: config.observations,
        })
    }

    #[getter]
    fn lane_count(&self) -> usize {
        self.inner.lane_count()
    }

    #[getter]
    fn step_frames(&self) -> u32 {
        self.inner.config().step_frames
    }

    #[getter]
    fn execution(&self) -> &'static str {
        match self.inner.config().execution {
            ExecutionMode::Serial => "serial",
            ExecutionMode::Parallel => "parallel",
        }
    }

    #[getter]
    fn observation_flags<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let flags = PyDict::new(py);
        flags.set_item("full_state", self.flags.full_state)?;
        flags.set_item("pixels", self.flags.pixels)?;
        flags.set_item("board", self.flags.board)?;
        flags.set_item(
            "include_offscreen_board",
            self.flags.include_offscreen_board,
        )?;
        flags.set_item("ml", self.flags.ml)?;
        flags.set_item("ml_grid_spacing", self.flags.ml_grid_spacing)?;
        Ok(flags)
    }

    fn reset_batch<'py>(
        &mut self,
        py: Python<'py>,
        seeds: PyReadonlyArray1<'_, u32>,
    ) -> PyResult<Bound<'py, PyDict>> {
        let seeds = seeds
            .as_slice()
            .map_err(|error| PyValueError::new_err(error.to_string()))?
            .to_vec();
        let observations = py
            .detach(|| self.inner.reset(&seeds))
            .map_err(batch_error)?;
        observations_to_dict(py, observations, self.flags)
    }

    fn reset_lanes<'py>(
        &mut self,
        py: Python<'py>,
        lanes: PyReadonlyArray1<'_, u32>,
        seeds: PyReadonlyArray1<'_, u32>,
    ) -> PyResult<Bound<'py, PyDict>> {
        let lanes = lanes
            .as_slice()
            .map_err(|error| PyValueError::new_err(error.to_string()))?
            .iter()
            .copied()
            .map(|lane| lane as usize)
            .collect::<Vec<_>>();
        let seeds = seeds
            .as_slice()
            .map_err(|error| PyValueError::new_err(error.to_string()))?
            .to_vec();
        let observations = py
            .detach(|| self.inner.reset_lanes(&lanes, &seeds))
            .map_err(batch_error)?;
        observations_to_dict(py, observations, self.flags)
    }

    fn reset_ml_batch<'py>(
        &mut self,
        py: Python<'py>,
        seeds: PyReadonlyArray1<'_, u32>,
    ) -> PyResult<Bound<'py, PyDict>> {
        let seeds = seeds
            .as_slice()
            .map_err(|error| PyValueError::new_err(error.to_string()))?
            .to_vec();
        let observations = py
            .detach(|| self.inner.reset_ml(&seeds))
            .map_err(batch_error)?;
        ml_observations_to_dict(py, observations)
    }

    fn reset_ml_lanes<'py>(
        &mut self,
        py: Python<'py>,
        lanes: PyReadonlyArray1<'_, u32>,
        seeds: PyReadonlyArray1<'_, u32>,
    ) -> PyResult<Bound<'py, PyDict>> {
        let lanes = lanes
            .as_slice()
            .map_err(|error| PyValueError::new_err(error.to_string()))?
            .iter()
            .copied()
            .map(|lane| lane as usize)
            .collect::<Vec<_>>();
        let seeds = seeds
            .as_slice()
            .map_err(|error| PyValueError::new_err(error.to_string()))?
            .to_vec();
        let observations = py
            .detach(|| self.inner.reset_ml_lanes(&lanes, &seeds))
            .map_err(batch_error)?;
        ml_observations_to_dict(py, observations)
    }

    fn step_batch<'py>(
        &mut self,
        py: Python<'py>,
        actions: PyReadonlyArray1<'_, u8>,
    ) -> PyResult<Bound<'py, PyDict>> {
        let action_values = actions
            .as_slice()
            .map_err(|error| PyValueError::new_err(error.to_string()))?;
        let native_actions = action_values
            .iter()
            .copied()
            .map(action_from_index)
            .collect::<PyResult<Vec<_>>>()?;
        let observations = py
            .detach(|| self.inner.step(&native_actions))
            .map_err(batch_error)?;
        observations_to_dict(py, observations, self.flags)
    }

    fn step_ml_batch<'py>(
        &mut self,
        py: Python<'py>,
        actions: PyReadonlyArray1<'_, u8>,
    ) -> PyResult<Bound<'py, PyDict>> {
        let action_values = actions
            .as_slice()
            .map_err(|error| PyValueError::new_err(error.to_string()))?;
        let native_actions = action_values
            .iter()
            .copied()
            .map(action_from_index)
            .collect::<PyResult<Vec<_>>>()?;
        let observations = py
            .detach(|| self.inner.step_ml(&native_actions))
            .map_err(batch_error)?;
        ml_observations_to_dict(py, observations)
    }

    /// Return survival-frame deltas for every action from each canonical state.
    /// The live batch is borrowed immutably and is never advanced by scoring.
    fn score_actions<'py>(
        &self,
        py: Python<'py>,
        snapshots: &Bound<'_, PyList>,
        lookahead_steps: u32,
    ) -> PyResult<Bound<'py, PyDict>> {
        let snapshots = snapshots
            .iter()
            .map(|value| {
                let bytes = value.cast::<PyBytes>().map_err(|_| {
                    PyValueError::new_err(
                        "counterfactual snapshots must be a non-empty list of bytes",
                    )
                })?;
                Ok(bytes.as_bytes().to_vec())
            })
            .collect::<PyResult<Vec<_>>>()?;
        let snapshot_count = snapshots.len();
        let scores = py
            .detach(|| self.inner.score_actions(&snapshots, lookahead_steps))
            .map_err(batch_error)?;
        let values = scores
            .into_iter()
            .flat_map(|row| row.into_iter())
            .collect::<Vec<_>>();
        let scores = Array2::from_shape_vec((snapshot_count, ACTION_COUNT), values)
            .map_err(|error| PyRuntimeError::new_err(error.to_string()))?
            .into_pyarray(py);
        let result = PyDict::new(py);
        result.set_item("schema_version", BATCH_SCHEMA_VERSION)?;
        result.set_item("snapshot_count", snapshot_count)?;
        result.set_item("action_count", ACTION_COUNT)?;
        result.set_item("lookahead_steps", lookahead_steps)?;
        result.set_item("scores", scores)?;
        Ok(result)
    }
}

#[pymodule]
fn dodge_native(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add("ML_OBSERVATION_SHAPE", (ML_OBSERVATION_SIZE,))?;
    module.add("BATCH_SCHEMA_VERSION", BATCH_SCHEMA_VERSION)?;
    module.add("BOARD_SHAPE", (BOARD_CHANNELS, BOARD_HEIGHT, BOARD_WIDTH))?;
    module.add("PIXEL_SHAPE", (PIXEL_HEIGHT, PIXEL_WIDTH))?;
    module.add_class::<NativeBatchEnv>()?;
    Ok(())
}

fn parse_execution(value: &str) -> PyResult<ExecutionMode> {
    match value {
        "serial" => Ok(ExecutionMode::Serial),
        "parallel" => Ok(ExecutionMode::Parallel),
        _ => Err(PyValueError::new_err(
            "execution must be 'serial' or 'parallel'",
        )),
    }
}

fn action_from_index(index: u8) -> PyResult<Action> {
    Action::ALL
        .get(usize::from(index))
        .copied()
        .ok_or_else(|| PyValueError::new_err(format!("invalid Dodge action index: {index}")))
}

fn batch_error(error: BatchError) -> PyErr {
    match error {
        BatchError::Core(core) => PyRuntimeError::new_err(core.to_string()),
        other => PyValueError::new_err(other.to_string()),
    }
}

fn observations_to_dict<'py>(
    py: Python<'py>,
    observations: Vec<BatchObservation>,
    flags: ObservationFlags,
) -> PyResult<Bound<'py, PyDict>> {
    let lane_count = observations.len();
    let frames = observations.iter().map(|value| value.frame).collect();
    let lane_ids = observations.iter().map(|value| value.lane as u32).collect();
    let frames_advanced = observations
        .iter()
        .map(|value| value.frames_advanced)
        .collect();
    let rewards = observations
        .iter()
        .map(|value| value.reward as f32)
        .collect();
    let done = observations.iter().map(|value| value.done).collect();
    let seeds = observations.iter().map(|value| value.seed).collect();
    let state_hashes = observations.iter().map(|value| value.state_hash).collect();
    let pixel_hashes = observations.iter().map(|value| value.pixel_hash).collect();
    let modes = observations
        .iter()
        .map(|value| mode_code(value.mode))
        .collect();
    let event_flags = observations
        .iter()
        .map(|value| event_flags_code(&value.events))
        .collect();

    let frames = Array1::from_vec(frames).into_pyarray(py);
    let lane_ids = Array1::from_vec(lane_ids).into_pyarray(py);
    let frames_advanced = Array1::from_vec(frames_advanced).into_pyarray(py);
    let rewards = Array1::from_vec(rewards).into_pyarray(py);
    let done = Array1::from_vec(done).into_pyarray(py);
    let seeds = Array1::from_vec(seeds).into_pyarray(py);
    let state_hashes = Array1::from_vec(state_hashes).into_pyarray(py);
    let pixel_hashes = Array1::from_vec(pixel_hashes).into_pyarray(py);
    let modes = Array1::from_vec(modes).into_pyarray(py);
    let event_flags = Array1::from_vec(event_flags).into_pyarray(py);

    let pixels = if flags.pixels {
        let values = observations
            .iter()
            .flat_map(|value| {
                value
                    .pixels
                    .as_ref()
                    .into_iter()
                    .flat_map(|pixels| pixels.iter().copied())
            })
            .collect::<Vec<_>>();
        Some(
            Array3::from_shape_vec((lane_count, PIXEL_HEIGHT, PIXEL_WIDTH), values)
                .map_err(|error| PyRuntimeError::new_err(error.to_string()))?
                .into_pyarray(py),
        )
    } else {
        None
    };
    let board = if flags.board {
        let values = observations
            .iter()
            .flat_map(|value| {
                value
                    .board
                    .as_ref()
                    .into_iter()
                    .flat_map(|board| board.as_slice().iter().copied())
            })
            .collect::<Vec<_>>();
        Some(
            Array4::from_shape_vec(
                (lane_count, BOARD_CHANNELS, BOARD_HEIGHT, BOARD_WIDTH),
                values,
            )
            .map_err(|error| PyRuntimeError::new_err(error.to_string()))?
            .into_pyarray(py),
        )
    } else {
        None
    };
    let ml_observation = if flags.ml {
        let values = observations
            .iter()
            .flat_map(|value| {
                value
                    .ml_observation
                    .as_ref()
                    .into_iter()
                    .flat_map(|observation| observation.iter().copied())
            })
            .collect::<Vec<_>>();
        Some(
            Array2::from_shape_vec((lane_count, ML_OBSERVATION_SIZE), values)
                .map_err(|error| PyRuntimeError::new_err(error.to_string()))?
                .into_pyarray(py),
        )
    } else {
        None
    };
    let player_positions = if flags.ml {
        let values = observations
            .iter()
            .flat_map(|value| {
                value
                    .player_position
                    .as_ref()
                    .into_iter()
                    .flat_map(|position| position.iter().copied())
            })
            .collect::<Vec<_>>();
        Some(
            Array2::from_shape_vec((lane_count, 2), values)
                .map_err(|error| PyRuntimeError::new_err(error.to_string()))?
                .into_pyarray(py),
        )
    } else {
        None
    };

    let snapshots = PyList::empty(py);
    for observation in &observations {
        match observation.canonical_snapshot.as_deref() {
            Some(bytes) => snapshots.append(PyBytes::new(py, bytes))?,
            None => snapshots.append(py.None())?,
        }
    }

    let result = PyDict::new(py);
    result.set_item("schema_version", BATCH_SCHEMA_VERSION)?;
    result.set_item("lane_count", lane_count)?;
    result.set_item("lane_ids", lane_ids)?;
    result.set_item("frames", frames)?;
    result.set_item("frames_advanced", frames_advanced)?;
    result.set_item("rewards", rewards)?;
    result.set_item("done", done)?;
    result.set_item("seeds", seeds)?;
    result.set_item("state_hashes", state_hashes)?;
    result.set_item("pixel_hashes", pixel_hashes)?;
    result.set_item("modes", modes)?;
    result.set_item("ml_observation", ml_observation)?;
    result.set_item("player_positions", player_positions)?;
    result.set_item("event_flags", event_flags)?;
    result.set_item("pixels", pixels)?;
    result.set_item("board", board)?;
    result.set_item("snapshot_bytes", snapshots)?;
    Ok(result)
}

fn ml_observations_to_dict<'py>(
    py: Python<'py>,
    observations: Vec<dodge_batch::MlBatchObservation>,
) -> PyResult<Bound<'py, PyDict>> {
    let lane_count = observations.len();
    let lane_ids = observations.iter().map(|value| value.lane as u32).collect();
    let frames = observations.iter().map(|value| value.frame).collect();
    let frames_advanced = observations
        .iter()
        .map(|value| value.frames_advanced)
        .collect();
    let rewards = observations
        .iter()
        .map(|value| value.reward as f32)
        .collect();
    let done = observations.iter().map(|value| value.done).collect();
    let seeds = observations.iter().map(|value| value.seed).collect();
    let modes = observations
        .iter()
        .map(|value| mode_code(value.mode))
        .collect();
    let ml_values = observations
        .iter()
        .flat_map(|value| value.ml_observation)
        .collect::<Vec<_>>();
    let positions = observations
        .iter()
        .flat_map(|value| value.player_position)
        .collect::<Vec<_>>();

    let lane_ids = Array1::from_vec(lane_ids).into_pyarray(py);
    let frames = Array1::from_vec(frames).into_pyarray(py);
    let frames_advanced = Array1::from_vec(frames_advanced).into_pyarray(py);
    let rewards = Array1::from_vec(rewards).into_pyarray(py);
    let done = Array1::from_vec(done).into_pyarray(py);
    let seeds = Array1::from_vec(seeds).into_pyarray(py);
    let modes = Array1::from_vec(modes).into_pyarray(py);
    let ml_observation = Array2::from_shape_vec((lane_count, ML_OBSERVATION_SIZE), ml_values)
        .map_err(|error| PyRuntimeError::new_err(error.to_string()))?
        .into_pyarray(py);
    let player_positions = Array2::from_shape_vec((lane_count, 2), positions)
        .map_err(|error| PyRuntimeError::new_err(error.to_string()))?
        .into_pyarray(py);

    let result = PyDict::new(py);
    result.set_item("schema_version", BATCH_SCHEMA_VERSION)?;
    result.set_item("fast_ml", true)?;
    result.set_item("lane_count", lane_count)?;
    result.set_item("lane_ids", lane_ids)?;
    result.set_item("frames", frames)?;
    result.set_item("frames_advanced", frames_advanced)?;
    result.set_item("rewards", rewards)?;
    result.set_item("done", done)?;
    result.set_item("seeds", seeds)?;
    result.set_item("modes", modes)?;
    result.set_item("ml_observation", ml_observation)?;
    result.set_item("player_positions", player_positions)?;
    Ok(result)
}

fn mode_code(mode: Mode) -> u8 {
    match mode {
        Mode::Menu => 0,
        Mode::TransitionToGame => 1,
        Mode::Game => 2,
        Mode::Terminal => 3,
        Mode::Settings => 4,
        Mode::TransitionToSettings => 5,
        Mode::TransitionToMenu => 6,
    }
}

fn event_flags_code(events: &[FrameEvent]) -> u32 {
    events.iter().fold(0_u32, |flags, event| {
        flags
            | match event {
                FrameEvent::EnemySpawn => 1 << 0,
                FrameEvent::Collision => 1 << 1,
                FrameEvent::Death => 1 << 2,
                FrameEvent::PatternActive => 1 << 3,
                FrameEvent::Terminal => 1 << 4,
            }
    })
}

#[cfg(test)]
mod tests {
    use super::{action_from_index, event_flags_code, mode_code};
    use dodge_core::{Action, FrameEvent, Mode};

    #[test]
    fn action_indices_preserve_the_nine_action_contract() {
        assert_eq!(action_from_index(0).ok(), Some(Action::Neutral));
        assert_eq!(action_from_index(8).ok(), Some(Action::DownRight));
        assert!(action_from_index(9).is_err());
    }

    #[test]
    fn metadata_codes_are_stable() {
        assert_eq!(mode_code(Mode::Menu), 0);
        assert_eq!(mode_code(Mode::Game), 2);
        assert_eq!(
            event_flags_code(&[FrameEvent::EnemySpawn, FrameEvent::Terminal]),
            17
        );
    }
}
