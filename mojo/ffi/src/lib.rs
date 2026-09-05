//! Narrow C ABI for the independent Mojo waypoint-DQN investigation.
//!
//! The simulator remains the existing Rust native batch engine.  This crate
//! deliberately does not depend on PyO3 or the Python package: Mojo receives
//! the same 225-float ML observations, positions, rewards, and terminal flags
//! through a small owned buffer boundary.

#![allow(unsafe_code)]

use std::ffi::{CString, c_char};
use std::ptr;

use dodge_batch::{
    BatchConfig, BatchEnvironment, ExecutionMode, ML_OBSERVATION_SIZE, MlBatchObservation,
    ObservationFlags,
};
use dodge_core::Action;

const ABI_VERSION: u32 = 1;
const INVALID_HANDLE: &[u8] = b"invalid batch handle\0";
const INVALID_INPUT: &str = "invalid input pointer or length";

struct MojoBatch {
    inner: BatchEnvironment,
    observations: Vec<f32>,
    positions: Vec<f32>,
    rewards: Vec<f32>,
    done: Vec<u8>,
    last_error: CString,
}

impl MojoBatch {
    fn new(
        lane_count: u32,
        step_frames: u32,
        grid_spacing: u32,
        parallel: bool,
    ) -> Result<Self, String> {
        if lane_count == 0 {
            return Err("lane count must be positive".to_owned());
        }
        let mut config = BatchConfig::new(step_frames);
        config.observations = ObservationFlags {
            full_state: false,
            pixels: false,
            board: false,
            include_offscreen_board: false,
            preserve_offscreen_coordinates: false,
            ml: true,
            ml_grid_spacing: grid_spacing,
        };
        config.execution = if parallel {
            ExecutionMode::Parallel
        } else {
            ExecutionMode::Serial
        };
        let inner = BatchEnvironment::new(config).map_err(|error| error.to_string())?;
        Ok(Self {
            inner,
            observations: Vec::with_capacity(lane_count as usize * ML_OBSERVATION_SIZE),
            positions: Vec::with_capacity(lane_count as usize * 2),
            rewards: Vec::with_capacity(lane_count as usize),
            done: Vec::with_capacity(lane_count as usize),
            last_error: CString::new("").expect("empty error string has no nul byte"),
        })
    }

    fn set_error(&mut self, error: impl Into<String>) {
        let message = error.into();
        self.last_error = CString::new(message)
            .unwrap_or_else(|_| CString::new("native error contained a nul byte").unwrap());
    }

    fn store(&mut self, values: Vec<MlBatchObservation>) {
        self.observations.clear();
        self.positions.clear();
        self.rewards.clear();
        self.done.clear();
        for value in values {
            self.observations.extend_from_slice(&value.ml_observation);
            self.positions.extend_from_slice(&value.player_position);
            self.rewards.push(value.reward as f32);
            self.done.push(u8::from(value.done));
        }
    }
}

unsafe fn batch_mut(handle: u64) -> Option<&'static mut MojoBatch> {
    if handle == 0 {
        None
    } else {
        unsafe { (handle as *mut MojoBatch).as_mut() }
    }
}

unsafe fn input_slice<'a, T>(pointer: *const T, length: usize) -> Option<&'a [T]> {
    if length == 0 {
        Some(&[])
    } else if pointer.is_null() {
        None
    } else {
        Some(unsafe { std::slice::from_raw_parts(pointer, length) })
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn mojo_batch_abi_version() -> u32 {
    ABI_VERSION
}

#[unsafe(no_mangle)]
pub extern "C" fn mojo_batch_observation_size() -> u32 {
    ML_OBSERVATION_SIZE as u32
}

#[unsafe(no_mangle)]
pub extern "C" fn mojo_batch_new(
    lane_count: u32,
    step_frames: u32,
    grid_spacing: u32,
    parallel: u8,
) -> u64 {
    match MojoBatch::new(lane_count, step_frames, grid_spacing, parallel != 0) {
        Ok(batch) => Box::into_raw(Box::new(batch)) as u64,
        Err(_) => 0,
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn mojo_batch_free(handle: u64) {
    if handle != 0 {
        unsafe {
            drop(Box::from_raw(handle as *mut MojoBatch));
        }
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn mojo_batch_lane_count(handle: u64) -> u32 {
    unsafe { batch_mut(handle) }
        .map(|batch| batch.inner.lane_count() as u32)
        .unwrap_or(0)
}

#[unsafe(no_mangle)]
pub extern "C" fn mojo_batch_reset(handle: u64, seeds: *const u32, lane_count: u32) -> i32 {
    let Some(batch) = (unsafe { batch_mut(handle) }) else {
        return -1;
    };
    let Some(seeds) = (unsafe { input_slice(seeds, lane_count as usize) }) else {
        batch.set_error(INVALID_INPUT);
        return -1;
    };
    match batch.inner.reset_ml_with_startup(seeds) {
        Ok(values) => {
            batch.store(values);
            0
        }
        Err(error) => {
            batch.set_error(error.to_string());
            -1
        }
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn mojo_batch_step(handle: u64, actions: *const u8, lane_count: u32) -> i32 {
    let Some(batch) = (unsafe { batch_mut(handle) }) else {
        return -1;
    };
    let Some(actions) = (unsafe { input_slice(actions, lane_count as usize) }) else {
        batch.set_error(INVALID_INPUT);
        return -1;
    };
    let mut native_actions = Vec::with_capacity(actions.len());
    for &index in actions {
        let Some(action) = Action::ALL.get(index as usize).copied() else {
            batch.set_error(format!("invalid native action index: {index}"));
            return -1;
        };
        native_actions.push(action);
    }
    match batch.inner.step_ml(&native_actions) {
        Ok(values) => {
            batch.store(values);
            0
        }
        Err(error) => {
            batch.set_error(error.to_string());
            -1
        }
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn mojo_batch_reset_lanes(
    handle: u64,
    lanes: *const u32,
    seeds: *const u32,
    count: u32,
) -> i32 {
    let Some(batch) = (unsafe { batch_mut(handle) }) else {
        return -1;
    };
    let Some(lanes) = (unsafe { input_slice(lanes, count as usize) }) else {
        batch.set_error(INVALID_INPUT);
        return -1;
    };
    let Some(seeds) = (unsafe { input_slice(seeds, count as usize) }) else {
        batch.set_error(INVALID_INPUT);
        return -1;
    };
    let lanes = lanes.iter().map(|lane| *lane as usize).collect::<Vec<_>>();
    match batch.inner.reset_ml_lanes_with_startup(&lanes, seeds) {
        Ok(values) => {
            batch.store(values);
            0
        }
        Err(error) => {
            batch.set_error(error.to_string());
            -1
        }
    }
}

fn copy_out<T: Copy>(source: &[T], destination: *mut T, count: u32) -> bool {
    let count = count as usize;
    if count != source.len() || destination.is_null() {
        return false;
    }
    unsafe {
        ptr::copy_nonoverlapping(source.as_ptr(), destination, count);
    }
    true
}

#[unsafe(no_mangle)]
pub extern "C" fn mojo_batch_copy_observations(
    handle: u64,
    destination: *mut f32,
    count: u32,
) -> i32 {
    let Some(batch) = (unsafe { batch_mut(handle) }) else {
        return -1;
    };
    if copy_out(&batch.observations, destination, count) {
        0
    } else {
        batch.set_error("observation output buffer has the wrong size");
        -1
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn mojo_batch_copy_positions(handle: u64, destination: *mut f32, count: u32) -> i32 {
    let Some(batch) = (unsafe { batch_mut(handle) }) else {
        return -1;
    };
    if copy_out(&batch.positions, destination, count) {
        0
    } else {
        batch.set_error("position output buffer has the wrong size");
        -1
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn mojo_batch_copy_rewards(handle: u64, destination: *mut f32, count: u32) -> i32 {
    let Some(batch) = (unsafe { batch_mut(handle) }) else {
        return -1;
    };
    if copy_out(&batch.rewards, destination, count) {
        0
    } else {
        batch.set_error("reward output buffer has the wrong size");
        -1
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn mojo_batch_copy_done(handle: u64, destination: *mut u8, count: u32) -> i32 {
    let Some(batch) = (unsafe { batch_mut(handle) }) else {
        return -1;
    };
    if copy_out(&batch.done, destination, count) {
        0
    } else {
        batch.set_error("done output buffer has the wrong size");
        -1
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn mojo_batch_copy_all(
    handle: u64,
    observations: *mut f32,
    observation_count: u32,
    positions: *mut f32,
    position_count: u32,
    rewards: *mut f32,
    reward_count: u32,
    done: *mut u8,
    done_count: u32,
) -> i32 {
    let Some(batch) = (unsafe { batch_mut(handle) }) else {
        return -1;
    };
    if !copy_out(&batch.observations, observations, observation_count) {
        batch.set_error("observation output buffer has the wrong size");
        return -1;
    }
    if !copy_out(&batch.positions, positions, position_count) {
        batch.set_error("position output buffer has the wrong size");
        return -1;
    }
    if !copy_out(&batch.rewards, rewards, reward_count) {
        batch.set_error("reward output buffer has the wrong size");
        return -1;
    }
    if !copy_out(&batch.done, done, done_count) {
        batch.set_error("done output buffer has the wrong size");
        return -1;
    }
    0
}

#[unsafe(no_mangle)]
pub extern "C" fn mojo_batch_last_error(handle: u64) -> *const c_char {
    unsafe { batch_mut(handle) }
        .map(|batch| batch.last_error.as_ptr())
        .unwrap_or(INVALID_HANDLE.as_ptr() as *const c_char)
}
