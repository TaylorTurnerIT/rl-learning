#![doc = "Engine-free deterministic core for the native Dodge runtime."]

mod action;
mod config;
mod error;
mod fixed;
mod game;
mod input;
mod lifecycle;
mod rng;
mod state;

pub use action::{Action, BUTTON_X_MASK};
pub use config::NativeConfig;
pub use error::CoreError;
pub use fixed::{PicoFixed, pico_ceil, pico_floor, pico_mid, pico_mod};
pub use game::{FrameResult, NativeGame};
pub use input::{BUTTON_MASK_LIMIT, Button, InputState};
pub use lifecycle::{LifecycleState, Mode};
pub use rng::{PicoRng, RngCheckpoint};
pub use state::{EnemyState, PlayerState};

/// Workspace-level version of native state contract.
pub const CORE_SCHEMA_VERSION: u32 = 1;
