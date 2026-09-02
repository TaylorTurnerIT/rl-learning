#![doc = "Engine-free deterministic core for the native Dodge runtime."]

/// Workspace-level version of the native state contract.
pub const CORE_SCHEMA_VERSION: u32 = 1;

/// Placeholder root for the P3 typed simulation API.
#[derive(Debug, Default)]
pub struct NativeGame;
