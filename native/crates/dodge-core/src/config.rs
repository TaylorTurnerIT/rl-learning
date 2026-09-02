/// Configuration shared by reset and future native runner boundaries.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct NativeConfig {
    pub seed: u32,
}

impl NativeConfig {
    pub const fn new(seed: u32) -> Self {
        Self { seed }
    }
}

impl Default for NativeConfig {
    fn default() -> Self {
        Self::new(42)
    }
}
