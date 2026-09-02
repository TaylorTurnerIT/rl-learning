use std::{
    error::Error,
    ffi::OsStr,
    fmt::{Display, Formatter, Write as FmtWrite},
    fs,
    io::Write,
    path::Path,
};

use dodge_core::{CoreError, FRAMEBUFFER_HEIGHT, FRAMEBUFFER_SIZE, FRAMEBUFFER_WIDTH, Snapshot};
use serde::Deserialize;

pub const PICO8_PALETTE: [[u8; 3]; 16] = [
    [0, 0, 0],
    [29, 43, 83],
    [126, 37, 83],
    [0, 135, 81],
    [171, 82, 54],
    [95, 87, 79],
    [194, 195, 199],
    [255, 241, 232],
    [255, 0, 77],
    [255, 163, 0],
    [255, 236, 39],
    [0, 228, 54],
    [41, 173, 255],
    [131, 118, 156],
    [255, 119, 168],
    [255, 204, 170],
];

#[derive(Debug)]
pub enum ViewerError {
    Io(std::io::Error),
    Json(serde_json::Error),
    Core(CoreError),
    InvalidTrace(String),
    InvalidPixel(u8),
    UnsupportedCaptureFormat(String),
}

impl Display for ViewerError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Io(error) => write!(formatter, "viewer I/O failed: {error}"),
            Self::Json(error) => write!(formatter, "viewer trace JSON failed: {error}"),
            Self::Core(error) => write!(formatter, "native snapshot failed: {error}"),
            Self::InvalidTrace(message) => write!(formatter, "invalid viewer trace: {message}"),
            Self::InvalidPixel(pixel) => write!(formatter, "invalid palette index: {pixel}"),
            Self::UnsupportedCaptureFormat(extension) => {
                write!(formatter, "unsupported capture extension: {extension}")
            }
        }
    }
}

impl Error for ViewerError {}

impl From<std::io::Error> for ViewerError {
    fn from(error: std::io::Error) -> Self {
        Self::Io(error)
    }
}

impl From<serde_json::Error> for ViewerError {
    fn from(error: serde_json::Error) -> Self {
        Self::Json(error)
    }
}

impl From<CoreError> for ViewerError {
    fn from(error: CoreError) -> Self {
        Self::Core(error)
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Viewport {
    pub x: f32,
    pub y: f32,
    pub size: f32,
    pub scale: u32,
}

pub fn integer_viewport(window_width: f32, window_height: f32) -> Option<Viewport> {
    if !window_width.is_finite()
        || !window_height.is_finite()
        || window_width <= 0.0
        || window_height <= 0.0
    {
        return None;
    }
    let scale = (window_width.min(window_height) / FRAMEBUFFER_WIDTH as f32)
        .floor()
        .max(1.0) as u32;
    let size = scale as f32 * FRAMEBUFFER_WIDTH as f32;
    Some(Viewport {
        x: (window_width - size) / 2.0,
        y: (window_height - size) / 2.0,
        size,
        scale,
    })
}

pub fn indexed_to_rgba(
    pixels: &[u8; FRAMEBUFFER_SIZE],
    palette: &[[u8; 3]; 16],
) -> Result<Vec<u8>, ViewerError> {
    let mut rgba = Vec::with_capacity(FRAMEBUFFER_SIZE * 4);
    for pixel in pixels {
        let Some(rgb) = palette.get(usize::from(*pixel)) else {
            return Err(ViewerError::InvalidPixel(*pixel));
        };
        let [red, green, blue] = *rgb;
        rgba.extend_from_slice(&[red, green, blue, 255]);
    }
    Ok(rgba)
}

#[derive(Clone, Debug)]
pub struct PresentedFrame {
    snapshot: Snapshot,
    rgba: Vec<u8>,
}

impl PresentedFrame {
    pub fn from_snapshot(snapshot: Snapshot) -> Result<Self, ViewerError> {
        let rgba = indexed_to_rgba(snapshot.pixels(), &PICO8_PALETTE)?;
        Ok(Self { snapshot, rgba })
    }

    pub fn snapshot(&self) -> &Snapshot {
        &self.snapshot
    }

    pub fn pixels(&self) -> &[u8; FRAMEBUFFER_SIZE] {
        self.snapshot.pixels()
    }

    pub fn rgba(&self) -> &[u8] {
        &self.rgba
    }

    pub fn frame(&self) -> u32 {
        self.snapshot.logical_state().lifecycle.frame
    }

    pub fn state_hash(&self) -> u64 {
        self.snapshot.state_hash()
    }

    pub fn pixel_hash(&self) -> u64 {
        self.snapshot.pixel_hash()
    }

    pub fn source_hash_hex(&self) -> String {
        bytes_to_hex(&self.snapshot.provenance().cartridge_sha256)
    }
}

pub fn decode_snapshot_hex(value: &str) -> Result<Snapshot, ViewerError> {
    let bytes = decode_hex(value.trim())?;
    Snapshot::from_canonical_bytes(&bytes).map_err(ViewerError::from)
}

#[derive(Debug, Deserialize)]
pub struct TraceRecord {
    pub frame: u32,
    pub state_hash: u64,
    pub pixel_hash: u64,
    pub snapshot_hex: String,
}

#[derive(Debug, Deserialize)]
pub struct ReplayTrace {
    pub schema_version: u32,
    pub seed: u32,
    pub frames: Vec<TraceRecord>,
}

pub fn load_trace(path: &Path) -> Result<ReplayTrace, ViewerError> {
    let bytes = fs::read_to_string(path)?;
    let trace = serde_json::from_str::<ReplayTrace>(&bytes)?;
    if trace.schema_version != 1 {
        return Err(ViewerError::InvalidTrace(format!(
            "unsupported schema version {}",
            trace.schema_version
        )));
    }
    if trace.frames.is_empty() {
        return Err(ViewerError::InvalidTrace(
            "trace contains no frames".to_owned(),
        ));
    }
    Ok(trace)
}

pub fn present_trace_frame(
    trace: &ReplayTrace,
    frame_index: usize,
) -> Result<PresentedFrame, ViewerError> {
    let Some(record) = trace.frames.get(frame_index) else {
        return Err(ViewerError::InvalidTrace(format!(
            "frame index {frame_index} is outside 0..{}",
            trace.frames.len()
        )));
    };
    let snapshot = decode_snapshot_hex(&record.snapshot_hex)?;
    if snapshot.logical_state().lifecycle.frame != record.frame {
        return Err(ViewerError::InvalidTrace(format!(
            "trace frame {} carries snapshot frame {}",
            record.frame,
            snapshot.logical_state().lifecycle.frame
        )));
    }
    if snapshot.state_hash() != record.state_hash {
        return Err(ViewerError::InvalidTrace(format!(
            "state hash mismatch at frame {}",
            record.frame
        )));
    }
    if snapshot.pixel_hash() != record.pixel_hash {
        return Err(ViewerError::InvalidTrace(format!(
            "pixel hash mismatch at frame {}",
            record.frame
        )));
    }
    PresentedFrame::from_snapshot(snapshot)
}

pub fn write_capture(path: &Path, frame: &PresentedFrame) -> Result<(), ViewerError> {
    match path.extension().and_then(OsStr::to_str) {
        Some(extension) if extension.eq_ignore_ascii_case("ppm") => write_rgba_ppm(path, frame),
        Some(extension) if extension.eq_ignore_ascii_case("pgm") => write_indexed_pgm(path, frame),
        Some(extension) => Err(ViewerError::UnsupportedCaptureFormat(extension.to_owned())),
        None => write_indexed_pgm(path, frame),
    }
}

pub fn write_indexed_pgm(path: &Path, frame: &PresentedFrame) -> Result<(), ViewerError> {
    let header = format!(
        "P5\n# source_sha256={}\n# frame={}\n# state_hash={:016x}\n# pixel_hash={:016x}\n{} {}\n15\n",
        frame.source_hash_hex(),
        frame.frame(),
        frame.state_hash(),
        frame.pixel_hash(),
        FRAMEBUFFER_WIDTH,
        FRAMEBUFFER_HEIGHT
    );
    let mut file = fs::File::create(path)?;
    file.write_all(header.as_bytes())?;
    file.write_all(frame.pixels().as_slice())?;
    Ok(())
}

fn write_rgba_ppm(path: &Path, frame: &PresentedFrame) -> Result<(), ViewerError> {
    let header = format!(
        "P6\n# source_sha256={}\n# frame={}\n# state_hash={:016x}\n# pixel_hash={:016x}\n{} {}\n255\n",
        frame.source_hash_hex(),
        frame.frame(),
        frame.state_hash(),
        frame.pixel_hash(),
        FRAMEBUFFER_WIDTH,
        FRAMEBUFFER_HEIGHT
    );
    let mut file = fs::File::create(path)?;
    file.write_all(header.as_bytes())?;
    for chunk in frame.rgba().chunks_exact(4) {
        let Ok(rgba) = <&[u8; 4]>::try_from(chunk) else {
            return Err(ViewerError::InvalidTrace(
                "RGBA presentation buffer is not four-byte aligned".to_owned(),
            ));
        };
        let [red, green, blue, _alpha] = *rgba;
        file.write_all(&[red, green, blue])?;
    }
    Ok(())
}

fn decode_hex(value: &str) -> Result<Vec<u8>, ViewerError> {
    if !value.len().is_multiple_of(2) {
        return Err(ViewerError::InvalidTrace(
            "snapshot hex has an odd number of digits".to_owned(),
        ));
    }
    let mut bytes = Vec::with_capacity(value.len() / 2);
    let mut digits = value.as_bytes().chunks_exact(2);
    for pair in &mut digits {
        let Some(high) = pair.first().copied().and_then(hex_digit) else {
            return Err(ViewerError::InvalidTrace(
                "snapshot hex has invalid digits".to_owned(),
            ));
        };
        let Some(low) = pair.get(1).copied().and_then(hex_digit) else {
            return Err(ViewerError::InvalidTrace(
                "snapshot hex has invalid digits".to_owned(),
            ));
        };
        bytes.push((high << 4) | low);
    }
    Ok(bytes)
}

fn hex_digit(value: u8) -> Option<u8> {
    match value {
        b'0'..=b'9' => Some(value - b'0'),
        b'a'..=b'f' => Some(value - b'a' + 10),
        b'A'..=b'F' => Some(value - b'A' + 10),
        _ => None,
    }
}

fn bytes_to_hex(bytes: &[u8]) -> String {
    let mut output = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        let _ = write!(output, "{byte:02x}");
    }
    output
}

#[cfg(test)]
mod tests {
    use super::{
        FRAMEBUFFER_SIZE, PICO8_PALETTE, PresentedFrame, ReplayTrace, TraceRecord,
        decode_snapshot_hex, indexed_to_rgba, integer_viewport, present_trace_frame, write_capture,
    };
    use dodge_core::{BUTTON_X_MASK, NativeConfig, NativeGame};

    fn sample_frame() -> PresentedFrame {
        let mut game = NativeGame::new(NativeConfig::new(42));
        let result = game.advance_frame(BUTTON_X_MASK);
        assert!(result.is_ok());
        let snapshot = match result {
            Ok(value) => value.snapshot,
            Err(_) => {
                return PresentedFrame::from_snapshot(game.snapshot())
                    .unwrap_or_else(|_| unreachable!("native reset snapshot must be present"));
            }
        };
        PresentedFrame::from_snapshot(snapshot)
            .unwrap_or_else(|_| unreachable!("native snapshot palette must be valid"))
    }

    #[test]
    fn viewport_uses_integer_scale_and_centers_square() {
        assert_eq!(
            integer_viewport(1280.0, 720.0),
            Some(super::Viewport {
                x: 320.0,
                y: 40.0,
                size: 640.0,
                scale: 5,
            })
        );
        assert_eq!(integer_viewport(0.0, 720.0), None);
        assert_eq!(
            integer_viewport(127.0, 127.0).map(|value| value.scale),
            Some(1)
        );
    }

    #[test]
    fn palette_conversion_preserves_index_order_and_alpha() {
        let pixels = [0_u8; FRAMEBUFFER_SIZE];
        let rgba = indexed_to_rgba(&pixels, &PICO8_PALETTE);
        assert!(rgba.is_ok());
        let rgba = match rgba {
            Ok(value) => value,
            Err(_) => return,
        };
        assert_eq!(rgba.len(), FRAMEBUFFER_SIZE * 4);
        assert_eq!(rgba.first().copied(), Some(0));
        assert_eq!(rgba.get(3).copied(), Some(255));
    }

    #[test]
    fn repeated_presentation_is_byte_identical() {
        let first = sample_frame();
        let second = PresentedFrame::from_snapshot(first.snapshot().clone());
        assert!(second.is_ok());
        let second = match second {
            Ok(value) => value,
            Err(_) => return,
        };
        assert_eq!(first.pixels(), second.pixels());
        assert_eq!(first.rgba(), second.rgba());
        assert_eq!(first.state_hash(), second.state_hash());
        assert_eq!(first.pixel_hash(), second.pixel_hash());
    }

    #[test]
    fn canonical_snapshot_hex_roundtrips_at_viewer_boundary() {
        let frame = sample_frame();
        let hex = frame
            .snapshot()
            .canonical_bytes()
            .iter()
            .map(|byte| format!("{byte:02x}"))
            .collect::<String>();
        let restored = decode_snapshot_hex(&hex);
        assert!(restored.is_ok());
        let restored = match restored {
            Ok(value) => value,
            Err(_) => return,
        };
        assert_eq!(
            restored.canonical_bytes(),
            frame.snapshot().canonical_bytes()
        );
    }

    #[test]
    fn trace_replay_validates_frame_and_hash_metadata() {
        let frame = sample_frame();
        let snapshot_hex = frame
            .snapshot()
            .canonical_bytes()
            .iter()
            .map(|byte| format!("{byte:02x}"))
            .collect::<String>();
        let trace = ReplayTrace {
            schema_version: 1,
            seed: 42,
            frames: vec![TraceRecord {
                frame: frame.frame(),
                state_hash: frame.state_hash(),
                pixel_hash: frame.pixel_hash(),
                snapshot_hex,
            }],
        };
        let replayed = present_trace_frame(&trace, 0);
        assert!(replayed.is_ok());
        let replayed = match replayed {
            Ok(value) => value,
            Err(_) => return,
        };
        assert_eq!(replayed.rgba(), frame.rgba());
    }

    #[test]
    fn indexed_capture_is_lossless_and_has_native_dimensions() {
        let frame = sample_frame();
        let path = std::env::temp_dir().join(format!("dodge-viewer-{}.pgm", std::process::id()));
        let result = write_capture(&path, &frame);
        assert!(result.is_ok());
        let bytes = std::fs::read(&path);
        assert!(bytes.is_ok());
        let bytes = match bytes {
            Ok(value) => value,
            Err(_) => return,
        };
        assert!(bytes.starts_with(b"P5\n"));
        assert!(bytes.windows(8).any(|window| window == b"128 128\n"));
        assert!(bytes.len() > FRAMEBUFFER_SIZE);
        let _ = std::fs::remove_file(path);
    }
}
