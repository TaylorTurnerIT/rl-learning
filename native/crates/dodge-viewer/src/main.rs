use std::{env, path::PathBuf, process};

use dodge_core::{BUTTON_X_MASK, Button, NativeConfig, NativeGame};
use dodge_viewer::{
    PresentedFrame, ViewerError, integer_viewport, load_trace, present_trace_frame, write_capture,
};
use macroquad::{
    color::{BLACK, WHITE},
    prelude::{
        Conf, DrawTextureParams, FilterMode, KeyCode, Texture2D, clear_background, draw_text,
        draw_texture_ex, get_frame_time, is_key_down, is_key_pressed, next_frame, screen_height,
        screen_width, vec2,
    },
};

const DEFAULT_WINDOW_SIZE: i32 = 768;
const FRAME_CADENCE_SECONDS: f32 = 1.0 / 60.0;

struct Options {
    trace: Option<PathBuf>,
    frame: usize,
    capture: Option<PathBuf>,
    debug: bool,
    pause: bool,
    seed: u32,
}

#[macroquad::main(window_conf)]
async fn main() {
    if let Err(error) = run().await {
        eprintln!("dodge-viewer: {error}");
        process::exit(2);
    }
}

fn window_conf() -> Conf {
    Conf {
        window_title: "Dodge Native Viewer".to_owned(),
        window_width: DEFAULT_WINDOW_SIZE,
        window_height: DEFAULT_WINDOW_SIZE,
        high_dpi: true,
        sample_count: 0,
        ..Default::default()
    }
}

async fn run() -> Result<(), ViewerError> {
    let options = parse_options(env::args().skip(1))?;
    let trace = options.trace.as_deref().map(load_trace).transpose()?;
    let mut live_game = trace
        .is_none()
        .then(|| NativeGame::new(NativeConfig::new(options.seed)));
    let initial_frame = if let Some(trace) = trace.as_ref() {
        present_trace_frame(trace, options.frame)?
    } else {
        let Some(game) = live_game.as_mut() else {
            return Err(ViewerError::InvalidTrace(
                "live viewer game was not initialized".to_owned(),
            ));
        };
        let result = game
            .advance_frame(BUTTON_X_MASK)
            .map_err(|error| ViewerError::InvalidTrace(error.to_string()))?;
        PresentedFrame::from_snapshot(result.snapshot)?
    };
    if let Some(path) = options.capture.as_deref() {
        write_capture(path, &initial_frame)?;
    }

    let mut frame_index = options.frame;
    let mut elapsed = 0.0_f32;
    loop {
        let frame = if let Some(trace) = trace.as_ref() {
            present_trace_frame(trace, frame_index.min(trace.frames.len() - 1))?
        } else {
            let Some(game) = live_game.as_mut() else {
                return Err(ViewerError::InvalidTrace(
                    "live viewer game was not initialized".to_owned(),
                ));
            };
            if options.pause {
                initial_frame.clone()
            } else {
                let result = game
                    .advance_frame(keyboard_mask())
                    .map_err(|error| ViewerError::InvalidTrace(error.to_string()))?;
                PresentedFrame::from_snapshot(result.snapshot)?
            }
        };
        let rgba_len = frame.rgba().len();
        let texture = Texture2D::from_rgba8(
            dodge_core::FRAMEBUFFER_WIDTH as u16,
            dodge_core::FRAMEBUFFER_HEIGHT as u16,
            frame.rgba(),
        );
        texture.set_filter(FilterMode::Nearest);
        clear_background(BLACK);
        if let Some(viewport) = integer_viewport(screen_width(), screen_height()) {
            draw_texture_ex(
                &texture,
                viewport.x,
                viewport.y,
                WHITE,
                DrawTextureParams {
                    dest_size: Some(vec2(viewport.size, viewport.size)),
                    ..Default::default()
                },
            );
        }
        if options.debug {
            let text = format!(
                "frame={} seed={} state={:016x} pixels={:016x} rgba={}",
                frame.frame(),
                trace.as_ref().map_or(options.seed, |value| value.seed),
                frame.state_hash(),
                frame.pixel_hash(),
                rgba_len
            );
            draw_text(&text, 8.0, 20.0, 18.0, WHITE);
        }
        if is_key_pressed(KeyCode::Escape) {
            break;
        }
        if let Some(trace) = trace.as_ref()
            && !options.pause
        {
            elapsed += get_frame_time();
            while elapsed >= FRAME_CADENCE_SECONDS {
                elapsed -= FRAME_CADENCE_SECONDS;
                if frame_index + 1 < trace.frames.len() {
                    frame_index += 1;
                }
            }
        }
        next_frame().await;
    }
    Ok(())
}

fn keyboard_mask() -> u8 {
    let mut mask = 0_u8;
    if is_key_down(KeyCode::Left) {
        mask |= Button::Left.mask();
    }
    if is_key_down(KeyCode::Right) {
        mask |= Button::Right.mask();
    }
    if is_key_down(KeyCode::Up) {
        mask |= Button::Up.mask();
    }
    if is_key_down(KeyCode::Down) {
        mask |= Button::Down.mask();
    }
    if is_key_down(KeyCode::X) || is_key_down(KeyCode::Space) {
        mask |= Button::X.mask();
    }
    if is_key_down(KeyCode::Z) {
        mask |= Button::O.mask();
    }
    mask
}

fn parse_options<I>(arguments: I) -> Result<Options, ViewerError>
where
    I: IntoIterator<Item = String>,
{
    let mut values = arguments.into_iter();
    let mut trace = None;
    let mut frame = 0_usize;
    let mut capture = None;
    let mut debug = false;
    let mut pause = false;
    let mut seed = 42_u32;
    while let Some(argument) = values.next() {
        match argument.as_str() {
            "--trace" => trace = Some(PathBuf::from(next_value(&mut values, "--trace")?)),
            "--frame" => {
                frame = next_value(&mut values, "--frame")?
                    .parse::<usize>()
                    .map_err(|error| {
                        ViewerError::InvalidTrace(format!("invalid frame: {error}"))
                    })?;
            }
            "--capture" => capture = Some(PathBuf::from(next_value(&mut values, "--capture")?)),
            "--debug" => debug = true,
            "--pause" => pause = true,
            "--seed" => {
                seed = next_value(&mut values, "--seed")?
                    .parse::<u32>()
                    .map_err(|error| ViewerError::InvalidTrace(format!("invalid seed: {error}")))?;
            }
            "--help" | "-h" => return Err(ViewerError::InvalidTrace(usage().to_owned())),
            value => {
                return Err(ViewerError::InvalidTrace(format!(
                    "unknown argument {value}\n{}",
                    usage()
                )));
            }
        }
    }
    Ok(Options {
        trace,
        frame,
        capture,
        debug,
        pause,
        seed,
    })
}

fn next_value<I>(values: &mut I, option: &str) -> Result<String, ViewerError>
where
    I: Iterator<Item = String>,
{
    values
        .next()
        .ok_or_else(|| ViewerError::InvalidTrace(format!("{option} requires a value\n{}", usage())))
}

const fn usage() -> &'static str {
    "usage: dodge-viewer [--trace PATH] [--frame N] [--capture PATH] [--debug] [--pause] [--seed N]"
}
