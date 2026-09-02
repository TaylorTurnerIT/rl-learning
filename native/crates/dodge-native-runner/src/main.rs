use std::{
    env,
    fmt::Write as FmtWrite,
    fs,
    io::{self, Read, Write as IoWrite},
    path::{Path, PathBuf},
    process,
};

use dodge_core::{Action, BUTTON_X_MASK, Button, NativeConfig, NativeGame};
use serde::{Deserialize, Serialize};

const DEFAULT_SEED: u32 = 42;
const DEFAULT_MAX_FRAMES: u32 = 2_048;
const START_HOLD_FRAMES: usize = 13;

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct InputCommand {
    #[serde(rename = "move")]
    movement: String,
    duration_ms: u32,
}

#[derive(Debug, Serialize)]
struct Output {
    schema_version: u32,
    seed: u32,
    commands: Vec<OutputCommand>,
    frames: Vec<OutputFrame>,
    result: OutputResult,
}

#[derive(Debug, Serialize)]
struct OutputCommand {
    index: usize,
    #[serde(rename = "move")]
    movement: String,
    duration_ms: u32,
    duration_frames: u32,
    mask: u8,
}

#[derive(Debug, Serialize)]
struct OutputFrame {
    frame: u32,
    input_mask: u8,
    previous_input_mask: u8,
    mode: String,
    game_ready: bool,
    started: bool,
    dead: bool,
    done: bool,
    reward_raw: i32,
    events: Vec<String>,
    audio: Vec<OutputAudio>,
    state_hash: u64,
    pixel_hash: u64,
    snapshot_hex: String,
}

#[derive(Debug, Serialize)]
struct OutputAudio {
    kind: &'static str,
    id: u8,
    channel: Option<i8>,
}

#[derive(Debug, Serialize)]
struct OutputResult {
    frames: u32,
    done: bool,
}

struct Options {
    commands: String,
    seed: u32,
    output: Option<PathBuf>,
    max_frames: u32,
}

fn main() {
    if let Err(error) = run() {
        eprintln!("dodge-native-runner: {error}");
        process::exit(2);
    }
}

fn run() -> Result<(), String> {
    let options = parse_options(env::args().skip(1))?;
    let commands = load_commands(&options.commands)?;
    let output_commands = commands
        .iter()
        .enumerate()
        .map(|(index, command)| {
            let duration_frames = duration_to_frames(command.duration_ms);
            Ok(OutputCommand {
                index,
                movement: command.movement.clone(),
                duration_ms: command.duration_ms,
                duration_frames,
                mask: command.mask()?,
            })
        })
        .collect::<Result<Vec<_>, String>>()?;
    let schedule = expand_schedule(&commands)?;
    let mut game = NativeGame::new(NativeConfig::new(options.seed));
    let mut frames = Vec::new();
    for (simulation_mask, post_frame_mask) in schedule.into_iter().chain(std::iter::repeat((0, 0)))
    {
        if frames.len() >= options.max_frames as usize {
            break;
        }
        let result = game
            .advance_frame_with_post_mask(simulation_mask, post_frame_mask)
            .map_err(|error| format!("native frame failed: {error}"))?;
        frames.push(output_frame(&result));
        if result.done {
            break;
        }
    }
    let frame_count = frames.len() as u32;
    let done = frames.last().is_some_and(|frame| frame.done);
    let output = Output {
        schema_version: 1,
        seed: options.seed,
        commands: output_commands,
        frames,
        result: OutputResult {
            frames: frame_count,
            done,
        },
    };
    let mut bytes = Vec::new();
    serde_json::to_writer_pretty(&mut bytes, &output)
        .map_err(|error| format!("could not encode runner output: {error}"))?;
    bytes.push(b'\n');
    write_output(options.output.as_deref(), &bytes)
}

fn parse_options<I>(arguments: I) -> Result<Options, String>
where
    I: IntoIterator<Item = String>,
{
    let mut values = arguments.into_iter();
    let mut commands = None;
    let mut seed = DEFAULT_SEED;
    let mut output = None;
    let mut max_frames = DEFAULT_MAX_FRAMES;
    while let Some(argument) = values.next() {
        match argument.as_str() {
            "--commands" => commands = Some(next_value(&mut values, "--commands")?),
            "--seed" => seed = parse_u32(&next_value(&mut values, "--seed")?, "seed")?,
            "--output" => output = Some(PathBuf::from(next_value(&mut values, "--output")?)),
            "--max-frames" => {
                max_frames = parse_u32(&next_value(&mut values, "--max-frames")?, "max-frames")?;
                if max_frames == 0 {
                    return Err("--max-frames must be greater than zero".to_owned());
                }
            }
            "--help" | "-h" => return Err(usage()),
            value => return Err(format!("unknown argument {value}\n{}", usage())),
        }
    }
    let commands = commands.ok_or_else(usage)?;
    Ok(Options {
        commands,
        seed,
        output,
        max_frames,
    })
}

fn next_value<I>(values: &mut I, option: &str) -> Result<String, String>
where
    I: Iterator<Item = String>,
{
    values
        .next()
        .ok_or_else(|| format!("{option} requires a value"))
}

fn parse_u32(value: &str, name: &str) -> Result<u32, String> {
    value
        .parse::<u32>()
        .map_err(|error| format!("{name} must be an unsigned integer: {error}"))
}

fn usage() -> String {
    "usage: dodge-native-runner --commands PATH [--seed N] [--output PATH] [--max-frames N]"
        .to_owned()
}

fn load_commands(source: &str) -> Result<Vec<InputCommand>, String> {
    let mut input = String::new();
    if source == "-" {
        io::stdin()
            .read_to_string(&mut input)
            .map_err(|error| format!("could not read commands from stdin: {error}"))?;
    } else {
        input = fs::read_to_string(source)
            .map_err(|error| format!("could not read commands {}: {error}", source))?;
    }
    let commands = serde_json::from_str::<Vec<InputCommand>>(&input)
        .map_err(|error| format!("invalid command JSON: {error}"))?;
    validate_commands(&commands)?;
    Ok(commands)
}

fn validate_commands(commands: &[InputCommand]) -> Result<(), String> {
    if commands.is_empty() {
        return Err("commands must not be empty".to_owned());
    }
    if commands
        .first()
        .is_none_or(|command| command.movement != "x")
    {
        return Err("commands must start with an x move".to_owned());
    }
    for (index, command) in commands.iter().enumerate() {
        if !(1..=60_000).contains(&command.duration_ms) {
            return Err(format!(
                "command {index} duration_ms must be from 1 to 60000"
            ));
        }
        let _ = command
            .mask()
            .map_err(|error| format!("command {index}: {error}"))?;
    }
    Ok(())
}

fn duration_to_frames(duration_ms: u32) -> u32 {
    (duration_ms * 60).div_ceil(1_000)
}

fn expand_schedule(commands: &[InputCommand]) -> Result<Vec<(u8, u8)>, String> {
    let mut schedule = Vec::new();
    schedule.extend(std::iter::repeat_n(
        (BUTTON_X_MASK, BUTTON_X_MASK),
        START_HOLD_FRAMES,
    ));
    for command in commands.iter().skip(1) {
        let frames = duration_to_frames(command.duration_ms);
        let mask = command.mask()?;
        for frame in 0..frames {
            let post_frame_mask = if frame + 1 == frames { 0 } else { mask };
            schedule.push((mask, post_frame_mask));
        }
    }
    Ok(schedule)
}

impl InputCommand {
    fn mask(&self) -> Result<u8, String> {
        if self.movement == "x" {
            return Ok(BUTTON_X_MASK);
        }
        if self.movement == "o" {
            return Ok(Button::O.mask());
        }
        Action::from_name(&self.movement)
            .map(Action::mask)
            .map_err(|error| error.to_string())
    }
}

fn output_frame(result: &dodge_core::FrameResult) -> OutputFrame {
    let snapshot = &result.snapshot;
    OutputFrame {
        frame: result.frame,
        input_mask: result.input_mask,
        previous_input_mask: result.previous_input_mask,
        mode: mode_name(result.mode).to_owned(),
        game_ready: result.game_ready,
        started: result.started,
        dead: result.dead,
        done: result.done,
        reward_raw: result.reward.raw(),
        events: result
            .events
            .iter()
            .map(|event| event.name().to_owned())
            .collect(),
        audio: result
            .audio
            .iter()
            .map(|event| OutputAudio {
                kind: event.name(),
                id: event.id(),
                channel: event.channel(),
            })
            .collect(),
        state_hash: snapshot.state_hash(),
        pixel_hash: snapshot.pixel_hash(),
        snapshot_hex: hex_encode(&snapshot.canonical_bytes()),
    }
}

fn mode_name(mode: dodge_core::Mode) -> &'static str {
    match mode {
        dodge_core::Mode::Menu => "menu",
        dodge_core::Mode::TransitionToGame => "transition_to_game",
        dodge_core::Mode::Game => "game",
        dodge_core::Mode::Terminal => "terminal",
        dodge_core::Mode::Settings => "settings",
        dodge_core::Mode::TransitionToSettings => "transition_to_settings",
        dodge_core::Mode::TransitionToMenu => "transition_to_menu",
    }
}

fn hex_encode(bytes: &[u8]) -> String {
    let mut output = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        let _ = write!(&mut output, "{byte:02x}");
    }
    output
}

fn write_output(path: Option<&Path>, bytes: &[u8]) -> Result<(), String> {
    match path {
        Some(path) => fs::write(path, bytes)
            .map_err(|error| format!("could not write output {}: {error}", path.display())),
        None => {
            let mut stdout = io::stdout();
            stdout
                .write_all(bytes)
                .map_err(|error| format!("could not write stdout: {error}"))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{
        InputCommand, duration_to_frames, expand_schedule, load_commands, validate_commands,
    };
    use dodge_core::{BUTTON_X_MASK, NativeConfig, NativeGame};

    #[test]
    fn p1_duration_rounding_and_boundary_masks_are_deterministic() {
        assert_eq!(duration_to_frames(50), 3);
        assert_eq!(duration_to_frames(100), 6);
        let commands = vec![
            InputCommand {
                movement: "x".to_owned(),
                duration_ms: 50,
            },
            InputCommand {
                movement: "left".to_owned(),
                duration_ms: 100,
            },
        ];
        let schedule = expand_schedule(&commands);
        assert!(schedule.is_ok());
        assert_eq!(
            schedule.unwrap_or_default(),
            [(BUTTON_X_MASK, BUTTON_X_MASK); 13]
                .into_iter()
                .chain([(1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 0)])
                .collect::<Vec<_>>()
        );
    }

    #[test]
    fn invalid_commands_are_rejected_before_game_creation() {
        let commands = vec![InputCommand {
            movement: "left".to_owned(),
            duration_ms: 100,
        }];
        assert!(validate_commands(&commands).is_err());
    }

    #[test]
    fn runner_schedule_advances_core_without_emulator_process() {
        let commands = vec![InputCommand {
            movement: "x".to_owned(),
            duration_ms: 50,
        }];
        let schedule = expand_schedule(&commands);
        assert!(schedule.is_ok());
        let mut game = NativeGame::new(NativeConfig::new(42));
        for (simulation_mask, post_frame_mask) in schedule.unwrap_or_default() {
            assert!(
                game.advance_frame_with_post_mask(simulation_mask, post_frame_mask)
                    .is_ok()
            );
        }
        assert_eq!(game.lifecycle().frame, 13);
    }

    #[test]
    fn command_loader_requires_exact_p1_shape() {
        let path = std::env::temp_dir().join("dodge-native-runner-invalid.json");
        let write_result = std::fs::write(&path, r#"[{"move":"x","duration_ms":50,"extra":1}]"#);
        assert!(write_result.is_ok());
        assert!(load_commands(path.to_string_lossy().as_ref()).is_err());
        let _ = std::fs::remove_file(path);
    }
}
