# RL Learning

A from-first-principles reinforcement learning and evolutionary algorithm sandbox. The goal is to understand
how agents, environments, states, actions, rewards, policies, and returns fit
together by implementing them directly in Python without an ML or RL library.

## Run

Enter the reproducible development environment with `direnv allow` or
`devenv shell`, then run:

```bash
uv run rl-learning
```

Useful checks:

```bash
ruff format --check .
ruff check .
```

## Dodge (PICO-8)

The complete cartridge extracted from the Linux export lives at
`src/dodge/game/dodge.p8`. The ignored `src/dodge/runtime/` directory contains
the prebuilt Pemsa Linux runner, so no emulator compilation is required.

Launch the cartridge with:

```bash
just dodge-run
```

You can also run it directly through the development environment with
`devenv shell -- dodge-run`.

### Programmatic controls

Pass `dodge-control` a JSON list of timed movements:

```json
[
  {"move": "x", "duration_ms": 50},
  {"move": "neutral", "duration_ms": 750},
  {"move": "left", "duration_ms": 250},
  {"move": "up_right", "duration_ms": 400},
  {"move": "neutral", "duration_ms": 100},
  {"move": "down", "duration_ms": 300}
]
```

Each list must start with `x` to pass the main menu. Subsequent movements may
be `neutral`, `left`, `right`, `up`, `down`, `up_left`, `up_right`, `down_left`,
or `down_right`. Durations are integer milliseconds from 1 through 60000. The
complete list is validated before the game launches.

Run a file with:

```bash
just dodge-control src/dodge/game/movements.json
```

The control runner defaults PICO-8's random seed to 42. Override it with:

```bash
just dodge-control src/dodge/game/movements.json --seed 7
```

Valid seeds are integers from 0 through 32767. Seeded runs use a temporary
cartridge copy; `src/dodge/game/dodge.p8` is never modified.

Or pipe the JSON list through stdin:

```bash
printf '[{"move":"left","duration_ms":250}]' | just dodge-control -
```

The controller launches its own game process, executes the listed controls
through targeted keyboard events, and closes that process afterward.

### Headless runs

Run the same command file without a display or audio device and receive the
final game score as JSON:

```bash
just dodge-headless src/dodge/game/movements.json
```

Example output:

```json
{"score": 0, "frames": 159, "seed": 42, "started": true}
```

Headless durations are rounded up to whole 60 Hz game frames. Each invocation
uses an isolated temporary cartridge, working directory, and `.cartdata`, so
multiple processes can run concurrently without sharing save state. Pemsa still
advances at real-time 60 Hz; headless mode removes graphical and audio overhead
but does not accelerate its clock.

For batches, enter `devenv shell` once and invoke `dodge-headless` directly from
worker processes instead of starting a new devenv shell through `just` per run.

### Direct PPO training

The direct learner trains a convolutional actor-critic against the step-wise
Dodge environment. It optimizes survived frames directly and uses only a small
capped preference for neutral actions; it does not depend on behavior-cloning
weights. Checkpoints and metrics are written under the ignored history tree:

```bash
just dodge-ppo-train --run-dir history/dodge/ppo/production --updates 1000
```

Resume an interrupted run by increasing the target update count:

```bash
just dodge-ppo-train --resume --run-dir history/dodge/ppo/production --updates 2000
```

### DQN dashboard

Start the dashboard in another terminal:

```bash
just dodge-ng-dashboard
```

It selects the latest run; pass `--run-dir` to choose one explicitly. Open the
printed URL for live progress, controls, and checkpoint replays. Telemetry is
best-effort and never waits on the dashboard.
