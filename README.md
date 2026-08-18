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
  {"move": "left", "duration_ms": 250},
  {"move": "up_right", "duration_ms": 400},
  {"move": "neutral", "duration_ms": 100},
  {"move": "down", "duration_ms": 300}
]
```

Each movement may be `neutral`, `left`, `right`, `up`, `down`, `up_left`,
`up_right`, `down_left`, or `down_right`. Durations are integer milliseconds
from 1 through 60000. The complete list is validated before the game launches.

Run a file with:

```bash
just dodge-control movements.json
```

The control runner defaults PICO-8's random seed to 42. Override it with:

```bash
just dodge-control movements.json --seed 42
```

Valid seeds are integers from 0 through 32767. Seeded runs use a temporary
cartridge copy; `src/dodge/game/dodge.p8` is never modified.

Or pipe the JSON list through stdin:

```bash
printf '[{"move":"left","duration_ms":250}]' | just dodge-control -
```

The controller launches its own game process, presses X to start, executes the
movements through targeted keyboard events, and closes that process afterward.
