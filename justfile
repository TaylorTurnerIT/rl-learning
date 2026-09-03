# Project task runner. Recipes enter `devenv` so they work from a plain shell.

[private]
default:
    @just --list

# Reinforcement-learning sandbox
[group("rl_learning")]
rl-run:
    devenv shell -- app-run

[group("rl_learning")]
rl-test:
    devenv shell -- app-test

[group("rl_learning")]
rl-check:
    devenv shell -- app-check

[group("rl_learning")]
rl-format:
    devenv shell -- app-fmt

# PICO-8 Dodge cartridge
[group("dodge")]
dodge-run:
    devenv shell -- dodge-run

[group("dodge")]
dodge-control commands *options:
    devenv shell -- dodge-control {{ commands }} {{ options }}

[group("dodge")]
dodge-headless commands *options:
    @DODGE_HEADLESS=1 devenv -q shell -- dodge-headless {{ commands }} {{ options }}

[group("dodge")]
dodge-native-oracle commands output *options:
    @DODGE_HEADLESS=1 devenv -q shell -- dodge-native-oracle --commands {{ commands }} --output {{ output }} {{ options }}

[group("dodge")]
dodge-native-probe *options:
    @DODGE_HEADLESS=1 devenv -q shell -- dodge-native-probe {{ options }}

[group("dodge")]
dodge-native-test:
    @DODGE_HEADLESS=1 devenv -q shell -- cargo nextest run --locked --manifest-path native/Cargo.toml --workspace

[group("dodge")]
dodge-native-check:
    @DODGE_HEADLESS=1 devenv -q shell -- cargo fmt --manifest-path native/Cargo.toml --all -- --check
    @DODGE_HEADLESS=1 devenv -q shell -- cargo clippy --locked --manifest-path native/Cargo.toml --workspace --all-targets --all-features -- -D warnings
    @DODGE_HEADLESS=1 devenv -q shell -- cargo nextest run --locked --manifest-path native/Cargo.toml --workspace

[group("dodge")]
dodge-native-kani:
    @DODGE_HEADLESS=1 devenv -q shell -- cargo kani --manifest-path native/Cargo.toml -p dodge-kani

[group("dodge")]
dodge-native-bench *options:
    @DODGE_HEADLESS=1 devenv -q shell -- cargo bench --locked --manifest-path native/Cargo.toml -p dodge-batch --bench batch -- {{ options }}

[group("dodge")]
dodge-native-fuzz *options:
    @DODGE_HEADLESS=1 devenv -q shell -- dodge-native-fuzz {{ options }}

[group("dodge")]
dodge-native-extract-assets source output *options:
    devenv -q shell -- dodge-native-extract-assets --source {{ source }} --output {{ output }} {{ options }}

[group("dodge")]
dodge-native-p2-report assets output *options:
    @DODGE_HEADLESS=1 devenv -q shell -- dodge-native-p2-report --assets {{ assets }} --output {{ output }} {{ options }}

[group("dodge")]
dodge-dataset-collect *options:
    @DODGE_HEADLESS=1 devenv -q shell -- uv run dodge-dataset-collect {{ options }}

[group("dodge")]
dodge-dataset-replay database seed:
    devenv shell -- uv run dodge-dataset-replay {{ database }} --seed {{ seed }}

[group("dodge")]
dodge-dataset-reconstruct *options:
    @DODGE_HEADLESS=1 devenv -q shell -- uv run dodge-dataset-reconstruct {{ options }}

[group("dodge")]
dodge-dataset-reset *options:
    @DODGE_HEADLESS=1 devenv -q shell -- uv run dodge-dataset-reset --yes {{ options }}

[group("dodge")]
dodge-dataset-export output *options:
    devenv -q shell -- uv run dodge-dataset-export {{ output }} {{ options }}

[group("dodge")]
dodge-bc-train *options:
    devenv -q shell -- uv run dodge-bc-train {{ options }}

[group("dodge")]
dodge-bc-plot history *options:
    devenv -q shell -- uv run dodge-bc-plot {{ history }} {{ options }}

[group("dodge")]
dodge-ppo-train *options:
    @mkdir -p history/dodge/ppo/.launch-tmp
    @DODGE_HEADLESS=1 TMPDIR="$PWD/history/dodge/ppo/.launch-tmp" devenv -q shell -- dodge-ppo-train {{ options }}

[group("dodge")]
dodge-marimo:
    devenv -q shell -- uv run --group gpu marimo edit notebooks/dodge_behavior_cloning.py

[group("dodge")]
dodge-replay history:
    devenv shell -- dodge-replay {{ history }}

[group("dodge")]
dodge-replay-run history:
    devenv shell -- dodge-replay-run {{ history }}

[group("dodge")]
dodge-replay-latest epoch:
    devenv shell -- dodge-replay-latest {{ epoch }}

[group("dodge")]
dodge-neat-train *options:
    @DODGE_HEADLESS=1 devenv -q shell -- dodge-neat-train {{ options }}

[group("dodge")]
dodge-neat-resume run *options:
    @DODGE_HEADLESS=1 devenv -q shell -- dodge-neat-train --resume {{ run }} {{ options }}

[group("dodge")]
dodge-neat-replay episode:
    devenv shell -- dodge-neat-replay {{ episode }}

[group("dodge")]
dodge-neat-replay-latest epoch:
    devenv shell -- dodge-neat-replay-latest {{ epoch }}
