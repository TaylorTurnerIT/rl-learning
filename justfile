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
    devenv shell -- dodge-control {{commands}} {{options}}

[group("dodge")]
dodge-headless commands *options:
    @DODGE_HEADLESS=1 devenv -q shell -- dodge-headless {{commands}} {{options}}

[group("dodge")]
dodge-replay history:
    devenv shell -- dodge-replay {{history}}

[group("dodge")]
dodge-replay-run history:
    devenv shell -- dodge-replay-run {{history}}

[group("dodge")]
dodge-replay-latest epoch:
    devenv shell -- dodge-replay-latest {{epoch}}

[group("dodge")]
dodge-neat-train *options:
    @DODGE_HEADLESS=1 devenv -q shell -- dodge-neat-train {{options}}

[group("dodge")]
dodge-neat-resume run *options:
    @DODGE_HEADLESS=1 devenv -q shell -- dodge-neat-train --resume {{run}} {{options}}

[group("dodge")]
dodge-neat-replay episode:
    devenv shell -- dodge-neat-replay {{episode}}

[group("dodge")]
dodge-neat-replay-latest epoch:
    devenv shell -- dodge-neat-replay-latest {{epoch}}
