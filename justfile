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
dodge-control commands:
    devenv shell -- dodge-control {{commands}}
