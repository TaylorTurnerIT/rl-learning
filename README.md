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
