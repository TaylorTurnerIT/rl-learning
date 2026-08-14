# RL Learning

A from-first-principles reinforcement learning sandbox. The goal is to understand
how agents, environments, states, actions, rewards, policies, and returns fit
together by implementing them directly in Python—without an ML or RL library.

The project follows a bottom-up path: build tiny environments, study exploration
versus exploitation with multi-armed bandits, then add state and implement
tabular methods such as value iteration, SARSA, and Q-learning. Each step favors
small experiments and readable mechanics over black-box training APIs, following
the approach outlined in [Getting Started with RL](https://chatgpt.com/share/6a7e2d90-36dc-83ea-b3d3-630ff2775383).

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
uv run pytest
```
