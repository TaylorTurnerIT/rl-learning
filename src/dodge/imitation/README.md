# Dodge behavior cloning

The genetic algorithm searches open-loop action lists. This package learns a
reactive policy from its accepted examples: at every eight-frame decision it
sees the current projected Dodge state and selects a fresh direction.

## First model: MLP classifier

An MLP (multi-layer perceptron) is a stack of fully connected layers. Every
input feature contributes to every neuron in the next layer; ReLU nonlinearities
between layers let the network combine features into rules more complex than a
single weighted sum.

```text
221 projected state floats → Linear(256) → ReLU → Linear(256) → ReLU → 9 logits
```

The nine output values are **logits**: unnormalized evidence for the collector's
nine directions. `CrossEntropyLoss` compares those logits with the GA action
label. It increases the chosen direction's score relative to all other choices;
we take the highest-scoring direction during deterministic play.

The input is the collector's packed `observation_f32` vector. It already contains
the player and danger-ordered enemy/AOE features, including time-to-intersection.
The fixed start/bootstrap rows are intentionally excluded: they are game setup,
not tactical demonstrations.

## Run

```bash
just dodge-dataset-export /tmp/dodge-dataset.sqlite3
```

The export uses SQLite's backup API, so it captures committed WAL data without
mutating or stopping the running collector. Upload that snapshot to the cloud
workspace; do not copy a live `dataset.sqlite3` file directly.

Open `notebooks/dodge_behavior_cloning.py` in
[Molab](https://molab.marimo.io/), set the snapshot path, and press **Train**.
The notebook imports this package rather than keeping a second training
implementation. It writes a PyTorch model artifact to
`history/dodge/models/behavior-cloning.pt` by default.

The same command works locally and in the cloud:

```bash
uv run dodge-bc-train --database /tmp/dodge-dataset.sqlite3 --epochs 50
```

It automatically selects CUDA when PyTorch can use it, otherwise CPU. The JSON
result reports the chosen `device`. Use `--device cuda` to require a GPU or
`--device cpu` to force the local CPU path. Marimo itself remains optional:
install it with `uv sync --group gpu` when you need the notebook UI. Only load
model artifacts you trust: PyTorch model loading uses serialization.

## What this baseline proves

It tests whether similar danger states across many seeds imply similar good
actions. Classification accuracy is diagnostic only; real success requires a
closed-loop replay that queries this policy every eight frames on never-trained
seeds.

This first command deliberately does not claim that evaluation yet. The next
piece is a persistent Dodge `reset(seed)` / `step(direction)` environment, then
we can measure survival on held-out seeds and later fine-tune the policy with
PPO.

## Follow-up experiments

1. Compare this MLP with a small recent-observation stack or GRU.
2. Add a held-out validation split of ordinary training seeds; retain the ten
   reserved high seeds for final testing.
3. Store `state, action, reward, next_state, terminated` for a bounded mix of
   champions and near-misses before trying Discrete CQL.
4. Try BC-initialized PPO once the interactive environment exists.

References: [PyTorch cross-entropy](https://docs.pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html), [PPO](https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html), [IQL](https://arxiv.org/abs/2110.06169), and [Decision Transformer](https://arxiv.org/abs/2106.01345).
