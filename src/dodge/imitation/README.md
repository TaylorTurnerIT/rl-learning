# Dodge behavior cloning

The genetic algorithm searches open-loop action lists. This package learns a
reactive policy from its accepted examples: at every eight-frame decision it
sees the current projected Dodge state and selects a fresh direction.

## Board CNN classifier

The trainer rasterizes each raw Dodge state into a 16×16 board with separate
channels for the player, enemies, and areas of effect. Each channel carries
spatial presence plus the relevant velocity, size, or stage information. This
lets convolutional filters learn local danger patterns and pooling combine them
into a board-wide move decision.

```text
19×16×16 board tensor → Conv2d/Pool → Conv2d/Pool → Conv2d → 2×2 pooling
→ Linear(128) → 9 logits
```

The nine output values are **logits**: unnormalized evidence for the collector's
nine directions. `CrossEntropyLoss` compares those logits with the GA action
label. It increases the chosen direction's score relative to all other choices;
we take the highest-scoring direction during deterministic play.

The input is reconstructed from each row's complete `raw_state_json`; the packed
221-float projection is still validated as collector data but is not reshaped
into an artificial image. The fixed start/bootstrap rows are intentionally
excluded: they are game setup, not tactical demonstrations.

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

The trainer prints `epoch=N/T train_loss=F validation_loss=F` after each epoch,
then emits the final JSON summary. This gives immediate loss progress in a
terminal or Marimo logs.
It holds out the highest ten accepted training seed IDs, logs both losses in
`history/dodge/models/behavior-cloning.metrics.json`, and never sends those
rows through the optimizer.

Plot the saved history after training:

```bash
just dodge-bc-plot history/dodge/models/behavior-cloning.metrics.json
```

This writes `behavior-cloning.metrics.png`; rising validation loss while
training loss falls is the overfitting signal.

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

1. Compare this CNN with a small recent-observation stack or GRU.
2. Run closed-loop policy survival on the reserved development-validation seeds;
   retain the ten high evaluation seeds for final testing.
3. Store `state, action, reward, next_state, terminated` for a bounded mix of
   champions and near-misses before trying Discrete CQL.
4. Try BC-initialized PPO once the interactive environment exists.

References: [PyTorch cross-entropy](https://docs.pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html), [PPO](https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html), [IQL](https://arxiv.org/abs/2110.06169), and [Decision Transformer](https://arxiv.org/abs/2106.01345).
