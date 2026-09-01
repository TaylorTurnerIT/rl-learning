# Direct Dodge PPO

This package trains a policy directly against the live step-wise Dodge
environment. It does not load behavior-cloning weights or action labels. The
actor and value function share a convolutional encoder over the complete raw
board state and produce nine direction logits plus a scalar survival value.

The default reward is the environment's survived-frame delta. A capped neutral
bonus (`0.02` per neutral decision, at most `1.0` per episode) supplies a small
stability preference; it cannot outweigh one additional survived frame.

Run a durable experiment with:

```bash
just dodge-ppo-train --run-dir history/dodge/ppo/production --updates 1000
```

Each update writes `checkpoint-latest.pt`, periodic checkpoints, `metrics.jsonl`,
and `run.json`.

PPO places Pemsa/Xvfb scratch workspaces under `<run>/.runtime` on the same
durable filesystem as the checkpoint. It cleans stale bridge workspaces there
and refuses to start when that filesystem has less than 512 MiB free. The
`just dodge-ppo-train` recipe also routes launcher scratch away from the
shared system `/tmp` mount.

Resume after interruption with:

```bash
just dodge-ppo-train --resume --run-dir history/dodge/ppo/production --updates 2000
```

Training samples only non-held-out seeds. Development validation uses
`29991..30000`; final evaluation uses `30001..30010`. The policy is selected by
deterministic argmax during evaluation. The final score to compare is mean
closed-loop `survival_frames`, not action accuracy or training reward alone.
