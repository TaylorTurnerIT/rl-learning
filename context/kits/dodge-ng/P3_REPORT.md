# Dodge NG P3 pixel decision report

P3 used the frozen [`ng-v1.json`](ng-v1.json): 100 fresh seeds, 70 training
seeds, and 30 locked holdout seeds. The pixel run used native indexed pixels,
a four-frame stack, the fast 16/32/64 CNN, the existing nine-action contract,
and the same 51,200-transition PPO budget as the matched board control.

## Matched controls

Checkpoint selection used only the first 10 training seeds. The locked holdout
was evaluated after selection.

| Control | Selected update | Inner mean | Train mean | Holdout mean | Holdout p10 | Wall time | Throughput |
|---|---:|---:|---:|---:|---:|---:|---:|
| Board PPO scratch | 25 | 195.9 | 196.4 | 209.9 | 195 | 196.4s | 260.6 transitions/s |
| Pixel PPO fast | 25 | 195.9 | 196.4 | 209.9 | 195 | 31.9s | 1,604.3 transitions/s |

The selected controls produced the same aggregate split statistics under this
budget. Both selected checkpoints had a 0.0-frame train-minus-holdout p10 gap;
the selected mean gap was -13.5 frames for both. Pixel rollout/training was
6.16x faster in this CPU run, so the exact-raster path is cheap enough for
larger searches even though it did not improve the policy by itself.

## Pixel learning trend

The pixel run selected update 25 at 195.9 inner-validation frames. The final
update-200 policy fell to 102.7 inner, 104.2 training, and 108.8 holdout
frames. Entropy ended at 1.871 and the rollout neutral fraction at 5.1%; this
was not the earlier 100%-neutral collapse, but it still failed to retain the
selected behavior. The final model is therefore not promoted over its selected
checkpoint.

The full generated report and curves are in
[`p3-pixel-ppo-fast-20260903/REPORT.md`](../../../history/dodge/ng/p3-pixel-ppo-fast-20260903/REPORT.md).

## Representative visual failures

The selected-checkpoint worst training and holdout traces each survived 193
frames. The final-checkpoint worst traces survived 101 frames and reach the
game-over raster. Each replay includes initial, early, midpoint, and terminal
indexed-palette frames plus the deterministic action trace:

[`failure-replays/index.html`](../../../history/dodge/ng/p3-pixel-ppo-fast-20260903/failure-replays/index.html)

The report's [`per_seed_survival.png`](../../../history/dodge/ng/p3-pixel-ppo-fast-20260903/per_seed_survival.png)
shows the complete 70/30 per-seed distribution; no holdout seed was used for
selection.

## Decision

P3 validates the native pixel observation ABI, reset-safe temporal stack, and a
useful six-fold throughput advantage. Raw pixel PPO is not yet a competitive
learner: its selected result matches the scratch board control while its later
updates degrade sharply. The next intervention is fresh planner-labeled pixel
data and a pixel-CNN behavior-cloning actor warm start, followed by a matched
pixel PPO run. The holdout remains locked until that selection is complete.
