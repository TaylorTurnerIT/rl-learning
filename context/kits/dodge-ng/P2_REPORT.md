# Dodge NG P2 decision report

P2 used the frozen NG manifest
[`ng-v1.json`](ng-v1.json): 100 fresh seeds, 70 training seeds, and 30 locked
holdout seeds. No legacy trajectories, checkpoints, or GA database rows were
used.

## Matched results

| Candidate | Selected inner mean | Final training mean | Locked holdout mean | Holdout p10 |
|---|---:|---:|---:|---:|
| Board BC | 284.1 | 321.0 | 348.3 | see report |
| PPO from scratch | 195.9 | 196.4 | 209.9 | 195 |
| PPO from BC actor | 394.0 | 378.3 | 360.2 | 255 |
| DAgger aggregate BC | 399.6 | 397.6 | 404.7 | see report |

Detailed artifacts:

- [Board BC report](../../../history/dodge/ng/bc-p2-v1/REPORT.md)
- [PPO comparison](../../../history/dodge/ng/p2-ppo-comparison-20260912/COMPARISON.md)
- [DAgger round report](../../../history/dodge/ng/dagger-p2-r1/REPORT.md)

## Decision

Retain the DAgger aggregate as the strongest current board-state teacher and
warm-start source. It improved locked holdout mean by 56.4 frames over the
previous BC corpus and ended with a small 7.1-frame training-minus-holdout gap.
The selected PPO warm-start checkpoint remains the matched RL reference; later
updates degraded, so selection and resumability stay separate.

The next intervention is the visual branch: exact native indexed pixels with a
four-frame stack, first without augmentation. Its purpose is to measure whether
the policy can recover the useful temporal information that the privileged board
representation supplied, while recording the transfer and convolution cost.

The holdout numbers above are retrospective reporting after the training-side
selection boundary was frozen. They do not change any P2 selection decision.
