---
name: dodge-ng-p1-diagnosis
description: Phase 1 diagnosis from repeated board PPO controls and the neutral-bonus ablation
metadata:
  type: experiment-decision-record
  generation: NG
  phase: P1
  created: "2026-09-03"
  status: complete
---

# Dodge NG Phase 1 diagnosis

## Decision

Do not promote the current board PPO configuration. Keep the neutral bonus as a
controlled ablation, not as the explanation for the collapse. The next learner
should address cold-start action credit with fresh native teacher labels and
behavior-cloning initialization, followed by PPO and a DAgger-style learner-state
refresh.

## Experiment integrity

All valid runs below use the immutable `dodge-ng-v1` manifest with hash
`c75e2c8327888d3f2b31a3bb4681c0a537c8603161f8ffbf4a2539f32d2e01f6`, the same
70/30 seed split, board observation, nine-action policy, 4-frame action hold,
32 native lanes, 200 updates, 256 rollout steps, four update epochs, and CPU
native parallel execution. They each completed 51,200 transitions with zero
environment errors. The locked holdout appears only in the final run reports;
the diagnosis and next-step choice use training-side controls and inner
validation.

The first no-neutral attempt is retained at
`history/dodge/ng/p1-no-neutral-20260903/` as an incomplete harness failure and
is excluded from every result below. Its failure led to the V14 evaluator fix.
The V15 checkpoint fix was applied after this sweep; its regression test now
preserves `checkpoint-best.pt` separately from the final resumable checkpoint.

## Repeated controls

Values are final report values unless marked `peak inner`, which is the highest
training-side validation checkpoint observed during the run.

| Run | learner seed | neutral bonus | final train | final holdout | peak inner | peak update | final neutral | final entropy | transitions/s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline-p1 | 20260903 | 0.02 | 187.8 | 188.2 | 187.9 | 25 | 1.000 | ~0 | 255.3 |
| control-20260904 | 20260904 | 0.02 | 196.4 | 209.9 | 201.4 | 100 | 0.160 | 0.833 | 272.7 |
| control-20260905 | 20260905 | 0.02 | 187.8 | 188.2 | 195.9 | 50 | 1.000 | ~0 | 277.2 |
| no-neutral-20260903-retry | 20260903 | 0.00 | 187.8 | 188.2 | 512.5 | 175 | 0.492 | 0.838 | 254.5 |
| no-neutral-20260904 | 20260904 | 0.00 | 187.8 | 188.2 | 323.9 | 50 | 1.000 | ~0 | 265.7 |
| no-neutral-20260905 | 20260905 | 0.00 | 187.8 | 188.2 | 195.9 | 25 | 0.984 | 0.089 | 269.4 |

The three current controls had mean peak inner validation of 195.1 frames;
the three no-neutral runs had mean peak inner validation of 344.1 frames. That
larger no-neutral number is transient: its mean final training result was 187.8
frames for all three runs, compared with 190.7 for the current controls. The
no-neutral group therefore changed the optimization trajectory but did not
produce a stable final policy. The final holdout means were 195.4 and 188.2
frames respectively; these are reporting outcomes, not selection evidence.

## Fixed-action evidence

The action-control diagnostic evaluated one fixed action over every seed in both
manifest partitions:

| Fixed action | training mean | holdout mean |
|---|---:|---:|
| neutral | 187.8 | 188.2 |
| left, right, up, or down | 196.4 | 209.9 |
| up-left | 86.3 | 91.7 |
| down-left, down-right, or up-right | 104.2 | 108.8 |

The four cardinal controls beat neutral by 8.6 training frames and 21.7
holdout frames on average. The action space therefore contains a measurable
survival signal. The learner's neutral policy is not explained by all nine
actions being equivalent.

## Interpretation

1. The dominant failure is optimization and action credit. Identical
   environment and architecture settings diverge by learner seed. Some runs
   briefly use useful movement, but their value estimates and policy entropy
   become unstable; other runs converge directly to neutral.
2. The neutral bonus is not the primary cause. With the bonus removed, one run
   reached a 512.5-frame inner peak and another reached 323.9, proving that
   non-neutral learning is possible without it. Neither result remained stable
   through the final checkpoint.
3. There is no useful overfitting conclusion yet. Final train and holdout
   survival remain near the same ~188-frame baseline because the policies have
   not learned a durable controller. A small train/holdout gap is not a success
   signal when both sides are at the failure floor.
4. The native execution path is fast enough for broader experiments: these
   runs processed roughly 255–277 transitions per second on CPU. The next
   bottleneck is learning signal quality, not display rendering.

## Next intervention

Implement P2 in this order:

1. Clone native states on the training partition and score all nine actions over
   a short, explicitly tested lookahead. Store the selected action and score
   margin as fresh NG teacher data.
2. Train a board classifier on those labels and measure closed-loop survival on
   inner training seeds before any holdout report.
3. Initialize PPO from the best classifier checkpoint under the same native
   interaction budget.
4. Add learner-visited-state relabeling so the teacher corrects the states the
   learner actually reaches, rather than only easy expert-start states.

Pixel CNNs, recurrent state, replay learners, and the vector/gradient branch
remain planned. They should be compared after this board credit-assignment
intervention has a stable control, so a second representation does not obscure
the current failure.

## Evidence paths

- [`ACTION_CONTROLS.md`](../../../history/dodge/ng/p1-action-controls/ACTION_CONTROLS.md)
- [`baseline-p1/REPORT.md`](../../../history/dodge/ng/baseline-p1/REPORT.md)
- [`control-20260904/REPORT.md`](../../../history/dodge/ng/p1-control-20260904/REPORT.md)
- [`control-20260905/REPORT.md`](../../../history/dodge/ng/p1-control-20260905/REPORT.md)
- [`no-neutral-20260903-retry/REPORT.md`](../../../history/dodge/ng/p1-no-neutral-20260903-retry/REPORT.md)
- [`no-neutral-20260904/REPORT.md`](../../../history/dodge/ng/p1-no-neutral-20260904/REPORT.md)
- [`no-neutral-20260905/REPORT.md`](../../../history/dodge/ng/p1-no-neutral-20260905/REPORT.md)
