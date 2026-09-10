# Pixel DQN learning investigation — 2026-09-06

## What the failed CPU run established

The paused one-life run reached 12,094 of 200,000 collection updates. Its best
inner-training mean was 205.5 survival frames at update 10,000, far below the
800-frame relevance gate.

Failure events were not rare: the replay stream recorded 26,214 life losses,
and a replay audit found about 75% of sampled n-step targets terminal. More
lives would therefore increase repeated resets, but would not solve a shortage
of death examples.

The late learner was poorly conditioned. Over its final 500 records, mean loss
was 61.85, mean absolute TD error was 62.35, mean Q was 29.77, and mean target
was 50.15. Each action lasted up to 32 game frames, while five-step returns
spanned up to 160 frames. Raw survival and death rewards then produced large,
frequently clipped updates. This explains instability more directly than a
generic claim that DDQN collapsed.

## Representation risk

The `fast` encoder has 65,018 parameters, downsamples a 128×128 raster to a 2×2
feature map, and treats PICO-8 palette indices as scalar intensity values.
Palette numbers are labels, not an ordered brightness scale. Sparse hazards can
also disappear through aggressive downsampling, especially in early frames
where almost the entire screen is background.

The observed policy repeated similar routes across seeds and still selected
corner targets despite terminal examples being common there. That is consistent
with weak visual discrimination, but it does not prove the encoder is the sole
cause. A supervised perception probe must answer that before another long run.

## Pixel probe result

The first meaningful probe used 6 decisive states per seed, an 800-decision
collection limit, and the exact native waypoint counterfactual scorer. It
produced 399 examples from 69 of the 84 probe seeds: 317 training examples
from 55 seeds and 82 validation examples from 14 disjoint training-side
validation seeds. The final 30 manifest holdout seeds were not opened.

The palette-spatial classifier was materially better than the existing scalar
fast encoder:

| encoder | train accuracy | validation accuracy | train regret | validation regret |
|---|---:|---:|---:|---:|
| scalar-fast | 13.9% | 20.7% | 35.7 frames | 24.0 frames |
| palette-spatial | 54.3% | 35.4% | 20.4 frames | 25.5 frames |

The palette-spatial model fit a 128-example diagnostic subset to 99.2% exact
action agreement, while scalar-fast reached only 30.5%. This demonstrates that
the raw pixel stack contains useful information and that palette identity and
preserved spatial structure matter. It does not yet demonstrate a viable
closed-loop DQN: the validation gap, the 25.5-frame residual regret, and the 31
seeds without a decisive example are still substantial.

The probe also exposed a compute trap. On this CPU, palette-spatial training
took 119.6 seconds for 20 regular epochs plus the tiny-set diagnostic, versus
8.6 seconds for scalar-fast. The native collection was fast enough; full
128×128 convolution on CPU dominated the supervised comparison. The
palette-spatial representation is therefore being added as an explicit DQN
ablation and should be trained on the Colab GPU, while the local probe should
use smaller diagnostic budgets.

## Correctness changes completed before new learning claims

- Native active-lane pixel stepping freezes completed lanes while surviving
  lanes finish the full action duration. Batch size no longer changes action
  cadence when another lane dies.
- Every life loss terminates and flushes n-step replay. Reset pixels cannot be
  bootstrapped across an invisible episode boundary.
- Rewards can be scaled at replay insertion; target updates can be tied to
  optimizer steps; update/sample ratios and clipping are recorded explicitly.
- Final evaluation selects the best inner-training model when present. Resuming
  marks old final reports stale.
- Each checkpoint owns an immutable compressed replay snapshot. Overwriting the
  live ring cannot mutate older checkpoint observations.

## Performance evidence

The corrected T4 smoke completed 128 collection updates and 97 optimizer
updates in 12.90 seconds end to end, including finalization. During the live
recovery drill, steady telemetry reported roughly 1,850–1,900 game frames per
second with 64-sample learner updates around 12 ms. The prior CPU path measured
about 3.8 collection updates per second with learner calls around 208 ms.

These are not yet a controlled GPU scaling benchmark: run lengths, machine
load, and finalization differ. They do show that training is tractable enough to
run short diagnostic experiments before committing to 200,000 updates.

## Next experimental sequence

1. [x] Train a small classifier on training-seed pixel stacks labeled by the
   native counterfactual action scorer. It must overfit a tiny subset, then
   generalize to disjoint training seeds. This separates perception failure from
   RL credit assignment.
2. [x] Compare palette handling: learned 16-color embedding or one-hot planes versus
   scalar index division. Preserve exact raw pixels in both cases.
3. [x] Compare the current aggressive encoder with a spatial encoder that
   retains a larger feature map. Choose architecture from perception accuracy
   and latency, not survival reward.
4. Integrate the palette-spatial encoder into isolated pixel DQN and run a
   short matched GPU pilot against scalar-fast. Select the next representation
   using training-side regret, stability, and throughput.
5. Run short DDQN trials with reward scale `1/32`, `n_step=3`, learning rate
   `1e-4`, optimizer-step target cadence, and corrected fixed-duration actions.
6. Sweep batch size, lane count, and gradient steps while holding replay ratio
   explicit. Select throughput settings before HPO.
7. Use training seeds only for pruning and selection. Open the 30-seed holdout
   once after the model family and hyperparameters are frozen.
8. Advance to longer runs only when the model beats the 800-frame training gate
   and diagnostics show bounded TD error, useful action margins, and no sustained
   clipping plateau.

Three-life training remains a matched ablation, not the default remedy. It
still resets the same seed after collision; it does not continue the dead game
timeline into later content.
