---
name: dodge-ng-plan
description: Master plan for the next-generation Dodge reinforcement-learning and visual-training tournament
metadata:
  type: project-generation-plan
  generation: NG
  created: "2026-09-03"
  last_edited: "2026-09-03"
  status: phase-3-pixel-teacher-in-progress
---

# Dodge NG: reinforcement learning and visual training plan

## Purpose

Dodge NG is the next training generation built on the accepted native Dodge
runtime. Its job is to make learning fast enough to iterate on, broad enough to
discover a genuinely good controller, and rigorous enough to distinguish
generalization from memorizing a few seeds.

This plan does not replace the native-port kits in
`context/kits/dodge-native/`. The checked-in PICO-8 cartridge and the accepted
native differential tests remain the game authority. NG adds experiment
orchestration, CNN/RL alternatives, seed-safe evaluation, and performance
evidence around that authority.

## Current boundary

Phase 0, Phase 1, and the completed P2 data/teacher branch are implemented under the scoped root
[`SPEC.md`](../../../SPEC.md). The current NG boundary is:

- [`native/batch.py`](../../../src/dodge/native/batch.py) exposes contiguous native
  batch results, including optional pixels and the derived board tensor.
- [`rl/ppo.py`](../../../src/dodge/rl/ppo.py) provides a resumable board-CNN PPO
  trainer with a nine-action categorical policy and explicit training-side
  evaluation hooks.
- [`ng/manifest.py`](../../../src/dodge/ng/manifest.py) freezes the fresh NG
  sample space and exact 70/30 split; [`ng/train.py`](../../../src/dodge/ng/train.py)
  routes it into native PPO without reading legacy trajectories.
- [`ng/report.py`](../../../src/dodge/ng/report.py) produces split statistics,
  learning curves, diagnostics, and a locked-holdout report.
- [`ng/diagnostics.py`](../../../src/dodge/ng/diagnostics.py) measures all nine
  fixed-action controls without changing the learner or the game contract.
- [`native/batch.py`](../../../src/dodge/native/batch.py) now scores all nine
  actions from cloned canonical snapshots, with restore-once/clone-per-action
  execution in Rust.
- [`ng/teacher.py`](../../../src/dodge/ng/teacher.py) collects fresh planner
  labels from training seeds and memoizes exact snapshot/lookahead scores.
- [`ng/bc.py`](../../../src/dodge/ng/bc.py) trains board behavior cloning and
  selects checkpoints on a training-side inner split.
- [`ng/compare.py`](../../../src/dodge/ng/compare.py) evaluates selected PPO
  checkpoints and writes matched sample-efficiency comparisons.
- [`ng/dagger.py`](../../../src/dodge/ng/dagger.py) collects states visited by
  the selected learner, scores them with the native teacher, aggregates them,
  and retrains BC under the same split protocol.
- [`rl/ppo.py`](../../../src/dodge/rl/ppo.py) now supports an explicit native
  indexed-pixel PPO path with reset-safe four-frame stacks, pixel checkpoints,
  and selectable small/current CNN widths.
- [`P3_REPORT.md`](P3_REPORT.md) records the matched pixel control, its 6.16x
  CPU throughput advantage over board PPO, its later training collapse, and
  representative indexed-pixel failure replays.
- [`imitation/data.py`](../../../src/dodge/imitation/data.py) still loads the
  existing SQLite trajectories for historical experiments; NG does not use it.
- [`neat/evaluator.py`](../../../src/dodge/neat/evaluator.py) preserves the older
  evolutionary path, which remains a comparison track rather than the default
  architecture.
- [`dataset.sqlite3`](../../../history/dodge/dataset.sqlite3) contains useful
  historical GA trajectories. NG treats them as archival evidence only; they
  are not NG training data, demonstrations, or evaluation seeds.

The existing GA, NEAT, PPO, and behavior-cloning experiments are not NG
baselines. Their results, checkpoints, trajectories, seed assignments,
hyperparameters, and scores remain untouched for postmortem context only. NG
starts with a new sample space and produces new evidence.

The native engine and its parity tests are different: they remain the game
authority and the execution foundation for NG. Preserving that authority does
not promote any previous learning result.

The first NG baseline ran 200 native PPO updates over 70 fresh training seeds.
It processed 51,200 transitions in about 200.5 seconds on CPU (255.3
transitions/second). Training and locked holdout survival were effectively the
same at 187.8 and 188.2 frames respectively, with zero horizon completions.
The policy collapsed to 100% neutral actions and near-zero entropy without
improving survival. This is a useful baseline failure signal, not evidence that
the later pixel, replay, teacher, or continuous-action tracks cannot work. The
full artifacts are in
[`history/dodge/ng/baseline-p1/REPORT.md`](../../../history/dodge/ng/baseline-p1/REPORT.md).

Three matched learner-control seeds and three neutral-bonus-off controls are now
complete. The fixed-action diagnostic shows a real action advantage: cardinal
actions survive longer than neutral, while diagonal actions are substantially
worse. The learner controls remain seed-sensitive: one current control retains
non-neutral behavior, while the others collapse; removing the neutral bonus
produces larger transient inner-validation peaks but still collapses or becomes
unstable by the final checkpoint. This closes P1 as a diagnosis rather than a
promotion. The measured decision record is
[`P1_DIAGNOSIS.md`](P1_DIAGNOSIS.md).

P2 is frozen in [`P2_REPORT.md`](P2_REPORT.md). The board DAgger aggregate is
the current privileged-state teacher reference; the next experiment is visual
learning from the exact native raster.

## Target

Build a native-backed experiment tournament that can train and compare:

1. a fast privileged board-state controller;
2. a pixels-only CNN controller with temporal context;
3. imitation-initialized and planner-distilled controllers;
4. replay-based discrete value learners;
5. recurrent policies;
6. continuous vector and spatial-gradient controllers;
7. repaired evolutionary baselines;
8. tuned, regularized variants selected without holdout leakage.

The winner is the policy with strong closed-loop survival across unseen seeds,
low lower-tail failure, reasonable wall-clock cost, and no hidden dependence on
the display or Pemsa process.

## Non-negotiable constraints

- The PICO-8 source, native physics, action order, collision rules, reward
  contract, and indexed-pixel parity remain unchanged.
- Training uses the native Rust batch path. Macroquad, Pemsa, Xvfb, xdotool,
  and full rendering stay outside the hot loop.
- Every observation mode is explicit: board, pixels, full state, hybrid, or
  recurrent history. A model cannot silently receive privileged state.
- NG defines a new, finite seed sample space before training. Exactly 70% of
  that space is training and the remaining 30% is a locked holdout used to
  detect seed overfitting.
- The split is grouped by environment seed. Frames from one seed never appear
  on both sides of the split.
- Hyperparameter search and checkpoint selection use only resampling within
  the 70% training side. The 30% holdout is never an optimization input.
- No prior trajectory, checkpoint, seed manifest, or evaluation result enters
  NG scoring, even if its seed ID happens to be convenient.
- A single successful seed is not a promotion signal.
- Learning claims use closed-loop survival and wall-clock/sample efficiency,
  not classifier accuracy or training loss alone.
- Exact visual comparison remains an engine-conformance gate. Training does
  not render every frame just to produce a screenshot.
- Existing Python/Pemsa paths remain available as oracle and fallback until a
  replacement has passed its relevant behavior and performance gate.

## Dependency spine

```text
NG-P0 evaluation protocol and speed baseline
             |
NG-P1 discrete board PPO reference
       /       |        \
NG-P2 data   NG-P3 visual  NG-P4 replay/value
and teacher  and recurrent learning
       \       |        /
        NG-P5 continuous/vector/gradient branch
             |
NG-P6 tournament, HPO, pruning, and regularization
             |
NG-P7 locked-holdout generalization and release report
```

P0 must precede every learning comparison because an invalid split or metric
would invalidate every later result. P1 must precede promotion decisions
because each alternative needs a stable reference. P2–P5 can then be explored
as separate branches; none should be allowed to hide a failing baseline. P6
needs stable candidates and comparable reports. P7 is last because it must use
the selected configuration without further tuning.

## Seed and leakage protocol

NG starts with a new, explicitly defined seed sample space `S_NG`. It is
disjoint from the seeds and trajectories used by the GA, NEAT, PPO, and
behavior-cloning experiments. The old database and old reserved evaluation
sets remain untouched archives; they are not members of `S_NG`.

Choose the size of `S_NG` before training and make it divisible by ten. I
recommend starting with 100 fresh seeds: 70 training seeds and 30 locked
holdout seeds. The ratio, not the particular count, is the requirement. If a
smaller first smoke corpus is needed, it must still be an explicitly versioned
70/30 sample of the new NG space rather than a slice of the old database.

The 70% side is the only side used for model fitting. Hyperparameter search can
use cross-validation or rotating inner validation drawn from those 70 seeds,
but it may not inspect the 30% holdout to choose a model. After the
configuration is frozen, retrain on all 70% training seeds and evaluate on the
30% holdout as the formal overfit/generalization check.

The manifest must record:

- the complete `S_NG` seed list and its generation method;
- seed role: training or locked holdout;
- split version and assignment rationale;
- pattern-coverage or difficulty metadata, if used before training;
- episode count and maximum evaluation horizon;
- separate environment seeds and learner/RNG seeds;
- an explicit exclusion list for all legacy seed manifests and artifacts.

The split is selected once, then treated as an immutable experiment input. It is
not recomputed after seeing which configuration wins.

## Legacy experiment boundary

The following are preserved but excluded from NG learning and scoring:

- the GA SQLite trajectories and all prior behavior-cloning examples;
- prior GA, NEAT, PPO, and CNN checkpoints;
- prior seed banks, including old development and evaluation ranges;
- prior hyperparameters, rankings, and survival scores as promotion evidence.

They may be cited in a postmortem explaining what failed. They may not seed a
model, label a fresh dataset, select a checkpoint, define the 70/30 split, or
appear in an NG leaderboard.

## Experiment tracks

| Priority | Track | Input | Policy or learner | Reason to run | Main risk |
|----------|-------|-------|-------------------|---------------|-----------|
| P0 | Board PPO | 19-channel board | Nine-action categorical PPO | Fast reference and mechanics diagnostic | Can hide visual difficulty |
| P0 | Pixel PPO | Indexed pixels or fixed RGB | CNN plus frame stack | Tests genuine visual control | More data and transfer cost |
| P0 | BC to PPO | Fresh NG demonstrations | CNN classifier, then PPO | Cheap initialization | Teacher and action-distribution bias |
| P1 | Planner teacher | Fresh native cloned states | Short-horizon action scoring | Generates new labels without legacy data | Teacher objective may be myopic |
| P1 | Recurrent PPO | Pixels or board history | CNN plus GRU | Recovers motion information | Hidden-state reset bugs |
| P1 | Rainbow-lite | Board or pixels | Double/dueling/n-step replay learner | Reuses transitions efficiently | Replay and target instability |
| P1 | Continuous vector | Board or pixels | Bounded two-dimensional action plus SAC | Smooth steering experiment | Changes action semantics |
| P1 | Danger field | Board or pixels | Spatial danger map and derived gradient | Direct test of gradient idea | Field may be poorly supervised |
| P2 | Reactive evolution | Small board controller | GA, ES, or NEAT | Keeps derivative-free search honest | Raw CNN evolution is expensive |
| P2 | Intrinsic exploration | Board or pixels | PPO or Q learner plus small RND bonus | Escape repeated local habits | Curiosity can fight survival |
| P2 | World model | Pixels | Dreamer-style imagined control | Long-term reach goal | Approximate model is unnecessary risk now |

The first comparison should use the board PPO reference, fresh-teacher BC-to-
PPO, pixel PPO, and Rainbow-lite. Recurrent and gradient branches follow once
the evaluator and plots can compare them automatically.

## NG-P0: evaluation protocol and speed baseline

### §G Goal

Produce one evaluator and report format that can score every future learner on
the same seed manifest, checkpoint schedule, observation contract, and wall-clock
measurement.

### §C Constraints

- No learner changes are required to establish the protocol.
- The locked holdout may be plotted for visibility but may not select a model.
- A visual replay is generated only for selected failures, finalists, and
  parity checks.
- All native throughput measurements name machine, toolchain, workload,
  observation mode, lane count, repetitions, statistic, and raw artifact.

### §I Interfaces

```text
seed manifest + run config -> experiment record
checkpoint -> deterministic evaluation report
rollout metrics -> JSONL/SQLite record + plots
native batch -> frames per second, learner steps per second, wall time
failure episode -> action trace, state/pixel hashes, optional visual replay
```

### §V Gate

- The new training and locked-holdout roles are disjoint and together cover
  all of `S_NG`.
- Repeating an evaluation with the same checkpoint and seed produces the same
  result.
- Reports include mean, median, p10, worst-seed survival, horizon completion,
  train/holdout gap, action entropy, and throughput.
- Plots show aggregate curves and individual-seed behavior.
- The evaluator can score board and pixel observations without changing the
  game contract.

### §T Tasks

1. Generate and freeze the new `S_NG` seed manifest and exact 70/30 split.
2. Add a common checkpoint evaluator for every learner family.
3. Store config, Git SHA, native version, observation mode, RNGs, and hardware
   provenance with every run.
4. Add train/inner-validation/holdout plots and per-seed heatmaps.
5. Benchmark native board, pixels-on, and full-state batch workloads.
6. Add a compact HTML report with ETA, best checkpoint, failure replays, and
   comparisons across runs.

### §B Bugs to watch

- accidental frame-level leakage;
- holdout use in early stopping or HPO;
- comparing different action-hold durations as though they were the same task;
- counting native lane throughput without including observation-copy cost;
- displaying a training loss curve without a closed-loop performance curve.

## NG-P1: discrete board PPO reference

### §G Goal

Establish a reproducible, native-batched board-state PPO reference that shows a
real learning signal across multiple learner RNG seeds.

### §C Constraints

Keep the existing nine-action order, survival-frame reward, checkpoint format,
and native/Pemsa parity boundary. Compare `step_frames` 3, 4, and 5 before
expanding the action contract.

### §V Gate

The reference must produce finite metrics, recover from checkpoint/resume,
show non-degenerate action use, and provide training-side and holdout results
for at least three independent learner seeds. If it does not learn, stop and
diagnose before adding algorithms.

### §T Tasks

- Establish current AdamW/ReLU configuration as the control.
- Ablate action hold duration, neutral bonus, learning-rate schedule, rollout
  horizon, entropy coefficient, and GAE parameters.
- Compare AdamW, Adam, and RMSprop under matched learning-rate budgets.
- Compare ReLU and SiLU before searching wider activation families.
- Record survival against environment frames and wall-clock time.

## NG-P2: data, imitation, and exact-simulator teachers

P2 is the current intervention. The P1 evidence supports improving action
credit and initialization before adding pixels, recurrence, or continuous
gradients. The completed first comparison used fresh native planner labels,
board behavior cloning, and matched BC-to-PPO. The remaining P2 experiment is
learner-visited DAgger-style relabeling. All labels and states come from the NG
training partition only.

### P2 evidence so far

- Lookahead 8 and 16 produced almost no decisive labels because most actions
  tied over such a short window. Lookahead 64 produced 3,217 decisive examples
  out of 4,480 (71.8%) across all 70 training seeds.
- The 40-epoch board BC run selected epoch 40 on the ten-seed inner split at
  284.1 mean survival frames. Its final training mean was 321.0 and its locked
  holdout mean was 348.3; the holdout was not used for selection.
- Matched PPO used the same 51,200 native transitions and learner seed. Scratch
  PPO selected update 25 at 195.9 inner / 209.9 holdout mean. BC warm-start PPO
  selected update 75 at 394.0 inner / 360.2 holdout mean, a +150.3 holdout
  delta. Later updates degraded, so `checkpoint-best.pt` is the candidate
  artifact and `checkpoint-latest.pt` remains the resumable final state.
- Native counterfactual scoring restores each source once and clones it for
  each action. Teacher collection also memoizes exact `(snapshot, lookahead)`
  requests; this is a correctness-preserving optimization even when gameplay
  states are mostly unique.
- DAgger round 1 added 2,240 states visited by the selected PPO warm-start
  learner. Aggregate BC selected epoch 40 at 399.6 inner frames and reached
  397.6 training / 404.7 holdout frames, improving the prior BC holdout by
  56.4 frames with a 7.1-frame train–holdout gap.

### §G Goal

Determine whether fresh demonstrations or native short-horizon action scoring
can give RL a useful initialization without teaching seed-specific behavior.

### §C Constraints

Legacy SQLite data is excluded from NG. New behavior-cloning data must be
collected from `S_NG`, and all training uses only the 70% training side.
Classifier loss and action accuracy are diagnostic only.

### §V Gate

BC must be evaluated closed-loop on unseen seeds. BC-to-PPO must be compared
with PPO-from-scratch under equal native interaction budgets. A teacher is
retained only if its labels improve generalization or sample efficiency.

### §T Tasks

1. Generate fresh demonstrations on the NG training seeds.
2. Train a board CNN classifier using only the 70% training side.
3. Evaluate its closed-loop survival with training-side validation, then on the
   locked 30% holdout only for reporting.
4. Initialize PPO from the best checkpoint and compare learning curves.
5. Build a native short-horizon planner that scores all nine actions from a
   cloned state.
6. Distill planner scores or actions into a CNN.
7. If useful, aggregate learner-visited states and relabel them in a DAgger-like
   loop rather than collecting only expert-start states.

## NG-P3: pixel CNN and temporal context

### §G Goal

Train a policy from the exact native raster while measuring the cost of visual
learning separately from the cost of the game simulation.

### §C Constraints

The pixel policy receives no derived board channels unless the run is explicitly
marked hybrid. Begin with a four-frame stack. Use a no-augmentation control
before enabling visual augmentation.

### P3 implementation evidence so far

- The indexed-pixel smoke path completed a full train/inner/training/holdout
  report with the frozen manifest. The first 32/64/128-channel control took
  27.7 seconds for 128 transitions under an intentionally short eight-step
  horizon (4.6 reported transitions/s).
- A matched 16/32/64-channel small control took 17.6 seconds (7.3 reported
  transitions/s). It is now the default architecture; the larger current
  control remains selectable for a direct representation-capacity comparison.
- The optional fast encoder took 13.2 seconds (9.7 reported transitions/s) on
  the same smoke workload. T18 will use it as the speed-oriented control and
  retain small/current as reproducible comparison variants.
- These smoke horizons are plumbing and speed evidence only. They do not
  constitute a learning or generalization result; T18 is the matched budgeted
  run.
- The matched fast pixel control completed 51,200 transitions in 31.9 seconds
  (1,604.3 transitions/second), versus 196.4 seconds (260.6 transitions/second)
  for board PPO with the same budget. Both selected update-25 controls scored
  195.9 inner, 196.4 training, and 209.9 holdout mean frames.
- The pixel run selected update 25; its final update-200 checkpoint scored
  102.7 inner, 104.2 training, and 108.8 holdout frames. Raw pixel PPO is
  therefore a speed-ready observation path, not a promoted learner.
- The detailed decision and worst-seed indexed-pixel replays are in
  [`P3_REPORT.md`](P3_REPORT.md).

### §V Gate

Pixel models must report their performance gap to the board reference, their
throughput cost, and their train/holdout gap. A recurrent model must reset hidden
state on every gameplay reset and retry.

### §T Tasks

- Compare indexed-pixel and fixed RGB input representations.
- Compare small, current, and residual CNNs.
- Compare one frame, four-frame, and eight-frame stacks.
- Add CNN plus GRU after the feed-forward control is measured.
- Test semantics-preserving random shifts or crops.
- Add optional auxiliary heads for danger, velocity, and time-to-intersection.

DrQ is the planned source for the first pixel augmentation experiment because
it studies simple image augmentation as a regularizer for pixel-based RL:
[DrQ](https://arxiv.org/abs/2004.13649).

## NG-P4: replay-based discrete learning

### §G Goal

Test whether replay and multi-step value learning use native experience more
efficiently than on-policy PPO for the nine-action task.

### §C Constraints

Add components progressively. Do not introduce a fully combined Rainbow
implementation without ablations that identify which components help.

### §V Gate

Promote this path only if it improves survival per environment frame or lowers
wall-clock cost while maintaining unseen-seed robustness.

### §T Tasks

- Start with Double Q-learning and a target network.
- Add dueling values, n-step targets, and prioritized replay.
- Compare standard and distributional value targets.
- Add noisy exploration only after replay is stable.
- Evaluate pixels with and without visual augmentation.
- Use IQL or CQL only after NG has collected enough diverse transitions; never
  load the legacy GA database into this path.

Rainbow is the relevant discrete-RL reference for combining and ablating these
improvements: [Rainbow](https://aaai.org/papers/11796-rainbow-combining-improvements-in-deep-reinforcement-learning/).
IQL and CQL are later offline-learning options for biased prior data:
[IQL](https://arxiv.org/abs/2110.06169) and
[CQL](https://arxiv.org/abs/2006.04779).

## NG-P5: continuous, vector, and gradient control

### §G Goal

Test smooth action representations and spatial danger gradients without
changing the underlying game physics or giving the policy unfair control.

### §C Constraints

A policy gradient is an optimization method, not automatically a direction on
the screen. Every gradient experiment must define what the network output
means, how it maps to movement, and how it is rate-limited.

### §I Interfaces

```text
CNN -> bounded (dx, dy) -> native movement adapter
CNN -> danger field D(x,y) -> finite-difference -gradient near player
CNN -> discrete logits + vector/field head -> hybrid fallback controller
```

### §V Gate

All variants use the same movement authority, frame schedule, collision rules,
and reward. The gradient branch must beat or complement the discrete reference
on unseen seeds; visual smoothness alone is not sufficient.

### §T Tasks

1. Implement a continuous-vector experiment with a bounded action envelope.
2. Use SAC as the first continuous-control candidate:
   [SAC](https://arxiv.org/abs/1812.05905).
3. Implement a danger-potential head whose semantics are explicitly “higher is
   more dangerous.”
4. Compute a local finite-difference gradient around the player.
5. Add a discrete fallback when the field is flat or contradictory.
6. Compare discrete, vector, field, and hybrid policies under matched budgets.

## NG-P6: tournament, HPO, and regularization

### §G Goal

Select configurations using repeatable, pruned experiments without overfitting
the locked holdout or wasting time on obviously weak trials.

### §C Constraints

No large Cartesian product. Search one family at a time, use only
training-side resampling for inner validation, and re-run finalists across
independent learner seeds. The 30% holdout remains locked until selection is
finished.

### §V Gate

The selected configuration must improve lower-tail unseen-seed performance or
sample efficiency across repeated runs, not merely win one noisy trial.

### §T Tasks

Use a staged budget:

1. cheap trials on a small inner-validation subset;
2. early pruning of weak trials;
3. three-seed confirmation for finalists;
4. full training-manifest runs;
5. one locked-holdout evaluation per finalist after selection is frozen.

Optuna's TPE sampler is the first HPO candidate:
[Optuna TPE](https://optuna.readthedocs.io/en/stable/reference/samplers/generated/optuna.samplers.TPESampler.html).
ASHA-style pruning is a candidate for early stopping:
[Ray ASHA](https://docs.ray.io/en/latest/tune/examples/tune_pytorch_asha/content/tune_pytorch_asha.html).
Population-based training is deferred until the basic search is stable because
it changes hyperparameters during training and adds attribution complexity:
[Ray PBT](https://docs.ray.io/en/latest/tune/api/doc/ray.tune.schedulers.PopulationBasedTraining.html).

Search dimensions:

- learning rate, optimizer, and weight decay;
- constant versus linear learning-rate decay;
- activation and CNN width/depth;
- frame-stack length and recurrent versus feed-forward policy;
- rollout horizon, minibatch size, and update epochs;
- entropy, value, gamma, and GAE coefficients;
- action-hold duration;
- augmentation and auxiliary-loss weights;
- reward-shaping ablations, including neutral bonus on versus off.

Regularization order:

1. correct grouped split and input normalization;
2. weight decay and learning-rate scheduling;
3. visual augmentation;
4. temporal masking or history augmentation;
5. auxiliary danger and motion prediction;
6. dropout only if the reports show genuine representation overfitting.

Promotion uses mean survival with a lower-tail guard. Report p10 and worst
seed separately; do not optimize the raw minimum because one anomalous seed can
dominate the decision.

## NG-P7: locked-holdout generalization and release report

### §G Goal

Measure the selected policy on the untouched 30% NG holdout and publish a clear
comparison of performance, speed, failure modes, and visual behavior.

### §C Constraints

No architecture, reward, or hyperparameter changes after holdout evaluation
begins. A new idea starts a new NG experiment version.

### §V Gate

- locked-holdout mean, median, p10, and worst-case results are recorded;
- training/holdout gap and learner-seed variance are reported;
- sample efficiency and wall-clock cost are compared;
- representative failures have action traces and visual replays;
- exact native visual parity remains green for the relevant observation path;
- the report distinguishes measured facts, inference, and future hypotheses.

## Required report set

Every promoted run produces:

- `run.json` with provenance, split, configuration, and Git SHA;
- checkpoint and resume metadata;
- JSONL metrics for every update and evaluation checkpoint;
- train/inner-validation/holdout survival curves;
- per-seed heatmap and distribution plots;
- train/holdout gap plot;
- action entropy and action-frequency plots;
- survival versus environment frames and wall-clock plots;
- HPO configuration and pruning history;
- selected failure replays and optional exact pixel comparisons;
- a short decision record explaining why the run was promoted.

## Cavekit routing

| Need | Route |
|------|-------|
| Unresolved split, objective, or gradient meaning | grill |
| External algorithm, library, or current API fact | research |
| Durable behavior, interface, invariant, or task change | spec |
| New learner, observation ABI, or HPO boundary | review |
| Accepted open phase implementation | build |
| Drift or gate audit | check |
| Failed test, parity mismatch, or misleading metric | backprop |
| Accepted phase with behavior held constant | deepen |

NG planning must not silently rewrite `src/dodge/game/SPEC.md` or
`src/dodge/neat/SPEC.md`. Before implementation, route any new observation,
seed, reward, checkpoint, or evaluation contract through the relevant spec.

## First implementation slice when NG resumes

Implement only NG-P0 first:

1. generate and freeze the new `S_NG` sample space and exact 70/30 split;
2. build the shared evaluator and provenance record;
3. add closed-loop plots and the HTML comparison report;
4. benchmark board, pixels-on, and full-state native throughput;
5. run the unchanged PPO implementation on three learner seeds from the new
   70% training side;
6. stop at the P0 gate before adding a new algorithm.

The purpose of NG is not to make the codebase contain every algorithm. It is to
make every serious candidate cheap to run, fair to compare, and difficult to
mistake for a policy that only memorized one seed.
