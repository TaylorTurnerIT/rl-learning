---
name: dodge-ng-research-process
description: Evidence trail and repeatable research method used to design the Dodge NG training roadmap
metadata:
  type: research-method
  generation: NG
  created: "2026-09-03"
  last_edited: "2026-09-03"
  status: recorded
---

# How the Dodge NG research was done

This document explains the process behind `PLAN.md`. The useful part was not
finding a long list of fashionable algorithms. It was connecting the observed
failures to the local code and data, then using primary sources to decide which
experiments were worth making comparable.

## The starting question

The request contained several problems that look related but need different
tests:

- GA produced impressive behavior on too few seeds.
- NEAT was expensive and too conservative to learn useful behavior.
- Training progress was slow enough to discourage iteration.
- CNN and reinforcement-learning choices had not been compared under one
  evaluation protocol.
- A spatial-gradient controller might be better than discrete directions, but
  its output semantics were not defined.
- All previous experiments should remain untouched as lessons, but none of
  their data or results should be reused by the next generation.

I converted those observations into three research questions:

1. Which learning families fit nine discrete actions, and which fit a bounded
   continuous steering action?
2. What makes a pixel-based CNN generalize across seeds instead of memorizing
   visual trajectories?
3. How can data collection, replay, hyperparameter search, and native batching
   improve sample efficiency and wall-clock iteration time?

The gradient idea was kept as a fourth design hypothesis. I did not treat it as
an established method for this game because no source was found that proves a
gradient-derived controller is better for Dodge.

## Step 1: inspect local authority before searching

I first used the repository as the source of truth. The purpose was to avoid
researching a hypothetical environment that differs from the one we actually
have.

The relevant local evidence was:

| Question | Local evidence | Consequence |
|----------|----------------|-------------|
| What can the native runtime expose? | [`src/dodge/native/batch.py`](../../../src/dodge/native/batch.py) | Board, pixels, hashes, and structured batch results can be compared as separate observation modes |
| What does the current PPO actually do? | [`src/dodge/rl/ppo.py`](../../../src/dodge/rl/ppo.py) | Current reference is a board CNN, nine categorical actions, GAE, clipped PPO, AdamW, and native batching |
| What data exists? | [`history/dodge/dataset.sqlite3`](../../../history/dodge/dataset.sqlite3) and [`src/dodge/native/ga.py`](../../../src/dodge/native/ga.py) | Existing GA trajectories explain an earlier failure mode, but NG excludes them from training, demonstrations, and scoring |
| How is imitation split today? | [`src/dodge/imitation/data.py`](../../../src/dodge/imitation/data.py) | The existing seed split needs an explicit NG manifest and grouped 70/30 protocol |
| Why might NEAT be unstable? | [`src/dodge/neat/evaluator.py`](../../../src/dodge/neat/evaluator.py) and [`src/dodge/neat/SPEC.md`](../../../src/dodge/neat/SPEC.md) | The current contract uses a small seed bank, so robustness must be measured rather than assumed |
| What speed boundary already exists? | [`context/kits/dodge-native/cavekit-dodge-native-p7-benchmark-report.json`](../dodge-native/p7-benchmark-report.json) | Native batching should be the training hot path; Pemsa remains an oracle and visual regression path |

This local pass also exposed an important distinction: the GA database has
surviving episodes across many seeds, but that does not prove that one reactive
policy transfers across those seeds. A trajectory can be good while its policy
representation remains seed-specific or open-loop. Because the new generation
is intended to measure that question cleanly, the database is now archival
only. It will not supply NG labels, pretraining, evaluation seeds, or leaderboard
entries.

## Step 2: turn failures into hypotheses

I did not start by choosing an algorithm. I wrote down hypotheses that could be
disproved:

| Hypothesis | Evidence needed | Disproof |
|------------|-----------------|----------|
| GA was selecting seed-specific action sequences | Per-seed policy transfer and reactive closed-loop evaluation | One fixed policy remains strong across held-out seeds |
| NEAT's search signal was too noisy or conservative | Fitness variance, species statistics, mutation/topology changes, larger fixed seed banks | It learns consistently after a fair evaluator and sufficient budget |
| PPO may be under-tuned rather than unsuitable | Repeated learner seeds, longer native budgets, action/entropy/value diagnostics | It still fails while replay and imitation paths learn |
| Pixels need temporal context | One-frame versus stacked-frame versus recurrent CNN | One-frame pixel policy matches recurrent performance |
| Visual augmentation may reduce seed memorization | Same architecture with and without semantics-preserving augmentation | Augmentation worsens held-out survival consistently |
| A gradient output may improve control smoothness | Matched-action-authority comparison against discrete actions | It does not improve survival or robustness |

This step keeps a plausible explanation from becoming a fact. A research source
can support a technique, but only an experiment can establish that it helps this
game.

## Step 3: choose narrowly scoped external questions

The research pass used one to three questions at a time instead of searching
for “the best reinforcement-learning algorithm.” The questions were:

### Q1: discrete action learning

- Is PPO a reasonable reference for a nine-action policy?
- Is a replay-based value learner a meaningful alternative?

### Q2: visual representation and regularization

- What established techniques address pixel-based overfitting?
- Is temporal memory worth testing when a single frame does not reveal motion?

### Q3: data and experiment efficiency

- Can imitation learning reduce the cold-start problem?
- Which search and pruning methods reduce wasted training runs?

The continuous/vector branch was researched only far enough to choose a
reasonable first learner. The spatial danger-field interpretation remains an
NG experiment, not a literature-backed conclusion.

## Step 4: prefer primary sources

The source hierarchy was:

1. the checked-in game, native runtime, tests, metrics, and accepted reports;
2. original research papers for algorithmic claims;
3. official library documentation for tool behavior;
4. secondary explanations only when a primary source was unavailable.

The AWS Automated Reasoning announcement provided useful motivation for
separating claims from verification evidence. It was not used as evidence that
an RL algorithm would work, nor as evidence for a Lua-to-Rust transpilation
path. Those are separate technical questions.

## Step 5: distill each source to one decision-relevant finding

The external research ledger was deliberately short. Each finding has a source
and a plan consequence.

| ID | Topic | Source-backed finding | Plan consequence |
|----|-------|-----------------------|------------------|
| R1 | PPO | PPO uses a clipped surrogate policy-update objective and alternates environment interaction with minibatch updates | Retain discrete PPO as the reference path, not as the assumed winner: [PPO](https://arxiv.org/abs/1707.06347) |
| R2 | Imitation distribution shift | DAgger aggregates states visited by the learner and obtains labels under an online imitation-learning procedure | Use learner-visited states for a later fresh NG planner-teacher loop instead of relying on legacy GA trajectories: [DAgger](https://arxiv.org/abs/1011.0686) |
| R3 | Pixel regularization | DrQ studies simple image augmentation as a way to improve pixel-based model-free RL | Add augmentation only after a no-augmentation pixel control is measured: [DrQ](https://arxiv.org/abs/2004.13649) |
| R4 | Discrete replay | Rainbow combines multiple value-learning improvements and evaluates their contributions through ablations | Test replay, n-step, double, dueling, prioritized, and noisy components progressively: [Rainbow](https://aaai.org/papers/11796-rainbow-combining-improvements-in-deep-reinforcement-learning/) |
| R5 | Continuous control | SAC is an off-policy maximum-entropy actor-critic method designed for continuous control | Use SAC as the first learner for a bounded two-dimensional steering head: [SAC](https://arxiv.org/abs/1812.05905) |
| R6 | Offline data | IQL and CQL address different problems created by learning from fixed, potentially biased offline datasets | Keep offline RL as a later data path, after collecting diverse NG transitions; never import the legacy GA database: [IQL](https://arxiv.org/abs/2110.06169), [CQL](https://arxiv.org/abs/2006.04779) |
| R7 | HPO search | Optuna's TPE sampler models promising and non-promising regions of a search space and supports seeded sampling | Use a lightweight TPE search after the evaluator is stable: [Optuna TPE](https://optuna.readthedocs.io/en/stable/reference/samplers/generated/optuna.samplers.TPESampler.html) |
| R8 | Early stopping | Ray Tune documents ASHA as an aggressive early-stopping scheduler for underperforming trials | Use multi-fidelity pruning so weak trials do not consume full native budgets: [ASHA](https://docs.ray.io/en/latest/tune/examples/tune_pytorch_asha/content/tune_pytorch_asha.html) |
| R9 | Schedule adaptation | Ray Tune's PBT periodically replaces weak trials with strong ones and mutates selected hyperparameters | Defer PBT until ordinary search is attributable and stable: [PBT](https://docs.ray.io/en/latest/tune/api/doc/ray.tune.schedulers.PopulationBasedTraining.html) |
| R10 | Intrinsic exploration | RND uses prediction error against a fixed random target network as an intrinsic exploration signal | Keep RND optional and small because Dodge already has dense survival feedback: [RND](https://arxiv.org/abs/1810.12894) |
| R11 | World models | DreamerV3 learns a world model and trains partly through imagined trajectories | Keep world-model RL as a reach goal because the exact native simulator is already available: [DreamerV3](https://arxiv.org/abs/2301.04104) |

The wording matters. “This paper reports X” is a sourced fact. “Therefore this
is a candidate for Dodge” is an inference. “This should be first” is a project
recommendation. I kept those levels separate.

## Step 6: map findings to the actual environment

The sources were not ranked in isolation. They were filtered through the local
constraints:

- Nine discrete directions make PPO and replay-based discrete learning natural
  first comparisons.
- Exact native batching makes larger seed banks and repeated learner seeds
  practical, so the previous single-seed evidence is not sufficient.
- Native pixels already exist, so a real visual path is possible without a
  second renderer or window in the training loop.
- The game has dense survival feedback, so intrinsic curiosity is not the first
  exploration intervention.
- An exact simulator enables short-horizon action scoring and data generation;
  that is more immediately useful than learning an approximate world model.
- The existing GA data is narrow and selected, so it is excluded from NG.
  Behavior cloning must use fresh NG demonstrations and remains an
  initialization experiment, not the final objective.
- Continuous steering changes the action interface, so it must be compared
  under a matched movement envelope rather than treated as a drop-in policy
  head.

This filtering step is where most of the practical value came from. The papers
provided candidate mechanisms; the native runtime and failure history decided
their order.

## Step 7: design the evaluation before HPO

Hyperparameter optimization can make a bad evaluation process look scientific.
The evaluation contract therefore comes first:

1. split by environment seed, never by correlated frames;
2. freeze the manifest;
3. reserve an inner validation group for search;
4. keep the locked 30% holdout out of selection;
5. repeat promising configurations across learner RNG seeds;
6. report lower-tail and per-seed behavior, not only the mean;
7. measure both environment frames and wall-clock time;
8. inspect visual replays only for finalists and failures.

The 70/30 request means the entire new NG seed sample space is partitioned
before training: 70% for model fitting and 30% as a locked holdout for the
overfit/generalization check. The sample space must be new and disjoint from
all prior experiments. A count divisible by ten makes the ratio exact; 100
fresh seeds is the recommended first NG corpus. Inner validation for HPO is
drawn only from the 70% training side and does not turn the 30% holdout into a
tuning set.

## Step 8: turn the research into phase gates

Each technique received a place in the dependency graph only if it had a
testable gate:

- evaluation first, because every later result depends on it;
- board PPO next, because it is the fastest reference;
- imitation, pixels, recurrence, and replay as independently rejectable
  branches;
- gradient control only after the discrete action contract is measured;
- HPO after candidates produce comparable metrics;
- fresh-seed generalization last, after selection is frozen.

This prevents a slower algorithm, a larger CNN, or a novel action head from
being introduced as an explanation for a broken evaluator.

## What remained intentionally unresolved

These are questions, not hidden assumptions:

- the size and generation method of the fresh NG seed sample space;
- the manifest mechanism that proves every legacy seed and artifact is
  excluded;
- whether indexed pixels or fixed RGB should be the primary pixel encoding;
- whether frame stacking is enough or a GRU is needed;
- the planner horizon and action-scoring function;
- whether the gradient field should represent danger or safety;
- how much lower-tail performance should influence promotion;
- whether board-state PPO, visual PPO, or replay learning wins after equal
  native budgets.

The plan assigns each unknown to an experiment instead of pretending it was
resolved by literature.

## A repeatable version of this process

For the next substantial training question, the same method can be reused:

1. State the desired behavior and the failure that would make the result
   misleading.
2. Inspect the current implementation, artifacts, tests, and live boundary.
3. Write two to five falsifiable hypotheses.
4. Scope no more than three external questions at a time.
5. Search original papers and official documentation first.
6. Record one source-backed finding per source.
7. Mark each statement as fact, inference, recommendation, or unknown.
8. Map candidates to a controlled comparison and an acceptance gate.
9. Freeze data splits and metrics before tuning.
10. Write the plan and research record before changing implementation.

The process is successful when it narrows the next experiment, makes failure
diagnostic, and prevents a plausible result from being mistaken for a proven
one.
