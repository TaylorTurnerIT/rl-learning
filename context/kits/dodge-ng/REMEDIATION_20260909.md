# Hazard DDQN audit and remediation plan

Date: 2026-09-09. Status: audit verified; implementation proposed.

Goal: repeatable 9,000+ native-frame survival with an affordable training loop.
Retain native game physics, centered waypoints, DDQN action selection,
`hold_decisions=1`, and `step_frames=3`. No new HPO campaign. Initial work and
experiments run on CPU; a later Colab deployment requires a separate launch
decision after measured local gates.

## Audit confirmation and corrections

Reference run:
[`waypoint-hazard-cpu-iterations-20260909`](../../../history/dodge/ng/waypoint-hazard-cpu-iterations-20260909/run.json).
Selected checkpoint: step 13,824; final collection step: 16,384.

| Finding | Verdict and evidence |
|---|---|
| Wall policy | Confirmed. Fresh independent rollouts of checkpoint-best reproduce every saved native action for seeds 30181 and 30182. Waypoint `left` counts: 86/92 and 104/106; survival: 275 and 317; minimum x: 2.0. |
| Architecture | Confirmed: 3,588 inputs, Linear(3588,256), LayerNorm, ReLU, Linear(256,256), ReLU, value/advantage heads, 987,658 parameters. DDQN target selects with online network and evaluates with target network. |
| Weak spatial structure | MLP has no built-in spatial weight sharing. The earlier claim that flattened cells are inherently unrelated was too strong: an MLP can learn their relationships. A CNN improvement requires an experiment. |
| Hazard semantics | Confirmed: 16×16 fixed-center queries, zero initial hypothetical velocity, neutral simulation, H=32. First lethal frame is retained; raw reference no-hit is infinity; learner no-hit is 33. This implements the agreed fixed-center definition. Trajectory risk would be an additional field. |
| Sparse startup map | Confirmed: 251/256 cells have no predicted death within H; five cells have TTC 10,15,25,31,31. This is expected with one distant enemy and does not prove the map is broken. |
| Q margin | Initial left/up margin is approximately 0.0446. Q-values are return estimates, not calibrated confidence. A small margin alone cannot establish policy quality or collapse. |
| Input scales | Confirmed: raw TTC 0..33, corner bit codes 1/2/4/8, presence 0/1, velocity divided by 128. Scale mismatch is a plausible conditioning issue, not a proven cause of collapse. |
| Wall controller | Confirmed: repeated left at column zero targets x=5.84375; sign steering ignores velocity. Momentum and recentering produce corrections in both axes. Arrival state is recreated each policy decision, so crossing-based latching cannot protect the next decision with hold=1. |
| AOE coverage | Present in source: personality -1 explosion footprint and active pattern rectangles; cloned engine advances their dynamics. Existing footprint test is not an end-to-end test of future death time. Previous audit overstated validation of AOE correctness. |
| Exploration | Confirmed: recorded epsilon 0.737363 at selected checkpoint and 0.688723 at final collection. High exploration may hinder this short experiment, but off-policy DDQN can learn from random actions; causality remains untested. |
| Number of updates | Correction: 16,384 collection updates, **14,385 optimizer updates**, first optimizer call at collection step 2,000. There are 524,288 native lane steps. |
| Performance | Reported training mean 192.7714, median 170.5, best 385; inner mean 203.4. The 800-frame gate fails. No current-run holdout evaluation. |

Source owners: [DDQN](../../../src/dodge/ng/dqn.py),
[waypoints](../../../src/dodge/ng/waypoint.py),
[hazard fields](../../../native/crates/dodge-batch/src/hazard.rs),
[native simulation](../../../native/crates/dodge-core/src/game.rs).

### Findings missing from the earlier audit

1. **9,000 frames are currently impossible to measure.**
   `max_episode_steps=2000` at three frames/step stops evaluation at 6,000.
   Use a separately declared 12,000-frame safety cap for the new objective,
   with 9,000 as a survival threshold, not an episode cutoff.

2. **The inherited HPO settings belong to a different decision timescale.**
   Trial 5 used legacy 225-value observations, spacing 32, hold=8, step_frames=4,
   120,000 collection updates, replay capacity 100,000. The new run uses hazard
   inputs, centered spacing 7.6875, three-frame decisions, 16,384 collection
   updates, and replay capacity 8,192. These are not matched learning results.
   Five-step returns changed from up to 160 native frames to 15. Gamma 0.99
   changed from a nominal discount half-life of 2,207 frames to 207. Under the
   current discount, a reward 9,000 frames ahead has weight about 8.05e-14.
   Dense survival rewards can still support long episodes; this calculation
   identifies a changed objective, not proof that 9,000-frame learning is impossible.

3. **Learner conditioning needs measurement.**
   In the last 1,000 learner records, mean absolute TD error is 17.8655,
   mean Q is 220.7838, and 99.6% of pre-clipping gradient norms exceed 10.
   Clipping is allowed and sometimes useful; sustained clipping warrants
   checking feature/reward scaling and targets before increasing network size.

4. **Replay diversity is short.**
   8,192 transitions hold roughly 256 collection updates at 32 lanes.
   Observation and next-observation arrays alone occupy 224.25 MiB. Nominal
   sample/insert ratio for the full run is 3.512; after warmup it is approximately
   four sampled transitions per newly inserted transition. Report replay age,
   seed/difficulty coverage, and terminal fraction before enlarging it.

5. **AOE stage and overlapping entities lose information.**
   Explosion stage is size/max_size; it omits `isizing`, so growth and shrinkage
   at the same size share that feature. Velocity is averaged and type is max
   pooled when entities overlap. Opposing velocities can cancel. Player size,
   freeze state, and remaining effect timers are absent from the four scalars.
   TTC incorporates some consequences inside H, but the observation is not a
   complete Markov state and does not expose all information needed beyond H.

6. **Timeouts currently suppress bootstrap.**
   `NStepAccumulator._emit` treats both death and truncation as zero bootstrap.
   For an artificial cap on continuing survival, flush at the boundary but
   bootstrap from the final pre-reset observation. If instead defining a
   finite-horizon task, remaining time must be observable. This becomes material
   as survival approaches the cap; it does not explain the current early deaths.
   See [Gymnasium time-limit semantics](https://gymnasium.farama.org/main/tutorials/handling_time_limits/).

7. **The overlay exposes only TTC.**
   Replay stores a TTC plane, not all 14 model channels. Its green/safe label
   conflates no predicted collision within H with safety. An inspection UI must
   display the horizon, distinguish censored/no-hit cells, and expose raw values.

### What the current runtime evidence supports

Current-run metric sums:

| Recorded phase | Seconds | Share of these phase sums |
|---|---:|---:|
| Collection | 1,775.88 | 68.4% |
| Learning | 657.91 | 25.3% |
| Checkpoint | 129.08 | 5.0% |
| Evaluation | 35.18 | 1.4% |

Total is 2,598.05 seconds. These are instrumented sums, not complete wall time:
startup, final evaluation/finalization, telemetry, and some best-checkpoint work
are not fully attributed. Collection currently includes hazard construction;
its subphases need separate instrumentation.

The reference TTC algorithm permits 32 lanes × 256 cells × 32 simulated frames
= 262,144 hypothetical frame advances per collection update, versus at most
96 real lane-frame advances. Early collision exits reduce actual work. This
explains why a small neural network does not imply a cheap training loop.

Earlier [CPU profiling](../../../history/dodge/ng/profiling-cpu-20260909/CPU_PROFILE_REPORT.md)
found expensive hazard generation and thread oversubscription. Its legacy
collector timings and hold=8 cadence speedups cannot be applied to this run:
with hold=1, every-step and decision-boundary observation cadence coincide.

## Success criteria and experiment rules

Proposed acceptance target: across three independently trained learner seeds,
the median learner's evaluation median reaches at least 9,000 frames and at
least half of its evaluation episodes reach 9,000. Report all learner results,
mean, median, P10, minimum, maximum, completion fraction, and uncertainty over
environment seeds. A single 9,000-frame replay is a milestone, not this gate.

Use the existing 70 training seeds for development, with an explicit inner
partition for supervised probes and selection. The old HPO already published
the existing 30-seed holdout results; describe that set as previously evaluated,
not untouched. Before the final campaign, preregister an additional disjoint
confirmation manifest under the NG seed rules, reserve its holdout, and open
it once after selection. Do not tune on that result.

Keep hold=1, step_frames=3, and the native difficulty/powerup/pattern settings
fixed. Test one factor at a time, or explicitly name a coupled experiment.
Exploration, target synchronization, replay ratio, and budgets must name their
units: native frames, inserted transitions, or optimizer updates. A nominal
collection update is not an optimizer update or a fixed amount of gameplay.

## Delivery phases

| Phase | Depends on | Deliverable and gate | Deferred work |
|---|---|---|---|
| P0: establish trustworthy measurements | — | Reproducible failures, cap/timeout tests, AOE oracle corpus, phase timings, baseline controls | Model changes → P2/P3 |
| P1: make waypoint control predictable | P0 | Velocity-aware controller passes obstacle-free settling and collision cases at three-frame cadence | Learned policy changes → P2/P3 |
| P2: prove the representation carries useful decisions | P0 + P1 for trajectory features | New observation version; tiny-set fit and validation regret gates | Long training → P3/P4 |
| P3: demonstrate stable short learning | P1 + P2 | CPU DDQN pilots beat controls across learner seeds and pass 800-frame gate | 9,000 claim → P4 |
| O1: reduce runtime cost | P0; selected observation for final measurement | Matched repeated CPU improvement with parity and bounded memory | GPU-specific optimization → P4 |
| P4: expose later game and confirm 9,000 | P3 + O1 | Staged 2,000/4,000/9,000 survival gates, later-game coverage, final confirmation | Algorithm expansion only if measured failure warrants it |

O1 can proceed beside P1–P3 on independent code. Learning experiments need
not wait for every runtime optimization. Final throughput acceptance must use
the selected observation and controller, because they determine the workload.

### P0 — measurement and correctness

1. Save hashes/configuration of current checkpoint, source, native extension,
   manifest, and two reproductions. Add deterministic diagnostic rollouts for
   neutral, fixed directions, repeated-left waypoint, and a short native
   waypoint lookahead controller. Planner is a feasibility reference only;
   DDQN remains the deployed action selector.
2. Add collision fixtures: ordinary enemy, moving enemy, imminent enemy death
   spawning AOE, growing/shrinking explosion, pattern activation/movement,
   powerup-triggered explosion, simultaneous threats, and collision between
   sampled observations. Compare center TTC with independent canonical rollouts
   at horizons 1/32/96. Include live-state nonmutation and serial/parallel parity.
   Use first death, not maximum TTC. Zero means lethal overlap of the player's
   collision box at the center, not simply an enemy anywhere in the same cell.
3. Introduce explicit frame-cap and truncation contracts. Test death on the
   last permitted frame, timeout while alive, final-observation bootstrap,
   lane resets, and n-step flushing. Preserve legacy checkpoint semantics.
4. Record greedy action distribution separately from exploration, native action,
   target, player velocity, time near walls, reversals, finite-TTC fraction,
   episode phase, death type, terminal fraction, replay age, clipping, Q/target
   scales, and optimizer counts. Q margin is descriptive, not confidence.
5. Dashboard inspection: checkpoint identity/config, all feature planes,
   hover values, no-hit legend, chosen waypoint/native action and nine Q-values,
   trajectory, and expanding/shrinking AOE state. Record new diagnostic sidecars
   from saved checkpoints; preserve old replay compatibility.
6. Restore missing Pemsa dependency for cartridge comparison before claiming
   original-game parity. Native-only diagnostics can continue while it is missing.

Gate: deterministic native fixtures and train/eval/replay action agreement;
explicit unknowns for unavailable cartridge checks; baseline and timings saved.
No model change is credited for fixing incorrect measurement.

### P1 — waypoint movement

Retain centered targets and three-frame policy decisions. Replace position-only
sign steering with a small controller that evaluates the nine existing native
inputs through the actual three-frame player physics. Score target error and
residual velocity so it brakes before arrival. Keep native acceleration,
friction, collision order, and wall clamping unchanged; do not teleport or zero
the live player's velocity. Preserve current controller as the comparison.

Make target persistence explicit: retain the target/arrival state when the same
target is selected, release it for a new target, and expose any persistent
controller state to the model. Define neutral as current-cell settling, document
it, and ensure hold=1 does not reset settling memory every frame block.

Treat duplicate boundary targets separately from danger. Choose a deterministic
canonical action for identical targets; if masking is used, apply the same mask
to exploration, greedy actions, DDQN next-action selection, evaluation and replay.
Never mask a move solely because its TTC is low; that may be the only escape.
Spawn labels/halo remain labels, with no corner penalty added by default.

Gate: on legal generated player states near every wall/corner and cell center,
the controller settles inside tolerance without sustained oscillation in an
obstacle-free fixture; reaches requested neighboring centers; responds to a new
target within the next three-frame block. Report settling frames, overshoot,
reversals and collision outcomes against old steering. Verify the P0 planner
can exploit the controller before spending on learned policy trials.

### P2 — representation before architecture guesses

First candidate retains the fixed-center map and adds missing distinctions:

- Encode TTC as normalized finite value plus a no-hit/censored mask. No-hit
  means only survival beyond H. Retain raw frame TTC for inspection/reference.
- Replace raw corner bit magnitudes with binary decoded labels; replace ordinal
  enemy type with categorical indicators. Scale velocities in physical movement
  units, not screen width. Resolve overlaps without averaging away threats.
- Add explosion expanding/shrinking state and imminent-death information,
  player size/effects, relevant pattern phase/timers, and controller target/error
  and velocity. Audit off-screen position loss. Expose only declared game-state
  features, not seeds or RNG state that could shortcut learning.
- Compare H=32 with 96 on a frozen corpus; retain earliest collision semantics.
  Do not equate larger N with more useful prediction. Keep N=16 for the first
  learning comparison so geometry does not change simultaneously.
- Optional trajectory branch: for each of nine waypoint actions, simulate the
  candidate with current velocity and P1 steering for the next three frames,
  then a declared continuation to 12/32/96. Report collision/path clearance,
  endpoint and remaining velocity. Mark predictions as conditional on that
  continuation; they are not guaranteed safety under future policy choices.
  Budget these nine rollouts separately from the 256 center queries.

Collect training-side canonical snapshots spanning early/late game and decisive
states. Label all nine waypoint choices with the same native controller and
lookahead. Include ties and label horizon explicitly; never force an arbitrary
winner for equivalent actions. Split probe validation by environment seed.

Compare normalized MLP, then a small spatial encoder on identical data. Initial
CNN candidate: two or three 3×3 stride-one layers, 16/32/32 channels, preserving
16×16 resolution; gather local player/target features plus a coarse spatial
summary and scalars; feed a dueling DDQN head. Avoid averaging away the player's
relation to hazards. Parameter count and latency are measured, not success gates.

Gate: on 128 decisive examples, at least 95% agreement with the set of tied-best
actions; on seed-disjoint probe validation, at least 25% less mean lookahead
regret than the normalized MLP or fixed-action reference used for that comparison,
with no increase in avoidable imminent deaths. These are proposed screening
thresholds, not evidence of 9,000-frame ability. If all inputs fail, revisit
observability/labels; if only tiny fitting passes, revisit generalization and
data coverage. Only a passing representation advances.

### P3 — short, interpretable DDQN experiments

Reuse replay, target network, native lanes, checkpoints, evaluation, dashboard
and colabctl. Initial pilots use 32 lanes, batch 128, one optimizer update per
collection update after warmup; preserve that ratio across comparisons.

Run this ordered ladder, promoting only measured gains:

| Experiment | Change | Question |
|---|---|---|
| A | Existing checkpoint/config and corrected instrumentation | Reproduce measured baseline |
| B | P1 controller with otherwise matched training config | Does usable steering improve survival? |
| C | P2 observation, same MLP | Does representation improve decisions? |
| D | P2 spatial encoder, same observation | Does spatial sharing improve sample efficiency? |
| E | Exploration schedule matched to pilot duration | Does an exploitation period improve greedy learning? |
| F | Normalize reward and lower learning rate if conditioning warrants | Do TD scales/clipping improve along with survival? |
| G | Frame-based discount, separately declared n-step experiment | Does longer credit improve outcomes? |

Each new stage compares against the last accepted stage. Reject/stop unhelpful
stages; this is a finite diagnostic ladder, not an HPO search. Pilot budget:
32,768 collection updates (~3.15 million maximum real native-frame advances
at 32 lanes), subject to a preregistered 30-minute CPU cap including finalization.
Measure throughput first. If the matched budget exceeds the cap, run a diagnostic
subset and optimize O1; do not label unequal-budget results a learning win.

For E, start epsilon at 1, reach 0.1 by 40% of the declared pilot budget, then
0.05 by 70%; record actual random fraction. Warmup must be counted independently.
For F, candidate survival reward is advanced-alive-frames/3 with no new corner
or large death penalty. Test LR=1e-4 only as a separate change or explicitly
declared conditioning bundle; do not silently overwrite the inherited value.

For G, declare discount in native-frame units. One interpretable comparator
preserves the old nominal timescale: per-three-frame gamma = 0.99^(3/32), about
0.999058. This is a candidate, not a proven optimum. Larger gamma increases
return magnitude, so monitor conditioning and use the selected reward scale.
Keep n=5 initially; compare longer returns only after stabilizing targets.
Target synchronization should count optimizer updates. Pure survival objective
does not require a new death penalty; true death already ends future reward.

Inspect checkpoints at 4,096-update intervals, with greedy evaluation on the
frozen inner training subset; separately budget durable recovery checkpoints.
Terminate non-finite runs immediately. Persistent clipping/action concentration
triggers diagnosis, not automatic rejection if survival and regret improve.

Gate: candidate beats fixed controls and accepted baseline on matched interaction
budgets; at least two of three learner seeds exceed 800 mean frames on all 70
training seeds, with median at least 800 and no P10 regression. Replays and
death classifications must support actual dodging. Promote by aggregate evidence,
not the best of three seeds.

### O1 — speed and memory

1. Profile actual N=16, hold=1, step_frames=3 workload with early and dense late
   snapshots. Separate native stepping, TTC, semantic painting, allocations,
   replay insertion/sampling, learning, evaluation, checkpoints, and residual
   wall time. Measure three warmed repetitions under recorded host load.
2. Cache immutable grid geometry/spawn labels, reuse result buffers, and avoid
   computing unused legacy ML features before replacing them with hazard fields.
   Profile clone cost and whole-game work inside each TTC query.
3. Optimize the native simulation/query boundary: reusable scratch state,
   exact early exits, and bounded lane/cell parallelism. Any omitted render,
   particle or other work must preserve state/RNG semantics that affect hazards.
4. A single shared future hazard rollout is **not** automatically equivalent to
   256 counterfactual games: player location can change enemy behavior, powerup
   pickup, collisions and future RNG consumption. Use shared trajectories only
   where independence is proven; otherwise exact fallback or a versioned
   approximation with measured false-negative collision rates is required.
5. After float/reference parity, pack bounded TTC to u16 with a reserved no-hit
   sentinel, binary fields to bits, categories to integers; decode before the
   network. Keep continuous features float32 until a quantization experiment
   validates error and decisions. Power-of-two ring capacity permits indexing
   masks; fractional cell spacing 7.6875 cannot be replaced by a pixel bit shift.
6. Consider deduplicated state/next-state storage and larger replay only after
   measuring diversity needs. Snapshot publication must remain atomic; exact
   learner/replay/RNG restoration is separate from explicitly restarted native
   lanes. Benchmark checkpoint copy/write costs and peak RSS as capacity grows.
7. CPU thread/lane tuning precedes GPU transfer/AMP/compile. A GPU learner does
   not accelerate Rust CPU TTC automatically. GPU simulation is a separate,
   higher-risk project requiring fixed-point/RNG/collision parity.

Gate: fixed-state action/frame/reward/done/position parity, reference TTC parity,
checkpoint/replay integrity, and at least 20% lower median all-in CPU time over
three matched repetitions without a memory or learning regression. Target at
least 2× all-in speedup for the combined work, but claim only measured results.
Set pilot peak RSS budget to 1 GiB and report any required revision explicitly.
Checkpoint/evaluation reductions are overhead changes, not kernel speedups.

### P4 — later-game learning and 9,000-frame confirmation

Advance through training-side 2,000- and 4,000-frame median survival milestones
before the final 9,000 gate. Retain a 12,000-frame evaluation cap and one life.
Report deaths by enemy/AOE/pattern, game phase and elapsed frames to identify
which new mechanics cause each plateau.

If later states remain absent, collect them with the P0 planner on training
seeds, preserving authentic canonical trajectories. Use balanced replay or an
explicit curriculum of real snapshots, and record snapshot sampling/weights.
Synthetic impossible states are unit tests, not evaluation episodes. Always
evaluate the complete natural-start episode. Extra lives alone restart early
game and do not solve late-game coverage.

If P2 learns useful labels but P3 fails to learn control, test fresh training-side
demonstration pretraining followed by DDQN and compare against scratch. Reuse
existing teacher/BC collection machinery, adapting its action/controller and
observation contracts; never feed legacy datasets into NG. Add recurrence only
if missing temporal state is measured; prioritized replay or distributional
heads remain later isolated experiments. No blanket Rainbow migration.

Only after local gates, prepare one colabctl fixed-configuration job with exact
source/wheel hashes, a projected wall budget from matched measurement, early
progress/recovery checks, and cleanup/artifact verification. Accelerator choice
follows existing T4-first availability rules. This plan launches no job.

Final gate: selected recipe meets the 9,000 criteria above on development
evaluation, then on the preregistered confirmation holdout. Publish all seed
outcomes, representative successful/typical/failed replays, feature/action
inspection, learning curves, death analysis, full timing/memory costs and source
identity. A failed confirmation remains a failed generalization result; do not
silently tune on it.

## Specification handoff before implementation

This document proposes behavior; it does not mark current SPEC tasks complete.
Root FORMAT.md is absent; use existing root section/pipe-table conventions when
the implementation updates the spec. Preserve unrelated work already in it.

- Amend G8/G18 and V33/V42/V127: retain 800 as an entry gate; add 9,000 success
  definition, cap above target, full distribution and repeatability requirements.
- Amend V39 and hazard replay contract: death versus external timeout, final
  observation bootstrap, no cross-reset n-step return, legacy version handling.
- Extend V37/V38/V73/V146: velocity-aware reference controller, target persistence,
  any masks applied consistently, observable controller state, fixed cadence.
- Extend V120–V126: preserve frozen-center reference; add optional trajectory
  fields, categorical/phase schema, horizon censorship, and packed decoder gates.
- Extend V41/V124/V147–V160: physical time units, actual optimizer/frame counts,
  probe gates, performance budgets, tested source/native provenance, and final
  confirmation split without historical holdout mislabeling.
- Reorder runtime/learning task dependencies: O1 may run alongside diagnostic
  learning; combined promotion still requires parity and learning evidence.
- Backprop entries to add with fixes: boundary recentering/arrival lifetime,
  unreachable target cap, inherited cadence changing discount meaning, and
  timeout-as-death semantics. Do not record unproven CNN/scaling hypotheses as
  established bugs. Add targeted regression tests with each invariant change.

Implementation routing: spec amendments and review of P0/P1 boundaries first;
build the open phase; backprop concrete defects; check code/spec before accepting
each phase. Performance-only refactors follow green parity gates. Later phases
remain proposals until their dependencies pass.

## Verification performed for this audit

- Loaded current best checkpoint and run config; parameter count checked.
- Recreated both greedy episodes using current native environment and declared
  controller; saved native action traces matched exactly; rewards matched replay
  survival exactly.
- Recomputed optimizer counts, epsilon, timing sums and clipping from all 16,384
  metric rows; inspected selected old HPO trial config and current source.
- `devenv -q shell -- uv run --extra native pytest -q tests/dodge/test_ng_hazard.py tests/dodge/test_ng_waypoint.py tests/dodge/test_ng_dqn.py`
  → 53 passed in 10.26 seconds. These existing tests do not establish the new
  future-AOE, settling or long-survival gates.
- Original-cartridge raster verification unavailable: `src/dodge/runtime/pemsa`
  is absent. No Colab operations or new training were performed.

Algorithm references: [Double DQN](https://arxiv.org/abs/1509.06461) supports the
online-selection/target-evaluation distinction used here;
[Rainbow](https://arxiv.org/abs/1710.02298) motivates testing improvements through
ablations rather than attributing gains to an untested bundle. Neither paper
establishes this game's achievable survival or runtime.
