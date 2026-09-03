# Dodge NG Phase 0/1/2/3 control spec

§G
G1|Build a fresh, reproducible NG experiment boundary and run native board- and pixel-observation PPO references.
G2|Measure learning on the 70% training partition and expose the untouched 30% partition only as a final generalization check.
G3|Produce machine-readable provenance, metrics, plots, and a human-readable baseline report.
G4|Close the board-PPO gate by separating action-credit/evaluator limits from learner-seed variance before Phase 2.
G5|Use a fresh native counterfactual planner to provide board-action supervision, then test whether BC and BC-to-PPO improve closed-loop learning on unseen NG seeds.
G6|Give each later learner the same frozen manifest, native interaction budget, and training-side selection boundary so improvements and failures transfer across methods.
G7|Train an exact-raster pixel CNN with temporal context and measure visual learning cost against the board reference.

§C
C1|NG sample space is new and finite: default 100 native-valid seeds, disjoint from every legacy seed used by the prior experiments.
C2|Split is exact 70/30 over the complete NG sample space; train and holdout sets are disjoint and their union is the sample space.
C3|Legacy GA/NEAT/BC/PPO data, checkpoints, databases, manifests, and scores are archival context only and cannot be NG inputs.
C4|Holdout seeds stay locked: training, checkpoint selection, and future HPO may use training-side seeds only.
C5|Use the existing native batch environment and board encoder for the hot path; preserve legacy PPO defaults and behavior when NG fields are unset.
C6|Phase 0/1 scope stops at manifest/provenance/evaluation/plots and the board PPO baseline; later phases add pixels, replay, DAgger, gradient actions, and HPO.
C7|Do not alter the native game contract or nested game/NEAT specs as part of this slice.
C8|Neutral bonus is a controlled ablation, not the assumed cause of policy collapse; diagnosis must also test action advantage and learner-seed variance.
C9|P2 teacher states, labels, BC validation, DAgger aggregation, and warm-start selection use only manifest training seeds; holdout remains final-report-only.
C10|Teacher scores simulate every action from the same canonical snapshot with one fixed native config/lookahead and must not mutate the live batch environment.
C11|A teacher dataset is fresh NG data with manifest/config/provenance metadata; legacy databases, checkpoints, and prior experiment artifacts are never inputs.
C12|The first pixel control receives only native indexed pixels with shape `(N,4,128,128)`, normalizes palette indexes by `15`, and uses no augmentation or derived board channels.
C13|Pixel PPO uses the same frozen NG manifest, nine-action contract, native frame schedule, and training-side selection boundary as board PPO; holdout remains report-only.

§I
I1|`src/dodge/ng/manifest.py` owns the immutable NG seed manifest, validation, hashing, and CLI generation.
I2|`src/dodge/ng/report.py` owns split metrics, trend summaries, plots, Markdown, and JSON report generation.
I3|`src/dodge/ng/train.py` owns the baseline CLI and wires one manifest into native PPO without reading legacy artifacts.
I4|`src/dodge/rl/ppo.py` accepts an optional explicit training-seed tuple and optional training-side evaluation seeds; absent values retain legacy behavior.
I5|`context/kits/dodge-ng/ng-v1.json` is the committed frozen manifest; run artifacts live under `history/dodge/ng/`.
I6|NG commands are `dodge-ng-manifest`, `dodge-ng-train`, and `dodge-ng-report`; the training recipe uses the devenv runtime boundary.
I7|`src/dodge/ng/diagnostics.py` owns fixed-action controls and action-advantage evidence on the frozen manifest.
I8|`native/crates/dodge-batch` owns deterministic all-action counterfactual scoring from canonical snapshots; `src/dodge/native/batch.py` exposes it without changing lane state.
I9|`src/dodge/ng/teacher.py` owns fresh manifest-scoped planner data, score margins, dataset validation, and learner-state collection hooks.
I10|`src/dodge/ng/bc.py` owns board BC training, training-seed inner selection, closed-loop evaluation, artifacts, and plots.
I11|`src/dodge/rl/ppo.py` accepts an actor-only warm start while reinitializing value/optimizer state; `src/dodge/ng/bc.py` owns the compatible bridge.
I12|`src/dodge/ng/compare.py` owns matched PPO selected-checkpoint evaluation, sample-efficiency deltas, and comparison plots without holdout selection.
I13|`src/dodge/ng/dagger.py` owns learner-visited training-state collection, fresh counterfactual labels, dataset aggregation, and round BC retraining.
I14|`src/dodge/rl/ppo.py` owns native pixel PPO, exact pixel frame stacking, reset-safe stack lifecycle, pixel checkpoints, and pixel evaluation.

§R
R1|Native batch API exposes board, pixels, hashes, snapshots, rewards, done flags, and deterministic reset/step results|`src/dodge/native/batch.py`
R2|Existing PPO uses the board CNN, clipped objective, GAE, AdamW, native lanes, checkpoints, and evaluation hooks|`src/dodge/rl/ppo.py`
R3|Clipped PPO is the Phase 1 baseline algorithm|https://arxiv.org/abs/1707.06347
R4|The native runtime must be verified through devenv for the project graphics/runtime closure|`devenv.nix`, `justfile`

§V
V1|A valid manifest has a deterministic version/hash, unique native-valid seeds, exact 70/30 cardinalities, disjoint partitions, and complete union.
V2|Every NG seed is outside the recorded legacy seed range; loading malformed, duplicated, overlapping, or legacy-containing manifests fails closed.
V3|A baseline config names the manifest hash and uses exactly the manifest training seeds; no default legacy seed stream may leak into NG training.
V4|Holdout seeds are passed only to final evaluation; inner checkpoint evaluation is a deterministic subset of training seeds.
V5|Explicit PPO seed streams are reproducible and emit only their configured candidate set; legacy default streams remain unchanged.
V6|Per-split evaluation reports finite mean, median, p10, worst, best, and horizon-completion statistics with per-seed outcomes.
V7|Reports contain training/inner/holdout comparison, train-minus-holdout gap, learning curves, diagnostic curves, provenance, and throughput when available.
V8|A successful baseline run writes valid `run.json`, `metrics.jsonl`, checkpoints, `report.json`, `REPORT.md`, and plot files under its run directory.
V9|Phase 0/1 tests run without legacy databases or checkpoints and cover manifest, seed routing, evaluator statistics, report generation, and CLI configuration.
V10|Every NG provenance/report JSON artifact is JSON-serializable after path and typed-config normalization.
V11|Fixed-action controls evaluate all nine action choices on both manifest partitions and report finite per-action survival distributions.
V12|P1 comparison runs use the same manifest, interaction budget, evaluation protocol, and architecture; only declared learner/configuration variables differ.
V13|The locked holdout is absent from all P1 diagnosis and selection decisions; it is reported only after training-side comparison is frozen.
V14|Native policy evaluation resets every lane reported done, including lanes already excluded from the measured episode, before the next batch step.
V15|When inner validation selects a best PPO checkpoint, `checkpoint-best.pt` preserves that model while `checkpoint-latest.pt` remains the final resumable state.
V16|Counterfactual scoring restores each supplied canonical snapshot independently, evaluates all nine actions for the fixed lookahead, returns finite deterministic scores, and leaves the source batch state unchanged.
V17|A valid teacher dataset contains only manifest training seeds, board tensors with the documented shape, one finite action/score/margin record per example, and manifest/config/lookahead provenance; legacy inputs are absent.
V18|Teacher/BC train and inner-validation subsets are disjoint by environment seed and selected before training; holdout seeds never enter data collection, checkpoint selection, or HPO.
V19|Repeated teacher scoring of the same snapshot/config is byte-for-byte or numerically identical, action ties/margins are explicit, and invalid snapshots/lookaheads fail closed.
V20|BC artifacts preserve model/action/board metadata, manifest hash, teacher-data hash, split seeds, normalization/config, per-epoch metrics, and the selected inner checkpoint.
V21|PPO warm start copies only compatible actor feature/policy weights from BC, resets value and optimizer state, records initialization provenance, and remains resumable under the matched PPO config.
V22|DAgger rounds append only learner-visited training-seed states with fresh teacher labels, record round/version provenance, and compare rounds only under the frozen BC/PPO evaluation protocol.
V23|Manifest-bound teacher loading identifies a held-out seed as a holdout violation before the broader non-training-seed violation, preserving an actionable provenance diagnosis.
V24|Saved teacher metadata counters and action histograms equal their serialized arrays; stale aggregate counters fail teacher loading.
V25|Pixel PPO accepts only native indexed-pixel batches with finite palette values in `0..15`, the configured stack/channel shape, and deterministic normalization.
V26|Pixel frame stacks repeat the first reset frame and replace all channels on lane reset; no frame from a prior episode is present in the first post-reset observation.
V27|Pixel checkpoints and run records identify the pixel model, observation mode, stack size, raster shape, and action contract; matching pixel resumes reproduce configuration validation.

§T
id|status|task|cites
T1|x|Implement `SeedManifest`, deterministic fresh default, validation/hash, committed NG manifest, and unit tests.|V1,V2,I1,I5
T2|x|Add explicit PPO training-seed routing and training-side checkpoint evaluation while preserving legacy defaults; add regression tests.|V3,V4,V5,I4
T3|x|Implement split evaluator, trend metrics, plot/report generation, and tests using synthetic run artifacts.|V6,V7,I2
T4|x|Add NG baseline CLI, package entry points, devenv-backed just recipe, provenance wiring, and CLI tests.|V3,V4,V8,V9,V10,I3,I6
T5|x|Run Phase 0 native smoke/throughput checks and freeze their evidence without touching legacy artifacts.|V8,V9,C1,C3,C5
T6|x|Run the Phase 1 native board PPO baseline on the frozen 70% partition, evaluate the locked 30%, and deliver the generated trend/performance report.|V4,V6,V7,V8,C2,C4,C6
T7|x|Implement fixed-action controls and action-advantage reporting across the frozen manifest.|V6,V11,I7
T8|x|Run two additional current-control learner seeds with the matched P1 budget and compare training-side curves.|V7,V12,V13
T9|x|Run matched neutral-bonus-off controls across three learner seeds and compare against the current control.|V7,V12,V13
T10|x|Freeze the P1 diagnosis and select the next intervention without using holdout results.|V7,V12,V13,G4
T11|x|Add native canonical-snapshot counterfactual scoring for all nine actions, Python exposure, validation errors, and serial/parallel/nonmutation/determinism tests.|V16,V19,I8,C10
T12|x|Collect fresh native planner demonstrations on training seeds, persist board/action/score/margin data with provenance, and validate legacy exclusion and seed routing.|V17,V18,V19,I9,C9,C11
T13|x|Train compatible board BC with training-side inner selection, closed-loop evaluation, and metrics/plots/artifacts.|V18,V20,I10
T14|x|Add actor-only BC-to-PPO initialization, run matched from-scratch/warm-start controls, and report whether sample efficiency/generalization improves.|V21,G5,G6,I11,I12
T15|x|Implement learner-state DAgger aggregation/retraining and compare rounds only if T14 supplies a viable teacher/learner baseline.|V22,G6,C9,I13
T16|x|Freeze the P2 method decision from training-side evidence, report final locked holdout comparisons, and select the next pixels/replay/gradient/HPO phase.|V7,V18,V20,V21,V22,G6
T17|~|Extend native PPO with indexed-pixel four-frame stacks, reset-safe lifecycle, pixel model/checkpoint metadata, CLI configuration, and regression tests.|V25,V26,V27,I14,C12,C13
T18|.|Run a matched native pixel-PPO control on the frozen training split, select only on inner training seeds, and report locked holdout performance and throughput.|V7,V18,V25,V27,G7
T19|.|Compare board and pixel controls by sample efficiency, wall-clock throughput, split gap, lower tail, and representative visual failures.|V7,V18,V25,V27,G6,G7

§B
id|date|cause|fix
B1|2026-09-03|NG provenance serialized `Path` fields from a dataclass directly to JSON.|Add an explicit JSON serializer for baseline configuration and enforce V10 with the runner test.
B2|2026-09-03|Native policy evaluation reset only newly completed lanes, so an already excluded lane could die again and remain done before the next batch step.|Reset every done lane and enforce V14 with a multi-lane evaluator regression test.
B3|2026-09-03|PPO saved the final model over `checkpoint-latest.pt` after recording a stronger inner-validation checkpoint, so the selected model was not preserved.|Write `checkpoint-best.pt`, keep latest as final, and enforce V15 with a two-update checkpoint test.
B4|2026-09-03|Teacher validation checked the generic non-training set before the specifically forbidden holdout set, hiding the actionable provenance cause.|Check holdout membership first and enforce V23 with the manifest-bound loader test.
B5|2026-09-03|The cache test counted duplicate positions in one request as cache hits even though they are deduplicated before the single computation.|Count repeated-call reuse as hits and verify the one-miss/one-hit accounting in the cache test.
B6|2026-09-03|New BC/warm-start tests exceeded repository import/line-width lint rules.|Format imports and signatures before next verification gate.
B7|2026-09-03|Comparison report called an omitted JSON writer and exceeded line width in its Markdown table.|Add atomic comparison JSON write and format table literal before next verification gate.
B8|2026-09-03|Comparison fixture expected holdout delta with incorrect hardcoded arithmetic.|Derive expected delta from fixture values and keep comparison arithmetic covered.
B9|2026-09-03|DAgger collector imported unused tensor alias after switching to module-qualified torch calls.|Remove unused import before next verification gate.
B10|2026-09-03|DAgger aggregation inherited base metadata counters after concatenating arrays, so aggregate metadata underreported examples.|Recompute examples, decisive count, and action histogram on every teacher save and validate V24.
B11|2026-09-03|The first full-suite run recorded the live NEAT checkpoint test as failed while an isolated devenv rerun passed in 248.93s; concurrent/stale Pemsa activity made the result transient rather than a P3 regression.|Keep P3 source scoped away from NEAT, rerun live Pemsa tests in isolation when the suite records this failure, and do not add a source workaround without deterministic reproduction.
B12|2026-09-03|Repository-wide `app-check` still reports five unrelated pre-existing lint violations in native fuzz tooling and the legacy `rl_learning` package; none overlap T17 files.|Use targeted clean lint for the T17 gate and defer unrelated lint cleanup to its owning legacy modules.
