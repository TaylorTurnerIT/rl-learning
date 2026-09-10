# Dodge NG Phase 0/1/2/3 control spec

§G
G1|Build a fresh, reproducible NG experiment boundary and run native board- and pixel-observation PPO references.
G2|Measure learning on the 70% training partition and expose the untouched 30% partition only as a final generalization check.
G3|Produce machine-readable provenance, metrics, plots, and a human-readable baseline report.
G4|Close the board-PPO gate by separating action-credit/evaluator limits from learner-seed variance before Phase 2.
G5|Use a fresh native counterfactual planner to provide board-action supervision, then test whether BC and BC-to-PPO improve closed-loop learning on unseen NG seeds.
G6|Give each later learner the same frozen manifest, native interaction budget, and training-side selection boundary so improvements and failures transfer across methods.
G7|Train an exact-raster pixel CNN with temporal context and measure visual learning cost against the board reference.
G8|Treat 800 native survival frames as the first true-learning threshold for the focused baseline campaign; do not call shorter survival progress learned.
G9|Test whether waypoint-discretized movement plus DQN improves action credit, sample efficiency, and 800-frame learning over direct nine-action PPO.
G10|Use predictive frozen-center hazard field as opt-in waypoint-DDQN observation; compare against legacy waypoint DQN under same NG protocol; defer gradient controller.
G11|Move waypoint ML observation extraction into the native batch runtime; preserve Python reference parity and expose uncapped survival beyond the 800-frame gate.
G12|Make every saved NG replay an original-cartridge pixel-regression sample; expose movement/corner safety controls for measured waypoint ablations.
G13|Measure isolated pixel-only DQN learning at exactly 200,000 updates with matched one-life and three-life runs.
G14|Keep rendered pixel-only collection free of unused snapshot/ML/board payloads while preserving exact game behavior.
G15|Submit optimized NG training as observable, resumable Colab GPU batch jobs; return verified artifacts locally and escalate hardware only when evidence requires it.
G16|Reduce full-loop pixel-DQN wall time through measured checkpoint/replay, host/device, learner, native-collection, and GPU-tier optimization while preserving learning and resume semantics.
G17|After runtime optimization, systematically audit every pixel-DQN input parameter and test declared learning/performance hypotheses without invalidating matched comparisons.
G18|Train opt-in fixed-N waypoint DDQN on native frozen-center TTC plus compact threat/spawn fields; reach 800 training survival frames before promotion.
G19|Maximize current native headless `dodge.ng.dqn` end-to-end throughput, measured as actual native game frames per all-in wall second; use CPU reductions as primary GPU-speed proxy.
G20|Make real Colab GPU training, not synthetic microbenchmarks, final optimization target; use occasional matched T4 checks and one representative full GPU run.
G21|Exhaust speedup surface in order: measure → profile → parity-preserving implementation → matched CPU comparison → real GPU gate → promotion; preserve learning quality, resume, and artifacts.

§C
C1|NG sample space is new and finite: default 100 native-valid seeds, disjoint from every legacy seed used by the prior experiments.
C2|Split is exact 70/30 over the complete NG sample space; train and holdout sets are disjoint and their union is the sample space.
C3|Legacy GA/NEAT/BC/PPO data, checkpoints, databases, manifests, and scores are archival context only and cannot be NG inputs.
C4|Holdout seeds stay locked: training, checkpoint selection, and future HPO may use training-side seeds only.
C5|Use the existing native batch environment and board encoder for the hot path; preserve legacy PPO defaults and behavior when NG fields are unset.
C6|Phase 0/1 scope stops at manifest/provenance/evaluation/plots and board PPO baseline; later phases add waypoint DQN, replay, DAgger, predictive hazard gradients, pixels, and HPO.
C7|Do not alter the native game contract or nested game/NEAT specs as part of this slice.
C8|Neutral bonus is a controlled ablation, not the assumed cause of policy collapse; diagnosis must also test action advantage and learner-seed variance.
C9|P2 teacher states, labels, BC validation, DAgger aggregation, and warm-start selection use only manifest training seeds; holdout remains final-report-only.
C10|Teacher scores simulate every action from the same canonical snapshot with one fixed native config/lookahead and must not mutate the live batch environment.
C11|A teacher dataset is fresh NG data with manifest/config/provenance metadata; legacy databases, checkpoints, and prior experiment artifacts are never inputs.
C12|The first pixel control receives only native indexed pixels with shape `(N,4,128,128)`, normalizes palette indexes by `15`, and uses no augmentation or derived board channels.
C13|Pixel PPO uses the same frozen NG manifest, nine-action contract, native frame schedule, and training-side selection boundary as board PPO; holdout remains report-only.
C14|The focused baseline target is 800 survival frames, equal to 200 native decisions at `step_frames=4`; target completion is evaluated on the training split and holdout is report-only.
C15|Waypoint actions never teleport, alter native physics, bypass collision, or emit controls outside existing nine-action contract.
C16|Waypoint candidate resolutions start at 8, 16, and 32 pixels; valid centers respect native player-center bounds and resolve wall targets deterministically.
C17|Waypoint transitions use fixed native frame cadence and explicit target reached, timeout, terminal, and reset semantics; replay never crosses episode boundaries.
C18|Waypoint oracle feasibility may use canonical full state; learned DQN input contains only declared observation channels and no hidden simulator state.
C19|Waypoint/DQN training, inner selection, and HPO use training seeds only; holdout remains final report-only under same 800-frame gate.
C20|Hazard-field labels predict future collision risk for fixed player-center cells at declared horizons; instantaneous occupancy alone cannot define labels or pass method comparison.
C21|Native ML observation batches may omit canonical snapshots; legacy full-state, board, pixel, and snapshot APIs remain unchanged.
C22|Dashboard connects through run-directory files/WebSocket; trainer never waits on dashboard clients.
C23|Replay capture runs from saved checkpoint in separate process; DQN hot path stays render-free ML path.
C24|Training lives are a DQN collection wrapper only: evaluation remains one-life, and the native game contract is unchanged.
C25|The overnight campaign inherits the selected HPO parameters, changes only declared lives/batch/budget controls, and uses training-only interim evaluation.
C26|Stopping may happen through the dashboard mailbox, SIGINT, or SIGTERM; the trainer must finish the current safe boundary and write a resumable latest checkpoint.
C27|Replay regression runs original `src/dodge/game/dodge.p8` through Pemsa/Xvfb under same seed, startup boundary, action trace, and frame cadence; unavailable or mismatched oracle cannot pass.
C28|Movement interventions remain opt-in, native-physics/action-space preserving, checkpoint-provenanced, and comparable against legacy defaults.
C29|Pixel ablation keeps NG manifest, learner seed, HPO values, action/controller contract, cadence, budget, and death penalty fixed; only `training_lives` differs; waypoint artifacts untouched.
C30|Fast pixel boundary remains opt-in until parity/performance gates pass; current 200k life ablation and existing artifacts remain on legacy boundary.
C31|Batch control plane uses pinned `colabctl[cli]` over official Google `colab` CLI; ADC auth is required for live allocation/remote inspection; `colab-mcp` remains optional browser bridge, not job queue.
C32|GPU policy starts at T4, advances only through declared ordered tiers, and never silently requests hardware outside job manifest or configured maximum.
C33|`colabctl` owns durable remote job state, detached supervision, polling, logs, and cancellation; local adapter retains small job/status/log/artifact records; runtime release follows verified retrieval or explicit stop.
C34|Remote source and native wheel artifacts derive from current repository snapshot; legacy histories/checkpoints stay excluded unless explicit resume input names them.
C35|Returned archive includes verified run files, reports, metrics, checkpoints, and replay snapshots; raw pixel frame stores remain opt-in; resume is explicit and no automatic lane restart is promised.
C36|Optimization pilots use fresh run directories and matched manifest, learner seed, model, action/controller, evaluation, and update budget; prior runs remain untouched.
C37|Performance claims require reconciled wall-clock phase timings and matched behavior/learning evidence; GPU occupancy alone cannot select a change.
C38|Replay/checkpoint format changes preserve current restore semantics, retain explicit legacy-format support or migration, and never replace a valid snapshot with incomplete data.
C39|CUDA, pinned-transfer, prefetch, compile, AMP, and native multi-step paths remain opt-in until correctness and matched-performance gates pass; CPU path and defaults remain available.
C40|Colab GPU selection starts at T4 and advances only after typed `colabctl` accelerator-unavailable errors; auth, quota, parser, and transport failures remain fail-closed.
C41|Parameter review uses fresh runs, fixed manifest/splits, fixed learner seed where possible, one declared factor or interaction per comparison, and training-side selection; holdout remains report-only.
C42|Hazard DDQN remains opt-in sibling; legacy waypoint DQN/checkpoints remain reproducible; matched runs vary only declared observation/config factors.
C43|Hazard waypoint grid uses fixed per-checkpoint N×N; cell i center = min_center + (i+0.5)(max_center-min_center)/N; native player-center bounds remain authoritative.
C44|Hazard TTC fixes hypothetical player at each cell center with zero velocity; H=32 native frames initially; no-hit within H → ∞ semantic.
C45|TTC precision one native frame; model input finite; packed u16 max sentinel may represent ∞; float reference remains authority.
C46|Hazard observation includes TTC plus declared threat type/velocity/occupancy/pattern channels; four-bit exact spawn-corner mask plus static configurable halo; halo never hard-masks actions.
C47|No gradient controller; DDQN owns waypoint action; legacy corner ban/penalty excluded from hazard mode; dynamic halo deferred.
C48|Float/reference hazard path verified before bit-shift/packed path; packed promotion requires value/action parity and benchmark.
C49|Canonical target = current native headless `dodge.ng.dqn` workload; record exact argv/config, source/native hashes, manifest, learner seed, and hardware; active run remains reference-only.
C50|Active remote job receives ⊥ control/replay/stop or remote writes; local inspection remains read-only; no overlapping GPU trial until terminal unless explicitly authorized.
C51|CPU speedup drives prioritization as GPU proxy; GPU effect remains unverified until matched real Colab check.
C52|Microbenchmarks locate bottlenecks only; candidate acceptance requires representative end-to-end headless run.
C53|Timed trials use fresh bounded dirs, explicit CPU affinity/thread/priority, warm-up/repetitions, and ambient-load record; contention samples discarded or labeled.
C54|Baseline/candidate differ by one runtime factor or declared interaction; manifest, seed, model, action/controller, observation/cadence, interaction/update budget, eval, and checkpoint policy fixed unless named.
C55|Reports separate collection, native observation/hazard, Python boundary, replay, transfer, learner, evaluation, checkpoint, and all-in wall; cadence reductions never masquerade as kernel speed.
C56|Real GPU checks run at optimization phase boundaries and after CUDA/transfer/compile/AMP/native-wheel changes; T4 first; no Colab allocation per microtrial.
C57|Real GPU evidence requires actual Colab device, current packaged source/native wheel, matched config, remote phase telemetry, and verified artifacts; CPU/synthetic CUDA insufficient.
C58|Speedup promotion requires action/frame/reward/done/position parity, checkpoint/resume/replay validity, and relevant learning gate; speed-only wins rejected.
C59|Holdout locked and never used for performance/parameter selection; training-side selection only.
C60|Optimization defaults remain behavior-compatible; schedule/parameter changes reported separately and reviewed after runtime track.

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
I15|`src/dodge/ng/relevance.py` owns multi-horizon decision-relevance audits, policy regret/action dynamics, gate output, and visual checkpoint references.
I16|`src/dodge/rl/ppo.py` owns lane-independent GAE for the interleaved native batch rollout and records the configured baseline target in NG provenance.
I17|Native board encoders and PPO observation-mode routing own the opt-in off-screen-preserving `board_full` representation; the Python reference encoder mirrors it for parity tests.
I18|`src/dodge/rl/ppo.py` owns explicit board spatial-pooling configuration and reconstructs the selected pooling mode for PPO checkpoints and relevance audits.
I19|Native full-coordinate board encoding owns canonical off-screen hazard position channels while legacy `board` and edge-pinned `board_full` remain unchanged.
I20|`src/dodge/ng/waypoint.py` owns grid geometry, valid-center generation, waypoint action encoding, bounded native steering, and feasibility metrics.
I21|`src/dodge/ng/dqn.py` owns waypoint DQN observations, replay, target updates, checkpoints, evaluation, and run provenance.
I22|`src/dodge/ng/hazard.py` owns fixed-center TTC labels, hazard observation encoding, spawn labels/halo, and matched comparison artifacts; no action controller.
I23|NG waypoint commands are `dodge-ng-waypoint-feasibility` and `dodge-ng-dqn`; hazard-field command is `dodge-ng-hazard`.
I24|`native/crates/dodge-batch/src/ml.rs` owns native waypoint ML feature extraction; `NativeBatchEnv` optionally returns `ml_observation` with shape `(N,225)` and player coordinates.
I25|`src/dodge/ng/telemetry.py` owns bounded latest-status publication and file control mailbox.
I26|`src/dodge/ng/dashboard.py` owns HTTP/WebSocket dashboard, controls, run inspection, and replay launch.
I27|`src/dodge/ng/replay.py` owns deterministic checkpoint-to-pixel replay artifacts.
I28|NG dashboard command is `dodge-ng-dashboard`.
I29|`src/dodge/ng/overnight.py` owns the reproducible long-run DQN preset, HPO parameter inheritance, CLI overrides, and training-only evaluation defaults.
I30|`src/dodge/ng/pixel_regression.py` owns replay action traces, original-cartridge execution, exact indexed-pixel comparison, and regression metadata.
I31|`src/dodge/ng/waypoint.py` and `DQNConfig` expose grid spacing, decision interval, arrival tolerance/latching, corner policy, and optional corner safety penalty.
I32|`src/dodge/ng/pixel_dqn.py` owns pixel-only DQN model, uint8 frame-stack replay, life ablation runner, checkpoints, reports, and matched-run comparison.
I33|`NativeBatchEnvironment.step_pixels` and `dodge_native.NativeBatchEnv.step_pixels` own minimal rendered pixel results; `PixelDQNConfig.native_pixel_boundary` selects legacy/fast collection.
I34|`src/dodge/ng/colabctl_batch.py` owns source/native-wheel/resume packaging, payload identity, local records, GPU ladder, artifact verification, and the NG CLI; `colabctl` owns runtime lifecycle.
I35|`dodge-ng-colab` owns batch `plan`, `submit`, `status`, `watch`, `stop`, and `retrieve` commands; trainer arguments remain explicit payload data.
I36|`history/dodge/ng/colab-jobs/<job-id>/` stores compact `job.json`, `status.json`, bounded `colabctl.log`, downloaded archive, and artifact receipt.
I37|Remote root `/content/.colabctl/jobs/<job-id>/` is owned by colabctl; generated bootstrap adds extracted source, native wheel, run directory, and finalized artifact archive.
I38|`colabctl` `DetachedColabBackend` over `ColabCliTransport` owns allocation, detached runner, status/log/result/cancel, reattachment, and state store; generated bootstrap owns trainer/artifact work.
I39|`src/dodge/ng/pixel_dqn.py` owns pixel replay serialization, phase timings, optional profiler, host/device transfer modes, learner optimization flags, and checkpoint provenance.
I40|`history/dodge/ng/<run>/profile/` owns bounded optional CPU/CUDA trace and summary artifacts; absent unless profiling explicitly enabled.
I41|`native/crates/dodge-batch` and `src/dodge/native/batch.py` own optional batched pixel microsteps and lane-count/execution benchmarks; single-step API remains stable.
I42|`src/dodge/ng/parameter_review.py` owns parameter inventory, factor classification, matched trial manifests, training-side summaries, and promotion/report artifacts.
I43|`src/dodge/ng/hazard.py` owns fixed-center TTC reference encoding, semantic threat channels, spawn-corner labels/halo, and matched comparison artifacts; no action controller.
I44|`native/crates/dodge-core` owns canonical fixed-center TTC simulation; `native/crates/dodge-batch` and `src/dodge/native/batch.py` own serial/parallel exposure.
I45|`src/dodge/ng/dqn.py` owns opt-in fixed-N hazard observation wiring, replay/checkpoint contract, and waypoint-DDQN evaluation; legacy mode remains unchanged.
I46|`src/dodge/ng/colab_batch.py` and `src/dodge/ng/colab_worker.py` remain legacy-only Colab implementation surfaces during migration.
I47|`src/dodge/ng/benchmark.py` owns immutable benchmark manifest, bounded runner, repetition timing, counters, phase aggregation, comparison, and report.
I48|`dodge-ng-benchmark` owns local native headless CPU/microbench/profile commands; it never controls Colab jobs.
I49|`history/dodge/ng/benchmarks/<benchmark-id>/` stores raw repetitions, machine/toolchain/config identity, profile artifacts, `report.json`, and `REPORT.md`.
I50|`src/dodge/ng/dqn.py` and `src/dodge/ng/pixel_dqn.py` emit common throughput/phase schema; instrumentation disabled → existing default behavior.
I51|`src/dodge/ng/parameter_review.py` owns optimization candidate ledger, factor classification, promotion/rejection decisions, and final speed/learning report.
I52|`dodge-ng-colab` submits benchmark/training payloads with same source/native-wheel identity and artifact contract; `colabctl` remains lifecycle owner.

§R
R1|Native batch API exposes board, pixels, hashes, snapshots, rewards, done flags, and deterministic reset/step results|`src/dodge/native/batch.py`
R2|Existing PPO uses the board CNN, clipped objective, GAE, AdamW, native lanes, checkpoints, and evaluation hooks|`src/dodge/rl/ppo.py`
R3|Clipped PPO is the Phase 1 baseline algorithm|https://arxiv.org/abs/1707.06347
R4|The native runtime must be verified through devenv for the project graphics/runtime closure|`devenv.nix`, `justfile`
R5|`colab-mcp` bridges local MCP clients to a Colab browser session; README defines no durable batch queue|https://github.com/googlecolab/colab-mcp
R6|`colab run` combines fresh allocation, execution, exit propagation, and auto-stop unless `--keep`|https://raw.githubusercontent.com/googlecolab/google-colab-cli/main/docs/05_run_command.md
R7|Named `colab` sessions support `new`, `exec`, `status`, `stop`, and GPU variants including T4/L4/G4/A100/H100|https://raw.githubusercontent.com/googlecolab/google-colab-cli/main/docs/01_session_management.md
R8|Colab CLI operator guidance treats sessions as billable until explicit stop; keep-alive has finite lifetime|https://raw.githubusercontent.com/googlecolab/google-colab-cli/main/skills/colab-operator/SKILL.md
R9|`colabctl` 0.5.0 provides durable detached jobs, a CLI extra bundling Google Colab CLI, and SDK/backend control; package status is alpha|https://pypi.org/project/colabctl/

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
V28|Decision-relevance audit replays canonical snapshots with common-random-number state, scores all nine actions at configured horizons, remains finite/deterministic, and leaves live batch state unchanged.
V29|Audit reports first enemy/active-pattern/near-collision/decisive frames, per-horizon action margins, policy regret, action-change rate, and maximum action run.
V30|Gate marks rollout uninformative when no decisive state exists or all-action margin ≤ configured threshold; uninformative scores cannot advance learner selection.
V31|Audit selection inputs use training seeds only; holdout audit optional report-only and cannot set pass/selection fields.
V32|Native PPO computes GAE independently along each lane's time axis; interleaved lane transitions cannot contribute future return or advantage across lane boundaries.
V33|A focused baseline run declares target completion only when its configured training evaluation reaches 800 survival frames; shorter runs remain diagnostic progress regardless of holdout performance.
V34|Opt-in `board_full` pins entirely off-screen canonical hazards to the nearest 19x16 edge cells, leaves legacy `board` unchanged, and agrees across native serial/parallel and Python reference encoders.
V35|Board pooling is explicit in PPO configuration/checkpoint provenance; average pooling remains the default and max pooling is an isolated board-trial option that preserves sparse hazard evidence without changing observation shape.
V36|Opt-in `board_full_coords` preserves normalized canonical x/y for enemy and AOE entities in four extra channels, pins their cells only for indexing, and agrees across native/Python/serial/parallel; legacy board shapes and values remain unchanged.
V37|∀ waypoint resolutions → candidate centers stay within native player-center bounds; target quantization, clamping, and wall handling deterministic; no teleport.
V38|Waypoint controller emits only existing native `ACTION_CHOICES`; same state, target, config, and cadence → same native action sequence and unchanged game hashes.
V39|Waypoint transition records target/action/reward/next-state plus terminated/truncated boundary; replay sampling never bootstraps across terminal or reset.
V40|Oracle feasibility evaluates every candidate waypoint resolution across complete training partition before resolution selection; holdout cannot influence choice.
V41|DQN checkpoints identify manifest hash, grid resolution, observation shape, controller/cadence, replay/target configuration, and nine-action contract; matching resumes validate metadata.
V42|DQN selection/evaluation preserves exact 70/30 split and 800-frame gate; holdout scores remain report-only.
V43|Hazard-field labels encode future collision risk at each declared horizon; controller considers only reachable candidate moves and records label/controller provenance.
V44|Waypoint runners decode canonical snapshot bytes before typed-state conversion; invalid or missing snapshots fail closed.
V45|Waypoint batch evaluation resets every native-done lane before next step, including inactive dummy lanes; measured lane survival remains unchanged by dummy resets.
V46|Fast waypoint steering reads player coordinates from validated canonical snapshot prefix; coordinates equal full decode and invalid prefix fails closed.
V47|Native waypoint ML features equal `encode_waypoint_observation` after canonical state conversion across enemies, AOEs, patterns, ordering ties, overflow, and grid spacing.
V48|`ml_observation` returns finite `(N,225)` `float32` rows; enabling it does not alter state, pixel, board, reward, done, hash, or snapshot results.
V49|DQN hot collection uses native ML observations and native player coordinates without full Python snapshot decoding; the reference encoder remains callable for parity tests.
V50|800 frames remains a relevance gate only; DQN training/evaluation horizons may exceed 800 and report uncapped survival until terminal or an explicit safety limit.
V51|Waypoint grid geometry is constructed once per grid; allocation-free scalar steering emits the same native action as full `WaypointDecision` steering across positions, targets, tolerances, and action choices.
V52|Native frame capture may avoid redundant intermediate snapshot construction only when it preserves the exact render-boundary framebuffer, render state, logical state, hashes, and canonical bytes.
V53|The explicit ML-only native path shares the canonical simulation update and preserves logical state, RNG, frame/mode, reward, terminal flags, player positions, and feature vectors for identical seed/action traces while omitting render, hashes, and snapshots by contract.
V54|Fast ML batch results contain ordered finite `float32` `(N,225)` observations and `(N,2)` player positions plus `uint32` frames/seeds, `float32` rewards, and boolean terminal flags.
V55|Python batch payload parsers return their declared full or ML result type; fast ML results expose no snapshot-only fields.
V56|Telemetry publication to absent/slow/disconnected dashboard never blocks DQN collection or learning; stale updates may be dropped.
V57|Status/control/replay metadata writes use atomic file replacement; dashboard never reads partial artifacts.
V58|Dashboard controls accept only `pause`, `resume`, `stop`; command delivery remains optional and trainer continues without dashboard.
V59|Replay uses saved checkpoint and seed in independent native environment; live training lanes remain unchanged.
V60|Replay pixel frames use fixed 128x128 palette-index contract and metadata identifies checkpoint/config/seed/frame cadence.
V61|Dashboard path resolution stays inside selected run directory; arbitrary file serving forbidden.
V62|A non-final training death adds the configured negative life-loss penalty, emits a nonterminal replay transition to the same-seed reset observation, and consumes no new training seed; the final death remains terminal and advances to a new seed.
V63|Training lives never enter one-life evaluation: survival-frame metrics, the frozen 70/30 split, and holdout report-only boundary remain comparable to prior DQN runs.
V64|Life counters, life-loss counts, and penalized rewards are finite, deterministic, checkpoint-provenanced, and reset consistently at non-final death, final death, and truncation.
V65|The overnight preset records its inherited HPO values, lives, penalty, batch size, budget, manifest hash, and evaluation mode; `--resume` accepts only a matching checkpoint contract.
V66|Dashboard stop, SIGINT, and SIGTERM request graceful termination; after the current collection/learning boundary the run writes `checkpoint-latest.pt`, final run metadata, and a stopped status.
V67|Representative replay roles use one checkpoint and manifest training-seed outcomes only; best=max survival, bad=min survival, mean=closest to training mean, ties→lowest seed; holdout cannot influence roles.
V68|Dashboard exposes valid best/mean/bad roles with existing path-safe 128x128 palette replays and finite seed/survival metadata; missing roles do not break ordinary replay listing.
V69|Dashboard metrics inspection reads bounded tail; large metrics files cannot block HTTP/WebSocket connection.
V70|Every saved DQN replay records complete native action/frame trace and compares each saved 128x128 palette frame against original-cartridge Pemsa output at same game frame; pass requires zero differing pixels.
V71|Replay oracle provenance names seed, manifest/checkpoint/config, source/Pemsa identities, startup mode, cadence, compared count, and first mismatch coordinates/values; missing oracle or trace alignment never reports pass.
V72|Replay regression uses isolated original execution and cannot mutate live training lanes, source cartridge, or Pemsa binary; replay metadata appears only after atomic frame/metadata finalization.
V73|Arrival-latched waypoint control emits neutral after target enters configured tolerance until next decision boundary; unlatching preserves prior steering behavior; all actions stay in native nine-action contract.
V74|Corner-ban policy maps any would-be corner target to current cell without teleport/physics bypass; grid spacing and decision interval remain explicit in config/checkpoint/replay provenance.
V75|Optional corner safety penalty is finite and non-positive, applies only to selected corner targets, and remains zero by default; training/evaluation/replay use same control settings.
V76|Pixel DQN forward input contains only validated native indexed pixels with exact `(N,stack,128,128)` shape; player coordinates remain action-controller plumbing and never model features.
V77|Matched pixel life runs share manifest hash, learner seed, pixel architecture, HPO, action/controller contract, cadence, and 200,000-update budget; `training_lives` remains sole declared difference.
V78|Pixel replay stores palette frames as `uint8`, n-step references never cross terminal/truncated/reset boundaries, checkpoints preserve frame-store/progress contracts, and final evaluation uses one life on both splits.
V79|Pixel run provenance identifies observation mode/source, stack/raster, model/action contract, life settings, exact update count, native steps, and train/holdout results; comparison rejects mismatched runs.
V80|For identical seed/action traces, fast `step_pixels` frame/pixel/reward/done/position outputs equal canonical `step_batch`; fast result carries no snapshot/ML/board payload.
V81|Legacy pixel checkpoints missing `native_pixel_boundary` normalize to `legacy`; fast boundary requires explicit matching config; life comparison treats missing legacy field as `legacy`.
V82|Every Colab job manifest records immutable job id, source/file hashes, manifest hash, trainer argv, requested/attempted GPU tier, session name, and local artifact root before allocation.
V83|Job state transitions are monotonic and explicit: `planned`→`provisioning`→`running`→`stopping`→`completed`/`failed`/`cancelled`; malformed or contradictory remote state fails closed.
V84|Remote heartbeat contains finite timestamp, worker pid/status, trainer state, update/native-step counters when available, and last error; stale heartbeat cannot be reported as running.
V85|GPU escalation tries only declared tiers in order, records each allocation result and reason, stops failed sessions, and never escalates after successful completion or without configured budget/trigger.
V86|Artifact retrieval uses allowlist and SHA-256 inventory; default includes immutable `replay-*.u8.gz`, excludes `.pixel-frames.u8`, temporary files, unrelated history; raw map opt-in.
V87|Completion/stop → verify artifacts before owned-session cleanup; prelaunch failure → owned-session cleanup; launched-job transfer/observation failure → retain session with explicit recovery-required status. Record cleanup/retention without erasing failure evidence.
V88|Remote worker refuses to train until CUDA/device, native extension, project manifest, and optimized `fast` pixel boundary checks pass; dependency failures become observable job failures.
V89|A retrieved artifact set is accepted only when archive/file hashes and job/source/manifest identities match its local manifest; partial retrieval remains incomplete, never successful.
V90|Every pixel life loss terminates/flushes replay even when outer seed reuse has lives remaining; default `freeze-lanes` stops completed lanes until macro boundary; legacy continuation explicit checkpoint provenance.
V91|Pixel final train/holdout evaluation uses the best inner-training checkpoint when available; resuming marks prior terminal reports stale until the next safe finalization.
V92|Pixel telemetry records optimizer/sample counts, update/sample replay ratios, game frames/throughput, collection/learning time, action exploration/entropy/margins, sampled terminal/reward/n-step statistics, and clipping incidence.
V93|Prepared Colab payload identity binds immutable job/session ownership, trainer argv, manifest, source bundle, and wheel hashes; remote status/artifacts must match that identity and authenticated returned archive hash before atomic installation.
V94|New pixel checkpoint → immutable compressed replay snapshot with raw hash/size; live ring overwrite cannot change saved pixels; failed checkpoint publication preserves prior checkpoint/snapshot; restore rejects corrupt snapshot before replacing map.
V95|Download/verification failure after worker launch → recoverable `stopping`, retain owned session; retry verifies atomically installed receipt/files before reusing results; cleanup only after verified return.
V96|Stop command/SIGINT/SIGTERM → durable local request; only leased supervisor mutates lifecycle state; prelaunch request → cancel without training; remote stop sent only after matching worker identity.
V97|Native active-pixel stepping advances only selected lanes, preserves original lane ids, matches independent-game pixels/frames in serial/parallel; other lane death cannot shorten surviving lane macro duration.
V98|Session-name collision → no allocation, no stop; cleanup requires durable evidence this job started allocation after absence check.
V99|Running Colab job periodically exports verified latest checkpoint, metrics prefix, and referenced immutable replay snapshot outside runtime; status records exported step; corrupt/incomplete export cannot replace prior recovery copy.
V100|Fresh Colab job may resume from minimal local checkpoint/metrics/snapshot bundle; bundle hash bound into job identity; trainer config/manifest checks remain authoritative; native lanes restart explicitly.
V101|Worker-finalized artifact archive immutable after publication; fallback packing writes separate path; failed fallback cannot corrupt authoritative worker archive.
V102|Pixel perception probe uses only manifest training seeds, keeps its validation seed subset disjoint, requires a tiny-set fit diagnostic, and records action regret alongside accuracy before pixel-DQN selection.
V103|Recovery export before the first checkpoint → matching-identity no-op; it cannot stop a valid job or replace the latest verified recovery copy.
V104|Evaluation boundary → `checkpoint-latest.pt` is published before the evaluator starts, so runtime loss during a long evaluation leaves a resumable boundary at that update.
V105|Recovery polling compares the immutable checkpoint/replay identity; metrics-only changes do not trigger another large transfer, but a changed same-step checkpoint is never skipped.
V106|Colab allocation errors with explicit capacity signals, including `TooManyAssignmentsError`/HTTP 412, trigger only ordered GPU escalation; authentication, identity, and other non-capacity failures remain fail-closed.
V107|Every profiled training boundary separates collection, replay materialization, host-to-device transfer, learner compute, evaluation, checkpoint, and recovery durations; reported phases reconcile with wall time within documented measurement tolerance.
V108|Latest/best checkpoint publication preserves model, optimizer, RNG, replay, and provenance state; failed or partial new snapshot publication leaves prior valid checkpoint/replay intact and matching resume reproduces the saved boundary.
V109|Incremental replay snapshots store sufficient changed/required frame chunks plus hash/size/index metadata to reconstruct the exact frame ring; restore rejects missing, corrupt, overlapping, or misordered chunks before replacing the live map, while `gzip-u8-v1` remains readable.
V110|Opt-in CPU/CUDA profiling is bounded and writes trace/summary artifacts without changing actions, RNG, gradients, checkpoint order, or default runtime behavior; CPU/no-profiler path remains equivalent.
V111|Pinned/non-blocking learner transfers use contiguous host buffers and explicit lifetime/synchronization ownership; CPU fallback and synchronous CUDA mode remain numerically valid and loadable.
V112|Replay prefetch preserves declared RNG/sample order across checkpoint/resume, bounds queued memory, and returns samples byte-equivalent to synchronous sampling for identical state.
V113|AMP, compile, and larger-batch variants record warmup/compile cost separately, reject non-finite loss/Q/gradients, preserve checkpoint contracts, and cannot become selected defaults without matched learning evidence.
V114|Native multi-microstep pixel results preserve ordered lane ids, frames, pixels, rewards, done flags, and positions exactly equal to repeated single-step calls; inactive lanes remain excluded.
V115|Lane-count and serial/parallel benchmarks isolate collection throughput, retain exact game semantics, and record selected runtime settings in provenance; no result may imply learner speedup from collection-only data.
V116|Hardware escalation changes only declared Colab tiers after profiling evidence, retains T4-first policy, and records unavailable/entitlement outcomes without treating them as successful training.
V117|Parameter review inventories every exposed pixel-DQN input, records default/type/range/dependency/semantic owner, keeps comparison factors explicit, and promotes only variants with matched training-side evidence and unchanged holdout lock.
V118|Every public Just recipe for pixel-DQN resolves to an installed devenv wrapper or explicit `uv run --extra native` invocation; recipe smoke reaches CLI help.
V119|∀ hazard grid N≥1 → N×N cells; cell centers lie within native player-center bounds, scale deterministically, and nearest-cell ties resolve deterministically.
V120|Reference TTC fixes hypothetical player at each cell center with zero velocity, advances canonical native physics for H=32 native frames, records first death frame, and returns no-hit as explicit infinity semantics.
V121|Identical canonical state/grid/horizon → reference TTC and semantic field values deterministic; serial/parallel outputs agree; live lanes remain unchanged.
V122|Hazard observation contains finite TTC model values plus declared threat type/velocity/occupancy/pattern channels; raw float reference remains available for parity.
V123|Spawn-corner mask marks four exact spawn cells with deterministic bits; static halo radius maps deterministically for every N; labels never hard-mask actions.
V124|Hazard DDQN checkpoint identifies fixed N, H, field/channel schema, encoding, waypoint controller/cadence, manifest, replay/target config, and nine-action contract; legacy checkpoints remain loadable only under legacy mode.
V125|Hazard DDQN owns waypoint action selection; no gradient controller or legacy corner ban/penalty participates in hazard mode; same state/config → same greedy action.
V126|Packed TTC/bitfield/fixed-point decoder equals reference within declared quantization tolerance and produces identical greedy actions on parity corpus; packed path remains opt-in until gate passes.
V127|Matched hazard-vs-legacy trials keep manifest, learner, interaction budget, controller/cadence, evaluation, and declared HPO factors aligned; selection uses training seeds and 800-frame gate; holdout report-only.
V128|∀ ML reset with scripted startup → returned frame/survival baselines equal current native state; first learner step reports only post-reset deltas.
V129|Hazard startup N → up-target uses same centered cell-center geometry as DQN waypoint grid; legacy startup remains spacing-based.
V130|Artifact extraction keeps destination paths distinct from copy handles, installs verified members atomically, and repeated retrieval is accepted only for a matching receipt hash; unrelated output is never overwritten.
V131|NG Colab submission uses `colabctl` CLI transport without native opt-in; local identity binds manifest/source/wheel/payload hashes, and artifact receipt verification precedes installation.
V132|Typed colabctl `TooManyAssignmentsError` records capacity and advances to next declared GPU tier; auth, quota, parser, transport, and untyped allocation errors remain fail-closed.
V133|Project Colab CLI environment exposes `jupyter_kernel_client.KernelClient` and `JupyterSubprotocol` before runtime allocation; the compatible CLI/fork pair remains pinned and launch preflight fails before GPU reservation when either is absent.
V134|A post-allocation detached-launch failure releases only the owned named session before propagating; capacity and unavailable allocation failures never stop unrelated sessions.
V135|Lifecycle subcommands parse and dispatch without submission-only trainer arguments; only `submit` consumes the remainder argument list.
V136|Remote bootstrap validates the manifest's canonical semantic hash against both its declared field and the submission identity; raw JSON formatting bytes are not confused with the semantic hash.
V137|Payload packaging preserves the native wheel filename; remote bootstrap discovers exactly one `.whl` member, verifies its bytes, and installs that validly named wheel.
V138|Remote bootstrap maps the preserved repository `src/` layout onto `PYTHONPATH` before launching the trainer; the packaged `dodge` module is importable from the detached subprocess.
V139|Read-only dashboard mode hides training controls while retaining status/run selection/replay access; control messages remain rejected unless controls are explicitly enabled.
V140|Compact ML stepping may omit feature vectors between decision boundaries; terminal transitions retain original reward/frame/done metadata while observation-only refresh supplies next-state features.
V141|Legacy held-action collection uses compact ML positions only before each decision boundary; boundary transitions materialize fresh full features before DDQN action selection.
V142|Compact intermediate ML transitions validate positions and transition fields without reading or rescanning cached feature vectors; full feature validation remains at reset, boundary, and terminal materialization.
V143|CPU DQN training uses batched AdamW updates while preserving AdamW optimizer state/checkpoint semantics; non-CPU optimizer selection remains unchanged.
V144|HPO checkpoint-timing diagnosis compares recorded `best_inner.step` with `updates_completed`; equal values report final-step selection, unequal values report non-final, missing values report unavailable.
V145|HPO relevance-gate diagnosis reports training-side and locked-holdout means separately; a training pass with holdout below 800 is never labeled overall pass.
V146|Batch waypoint steering emits same native action indices as scalar steering for every position, target, tolerance, and arrival-latching state; current-cell reuse does not change target selection.
V147|Canonical benchmark record binds source/native wheel, manifest, argv/config, learner seed, model/action/controller, observation/cadence, lane count/execution, device, threads/affinity, budget, eval/checkpoint cadence, and machine identity.
V148|Throughput reports actual native game frames, nominal frames, lane steps, transitions, optimizer updates, samples, collection FPS, learner updates/s, and all-in FPS with explicit denominators.
V149|Matched CPU baseline/candidate uses same V147 record except declared factor, ≥3 timed repetitions after warm-up, retains raw repetitions, and reports median plus spread; GPU check records full matched run.
V150|Phase timings reconcile with wall within documented tolerance and separate native collection, native observation/hazard, Python boundary, replay sample/materialization, host-to-device, learner, device-to-host, evaluation, checkpoint, telemetry, startup/finalization, and unaccounted overhead.
V151|Profiler overhead and profiler/no-profiler delta remain separate from unprofiled throughput; bounded cProfile/torch.profiler/native traces preserve actions, RNG, gradients, checkpoint order, and default behavior.
V152|Microbench result cannot promote candidate; representative end-to-end current headless run must confirm all-in speedup and no undeclared phase regression.
V153|Native collector optimization preserves fixed-seed reference action/frame/reward/done/position traces; inactive/dead-lane, reset, and cadence semantics remain unchanged.
V154|Replay, checkpoint, recovery, and resume optimization preserves byte/semantic contract and reproduces saved boundary with same RNG, optimizer, and replay state.
V155|Active remote job → read-only observation; benchmark/Colab tooling cannot send controls, write remote job files, start replay, or allocate overlapping trial.
V156|GPU-sensitive candidate promotion requires matched real Colab run on declared actual accelerator with source/native/config identity and remote phase/artifact verification; CPU-only and synthetic CUDA evidence remains screening.
V157|Colab confirmation cadence = optimization phase boundary plus every GPU-sensitive change, T4-first, ordered typed capacity escalation; no per-microtrial allocation.
V158|Eval/checkpoint/schedule changes report steady-state and all-in throughput, preserve declared selection/recovery obligations, and never claim compute speed from omitted work.
V159|Runtime promotion keeps holdout locked and requires relevant training-side learning gate; input-parameter changes begin after runtime baseline and use one-factor/declared-interaction matched trials.
V160|Accepted default chosen from Pareto of all-in throughput, learning evidence, artifact/recovery reliability, and variance; rejected candidates remain in ledger with cause.

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
T17|x|Extend native PPO with indexed-pixel four-frame stacks, reset-safe lifecycle, pixel model/checkpoint metadata, CLI configuration, and regression tests.|V25,V26,V27,I14,C12,C13
T18|x|Run a matched native pixel-PPO control on the frozen training split, select only on inner training seeds, and report locked holdout performance and throughput.|V7,V18,V25,V27,G7
T19|x|Compare board and pixel controls by sample efficiency, wall-clock throughput, split gap, lower tail, and representative visual failures.|V7,V18,V25,V27,G6,G7
T20|~|Collect fresh training-only indexed-pixel teacher examples and train a pixel CNN behavior-cloning checkpoint suitable for actor-only PPO warm starts.|V18,V20,V25,V27,G6,G7,I9,I14
T21|x|Implement frozen-manifest decision-relevance gate: canonical multi-horizon all-action audit, policy regret/action dynamics, hazard timeline, training-only gate, JSON/Markdown/plot artifacts, CLI/tests.|V16,V28,V29,V30,V31,I7,I15
T22|x|Repair native PPO lane-wise GAE, expose the 800-frame focused-baseline target in NG provenance, and add regression coverage before rerunning board PPO.|V32,V33,G8,C14,I3,I16
T23|~|Run the focused board-PPO baseline under the 800-frame target, audit checkpoints with T21 on training seeds, and iterate baseline-only until the target is reached or the blocker is diagnosed.|V30,V31,V33,G4,G8,C14
T24|~|Add an isolated `board_full` observation mode that preserves off-screen hazards at board edges, proves native/Python parity, and reruns the focused baseline with the same PPO protocol.|V34,C5,G8,I17
T25|~|Expose explicit average/max board pooling, preserve legacy defaults, and run the `board_full` max-pooling baseline under the same 800-frame gate.|V35,G8,I18
T26|~|Add coordinate-preserving `board_full_coords` observation mode, prove native/Python parity, and run the baseline max-pooling trial under the 800-frame relevance gate.|V36,C5,G8,I19
T27|x|Implement waypoint grid/target codec, deterministic bounded steering, transition contract, and native-physics regression tests.|V37,V38,V39,I20,C15,C16,C17
T28|x|Run full-state oracle waypoint feasibility at 8, 16, and 32 pixels across training seeds; select resolution on training-only 800-frame evidence; report holdout after freeze.|V40,V42,V44,V45,G9,C2,C4,I20
T29|x|Implement waypoint DQN observation/model/replay, Double-Dueling-n-step targets, checkpoints, resume validation, and provenance.|V39,V41,V46,I21,C18
T30|.|Train first waypoint DQN under frozen 70/30 manifest and 800-frame relevance gate; report training/inner/holdout trends and throughput.|V42,G9,G8,C19,I21
T31|~|Iterate waypoint baseline with declared replay, optimizer, regularization, and HPO variants only from training-side evidence; preserve matched interaction budget.|V42,G6,C19
T32|x|Implement fixed-N frozen-center TTC reference field, semantic threat/spawn channels, and opt-in hazard waypoint-DDQN observation; defer gradient controller.|V43,V119,V120,V121,V122,V123,V124,V125,V128,V129,G10,G18,I22,I43,I44,I45,C20,C42,C43,C44,C46,C47
T33|.|Run matched legacy waypoint-DDQN versus hazard-DDQN comparison; select fixed N/H/halo only from training-side evidence; report locked holdout after 800-frame gate.|V42,V127,G6,G10,G18,I21,I22,C4,C42,C44,C46
T34|x|Add native `ml.rs` waypoint feature encoding, optional batch/PyO3 `ml_observation` and player-coordinate outputs, and canonical Python parity tests.|V47,V48,I24,C21
T35|x|Migrate waypoint DQN collection/evaluation to native ML observations, retain a reference path for debugging, remove the 800-frame horizon cap, and benchmark the hot path.|V49,V50,G11,I21,I24
T36|x|Cache immutable waypoint geometry, add allocation-free scalar steering for the DQN hot path, and profile before/after under the same native-ML run.|V51,I20,G11
T37|x|Remove redundant native per-frame snapshot construction while preserving exact frame-result parity, then profile the matched DQN workload again.|V52,I20,G11
T38|x|Add a shared simulation ML-only frame path, expose fast batch reset/step results through PyO3/Python, migrate DQN to it, and verify canonical trace parity before profiling.|V53,V54,I20,G11,I24
T39|x|Add non-blocking DQN telemetry/control mailbox, minimal WebSocket dashboard, checkpoint replay worker, and tests.|V56,V57,V58,V59,V60,V61,V139,I25,I26,I27,I28
T40|x|Add training-only three-life DQN collection semantics, negative death penalty, life telemetry, checkpoint compatibility, graceful signal stopping, and regression tests.|V39,V41,V62,V63,V64,V66,C24,I21,I25
T41|x|Add reproducible overnight DQN preset using selected HPO parameters, larger batch, long budget, training-only interim evaluation, stop/resume CLI, and provenance tests.|V42,V50,V65,V66,C25,C26,I21,I29
T42|x|Add training-split best/mean/bad replay set and labeled dashboard comparison controls for one checkpoint, with deterministic selection, metadata, and tests.|V59,V60,V61,V67,V68,V69,I26,I27
T43|~|Record native action/frame traces and run mandatory original-cartridge indexed-pixel regression after every saved replay; expose failure provenance and tests.|V52,V59,V60,V70,V71,V72,C27,I26,I27,I30
T44|~|Add opt-in arrival latching, configurable steering tolerance, corner-node ban, corner safety penalty, and explicit checkpoint/replay provenance across DQN train/eval/replay.|V37,V38,V51,V73,V74,V75,C28,I20,I21,I31
T45|~|Implement isolated pixel-only DQN with native indexed frame stacks, compact uint8 replay/checkpoints, life semantics, split evaluation, reports, and matched 200k one-life/three-life comparison.|V25,V26,V27,V62,V63,V64,V76,V77,V78,V79,G13,C29,I32
T46|x|Add opt-in exact-parity rendered pixel boundary, Python adapter, DQN selection, backward-compatible resumes, focused tests, and benchmark.|V80,V81,G14,C30,I33
T47|x|Add Colab job schema, durable local state/events, source/native-wheel bundle manifest, artifact allowlist/hash inventory, dry-run CLI, and unit tests.|V82,V83,V86,V89,G15,C31,C34,C35,I34,I35,I36
T48|x|Add named-session Colab CLI adapter with T4-first ordered GPU escalation, authenticated preflight, bounded polling, graceful stop, cleanup, and mocked lifecycle tests.|V83,V84,V85,V87,V95,V96,V98,R6,R7,R8,I34,I35
T49|x|Add remote worker/bootstrap with optimized native-pixel checks, heartbeat/status forwarding, subprocess execution, checkpoint-aware stop, and finalized artifact archive.|V84,V87,V88,V101,I37,I38
T50|x|Run a short T4 optimized-path smoke job, retrieve artifacts, verify hashes/provenance, and record throughput/observability evidence without touching paused CPU runs.|V82,V84,V88,V89,G15,C32,C33
T51|.|Run the first full optimized NG training job through batch control, escalate only from recorded throughput/tractability evidence, and return selected artifacts/report.|V42,V50,V65,V79,V82,V85,V86,G15,C32,C35
T52|.|Run `check`/targeted/full regression gates, document operator workflow and recovery drills, and hand off accepted Colab batch infrastructure.|V56,V57,V66,V82,V83,V87,V89,C33,I34,I35,I36
T53|~|Repair pixel reset/lives replay semantics, best/stale reporting, configurable reward/update conditioning, and diagnostic telemetry before expensive GPU learning claims.|V78,V90,V91,V92,G13,I32
T54|x|Freeze finished native pixel lanes; publish immutable replay checkpoints; verify Colab-to-local learner/replay resume and corruption/failure recovery.|V78,V94,V97,I32,I33
T55|x|Add periodic durable off-runtime checkpoint export and explicit Colab resume payload; test runtime-loss recovery before unattended full-budget job.|V78,V89,V94,V99,V100,G15,I34,I38
T56|x|Build training-seed pixel perception probe with exact waypoint-controller oracle; compare palette encoding/spatial retention before further DDQN budget.|V76,G13,C4,I32,V102
T57|.|Integrate palette-aware spatial pixels as an isolated DQN architecture and run a matched training-side GPU pilot against scalar-fast before committing to a long pixel run.|V102,G13,I32,C4
T58|.|Repair pre-checkpoint recovery synchronization, add the regression gate, and rerun the interrupted Colab pilot.|V95,V99,V103,G15
T59|x|Publish pixel-DQN checkpoints before long evaluations and add recovery-order regression coverage.|V94,V99,V104,I32
T60|x|Avoid repeated Colab recovery downloads when checkpoint and immutable replay hashes are unchanged; retain same-step post-evaluation changes.|V99,V105,G15,I32
T61|x|Classify Colab `TooManyAssignmentsError`/HTTP 412 as safe capacity escalation and test the fail-closed distinction.|V85,V106
T62|~|Add reconciled phase timing plus optional bounded CPU/CUDA profiler artifacts for pixel-DQN; capture baseline without changing default behavior.|V107,V110,I39,I40,C37,C39
T63|.|Replace full replay gzip snapshots with content-addressed incremental/chunked snapshots; preserve legacy restore, atomic publication, and exact resume tests.|V108,V109,I39,C38
T64|.|Add reusable pinned host staging and non-blocking H2D learner mode with synchronous/CPU fallback and matched numerical tests.|V107,V111,I39,C39
T65|.|Add bounded replay sampling/materialization prefetch with checkpoint-safe RNG ownership and synchronous equivalence tests.|V107,V112,I39,C39
T66|.|Benchmark opt-in AMP, `torch.compile`, and larger batches; separate warmup cost and reject unstable/non-finite variants before selection.|V107,V113,I39,C36,C37
T67|.|Add native multi-microstep fast-pixel API, migrate DQN only after long-trace parity, and measure Python-boundary reduction.|V80,V97,V114,I33,I41,C39
T68|.|Benchmark native lane counts and serial/parallel collection separately from learner timing; persist matched throughput evidence and selected settings.|V107,V115,I33,I41,C36,C37
T69|.|Run ordered optimization ladder on available Colab hardware, starting T4, escalating only from recorded evidence, and publish matched wall-time/learning report.|V107,V113,V115,V116,G16,C36,C37,C40,I34,I39
T70|.|Inventory and systematically review all pixel-DQN input parameters after runtime optimizations; run matched one-factor/declared-interaction trials and publish promotion decisions.|V117,G17,C36,C37,C41,I39,I42
T71|x|Repair pixel-DQN Just/devenv command exposure and add recipe smoke coverage.|V118,I39
T72|.|Add packed TTC/bitfield/fixed-point hazard storage after reference parity; benchmark memory/throughput and promote only after value/action parity.|V126,C45,C48,I43,I44
T73|x|Switch active NG Colab entry point from custom supervisor to `colabctl` detached backend over official CLI; retain legacy entry point, compact local records, verified bootstrap/artifacts, and offline gates.|V130,V131,R9,G15,C31,C32,C33,C34,C35,C40,I34,I35,I36,I37,I38,I46
T74|x|Handle CLI `TooManyAssignmentsError` as declared GPU-capacity escalation; preserve fail-closed non-capacity errors.|V132,C40,I34
T75|x|Pin the project Colab CLI kernel-client pair, add pre-allocation API preflight, and cover the detached-launch dependency contract.|V133,V131,C40,I34
T76|x|Release the owned Colab session when detached launch fails after allocation and add the cleanup regression.|V134,V131,C40,I34
T77|x|Keep status/logs/watch/stop/retrieve dispatch independent of submit-only trainer arguments and add a parser regression.|V135,I34
T78|x|Validate the canonical manifest hash in the remote bootstrap and add regression coverage for the semantic/raw hash boundary.|V136,V130,V131,I34
T79|x|Preserve the native wheel filename in the payload and reject missing or ambiguous remote wheel members before installation.|V137,V130,V131,I34
T80|x|Set detached trainer `PYTHONPATH` to the packaged repository `src/` directory and cover the remote import layout.|V138,V137,I34
T81|x|Add action-equivalent batched waypoint steering and remove duplicate current-cell lookup; profile matched CPU collector before/after.|V51,V146,I20,G11
T82|.|Define benchmark manifest/schema and `dodge-ng-benchmark` runner for bounded local CPU, native microbench, profile, and comparison modes; active Colab lifecycle remains out of scope.|G19,G21,C50,C52,C53,V147,V148,V149,I47,I48,I49
T83|.|Capture current native headless `dodge.ng.dqn` CPU baseline across matched lane counts, serial/parallel execution, threads/affinity, observation/hazard modes, `step_frames`, decision cadence, replay, and learner phases; freeze representative workload and raw evidence.|G19,C49,C53,V147,V148,V149,I47,I49
T84|.|Extend DQN and pixel-DQN telemetry to common phase/counter schema with reconciled collection, native observation/hazard, Python boundary, replay, transfer, learner, evaluation, checkpoint, and all-in timings.|G16,G19,C55,V148,V150,V151,I39,I50
T85|.|Run bounded cProfile, `torch.profiler`, and native profiles on representative CPU baselines; build bottleneck ledger before optimization and separate profiler overhead.|G21,C52,V150,V151,V152,I47,I49
T86|.|Remove duplicate native/Python boundary work, allocations, copies, feature materialization, and current-state lookups; preserve fixed-seed trace parity before matched timing.|C54,C58,V146,V153,I20,I24,I41
T87|.|Optimize native headless collection with active-lane compaction, preallocated results, batched actions, multi-step API, and thread/Rayon tuning; benchmark each factor separately.|G19,C54,V148,V152,V153,I41,I47
T88|.|Optimize fixed-center TTC/hazard path with immutable geometry reuse, packed storage, and serial/parallel exposure; float/reference parity precedes promotion.|G18,C54,C58,V121,V126,V153,I43,I44
T89|.|Optimize replay sampling/materialization plus checkpoint/recovery I/O with bounded memory reuse, asynchronous publication, and incremental snapshots; preserve exact resume.|G16,C38,C54,V108,V109,V150,V154,I39,I49
T90|.|Optimize CPU learner path across thread affinity, batch/update kernels, memory reuse, AdamW/foreach, and Python overhead; report learner and all-in deltas separately.|G19,C51,C54,C55,V148,V152,V160,I21,I39
T91|.|Add opt-in overlap pipeline for collection, replay, host staging, H2D, learner, and checkpoint work with synchronous/CPU fallback and numerical/resume equivalence.|G19,C39,C54,C57,V150,V154,V156,I39,I50
T92|.|Benchmark GPU-sensitive AMP, `torch.compile`, pinned/non-blocking transfer, prefetch, fused kernels, and larger batches; isolate warm-up/compile cost and reject instability.|G20,C39,C56,C57,V113,V151,V156,I39,I52
T93|.|Run native lane/step/cadence scaling matrix, find saturation and contention points, and select settings only from collection plus all-in evidence.|G19,C53,C54,C55,V147,V148,V152,V157,I41,I47
T94|.|Create candidate ledger with baseline/candidate hashes, factor, phase deltas, FPS/update metrics, parity/learning/resume gates, promotion or rejection cause, and raw artifact links.|G21,C54,C58,V149,V152,V153,V154,V160,I49,I51
T95|.|Run occasional matched real-Colab T4 checks at completed optimization boundaries and after GPU-sensitive changes; verify remote phase telemetry, source/native identity, artifacts, and cleanup.|G20,C50,C56,C57,V155,V156,V157,I34,I48,I52
T96|.|Run representative full native headless GPU training target on selected settings; compare all-in native FPS, learner throughput, phase shares, learning gate, checkpoint/resume, and artifact integrity.|G15,G19,G20,C49,C57,C58,V148,V150,V156,V158,V160,I34,I49,I51
T97|.|After runtime track, inventory every exposed DQN/pixel-DQN input and run training-side one-factor/declared-interaction review from selected runtime baseline; holdout remains locked.|G17,C54,C59,V159,I42,I51
T98|.|Run final parity, checkpoint/resume, profiler-schema, clean-repetition, artifact, and real-Colab acceptance gates; publish selected default plus rejected-candidate ledger and report.|G19,G20,G21,C52,C55,C58,V149,V150,V153,V154,V156,V160,I47,I49,I51

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
B13|2026-09-03|T21 first lint gate found type-erased result access, unused import, and Markdown line-width drift.|Mechanical format and typed result boundary; no new invariant.
B14|2026-09-03|Matplotlib boxplot API in current environment rejects legacy `labels=` keyword.|Use current `tick_labels=` plotting API and retain plot-generation coverage.
B15|2026-09-03|T21 custom-horizon fixtures omitted newly required near-term gate horizon.|Set gate horizon explicitly in custom-horizon tests; no new invariant.
B16|2026-09-03|Full-suite rerun reproduced B11's concurrent Pemsa/Xvfb hidden-window timeout in three legacy NEAT tests.|Rerun live Pemsa tests in isolation; no T21 source workaround.
B17|2026-09-03|Native PPO flattened time-major transitions from independent lanes before GAE, so advantages crossed lane boundaries and mixed unrelated episodes.|Compute GAE on `[time,lane]` tensors before flattening; enforce V32 with a multi-lane boundary regression test.
B18|2026-09-04|The legacy 19x16 board and pixels dropped enemies entirely while their canonical positions remained outside the screen, leaving the baseline blind to the first incoming hazard.|Add an opt-in off-screen-preserving board observation and enforce V34 with native/Python parity and legacy-unchanged tests.
B19|2026-09-04|The native workspace Clippy gate found pre-existing unchecked indexing in counterfactual scoring and its regression assertion.|Use iterator/get access at the native boundary; no new behavior invariant is required.
B20|2026-09-04|The relevance gate could pass while `board_full` still collapsed every off-screen hazard onto an edge cell, so the CNN saw hazard presence but not time-to-arrival; max pooling then learned a vertical edge policy and plateaued below 800.|Add coordinate-preserving `board_full_coords` and require native/Python parity before another baseline campaign.
B21|2026-09-03|Waypoint feasibility retained canonical snapshot bytes but passed them directly to typed-state conversion, causing an attribute failure before evaluation.|Decode bytes at runner boundary; enforce V44 with native oracle regression.
B22|2026-09-03|Waypoint feasibility reset measured done lanes but left inactive dummy lanes unreset, allowing a dummy lane to die and block later batch steps.|Reset every native-done lane before next step; enforce V45 with completed-lane regression.
B23|2026-09-03|First waypoint DQN lint gate found unused import and line-width drift before behavioral verification.|Format DQN module and rerun targeted lint before next verification gate.
B24|2026-09-03|DQN formatter left long metadata literals and report strings outside line limit.|Wrap remaining literals and rerun targeted lint before behavioral verification.
B25|2026-09-03|Waypoint DQN decoded full canonical entity/pattern graphs to read player coordinates on every held steering frame, making long training impractical.|Read validated fixed snapshot prefix for player position; enforce V46 with full-decoder parity and malformed-prefix tests.
B26|2026-09-03|Fast snapshot reader regression test added private imports out of Ruff order.|Sort test imports and rerun targeted lint before behavioral verification.
B27|2026-09-04|Native ML encoder assigned an `f64` stage calculation directly into its `f32` feature buffer.|Cast every native feature at the output boundary and keep the native ML lint/build gate mandatory.
B28|2026-09-04|Native ML encoder initially used dynamic array indexing and a derived partial order that violated the native safety lint gate.|Use checked feature writes and an explicit total float ordering before native verification.
B29|2026-09-04|Waypoint optimization regression test exceeded the repository's 88-column Ruff gate.|Wrap new hot-path assertions before verification; no behavior invariant added.
B30|2026-09-04|Native verification was invoked from a shell without the project's Cargo toolchain on PATH.|External toolchain boundary; rerun through the repository development environment with no behavior invariant added.
B31|2026-09-04|The native Cargo gate was invoked from the Python project root, which has no Cargo manifest.|Use the native workspace directory or explicit manifest path; no behavior invariant added.
B32|2026-09-04|The native frame-capture change was not yet rustfmt-normalized when the formatting gate ran.|Run the repository formatter before the native verification gate; no behavior invariant added.
B33|2026-09-04|The workspace native test build reached the PyO3 linker without Python C symbols in the active environment.|External PyO3/Python linker boundary; isolate core and batch verification, then restore the configured Python link environment with no behavior invariant added.
B34|2026-09-04|The new frame-capture regression fixture included input mask 64, outside the core button domain.|Use only valid button masks in the capture sequence; no behavior invariant added.
B35|2026-09-04|Rebuilding the editable PyO3 extension exposed a 10-argument constructor while the Python adapter supplies 12 arguments.|Keep the installed native extension constructor synchronized with the Python adapter before runtime verification; no frame-capture invariant added.
B36|2026-09-04|ML reset borrowed lane mutably before reading lane count for its fallback error.|Capture lane count before mutable borrow; no behavior invariant added.
B37|2026-09-04|Parallel ML result wrapper retained unused lane metadata after lane identity moved into result observation.|Remove unused wrapper field; no behavior invariant added.
B38|2026-09-04|Python full-result parser lost its return when ML parser insertion split its body.|Restore typed full-result return and cover V55 with full/fast boundary tests.
B39|2026-09-04|New native parity fixtures used direct action-table indexing, violating repository panic-safety lint.|Use checked action lookup in fixtures; no behavior invariant added.
B40|2026-09-04|Native parity fixture formatting drifted from rustfmt output.|Run rustfmt before format verification; no behavior invariant added.
B41|2026-09-04|Canonical snapshot construction moved across transition trail mutation during ML-path refactor, then closure formatting drifted.|Construct full snapshot at original post-input/pre-trail boundary and run rustfmt; V52.
B42|2026-09-04|New Python ML boundary added two lines beyond repository Ruff width.|Wrap adapter and regression assertions before lint verification; no behavior invariant added.
B43|2026-09-04|Full-suite PPO tests hit the existing 512 MiB runtime preflight on `/tmp` with about 299 MiB free; the same suite passed on `/home/taylor`.|Use a spacious pytest base directory for verification; no T39 code or invariant change.
B44|2026-09-04|Native `ObservationFlags` gained `preserve_offscreen_coordinates` while one test initializer remained stale.|Update test initializer; no behavior invariant added.
B45|2026-09-04|Mojo checkpoint option used Python-style `len(String)`, rejected by UTF-8-aware Mojo compiler.|Use `byte_length()` for String emptiness before Mojo build; no behavior invariant added.
B46|2026-09-04|Initial HPO module retained an unused manifest type import after switching to the loader boundary.|Remove the unused import before HPO verification; no behavior invariant added.
B47|2026-09-04|HPO regression imports were not in Ruff's private-symbol ordering.|Sort the imported checkpoint helpers before the next targeted lint gate; no behavior invariant added.
B48|2026-09-04|The targeted pytest command pointed `TMPDIR` at a directory that had not been created, so devenv could not capture its environment.|Use the existing spacious project-home temp boundary for test execution; no code or invariant change.
B49|2026-09-04|The first HPO test exposed a queued Optuna baseline being counted as a completed trial, while the parser test expected `ValueError` instead of its documented argparse type error.|Count only terminal study trials toward the requested campaign size and assert the parser's public exception type; preserve the holdout boundary test.
B50|2026-09-04|The HPO project entry point and Just recipe were present, but devenv did not expose the command wrapper used by the launch path.|Add the native-aware `dodge-ng-hpo` devenv command with the same library/Python path setup as DQN; no behavior invariant added.
B51|2026-09-04|The first overnight lint gate found four line-width violations in the new preset metadata and regression test.|Wrap the new literals and rerun targeted lint before the behavioral gate; no behavior invariant added.
B52|2026-09-04|The stop-signal regression test name exceeded the repository's 88-column Ruff gate after the first lint fix.|Shorten the test identifier and rerun the targeted lint gate; no behavior invariant added.
B53|2026-09-04|The first real stop smoke sent SIGINT during DQN model initialization, before the training-loop-only handler scope, so no stopped checkpoint was finalized.|Install the graceful handler around the complete DQN call and verify a real interrupted run reaches stopped finalization; enforce V66.
B54|2026-09-05|T42 first lint gate found one unused import and three line-width violations in representative replay code/tests.|Remove unused import and wrap new signatures/path before rerun targeted lint; no behavior invariant added.
B55|2026-09-05|Dashboard parsed entire 365 MB metrics log before WebSocket handshake, making live dashboard appear unavailable.|Read bounded metrics tail; enforce V69 with large-log dashboard smoke test.
B56|2026-09-05|Representative button updater used `some`, stopping after Best and leaving Mean/Bad labels unset.|Iterate all role buttons; verify live dashboard renders/selects all roles; V68.
B57|2026-09-05|Final format gate found two formatter-only wrapping differences in representative replay module after lint passed.|Run `ruff format` on touched Python files before final gate; no behavior invariant added.
B58|2026-09-05|Controller-factory replacement removed training-local `grid` while checkpoint/report code still referenced it.|Keep shared grid/controller bindings available through training finalization; rerun targeted lint before the behavioral gate; no behavior invariant added.
B59|2026-09-05|Long replay oracle exceeded fixed 60-second timeout before comparison, so valid saved replay became `unavailable` rather than a result.|Scale isolated oracle timeout with replay trace length while preserving non-passing unavailable status; no behavior invariant added.
B60|2026-09-05|Pixel regression overwrote an earlier frame mismatch with a later missing oracle frame, obscuring the first divergence in a long replay.|Preserve the first mismatch reason and coordinates while still reporting the missing-frame boundary; require long replay diagnostics to identify the earliest mismatch.
B61|2026-09-05|Native explosion update tested the post-growth size when deciding whether to switch from expansion to shrink, while the cartridge tests the pre-growth local size.|Use the pre-growth size for the phase transition and add a focused expanding-kamikaze regression; preserve V70's long-trace pixel gate.
B62|2026-09-05|A personality-1 enemy marked for death spawned its kamikaze from the post-growth `current` state, while the cartridge's `kamikaze(_e)` uses the pre-growth source position and size.|Pass pre-growth position/size into kamikaze creation and add a focused dying-personality-1 regression; preserve V70's long-trace pixel gate.
B63|2026-09-05|Native applied pattern bounce logic to personality -1 kamikazes, but the cartridge guards pattern collision with `p!=-1`; the explosion acquired velocity on the frame it entered a pattern rectangle.|Skip pattern collision for kamikazes and add a focused explosion/pattern regression; preserve V70's long-trace pixel gate.
B64|2026-09-05|Pixel smoke harness passed a NumPy array directly to a PyTorch module instead of using the declared tensor boundary.|Use `torch.from_numpy` in model-bound tests and keep V76's tensor/input contract explicit; no production fix required.
B65|2026-09-05|The pixel throughput probe kept stepping a native lane after it reached terminal, so the probe violated the batch lane contract.|Reset completed lanes or exclude them before the next timed step; production semantics are unchanged and no new invariant is required.
B66|2026-09-05|Native verification used Cargo 1.82 outside configured toolchain, which cannot parse edition 2024 manifests.|Run native formatting/checks through configured devenv/Nix toolchain; no behavior invariant is required.
B67|2026-09-05|Standalone Nix Cargo 1.98 lacked installed `x86_64-unknown-linux-gnu` standard-library target, so native compilation stopped before source checking.|Use configured rustup/devenv target environment for native verification; no behavior invariant is required.
B68|2026-09-05|`BatchObservation` gained `survival_frames` without updating existing snapshot-result construction, so native compilation rejected the incomplete initializer.|Populate new observation metadata at every constructor before the next native gate; no behavior invariant is required.
B69|2026-09-05|Core pixel parity fixture compared `IndexedFramebuffer` and fixed-point coordinates against Python-boundary array types, so the fixture failed to compile.|Compare core framebuffer/fixed-point values at core layer; reserve `uint8`/`float32` conversion assertions for Python boundary tests; no behavior invariant is required.
B70|2026-09-05|Batch pixel parity fixture expected optional canonical ML positions while its `ObservationFlags::all()` configuration leaves that field unset.|Derive canonical position from requested full state when comparing mandatory pixel-boundary position; no behavior invariant is required.
B71|2026-09-05|Python pixel parity fixture compared mandatory fast positions with canonical optional `ml=False` positions, so the reference array was correctly absent.|Enable canonical ML position plumbing in boundary fixture before comparing converted position arrays; no behavior invariant is required.
B72|2026-09-05|Pixel boundary regression assertion exceeded Ruff's 88-column limit by two characters.|Wrap assertion before the next Python lint gate; no behavior invariant is required.
B73|2026-09-05|Workspace Clippy with all targets re-exposed pre-existing panic assertions in older game tests and unchecked board-point indexing outside pixel-boundary code.|Use clean library-only Clippy for touched production crates; defer unrelated legacy test/board lint to owning modules; no behavior invariant is required.
B74|2026-09-05|New pixel-boundary regression imported private checkpoint helpers out of Ruff's lexical order.|Sort private imports before the next Python lint gate; no behavior invariant is required.
B75|2026-09-05|Colab CLI `status` exits zero even when a named session is absent, so return code alone made collision detection reject every fresh job.|Classify the explicit `not found` response as absent while retaining V82 ownership identity and V85 fail-closed escalation checks.
B76|2026-09-05|A dead pixel lane was reset inside a multi-step macro then advanced under neutral actions while its transition boundary stayed frozen; multi-life loss also bootstrapped across an invisible reset.|Make every life loss terminal for replay and stop the whole macro after a reset by default; retain explicit `legacy-continue` only for old-checkpoint provenance under V90.
B77|2026-09-05|Resumed pixel training left prior `run.json`, `report.json`, and `REPORT.md` looking final while live progress had advanced.|Mark prior finalization artifacts stale immediately after checkpoint load and replace them only at safe finalization under V91.
B78|2026-09-05|Pixel final evaluation always used budget-final weights even when `checkpoint-best.pt` held a better inner-training policy.|Load recorded best model state for final split evaluation and report its exact selection step under V91.
B79|2026-09-05|Stop-whole-macro reset workaround coupled action duration to earliest death across batch.|Native active-lane stepping; fixed survivor cadence; V90,V97.
B80|2026-09-05|Saved replay referenced mutable mmap; later ring overwrite silently changed checkpoint observations.|Immutable compressed snapshot, verified restore, failed-publication regression; V94.
B81|2026-09-05|Submit exception handler stopped colliding session despite never allocating it.|Persist allocation ownership only after absence check; collision regression; V98.
B82|2026-09-05|PyPI Jupyter client lacks `KernelClient` required by Colab CLI; first T4 worker never launched.|Pin CLI 0.6.0 with official Google Colab Jupyter fork commit f18e982c3265df5e923aa9def101ab3fd737e139; dependency issue, no new invariant.
B83|2026-09-05|Artifact transfer failure triggered runtime deletion; recovery lost only checkpoint copy.|Retain launched session, explicit recovery state, atomic local receipt; V95.
B84|2026-09-05|Standalone stop mutated state during allocation; supervisor then attempted forbidden state regression.|Durable stop mailbox, single-writer lifecycle, signal forwarding; V96.
B85|2026-09-06|Hung Colab poll delayed queued stop transmission while remote training continued.|Send stop out-of-band after validating cached remote identity; durable mailbox remains fallback; V96.
B86|2026-09-06|Fallback packer reopened worker-published archive; failed command corrupted authoritative bytes and hash verification rejected return.|Use immutable worker archive plus distinct fallback path; V101.
B87|2026-09-06|Recovery polling ran before the first checkpoint; Colab exec returned success with a remote traceback/no JSON, which the controller misclassified as an identity mismatch and stopped valid training.|Make pre-checkpoint export return a matching-identity no-op, skip download, and enforce V103.
B88|2026-09-06|Pixel training saved `checkpoint-latest.pt` only after evaluation; a runtime loss during a long evaluation could discard the completed update.|Save the latest checkpoint before entering evaluation and enforce V104.
B89|2026-09-06|Recovery polling re-downloaded the same immutable checkpoint/replay pair because only `metrics.jsonl` had changed.|Compare checkpoint and replay hashes; acknowledge unchanged recovery without transfer while retaining same-step checkpoint changes; V105.
B90|2026-09-06|Colab returned `TooManyAssignmentsError`/HTTP 412 for T4 allocation, but the escalation classifier treated it as unsafe and stopped instead of trying the next tier.|Recognize explicit capacity signals while preserving fail-closed handling for non-capacity errors; V106.
B91|2026-09-06|Pixel-DQN Just recipe invoked a devenv command absent from `devenv.nix`, so real CLI profiling never reached trainer startup.|Add matching native-aware devenv wrapper and recipe smoke coverage; V118.
B92|2026-09-08|Reference hazard size helper used non-const `Option::and_then` unsupported by pinned Rust 1.97 compiler.|Use const-safe branches before native verification; no behavior invariant added.
B93|2026-09-08|Direct host-shell Ruff invocation lacked project devenv tools.|Run Python lint through configured devenv wrapper; no behavior invariant added.
B94|2026-09-08|First devenv Ruff gate found new hazard adapter lines over width and DQN import order.|Format touched Python files and rerun targeted lint; no behavior invariant added.
B95|2026-09-08|Hazard regression first lint pass retained one unused import and two over-width declarations.|Remove unused test import and wrap declarations before behavioral verification; no behavior invariant added.
B96|2026-09-08|Formatter preserved one hazard test identifier at 89 columns.|Shorten test identifier before lint rerun; no behavior invariant added.
B97|2026-09-08|Centered-grid regression expected 64.0 to quantize below midpoint although upper center 94.25 is nearer than 32.75.|Use nearest-center expectation in test; no implementation change.
B98|2026-09-08|Combined format/lint command ran devenv from `native/`, outside repository environment root.|Run devenv checks from repository root; no behavior invariant added.
B99|2026-09-08|Hazard TTC conversion loop enumerated one axis, producing N raw TTC values for an N×N field.|Iterate every row/column center pair and preserve V120 coverage.
B100|2026-09-08|Checkpoint regression test called private contract matcher absent from its import block; Ruff import order then hid exact missing symbol.|Keep checkpoint contract tests import-complete and lint-clean; no behavior invariant added.
B101|2026-09-08|Repository-wide oracle tests could not launch or fingerprint missing `src/dodge/runtime/pemsa`.|Restore Pemsa runtime before original-cartridge compatibility/raster verification; no hazard/DDQN code change.
B102|2026-09-08|Hazard practice startup reused legacy waypoint spacing and stale ML frame/survival baselines, so metrics included scripted startup and ignored centered N geometry.|Use centered hazard startup and synchronize reset baselines; enforce V128,V129.
B103|2026-09-08|Centered-startup Python wrapper exceeded Ruff width after native API addition.|Run configured formatter before behavioral verification; no behavior invariant added.
B104|2026-09-08|Centered hazard startup kept fixed Up control while target x could fall outside tolerance, so player never reached target and startup returned terminal lanes.|Steer centered startup toward both target axes; preserve legacy fixed-Up startup; V129.
B105|2026-09-08|Artifact extraction shadowed the destination Path with a file handle, so verification succeeded but final installation failed.|Keep archive destination paths distinct from copy handles and enforce idempotent receipt-guarded installation; V130.
B106|2026-09-08|`colabctl auth login` delegated ADC setup to `gcloud`, but the project devenv omitted the Google Cloud SDK, so login failed before OAuth.|Declare the external Google Cloud SDK dependency in the same devenv as colabctl; no behavior invariant added.
B107|2026-09-08|CLI transport surfaced typed `TooManyAssignmentsError` outside `AcceleratorUnavailableError`, so T4 capacity stopped GPU ladder before A100/H100.|Catch typed capacity error, record attempt, continue declared ladder; V132.
B108|2026-09-08|Project uv resolution first installed jupyter-kernel-client 1.0.2, then plain PyPI 0.8.0 still lacked `JupyterSubprotocol`; google-colab-cli 0.6.0 imports both and detached launch failed after GPU allocation.|Pin Google Colab fork commit `f18e982c3265df5e923aa9def101ab3fd737e139`, preflight both APIs before allocation, and cover V133.
B109|2026-09-08|The first direct-URL Colab dependency omitted the whitespace Hatchling requires before its environment marker, so the editable project build failed before tests.|Use valid PEP 508 direct-URL marker spacing and rerun the package/test gates; V133.
B110|2026-09-08|Hatchling rejected the required Colab fork direct reference until its explicit direct-reference metadata opt-in was declared, so the editable build still failed before tests.|Enable Hatchling direct references for the pinned Colab fork and rerun the package/test gates; V133.
B111|2026-09-08|`colabctl` allocated the named runtime before detached launch, then propagated the launch error without releasing it; the failed serious submission left an A100 session active.|Release only the requested owned session on post-allocation launch failure and enforce V134.
B112|2026-09-08|The cleanup regression used an unused import guard and a raw suppression block that violated the configured Ruff rules after behavior passed.|Use `contextlib.suppress` and an unassigned import guard before the lint gate; no behavior invariant added.
B113|2026-09-08|The CLI main path unconditionally read `trainer_args`, a field defined only on plan/submit parsers, so status failed before reattachment to the running job.|Read the remainder arguments only inside `submit` dispatch and enforce V135.
B114|2026-09-08|Remote bootstrap compared raw manifest JSON bytes with `SeedManifest.sha256`, which hashes canonical manifest content; valid submissions failed integrity verification after launch.|Recompute the canonical manifest body hash remotely and compare it with the declared field and submission identity; enforce V136.
B115|2026-09-08|The payload renamed a valid native wheel to `native.whl`, which pip rejected because wheel filenames encode distribution and compatibility tags.|Preserve the built wheel basename in the payload, discover exactly one `.whl` remotely, and enforce V137.
B116|2026-09-08|The source bundle preserved `src/dodge`, but bootstrap set `PYTHONPATH` to the repository root, so the detached trainer could not import `dodge` after all integrity and wheel checks passed.|Add the packaged `src/` directory to remote `PYTHONPATH` and enforce V138.
B117|2026-09-08|systemd user service lacked devenv native library path, so colabctl ZeroMQ import failed before status.|Export resolved `libstdc++` path and `NIX_LD` before invoking `just`; no behavior invariant added.
B118|2026-09-08|Monitor read `status.state`/`status.accelerator`, but durable `status.json` stores fields at top level, so live job logged unknown.|Read top-level fields; no behavior invariant added.
B119|2026-09-08|Dashboard `.controls`/`.replay-comparison` display rules overrode the HTML `hidden` attribute, exposing read-only training controls.|Add an author-level hidden rule and V139 page regression coverage.
B120|2026-09-08|The dashboard replay guard introduced one 90-character error literal beyond the repository Ruff width gate.|Wrap the literal and rerun the touched-file lint gate; no behavior invariant added.
B121|2026-09-08|New HPO representation imports were not Ruff-sorted, so lint stopped before tests.|Order NG imports before the lint gate; no behavior invariant added.
B122|2026-09-08|Verification command concatenated two pytest paths, so intended test gate did not run.|Separate path arguments before rerun; no behavior invariant added.
B123|2026-09-08|HPO report Markdown table patch inserted raw newline inside string literal, so lint stopped at parse.|Keep generated Markdown rows as separate literals; no behavior invariant added.
B124|2026-09-08|First HPO report lint pass retained unused imports and overlong generated-Markdown lines.|Run configured autofix and wrap report literals before behavioral tests; no behavior invariant added.
B125|2026-09-08|Parallel devenv test dispatch resolved report test path under itself, so pytest never loaded test file.|Run report test serially from repository root; no behavior invariant added.
B126|2026-09-08|HPO report fixture created trial parent before plain parent mkdir, so test failed during setup.|Use idempotent fixture directory creation; no behavior invariant added.
B127|2026-09-08|HPO report seed-routing audit passed root object to `_object` instead of nested evaluation, so valid report fixture failed before artifact generation.|Read nested evaluation fields before seed checks; no behavior invariant added.
B128|2026-09-08|Direct pytest invocation was unavailable in the repository shell, so the report test gate did not execute.|Run Python verification through the declared devenv wrapper; no behavior invariant added.
B129|2026-09-08|HPO report selected-trial audit passed the root object to `_object`, so valid selection metadata failed before aggregation.|Read the nested selected-trial record before extracting its number; no behavior invariant added.
B130|2026-09-08|HPO report split comparison treated the split wrapper as one evaluation summary, so valid training and holdout summaries failed numeric extraction.|Read the named split summaries before comparison; no behavior invariant added.
B131|2026-09-08|HPO report failure-mode analysis repeated the split-wrapper lookup mistake, so valid summaries failed before findings were generated.|Read each named split summary in the failure-mode analyzer; no behavior invariant added.
B132|2026-09-08|HPO report plot generation repeated nested report-wrapper lookups, so valid report data failed while writing plots.|Read report sections and split names through their containing mappings; no behavior invariant added.
B133|2026-09-08|HPO report validation-frame estimate patch introduced an unexpected indentation, so test collection failed before execution.|Format the arithmetic block and rerun collection before behavioral tests; no behavior invariant added.
B134|2026-09-08|Bounded hazard smoke configured zero warmup steps even though DQN validation requires positive counts and intervals, so the smoke stopped before exercising hazard state.|Use a one-step warmup for the smoke harness; no behavior invariant added.
B135|2026-09-08|Bounded hazard smoke fixture used four manifest seeds, violating the manifest sample-space divisibility contract before hazard execution.|Use ten fixture seeds for the bounded smoke; no behavior invariant added.
B136|2026-09-09|The temporary CPU profiling harness formed actions with uint8 arithmetic, so values above 255 overflowed before the native boundary was exercised.|Compute the action schedule in a wider integer dtype before casting to the native action dtype; no behavior invariant added.
B137|2026-09-09|The temporary CPU profiling harness generated replacement seeds above the native 32767 limit during long repetitions, so the benchmark stopped at reset.|Keep benchmark seed generation within the native seed domain before calling reset; no behavior invariant added.
B138|2026-09-09|Parallel devenv lint dispatch concatenated adjacent test paths, so Ruff checked a nonexistent path instead of the requested files.|Run repository verification commands serially when the wrapper shares argument state; no behavior invariant added.
B139|2026-09-09|Cadence regression test literals exceeded Ruff width, so lint stopped before behavioral verification.|Wrap test fixture calls before next Python lint gate; no behavior invariant added.
B140|2026-09-09|Cadence test fixture passed derived lane_count into concrete NativeHazardBatchResult, so its constructor rejected the non-field.|Construct fixtures from concrete native fields and rely on derived lane_count; no behavior invariant added.
B141|2026-09-09|Temporary cadence harness read non-returned metrics field after training had already persisted metrics.jsonl, so comparison wrapper stopped before second run.|Read trainer timing data from authoritative metrics.jsonl when summarizing completed runs; no behavior invariant added.
B142|2026-09-09|Native format/test gate ran from Python repository root without a Cargo manifest, so Cargo stopped before compiling touched Rust code.|Pass native manifest path or run from `native/` for Rust verification; no behavior invariant added.
B143|2026-09-09|Combined native format/test command continued after rustfmt reported differences and returned later test success, masking the failed format gate.|Run native format and test gates as separate commands and format touched Rust sources before retesting; no behavior invariant added.
B144|2026-09-09|Completed Colab handoff invoked the uv-managed HPO report entry point from bare devenv, so the report launcher was absent from PATH and the monitor recorded a false report failure.|Invoke the report module through the declared uv environment in both the recipe and unattended monitor; no behavior invariant added.
B145|2026-09-09|Compact terminal ML stepping replaced transition result with zero-reward observation-only refresh, dropping terminal reward/frame metadata before replay append.|Keep observation-only materialization separate from transition result; add terminal compact parity coverage; V140.
B146|2026-09-09|Existing training-lives test double implemented only full ML stepping after DQN collection gained compact ML stepping, so the regression suite failed before exercising life semantics.|Update test doubles to implement compact step and observation-only refresh; no behavior invariant added.
B147|2026-09-09|Compact legacy stepping was enabled at final held-action step, so DDQN reused prior observations instead of receiving fresh boundary features and behavior diverged.|Restrict compact stepping to pre-boundary frames and enforce V141 with cadence coverage.
B148|2026-09-09|Parallel devenv lint invocations shared wrapper argument state, so Ruff checked fabricated `tests/dodge/test_ng_dqn.py_hazard.py` and failed before linting sources.|Run devenv-backed verification serially; no behavior invariant added.
B149|2026-09-09|Compact ML stepping still rescanned cached full observations through `_native_ml_state`, erasing its intended boundary-reduction benefit.|Validate positions only on compact intermediate frames and enforce V142 with a feature-vector-free fixture.
B150|2026-09-09|HPO failure-mode report hard-coded every selected checkpoint as non-final, contradicting runs whose best inner step equaled `updates_completed`.|Derive checkpoint timing from recorded progress and enforce V144 with final-step fixture evidence.
B151|2026-09-09|HPO relevance-gate report labeled training-side success as an unqualified pass while locked holdout remained below 800 frames.|Report training/holdout gate status separately and enforce V145 with split-gate fixture evidence.
B152|2026-09-09|Completed Colab monitor kept polling released remote session after durable terminal handoff, producing repeated false command failures.|Exit cleanly from local terminal state plus completed handoff before remote status polling; no behavior invariant added.
B153|2026-09-09|CPU benchmark wrapper invoked absent `/usr/bin/time`, so measurement stopped before trainer startup.|Use shell `time` or an available timing command; no behavior invariant added.
B154|2026-09-09|Touched Python files were not formatter-normalized before dashboard verification, so format check failed after behavioral patch.|Run configured formatter on every touched Python file before verification; no behavior invariant added.
B155|2026-09-09|Full Dodge suite hit missing `src/dodge/runtime/pemsa` in four pre-existing cartridge compatibility/raster tests; replay comparison stayed unavailable.|Restore Pemsa runtime before full suite; no dashboard invariant added.
