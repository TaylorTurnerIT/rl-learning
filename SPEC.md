# Dodge NG Phase 0/1 control spec

§G
G1|Build a fresh, reproducible NG experiment boundary and run a native board-observation PPO baseline.
G2|Measure learning on the 70% training partition and expose the untouched 30% partition only as a final generalization check.
G3|Produce machine-readable provenance, metrics, plots, and a human-readable baseline report.

§C
C1|NG sample space is new and finite: default 100 native-valid seeds, disjoint from every legacy seed used by the prior experiments.
C2|Split is exact 70/30 over the complete NG sample space; train and holdout sets are disjoint and their union is the sample space.
C3|Legacy GA/NEAT/BC/PPO data, checkpoints, databases, manifests, and scores are archival context only and cannot be NG inputs.
C4|Holdout seeds stay locked: training, checkpoint selection, and future HPO may use training-side seeds only.
C5|Use the existing native batch environment and board encoder for the hot path; preserve legacy PPO defaults and behavior when NG fields are unset.
C6|Phase 0/1 scope stops at manifest/provenance/evaluation/plots and the board PPO baseline; pixels, replay, DAgger, gradient actions, and HPO remain later phases.
C7|Do not alter the native game contract or nested game/NEAT specs as part of this slice.

§I
I1|`src/dodge/ng/manifest.py` owns the immutable NG seed manifest, validation, hashing, and CLI generation.
I2|`src/dodge/ng/report.py` owns split metrics, trend summaries, plots, Markdown, and JSON report generation.
I3|`src/dodge/ng/train.py` owns the baseline CLI and wires one manifest into native PPO without reading legacy artifacts.
I4|`src/dodge/rl/ppo.py` accepts an optional explicit training-seed tuple and optional training-side evaluation seeds; absent values retain legacy behavior.
I5|`context/kits/dodge-ng/ng-v1.json` is the committed frozen manifest; run artifacts live under `history/dodge/ng/`.
I6|NG commands are `dodge-ng-manifest`, `dodge-ng-train`, and `dodge-ng-report`; the training recipe uses the devenv runtime boundary.

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
V6|Per-split evaluation reports finite mean, median, p10, worst, best, and solved-fraction statistics with per-seed outcomes.
V7|Reports contain training/inner/holdout comparison, train-minus-holdout gap, learning curves, diagnostic curves, provenance, and throughput when available.
V8|A successful baseline run writes valid `run.json`, `metrics.jsonl`, checkpoints, `report.json`, `REPORT.md`, and plot files under its run directory.
V9|Phase 0/1 tests run without legacy databases or checkpoints and cover manifest, seed routing, evaluator statistics, report generation, and CLI configuration.
V10|Every NG provenance/report JSON artifact is JSON-serializable after path and typed-config normalization.

§T
id|status|task|cites
T1|x|Implement `SeedManifest`, deterministic fresh default, validation/hash, committed NG manifest, and unit tests.|V1,V2,I1,I5
T2|x|Add explicit PPO training-seed routing and training-side checkpoint evaluation while preserving legacy defaults; add regression tests.|V3,V4,V5,I4
T3|x|Implement split evaluator, trend metrics, plot/report generation, and tests using synthetic run artifacts.|V6,V7,I2
T4|x|Add NG baseline CLI, package entry points, devenv-backed just recipe, provenance wiring, and CLI tests.|V3,V4,V8,V9,V10,I3,I6
T5|x|Run Phase 0 native smoke/throughput checks and freeze their evidence without touching legacy artifacts.|V8,V9,C1,C3,C5
T6|~|Run the Phase 1 native board PPO baseline on the frozen 70% partition, evaluate the locked 30%, and deliver the generated trend/performance report.|V4,V6,V7,V8,C2,C4,C6

§B
id|date|cause|fix
B1|2026-09-03|NG provenance serialized `Path` fields from a dataclass directly to JSON.|Add an explicit JSON serializer for baseline configuration and enforce V10 with the runner test.
