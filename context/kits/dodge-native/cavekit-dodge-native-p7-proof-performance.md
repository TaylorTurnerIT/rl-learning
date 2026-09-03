---
name: dodge-native-p7-proof-performance
description: Harden native Dodge with Kani properties, differential fuzzing, reproducibility, strict Rust configuration, and performance regression gates
metadata:
  type: project-phase
  phase: P7
  created: "2026-09-01"
  last_edited: "2026-09-01"
---

# Cavekit: P7 Proof and Performance Hardening

## Scope

Turn accepted native behavior into durable regression protection. Prove local
Rust safety/correctness properties with Kani, fuzz native-vs-Pemsa traces,
lock strict but usable Rust configuration inspired by Namtao, preserve
provenance, and keep the measured training speedup from regressing.

Depends on accepted P5 and P6.

## Phase contract

### §G GOAL

Native runtime → repeatable conformance evidence, local formal proofs,
strict-toolchain CI, reproducible artifacts, and performance regression
protection with no claim stronger than the evidence.

### §C CONSTRAINTS

- Kani proves explicit local properties; it does not magically prove
  equivalence to an external PICO-8 emulator.
- Differential fuzzing and finite corpus parity are reported separately from
  mathematical proof.
- Dafny/Lean may model selected abstract properties later, but no
  unvalidated Dafny-to-Rust code-generation path enters production.
- Stable application toolchain is pinned. Kani's compatible nightly, if
  required, is pinned separately and does not silently define application
  builds.
- Adopt strict Clippy/rustfmt/rust-analyzer/nextest/Criterion practices
  selectively. Core hot loops cannot depend on CLI, logging, or serialization
  conveniences.
- Unsafe code is forbidden by default; every exception is isolated, documented,
  reviewed, and covered by a proof/test boundary.
- Performance evidence retains workload, host, toolchain, repetitions, raw
  results, statistic, and budget.
- The P7 full-observation regression budget is at most 1.25× the accepted P6
  median and at most 2× the accepted P6 population standard deviation, with a
  0.05-second minimum variance limit; changing this budget requires a reviewed
  amendment.

### §I INTERFACES

proof harness → bounded FullState/action/input domain → Kani proof or explicit
counterexample/resource result.

fuzz scenario → deterministic seed, initial state, action/frame schedule →
native/Pemsa differential report and retained reproducer.

CI → format, lint, unit/integration/nextest, Kani, differential corpus,
benchmark, and legacy Python regression commands.

provenance → source and generated asset hashes, Pemsa identity, Rust/Python
toolchains, dependency lock, feature flags, backend, observation mode, host,
benchmark raw output, and acceptance status.

release report → parity tiers, proof properties, fuzz coverage/reproducers,
benchmark statistics, known limitations, and owner signoff.

### §R RESEARCH

| id | topic | finding | source |
|----|-------|---------|--------|
| R1 | Kani model checking | Kani checks Rust safety and custom correctness properties and returns proof, counterexample, or resource exhaustion | https://model-checking.github.io/kani/ |
| R2 | Kani status | Kani releases track a recent Rust nightly and support is feature-limited; pin the compatible proof toolchain and inspect limitations | https://model-checking.github.io/kani/ |
| R3 | Namtao strict lints | Suggested deny rules include unwrap/expect, panic, indexing, arithmetic side effects, and unchecked conversions; apply with explicit test exceptions only where justified | https://www.namtao.com/rust/ |
| R4 | Namtao development tools | Suggested Rust devenv includes rustfmt, clippy, rust-analyzer, bacon, cargo-nextest, Criterion, and Rayon | https://www.namtao.com/rust/ |
| R5 | AWS Automated Reasoning | AWS announcement is Bedrock Guardrails policy validation, not evidence of Dafny/Lean source transpilation | https://aws.amazon.com/blogs/aws/prevent-factual-errors-from-llm-hallucinations-with-mathematically-sound-automated-reasoning-checks-preview/ |
| R6 | Dafny Rust support | Current Dafny installation docs label Rust support partial and growing; do not make Dafny-generated Rust the native production path | https://dafny.org/latest/Installation |

### §V INVARIANTS

- V1: Kani proves framebuffer indexing, palette lookup, action-mask
  constraints, fixed-slot bounds, and any other listed bounded properties.
- V2: every Kani harness has a named property, bounded domain, source-map
  rationale, toolchain identity, and resource outcome.
- V3: retained fuzz reproducers replay identically and remain in the
  differential corpus.
- V4: native/Pemsa parity tiers A-C remain green on golden, held-out, and
  retained fuzz corpora.
- V5: repeated native batch runs with same seeds/actions/config are byte
  identical, including pixels and full-state hashes.
- V6: CI runs strict formatting/lint/test/proof commands under pinned
  configuration; test-only allowances cannot leak into production code.
- V7: generated assets, source maps, binaries, traces, and benchmark reports
  identify input/source/toolchain hashes.
- V8: benchmark median and variance remain within the accepted P6 budget for
  full-state + pixel workload; regression fails CI or opens a reviewed budget
  amendment.
- V9: legacy Python/Pemsa control, NEAT, PPO fallback, and replay commands
  remain runnable and retain their documented contracts.
- V10: reports never call finite differential coverage “mathematical proof” or
  call Kani local properties “full-game equivalence.”
- V11: unresolved parity, proof, fuzz, or benchmark failure blocks release and
  records owner, reproducer, classification, and next action.

### §T TASKS

| id | status | task | cites |
|----|--------|------|-------|
| P7-T1 | x | Add strict Rust workspace configuration, pinned toolchains, formatting, clippy, nextest, and benchmark commands | §C,R3,R4,V6 |
| P7-T2 | x | Add Kani harnesses for bounded state, action, framebuffer, palette, and buffer properties | V1,V2 |
| P7-T3 | x | Add deterministic differential fuzzing against Pemsa with retained reproducers | V3,V4 |
| P7-T4 | x | Add repeated-run reproducibility and source/generated-artifact provenance records | V5,V7 |
| P7-T5 | x | Add full-state + pixel benchmark regression and raw evidence retention | V8 |
| P7-T6 | x | Run legacy Python/Pemsa/NEAT/PPO regression and native backend smoke tests | V9 |
| P7-T7 | x | Write evidence report separating proof, conformance, visual review, and performance claims | V10,V11 |
| P7-T8 | . | Obtain owner signoff or record blocked release with reproducer and next action | V11 |

### §B BUGS

| id | date | cause | fix |
|----|------|-------|-----|
| B1 | 2026-09-02 | The first P7 format gate used `use_small_heuristics = "Max"`, which changed the existing native formatting rather than only checking the new benchmark target | Removed that override, formatted the benchmark with the pinned Rust toolchain, and reran format, locked check, Clippy, nextest, and benchmark compilation successfully |
| B2 | 2026-09-02 | Kani setup selected `/usr/bin/curl`, whose runtime libcurl did not provide the ABI expected by the Nix nix-ld library | Added `pkgs.curl` to the project devenv so the verifier download uses a directly provisioned Nix curl; rerun setup before accepting Kani results |
| B3 | 2026-09-02 | Kani downloaded its bundle but could not invoke the required `nightly-2025-11-21` toolchain because the project devenv had no `rustup` executable | Added `pkgs.rustup` to the project devenv; rerun Kani setup and keep the application channel pinned independently at Rust 1.97.1 |
| B4 | 2026-09-02 | The Kani proof crate needed the public palette-size contract, but `PALETTE_SIZE` was only re-exported inside the snapshot module | Re-exported `PALETTE_SIZE` from `dodge-core` and reran the native compile gates before proof execution |
| B5 | 2026-09-02 | The first composed `dodge-native-check` recipe passed `--locked` to `cargo fmt`, which has no Cargo lockfile option | Removed `--locked` from the format invocation and reran the composed format, Clippy, and nextest gate |
| B6 | 2026-09-02 | The first fuzz run passed the in-memory `OracleTrace.to_json()` tuple values directly to a comparator that expects JSON-decoded lists, producing a false particle-color mismatch | Compare against `json.loads(oracle.canonical_bytes())` so the fuzz path uses the same canonical JSON boundary as the persisted differential tool |
| B7 | 2026-09-02 | Seed 17 first-pattern fuzzing exposed a fixed-point weighted-selection index using raw Q16.16 units, so every sampled index above zero silently selected no pattern | Convert the floored sample to an integer weighted-list position before checked lookup; retain seed 17 as a regression case and rerun the complete fuzz corpus |
| B8 | 2026-09-02 | The first focused pattern-selection regression test referenced the private delay constant without importing it into the test module | Import the constant through the module boundary and rerun the focused test before the composed gate |
| B9 | 2026-09-02 | The corrected selection path and regression import were not yet normalized by rustfmt, so the strict check stopped at its format gate | Run the pinned workspace formatter, then rerun the complete strict check |
| B10 | 2026-09-02 | The strict Clippy profile applies to test targets and rejected the new regression test's `expect()` call under `clippy::expect_used` | Assert the result is successful and unwrap it through checked control flow with no panic-capable test helper |
| B11 | 2026-09-02 | The first P7-T4 Python lint pass found provenance line-width violations and a missing `NativeBatchEnvironment` test import | Wrap the provenance strings, use checked candidate selection, and import/type the native batch result in the reproducibility test |
| B12 | 2026-09-02 | The P7-T4 Ruff format check found one new list-comprehension wrapping issue and three legacy shapes in the touched native-backend test | Run the pinned Ruff formatter on the two touched Python files, then rerun Ruff checks |
| B13 | 2026-09-02 | The first P7-T5 benchmark checker lint pass found an unused hash import and two line-width violations | Remove the unused import and apply the pinned Ruff wrapping before executing the benchmark |
| B14 | 2026-09-02 | Removing the temporary local subprocess import from the benchmark checker left its command-version helper without a module import | Add the module-level subprocess import and rerun Ruff |
| B15 | 2026-09-02 | Adding retained Criterion estimate identity introduced a hash helper before restoring its module import | Import hashlib and rerun the benchmark-checker lint gate |
| B16 | 2026-09-02 | The first focused native/Pemsa smoke invocation bypassed the `app-test` wrapper and omitted its required X11 library path, causing a false hidden-window timeout | Rerun the focused suite through `app-test`, which owns the project X11 runtime environment |

## Gate

### Automated

- Required Kani harnesses prove or produce retained counterexamples; resource
  exhaustion is not recorded as proof.
- Golden, held-out, and fuzz-reproducer traces remain pixel/state-conformant.
- Strict Rust checks, nextest, legacy regression, and native training smoke
  checks pass.
- Repeated runs and generated artifacts carry complete provenance.
- Full-observation benchmark stays within the accepted P6 budget.

### Owner

Review the final report's wording. Confirm that formal local properties,
finite Pemsa conformance, visual approval, and performance results are clearly
separated. Approve release only with no unresolved blocking reproducer.

### Handoff

P7 closes the initial target. Later work may add audio waveform parity,
additional formal models, GPU-native policy inference, or architecture changes
only as separately reviewed reach kits.

## Cavekit routing

- grill: resolve proof claim, benchmark budget, or release-scope ambiguity.
- research: verify Kani/toolchain/current lint and benchmark behavior.
- spec: add durable proof/conformance/provenance/performance invariants.
- review: required before CI, ABI, unsafe, or claim-scope changes.
- build: execute P7 tasks only after P5 and P6 gates.
- check: final drift audit across nested specs, overview, and all kits.
- backprop: every failed gate records whether code, spec, phase order, or
  harness caused it.
- deepen: optional post-release design pass with all gates green.
