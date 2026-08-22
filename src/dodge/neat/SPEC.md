# Dodge NEAT environment

§G

NEAT agent observes live Dodge state, selects direction every fixed game-frame interval, trains across rotating seed banks, replays exact episodes.

§C

C1: retain checked-in `dodge.p8`; temporary instrumented cartridge only.
C2: retain prebuilt Pemsa; ⊥ source build or Python game conversion.
C3: live bridge uses hidden X11 window + no-op `_draw`; target window only.
C4: step interval integer `3..5`, fixed per episode.
C5: action space = `neutral|left|right|up|down|up_left|up_right|down_left|down_right`.
C6: `reset()` default → fresh entropy seed; explicit seed only test/replay.
C7: generation seed bank size = 3; every genome receives same bank for 5 generations; fresh held-out bank reports champion generalization.
C8: episode ends only cartridge death; ⊥ arbitrary survival cap.
C9: fixed NEAT projection = player + 16 enemies + 8 AOEs; raw state retains all entities.
C10: overflow telemetry ! report; ⊥ fail episode.
C11: existing uncommitted `src/dodge/neat/config-xor` + `neat-testing.py` ⊥ overwrite.
C12: train workers default ≤8; `--workers` selects positive process count.
C13: generation network visual ! self-contained HTML/SVG; ⊥ browser dependency or external service.
C14: retain ≤5 NEAT checkpoints per run; episode history ∉ checkpoint retention.
C15: v3 defaults: population 100, compatibility target 6 species, fixed 12-seed champion benchmark every 5 generations.

§I

py: `DodgeEnv(step_frames=4, enemy_slots=16, aoe_slots=8)` → env.
py: `env.reset(seed: int|None = None)` → `Observation` after menu transition.
py: `env.step(action: Direction)` → `Transition(observation, reward, done, result?)` after exact `step_frames` updates.
json: raw state → player `{x,y,vx,vy,size}` + all `enemies` + all `aoes`.
json: projection → fixed numeric vector, zero slots use `present=0`; v2 entity slots include bounded time-to-intersection.
json: episode history → seed, config, action trace, result, overflow telemetry.
cli: `just dodge-neat-replay <episode.json>` → visible replay from stored action trace + seed.
cli: `just dodge-neat-replay-latest <epoch>` → visible replay selected generation winner from newest NEAT run.
cli: `just dodge-neat-train` → genome progress + generation-best compact network summary.
cli: `just dodge-neat-train --workers <n>` → up to `n` concurrent genome evaluations.
cli: `just dodge-neat-train --evolution-seed <n>` → reproducible population evolution; game seed banks remain varied.
cli: `just dodge-neat-train` end → generation table + concise final fitness summary.
cli: `just dodge-neat-resume <run> --generations <n>` → same run + `n` generations.
file: `generation-####/network.html` → interactive best-genome weighted graph.

§R

R1|Pemsa output|`printh` writes process stdout|https://github.com/egordorichev/pemsa/blob/6c13c5879c800af33543f702a353285cfa9e6fb0/src/pemsa/util/pemsa_system_api.cpp#L35-L44
R2|Pemsa input|input module delegates `btn`/`btnp` to backend|https://github.com/egordorichev/pemsa/blob/6c13c5879c800af33543f702a353285cfa9e6fb0/src/pemsa/input/pemsa_input_api.cpp#L6-L44
R3|Pemsa backend|input backend exposes keyboard queue, no stdin contract|https://github.com/egordorichev/pemsa/blob/6c13c5879c800af33543f702a353285cfa9e6fb0/include/pemsa/input/pemsa_input_backend.hpp#L6-L20

§V

V1: `step(action)` → exactly configured game updates; ⊥ neutral gap or extra frame.
V2: reset→step state/action exchange targets owned hidden Pemsa window only; close/error → child + X server reaped.
V3: raw state exposes player position/vector, every enemy position/vector/bounds, every visible AOE bounds/vector.
V4: projection reserves 16 enemy + 8 AOE slots; slots danger-order by predicted player-box intersection then distance; overflow retains telemetry.
V5: same seed + config + action trace → same terminal result in headless and visible replay.
V6: reset seed omitted → entropy seed; explicit seed preserved in result/history.
V7: ∀ generation genomes → same 3 seeds; fitness = mean `survival_frames`; bank holds 5 generations; champion held-out result ⊥ selection fitness.
V8: stored episode trace replays visibly without physical controls.
V9: bridge holds injected keys until Pemsa acknowledges action; release follows acknowledgement.
V10: bridge discovers input window after game-ready boundary; ⊥ transient startup window.
V11: diagonal action → matching simultaneous directional mask across exact step frames.
V12: bridge serializes action bits while game paused; completed bit mask starts next exact step.
V13: ∀ consecutive actions → final collection press clears physical-held state before game frames advance.
V14: NEAT CLI recipes launch through `uv run` so declared Python dependencies are importable.
V15: ∀ generation → report completed genomes; end → report best fitness + compact topology and strongest edges.
V16: parallel worker result equals sequential per-genome 3-seed contract; parent writes history after worker result.
V17: accepted terminal action → lost final key-release window error ! terminal result handling; pre-accept key errors ! fail.
V18: ∀ completed generation with best genome → write self-contained visual; slider spans strongest visible edges through every enabled edge.
V19: train end → table row ∀ completed generation with population mean and best survival; ⊥ full genome dump.
V20: replay holds menu `x` through transition, pauses at game-ready, releases between actions; trace retains its mouse-input mode; stored action frames → exact game updates.
V21: transient hidden-X11 input ack → release + retry while paused; one failed episode → same-seed retry; repeat failure → error.
V22: completed generation → atomic checkpoint + run record; resume latest → saved population/species/RNG next gen; retain ≤5 checkpoints.
V23: `RunCheckpointer` ∈ full NEAT reporter lifecycle; ⊥ missing hook abort generation.
V24: checkpoint pickle ⊥ live callback; tiny live population completes gen1 → checkpoint exists.
V25: NEAT replay-latest selects newest `run-*`; epoch → recorded best genome trace with greatest `survival_frames`; absent run, epoch, or trace → fail before Pemsa.
V26: v2 default → 3-frame decisions, sparse direct initial graph, local weight/bias mutation; v1 config remains resume-compatible.
V27: v2 entity feature `time_to_intersection` ∈ [0,1]; overlap → 0; no future intersection → 1; visual labels identify feature.
V28: v3 generation record → species count/sizes, compatibility threshold, hidden-node stats; adaptive threshold targets 6 species within `4..8`.
V29: v3 champion benchmark uses same 12 held-out seeds every 5 generations; benchmark ∉ selection fitness.
V30: new run records entropy-generated `evolution_seed`; same seed/config/game banks → reproducible population trajectory.
V31: transient `InputAcknowledgementTimeout` retries same-seed episode 3 times; only exhausted retries fail worker task.
V32: CLI time-to-intersection omitted → derive config input width; explicit mode/input-width mismatch → fail before Pemsa.

§T

id|status|task|cites
T1|x|prove hidden Pemsa step bridge|V1,V2,C1,C2,C3,C4,C5
T2|x|capture raw state + fixed danger projection|V3,V4,C9,C10
T3|x|add `DodgeEnv` reset/step + episode history|V1,V2,V5,V6,I.py,I.json
T4|x|add 3-seed NEAT evaluation + replay command|V5,V7,V8,I.cli
T5|x|add focused + end-to-end regression tests|V1,V2,V3,V4,V5,V6,V7,V8
T6|x|add NEAT training progress + compact network summary|V15,I.cli
T7|x|parallelize NEAT genome evaluation|V7,V16,I.cli
T8|x|tolerate terminal window teardown after accepted action|V17
T9|x|write interactive best-network generation visual|V18,I.file
T10|x|report final NEAT generation table and summary|V19,I.cli
T11|x|fix NEAT replay menu bootstrap timing|V5,V8,V20
T12|x|retry transient hidden-X11 input and deterministic episode|V21
T13|x|add NEAT checkpoint retention + resume|V22,I.cli
T14|x|make checkpoint reporter full NEAT reporter|V23
T15|x|exclude checkpoint callback from NEAT pickle|V24
T16|x|add latest NEAT generation replay|V25,I.cli
T17|x|add v2 NEAT search profile, seed schedule, intersection feature|V7,V26,V27,I.json
T18|x|add v3 speciation diagnostics, benchmark, reproducible evolution, timeout recovery|C15,V28,V29,V30,V31,V32,I.cli

§B

id|date|cause|fix
B1|2026-08-21|bridge lines exceeded Ruff width|mechanical format
B2|2026-08-21|`stat(31)` ignored targeted X11 key|V9
B3|2026-08-21|raw-state Lua lines exceeded Ruff width|mechanical format
B4|2026-08-21|bridge captured transient Pemsa startup window|V10
B5|2026-08-21|Pemsa backend retains one injected button state|V11,V12
B6|2026-08-21|new-module lint scope included user-owned starter script|narrow verification scope
B7|2026-08-21|new environment imports were not Ruff-isort ordered|mechanical format
B8|2026-08-21|user-modified GA mutation invalidated fixed-length elite test|exclude unrelated dirty behavior
B9|2026-08-21|new evaluator missed Ruff import and assignment rules|mechanical format
B10|2026-08-21|Python formatter was applied to Nix and Just files|format Python scope only
B11|2026-08-21|final action press remained held into the next decision boundary|V13
B12|2026-08-21|NEAT devenv recipes bypassed the project uv environment|V14
B13|2026-08-21|parallel evaluator imports and test task access missed Ruff rules|mechanical format
B14|2026-08-21|spawn test import order missed Ruff rule|mechanical format
B15|2026-08-21|terminal cartridge exit destroyed X11 window before final action keyup|V17
B16|2026-08-21|embedded HTML lines exceed Python formatter width|template E501 exemption
B17|2026-08-21|visual test expected runtime label in static HTML|assert embedded graph data
B18|2026-08-22|final report imported `Iterable` from legacy module|mechanical format
B19|2026-08-22|v1 NEAT bridge observed virtual mouse while replay disabled it|V20
B20|2026-08-22|a single missed X11 key ack aborted a parallel generation|V21
B21|2026-08-22|checkpoint reporter lacked NEAT `post_evaluate` hook|V23
B22|2026-08-22|checkpoint pickle captured local run-record callback via species reporters|V24
B23|2026-08-22|latest NEAT replay test + CLI exceeded Ruff format|mechanical format
B24|2026-08-22|v2 scheduler import missed Ruff order|mechanical format
B25|2026-08-22|v1 report + resume tests omitted v2 validation metadata|V7,V26
B26|2026-08-22|repeated hidden Pemsa input timeout aborted parallel generation|V31
B27|2026-08-22|v3 NEAT additions missed Ruff formatter|mechanical format
B28|2026-08-22|v3 NEAT tests missed Ruff lint|mechanical format
B29|2026-08-22|explicit 197-input config inherited 221-input default mode|V32
B30|2026-08-22|v3 config-width test missed Ruff import order|mechanical format
B31|2026-08-22|v3 threshold calibrated on evolved genome distances fragmented fresh population|V28
