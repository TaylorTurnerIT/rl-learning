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
C7: generation seed bank size = 3; every genome receives same bank; next generation rotates bank.
C8: episode ends only cartridge death; ⊥ arbitrary survival cap.
C9: fixed NEAT projection = player + 16 enemies + 8 AOEs; raw state retains all entities.
C10: overflow telemetry ! report; ⊥ fail episode.
C11: existing uncommitted `src/dodge/neat/config-xor` + `neat-testing.py` ⊥ overwrite.
C12: train workers default ≤8; `--workers` selects positive process count.

§I

py: `DodgeEnv(step_frames=4, enemy_slots=16, aoe_slots=8)` → env.
py: `env.reset(seed: int|None = None)` → `Observation` after menu transition.
py: `env.step(action: Direction)` → `Transition(observation, reward, done, result?)` after exact `step_frames` updates.
json: raw state → player `{x,y,vx,vy,size}` + all `enemies` + all `aoes`.
json: projection → fixed numeric vector, zero slots use `present=0`.
json: episode history → seed, config, action trace, result, overflow telemetry.
cli: `just dodge-neat-replay <episode.json>` → visible replay from stored action trace + seed.
cli: `just dodge-neat-train` → genome progress + generation-best compact network summary.
cli: `just dodge-neat-train --workers <n>` → up to `n` concurrent genome evaluations.

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
V7: ∀ generation genomes → same 3 seeds; fitness = mean `survival_frames`; future generation → fresh bank.
V8: stored episode trace replays visibly without physical controls.
V9: bridge holds injected keys until Pemsa acknowledges action; release follows acknowledgement.
V10: bridge discovers input window after game-ready boundary; ⊥ transient startup window.
V11: diagonal action → matching simultaneous directional mask across exact step frames.
V12: bridge serializes action bits while game paused; completed bit mask starts next exact step.
V13: ∀ consecutive actions → final collection press clears physical-held state before game frames advance.
V14: NEAT CLI recipes launch through `uv run` so declared Python dependencies are importable.
V15: ∀ generation → report completed genomes; end → report best fitness + compact topology and strongest edges.
V16: parallel worker result equals sequential per-genome 3-seed contract; parent writes history after worker result.

§T

id|status|task|cites
T1|x|prove hidden Pemsa step bridge|V1,V2,C1,C2,C3,C4,C5
T2|x|capture raw state + fixed danger projection|V3,V4,C9,C10
T3|x|add `DodgeEnv` reset/step + episode history|V1,V2,V5,V6,I.py,I.json
T4|x|add 3-seed NEAT evaluation + replay command|V5,V7,V8,I.cli
T5|x|add focused + end-to-end regression tests|V1,V2,V3,V4,V5,V6,V7,V8
T6|x|add NEAT training progress + compact network summary|V15,I.cli
T7|x|parallelize NEAT genome evaluation|V7,V16,I.cli

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
