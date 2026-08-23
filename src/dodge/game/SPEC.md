# Dodge JSON control layer

§G

JSON movement sequence → visual or headless controlled Pemsa Dodge run; no emulator build.

§C

C1: Linux local runner = `src/dodge/runtime/pemsa` + `src/dodge/game/dodge.p8`.
C2: ⊥ modify cartridge logic/assets.
C3: keyboard injection targets launched game window only; ⊥ global blind keystrokes.
C4: force SDL X11 backend; use `xdotool` from devenv.
C5: Python standard library only; ⊥ new PyPI runtime dependency.
C6: validate complete input before launch or key injection.
C7: duration unit = integer milliseconds; range `1..60000`.
C8: interrupt/error → release held keys + terminate owned emulator.
C9: seeded run uses temporary cartridge; checked-in `dodge.p8` unchanged.
C10: headless run → SDL dummy video/audio + no-op `_draw`; ⊥ display server/keyboard injection.
C11: headless input duration → `ceil(duration_ms*60/1000)` game frames.
C12: ∀ headless run → unique temp cartridge + cwd + `.cartdata`; safe process parallelism.
C13: headless harness runs unpaced; visible replay retains Pemsa 60 Hz cadence.
C14: full visible world-state JSON deferred to T10.
C15: headless episode ends only on cartridge death; post-command input → neutral.
C16: collector train seeds ∈ `0..30000`; exactly 10 fixed eval seeds ∈ `30001..32767`; sets disjoint.
C17: teacher genome open-loop; game-ready bootstrap = `x:3`, `neutral:18`, `up:6`, `down:6`, `neutral:31`; evolved action begins when first normal enemy spawns; gene action = 8 game frames.
C18: accepted trace survival ≥1800 frames + deterministic replay match; retain ≤5 unique action hashes/train seed.
C19: pilot defaults = 5 train seeds × 100 generations; exhausted seed → unsolved/deferred.
C20: one SQLite DB; stdlib only; episode rows retain config/version + raw state; projected observation stored packed float32.
C21: generation + accepted episode checkpoint ! atomic; resume restores seed index, population, RNG, accepted hashes.
C22: collector stores per-seed historical champion; replay loads champion genome; reconstruct recovers missing champion from deterministic search.
C23: mutation preserves early survival prefix; final 25% genome receives higher exploration rate.
C24: explicit reset clears collector records in one SQLite DB; schema retained; confirmation required outside `just` recipe.
C25: collector `--resume` loads saved campaign configuration from its database; defaults/option values do not redefine a campaign.
C26: append mode extends only the stored train-seed list; all collection/evolution parameters remain unchanged.
C27: behavior-cloning baseline reads collector DB; never mutates collector rows or running campaign state.

§I

cli: `just dodge-control <commands.json|->` → launch game, start run, execute sequence, exit `0`.
cli: `just dodge-control <commands.json|-> --seed <0..32767>` → seed PICO-8 RNG before game init.
default: omitted `--seed` → seed `42`.
cli: source `-` → read JSON from stdin.
cli: invalid JSON/schema/window timeout/injection failure → stderr diagnostic + nonzero exit.
json: top-level array of `{"move":"<direction>","duration_ms":<int>}`.
enum: `<direction>` → `x|neutral|left|right|up|down|up_left|up_right|down_left|down_right`.
keys: `x|left|right|up|down` → keyboard `x|Left|Right|Up|Down` → PICO-8 `❎|btn(0..3)`.
start: first JSON command ! `x`; controller ⊥ hidden start key.
example: `src/dodge/game/movements.json` → valid runnable movement list.
cli: `just dodge-headless <commands.json|-> [--seed N]` → stdout one JSON result.
result: `{"score":number,"frames":int,"survival_frames":int,"seed":int,"started":bool,"died":bool}`.
py: `replay_commands(commands, seed)` → visible Pemsa replay; command input only.
history: `history/dodge/*.json` → winner seed, fitness, commands, epoch count, replay result.
cli: `just dodge-replay <history.json>` → visible controls-disabled replay from saved winner.
history: `history/dodge/run-*/epoch-*.json` → per-epoch winner + headless result.
cli: `just dodge-replay-run <history-dir>` → visible per-epoch replay sequence.
cli: `just dodge-replay-latest <epoch>` → visible replay requested epoch from newest saved run.
cli: `just dodge-dataset-collect [options]` → resumable GA demonstration collection into SQLite.
cli: `just dodge-dataset-replay <database> <seed>` → visible replay stored champion.
cli: `just dodge-dataset-reconstruct [options] --seed N` → recover historical champion for configured seed.
cli: `just dodge-dataset-reset [--database path]` → delete collector records; stdout removed-row counts.
cli: `just dodge-dataset-collect --resume` → continue with the database's saved collector configuration.
cli: `just dodge-dataset-collect --resume --append-seeds N` → append N sequential training seeds and continue at the first new seed.
cli: `dodge-bc-train [--database PATH] [--output PATH]` → train MLP from non-bootstrap collector steps; stdout JSON metrics.
db: one collector DB → metadata, seed roles, runs, episodes, ordered decision rows, checkpoints.

§R

R1|Pemsa `printh`|writes arguments + newline to stdout via `printf`|https://github.com/egordorichev/pemsa/blob/6c13c5879c800af33543f702a353285cfa9e6fb0/src/pemsa/util/pemsa_system_api.cpp#L35-L44
R2|Pemsa `exit`|calls emulator stop; binary probe exited `0` + flushed `printh`|https://github.com/egordorichev/pemsa/blob/6c13c5879c800af33543f702a353285cfa9e6fb0/src/pemsa/util/pemsa_system_api.cpp#L100-L103
R3|Pemsa cart data|relative `.cartdata/` path ∴ isolate cwd per process|https://github.com/egordorichev/pemsa/blob/6c13c5879c800af33543f702a353285cfa9e6fb0/src/pemsa/cart/pemsa_cartridge_module.cpp#L621-L637
R4|SDL driver selection|`SDL_VIDEODRIVER` + `SDL_AUDIODRIVER` select backends|https://wiki.libsdl.org/SDL2/FAQUsingSDL

§V

V1: ∀ JSON input valid list + exact fields/types/enums/ranges before side effects.
V2: direction → exact key set; diagonal keys held simultaneously; `neutral` → no held keys; `x` → keyboard `x`.
V3: ∀ command execute in list order for requested `duration_ms`; previous keys released before next command.
V4: ∀ injected key event targets launched Pemsa window id.
V5: normal completion | exception | signal → zero held keys.
V6: controller owns emulator lifecycle; completion/error/interrupt → emulator terminated + reaped.
V7: game startup bounded by window timeout; timeout → clear error + no orphan process.
V8: existing `just dodge-run` interactive path remains functional.
V9: first key injection waits bounded SDL/X11 settle interval after window discovery.
V10: `--seed N` → `srand(N)` first `_init` statement in launched temporary cartridge.
V11: seed absent → seed `42`; explicit valid seed overrides; invalid seed → fail before side effects.
V12: checked-in movement example conforms to JSON interface + covers start, cardinal, diagonal, neutral moves.
V13: command list nonempty + first move `x`; execution emits only listed control keys.
V14: headless cartridge overrides `_draw` with no-op + runs dummy video/audio.
V15: headless `btn`/`btnp` derive only from frame-converted JSON commands.
V16: successful headless stdout = one valid result JSON object containing final in-game `score`.
V17: headless failure → stderr diagnostic + nonzero exit + child reaped.
V18: same commands + seed + initial data → identical result JSON.
V19: ≥4 concurrent headless runs → isolated success; ⊥ shared `.cartdata` or residue.
V20: checked-in cartridge unchanged by headless runs.
V21: non-UTF-8 Pemsa diagnostics → replacement text; result parser remains operational.
V22: headless `survival_frames` → live `updategame` frames before death; ⊥ menu/transition frames.
V23: Agent fitness → headless `survival_frames`; ⊥ cartridge `score`.
V24: `dodge.control.main` → `control` CLI behavior.
V25: headless mode advances cartridge transitions without calling cartridge `_draw`.
V26: headless final input → neutral frames until cartridge death; success result `died:true`.
V27: winner replay → visible Pemsa + instrumented command input; ⊥ keyboard injection.
V28: epoch dispatches each agent once concurrently; selection/mutation starts after all results return.
V29: completed training replay → saved local history JSON with exact commands + seed + fitness.
V30: `dodge-replay` validates saved history JSON before visible replay; ⊥ keyboard injection.
V31: same commands + seed + initial cartridge → headless and visible replay result JSON equal.
V32: replay renderer ⊥ advance gameplay RNG or read host mouse state.
V33: training run creates one history directory; each evaluated epoch writes its own winner before mutation.
V34: replay-run replays epoch files ordered by epoch and fails on visible/headless result mismatch.
V35: replay-latest selects newest valid `run-*` directory; absent run → fail before replay.
V36: replay-latest `<epoch>` → only matching saved epoch; invalid or absent epoch → fail before Pemsa.
V37: next generation retains 5 epoch-ranked brains; remaining agents round-robin clone 1 then mutate.
V38: headless run repeats exact frame routine until terminal; visible replay invokes routine once per Pemsa tick.
V39: collector genome action → exactly 8 game updates; ⊥ millisecond duration gene.
V40: bootstrap starts after game-ready; `up:6` leaves center-idle box before evolved action; no bootstrap neutral settle.
V41: train seed ≤30000; eval seed >30000; exactly 10 eval seeds; ∀ run → no overlap.
V42: only replay-verified trace with survival ≥1800 → accepted dataset episode.
V43: ∀ train seed → ≤5 accepted distinct action hashes; bootstrap ∉ hash.
V44: accepted episode + ordered raw/projected state-action rows → one SQLite transaction; ⊥ partial episode.
V45: resume → same pending seed, generation, population, RNG, accepted hashes as interrupted run.
V46: accepted `steps` → action labels `neutral,up,down,neutral,*genome`; bootstrap rows=4; `observation_f32`=221 little-endian f32 values.
V47: same action hash ∈ multiple training seeds; hash unique only within seed.
V48: `Genome` = `tuple[Direction, ...]`; collector action sets, population, checkpoint, trace labels preserve `Direction`.
V49: ∀ completed generation → DB champion retains greatest survival genome seen for seed; ties retain earlier champion.
V50: champion replay → stored genome + stored seed; reconstruction uses stored collector config + deterministic evolution seed.
V51: bootstrap neutral phase ends on first normal enemy spawn; evolved action starts after enemy-visible state.
V52: mutation rate = 2% first 75% genome, 20% final 25%; ranked elites unchanged.
V53: reset without `--yes` → fail; confirmed reset deletes episodes, steps, champions, checkpoint, seeds, metadata; schema retained.
V54: accepted trace rows pair only states with next scheduled action; ⊥ terminal or post-script idle state rows.
V55: `dodge-dataset-collect --resume` → loads and validates stored collector config before `collect`; ⊥ CLI default configuration comparison.
V56: append mode atomically records only sequential new training seeds + enlarged config; checkpoint/RNG/population retained and next run starts at its existing seed index.
V57: behavior-cloning loader reads only non-bootstrap `steps`; ∀ row → 221 little-endian f32 observation + one known direction label.
V58: MLP ∀ batch `(N,221)` → logits `(N,9)` ordered by collector action choices; invalid feature shape → fail.
V59: legacy `Brain` mutation chance `0` → no action replacement or optional additions.

§T

id|status|task|cites
T1|x|add JSON model, full-list validation, direction mapping|V1,V2,I.json,I.enum,I.keys
T2|x|add targeted X11 keyboard backend + Pemsa lifecycle|V3,V4,V5,V6,V7,I.start
T3|x|add CLI, devenv dependency/scripts, just recipe, docs|V8,I.cli,C4,C5
T4|x|add unit tests + controlled-run smoke test|V1,V2,V3,V4,V5,V6,V7,V8,V9
T5|x|add deterministic seed option via temporary cartridge|V10,V11,I.cli,C9
T6|x|default control seed to `42`|V10,V11,I.cli
T7|x|add runnable example movement file|V1,V12,I.json,I.example
T8|x|make menu-start X explicit in command list|V1,V2,V3,V12,V13,I.json,I.start
T9|x|add isolated headless command runner + score JSON|V1,V10,V11,V13,V14,V15,V16,V17,V18,V19,V20,V21,I.cli,I.result
T10|.|expose per-frame visible world state as JSON|V14,V15,C14
T11|x|add headless survival telemetry + agent fitness|V22,V23,V25,I.result
T12|x|retain `dodge.control.main` compatibility alias|V24
T13|x|end episode on death + replay winner visually|V26,V27,C15,I.result,I.py
T14|x|parallelize agent evaluation per epoch|V19,V28
T15|x|save winner history + replay history command|V29,V30,I.history,I.cli
T16|x|unify headless and visible replay simulation state|V31,V32
T17|x|save per-epoch winners + replay run sequence|V33,V34,I.history,I.cli
T18|x|replay newest run command|V35,I.cli
T19|x|replay requested epoch from newest run|V36,I.cli
T20|x|retain five ranked elite lineages|V37
T21|~|unpace headless simulation; retain visible 60 Hz replay|V31,V38,C13
T22|x|capture headless game-ready state/action trace|V39,V40,I.json
T23|x|add resumable open-loop collector + SQLite dataset|V41,V42,V43,V44,V45,V46,V47,V48,C16,C17,C18,C19,C20,C21,I.cli,I.db
T24|x|add collector recipe + focused behavior tests|V39,V40,V41,V42,V43,V44,V45,V46,V47,I.cli
T25|x|persist/replay/reconstruct per-seed champion|V49,V50,C22,I.cli,I.db
T26|x|increase action horizon + delay genes until first enemy|V39,V46,V51,C17
T27|x|bias mutation toward late-game actions|V52,C23
T28|x|add explicit collector reset command|V53,C24,I.cli,I.db
T29|x|exclude terminal/post-script trace states from accepted rows|V54,V44
T30|x|load stored collector configuration for bare resume|V55,C25,I.cli
T31|x|append training seeds to completed collector campaign|V56,C26,I.cli,I.db
T32|x|add documented MLP behavior-cloning baseline + SQLite reader|V57,V58,C27,I.cli
T33|x|fix zero-rate legacy Dodge brain mutation|V59

§B

id|date|cause|fix
B1|2026-08-17|Pemsa window discovered before SDL X11 init settled|V9
B2|2026-08-17|Pemsa glyph diagnostics contain invalid UTF-8 bytes|V21
B3|2026-08-19|`main` renamed `control`; control tests/import callers broke|V24
B4|2026-08-19|no-op headless `_draw` froze cartridge draw-driven transitions|V25
B5|2026-08-20|visible draw-driven transition clock diverged from headless update clock|V31
B6|2026-08-20|visible draw consumed gameplay RNG + host mouse state|V32
B7|2026-08-20|new replay lines exceeded Ruff width|mechanical wrap
B8|2026-08-20|new elite lines exceeded Ruff width|mechanical wrap
B9|2026-08-20|new elite print bypassed Ruff formatter|mechanical format
B10|2026-08-23|legacy 197 projection + bootstrap labels omitted neutral|V46
B11|2026-08-23|checkpoint test imports unsorted|ruff fix
B12|2026-08-23|checkpoint test wrote before schema init|fixture init
B13|2026-08-23|justfile interpolation spacing stale|just fmt
B14|2026-08-23|collector genome widened `Direction` to `str`|V48
B15|2026-08-23|collector log line + test imports violate Ruff format|mechanical format
B16|2026-08-23|progress test mocked 5 scores for 50 genomes|fixture population=5
B17|2026-08-23|champion SQL + test import violate Ruff|mechanical format
B18|2026-08-23|new replay recipe bypassed `uv run` console script|recipe wrapper
B19|2026-08-23|post-script trace state had no action label|V54
B20|2026-08-23|bare resume rebuilt CLI defaults instead of loading database campaign|V55
B21|2026-08-23|completed campaign had no way to extend stored seed set without changing parameters|V56
B22|2026-08-23|new imitation modules retained unused imports|mechanical ruff
B23|2026-08-23|legacy zero mutation rate appended optional random actions|V59
B24|2026-08-23|NEAT recipe test expected pre-formatter interpolation spelling|mechanical test update
B25|2026-08-23|parallel collector delayed visible Pemsa input acknowledgement|external resource contention
