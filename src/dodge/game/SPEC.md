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
C28: PyTorch ∈ default project dependencies; Marimo ∈ GPU dependency group; trainer device=auto.
C29: cloud transfer uses SQLite backup snapshot; ⊥ copy live WAL main file as dataset handoff.
C30: development-validation seeds = `29991..30000`; ⊥ collector training or MLP gradients on them.
C31: behavior-cloning validation = top 10 accepted training seed IDs; split recorded per training history; validation ⊥ gradients.
C32: native oracle/runtime additive; checked-in `dodge.p8` remains immutable source + oracle.
C33: canonical native trace = full cartridge draw path + indexed 128×128 pixels; legacy no-op headless trace remains distinct.
C34: canonical trace artifact includes source/Pemsa identity, seed, initial state, action schedule, capture mode, frame records, and hashes.
C35: native extraction reads only checked-in cartridge bytes; generated output records generator version + source hash.
C36: extracted gfx preserves 128×128 palette indices; sprite, sfx, music, and proven static-table records preserve source order + identity.
C37: compatibility helpers encode only probed PICO-8 behavior; unresolved behavior-affecting semantics block native phase handoff.
C38: generated asset output is disposable; stale or mismatched source/generator identity fails before consumption.
C39: P3 Rust workspace uses pinned stable toolchain; `dodge-core` has no Macroquad, Python, Pemsa, xdotool, subprocess, or window dependency.
C40: P3 slice scope = menu start → transition → player movement + representative normal enemy; ⊥ complete-game parity claim.
C41: native slice state uses named typed Rust fields; ⊥ opaque Lua table or untyped remaining-state escape hatch.
C42: native frame/action APIs separate one-frame inspection from multi-frame training stepping; state + indexed framebuffer share one simulation.

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
py: `encode_board(RawState)` → float32 `(19,16,16)` spatial board tensor.
py: `predict_action(BehaviorCloningCNN, RawState, mean, std)` → next `Direction`.
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
cli: `dodge-bc-train [--database PATH] [--output PATH] [--device auto|cuda|cpu]` → per-epoch loss lines then final stdout JSON metrics.
cli: `dodge-bc-plot HISTORY [--output PATH]` → PNG train/validation loss plot.
cli: `dodge-dataset-export OUTPUT [--database PATH]` → write consistent read-only collector SQLite snapshot.
notebook: `notebooks/dodge_behavior_cloning.py` → Marimo GPU training UI; imports reusable `dodge.imitation` code.
db: one collector DB → metadata, seed roles, runs, episodes, ordered decision rows, checkpoints.
db: seed roles → `training|validation|evaluation`.
cli: `dodge-native-oracle --commands FILE --seed N --output FILE` → canonical full-draw trace JSON.
file: canonical trace → schema version, provenance, source/section hashes, scenario, ordered frames, terminal result, state/pixel hashes.
json: frame → frame index, state, reward/done/events, pixel rows or canonical pixel buffer.
cli: `dodge-native-extract-assets --source PATH --output DIR` → deterministic indexed assets, source map, compatibility manifest, and hashes.
file: native asset manifest → generator version, cartridge identity, section identities, asset paths/content hashes, source-map hash, and compatibility report hash.
file: native source map → PICO-8 symbol/section/source span → planned native module, conversion note, and parity status.
api: `PicoCompat` → numeric/RNG/input/stat/palette/camera/fill/raster/sprite/sound compatibility primitives.
api: `dodge_core::NativeGame` → reset, advance one frame, exact action step, typed snapshot.
api: `dodge_core::Snapshot` → typed slice state, 16,384 indexed pixels, render state, provenance, canonical bytes.
cli: `dodge-native-runner` → consume P1 action JSON, execute native serial slice, emit canonical snapshot/frame JSON.
file: native differential report → first frame, field or pixel coordinate, expected/actual values, source-map span.

§R

R1|Pemsa `printh`|writes arguments + newline to stdout via `printf`|https://github.com/egordorichev/pemsa/blob/6c13c5879c800af33543f702a353285cfa9e6fb0/src/pemsa/util/pemsa_system_api.cpp#L35-L44
R2|Pemsa `exit`|calls emulator stop; binary probe exited `0` + flushed `printh`|https://github.com/egordorichev/pemsa/blob/6c13c5879c800af33543f702a353285cfa9e6fb0/src/pemsa/util/pemsa_system_api.cpp#L100-L103
R3|Pemsa cart data|relative `.cartdata/` path ∴ isolate cwd per process|https://github.com/egordorichev/pemsa/blob/6c13c5879c800af33543f702a353285cfa9e6fb0/src/pemsa/cart/pemsa_cartridge_module.cpp#L621-L637
R4|SDL driver selection|`SDL_VIDEODRIVER` + `SDL_AUDIODRIVER` select backends|https://wiki.libsdl.org/SDL2/FAQUsingSDL
R5|Pemsa math bridge|`sin`/`cos` receive fixed values as `float`, evaluate C math, narrow to `float`, then convert back to fixed|https://github.com/egordorichev/pemsa/blob/master/src/pemsa/util/pemsa_math_api.cpp

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
V49: ∀ completed generation → DB champion retains greatest `(survival_frames, neutral_actions)` genome seen for seed; exact fitness ties retain earlier champion.
V50: champion replay → stored genome + stored seed; reconstruction uses stored collector config + deterministic evolution seed.
V51: bootstrap neutral phase ends on first normal enemy spawn; evolved action starts after enemy-visible state.
V52: mutation rate = 2% first 75% genome, 20% final 25%; ranked elites unchanged.
V53: reset without `--yes` → fail; confirmed reset deletes episodes, steps, champions, checkpoint, seeds, metadata; schema retained.
V54: accepted trace rows pair only states with next scheduled action; ⊥ terminal or post-script idle state rows.
V55: `dodge-dataset-collect --resume` → loads and validates stored collector config before `collect`; ⊥ CLI default configuration comparison.
V56: append mode atomically records only sequential new training seeds + enlarged config; checkpoint/RNG/population retained and next run starts at its existing seed index.
V57: behavior-cloning loader reads only non-bootstrap training `steps`; ∀ row → validates 221 little-endian f32 projection, decodes raw state to float32 `(19,16,16)` board tensor, + one known direction label.
V58: `BehaviorCloningCNN` ∀ batch `(N,19,16,16)` → logits `(N,9)` ordered by collector action choices; invalid board shape → fail.
V59: legacy `Brain` mutation chance `0` → no action replacement or optional additions.
V60: dataset export uses SQLite backup API; source collector DB unchanged; snapshot includes committed WAL state.
V61: `dodge-bc-train` defaults to auto: available CUDA → CUDA; otherwise CPU; unavailable explicit CUDA → fail before training.
V62: Marimo notebook cells ⊥ conditional `return`; notebook registers without `SyntaxError`.
V63: notebook startup adds repository `src/` to import path before `dodge` import.
V64: `--resume --workers N` changes only collector process parallelism; stored campaign config unchanged.
V65: ∀ headless Pemsa run → `SDL_RENDER_DRIVER=software`.
V66: `29991..30000` ∈ validation role; train config ∩ validation = ∅; resume backfills missing validation rows.
V67: behavior-cloning observation normalization has finite positive standard deviation, including one-row datasets.
V68: ∀ completed behavior-cloning epoch → stdout `epoch=N/T train_loss=F [validation_loss=F]`; final JSON metrics last line.
V69: behavior-cloning split → highest 10 accepted training seed IDs validation; validation rows ∉ optimizer; split ∈ saved history.
V70: training history → per-epoch train/validation loss; `dodge-bc-plot` → PNG loss curves.
V71: saved behavior-cloning artifact → CNN model type, board shape/channels, nine actions, state dict, mean, and standard deviation.
V72: native oracle input bytes + source/section hashes unchanged during capture.
V73: same native-oracle scenario repeated ≥3 → identical canonical frames/results/hashes; timestamp excluded.
V74: full-draw and no-op-headless capture modes labeled distinct; canonical mode preserves cartridge draw RNG/side effects.
V75: canonical frame pixels → exactly 128×128 palette indices in row-major order; pixel hash covers all indices.
V76: canonical frame boundary → one post-update/post-draw record; frame index monotonic; terminal frame captured before exit.
V77: oracle failure/timeout/parse error → owned Pemsa/Xvfb reaped + no accepted partial output.
V78: canonical trace → provenance identifies cartridge, Pemsa, seed, action schedule, capture mode, and schema version.
V79: compatibility probes emit each optional/nullable PICO-8 value as a separate validated record; no unvalidated value enters a concatenated record.
V80: compatibility probes encode PICO-8 booleans as explicit numeric `0|1` values before `tostr`; probe output never concatenates `tostr(boolean)`.
V81: harness transition dispatch enters only when `_upd` is present and equals `updatetransition`; absent callbacks never execute transition state.
V82: canonical frame metadata records current/previous input masks, lifecycle mode, and dead flag at the same post-update/post-draw boundary as state and pixels.
V83: capture emits at most one canonical post-update/post-draw record for each game-frame index; duplicate renderer ticks cannot create accepted duplicate frames.
V84: extraction source identity + every section hash equal checked-in input before and after extraction; source mutation fails.
V85: decoded gfx contains exactly 16,384 row-major values ∈ `0..15`; source rows and implicit zero rows recorded; reassembly matches indexed image.
V86: sfx records retain 64 source identities and 168-hex payloads; music records retain 32 ordered source lines; no lossy audio normalization.
V87: same source bytes + generator version → byte-identical asset tree, manifest, source map, and compatibility report.
V88: every generated asset/compatibility symbol maps to source section + span; every behavior-affecting cartridge function is classified; unresolved status blocks handoff.
V89: numeric/RNG/input/stat probes record seed, source/Pemsa identity, separate values, and expected native representation; no unprobed substitution accepted.
V90: software raster state preserves palette remap, transparency, fill pattern, camera, primitive order, and indexed pixels without RGB conversion.
V91: stale output with changed source hash or generator version fails validation before asset load.
V92: P2 acceptance report names accepted, deferred, and unresolved semantics; deferred/unresolved behavior-affecting entries prevent P3 acceptance.
V93: native workspace toolchain + lock identity recorded; `dodge-core` dependency graph excludes viewer, Python, Pemsa, process, and window APIs.
V94: same native seed + config + initial persistent state → byte-identical reset Snapshot, typed state hash, and indexed pixels.
V95: nine actions → exact PICO-8 masks; neutral clears held input; `advance_frame` increments one; `step(action,n)` advances exactly n nonterminal frames.
V96: every P3 frame exposes named lifecycle, player, representative enemy, input, RNG, reward, terminal, and indexed-render fields; no opaque state.
V97: native renderer consumes simulation state and owns indexed framebuffer; draw cannot advance gameplay RNG, input, timers, or entities.
V98: Snapshot canonical serialization roundtrip → identical bytes, typed state, render state, and pixel hash.
V99: serial native runner consumes P1 scenarios without emulator IPC; invalid action/frame count fails before simulation mutation.
V100: native/Pemsa mismatch → first frame + field path or pixel coordinate + expected/actual + source-map span; aggregate hashes alone insufficient.
V101: P3 corpus acceptance report names exact slice boundary, corpus, parity result, deferred systems, and P4 handoff; no full-port claim.
V102: menu start `btnp` → transition advances on same frame; default game-ready frame = 13.
V103: collision/death sets the terminal flag while preserving the update-game mode; terminal reward/survival accounting stops on that frame.
V104: the native transition render cursor includes the same-frame transition tick; the source draw-side game update begins on the accepted frame-seven boundary and yields the seed-42 friendly spawn at frame 57.
V105: post-update frame metadata exposes current and previous masks at the same completed-frame boundary as the oracle; held input therefore records equal masks while `btnp` still sees the pre-frame edge during simulation.
V106: a command boundary may apply one simulation mask and publish a different cleared post-frame mask; the ordinary `advance_frame(mask)` path remains same-mask.
V107: native reset consumes the deterministic `initpatterns` random draws before frame updates, including its seed-dependent branch (21 or 25 draws); each source spawn roll therefore sees the same RNG stream for its seed.
V108: a normal spawn stores its selected `es` value globally; later friendly corner spawns use that current maximum size until another normal spawn changes it.
V109: normal-enemy overlap uses the source ordered loop: already-marked enemies are removed while iterating, each removed normal enemy consumes 11 shatter draws, adds 0.5 score, and applies the half-step speed/spawn-rate curve.
V110: native enemy-loop mutations use checked collection access under the workspace no-panic lint profile.
V111: player collision removes each colliding normal enemy at the source terminal boundary and consumes its 11 shatter RNG draws without adding score.
V112: normal spawns use the source low-score personality weighting, and personality-1 enemies reduce speed by 0.01 while within the source 25-pixel radius.
V113: personality-2-or-higher enemies expose an 8-pixel collision/render extent while retaining their typed internal size and source circle primitive.
V114: full-draw frame accounting consumes exactly two shared RNG draws per visible personality-2-or-higher trail while snapshot rasterization itself remains state-pure.
V115: power-up collision uses the source four-pixel expanded bounds; personality 4 removes the enemy, sets player size to 2, adds one score, applies full difficulty, and does not set terminal.
V116: the native pattern scheduler reaches the source first-pattern boundary after 420 game updates and consumes its one selection RNG draw before later frame-side effects.
V117: canonical differential frames include per-frame reward and ordered events; missing or mismatched values fail before state/pixel comparison.
V118: P3 report separates accepted logical-slice parity from deferred visual/full-state parity; no P3 artifact claims complete-game equivalence.
V119: every behavior-affecting source-manifest field has one typed native field or reviewed derived representation; no opaque remainder.
V120: same seed, persistent initialization, settings, action schedule, and frame boundary → identical full canonical trace through terminal state.
V121: native frame systems execute source-observed order; no hidden RNG, input, timer, list, or draw-side mutation.
V122: restore(snapshot) + same next input → identical next state, pixels, reward, events, done, and hashes.
V123: every active enemy, particle, pattern, rectangle, warning, timer, flag, setting, and side effect appears in FullState or proves derived.
V124: native indexed framebuffer equals canonical oracle pixels at every accepted frame; RGB/GPU output never proves parity.
V125: full-draw and renderless paths preserve classified draw-side effects and RNG accounting; render cannot silently alter simulation.
V126: camera, palette, fill pattern, clipping, sprite transparency, built-in text, primitive order, and indexed raster match PICO-8.
V127: input, stat, mouse mode, persistent reads, and external state remain replay-controlled; host state cannot alter canonical trace.
V128: reward, death, survival frames, score, transition boundaries, terminal events, and legacy Python contracts remain equal.
V129: any full-port mismatch blocks acceptance and records first frame, field/pixel, source span, expected/actual, and RNG delta.
V130: existing Python control, headless, NEAT, PPO, native-oracle, and extraction regression suites remain green.
V131: canonical indexed pixels use the same camera-relative coordinate space as
the source `pget` capture; camera movement affects the retained clear edge while
world primitives remain at their queried coordinates.
V132: indexed primitive colors include palette index zero; sprite transparency
applies only to sprite source pixels and never suppresses `pset`, line, rect, or
text writes.
V133: draw-side particle additions occur only when the source dispatch actually
renders `drawgame`; menu/settings transition frames expose no game trail.
V134: if the source renderer performs a duplicate draw after a canonical capture
callback, its side effects affect only the next frame and never the captured
frame's state or pixels.
V135: camera edge retention is determined by PICO's quantized integer camera
offset, so subpixel values use the same rounding boundary as the source and
can expose a retained clear edge.
V136: draw-side trails use the same floored enemy coordinates as the source
 draw loop before applying their random trigonometric offsets.
V137: pattern line endpoints preserve source fractional interpolation until the
 PICO line primitive's coordinate conversion; they are not rounded by the
 pattern caller.
V138: active pattern shape and timer updates occur after enemy updates and use
 the current source freeze-rate multiplier before the next draw.
V139: dead frames skip active-game primitives and render the source game-over
prompts, icons, logo, and centered score with the current input mode.
V140: each kamikaze enemy updates its proximity speed exactly once per source
enemy-update frame before movement; the post-movement position never performs a
second speed adjustment.
V141: opening `TransitionToGame` frames render source game-draw state beneath
exposed transition rows; overlay cursor and fill remain source-aligned.
V142: target-2 transitions render source `from` state while cursor >0 and target
settings state while cursor ≤0; game draw-side effects follow exposed layer.
V143: when target-2 exposes a game `from` state, the source invokes `updategame`
once from `drawtransition` after the outer update; the initiating frame therefore
executes both source updategame calls before its captured draw.
V144: the first menu X edge latches `hasplayed=true` before any later settings
frame; settings then locks gameplay rows and routes X back to game.
V145: transition mode completion follows the source draw cursor after its
same-frame ±10 tick: target 1/0 becomes active at cursor ≥128 and target 2 at
cursor ≤−128, even when the lifecycle bookkeeping counter has a different
intermediate value.
V146: the post-capture duplicate-draw trail compensation applies to the initial
menu-to-game transition only; settings-to-game completion contributes one
draw-side trail and no extra next-frame particle.
V147: terminal rendering preserves the camera-directed physical-screen edge
through the normal clear/project path; it does not force one physical row to a
constant color.
V148: a normal collision followed by `die` applies the source shake increment
 twice, once in `collide` and once in `die`, before the next `_update60` camera
 sample.
V149: player collision is enabled only while `shouldcollide` is true, and enemy
overlap marking is enabled only while `eshouldcollide` is true; both controls
are part of the canonical replay state.
V150: every source `difficultycurve(half)` call uses the difficulty-indexed speed,
spawn-rate, static-bounce, and moving-bounce increments and targets, dividing
each increment by two only for the `half` call.
V151: difficulty configuration maps only to bounded curve-table entries; invalid
configuration values cannot reach unchecked table indexing.
V152: native snapshot wire-version changes update every producer, decoder, and
fixture constructor together; current fixtures decode through the current field
layout before differential comparison.
V153: P2 collision iterates source mutable enemy order, including newly appended
P1 kamikaze entries; each visited entry adds one score and full difficulty step,
non-P1/non-P2 explosion entries shatter before removal, and initiating P2 is
visited again after its outer collision shatter.
V154: enemy updates preserve source pre-growth local position/size for pattern and
crush checks, then use post-growth entity position/size for overlap, edge bounce,
kamikaze speed, and movement.
V155: pattern metadata preserves source `special` numeric values, including 1.1,
separately from `autovar`; generated constructors and snapshots retain both
identities without sentinel substitutions.
V156: native rounding uses the source modulo-positive fractional part for
negative coordinates, so pattern interpolation and raster coordinates round
with the same floor/half-up boundary as PICO-8.
V157: active patterns generate source warnings, apply rectangle targets in order,
perform visibility/fade completion, restore default friendly/spawn state, and
reinitialize generated pattern data while preserving counters/probabilities.
V158: audio events preserve source call order, identity, and channel metadata;
terminal restart resets gameplay fields in-place while retaining the active
game lifecycle boundary.
V159: source `sin`/`cos` converts fixed input to `f32`, evaluates the Pemsa C
double-math path, narrows the result to `f32`, then converts to Q16.16; native
code never substitutes `f32`-only trig evaluation.

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
T32|x|add documented board-CNN behavior-cloning baseline + raw-state SQLite reader|V57,V58,V71,C27,I.cli
T33|x|fix zero-rate legacy Dodge brain mutation|V59
T34|x|add Marimo GPU notebook + SQLite snapshot cloud handoff|V60,V61,C28,C29,I.cli
T35|x|fix Marimo conditional-return cell compile failure|V62,I.notebook
T36|x|fix Molab repository `dodge` import path|V63,I.notebook
T37|x|stabilize Molab headless collector runtime|V64,V65,C10,C21,I.cli
T38|x|reserve seed-held development validation partition|V66,C30,I.db
T39|x|auto-select Dodge training CUDA or CPU; make PyTorch local dependency|V61,V67,C28,I.cli
T40|x|log behavior-cloning loss after each epoch|V68,I.cli
T41|x|add behavior-cloning validation loss history + plot CLI|V69,V70,C31,I.cli
T42|x|add hash-addressed cartridge/section manifest|V72,V78,I.file
T43|x|define canonical full-draw trace schema + provenance|V73,V74,V76,V78,I.file
T44|x|capture post-draw indexed pixels + full-draw frame records|V74,V75,V76
T45|x|add native-oracle CLI, parser tests, and cleanup/error tests|V73,V77,V78,I.cli
T46|x|extract p8 sections + source hash manifest|V84,V87,I.cli
T47|x|decode indexed gfx/palette/sprite + preserve sfx/music records|V85,V86,V90,I.file
T48|x|build static-table/source-span map + unresolved symbol inventory|V88,V92,I.file
T49|x|add PicoCompat numeric/RNG/input/stat probes|V89,I.api
T50|x|add indexed software raster palette/fill/camera primitives|V90,I.api
T51|x|add primitive differential fixtures against P1 indexed frames|V85,V89,V90
T52|x|add stale-output + unresolved-symbol validation|V84,V87,V88,V91,V92
T53|x|produce P2 conversion map + accepted compatibility report|V84-V92,I.file
T54|x|scaffold pinned Rust workspace + engine-free `dodge-core`|C39,V93
T55|x|port lifecycle, menu transition, actions, numeric helpers, and RNG state|C40,V94,V95
T56|x|port player movement, bounds, normal enemy, collision, reward, terminal behavior|C40,V95,V96,V103,V104,V111,V112,V115
T57|x|implement typed Snapshot, canonical serialization, and indexed framebuffer ownership|C41,C42,V96-V98,V113,V114,V116,I.api
T58|x|add serial native runner for P1 scenarios without emulator IPC|C39,V95,V99,I.cli
T59|x|add field/pixel differential comparison + source-map diagnostics|V100,I.file
T60|x|run defined slice corpus every frame + resolve in-scope logical mismatches; classify visual boundary for P4|V94-V100,V117,V118
T61|x|produce P3 vertical-slice acceptance report + P4 handoff|V94-V101,V117,V118,I.file
P4-T1|x|port lifecycle, initialization, settings, menu, transitions, persistent state, and high-score boundaries|V119,V120,V128,V143-V145
P4-T2|x|port player movement, collision, death, progression, freeze, sizing, and difficulty behavior|V120,V121,V128,V149,V150,V153
P4-T3|x|port particles, trails, enemy families, growth/shrink/death states, and spawn logic|V119,V121,V123,V125,V154,V159
P4-T4|x|port pattern tables, dynamic variants, interpolation, warnings, visibility, and completion|V119,V120,V121,V123,V157
P4-T5|x|port complete indexed draw path, palette/camera/fill state, sprite/text primitives, and sound events|V124,V125,V126,V158
P4-T6|x|expand FullState inventory, restore, canonical serializer, and source-map coverage|V119,V122,V123
P4-T7|x|run full corpus frame-by-frame; add targeted fixtures for every first mismatch|V120,V124,V129,V159
P4-T8|x|run held-out randomized traces and legacy regression suite|V120,V127,V130
P4-T9|.|produce P4 acceptance report and stable core handoff to P5/P6|V119-V130

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
B26|2026-08-23|optional PyTorch test import precedes project imports|mechanical Ruff exception
B27|2026-08-23|baseline `rl_learning` files retain unused imports|pre-existing full Ruff block
B28|2026-08-23|new dataset test split `test_v53` fixture scope|mechanical test placement
B29|2026-08-23|Marimo cell `return` nested in conditional became top-level `return`|V62
B30|2026-08-23|Marimo training cell used `mo.stop` without `mo` dependency|mechanical cell input
B31|2026-08-23|notebook execution path omitted repository `src/`|V63
B32|2026-08-23|concurrent Molab headless Pemsa child segfaulted|V64,V65
B33|2026-08-24|CPU fallback test imports were not Ruff-sorted|mechanical Ruff fix
B34|2026-08-24|one-row PyTorch `std` used unbiased estimator and produced NaN|V67
B35|2026-08-24|validation test train imports violated Ruff ordering|mechanical Ruff fix
B36|2026-08-27|CNN loader retained legacy 221-float assignment after board allocation|V57
B37|2026-09-01|compatibility probe concatenated multiple PICO-8 values and masked a runtime semantic boundary|V79
B38|2026-09-01|input probe passed a PICO-8 boolean directly to `tostr`, which returned nil|V80
B39|2026-09-01|probe harness treated two absent Lua callbacks as equal and entered transition code with nil state|V81
B40|2026-09-01|visible Pemsa rendered the same update counter more than once before the next update|V83
B41|2026-09-01|plain devenv pytest did not expose xdotool's transitive libXext runtime path and hid the process failure as a window timeout|external runtime setup
B42|2026-09-01|initial P2 extractor verification found line-width and import-order violations|mechanical format
B43|2026-09-01|T47 asset parity tests exceeded configured line width|mechanical format
B44|2026-09-01|T48 source-map test declaration exceeded configured line width|mechanical format
B45|2026-09-01|T49 compatibility report verification found unused import and line-width violations|mechanical format
B46|2026-09-01|T49 compatibility test imports required Ruff reordering|mechanical format
B47|2026-09-01|T50 raster implementation verification found line-width violations|mechanical format
B48|2026-09-01|fill-pattern reset value was reused as 16-bit validation maximum after Pemsa orientation fix|V90
B49|2026-09-01|T50 clip test expected a pixel outside its declared 2x2 clip|mechanical test fix
B50|2026-09-01|T51 raster fixture imports required Ruff reordering|mechanical format
B51|2026-09-01|T52 asset validation test import order and declaration exceeded configured style|mechanical format
B52|2026-09-01|T53 manifest installer verification found line-width violations|mechanical format
B53|2026-09-02|native lifecycle advanced transition one frame after menu start|V102
B54|2026-09-02|PicoFixed helpers shadowed standard arithmetic trait names|mechanical Clippy fix
B55|2026-09-02|T56 collision path called an unimplemented lifecycle death transition|V103
B56|2026-09-02|movement test compared a fixed result with an inexact decimal f32 conversion|mechanical test expectation
B57|2026-09-02|collision fixture placed a size-one normal enemy outside the source strict boundary|mechanical test fixture
B58|2026-09-02|native transition render cursor omitted the same-frame ten-pixel tick|V104
B59|2026-09-02|enemy movement test compared negative fixed arithmetic with an inexact decimal f32 conversion|mechanical test expectation
B60|2026-09-02|snapshot-owned FrameResult made legacy tests consume non-Copy Results|mechanical test ownership fix
B61|2026-09-02|snapshot module imported its own FullState type|mechanical import fix
B62|2026-09-02|snapshot validation and truncation test used unchecked indexing syntax|V8
B63|2026-09-02|runner duration conversion manually reimplemented integer ceiling division|mechanical Clippy fix
B64|2026-09-02|native snapshots exposed pre-finalization input history unlike the post-draw oracle boundary|V105
B65|2026-09-02|differential diagnostics initially violated Ruff import and line-width rules|mechanical format
B66|2026-09-02|differential fixture construction exceeded the configured line width|mechanical format
B67|2026-09-02|differential fixture timer packing shifted the canonical render payload|V98
B68|2026-09-02|runner command-boundary frame needed simulation input before post-draw mask clearing|V106
B69|2026-09-02|P3 logical differential reached the source RNG-selected normal-enemy size at frame 130|T60
B70|2026-09-02|P3 pixel differential reached the unported menu sprite at frame 1|T60,P4
B71|2026-09-02|native reset omitted the seed-dependent `initpatterns` RNG draws and shifted every gameplay spawn roll|V107
B72|2026-09-02|native normal spawns did not persist the selected `es` value for later friendly spawns|V108
B73|2026-09-02|native omitted ordered normal-enemy overlap deletion and its shatter/difficulty side effects|V109
B74|2026-09-02|V109 test borrowed logical state from a temporary Snapshot|bind Snapshot before borrowing fields
B75|2026-09-02|ordered enemy loop used unchecked vector indexing rejected by the native no-panic lint profile|V110
B76|2026-09-02|checked enemy mutation triggered Clippy collapsible-if under the warnings-as-errors gate|mechanical conditional rewrite
B77|2026-09-02|terminal collision parity retained colliding enemies that source `collide()` deletes|V111
B78|2026-09-02|seed-42 corpus reached source personality-1 radius-dependent enemy speed|V112
B79|2026-09-02|full-draw corpus reached source power-up trail RNG side effects|V114
B80|2026-09-02|seed-42 corpus reached the source personality-4 collision branch|V115
B81|2026-09-02|seed-42 corpus reached source first-pattern selection RNG before a later spawn|V116
B82|2026-09-02|differential comparison omitted canonical per-frame reward and ordered side-effect events|V117
B83|2026-09-02|snapshot module imported the FullState type it defines while extending the canonical state wire|V123
B84|2026-09-02|Python native snapshot decoder retained the pre-particle wire offsets|V123
B85|2026-09-02|native raster stored physical camera-shifted coordinates while the oracle captures camera-relative `pget` coordinates|V131
B86|2026-09-02|native raster reused sprite transparency when drawing a color-zero shatter particle|V132
B87|2026-09-02|particle oracle capture introduced import-order and line-width violations|mechanical format
B88|2026-09-02|native added the player trail on every transition instead of only game-rendered transition frames|V133
B89|2026-09-02|Pemsa emitted a duplicate transition draw after the canonical frame callback and advanced the next frame's particle list|V134
B90|2026-09-02|native clear classified a negative subpixel camera as an exposed edge although PICO truncates the camera offset|V135
B91|2026-09-02|persistent-screen renderer reborrowed a mutable framebuffer reference after changing its signature|mechanical borrow fix
B92|2026-09-02|v4 differential fixture supplied one fewer timer/state value than its packed wire format requires|fixture field-count fix
B93|2026-09-02|power-up trail side effects used the enemy's fixed-point position instead of the draw loop's floored coordinates|V136
B94|2026-09-02|game-side trail fix referenced a renderer-private coordinate helper|mechanical module boundary fix
B95|2026-09-02|pattern renderer introduced a missing fixed-to-pixel rounding helper|mechanical module boundary fix
B96|2026-09-02|pattern line interpolation rounded a fractional endpoint before the source line primitive converted it|V137
B97|2026-09-02|pattern selection was represented without advancing active rectangle opening state|V138
B98|2026-09-02|pattern opening fill applied the dotted mask using the destination row instead of the source fillp anchor row|V126
B99|2026-09-02|pattern opening used filled rectangles although the cartridge uses patterned outline rectangles|V126
B100|2026-09-02|pattern closing geometry used the clamped one-pixel progress instead of the source raw stage minus one|V137
B101|2026-09-02|terminal frames omitted the source drawgame game-over branch|V139
B102|2026-09-02|adding the terminal high-score flag left the Python fixture on the old packed field shape|V123
B103|2026-09-02|terminal draw selection used the physical mask instead of the cartridge keyboard-or-mouse input mode|V127,V139
B104|2026-09-02|terminal camera projection exposed the source-cleared physical bottom row as background|V124,V126
B105|2026-09-02|expanding EnemyState and NativeConfig left constructors and difficulty helpers incomplete|V119
B106|2026-09-02|enemy collision helper returned an unused removal value in a unit match|mechanical Rust type fix
B107|2026-09-02|native kamikaze speed was adjusted again after movement, shortening the source trajectory|V140
B108|2026-09-02|settings expansion used a non-const clamp and passed i16 coordinates into an i32 raster API|mechanical Rust type fix
B109|2026-09-02|adding settings lifecycle modes left the runner's exhaustive mode-name match stale|mechanical Rust match fix
B110|2026-09-02|strict no-panic Clippy exposed unchecked fixed-array and framebuffer indexing in the expanded renderer|V110
B111|2026-09-02|strict Clippy caught the active-pattern delay constant before the scheduler consumed it|mechanical Rust warning fix
B112|2026-09-02|adding typed settings and transition provenance changed the native snapshot payload without advancing its wire version|V123
B113|2026-09-02|expanded transition renderer selected menu beneath exposed opening game-transition rows|V141
B114|2026-09-02|settings button extension overconstrained legacy example-movement coverage|mechanical test fix
B115|2026-09-02|settings-opening update retained game draw-side trails although source dispatched drawsettings|V133
B116|2026-09-02|target-2 transition was treated as settings for positive cursor values although source retains `from` draw state|V142
B117|2026-09-02|target-2 game-side transition replay matched the exposed draw state but omitted drawtransition's second updategame call|V143
B118|2026-09-02|native menu start advanced the lifecycle but did not latch the cartridge hasplayed flag|V144
B119|2026-09-02|settings-to-game replay kept transition_to_game for one frame after the source draw cursor had reached its completion boundary|V145
B120|2026-09-02|generic transition-boundary trail compensation added a second settings-to-game trail that the source does not retain|V146
B121|2026-09-02|terminal rendering forced the physical bottom row to zero and broke a source trace whose quantized camera offset retained background there|V147
B122|2026-09-02|native die omitted the source's second shake increment, changing terminal camera quantization and the retained edge|V148
B123|2026-09-02|new difficulty tables used direct indexing under workspace no-panic lint|V151
B124|2026-09-02|native snapshot wire version changed without updating the Python fixture constructor|V152
B125|2026-09-02|native P2 collision removed initiating enemy before source mutable-list iteration|V153
B126|2026-09-02|native enemy update reused one size/position view across source pre-growth and post-growth checks|V154
B127|2026-09-02|pattern metadata wire and constructors assumed autovar could stand in for special values|V155
B128|2026-09-02|native round used signed remainder and rounded negative subpixels toward floor|V156
B129|2026-09-02|native pattern scheduler only opened rectangles and never executed source targets or completion reset|V157
B130|2026-09-02|pattern inventory regression test used one-based pattern IDs as zero-based vector indexes|V157
B131|2026-09-02|native particle trig evaluated `f32` sin/cos while Pemsa uses `f32` input plus C double math before `f32` fixed conversion|V159
