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
C13: prebuilt Pemsa remains 60 Hz; ⊥ promise faster-than-real-time simulation.
C14: full visible world-state JSON deferred to T10.

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
result: `{"score":number,"frames":int,"survival_frames":int,"seed":int,"started":bool}`.

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

§B

id|date|cause|fix
B1|2026-08-17|Pemsa window discovered before SDL X11 init settled|V9
B2|2026-08-17|Pemsa glyph diagnostics contain invalid UTF-8 bytes|V21
B3|2026-08-19|`main` renamed `control`; control tests/import callers broke|V24
B4|2026-08-19|no-op headless `_draw` froze cartridge draw-driven transitions|V25
