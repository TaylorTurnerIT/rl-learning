# Dodge JSON control layer

§G

JSON movement sequence → keyboard events → controlled Pemsa Dodge run; no emulator build.

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

§I

cli: `just dodge-control <commands.json|->` → launch game, start run, execute sequence, exit `0`.
cli: `just dodge-control <commands.json|-> --seed <0..32767>` → seed PICO-8 RNG before game init.
default: omitted `--seed` → seed `42`.
cli: source `-` → read JSON from stdin.
cli: invalid JSON/schema/window timeout/injection failure → stderr diagnostic + nonzero exit.
json: top-level array of `{"move":"<direction>","duration_ms":<int>}`.
enum: `<direction>` → `neutral|left|right|up|down|up_left|up_right|down_left|down_right`.
keys: `left|right|up|down` → keyboard `Left|Right|Up|Down` → PICO-8 `btn(0..3)`.
start: ready window → keyboard `x` tap → PICO-8 `❎`; wait transition before movement.

§V

V1: ∀ JSON input valid list + exact fields/types/enums/ranges before side effects.
V2: direction → exact key set; diagonal keys held simultaneously; `neutral` → no held keys.
V3: ∀ command execute in list order for requested `duration_ms`; previous keys released before next command.
V4: ∀ injected key event targets launched Pemsa window id.
V5: normal completion | exception | signal → zero held keys.
V6: controller owns emulator lifecycle; completion/error/interrupt → emulator terminated + reaped.
V7: game startup bounded by window timeout; timeout → clear error + no orphan process.
V8: existing `just dodge-run` interactive path remains functional.
V9: first key injection waits bounded SDL/X11 settle interval after window discovery.
V10: `--seed N` → `srand(N)` first `_init` statement in launched temporary cartridge.
V11: seed absent → seed `42`; explicit valid seed overrides; invalid seed → fail before side effects.

§T

id|status|task|cites
T1|x|add JSON model, full-list validation, direction mapping|V1,V2,I.json,I.enum,I.keys
T2|x|add targeted X11 keyboard backend + Pemsa lifecycle|V3,V4,V5,V6,V7,I.start
T3|x|add CLI, devenv dependency/scripts, just recipe, docs|V8,I.cli,C4,C5
T4|x|add unit tests + controlled-run smoke test|V1,V2,V3,V4,V5,V6,V7,V8,V9
T5|x|add deterministic seed option via temporary cartridge|V10,V11,I.cli,C9
T6|x|default control seed to `42`|V10,V11,I.cli

§B

id|date|cause|fix
B1|2026-08-17|Pemsa window discovered before SDL X11 init settled|V9
