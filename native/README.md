# Native Dodge workspace

This workspace contains the engine-free Rust simulation and, in later phases,
its Macroquad viewer and training adapters.

`crates/dodge-core` remains independent of Macroquad, Python, Pemsa, display
servers, subprocesses, and JSON in its frame loop. Offline runners and viewer
crates consume its typed snapshots at the workspace boundary.

Run a tracked P1 command schedule through the native core with no emulator or
window process:

```text
cargo run -p dodge-native-runner -- \
  --commands ../context/kits/dodge-native/corpus/seed-42-movement.json \
  --seed 42 --output native-trace.json
```

The runner emits one record per frame, including the canonical snapshot in
hexadecimal, typed-state hash, indexed-pixel hash, reward, and terminal flag.

Compare that stream with a canonical full-draw trace and receive the first
field or pixel mismatch with its cartridge source span:

```text
dodge-native-diff \
  --native native-trace.json \
  --oracle ../src/dodge/runtime/.native-p3-oracle/seed-42-movement.json \
  --no-pixels
```
