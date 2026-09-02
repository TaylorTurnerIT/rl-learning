# Native Dodge workspace

This workspace contains the engine-free Rust simulation and, in later phases,
its Macroquad viewer and training adapters.

`crates/dodge-core` remains independent of Macroquad, Python, Pemsa, display
servers, subprocesses, and JSON in its frame loop. Offline runners and viewer
crates consume its typed snapshots at the workspace boundary.
