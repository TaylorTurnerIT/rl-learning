# P1 oracle corpus

The JSON files are the tracked action schedules. Full canonical traces are
generated into the ignored `src/dodge/runtime/.native-oracle-check/` directory
because one 128 × 128 indexed-pixel trace is large. The acceptance report
records the source/Pemsa identities, trace hashes, frame counts, and terminal
results for every generated fixture.

Each schedule begins with the explicit PICO-8 `x` start action and then covers
neutral, cardinal, and diagonal controls. The fixed seed is part of the fixture
identity, not an ambient emulator setting.
