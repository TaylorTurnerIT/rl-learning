# P2 acceptance: extraction and compatibility boundary

Status: `accepted_primitive_boundary`

Source: `dodge.p8` (`7453a9658fd32577385ad72672a54ad84ff70567fadbde75ba6634aa5cc684a3`)

The checked-in cartridge was extracted into a deterministic, hash-addressed
asset bundle. The bundle preserves indexed 128×128 graphics, the default PICO-8
palette, sprite metadata, all 64 SFX records, all 32 music records, source
spans, and the current static-table inventory.

The compatibility report accepted 56 records covering Q16.16 numeric behavior,
the Pemsa `rand()` stream, list operations, input/stat/persistent-data behavior,
and indexed camera/palette/fill/sprite raster behavior. The raster fixture is
`p2-raster-fixture.json`.

Accepted output: `p2-acceptance-report.json`

Deferred to later phases:

- full gameplay-function and frame parity in P3/P4;
- audio waveform parity in P4.

Handoff: `p3_primitive_boundary_ready`
