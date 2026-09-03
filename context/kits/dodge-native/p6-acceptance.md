# P6 acceptance: batched native training

Status: `accepted_automated_owner_review_pending`

P6 adds an opt-in Rust/PyO3 batch environment and a native PPO backend. The
native path keeps the accepted `dodge-core` simulation and exposes owned NumPy
board, pixel, hash, metadata, and canonical full-state buffers. It does not
initialize Macroquad or use Pemsa, Xvfb, xdotool, subprocesses, or per-step
JSON. The existing Python/Pemsa path remains the default fallback.

## Measured workload

The fixed workload used 32 lanes, 1,024 lane-decisions, four-frame decisions,
the same nine-action schedule, and five repetitions on Linux x86_64. Median
throughput was:

| path | observation payload | median | lane-steps/s | speedup |
|---|---|---:|---:|---:|
| native | full state + 128 × 128 pixels | 0.6288 s | 1,628.5 | 315.9× |
| native control | pixels off, 19 × 16 × 16 board | 0.5088 s | 2,012.5 | 390.3× |
| Python/Pemsa | legacy raw state | 198.6047 s | 5.16 | 1× |

Raw per-repetition timings and workload details are retained in
`p6-benchmark-raw.json`; the machine-readable report is
`p6-acceptance-report.json`.

## Behavior and boundary evidence

- Rust serial and Rayon-parallel lanes match state, pixels, reward, events, and
  hashes; selective lane reset preserves unselected lane trajectories.
- The Python binding validates nine action indexes, returns owned arrays, and
  exposes selective resets with lane IDs. Missing native installation is an
  explicit error.
- Native PPO collects batched board rollouts, preserves survival-frame reward,
  capped neutral shaping, GAE, checkpoints, and held-out evaluation. Backend,
  lane count, execution mode, and observation mode are in `PPOConfig` and the
  checkpoint/run record.
- The short live comparison passed through the verified X11 runtime wrapper.
  The existing step bridge reports its first ready counter at 26 while the
  accepted native core boundary is 13; the test records that +13 counter offset
  and compares relative state/action/reward progression.

Owner visual/full-state review remains pending. Automated pixel equality and
lossless Macroquad capture are evidence, not a substitute for that review.
