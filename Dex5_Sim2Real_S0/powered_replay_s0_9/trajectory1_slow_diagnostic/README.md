# Trajectory 1 Slow Diagnostic

This directory contains a separately versioned hardware diagnostic derived
from frozen S0.9 source trajectory 1. It does not replace or modify
`../mapped_replay_100hz.npz`.

## Fixed profile

- command rate: `100 Hz`
- command speed cap: `0.03 rad/s`
- replay control: reduced-gain zero-torque hybrid
- feedback abort: one fresh 50 ms window above `0.12 rad/s`
- thermal abort: `75 C`
- source and diagnostic SHA-256 values are recorded in
  `trajectory1_slow_manifest.json`

The operator must first remove the pen and rerun mapped-P3 staging plus the
startup-only gate. Powered replay then requires a separate confirmation after
the pen is placed. Completion remains diagnostic and must be annotated with
drop and visible-rotation observations.

## First powered result

Loaded startup passed. Replay then executed 237/1007 rows (about 2.36 s) and
software-E-stopped when independent joint 4 produced a `0.4897 rad/s` 50 ms
feedback jump. Samples recorded before that validating frame peaked at
`0.0245 rad/s`. The result is stored in `powered_retry_result.json`; further
powered retry is frozen pending operator drop/jam annotation and transient
review.

The second attempt used a fresh empty-hand P3 gate and again passed loaded
startup. It stopped at 274/1007 rows (about 2.73 s) when joint 4 (`Roll_31R`,
middle-finger roll) reached `0.2853 rad/s`. Repetition on the same channel
identifies a trajectory/controller/load incompatibility. See
`powered_retry2_result.json`; do not run a third powered replay without a new
offline correction and review.

With the approximately 9.8 mm pen, a dedicated loaded hold passed and the
separately hashed 0-2.0 s prefix completed all 201 rows. The operator reported
no drop and visible rotation. The safety-relevant 50 ms position-difference
peak was `0.00143 rad/s`; raw SDK `dq=2.4544 rad/s` remains diagnostic-only.
See `thin_pen_9p8mm_prefix_powered_result.json`. This accepts only the prefix,
not the full 10.07 s replay.
