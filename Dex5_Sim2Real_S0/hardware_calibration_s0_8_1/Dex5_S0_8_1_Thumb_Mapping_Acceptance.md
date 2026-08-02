# Dex5 S0.8.1 Thumb Mapping Acceptance

## Result

```text
global segmented mapping: PASS
mapped P3 static gate: PASS
trajectory replay: NOT RUN
PPO/student/teacher: NOT RUN
```

All four thumb command channels were scanned independently over their audited
URDF ranges. Raw feedback, unwrapped feedback, forward/reverse branches,
direction, offset, scale, hysteresis, and piecewise inverse data are retained.

| Command | Feedback | Sign | Feedback-to-URDF scale | P3 SDK command |
|---:|---:|---:|---:|---:|
| 16 | 16 | +1 | 1.6414 | 0.1980 |
| 17 | 17 | +1 | 1.1430 | 0.7971 |
| 18 | 18 | +1 | 1.1448 | 0.9899 |
| 19 | 19 | +1 | 1.4075 | 0.6663 |

The joints exhibit direction-dependent hysteresis, so a single affine mapping
is rejected. Runtime conversion must select the applicable motion branch.

## Static P3 Gate

The corrected command was evaluated from a fresh powered open-hand baseline.
The hand was empty and motors were disabled after measurement.

```text
desired normalized thumb:
[0.489146, 0.655973, 0.917135, 0.450349]

measured normalized thumb:
[0.390648, 0.669668, 0.932860, 0.454395]

error rad:
[-0.098498, +0.013696, +0.015724, +0.004046]

maximum absolute error: 0.098498 rad
gate: <= 0.12 rad
```

## Frozen Evidence

- `thumb_global_mapping_all.json`
  - SHA-256: `273bafb3631a50907a7738d57212e1b83df1184c91953fac9056a1f2a43c415b`
- `mapped_p3_static_gate_joint16_corrected.json`
  - SHA-256: `46882c610578a3c40918af2a65404d026031f960af59ead0b9f3728f74adf95b`

The mapped P3 correction is local evidence for the simultaneous four-thumb
entry. It does not replace the global branch maps and does not validate a
trajectory, pen grasp, policy, or powered replay.
