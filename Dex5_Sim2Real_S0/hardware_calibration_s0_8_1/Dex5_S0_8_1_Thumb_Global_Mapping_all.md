# Dex5 S0.8.1 Thumb Global Mapping

## Status

```text
hardware: SCAN_COMPLETED
analysis: PASS_SEGMENTED_MAPPING_ONLY
P3 feedback covered: True
PPO/student/teacher/replay: NOT RUN
```

| command joint | dominant feedback joint | sign | offset rad | rank correlation | RMSE rad | hysteresis rad |
|---:|---:|---:|---:|---:|---:|---:|
| 16 | 16 | +1 | +0.0557 | 0.9984 | 0.1004 | 0.2878 |
| 17 | 17 | +1 | +0.0316 | 0.9998 | 0.0819 | 0.2341 |
| 18 | 18 | +1 | +0.0000 | 0.9998 | 0.1160 | 0.3144 |
| 19 | 19 | +1 | +0.0978 | 0.9972 | 0.2521 | 0.8016 |

The scan records raw and unwrapped thumb feedback for every command point.
Estimated inverse P3 commands are not transmitted automatically.
