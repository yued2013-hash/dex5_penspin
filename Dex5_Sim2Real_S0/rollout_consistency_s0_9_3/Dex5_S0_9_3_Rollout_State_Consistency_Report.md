# Dex5 S0.9.3 Rollout State Consistency Repair

## Status

```text
Synchronized transition recording: PASS
Quaternion-based success gate: PASS_IMPLEMENTED
Existing checkpoint corrected success: 0/32
Instantaneous velocity integrability: FAIL
New successful oracle dataset: BLOCKED
Student / powered replay: BLOCKED
```

## Changes

- Each row now records `state_before`, action, `state_after`, and
  `transition_same_episode`.
- Automatic-reset boundaries are excluded from kinematic checks.
- 120 Hz PhysX substep quaternion and angular-velocity samples are recorded.
- `pen_angular_velocity_pose_delta` is derived from adjacent `root_quat_w`
  samples and is the finite-step motion channel.
- `root_ang_vel_w` remains available only as an instantaneous contact/impact
  diagnostic.
- S0 success and hold now use actual relative-quaternion target-axis rotation,
  target-axis alignment >=0.70, and the existing 150 ms physical hold gates.
- The old integrated angular-velocity quantity remains diagnostic and is not a
  success condition.

## Smoke Result

The smoke used 16 environments and collected 32 completed episodes with the
frozen `agent_4032.pt` checkpoint. It contains 3008 valid non-reset control
transitions and 6016 valid 120 Hz substeps.

| Metric | 60 Hz | 120 Hz |
|---|---:|---:|
| omega-z correlation | 0.116 | 0.554 |
| XYZ RMSE (rad/s) | 2.134 | 2.127 |
| quaternion/reported path ratio | 0.130 | 0.222 |

The improvement at 120 Hz confirms control-rate aliasing, but the instantaneous
velocity channel still does not approximate finite-step pose motion closely
enough. It must not be integrated as object-rotation evidence.

## Gate

No corrected success was observed. The synchronized smoke dataset remains a
quarantined diagnostic artifact. Recollecting a student or powered-replay
dataset requires a policy that reaches the corrected quaternion success gate.
