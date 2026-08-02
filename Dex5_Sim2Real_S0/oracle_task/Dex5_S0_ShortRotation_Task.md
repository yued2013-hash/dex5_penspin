# Dex5 S0 Short Rotation Oracle Task

## Contract

```text
Validated S0 hardware-compatible P3 reset
  -> 45 deg +/- 10 deg target-axis rotation
  -> active braking
  -> 150 ms hold
```

- Accepted task IDs: `Dex5-S0-ShortRotation-16-Nominal-Direct-v0` and
  `Dex5-S0-ShortRotation-16-MediumDR-Direct-v0`.
- Episode horizon: 1.6 s. The 0.8/1.2 s audits were retained as diagnostics;
  provisional hardware velocity limits truncated otherwise valid holds.
- Reset source: only `Dex5_S0_HW_P3_Reset_Bank.npz`.
- Action: 20D virtual delta target through the S0 coupling and dual-clock
  adapter. Bypassing the adapter is not supported.
- Oracle observation: 152D privileged state.
- Student observation: separately recorded 340D real-available history.
- Handoff, ring transition, and finger gait are outside this task.
- A contact violation must persist for 3 control steps before it is recorded.
  Earlier transient contact changes remain diagnostic; a later recovered
  window may qualify only after all strict gates hold continuously for 150 ms.

## Success

- relative `root_quat_w` target-axis rotation lies in the configured target
  window (30-40 degrees for the S0.9.4 35-degree audit task);
- relative-quaternion rotation-axis alignment is at least 0.70;
- target-axis speed <=2.5 rad/s;
- wrong-axis speed <=2.0 rad/s;
- height loss <=5 mm;
- TIM support remains and ring/pinky remain clear;
- conditions hold for 9 control steps (150 ms);
- no drop, invalid state, or excessive penetration.

`root_ang_vel_w` is an instantaneous impact/contact diagnostic. Its control-rate
integral is not object-rotation evidence and must not be used as a success gate.
Finite-step motion is derived from adjacent quaternion samples.

The reward contains rotation, axis alignment, braking/terminal hold, bounded
strict-hold counter progress, velocity, joint-motion, drop, and penetration
terms. It contains no contact-pattern or finger-order reward.
