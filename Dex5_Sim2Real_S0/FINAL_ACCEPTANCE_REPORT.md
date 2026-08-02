# Dex5 Sim2Real-S0 Offline Acceptance

Status: **PASS_OFFLINE / NOT_READY_FOR_POWERED_HARDWARE**

## Scope

The hardware-compatible P3 short-rotation oracle pipeline is complete through
offline real-replay preparation. Student BC and powered Dex5 execution remain
intentionally out of scope.

## Reset Distribution

- Validated S0 P3 states: 582
- Unique original P3 parents: 329
- Unique source trajectories: 209
- Final TIM contact: 100%
- Maximum reset penetration: 0.254 mm
- PIP/DIP target coupling error: 0 rad
- O0 source SHA-256 remains:
  `73ab3cc0ffada7759da428767956b353b39b556d77c60accbd95c4891073ac65`

## Oracle Gate

Task: P3 to 45 deg +/- 10 deg, active braking, strict 150 ms hold, no
handoff, no ring/pinky transition. The hardware-compatible horizon is 1.6 s;
the earlier 0.8/1.2 s audits showed that provisional actuator velocity limits,
not physical reachability, truncated valid trajectories.

| Seed | Nominal | Medium DR |
|---:|---:|---:|
| 20260801 | 95.51% | 84.89% |
| 20260802 | 94.92% | 88.00% |
| 20260803 | 96.01% | 93.38% |

All three seeds pass nominal >=90%, medium DR >=70%, and minimum seed >=60%.
All formal evaluations report zero non-finite observations, zero target
coupling error, and zero physical-failure episodes.

## Rollout Dataset

- Successful trajectories: 1200
- Transition rows: 83,218
- Unique source trajectories: 160
- Dataset split: 872 train / 136 validation / 192 test
- Source-trajectory split leakage: 0
- Executed virtual action bound: [-1, 1]
- Dataset SHA-256:
  `1451594f8d4e8519459eb93538945209e5d8fbde80641fd26aa8fb1fb3e54b71`

The dataset stores oracle and deployable student observations, raw and
executed virtual actions, projected 20D targets, 16D logical actuator
coordinates, SDK absolute targets, robot state, contact state/force, pen state,
adapter randomization, and source provenance.

## Replay Package

- Selected trajectories: 20 from unique training sources
- Commands: 3,486 at 100 Hz, replayed at 0.5x trajectory speed
- Maximum observed 100 Hz command delta: 0.00409 rad
- Coupling error: 0 rad
- Offline URDF limit check: PASS
- Privileged oracle included: no
- Dry run: PASS

The 16D values are logical coupled coordinates, not motor-shaft commands. The
public Dex5 SDK must remain responsible for its nonlinear URDF-to-motor
conversion.

## Hardware Boundary

Powered replay is blocked until the Dex5 has power and controlled acceptance
measures joint directions, physical PIP/DIP coupling, latency, PD response,
workspace limits, and emergency-stop behavior. The first powered experiment
must be human-supervised open-loop replay with reduced range and speed. Direct
deployment of the privileged oracle and online real-world PPO remain forbidden.
