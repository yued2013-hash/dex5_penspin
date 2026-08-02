# O1.1 Deterministic Stability and Gait Report

**Classification: `REACHABILITY_ONLY`**

- Checkpoint: `/home/linkerhand/IsaacLab/logs/skrl/dex5_penspin_privileged_oracle/2026-07-30_22-08-37_ppo_torch_o1_3_P3State60/checkpoints/best_agent.pt`
- Environments / steps: `256 / 240`
- Ballistic termination guard: `False`
- Completed episodes: `1298`
- Raw gait / qualified gait: `89.14% / 0.00%`
- Raw 180 deg / qualified 180 deg: `83.28% / 0.00%`
- Failure / contact loss / ballistic: `100.00% / 97.46% / 19.57%`
- Mean ballistic exposure: `0.21` control steps per completed episode

Qualified means: directional axis alignment >= 0.70, no ballistic, no contact-loss termination, and no failure.

## By Canonical

| Canonical | Episodes | Raw gait | Qualified gait | Raw 180 | Qualified 180 | Axis align | Failure | Contact loss | Ballistic |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| P1 | 0 | 0.00% | 0.00% | 0.00% | 0.00% | 0.000 | 0.00% | 0.00% | 0.00% |
| P2 | 0 | 0.00% | 0.00% | 0.00% | 0.00% | 0.000 | 0.00% | 0.00% | 0.00% |
| P3 | 1298 | 89.14% | 0.00% | 83.28% | 0.00% | 0.575 | 100.00% | 97.46% | 19.57% |
| P4 | 0 | 0.00% | 0.00% | 0.00% | 0.00% | 0.000 | 0.00% | 0.00% | 0.00% |
| P5 | 0 | 0.00% | 0.00% | 0.00% | 0.00% | 0.000 | 0.00% | 0.00% | 0.00% |

## Thumb And Height Recovery

A thumb attempt is any post-release raw thumb contact that has not yet passed the three-step re-contact debounce. Height recovery requires a prior >=5 mm drop followed by return to within 2 mm of reset height.

| Canonical | Thumb release | Attempt | Re-contact | Mean break step | Mean attempt step | Height drop | Height recovery | Min height delta |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| P1 | 0.00% | 0.00% | 0.00% | 0.0 | 0.0 | 0.00% | 0.00% | 0.00 mm |
| P2 | 0.00% | 0.00% | 0.00% | 0.0 | 0.0 | 0.00% | 0.00% | 0.00 mm |
| P3 | 98.07% | 97.38% | 58.01% | 3.0 | 10.6 | 98.07% | 0.00% | -17.35 mm |
| P4 | 0.00% | 0.00% | 0.00% | 0.0 | 0.0 | 0.00% | 0.00% | 0.00 mm |
| P5 | 0.00% | 0.00% | 0.00% | 0.0 | 0.0 | 0.00% | 0.00% | 0.00 mm |

## Primary Failure Type

| Canonical | contact_loss | ballistic | penetration | throw | drop | drift | invalid |
|---|---:|---:|---:|---:|---:|---:|---:|
| P1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| P2 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| P3 | 1011 | 254 | 0 | 26 | 7 | 0 | 0 |
| P4 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| P5 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |

## Debounced Finger Events

Each break and re-contact must remain stable for at least 3 control steps. Re-contact is counted as gait only after a prior break in the same episode.

| Canonical | Finger | Break episodes | Re-contact episodes | Gait episodes | Break events | Re-contact events |
|---|---|---:|---:|---:|---:|---:|
| P1 | thumb | 0 | 0 | 0 | 0 | 0 |
| P1 | index | 0 | 0 | 0 | 0 | 0 |
| P1 | middle | 0 | 0 | 0 | 0 | 0 |
| P1 | ring | 0 | 0 | 0 | 0 | 0 |
| P1 | pinky | 0 | 0 | 0 | 0 | 0 |
| P2 | thumb | 0 | 0 | 0 | 0 | 0 |
| P2 | index | 0 | 0 | 0 | 0 | 0 |
| P2 | middle | 0 | 0 | 0 | 0 | 0 |
| P2 | ring | 0 | 0 | 0 | 0 | 0 |
| P2 | pinky | 0 | 0 | 0 | 0 | 0 |
| P3 | thumb | 1273 | 753 | 753 | 2030 | 764 |
| P3 | index | 1273 | 1095 | 1095 | 2367 | 1095 |
| P3 | middle | 1273 | 259 | 259 | 1565 | 292 |
| P3 | ring | 0 | 0 | 0 | 0 | 0 |
| P3 | pinky | 0 | 0 | 0 | 0 | 0 |
| P4 | thumb | 0 | 0 | 0 | 0 | 0 |
| P4 | index | 0 | 0 | 0 | 0 | 0 |
| P4 | middle | 0 | 0 | 0 | 0 | 0 |
| P4 | ring | 0 | 0 | 0 | 0 | 0 |
| P4 | pinky | 0 | 0 | 0 | 0 | 0 |
| P5 | thumb | 0 | 0 | 0 | 0 | 0 |
| P5 | index | 0 | 0 | 0 | 0 | 0 |
| P5 | middle | 0 | 0 | 0 | 0 | 0 |
| P5 | ring | 0 | 0 | 0 | 0 | 0 |
| P5 | pinky | 0 | 0 | 0 | 0 | 0 |

This is deterministic evaluation of the existing checkpoint. No PPO training, teacher generation, or O0 mutation is part of this report.
