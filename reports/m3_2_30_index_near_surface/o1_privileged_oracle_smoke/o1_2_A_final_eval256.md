# O1.1 Deterministic Stability and Gait Report

**Classification: `REACHABILITY_ONLY`**

- Checkpoint: `/home/linkerhand/IsaacLab/logs/skrl/dex5_penspin_privileged_oracle/2026-07-30_21-34-06_ppo_torch_o1_2_A/checkpoints/agent_720.pt`
- Environments / steps: `256 / 240`
- Ballistic termination guard: `True`
- Completed episodes: `6149`
- Raw gait / qualified gait: `0.00% / 0.00%`
- Raw 180 deg / qualified 180 deg: `0.00% / 0.00%`
- Failure / contact loss / ballistic: `100.00% / 0.00% / 85.54%`
- Mean ballistic exposure: `0.86` control steps per completed episode

Qualified means: directional axis alignment >= 0.70, no ballistic, no contact-loss termination, and no failure.

## By Canonical

| Canonical | Episodes | Raw gait | Qualified gait | Raw 180 | Qualified 180 | Axis align | Failure | Contact loss | Ballistic |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| P1 | 3074 | 0.00% | 0.00% | 0.00% | 0.00% | 0.566 | 100.00% | 0.00% | 100.00% |
| P2 | 1241 | 0.00% | 0.00% | 0.00% | 0.00% | 0.255 | 100.00% | 0.00% | 29.90% |
| P3 | 1834 | 0.00% | 0.00% | 0.00% | 0.00% | 0.580 | 100.00% | 0.00% | 98.96% |
| P4 | 0 | 0.00% | 0.00% | 0.00% | 0.00% | 0.000 | 0.00% | 0.00% | 0.00% |
| P5 | 0 | 0.00% | 0.00% | 0.00% | 0.00% | 0.000 | 0.00% | 0.00% | 0.00% |

## Primary Failure Type

| Canonical | contact_loss | ballistic | penetration | throw | drop | drift | invalid |
|---|---:|---:|---:|---:|---:|---:|---:|
| P1 | 0 | 3074 | 0 | 0 | 0 | 0 | 0 |
| P2 | 0 | 357 | 315 | 0 | 569 | 0 | 0 |
| P3 | 0 | 1781 | 0 | 53 | 0 | 0 | 0 |
| P4 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| P5 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |

## Debounced Finger Events

Each break and re-contact must remain stable for at least 3 control steps. Re-contact is counted as gait only after a prior break in the same episode.

| Canonical | Finger | Break episodes | Re-contact episodes | Gait episodes | Break events | Re-contact events |
|---|---|---:|---:|---:|---:|---:|
| P1 | thumb | 1337 | 0 | 0 | 1337 | 0 |
| P1 | index | 2704 | 0 | 0 | 2704 | 0 |
| P1 | middle | 426 | 0 | 0 | 426 | 0 |
| P1 | ring | 0 | 0 | 0 | 0 | 0 |
| P1 | pinky | 0 | 0 | 0 | 0 | 0 |
| P2 | thumb | 950 | 0 | 0 | 950 | 0 |
| P2 | index | 1241 | 0 | 0 | 1241 | 0 |
| P2 | middle | 1029 | 0 | 0 | 1029 | 0 |
| P2 | ring | 0 | 0 | 0 | 0 | 804 |
| P2 | pinky | 0 | 0 | 0 | 0 | 0 |
| P3 | thumb | 1815 | 0 | 0 | 1815 | 0 |
| P3 | index | 40 | 0 | 0 | 40 | 0 |
| P3 | middle | 5 | 0 | 0 | 5 | 0 |
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
