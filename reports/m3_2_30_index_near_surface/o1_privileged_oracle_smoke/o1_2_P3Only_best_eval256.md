# O1.1 Deterministic Stability and Gait Report

**Classification: `REACHABILITY_ONLY`**

- Checkpoint: `/home/linkerhand/IsaacLab/logs/skrl/dex5_penspin_privileged_oracle/2026-07-30_21-35-22_ppo_torch_o1_2_P3Only/checkpoints/best_agent.pt`
- Environments / steps: `256 / 240`
- Ballistic termination guard: `True`
- Completed episodes: `2170`
- Raw gait / qualified gait: `6.41% / 0.00%`
- Raw 180 deg / qualified 180 deg: `0.00% / 0.00%`
- Failure / contact loss / ballistic: `99.95% / 12.30% / 87.51%`
- Mean ballistic exposure: `0.88` control steps per completed episode

Qualified means: directional axis alignment >= 0.70, no ballistic, no contact-loss termination, and no failure.

## By Canonical

| Canonical | Episodes | Raw gait | Qualified gait | Raw 180 | Qualified 180 | Axis align | Failure | Contact loss | Ballistic |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| P1 | 0 | 0.00% | 0.00% | 0.00% | 0.00% | 0.000 | 0.00% | 0.00% | 0.00% |
| P2 | 0 | 0.00% | 0.00% | 0.00% | 0.00% | 0.000 | 0.00% | 0.00% | 0.00% |
| P3 | 2170 | 6.41% | 0.00% | 0.00% | 0.00% | 0.529 | 99.95% | 12.30% | 87.51% |
| P4 | 0 | 0.00% | 0.00% | 0.00% | 0.00% | 0.000 | 0.00% | 0.00% | 0.00% |
| P5 | 0 | 0.00% | 0.00% | 0.00% | 0.00% | 0.000 | 0.00% | 0.00% | 0.00% |

## Primary Failure Type

| Canonical | contact_loss | ballistic | penetration | throw | drop | drift | invalid |
|---|---:|---:|---:|---:|---:|---:|---:|
| P1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| P2 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| P3 | 238 | 1876 | 0 | 55 | 0 | 0 | 0 |
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
| P3 | thumb | 2141 | 139 | 139 | 2265 | 139 |
| P3 | index | 456 | 0 | 0 | 456 | 0 |
| P3 | middle | 498 | 0 | 0 | 498 | 0 |
| P3 | ring | 0 | 0 | 0 | 0 | 1 |
| P3 | pinky | 0 | 0 | 0 | 0 | 1 |
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
