# O1.1 Deterministic Stability and Gait Report

**Classification: `REACHABILITY_ONLY`**

- Checkpoint: `/home/linkerhand/IsaacLab/logs/skrl/dex5_penspin_privileged_oracle/2026-07-30_21-34-43_ppo_torch_o1_2_B/checkpoints/best_agent.pt`
- Environments / steps: `256 / 240`
- Ballistic termination guard: `True`
- Completed episodes: `6428`
- Raw gait / qualified gait: `0.09% / 0.00%`
- Raw 180 deg / qualified 180 deg: `0.00% / 0.00%`
- Failure / contact loss / ballistic: `100.00% / 0.00% / 91.10%`
- Mean ballistic exposure: `0.91` control steps per completed episode

Qualified means: directional axis alignment >= 0.70, no ballistic, no contact-loss termination, and no failure.

## By Canonical

| Canonical | Episodes | Raw gait | Qualified gait | Raw 180 | Qualified 180 | Axis align | Failure | Contact loss | Ballistic |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| P1 | 1960 | 0.00% | 0.00% | 0.00% | 0.00% | 0.520 | 100.00% | 0.00% | 100.00% |
| P2 | 1228 | 0.49% | 0.00% | 0.00% | 0.00% | 0.319 | 100.00% | 0.00% | 57.65% |
| P3 | 2546 | 0.00% | 0.00% | 0.00% | 0.00% | 0.627 | 100.00% | 0.00% | 98.00% |
| P4 | 694 | 0.00% | 0.00% | 0.00% | 0.00% | 0.201 | 100.00% | 0.00% | 99.86% |
| P5 | 0 | 0.00% | 0.00% | 0.00% | 0.00% | 0.000 | 0.00% | 0.00% | 0.00% |

## Primary Failure Type

| Canonical | contact_loss | ballistic | penetration | throw | drop | drift | invalid |
|---|---:|---:|---:|---:|---:|---:|---:|
| P1 | 0 | 1960 | 0 | 0 | 0 | 0 | 0 |
| P2 | 0 | 672 | 105 | 0 | 451 | 0 | 0 |
| P3 | 0 | 2445 | 1 | 100 | 0 | 0 | 0 |
| P4 | 0 | 693 | 1 | 0 | 0 | 0 | 0 |
| P5 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |

## Debounced Finger Events

Each break and re-contact must remain stable for at least 3 control steps. Re-contact is counted as gait only after a prior break in the same episode.

| Canonical | Finger | Break episodes | Re-contact episodes | Gait episodes | Break events | Re-contact events |
|---|---|---:|---:|---:|---:|---:|
| P1 | thumb | 728 | 0 | 0 | 728 | 0 |
| P1 | index | 1913 | 0 | 0 | 1913 | 0 |
| P1 | middle | 498 | 0 | 0 | 498 | 0 |
| P1 | ring | 0 | 0 | 0 | 0 | 0 |
| P1 | pinky | 0 | 0 | 0 | 0 | 0 |
| P2 | thumb | 526 | 0 | 0 | 526 | 0 |
| P2 | index | 1228 | 6 | 6 | 1234 | 6 |
| P2 | middle | 835 | 0 | 0 | 835 | 0 |
| P2 | ring | 0 | 0 | 0 | 0 | 474 |
| P2 | pinky | 0 | 0 | 0 | 0 | 0 |
| P3 | thumb | 2495 | 0 | 0 | 2495 | 0 |
| P3 | index | 87 | 0 | 0 | 87 | 0 |
| P3 | middle | 8 | 0 | 0 | 8 | 0 |
| P3 | ring | 0 | 0 | 0 | 0 | 0 |
| P3 | pinky | 0 | 0 | 0 | 0 | 0 |
| P4 | thumb | 4 | 0 | 0 | 4 | 0 |
| P4 | index | 7 | 0 | 0 | 7 | 0 |
| P4 | middle | 5 | 0 | 0 | 5 | 0 |
| P4 | ring | 0 | 0 | 0 | 0 | 0 |
| P4 | pinky | 0 | 0 | 0 | 0 | 0 |
| P5 | thumb | 0 | 0 | 0 | 0 | 0 |
| P5 | index | 0 | 0 | 0 | 0 | 0 |
| P5 | middle | 0 | 0 | 0 | 0 | 0 |
| P5 | ring | 0 | 0 | 0 | 0 | 0 |
| P5 | pinky | 0 | 0 | 0 | 0 | 0 |

This is deterministic evaluation of the existing checkpoint. No PPO training, teacher generation, or O0 mutation is part of this report.
