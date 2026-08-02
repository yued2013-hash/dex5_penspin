# O1.1 Deterministic Stability and Gait Report

**Classification: `REACHABILITY_ONLY`**

- Checkpoint: `/home/linkerhand/IsaacLab/logs/skrl/dex5_penspin_privileged_oracle/2026-07-30_17-42-49_ppo_torch/checkpoints/best_agent.pt`
- Environments / steps: `256 / 240`
- Ballistic termination guard: `False`
- Completed episodes: `559`
- Raw gait / qualified gait: `29.16% / 0.00%`
- Raw 180 deg / qualified 180 deg: `72.99% / 0.00%`
- Failure / contact loss / ballistic: `60.64% / 36.49% / 46.51%`
- Mean ballistic exposure: `1.24` control steps per completed episode

Qualified means: directional axis alignment >= 0.70, no ballistic, no contact-loss termination, and no failure.

## By Canonical

| Canonical | Episodes | Raw gait | Qualified gait | Raw 180 | Qualified 180 | Axis align | Failure | Contact loss | Ballistic |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| P1 | 202 | 37.13% | 0.00% | 100.00% | 0.00% | 0.367 | 50.50% | 48.02% | 50.99% |
| P2 | 122 | 61.48% | 0.00% | 95.08% | 0.00% | 0.298 | 99.18% | 69.67% | 100.00% |
| P3 | 100 | 0.00% | 0.00% | 90.00% | 0.00% | 0.437 | 8.00% | 0.00% | 0.00% |
| P4 | 107 | 11.21% | 0.00% | 0.00% | 0.00% | 0.176 | 100.00% | 19.63% | 31.78% |
| P5 | 28 | 3.57% | 0.00% | 0.00% | 0.00% | -0.070 | 3.57% | 3.57% | 3.57% |

## Primary Failure Type

| Canonical | contact_loss | ballistic | penetration | throw | drop | drift | invalid |
|---|---:|---:|---:|---:|---:|---:|---:|
| P1 | 0 | 100 | 0 | 2 | 0 | 0 | 0 |
| P2 | 0 | 113 | 0 | 8 | 0 | 0 | 0 |
| P3 | 0 | 0 | 6 | 2 | 0 | 0 | 0 |
| P4 | 0 | 34 | 27 | 0 | 46 | 0 | 0 |
| P5 | 0 | 1 | 0 | 0 | 0 | 0 | 0 |

## Debounced Finger Events

Each break and re-contact must remain stable for at least 3 control steps. Re-contact is counted as gait only after a prior break in the same episode.

| Canonical | Finger | Break episodes | Re-contact episodes | Gait episodes | Break events | Re-contact events |
|---|---|---:|---:|---:|---:|---:|
| P1 | thumb | 201 | 38 | 38 | 232 | 42 |
| P1 | index | 202 | 31 | 31 | 235 | 33 |
| P1 | middle | 202 | 5 | 5 | 207 | 5 |
| P1 | ring | 93 | 4 | 4 | 94 | 193 |
| P1 | pinky | 0 | 0 | 0 | 0 | 0 |
| P2 | thumb | 122 | 26 | 26 | 149 | 27 |
| P2 | index | 122 | 0 | 0 | 122 | 0 |
| P2 | middle | 122 | 0 | 0 | 122 | 0 |
| P2 | ring | 121 | 65 | 65 | 201 | 202 |
| P2 | pinky | 0 | 0 | 0 | 0 | 0 |
| P3 | thumb | 100 | 0 | 0 | 100 | 0 |
| P3 | index | 0 | 0 | 0 | 0 | 0 |
| P3 | middle | 0 | 0 | 0 | 0 | 0 |
| P3 | ring | 0 | 0 | 0 | 0 | 0 |
| P3 | pinky | 0 | 0 | 0 | 0 | 0 |
| P4 | thumb | 71 | 10 | 10 | 78 | 10 |
| P4 | index | 38 | 0 | 0 | 38 | 0 |
| P4 | middle | 77 | 2 | 2 | 78 | 2 |
| P4 | ring | 0 | 0 | 0 | 0 | 0 |
| P4 | pinky | 0 | 0 | 0 | 0 | 0 |
| P5 | thumb | 1 | 0 | 0 | 1 | 0 |
| P5 | index | 28 | 1 | 1 | 29 | 1 |
| P5 | middle | 0 | 0 | 0 | 0 | 0 |
| P5 | ring | 0 | 0 | 0 | 0 | 0 |
| P5 | pinky | 0 | 0 | 0 | 0 | 0 |

This is deterministic evaluation of the existing checkpoint. No PPO training, teacher generation, or O0 mutation is part of this report.
