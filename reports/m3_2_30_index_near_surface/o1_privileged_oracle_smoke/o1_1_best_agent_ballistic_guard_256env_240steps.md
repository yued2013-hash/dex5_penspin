# O1.1 Deterministic Stability and Gait Report

**Classification: `REACHABILITY_ONLY`**

- Checkpoint: `/home/linkerhand/IsaacLab/logs/skrl/dex5_penspin_privileged_oracle/2026-07-30_17-42-49_ppo_torch/checkpoints/best_agent.pt`
- Environments / steps: `256 / 240`
- Ballistic termination guard: `True`
- Completed episodes: `569`
- Raw gait / qualified gait: `28.47% / 0.00%`
- Raw 180 deg / qualified 180 deg: `73.11% / 0.00%`
- Failure / contact loss / ballistic: `61.51% / 1.05% / 45.17%`
- Mean ballistic exposure: `0.45` control steps per completed episode

Qualified means: directional axis alignment >= 0.70, no ballistic, no contact-loss termination, and no failure.

## By Canonical

| Canonical | Episodes | Raw gait | Qualified gait | Raw 180 | Qualified 180 | Axis align | Failure | Contact loss | Ballistic |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| P1 | 199 | 38.69% | 0.00% | 100.00% | 0.00% | 0.367 | 52.26% | 3.02% | 51.76% |
| P2 | 123 | 61.79% | 0.00% | 97.56% | 0.00% | 0.300 | 100.00% | 0.00% | 100.00% |
| P3 | 112 | 0.00% | 0.00% | 86.61% | 0.00% | 0.432 | 11.61% | 0.00% | 0.00% |
| P4 | 108 | 8.33% | 0.00% | 0.00% | 0.00% | 0.188 | 100.00% | 0.00% | 26.85% |
| P5 | 27 | 0.00% | 0.00% | 0.00% | 0.00% | -0.071 | 7.41% | 0.00% | 7.41% |

## Primary Failure Type

| Canonical | contact_loss | ballistic | penetration | throw | drop | drift | invalid |
|---|---:|---:|---:|---:|---:|---:|---:|
| P1 | 0 | 102 | 0 | 1 | 1 | 0 | 0 |
| P2 | 0 | 121 | 0 | 2 | 0 | 0 | 0 |
| P3 | 0 | 0 | 9 | 4 | 0 | 0 | 0 |
| P4 | 0 | 29 | 27 | 0 | 52 | 0 | 0 |
| P5 | 0 | 2 | 0 | 0 | 0 | 0 | 0 |

## Debounced Finger Events

Each break and re-contact must remain stable for at least 3 control steps. Re-contact is counted as gait only after a prior break in the same episode.

| Canonical | Finger | Break episodes | Re-contact episodes | Gait episodes | Break events | Re-contact events |
|---|---|---:|---:|---:|---:|---:|
| P1 | thumb | 181 | 36 | 36 | 210 | 39 |
| P1 | index | 199 | 35 | 35 | 234 | 36 |
| P1 | middle | 199 | 6 | 6 | 205 | 6 |
| P1 | ring | 18 | 6 | 6 | 18 | 186 |
| P1 | pinky | 0 | 0 | 0 | 0 | 0 |
| P2 | thumb | 86 | 23 | 23 | 95 | 23 |
| P2 | index | 123 | 0 | 0 | 123 | 0 |
| P2 | middle | 123 | 1 | 1 | 124 | 1 |
| P2 | ring | 114 | 66 | 66 | 172 | 204 |
| P2 | pinky | 0 | 0 | 0 | 0 | 0 |
| P3 | thumb | 112 | 0 | 0 | 112 | 0 |
| P3 | index | 0 | 0 | 0 | 0 | 0 |
| P3 | middle | 0 | 0 | 0 | 0 | 0 |
| P3 | ring | 0 | 0 | 0 | 0 | 0 |
| P3 | pinky | 0 | 0 | 0 | 0 | 0 |
| P4 | thumb | 68 | 8 | 8 | 68 | 8 |
| P4 | index | 22 | 0 | 0 | 22 | 0 |
| P4 | middle | 77 | 1 | 1 | 77 | 1 |
| P4 | ring | 0 | 0 | 0 | 0 | 0 |
| P4 | pinky | 0 | 0 | 0 | 0 | 0 |
| P5 | thumb | 1 | 0 | 0 | 1 | 0 |
| P5 | index | 27 | 0 | 0 | 27 | 0 |
| P5 | middle | 0 | 0 | 0 | 0 | 0 |
| P5 | ring | 0 | 0 | 0 | 0 | 0 |
| P5 | pinky | 0 | 0 | 0 | 0 | 0 |

This is deterministic evaluation of the existing checkpoint. No PPO training, teacher generation, or O0 mutation is part of this report.
