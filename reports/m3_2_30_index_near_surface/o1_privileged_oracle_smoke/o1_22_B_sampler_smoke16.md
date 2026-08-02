# O1.1 Deterministic Stability and Gait Report

**Classification: `REACHABILITY_ONLY`**

- Checkpoint: `/home/linkerhand/IsaacLab/logs/skrl/dex5_penspin_privileged_oracle/2026-08-01_21-22-08_ppo_torch_o1_22_B_smoke/checkpoints/agent_24.pt`
- Environments / steps: `16 / 120`
- Ballistic termination guard: `False`
- Completed episodes: `16`
- Raw gait / qualified gait: `18.75% / 0.00%`
- Support handoff success: `0.00%`
- Raw 180 deg / qualified 180 deg: `0.00% / 0.00%`
- Failure / contact loss / ballistic: `25.00% / 6.25% / 6.25%`
- Mean ballistic exposure: `0.19` control steps per completed episode

Qualified means: directional axis alignment >= 0.70, no ballistic, no contact-loss termination, and no failure.

## By Canonical

| Canonical | Episodes | Raw gait | Qualified gait | Support handoff | Raw 180 | Qualified 180 | Axis align | Failure | Contact loss | Ballistic |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| P1 | 0 | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% | 0.000 | 0.00% | 0.00% | 0.00% |
| P2 | 0 | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% | 0.000 | 0.00% | 0.00% | 0.00% |
| P3 | 16 | 18.75% | 0.00% | 0.00% | 0.00% | 0.00% | 0.135 | 25.00% | 6.25% | 6.25% |
| P4 | 0 | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% | 0.000 | 0.00% | 0.00% | 0.00% |
| P5 | 0 | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% | 0.000 | 0.00% | 0.00% | 0.00% |

## Thumb And Height Recovery

A thumb attempt is any post-release raw thumb contact that has not yet passed the three-step re-contact debounce. Height recovery requires a prior >=5 mm drop followed by return to within 2 mm of reset height.

| Canonical | Thumb release | Attempt | Re-contact | Mean break step | Mean attempt step | Height drop | Height recovery | Min height delta |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| P1 | 0.00% | 0.00% | 0.00% | 0.0 | 0.0 | 0.00% | 0.00% | 0.00 mm |
| P2 | 0.00% | 0.00% | 0.00% | 0.0 | 0.0 | 0.00% | 0.00% | 0.00 mm |
| P3 | 100.00% | 25.00% | 18.75% | 5.6 | 67.8 | 6.25% | 0.00% | -2.18 mm |
| P4 | 0.00% | 0.00% | 0.00% | 0.0 | 0.0 | 0.00% | 0.00% | 0.00 mm |
| P5 | 0.00% | 0.00% | 0.00% | 0.0 | 0.0 | 0.00% | 0.00% | 0.00 mm |

## Primary Failure Type

| Canonical | contact_loss | ballistic | penetration | throw | drop | drift | invalid |
|---|---:|---:|---:|---:|---:|---:|---:|
| P1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| P2 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| P3 | 0 | 1 | 3 | 0 | 0 | 0 | 0 |
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
| P3 | thumb | 16 | 3 | 3 | 16 | 3 |
| P3 | index | 1 | 0 | 0 | 1 | 0 |
| P3 | middle | 1 | 0 | 0 | 1 | 0 |
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
