# Dex5 PenSpin Isaac/Lab migration notes

Current target:

```text
M3.2.30 first leg: earlier_pre_ring -> index_near_surface
```

This is intentionally approach-only. The MuJoCo smoke result says:

```text
near_surface keypose: usable
index_contact keypose: statically usable
near_surface -> index_contact active transition: not usable yet
```

So the first Isaac parallel RL task should train only:

```text
index_tip_dist_final < 14mm
```

and should not require index contact.

## Local setup status

Done:

```text
conda env: env_isaaclab, Python 3.11
IsaacLab repo: /home/linkerhand/IsaacLab
torch: 2.7.0+cu128, CUDA visible on RTX 5060 Laptop GPU
Isaac Sim package: 5.1.0.0
Isaac Lab import: OK
skrl: 2.1.0, installed without changing torch
visual summary: dex5_penspin/reports/m3_2_30_index_near_surface/visual_summary.html
Isaac first-leg bundle:
  dex5_penspin/reports/m3_2_30_index_near_surface/isaac_first_leg_bundle.npz
  dex5_penspin/reports/m3_2_30_index_near_surface/isaac_first_leg_bundle.json
Dex5 USD: dex5_penspin/isaac/assets/dex5_hand.usd
physical DirectRLEnv: Dex5-IndexNearSurface-Direct-v0
```

The NVIDIA Omniverse EULA was accepted by the user on 2026-07-14. Headless Isaac Lab and skrl launches are working.

Reproduce or repair install:

```bash
bash dex5_penspin/isaac/install_isaaclab_env.sh
```

The script uses:

```text
torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu128
isaacsim[all,extscache]==5.1.0 --extra-index-url https://pypi.nvidia.com
```

## Useful files

```text
dex5_penspin/isaac/export_m3_2_30_first_leg_bundle.py
dex5_penspin/isaac/m3_2_30_first_leg_reward.py
dex5_penspin/isaac/train_m3_2_30_first_leg_bc_warmstart.py
dex5_penspin/isaac/run_m3_2_30_index_near_surface_cache_skrl.sh
dex5_penspin/isaac/run_m3_2_30_index_near_surface_physics_skrl.sh
dex5_penspin/isaac/run_index_near_surface_physics_smoke.sh
dex5_penspin/reports/m3_2_30_index_near_surface/isaac_parallel_training_handoff.md
```

## Warm-start before Isaac physics training

Run a GPU behavior-cloning smoke from the MuJoCo first-leg source states to the near-surface target hand-qpos deltas:

```bash
source /home/linkerhand/anaconda3/etc/profile.d/conda.sh
conda activate env_isaaclab
python dex5_penspin/isaac/train_m3_2_30_first_leg_bc_warmstart.py
```

Outputs:

```text
dex5_penspin/reports/m3_2_30_index_near_surface/isaac_bc_warmstart_policy.pt
dex5_penspin/reports/m3_2_30_index_near_surface/isaac_bc_warmstart_metrics.json
```

## Isaac physics task

Done for smoke training:

```text
Dex5-IndexNearSurface-Cache-Direct-v0
```

This cache-backed DirectEnv is intentionally not the final hand+pen physics task. It lets Isaac Lab/skrl exercise
parallel rollout, PPO plumbing, the first-leg reward shape, and the source->near curriculum using the exported bundle.

Run after accepting EULA:

```bash
bash dex5_penspin/isaac/run_m3_2_30_index_near_surface_cache_skrl.sh
```

Implemented physical task:

```text
Dex5-IndexNearSurface-Direct-v0
```

It uses:

```text
real Dex5 URDF-derived USD articulation, meshes, joints, limits, inertias, and actuators
9g, 170mm x 12mm PhysX cylinder pen with gravity
five fingertip sphere collision proxies matching the MuJoCo proxy-only curriculum model
reset distribution: *_source_qpos / *_source_qvel from isaac_first_leg_bundle.npz
target reference: *_target_hand_qpos / *_target_pen_pos / *_target_pen_quat
reward: dex5_penspin.isaac.m3_2_30_first_leg_reward.compute_index_near_surface_reward
20 residual joint actions around the scripted source-to-target teacher
126 observations including physical hand/pen state, fingertip geometry, and contact forces
30-40ms episode horizon matching each split first-leg primitive
```

Run the deterministic physical smoke first:

```bash
bash dex5_penspin/isaac/run_index_near_surface_physics_smoke.sh --num_envs 16 --steps 2
```

Run parallel PPO:

```bash
bash dex5_penspin/isaac/run_m3_2_30_index_near_surface_physics_skrl.sh \
  --num_envs 256 --max_iterations 100
```

The corrected 2026-07-14 256-env, 20-iteration smoke completed in 18.0s. Its total reward mean rose from 0.76 to
2.91, final near-14mm ratio was 100%, index contact ratio was 98.7%, and the per-step failure fraction fell from
about 10.3% to 2.9%. The reward uses the measured post-reset index distance, not the old fixed 30mm cache value.
The stable checkpoint and run summary are in
`dex5_penspin/reports/m3_2_30_index_near_surface/isaac_physics_direct/`; full TensorBoard data remain under
`/home/linkerhand/IsaacLab/logs/skrl/dex5_index_near_surface_physics_direct/`.

## Physical recapture and P3 curriculum

The first physical curriculum now extends beyond `index_near_surface` without treating the old MuJoCo caches as
physically valid reset states:

```text
Dex5-IndexContact-Direct-v0   near surface -> index contact
Dex5-IndexRecapture-Direct-v0 index contact -> contact-frame recapture
Dex5-RingTouch-Direct-v0      physical recapture -> B23 ring touch
Dex5-P3Entry-Direct-v0        physical ring touch -> P3 dynamic pre-entry
```

The MuJoCo state exporters reorder the raw thumb-first state into the DEX5 control order before constructing Isaac
resets. Recapture and ring-touch reset banks are collected from successful PhysX rollouts. Old MuJoCo transitions
are retained only as relative hand/pen hints, which avoids importing their proxy penetration into PhysX.

Run one phase at a time:

```bash
DEX5_TASK=Dex5-RingTouch-Direct-v0 \
  bash dex5_penspin/isaac/run_recapture_p3_physics_skrl.sh \
  --num_envs 256 --max_iterations 60
```

Stable checkpoints and the machine-readable result summary are in:

```text
dex5_penspin/reports/m3_2_30_index_near_surface/isaac_recapture_p3_physics/
```

The 2026-07-14 physical results are:

```text
phase                    success   max proxy penetration   pen position error
index contact             23.2%             0.10mm                2.79mm
contact-frame recapture   20.9%             2.33mm                2.20mm
ring touch                48.5%             0.21mm                1.35mm
P3 old static pre-entry    0.0%             0.19mm                6.77mm
```

The previous roughly 5mm proxy depth has therefore been reduced below 0.25mm in index contact, ring touch, and P3
pre-entry. Recapture is below 2.4mm but still needs refinement. The old static-quaternion P3 objective was retired;
the current task measures signed planar deflection relative to each physical B23 reset.

### Current stage goal and measurable gates

The current Isaac Lab stage defines P2->P3 as a measurable dynamic task and begins the P3->braked-P4 curriculum.

Current stage status:

```text
index contact:  low-penetration contact acquisition
recapture:      partially learned, but still too much proxy penetration
ring touch:     low-penetration guard/contact phase
P2->P3:         nominal dynamic deflection task passes; not yet canonical Manus P3
P3->P4:         20deg approach learns progress but is below its hold-success gate
```

The near-term measurable gates are:

| Gate | Current | Next target |
|---|---:|---:|
| index contact success | 23.2% | >= 50% |
| index contact proxy penetration | 0.10mm | <= 0.25mm |
| recapture success | 20.9% | >= 40% |
| recapture proxy penetration | 2.33mm | <= 1.0mm, then <= 0.5mm |
| ring touch success | 48.5% | >= 60% |
| ring touch proxy penetration | 0.21mm | <= 0.25mm |
| P3 pre-entry contact hold | stable | keep stable |
| P3 pre-entry proxy penetration | 0.19mm | <= 0.25mm |
| P2->P3 nominal episode success | 99.93% / 4096 episodes | keep >=60% under broader reset randomization |
| P2->P3 mean max deflection | 11.99deg | keep >=7deg |
| P4 approach total phase | 16.08deg | 17-23deg stage band |
| P4 approach hold success | 0.57% | >=30%, then advance to 35deg |
| braked P4 | not trained | 50-60deg, <=12rad/s, >=100ms hold |

This means the next engineering focus is:

```text
1. Raise P4-approach thumb/index/middle contact retention before advancing to 35deg.
2. Re-evaluate P2->P3 with broader reset and actuator randomization; nominal success alone is not robustness.
3. Build the 50-100ms MuJoCo/PhysX alignment report and audit canonical P3 against Manus RMS <0.12rad.
```

Do not treat the nominal P2->P3 result as complete canonical P3. The policy passes the dynamic contact/deflection
task, while the separate Manus pose-RMS gate remains open.

The P4 curriculum tasks are:

```text
Dex5-P4Approach-Direct-v0  20deg contact-preserving approach
Dex5-P4Midway-Direct-v0    35deg transition
Dex5-BrakedP4-Direct-v0    55deg final target, <=12rad/s, >=100ms hold
```

Independent P2->P3 evaluation:

```bash
bash dex5_penspin/isaac/run_evaluate_p2_p3_policy.sh \
  --task Dex5-P3Entry-Direct-v0 \
  --checkpoint dex5_penspin/reports/m3_2_30_index_near_surface/isaac_recapture_p3_physics/p2_p3_best_agent.pt \
  --output dex5_penspin/reports/m3_2_30_index_near_surface/isaac_recapture_p3_physics/p2_p3_evaluation.json
```

Collect successful physical states for the next phase with:

```bash
bash dex5_penspin/isaac/run_collect_physics_recapture_states.sh
```

## Isaac Lab visualization

Isaac Lab normally provides an interactive Kit window. The single-environment policy viewer is:

```bash
bash dex5_penspin/isaac/run_isaaclab_physics_viewer.sh
```

Select a different phase/checkpoint with environment variables:

```bash
DEX5_TASK=Dex5-P3Entry-Direct-v0 \
DEX5_CHECKPOINT="/home/linkerhand/桌面/deep learning/dex5_penspin/reports/m3_2_30_index_near_surface/isaac_recapture_p3_physics/p3_pre_entry_best_agent.pt" \
  bash dex5_penspin/isaac/run_isaaclab_physics_viewer.sh
```

On the current RTX 5060 laptop setup, both the interactive window and Replicator video path currently crash during
Isaac Sim 5.1 Hydra RTX startup in `librtx.scenedb.plugin.so`, before the Dex5 environment is created. Headless
PhysX/CUDA simulation and 256-environment training are verified working. Treat this as a renderer/driver issue,
not a policy or DirectRLEnv failure; do not add `--video` to training until the RTX path is repaired.
