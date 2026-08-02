# Dex5 S0.11 Student BC Training Report

## Status

```text
BC overfit smoke: PASS
Full BC: PASS
Simulation closed-loop: PASS_EVALUATION_COMPLETE
Hardware control: NOT RUN
```

## Frozen inputs

- Dataset: `/home/linkerhand/桌面/deep learning/dex5_penspin/Dex5_Sim2Real_S0/student_s0_10/student_bc_dataset.npz`
- Dataset SHA256: `04fdd611ba255715d5d4159a1e7d933ee1c77d58871d00bcdc9e26d61db7e746`
- Observation: deployable 340D student history only
- Action: bounded 20D virtual normalized delta action
- Privileged observation used: `false`

## 100-episode overfit smoke

- Rows: `5961`
- Initial MSE: `0.06679531`
- Best MSE: `0.00064900`
- Final/initial ratio: `0.009716`

## Full BC

- Train/validation/test episodes: `891 / 112 / 197`
- Best epoch: `177`
- Best validation MSE: `0.00078029`
- Offline test MSE: `0.00065356`
- Checkpoint: `/home/linkerhand/桌面/deep learning/dex5_penspin/Dex5_Sim2Real_S0/student_s0_11/student_checkpoint.pt`
- Checkpoint SHA256: `d350630ddba6c975858f2f0aef6a493167fa66af43a92fc5904c446133877cf9`

The offline test MSE is not a task-success result. Closed-loop PhysX evaluation is required before accepting the student policy.

## Simulation closed-loop evaluation

The frozen 197-episode source-isolated test split was evaluated with the
hardware-compatible Medium DR wrapper. The policy read only the 340D student
observation; oracle observation was not passed to the model.

| Metric | Result |
|---|---:|
| Strict quaternion/axis/150 ms hold success | `152/197 = 77.16%` |
| Quaternion rotation >=30 deg | `184/197 = 93.40%` |
| Axis/target-band gate | `169/197 = 85.79%` |
| 150 ms hold | `152/197 = 77.16%` |
| Mean TIM contact ratio | `87.62%` |
| Maximum penetration | `0.224 mm` |
| Penetration >0.5 mm | `0/197` |

Failure taxonomy: 15 axis/target-band failures, 17 incomplete holds, and 13
episodes below 30 degrees. A full-episode TIM ratio of 100% is not expected:
the frozen success definition requires TIM during the strict terminal hold,
not during every transient control step.

Closed-loop details are stored in `student_closed_loop_eval.json`. This result
validates the first non-privileged, simulation-pretrained student baseline. It
does not establish real Dex5 task success and does not authorize hardware
deployment. The checkpoint must remain frozen as the sim-only baseline for the
later sim-only versus sim-plus-real comparison.

## Next gate

S0.12 replays a separately selected set of oracle trajectories under controlled
real-hardware execution. It must measure actual pen orientation, tracking,
contact, temperature, and object retention before assigning a real success
label. S0.13 may then fine-tune a separately versioned student using successful
real replay data. The S0.11 student itself is not the S0.12 replay controller.
