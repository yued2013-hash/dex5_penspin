# Dex5 S0.12 Real Replay Dataset Validation

## Position in the pipeline

```text
Privileged oracle
  -> quaternion-valid simulation rollouts
  -> S0.11 simulation-pretrained student baseline
  -> S0.12 controlled real oracle-trajectory replay
  -> real successful replay dataset
  -> S0.13 sim-plus-real student adaptation
  -> S0.14 controlled real closed-loop student evaluation
```

S0.11 remains a valid sim-only baseline. S0.12 does not deploy that student and
does not train or fine-tune any policy.

## Goal

Determine which simulation-successful oracle trajectories remain executable on
the real Dex5, and produce source-isolated real replay records suitable for a
later S0.13 adaptation phase.

## Frozen inputs

- `quaternion_validated_rollouts_1200_v2.npz`
- Existing quaternion success definition
- Existing S0.8 measured hardware adapter
- Existing joint mapping, coupling model, limits, watchdog, and replay logger
- Frozen S0.11 sim-only checkpoint for later comparison only

## Stage A: offline ranking

Rank, do not modify, the 1200 rollouts using:

1. actual target-axis quaternion rotation and axis alignment;
2. strict hold duration and terminal angular speed;
3. TIM contact margin and absence of forbidden contacts;
4. maximum penetration and height loss;
5. action total variation, peak delta, and peak commanded speed;
6. diversity by `source_trajectory_id` and reset-bank state.

Select a small first batch, preferably 20 trajectories from distinct sources.
Expansion toward 100 trajectories requires the first batch to pass the real
measurement and replay-integrity gates.

## Stage B: replay packages

Generate immutable packages at progressively accepted speed tiers. The proposed
tiers are `0.12`, `0.18`, and `0.24 rad/s`. Each package must preserve the same
trajectory identity and include source hashes, mapped commands, timestamps,
limits, expected duration, and the adapter version.

Do not infer that faster replay is better. A higher tier is permitted only after
the preceding tier has acceptable tracking, temperature, contact, and object
retention. Any future `0.36 rad/s` tier requires a separate authorization.

## Stage C: real measurement contract

Every replay must record:

- commanded and measured joint positions;
- joint tracking error and response timing;
- fingertip force/contact channels;
- temperature and watchdog events;
- dropped-object and operator-abort labels;
- externally measured pen orientation over time.

The pen-orientation channel is mandatory. It may come from a calibrated camera,
marker-based pose tracker, or a validated frame-by-frame video annotation. Joint
motion, command motion, tactile changes, and integrated angular velocity are not
substitutes for actual pen quaternion change.

## Real success

Reuse the frozen task semantics on measured real motion:

```text
actual target-axis pen rotation >= 30 deg
axis alignment >= 0.70
stable terminal hold >= 150 ms
object retained
no safety abort
```

Tracking, contact, temperature, and penetration/proxy checks remain additional
validity gates. Do not relabel a failed trajectory merely to increase dataset
size.

## Split and provenance

Store simulation trajectory ID, source trajectory ID, replay package hash,
speed tier, trial ID, hardware configuration hash, and real outcome. Split later
S0.13 data by source trajectory and real trial session, never by individual
frames.

## Required outputs

```text
real_replay_selection_manifest.json
real_replay_trial_log.jsonl
real_replay_success_manifest.json
Dex5_S0_12_Real_Replay_Report.md
```

The report must include real success rate by speed tier, sim-to-real rotation
error, tracking error, contact/temperature statistics, failure taxonomy, and the
number of source-isolated trajectories authorized for S0.13.

## Current status

```text
Offline selection: NOT RUN
Real pen-orientation measurement: NOT ESTABLISHED
Powered replay: NOT RUN
Real successful dataset: NOT AVAILABLE
S0.13 adaptation: BLOCKED
```
