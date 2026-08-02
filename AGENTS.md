# Dex5 PenSpin Development Rules

## Current Phase

The active sim-to-real branch is **S0.9 Trajectory-0 Held-Pen Replay:
INTERFACE_PASS_TASK_FAIL**.
S0.8.1 Thumb Global Mapping and Mapped P3 Static Validation have passed, and
the mapped replay package has passed offline preflight. Trajectory 0 completed
once under explicit human arming without dropping the pen, but the pen did not
visibly rotate. Empty-hand,
one-channel-at-a-time full-range scans
for thumb joints 16-19 and the merged direction-aware mapping have passed.
Preserve raw/unwrapped feedback, sign, offset, hysteresis, and the separate
forward/reverse piecewise inverses.

The accepted mapped P3 thumb SDK command is
`[0.1980349710, 0.7970652983, 0.9899153359, 0.6662725608]`. It passed a
fresh-baseline empty-hand gate with maximum normalized thumb error `0.0985 rad`.
Do not replace the full-range branch curves with this local joint-16 correction.
The first powered acceptance is restricted to trajectory 0, count 1, after
empty-hand mapped-P3 staging and explicit confirmation that the pen, soft pad,
and physical emergency stop are ready. PPO, teacher generation, student training, autonomous
policy execution, online learning, and modification of O0 or frozen S0 data
remain prohibited.

Do not call trajectory 0 a task pass or generalize it to all 20 replay trajectories,
repeatability, force validation, or autonomous policy deployment. The interface-complete
run contains 308 commands over 3.178 s with 0.07962 rad/s peak command speed.
No force samples or temperature events were available. Raw SDK dq reached an
implausible 242.98 rad/s and must remain diagnostic-only; use the 50 ms
position-difference signal and operator drop annotation for this acceptance.

Freeze the remaining 19 powered trajectories. The source simulation episode
integrates world-z angular velocity to about 38.4 degrees, but its quaternion
net orientation changes only about 7.7 degrees (about 20.7 degrees cumulative
shortest-step orientation change). The original 46-step/60-Hz episode was also
time-stretched to 308 steps/100 Hz for hardware, removing the dynamics assumed
by an open-loop rotation skill. Before any further powered replay, reconcile
the angular-velocity and quaternion rotation metrics, select a trajectory that
passes both, and validate a hardware timing profile without treating arbitrary
time scaling as dynamically equivalent.

The first two trajectory-0 attempts were aborted before trajectory recording
by a single-window 20D feedback-speed gate. The second trigger was derived,
coupled DIP joint 15 at 0.2146 rad/s, without visible displacement. The approved
direct-replay gate now monitors only the 16 independent actuator-associated
feedback channels and requires three consecutive fresh 50 ms violations above
0.12 rad/s. Derived DIP joints 3/7/11/15 remain recorded but do not trigger the
hard gate. Command speed, command delta, watchdog, thermal abort, and per-run
human arming remain unchanged. The mapped replay package is unchanged.

Trajectory 1 subsequently exposed repeatable enable transients on independent
PIP feedback channels before any replay row was recorded: joint 10 reached
0.2279 and 0.2951 rad/s under direct start, while joint 6 reached 1.0691 rad/s
under the legacy soft-resume path. New powered replay must therefore use the
guarded `low_gain_preload` startup: zero-torque hybrid control with reduced
controller gains, 0.01 rad/s P3 alignment, three stable 50 ms windows below
0.04 rad/s, and a single-window 0.12 rad/s abort during preload and the final
normal-gain switch. The replay process must query the controller parameter
service before ARM and reject preload when non-thumb or thumb hybrid kp exceeds
0.20, or hybrid kd exceeds 0.40. `legacy_direct` and `legacy_soft_resume` are retained only
for reproducing historical diagnostics. No trajectory may execute until the
new startup gate passes; O0, replay bytes, mapping, and training assets remain
frozen.

Trajectory 1 also produced a loaded replay transient on independent PIP joint
14: the original command stayed below `0.0758 rad/s`, while 50 ms feedback
reached `0.3723 rad/s`. Do not relax feedback, watchdog, thermal, joint-limit,
or emergency-stop gates to force completion. The only approved retry is the
separately versioned `trajectory1_slow_diagnostic` package. It preserves the
frozen trajectory-1 endpoints, limits command speed to `0.03 rad/s`, keeps the
reduced-gain hybrid controller active throughout replay, and uses a
single-window `0.12 rad/s` independent-joint feedback abort. This package is a
hardware diagnostic only: completion does not establish visible rotation,
dynamic equivalence, task success, or approval of trajectories 2-19. Before
each powered retry, restage mapped P3 with an empty hand, pass a fresh startup
gate, and obtain a new explicit operator confirmation after the pen is placed.

The first powered slow-diagnostic retry passed loaded low-gain startup and
executed 237/1007 command rows (about 2.36 s) before the single-window gate
aborted on independent joint 4 at `0.4897 rad/s`. Recorded samples before the
trigger stayed at or below `0.0245 rad/s`; the triggering frame occurs during
command validation and is reported in `abort_reason`, not in the pre-trigger
sample maximum. The controller was stopped after software E-stop. Do not retry,
loosen the `0.12 rad/s` gate, or extend this profile until operator drop/jam
annotation and the joint-4 transient source are reviewed.

A second independently restaged powered attempt reproduced the same failure:
274/1007 rows (about 2.73 s), then joint 4 (`Roll_31R`, middle-finger roll)
reached `0.2853 rad/s`. The two trigger times and amplitudes differ, but the
same physical channel and early trajectory region fail under load. Treat this
as a repeatable trajectory/controller/load incompatibility, not isolated
measurement noise. A third powered replay is prohibited until an offline
joint-4 command/contact audit produces a separately reviewed correction.

The operator subsequently replaced the oversized pen with a measured
approximately `9.8 mm` middle-diameter pen matching the simulation reference.
The new condition passed a dedicated loaded low-gain hold: 15 preload, 16
alignment, and 28 final stable windows, with peak 50 ms speed below
`0.00033 rad/s`. This validates static load only. The next permitted powered
step is the separately hashed 0-2.0 s prefix, which ends before both historical
joint-4 failures; full trajectory replay remains frozen until that prefix is
observed and accepted.

The 9.8 mm pen 0-2.0 s prefix is now accepted: 201/201 rows completed, the pen
did not drop, the operator observed visible rotation, and the safety-relevant
50 ms position-difference peak was `0.00143 rad/s`. Raw SDK `dq` still produced
an implausible `2.4544 rad/s` diagnostic spike and is not a safety metric. This
is a prefix-only task pass, not full trajectory-1 acceptance. The next powered
boundary must cross the historical failure region in a separately hashed
limited-duration package before full 10.07 s replay can be considered.

S0.8 Real Calibration Closure is accepted.
S0.1 mapping, S0.2 simulation audits, and guarded S0.8 local hardware
identification have passed their applicable gates. Joint direction and public
URDF-command mapping are confirmed for 16/16 actuator channels. The measured
PIP/DIP response ratio is about 0.29; detectable physical onset was 14-31 ms.
This model remains local to the acceptance posture and is not a motor-shaft
model. A separately versioned hardware-compatible P3
bank now contains 582 R0-R3 validated states and is the only permitted reset
source for `Dex5-S0-ShortRotation`. The task applies forced PIP/DIP coupling,
a 60 Hz policy to 100 Hz zero-order-hold command chain, and provisional
delay, tracking-gain, and velocity-limit randomization. The opt-in S0.8 audit
configuration replaces provisional values with the measured local response.

The three-seed oracle gate and successful-rollout dataset are complete. Freeze
oracle training, reward, task horizon, checkpoint selection, and rollout data.
PPO is no longer permitted on this branch without a new explicit phase.
The curriculum may change only the target angle, tolerance, and episode
horizon; it must preserve the final 45-degree gate, reward, observation,
hardware adapter, and validated reset bank. O0/O1 training, handoff experiments,
teacher generation, student training, autonomous hardware rollout, and
modification of O0 remain frozen. Powered replay remains a separate guarded
acceptance step. The S0.8 `0.04-0.08 rad/s` range is a safety envelope, not a
measured or approved robot limit. The public SDK command remains a 20D
absolute URDF-joint target; do
not bypass the vendor anemone URDF-to-motor conversion or describe the 16D
logical coordinates as motor-shaft commands.

The project has completed **O1.19: Reduced-order Handoff MPC Feasibility Audit** as a
diagnostic phase. O1.15 established sparse long-horizon existence, O1.16 found
one narrow fixed-plan basin around bank 1155, O1.17 rejected nominal joint
trajectory tracking as a widening mechanism, and O1.18 found that the tested
one-step task-Jacobian residual controller also fails its calibration safety
gate and does not beat matched open-loop replay. O1.19 further shows that the
tested 50/100 ms reduced-order receding-horizon controller also fails
calibration safety and all widening gates. These artifacts must not be used as
reset samplers, canonical filters, teacher sources, or policy-training distributions.

O0 was explicitly accepted on 2026-07-30. Its frozen input is:

- `reports/m3_2_30_index_near_surface/canonical_grasp_v9_o0/canonical_reset_bank/P1_P5_v9_physx_projected_R3_validated.npz`
- SHA-256: `73ab3cc0ffada7759da428767956b353b39b556d77c60accbd95c4891073ac65`

The purpose of O1 is to determine whether a privileged oracle can produce
rotation and emergent finger gait from the validated canonical reset
distribution. O1 is a reachability and behavior-diagnostics smoke phase, not
a final performance run.

## Do Not

During O1, do not:

- modify or expand the frozen P1-P5 bank without returning the affected data
  to O0 and repeating R0-R3 validation;
- add P1-P5 pose-tracking rewards or require the policy to visit canonical
  states as trajectory waypoints;
- start multi-seed, multi-GPU, 5090/H200, or long training before the single
  seed reachability smoke passes;
- claim finger gait from isolated one-step contact noise;
- claim success from rotation alone without checking throw, slip, penetration,
  object translation, and wrong-axis angular velocity;
- start student distillation, real replay, or real fine-tuning;
- create, expand, or optimize new teacher trajectories;
- optimize local primitive success rates;
- resume H1-H6, ring-contact, force-servo, adapter, or full-chain training as
  the main line of work;
- treat a state observed in a successful trajectory as a valid reset state;
- use an unvalidated dataset for training, evaluation, or headline results.

## Priority Order

Work in this order:

1. preserve the frozen O0 bank and provenance;
2. verify oracle observation and reset-mixture wiring;
3. run one-seed reachability smoke at modest scale;
4. diagnose rotation, sustained contact transitions, and exploit behavior;
5. scale only after the smoke gates pass.

Do not skip an earlier priority to improve a later one.

## Reset-State Requirement

A trajectory success is **not** equal to a valid reset state.

A canonical reset record should preserve all available continuation context,
including:

- hand joint position and velocity;
- pen pose, linear velocity, and angular velocity;
- PD target;
- current/previous action history;
- control-delay buffers when enabled;
- randomized physics parameters;
- canonical ID and source trajectory ID;
- asset, environment, joint-order, and state-schema versions.

Do not assume that PhysX contact manifolds are serializable. Rebuild contacts
through controlled simulation steps and validate the resulting state.

## Required Validation

Every new dataset must pass the applicable R0-R3 validation before it is
promoted for use:

- **R0 - State integrity:** valid dimensions and ranges, no NaN/Inf, bounded
  penetration, and complete metadata.
- **R1 - Physics reinstantiation:** after state restoration and 1-2 physics
  steps, PhysX remains stable and contacts can be reconstructed without an
  explosion or immediate invalid state.
- **R2 - Behavior continuation:** static states pass the defined hold test;
  dynamic states pass saved-control or source-policy continuation. Do not
  require a dynamic contact pattern to remain frozen.
- **R3 - Oracle readiness:** repeated resets are reproducible and small local
  perturbations remain within a physically valid, explorable neighborhood.

Until validation is complete, store data only as a clearly labeled
`candidate` or `quarantine` artifact. Record validation results, rejection
reasons, source trajectory IDs, configuration, and schema versions alongside
the data.

Train/evaluation splits must be made by `source_trajectory_id`, never by
randomly splitting nearby states derived from the same trajectory.

## Scope of Existing Assets

### Canonical Naming

Keep the two canonical namespaces separate:

- `P1-P5` are the formal V9 grasp semantics. P1/P2/P3/P4 require
  thumb-index-middle control, with ring moving under the pen as auxiliary in
  P2; P4 places the thumb under the pen end; P5 is a thumb-index pinch while
  middle and ring withdraw.
- `C1-C6` are dynamics-role exploration entries. They are not aliases for
  P1-P5 and must not be relabeled as formal V9 grasps.

Build, validate, report, and visualize the two namespaces in separate
directories. A state may move between them only through an explicit semantic
and R0-R3 audit.

Existing local policies, teachers, contact/wrench tools, and primitive results
may be used as candidate-state sources, baselines, and diagnostics. They must
not prescribe a required contact sequence or be expanded into a manually
stitched full-chain controller during O0.

## O1 Smoke Contract

- Start with one seed and about 1024 environments.
- Use the reset mixture P1/P2/P3/P4/P5 = 40/20/20/15/5 unless an experiment
  explicitly records another mixture.
- Keep canonical IDs as reset provenance and evaluation labels, not policy
  waypoints.
- First inspect a short run before committing to 5M-20M simulation steps.
- Record pen target-axis rotation, wrong-axis angular velocity, linear speed,
  penetration, drop/throw events, and per-finger contact transitions.
- A gait event requires a debounced contact release or re-contact lasting at
  least 2-3 control steps. Report the exact debounce rule.
- Preserve every run's bank hash, observation schema, reward configuration,
  seed, environment count, checkpoint, and evaluation artifact.

O1 smoke passing means that useful rotation and sustained contact switching
appear without catastrophic exploits. It does not mean that a complete
PenSpin oracle, student policy, or sim-to-real system is complete.

## O1.2 Current Finding

O1.2 uses qualified gait, not raw rotation or raw contact switching, as its
gate. As of the first local ablations, qualified gait remains zero. The most
useful candidate is P3-only training with contact-gated rotation and a dense
ballistic penalty: it nearly removes ballistic behavior and produces sustained
middle-finger break/re-contact, but still drops the pen and misses the axis
alignment threshold. Do not scale this candidate or claim finger-gait success.

Do not increase axis reward weights blindly. The P3Axis ablation demonstrated
that a stronger axis-quality reward can restore high-speed contact-loss
exploitation. The next objective change must address thumb retention and
height support without prescribing a complete contact-state sequence.

## O1.3 Current Finding

The 158-dimensional P3State ablation adds rotation history, reset-relative
height, and thumb release/re-contact history without changing the O1.2 reward.
It proves the policy can attempt and complete thumb re-contact, but nearly all
TIM contacts still release and the support chain collapses before height
recovery. Treat zero drop terminations as censored by earlier contact-loss
termination, not as solved object-height stability.

Do not duplicate current thumb force as a new observation: the oracle already
receives binary contact, 3-D contact force, force norm, and slip for every
finger. Do not increase training scale yet. The next experiment must test
support handoff/overlap and recovery feasibility while keeping reward and O0
frozen.

## O1.4 Current Finding

Matched fixed-horizon diagnostics identify two distinct P3 support failures.
The O1.2 penalty60 policy releases the thumb almost immediately and never
recontacts it; index-middle contact persists temporarily, but it does not
provide a recoverable upward support wrench. The O1.3 P3State60 policy causes
near-synchronous TIM release, a large force collapse, and only late thumb/index
recontacts after the pen has already lost height. A recontact is not a valid
handoff unless support overlap and object-height recovery also occur.

The matched zero-action control remains stable on the same P3 reset rows, so
reset instability is not the primary explanation. Existing evidence also does
not support declaring the handoff physically unreachable. Treat the current
objective and temporal support-credit assignment as the leading bottleneck,
while keeping the distinction between objective, policy representation, and
optimizer limitations explicit. Do not modify reward or scale training inside
O1.4; this phase produces diagnostics only.

## O1.5 Current Finding

The matched P3-only, one-seed, 60-iteration comparison keeps O0, reward, PPO
configuration, and transition budget fixed. The temporal policy receives
generic per-finger contact-duty and force-history features plus object-height
and net-support-force trends; it receives no required contact order or finger
phase label.

Temporal input greatly increases raw debounced transitions, but qualified gait,
support handoff, and height recovery remain zero. TIM overlap decreases from
about 97.6% to 50.1%, while the failure rate rises to 100%. Therefore raw
break/re-contact frequency is not evidence of temporal-credit success. Do not
scale this temporal checkpoint or treat its high raw-gait rate as skill
improvement. Any subsequent objective or architecture experiment must retain
the joint gate: sustained support overlap, recoverable height, non-ballistic
motion, and qualified gait.

## O1.6 Current Finding

The matched P3-only O1.6 ablation compares the O1.5 temporal baseline against
a support-margin objective and a finite-window height-recoverability objective.
All three conditions use the same frozen O0 bank, 172-dimensional temporal
observation, seed, environment count, PPO iteration budget, and evaluation
horizon. Neither new objective prescribes a finger identity or contact order.

The support-margin and recoverability policies do not improve load transfer.
They reduce the approximate mean episode length from about 62 control steps to
about 10, with contact-loss and ballistic rates near 99%. This is a dense
penalty/termination-return exploit: terminating early avoids accumulating
future per-step support penalties. The higher reported TIM-overlap ratio in
these short episodes is a truncation artifact caused by retaining the initial
P3 overlap for a few steps; it is not evidence of successful handoff.

Do not scale, distill, or use the O1.6 B/C checkpoints as skill policies. Do
not conclude that support continuity is irrelevant or physically unreachable;
the experiment rejects the naive persistent negative shaping under the current
termination semantics. Before another support-objective PPO run, prove with
zero-action, random, and intentional-failure controls that early termination
has lower return than continued physically valid behavior. The next objective
must be termination-invariant, for example through fixed-horizon/absorbing
failure evaluation, a remaining-horizon terminal cost, or validated
potential-based shaping.

## O1.7 Current Finding

O1.7 keeps the frozen O0 bank, P3-only resets, 172-dimensional temporal
observation, O1.5 baseline reward, PPO configuration, seed, and transition
budget fixed. The only intended differences are fixed-horizon termination,
an additional one-time terminal failure penalty, and a 12-step recovery grace
window. All three conditions pass zero/random/intentional-release return-order
audits: intentional early failure has lower return than zero-action behavior.

The O1.6 short-episode exploit is removed. All learned O1.7 policies average
119 control steps, and none exhibits contact-loss collapse. Nevertheless,
qualified gait, strict TIM support handoff, and height recovery remain zero in
all conditions. Fixed-horizon policies retain index-middle contact while the
thumb releases and never recontacts; their near-100% TIM overlap is static
support, not load transfer. The grace policy produces 1.56% thumb recontact,
but every evaluated episode loses at least 5 mm of height and none recovers.

Treat termination credit assignment as a confirmed cause of the O1.6
exploitation, but not as a sufficient explanation for the P3 handoff failure.
Do not scale or distill the O1.7 policies. The strict support-handoff metric
must require a debounced thumb/index/middle release-recontact event; ring or
pinky chatter alone is not a P3 TIM handoff. Preserve episode-level failure
latches whenever early termination is disabled.

## O1.8 Current Finding

The first O1.8 result was invalid because it read the contact sensor directly
from the reset-return frame, before PhysX refreshed contact data. Its reported
6/12 feasibility and `B_POLICY_LEARNING_BOTTLENECK` conclusion are withdrawn.
Never use `O1_8_Handoff_Reachability_Audit.pre_contact_rebuild_invalid.json` as
reachability evidence.

The corrected audit performs two zero-residual contact-rebuild steps before
capturing initial force, selecting the strongest TIM support finger, clearing
handoff event counters, and starting the 100-300 ms horizon. It finds 0/12
strict feasible states at the original 64 x 10 CEM budget. A sensitivity run
at 128 x 30 also finds 0/12. The formal run contains one safe middle-support
release/re-contact candidate, bank 996, but its axis alignment is 0.385 under
the 0.5 gate. Therefore the current result is only
`A_UNREACHABLE_WITHIN_AUDITED_BOUNDS`; it is not a mathematical impossibility
claim and it does not establish a policy-learning bottleneck.

The CEM action plans remain unserialized. Do not reconstruct, export, distill,
or label them as teachers. Any reset-time contact or wrench audit must perform
the controlled contact-rebuild steps before reading sensor values.

## O1.9 Current Finding

O1.9 read-only reset-replay extracts 66 geometry, contact-wrench, pen-state,
velocity, and support-margin features from all 396 P3-train resets after the
same two-step contact reconstruction. The 12 corrected O1.8 rows represent
12 independent parents and sources, about 3.03% of P3-train. Their median
per-feature span coverage is about 65.2%, and one row is near-feasible only in
the limited sense that it completes a safe TIM transition but misses the axis
gate.

Because corrected O1.8 has zero strict positive labels, a handoff-feasibility
classifier, ROC AUC, and feasible-basin probability map are not identifiable.
Do not train a one-class surrogate, relabel the near-feasible row as positive,
or use the withdrawn labels. A reset-coverage, exploration, or representation
cause cannot yet be selected among A/B/C. All requested raw feature families
are already observed by the privileged oracle or algebraically derivable, but
this alone does not exclude architecture, temporal encoding, or optimization
limitations.

## O1.10 Current Finding

The valid measured boundary is 12 spread-sampled P3 entries, 100-300 ms,
three-waypoint 20-dimensional residual joint targets, and +/-0.18 rad. Strict
feasibility is 0/12 at population 64 x 10 iterations and remains 0/12 at the
128 x 30 search-budget sensitivity. The formal taxonomy is 11 no-TIM-transition
and one axis-misaligned entry. Bank 996 is the only safe near-boundary entry:
it completes the dominant-TIM release/re-contact but reaches axis alignment
0.385 under the 0.5 gate.

The O1.10 400-600 ms, TIM-only small-cooperation, and +/-0.30 rad conditions
were `NOT_MEASURED`, not failures. The apparent low-throughput path was later
traced to a report-serialization exception (`torch.flatnonzero` is unavailable
in the installed PyTorch) hidden by a slow Isaac Sim shutdown. O1.11 fixes the
serializer, prints the original traceback before shutdown, and records
per-iteration wall time. Do not retroactively promote aborted O1.10 runs; use
only completed JSON artifacts.

Do not claim absolute dynamics unreachability. Current evidence only closes
the <=300 ms, +/-0.18 rad boundary and rejects the hypothesis that a larger
CEM sample budget alone solves it. Longer horizons, a lower-dimensional TIM
cooperation parameterization, larger action limits, and different P3 entries
remain open hypotheses. No PPO, reward change, O0 change, teacher, or trajectory
export is allowed while this boundary remains open.

## O1.11 Current Finding

O1.11 uses the same frozen O0 bank, P3 entries 817/948/996, population 64,
10 CEM iterations, three waypoints, two-step contact reconstruction, and the
unchanged strict handoff gate. No PPO, reward, teacher, or O0 change is part of
this audit.

The 300 ms 20-D +/-0.18 rad baseline and the 300 ms TIM-only +/-0.12 rad
condition both produce 0/3 TIM transitions and 0/3 strict handoffs. The 400 ms
20-D +/-0.18 rad condition also produces 0/3 strict handoffs, but bank 996
reaches a safe TIM release/re-contact with only 0.744 mm height loss. It fails
the axis gate at 0.108 versus the required 0.5. Banks 817 and 948 remain
no-transition entries.

Therefore small TIM-only cooperation does not support the action-space
insufficiency hypothesis within the tested bound. Horizon is a contributing
but insufficient factor, and P3 entry dependence is observed. Treat bank 996
at 400 ms as a near-boundary diagnostic, not as a feasible state, trajectory,
or teacher. The next boundary experiment must target the coupled entry-axis
problem without relaxing the strict gate or prescribing a contact sequence.

## O1.12 Current Finding

O1.12 audits P3 bank 996 and the three nearest independent-parent/source O1.9
initial-feature entries 1245/1203/1171. It runs a 400 ms handoff CEM followed
in the same PhysX episode by a 200 ms post-handoff stabilization CEM. Dynamic
states and CEM plans remain in memory and are not serialized as reset data or
teachers.

The handoff search finds existence candidates for 3/4 entries. Continuous
post-stage replay also finds a transition candidate for 3/4 entries, but only
bank 996 passes the strict post gate. Its selected plan reaches axis alignment
0.852, zero measured height loss, 100% TIM overlap, 0.305-3.479 N total contact
force, and no ballistic/contact-loss/penetration/drop/throw event. Across 128
clone replays, its transition rate is 73.4% and strict post success is 71.9%.

Banks 1245 and 1171 preserve height and support after an occasional transition
but remain axis-misaligned at about 0.356 and 0.347. Bank 1203 produces a
handoff candidate in the first-stage search but does not reproduce it in the
post-stage selected-plan replay. The formal taxonomy is one success, two
axis-misaligned entries, and one transition-not-reproduced entry.

Therefore do not claim that post-handoff axis stabilization is dynamically
unreachable. The current bottlenecks are a narrow entry-dependent axis-control
basin and transition reproducibility, not height or support-force collapse.
Treat the bank-996 result as reachability evidence only. Do not export its CEM
plan, call it a teacher, distill it, or resume PPO scaling from this result.

## O1.13 Current Finding

O1.13 compares feasible prototype 996, axis-failure prototypes 1245/1171, and
transition-not-reproduced prototype 1203 using 66 O1.9 initial features plus
the 20-dimensional joint configuration. Geometry, wrench, support/contact
margin, pen pose, velocity, and joint configuration are group-balanced so the
31 wrench dimensions cannot dominate by count.

The resulting readiness score is an uncalibrated prototype margin, not a
classifier probability. Twelve high-score P3 entries are selected with no
prototype parent/source leakage and no parent/source duplication inside the
selection. A matched 400 ms, 20-D, +/-0.18 rad, population-64 x 10 CEM audit
compares them against the original O1.8 12-entry spread sample.

The readiness-selected distribution produces 0/12 raw TIM transitions, 0/12
admissible transitions, and 0/12 strict handoffs. The original spread produces
1/12 raw TIM transitions and 0/12 admissible or strict handoffs. Its only raw
transition is bank 996, which reaches 8.223 N versus the 6 N force ceiling and
axis alignment 0.220 versus the 0.5 gate. Therefore the static readiness score
does not expand the handoff-ready basin.

Do not filter, replace, relabel, or modify frozen P3 from this score. Preserve
it only as an analysis hypothesis. Any future handoff-ready overlay must use a
materially larger provenance-isolated set of dynamic CEM labels and include
action-conditioned 50-100 ms response features such as axis response,
support-force slope, and height acceleration. Transition reachability and
post-axis stabilization must remain separate labels, and held-out parent/source
plus clone-replay validation must beat the spread baseline before promotion.

## O1.14 Current Finding

O1.14 applies the same 31 deterministic probes to all 396 frozen P3/train
states. The probe set is one zero action plus 15 antithetic pairs of bounded
TIM target offsets at +/-0.06 rad. The audit uses the existing two-step PhysX
contact reconstruction and records matched 50 ms and 100 ms axis response,
support-force response, height acceleration, contact transitions, and safety
events. It does not optimize or serialize a successful action plan.

At 100 ms, bank 996 is the only one of the four O1.12 anchors with a nonzero
action passing the pre-specified short-window joint gate: 1/30 nonzero probes.
Banks 1245, 1171, and 1203 each have 0/30. Across the full P3/train set,
however, 37/396 states pass at least one nonzero probe and no state passes more
than two. Bank 996 is tied behind four two-pass states. The continuous dynamic
response score also does not separate all anchors: bank 1245 scores slightly
higher than 996 despite its known long-horizon axis failure.

Therefore action-conditioned response is more informative than the rejected
static O1.13 prototype score, but it is not sufficient evidence of 400-600 ms
handoff reachability or post-handoff stabilization. Preserve the 37 rows only
as an analysis candidate layer. Do not modify O0, train PPO from this layer,
call it handoff-ready, or promote a readiness overlay until provenance-isolated
long-horizon CEM labels and clone replay validate it against a matched spread
baseline. Keep transition reachability and post-axis stabilization as separate
labels.

## O1.15 Current Finding

O1.15 audits all 37 O1.14 dynamic candidates with matched 300/400/500 ms,
20-dimensional, +/-0.18 rad, three-waypoint CEM at population 64 x 10
iterations. Runs are split into GPU-safe batches but retain identical search
budget, seed, strict gate, and frozen O0 hash. Best action plans are not
serialized. Terminal height recovery is reported separately using a final
height-loss threshold of 2 mm.

Three independent states show recovered strict existence: bank 894 at 400 ms,
bank 1155 at 300 ms, and bank 831 at 500 ms. Their recovered-strict clone
rates are 10.94%, 1.56%, and 1.56%. Thus conditional existence is 3/37 (8.11%,
Wilson 95% interval about 2.80-21.30%), but the robust basin at clone replay
rate >=50% is 0/37. The three existence states come from three independent
parents and three independent source trajectories.

The best-horizon taxonomy is 17 no-TIM-transition, 10 contact-force overload,
7 axis-misaligned after transition, and 3 recovered but poorly reproducible.
Bank 996 does not reproduce strict feasibility under this one-stage 64 x 10
search and is classified as contact-force overload. This does not withdraw
the O1.12 result, which used a larger 128 x 20 two-stage handoff plus post-axis
stabilization search; it demonstrates search-budget and parameterization
sensitivity.

Interpret the measured basin size as conditional on the O1.14 37-state layer,
not as the measure of all frozen P3. Do not train PPO, modify O0, or generate
teachers from banks 894/1155/831. Their next admissible use is a matched
reproducibility and boundary audit that preserves parent/source isolation and
separately tests overload, axis alignment, and transition occurrence.

## O1.16 Current Finding

O1.16 independently re-finds in-memory CEM plans around O1.15 banks
1155/831/894 and replays each fixed plan under structured reset-state
perturbations. The audit covers joint position and velocity, pen position and
rotation, pen linear and angular velocity, controller target context, and
joint multi-factor perturbations. Contact forces are reconstructed by PhysX
after two settling steps rather than written directly. Plans and perturbed
states are not serialized.

Bank 1155 is the only measured narrow local open-loop basin: nominal replay is
34.38%, joint L1/L2/L3 recovered-strict rates are 10.16%/5.47%/3.12%. Its most
sensitive tested factors are hand joint position and pen position; velocity
perturbations are materially better tolerated. Bank 831 is approximately an
isolated trajectory solution: nominal replay is 6.25% and joint L1 is 0.78%.
Bank 894 has zero nominal recovery in this independent CEM run, so its basin is
not measured here. Preserve its O1.15 conditional-existence evidence; do not
reinterpret O1.16 as dynamics unreachability or as withdrawal of that result.

The O1.16 rates are a lower bound for a fixed open-loop plan, not the volume of
a closed-loop policy attractor. Bank 1155 is therefore a research seed for
future closed-loop basin widening, not a validated reset distribution,
teacher, or PPO curriculum. Do not modify O0, train PPO, export plans, or
generate teacher data from O1.16 artifacts without a separately approved next
phase.

## O1.17 Current Finding

O1.17 uses bank 1155 at the frozen 300 ms horizon and independently re-finds
the CEM plan in memory. Three deterministic high-level feedback candidates are
pre-registered. They track an in-memory nominal q/dq trajectory; one candidate
also adjusts tracking gain continuously from total support force and pen
height. No candidate prescribes a finger contact identity or sequence.
Calibration and held-out perturbation seeds are disjoint, while open-loop and
closed-loop modes receive matched perturbations within each split.

The calibration safety rule selects `support_adaptive`. On held-out Joint-L1,
open-loop recovered-strict rate is 19.53% with 1.56% unsafe, while the selected
feedback reaches only 3.91% with 10.94% unsafe. Joint-L2 changes from 7.81% to
0.78%. Nominal replay improves slightly from 26.56% to 29.69%, showing that
the controller can track the reference but does not widen the perturbation
basin. All four pre-registered widening gates fail.

Therefore the tested single-trajectory q/dq tracking plus support/height gain
adaptation is rejected as a basin-widening mechanism. This is not evidence
that every closed-loop method must fail. It specifically indicates that
forcing perturbed states back toward one brittle nominal joint trajectory can
increase throws and axis errors. Any later audit should test a genuinely
object-state-conditioned local controller or constrained receding-horizon
method with a new explicit phase, matched held-out perturbations, and the same
strict safety gates. Do not tune these tracking gains further, train PPO,
modify O0/reward, or serialize the CEM plan/reference as a teacher.

## O1.18 Current Finding

O1.18 compares matched A/B/C modes on bank 1155 at 300 ms: A is the nominal
open-loop CEM plan, B is the O1.17 support-adaptive q/dq tracker, and C is a
bounded task-space residual controller. C consumes pen height, pen angular
velocity, aggregate support force/torque on the pen, and aggregate contact
count. It does not target individual finger identities or a prescribed contact
sequence.

The task-space mapping is identified independently at the three plan stages
with +/-0.01 rad antithetic one-step joint-target probes. The resulting 11x20
Jacobians each have effective rank 10/11 and effective condition numbers about
14156, 3491, and 618. The Jacobians, CEM plan, and nominal reference remain in
memory and are not serialized. Calibration and held-out perturbation seeds are
disjoint, with matched perturbations across A/B/C inside each split.

No task-space candidate passes the calibration nominal-unsafe tolerance. The
soft DLS candidate is retained only as a diagnostic fallback. On held-out
Joint-L1, A/B/C recovered-strict rates are 15.62%/5.47%/8.59%, while unsafe
rates are 1.56%/7.03%/10.16%. On Joint-L2, recovered-strict rates are
6.25%/1.56%/4.69%. Thus C improves over B but remains worse than A and fails
four of five pre-registered widening gates. Final height recovery is 100% in
all tested modes; p10 downward excursion stays below 0.35 mm, so height collapse
is not the discriminating failure in this audit.

Therefore reject the tested one-step, stage-local DLS task controller as a
basin-widening mechanism. The result points to an ill-conditioned and
state-dependent local dynamics map, plus transition/axis/safety failures. Do
not tune DLS gains further, serialize the probes/Jacobians, train PPO, modify
O0/reward, or create teachers from O1.18. A later phase would need a separately
approved multi-step constrained model or receding-horizon audit with the same
held-out safety contract.

## O1.19 Current Finding

O1.19 compares matched A/B/C modes on bank 1155 at 300 ms: A is open-loop, B
is the O1.18 soft task-space DLS fallback, and C is reduced-order receding-
horizon random-shooting MPC. C evaluates 64 candidate residual sequences at
every control step, executes only the first residual, and replans. The tested
horizons are 3 and 6 control steps, corresponding to 50 and 100 ms. Only the
12 index/middle/thumb joints are active, with residuals bounded to 0.025 rad.

Neither MPC horizon passes the calibration nominal-unsafe tolerance. Both
produce 2.34% Joint-L1 recovered-strict calibration rate; unsafe is 11.72% at
50 ms and 12.50% at 100 ms, so 50 ms is retained only as the less unsafe
diagnostic fallback. On held-out Joint-L1, A/B/C recovered-strict rates are
10.94%/3.12%/1.56%, with unsafe rates 3.91%/7.03%/14.06%. On Joint-L2 the
rates are 5.47%/2.34%/3.12%. All five widening gates fail. MPC p10 downward
height excursion remains below 0.41 mm, so height collapse is again not the
primary discriminator.

The dominant MPC failures are no TIM transition, axis misalignment, and
throw. Therefore the tested reduced-order task model does not predict the
hybrid contact dynamics accurately enough for constrained shooting to improve
real PhysX outcomes. Do not increase candidate count, tune MPC costs/horizon,
serialize optimized sequences, train PPO, modify O0/reward, or generate
teachers from O1.19 without a separately approved model-validity phase. The
next admissible research question is model prediction error across contact
switches, not a larger MPC search budget.

## O1.20 Current Finding

O1.20 audits whether frozen, reset-valid P3 states contain a distinct
handoff-entry semantic layer. It reconciles the corrected O1.8 result (0/12
strict feasible) with the later O1.15 conditional positives: banks 831, 894,
and 1155 are the only strict positive labels. These three positives have
independent parent and source trajectory provenance. The 37-state O1.15 layer
contains 35 unique parents and 33 unique sources, so validation leaves the
test row's complete parent and source groups out of both class prototypes.

Static initial-state features separate the three positives from the 34
failures under this provenance holdout (AUC 0.980, permutation p=0.004).
Short-action response features alone do not (AUC 0.539, p=0.294). The combined
score passes the pre-registered confirmation gate (AUC 0.892, p=0.021), but
only three sparse positives exist and none is a robust handoff basin. The
result supports a P3 skill-entry hypothesis, not a deployable classifier or a
validated P3-H distribution.

The 20 provenance-disjoint ranked candidates in the O1.20 report are
quarantined analysis rows. Do not use them as reset filters, teachers, PPO
curricula, or modifications to O0/P3. Their only admissible next use is an
independent confirmatory audit under the frozen O1.15 long-horizon search
contract. Require additional independent strict positives and clone
robustness before defining P3-H. Do not interpret static feature contrasts as
causal control rules.

## O1.21 Current Finding

O1.21 measures static skill-entry coverage across all 396 frozen P3 rows. Only
37 rows have long-horizon labels; the remaining 359 rows are unknown and must
not be counted as non-feasible. Three strict positives are known, corresponding
to 0.76% of all P3 and 8.11% of the O1.14 conditional candidate layer.

Using the O1.20 static prototype margin, the zero-known-failure core contains
9/396 rows (2.27%) and recalls two of three strict positives. The envelope that
recalls all three positives contains 11/396 rows (2.78%), including seven
unmeasured candidates and one measured failure. A broader band allowing about
5% measured-failure selection contains 46/396 rows (11.62%). These are feature-
space densities, not physical feasibility rates.

The measured skill-entry region is sparse but nonzero and provenance-diverse.
The strongest descriptive differences lie in pen pose, fingertip geometry,
initial wrench, support margin, and joint configuration. They are not causal
rules. O1.21 therefore supports both insufficient high-confidence P3 coverage
and a continuing policy-exploration bottleneck. It does not yet support
canonical resampling: seven provenance-independent candidates from the O1.20
quarantine ranking remain available for confirmation.

Do not modify O0, create P3-H, train PPO, generate teachers, or use any O1.21
score as a reset filter. The only admissible next use of banks 1034, 1176,
1146, 1045, 1227, 852, and 1149 is independent long-horizon confirmation under
the frozen O1.15 contract. A later canonical-resampling proposal requires that
confirmation to show the existing candidate pool is insufficient.

## O1.22 Current Finding

O1.22 is the explicitly authorized exception to the O1.21 no-PPO rule. It
compares matched P3-only temporal-observation PPO runs with seed 20260730, 256
environments, 60 iterations, 24-step rollouts, and 368,640 transitions. Reward,
observation, PPO configuration, and frozen O0 bank hash are identical. A uses
uniform P3 resets. B samples seven provenance-independent O1.21 candidates at
75% for iterations 1-20, linearly decays to their natural P3 frequency during
iterations 20-40, and uses exact uniform P3 during iterations 41-60. No state
or label is added to O0. Both final checkpoints are evaluated on the same
uniform P3 task rather than B's biased training sampler.

The curriculum is rejected. Qualified gait, strict support handoff, and height
recovery remain zero in both runs. TIM transition activity falls from 56.47%
under A to 12.55% under B. Primary contact-loss/ballistic/throw exploitation
rises from 0.63% to 99.70%; B contact-loss exposure is 98.80%. A is not a skill
solution either: nearly every episode loses height and its overall failure
rate is 100%, mostly from drop rather than immediate contact-loss exploitation.

Static O1.21 entry-likeness is therefore not a sufficient PPO curriculum
variable under the frozen O1.5 objective. Do not increase the sampling weight,
extend the budget, distill B, or use its checkpoint as a skill policy. This
rejects the tested seven-row, 75%-then-decay schedule; it does not establish
that every state-distribution curriculum must fail. O0 remains frozen and no
P3-H or teacher is authorized by O1.22.

## O1.23 Current Finding

O1.23 changes the unit of analysis from P3 state identity to the dynamic
transition phase. It compares the O1.22 A/B checkpoints on the same ordered 256
P3 bank rows for 120 control steps with ordinary failure termination disabled.
The diagnostic trajectories are analysis-only artifacts and must not be used as
teachers, reset banks, or policy pretraining data.

The cumulative A-uniform transition funnel is 256 TIM releases, 173 same-finger
recontacts, 116 transitions satisfying the short-window TIM overlap/zero-contact
gate, and zero height recoveries. Its dominant first unmet gate is therefore
post-recontact height recovery, not release discovery. The B static-state
curriculum reduces the same funnel to 256, 55, 16, and zero; it moves failure
earlier rather than widening the transition skill.

Do not search for a better P3, increase static candidate weighting, create P3-H,
modify O0, generate teachers, or start skill pretraining/hierarchical control.
The user-confirmed V9 semantic correction supersedes the initial O1.23
curriculum decision. The visualized motion is the intended dynamic P4 process:
index and middle rotate the pen around the thumb. When the pen end moves from
the thumb-inside region into the thumb-index side, control must switch to P5,
where thumb and index pinch the pen. The observed policy instead continues the
index-middle drive, misses the P5 capture, and appears to use penetration to
force the pen around the thumb.

Therefore O1.23 does not authorize transition curriculum yet. Generic
same-finger recontact, TIM overlap, and height recovery are diagnostic values,
not V9 P5 success. The next admissible task is a read-only P4-to-P5 capture
window and penetration audit: locate pen-end passage relative to thumb/index,
measure mesh penetration, and determine whether thumb-index pinch force is
established before index-middle drive continues. Do not change reward, train
PPO, modify O0, or generate a teacher before this audit.

## S0 Quaternion Replay Gate

Powered replay selection must use the recorded pen quaternion, not integrated
angular velocity alone. A trajectory is eligible for speed-tier generation only
when both its maximum and final quaternion geodesic displacement from the first
frame are at least 30 degrees. The 2026-08-01 audit of all 1200 S0 rollouts found
zero eligible trajectories; the best was episode 1188 at 25.043 degrees despite
39.516 degrees of integrated world-z angular velocity. Therefore the existing
dataset must not be used to generate or execute the requested 0.12, 0.18, 0.24,
or 0.36 rad/s powered replay packages. Re-open this gate only after a corrected
simulation rollout records at least 30 degrees of actual quaternion change.

## S0.9.2 Current Finding

The read-only S0.9.2 validity audit classifies the 1200 historical oracle
"success" rollouts as 0 Tier A transfer candidates, 79 Tier B partial 15-30
degree motions, and 1121 Tier C false rotations. Student training, speed-tier
generation, and powered replay remain blocked. The dataset also fails a basic
pose/velocity consistency audit: reported world-z angular velocity and angular
velocity inferred from adjacent `root_quat_w` samples have mean episode
correlation -0.137, while quaternion-derived angular path is only 0.332 of the
reported path on average. Before changing reward or horizon, fix and validate
the rollout sampling/state contract. New data must pass pose/velocity
kinematic consistency before quaternion task gates are evaluated.

## S0.9.3 Current Finding

S0.9.3 records explicit before/action/after transitions, excludes automatic
reset boundaries, and captures both 60 Hz control states and 120 Hz PhysX
substeps. The corrected S0 success gate uses relative `root_quat_w` target-axis
rotation, axis alignment >=0.70, and the existing physical 150 ms hold gates.
The old checkpoint produced 0/32 corrected successes.

Instantaneous `root_ang_vel_w` is not finite-step motion evidence under the
current contact dynamics. In the synchronized smoke, quaternion/velocity
omega-z correlation is 0.116 at 60 Hz and 0.554 at 120 Hz; angular-path ratios
are 0.130 and 0.222. Keep that field only for impact/contact diagnostics. New
datasets must store `pen_angular_velocity_pose_delta`, derived from adjacent
quaternions, as the finite-step motion channel. The 32-episode smoke is
quarantined (`dataset_authorized=false`). Do not train a student, generate
powered-replay packages, or collect a large dataset until a policy reaches the
corrected quaternion success gate.

## S0.9.4 Authorized Exception

S0.9.4 explicitly authorizes one bounded, single-seed PPO smoke for the S0
short-rotation oracle. It may contain the initial 30-iteration run and one
same-seed 30-iteration continuation after correcting strict-hold temporal
credit; it must not add a second seed or scale the environment count. O0,
teacher generation, student training, and powered hardware replay remain
frozen. Rotation phase, progress, target error, braking,
and success must all use relative-quaternion finite-step motion; instantaneous
`root_ang_vel_w` is diagnostic only. The first acceptance gate is at least one
trajectory with actual target-axis quaternion rotation >=30 degrees, axis
alignment >=0.70, and the existing 150 ms strict hold. Do not scale training or
collect a large dataset before this gate passes.

S0.9.4 first-trajectory gate is now PASS. The bounded same-seed run produced
369/512 environment successes and 362/512 successes independently reconstructed
from synchronized quaternion/contact records. The accepted checkpoint is
`2026-08-01_23-41-07_ppo_torch/checkpoints/agent_720.pt`. This does not authorize
student training or powered replay: the full rollout remains quarantined because
instantaneous PhysX angular velocity fails the finite-step consistency thresholds.
Fix terminal-state recording and keep pose-delta angular velocity as the motion
channel before any expanded collection.

The terminal-state/data closure is now PASS. The pre-reset terminal snapshot
made environment and offline success counts agree exactly (49/64 regression),
and the authorized expanded dataset contains 1200/1200 independently verified
quaternion successes with zero split leakage. `pen_angular_velocity` and
`pen_angular_velocity_pose_delta` are identical canonical finite-step fields;
`pen_angular_velocity_physx_instantaneous` is diagnostic only. Do not let a
downstream consumer silently fall back to the instantaneous field. Student
training and powered replay remain separate, explicitly authorized phases.

## S0.10 Student Data Contract

S0.10 authorizes offline student dataset construction and BC interface smoke
only. Do not modify the oracle, O0, or the quaternion success definition, and
do not issue hardware commands. The student dataset may contain only the
deployable 340D observation history, bounded 20D virtual action target, matching
next observation, terminal/success labels, and provenance/split metadata. Split
by `source_trajectory_id`; row-level random splitting is forbidden. Any terminal
`next_obs` must come from a pre-reset snapshot, never from the next episode's
reset state. BC validation is limited to dataset loading, model forward, and
loss computation with no optimizer step and no trained checkpoint.

S0.10 is now PASS at the interface-smoke level. The authorized v2 source has
1200 quaternion-validated episodes and produces 69,600 transitions. The final
split contains 891/112/197 train/validation/test episodes with zero
`source_trajectory_id` leakage. Independent validation passes the deployable
field allowlist, terminal pre-reset `next_obs`, action-history, normalization,
and 20D adapter checks. The BC smoke performs forward and MSE computation only;
it creates no optimizer, calls no backward pass, and writes no checkpoint.
Formal BC training remains unauthorized.

## S0.11 Student BC Training

S0.11 authorizes training the first non-privileged Dex5 student from the frozen
S0.10 dataset. The oracle, O0, dataset bytes, quaternion success definition,
and hardware interface must remain unchanged. Training may read only the 340D
student observation and bounded 20D virtual action. Run a 100-episode overfit
gate before full BC on the 891 train episodes. Closed-loop simulation must use
the 197 source-isolated test episodes and the existing hardware-compatible
action wrapper. No hardware commands are authorized. Environment edits are
limited to read-only terminal diagnostic snapshots and must not affect reward,
termination, reset distributions, or success.

S0.11 is now PASS for BC training and simulation validation. The 100-episode
overfit MSE fell from 0.06680 to 0.000649. Full BC used all 891 train episodes;
the selected checkpoint has validation MSE 0.000780 and offline test MSE
0.000654. On the frozen 197-episode test split under the hardware-compatible
Medium DR wrapper, the non-privileged student achieved 152/197 strict successes,
184/197 episodes with at least 30 degrees quaternion rotation, and zero
penetration episodes above 0.5 mm. This authorizes no hardware deployment.

S0.11 is the simulation-pretrained student baseline, not a deployable or
real-validated student. Do not discard or overwrite it when real data becomes
available; later reports must compare the frozen sim-only checkpoint against a
separately versioned sim-plus-real adapted checkpoint.

## S0.12 Real Replay Dataset Validation

The next sim-to-real gate is controlled replay of selected oracle action
trajectories, not autonomous execution of the S0.11 student. S0.12 must first
rank the 1200 quaternion-valid simulation trajectories using actual quaternion
rotation, terminal hold, action smoothness, contact margin, and penetration.
Real success must be measured from an external pen-orientation signal or a
validated video pose annotation; joint motion, commanded motion, and integrated
angular velocity alone are not valid evidence of real pen rotation.

S0.12 hardware execution requires a separately armed replay command and keeps
the existing adapter, watchdog, delta bounds, thermal monitoring, soft pad, and
physical emergency stop. Speed tiers must be accepted progressively; a higher
tier cannot run before the lower tier passes tracking, temperature, contact,
and object-retention gates. S0.12 may collect real replay observations/actions
and outcome labels, but it must not fine-tune the student. S0.13 is the separate
phase authorized to adapt the frozen S0.11 baseline using source-isolated real
successful trajectories. S0.14 is reserved for controlled real closed-loop
student evaluation.

S0.12 offline selection is now PASS. The read-only audit found 885 eligible
rollouts across 149 unique sources after jointly requiring actual peak/final
quaternion rotation >=30 degrees, axis alignment >=0.70, strict hold >=150 ms,
TIM contact ratio >=75%, height loss <=5 mm, mapped target delta <=0.01 rad,
and reset baseline penetration <=0.5 mm. The first replay batch contains 20
trajectories from 20 distinct train sources. Their peak rotations are
34.46-35.33 degrees and mapped target deltas are 0.00527-0.00864 rad. Powered
replay and real pen-pose validation have not run. Start with a small subset at
0.12 rad/s; do not promote to 0.18 or 0.24 rad/s before the preceding tier is
accepted.
