# Dex5 S0.12 Offline Replay Candidate Audit

## Result

```text
Offline candidate quantity: PASS
Offline candidate quality: PASS
Powered real replay: NOT RUN
Real pen-pose measurement: REQUIRED
```

- Audited rollouts: `1200`
- Eligible rollouts: `885`
- Eligible unique sources: `149`
- Selected first batch: `20` trajectories from `20` distinct train sources

## Selected batch envelope

- Peak quaternion rotation: `34.46-35.33 deg`
- Final quaternion rotation: `34.46-35.33 deg`
- TIM contact ratio: `81.5-100.0%`
- Maximum mapped target delta: `0.00527-0.00864 rad`
- Strict hold: every selected trajectory is at least `150 ms`
- Source leakage inside the selected batch: `0`

The selected trajectories are sufficient for a first controlled replay batch. Start at `0.12 rad/s` and promote to `0.18` and `0.24 rad/s` only after the preceding tier passes. Actual pen orientation measurement remains mandatory; this audit does not authorize blind replay or student deployment.
