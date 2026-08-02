# O1.23 minimal reproduction assets

This directory contains only the assets needed to reproduce the O1.23
uniform-reset versus entry-curriculum diagnostic. Binary checkpoints and
rollouts are tracked by Git LFS. `manifest.sha256` verifies their contents.

Run `./scripts/reproduce.sh offline` for the deterministic offline audit, or
`./scripts/reproduce.sh rollout` to regenerate rollouts with Isaac Lab.
