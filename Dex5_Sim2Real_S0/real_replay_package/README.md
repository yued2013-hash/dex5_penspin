# Dex5 S0 Open-Loop Replay Package

Status: `OFFLINE_ONLY_NOT_HARDWARE_APPROVED`.

This package contains selected successful simulation trajectories resampled to
100 Hz as 20D absolute URDF-joint targets. It does not contain or deploy the
privileged oracle policy. The 16D logical actuator coordinates used in the
simulation audit are not motor-shaft commands and are intentionally absent.

Run the offline integrity check with:

```bash
python dry_run_replay.py
```

Do not connect this package to the powered hand until joint directions,
PIP/DIP coupling, command latency, PD response, limit behavior, and emergency
stop behavior have passed the controlled hardware acceptance procedure.
