from __future__ import annotations

import argparse
import os
import runpy
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CHECKPOINT = (
    ROOT
    / "reports/m3_2_30_index_near_surface/isaac_recapture_p3_physics/p2_p3_best_agent.pt"
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Play a trained Dex5 + pen PhysX policy.")
    parser.add_argument("--task", default="Dex5-P3Entry-Direct-v0")
    parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    parser.add_argument("--num_envs", type=int, default=1)
    parser.add_argument("--no-real-time", action="store_true")
    args, passthrough = parser.parse_known_args()

    import dex5_penspin.isaac.dex5_penspin_isaaclab  # noqa: F401

    isaaclab_root = Path(
        os.environ.get("ISAACLAB_ROOT", ROOT / ".deps/IsaacLab")
    ).resolve()
    play_script = isaaclab_root / "scripts/reinforcement_learning/skrl/play.py"
    sys.argv = [
        str(play_script),
        "--task",
        args.task,
        "--num_envs",
        str(args.num_envs),
        "--checkpoint",
        args.checkpoint,
    ]
    if not args.no_real_time:
        sys.argv.append("--real-time")
    sys.argv.extend(passthrough)
    runpy.run_path(str(play_script), run_name="__main__")


if __name__ == "__main__":
    main()
