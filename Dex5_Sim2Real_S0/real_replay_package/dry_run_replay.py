from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Offline-only validation; this script never connects to Dex5."
    )
    parser.add_argument(
        "commands",
        type=Path,
        nargs="?",
        default=Path(__file__).with_name("open_loop_commands_100hz.npz"),
    )
    args = parser.parse_args()
    with np.load(args.commands, allow_pickle=False) as data:
        commands = np.asarray(data["sdk_absolute_target_100hz"])
        virtual = np.asarray(data["virtual_joint_target_100hz"])
        logical = np.asarray(data["logical_actuator_target_100hz"])
        timestamps = np.asarray(data["timestamp_s"])
        offsets = np.asarray(data["trajectory_offsets"])
        hz = float(data["command_hz"])
    if commands.ndim != 2 or commands.shape[1] != 20:
        raise SystemExit("invalid 20D SDK command shape")
    if virtual.shape != commands.shape or logical.shape != (commands.shape[0], 16):
        raise SystemExit("invalid virtual/logical replay shapes")
    if timestamps.shape != (commands.shape[0],):
        raise SystemExit("invalid replay timestamp shape")
    if not np.isfinite(commands).all() or not np.all(np.diff(offsets) > 0):
        raise SystemExit("invalid replay values or offsets")
    print(
        f"OFFLINE PASS: {offsets.size - 1} trajectories, "
        f"{commands.shape[0]} commands at {hz:.1f} Hz"
    )
    print("No SDK connection was attempted. Powered replay remains blocked.")


if __name__ == "__main__":
    main()
