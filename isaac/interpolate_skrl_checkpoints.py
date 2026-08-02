from __future__ import annotations

import argparse
from pathlib import Path

import torch


parser = argparse.ArgumentParser(description="Linearly interpolate compatible skrl checkpoints.")
parser.add_argument("--base", type=Path, required=True)
parser.add_argument("--target", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--alpha", type=float, required=True)
args = parser.parse_args()

base = torch.load(args.base, map_location="cpu", weights_only=False)
target = torch.load(args.target, map_location="cpu", weights_only=False)

for section in ("policy", "value", "observation_preprocessor", "value_preprocessor"):
    if section not in base or section not in target:
        continue
    for key, base_value in base[section].items():
        target_value = target[section].get(key)
        if (
            isinstance(base_value, torch.Tensor)
            and isinstance(target_value, torch.Tensor)
            and base_value.shape == target_value.shape
            and torch.is_floating_point(base_value)
        ):
            base[section][key] = torch.lerp(
                base_value, target_value.to(dtype=base_value.dtype), args.alpha
            )

base.pop("optimizer", None)
args.output.parent.mkdir(parents=True, exist_ok=True)
torch.save(base, args.output)
print(args.output)
