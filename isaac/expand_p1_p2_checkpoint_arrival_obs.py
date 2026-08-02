from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Expand a P1/P2 skrl checkpoint from 164 to 172 observations."
    )
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--old-observations", type=int, default=164)
    parser.add_argument("--new-observations", type=int, default=172)
    args = parser.parse_args()

    source = Path(args.input).resolve()
    output = Path(args.output).resolve()
    checkpoint = torch.load(source, map_location="cpu", weights_only=False)

    old_observations = args.old_observations
    new_observations = args.new_observations
    if new_observations <= old_observations:
        raise ValueError("--new-observations must exceed --old-observations")
    first_weight_key = "net_container.0.weight"
    for model_name in ("policy", "value"):
        weight = checkpoint[model_name][first_weight_key]
        if tuple(weight.shape) != (256, old_observations):
            raise ValueError(
                f"Unexpected {model_name} first-layer shape: {tuple(weight.shape)}"
            )
        expanded = weight.new_zeros((weight.shape[0], new_observations))
        expanded[:, :old_observations] = weight
        checkpoint[model_name][first_weight_key] = expanded

    preprocessor = checkpoint["observation_preprocessor"]
    for key, fill in (("running_mean", 0.0), ("running_variance", 1.0)):
        values = preprocessor[key]
        if tuple(values.shape) != (old_observations,):
            raise ValueError(f"Unexpected preprocessor {key} shape: {tuple(values.shape)}")
        expanded = values.new_full((new_observations,), fill)
        expanded[:old_observations] = values
        preprocessor[key] = expanded

    checkpoint.pop("optimizer", None)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, output)
    result = {
        "input": str(source),
        "input_sha256": sha256(source),
        "output": str(output),
        "output_sha256": sha256(output),
        "old_observations": old_observations,
        "new_observations": new_observations,
        "optimizer_preserved": False,
        "new_input_weights": "zero_initialized",
    }
    print("P1_P2_CHECKPOINT_EXPANSION=" + json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
