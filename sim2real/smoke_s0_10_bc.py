from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn


class Dex5StudentPolicy(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(340, 256),
            nn.ELU(),
            nn.Linear(256, 256),
            nn.ELU(),
            nn.Linear(256, 20),
            nn.Tanh(),
        )

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        return self.network(observation)


def _batch_loss(
    model: nn.Module,
    obs: np.ndarray,
    action: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    indices: np.ndarray,
) -> tuple[torch.Tensor, torch.Tensor]:
    normalized = (obs[indices] - mean) / std
    input_tensor = torch.from_numpy(normalized.astype(np.float32))
    target_tensor = torch.from_numpy(action[indices].astype(np.float32))
    with torch.no_grad():
        prediction = model(input_tensor)
        loss = nn.functional.mse_loss(prediction, target_tensor)
    return prediction, loss


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=256)
    args = parser.parse_args()

    torch.manual_seed(20260802)
    rng = np.random.default_rng(20260802)
    with np.load(args.dataset, allow_pickle=False) as data:
        obs = np.asarray(data["student_obs"], dtype=np.float32)
        action = np.asarray(data["action"], dtype=np.float32)
        split = np.asarray(data["split_id"])[:, 0]
        mean = np.asarray(data["observation_mean_train"], dtype=np.float32)
        std = np.asarray(data["observation_std_train"], dtype=np.float32)

    train_rows = np.flatnonzero(split == 0)
    validation_rows = np.flatnonzero(split == 1)
    train_indices = rng.choice(train_rows, size=args.batch_size, replace=False)
    validation_indices = rng.choice(
        validation_rows, size=args.batch_size, replace=False
    )

    model = Dex5StudentPolicy().eval()
    train_prediction, train_loss = _batch_loss(
        model, obs, action, mean, std, train_indices
    )
    validation_prediction, validation_loss = _batch_loss(
        model, obs, action, mean, std, validation_indices
    )
    checks = {
        "train_input_shape": [args.batch_size, 340],
        "train_output_shape": list(train_prediction.shape),
        "validation_output_shape": list(validation_prediction.shape),
        "train_output_finite": bool(torch.isfinite(train_prediction).all()),
        "validation_output_finite": bool(torch.isfinite(validation_prediction).all()),
        "train_loss_finite": bool(torch.isfinite(train_loss)),
        "validation_loss_finite": bool(torch.isfinite(validation_loss)),
        "output_bounded": bool(
            torch.max(torch.abs(train_prediction)) <= 1.0
            and torch.max(torch.abs(validation_prediction)) <= 1.0
        ),
    }
    passed = (
        checks["train_output_shape"] == [args.batch_size, 20]
        and checks["validation_output_shape"] == [args.batch_size, 20]
        and all(value for key, value in checks.items() if isinstance(value, bool))
    )
    report = {
        "schema": "dex5-s0.10-bc-interface-smoke-v1",
        "status": "PASS" if passed else "FAIL",
        "dataset": str(args.dataset.resolve()),
        "seed": 20260802,
        "model": {
            "input_dimension": 340,
            "hidden_dimensions": [256, 256],
            "output_dimension": 20,
            "output_activation": "tanh",
            "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        },
        "batch_size": args.batch_size,
        "train_mse_smoke": float(train_loss),
        "validation_mse_smoke": float(validation_loss),
        "checks": checks,
        "backward_called": False,
        "optimizer_created": False,
        "optimizer_step_performed": False,
        "checkpoint_written": False,
        "training_status": "INTERFACE_SMOKE_ONLY",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
