from __future__ import annotations

import hashlib
from pathlib import Path

import torch
from torch import nn


OBSERVATION_DIMENSION = 340
ACTION_DIMENSION = 20
HIDDEN_DIMENSIONS = (256, 256)
NORMALIZED_OBSERVATION_CLIP = 10.0


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class Dex5StudentPolicy(nn.Module):
    def __init__(self, observation_mean: torch.Tensor, observation_std: torch.Tensor) -> None:
        super().__init__()
        if observation_mean.shape != (OBSERVATION_DIMENSION,):
            raise ValueError("student observation mean must be 340D")
        if observation_std.shape != (OBSERVATION_DIMENSION,):
            raise ValueError("student observation std must be 340D")
        self.register_buffer("observation_mean", observation_mean.float().clone())
        self.register_buffer("observation_std", observation_std.float().clone())
        self.network = nn.Sequential(
            nn.Linear(OBSERVATION_DIMENSION, HIDDEN_DIMENSIONS[0]),
            nn.ELU(),
            nn.Linear(HIDDEN_DIMENSIONS[0], HIDDEN_DIMENSIONS[1]),
            nn.ELU(),
            nn.Linear(HIDDEN_DIMENSIONS[1], ACTION_DIMENSION),
            nn.Tanh(),
        )

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        normalized = (observation - self.observation_mean) / self.observation_std
        normalized = normalized.clamp(
            -NORMALIZED_OBSERVATION_CLIP, NORMALIZED_OBSERVATION_CLIP
        )
        return self.network(normalized)


def checkpoint_payload(
    model: Dex5StudentPolicy,
    *,
    dataset_path: Path,
    dataset_sha256: str,
    seed: int,
    epoch: int,
    validation_mse: float,
) -> dict:
    return {
        "schema": "dex5-s0.11-student-checkpoint-v1",
        "model_state_dict": model.state_dict(),
        "architecture": {
            "observation_dimension": OBSERVATION_DIMENSION,
            "hidden_dimensions": list(HIDDEN_DIMENSIONS),
            "action_dimension": ACTION_DIMENSION,
            "activation": "ELU",
            "output_activation": "tanh",
            "normalized_observation_clip": NORMALIZED_OBSERVATION_CLIP,
        },
        "observation_contract": "dex5-s0-student-observation-v1",
        "action_contract": "bounded 20D virtual normalized delta action",
        "dataset_path": str(dataset_path.resolve()),
        "dataset_sha256": dataset_sha256,
        "seed": seed,
        "epoch": epoch,
        "validation_mse": validation_mse,
        "privileged_observation_used": False,
    }


def load_student_checkpoint(
    checkpoint_path: Path, device: torch.device
) -> tuple[Dex5StudentPolicy, dict]:
    payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if payload.get("schema") != "dex5-s0.11-student-checkpoint-v1":
        raise RuntimeError("unexpected student checkpoint schema")
    state = payload["model_state_dict"]
    model = Dex5StudentPolicy(
        state["observation_mean"].to(device),
        state["observation_std"].to(device),
    ).to(device)
    model.load_state_dict(state)
    model.eval()
    return model, payload
