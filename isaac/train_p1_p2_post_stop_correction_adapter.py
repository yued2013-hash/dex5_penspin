from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

from train_p1_p2_ring_only_bc_checkpoint import PolicyNet
from train_p1_p2_terminal_stop_adapter import LEARNED_INDICES, TerminalStopAdapter


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class PostStopCorrectionAdapter(nn.Module):
    def __init__(
        self,
        stop_adapter: TerminalStopAdapter,
        hidden_size: int,
        correction_scale: float,
    ) -> None:
        super().__init__()
        self.stop_adapter = stop_adapter
        self.correction_net = nn.Sequential(
            nn.Linear(12, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, len(LEARNED_INDICES)),
        )
        self.correction_scale = correction_scale

    def forward(
        self, condition: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        residual, stop = self.stop_adapter(condition)
        gate = condition[:, :1] * condition[:, 11:12]
        correction = gate * self.correction_scale * torch.tanh(
            self.correction_net(condition)
        )
        return residual, stop, correction


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train a correction head on top of a frozen terminal stop adapter."
    )
    parser.add_argument("--base-checkpoint", required=True, type=Path)
    parser.add_argument("--stop-adapter", required=True, type=Path)
    parser.add_argument("--v14-data", required=True, type=Path)
    parser.add_argument("--long-hold-teacher", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--metrics", required=True, type=Path)
    parser.add_argument("--steps", type=int, default=6000)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--hidden-size", type=int, default=32)
    parser.add_argument("--correction-scale", type=float, default=0.25)
    parser.add_argument("--lr", type=float, default=3.0e-4)
    parser.add_argument("--stable-zero-weight", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=2821)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    base_checkpoint = args.base_checkpoint.resolve()
    stop_adapter_path = args.stop_adapter.resolve()
    stop_artifact = torch.load(stop_adapter_path, map_location="cpu", weights_only=False)
    if stop_artifact.get("base_checkpoint_sha256") != sha256(base_checkpoint):
        raise ValueError("Stop adapter and base checkpoint SHA256 do not match")

    with np.load(args.long_hold_teacher.resolve(), allow_pickle=False) as data:
        observation = np.asarray(data["observation"], dtype=np.float32)
        action = np.asarray(data["action"], dtype=np.float32)
        target = np.asarray(data["target_condition"], dtype=np.float32)
        delta = np.asarray(data["target_condition_delta"], dtype=np.float32)
        margin = np.asarray(data["semantic_margin"], dtype=np.float32)
        hold = np.asarray(data["hold_latched"], dtype=np.bool_)
        accepted = np.asarray(data["accepted_step"], dtype=np.bool_)
    if observation.shape[-1] != 186 or not accepted.all():
        raise ValueError("Long-hold teacher is not a strict 186-D H6 teacher")
    terminal = np.ones((*hold.shape, 1), dtype=np.float32)
    phase = np.ones_like(terminal)
    conditioned = np.concatenate(
        (
            observation,
            terminal,
            phase,
            target,
            margin[..., None],
            delta,
            hold[..., None].astype(np.float32),
        ),
        axis=-1,
    )
    positive_mask = hold.reshape(-1)
    positive_obs_np = conditioned.reshape(-1, 198)[positive_mask]
    positive_action_np = action.reshape(-1, 20)[positive_mask]

    with np.load(args.v14_data.resolve(), allow_pickle=False) as data:
        v14_obs = np.asarray(data["teacher_observation"], dtype=np.float32)[:, 0]
        stop_target = np.asarray(data["stop_target"], dtype=np.float32)
    stable_obs_np = v14_obs[stop_target > 0.5]
    if len(positive_obs_np) == 0 or len(stable_obs_np) == 0:
        raise ValueError("Positive correction and stable-zero pools must be non-empty")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    positive_obs = torch.as_tensor(positive_obs_np, device=device)
    positive_action = torch.as_tensor(positive_action_np, device=device)
    stable_obs = torch.as_tensor(stable_obs_np, device=device)
    checkpoint = torch.load(base_checkpoint, map_location="cpu", weights_only=False)
    base = PolicyNet(198).to(device)
    base.load_skrl(checkpoint["policy"])
    base.eval()
    for parameter in base.parameters():
        parameter.requires_grad_(False)
    preprocessor = checkpoint["observation_preprocessor"]
    mean = preprocessor["running_mean"].to(device=device, dtype=torch.float32)
    variance = preprocessor["running_variance"].to(device=device, dtype=torch.float32)

    stop_adapter = TerminalStopAdapter(
        int(stop_artifact["hidden_size"]), float(stop_artifact["residual_scale"])
    ).to(device)
    stop_adapter.load_state_dict(stop_artifact["state_dict"])
    stop_adapter.eval()
    for parameter in stop_adapter.parameters():
        parameter.requires_grad_(False)
    adapter = PostStopCorrectionAdapter(
        stop_adapter, args.hidden_size, args.correction_scale
    ).to(device)
    learned = torch.as_tensor(LEARNED_INDICES, dtype=torch.long, device=device)

    def base_action(obs: torch.Tensor) -> torch.Tensor:
        return base((obs - mean) / torch.sqrt(variance + 1.0e-8))

    def compose_without_correction(obs: torch.Tensor) -> torch.Tensor:
        base_actions = base_action(obs)
        residual, stop = stop_adapter(obs[:, 186:198])
        return (1.0 - stop) * (base_actions[:, learned] + residual)

    optimizer = torch.optim.AdamW(
        adapter.correction_net.parameters(), lr=args.lr, weight_decay=1.0e-5
    )
    generator = torch.Generator(device=device).manual_seed(args.seed)
    half = args.batch_size // 2
    trace = []
    for step in range(args.steps):
        positive_rows = torch.randint(
            0, len(positive_obs), (half,), generator=generator, device=device
        )
        stable_rows = torch.randint(
            0, len(stable_obs), (args.batch_size - half,), generator=generator, device=device
        )
        pos_obs = positive_obs[positive_rows]
        neg_obs = stable_obs[stable_rows]
        with torch.no_grad():
            pos_base = compose_without_correction(pos_obs)
        _, _, pos_correction = adapter(pos_obs[:, 186:198])
        _, _, neg_correction = adapter(neg_obs[:, 186:198])
        positive_loss = torch.mean(
            (pos_base + pos_correction - positive_action[positive_rows][:, learned]) ** 2
        )
        stable_loss = torch.mean(neg_correction.square())
        loss = positive_loss + float(args.stable_zero_weight) * stable_loss
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(adapter.correction_net.parameters(), 1.0)
        optimizer.step()
        if step % 250 == 0 or step == args.steps - 1:
            trace.append(
                {
                    "step": step,
                    "loss": float(loss.detach()),
                    "positive_mse": float(positive_loss.detach()),
                    "stable_zero_mse": float(stable_loss.detach()),
                }
            )

    with torch.no_grad():
        pos_base = compose_without_correction(positive_obs)
        _, _, pos_correction = adapter(positive_obs[:, 186:198])
        _, _, stable_correction = adapter(stable_obs[:, 186:198])
        positive_mse = torch.mean(
            (pos_base + pos_correction - positive_action[:, learned]) ** 2
        )
        stable_zero_mse = torch.mean(stable_correction.square())

    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    artifact = {
        "state_dict": {key: value.detach().cpu() for key, value in adapter.state_dict().items()},
        "input_columns": tuple(range(186, 198)),
        "learned_action_indices": LEARNED_INDICES,
        "hidden_size": int(stop_artifact["hidden_size"]),
        "residual_scale": float(stop_artifact["residual_scale"]),
        "has_stop_head": True,
        "has_post_stop_correction": True,
        "correction_hidden_size": args.hidden_size,
        "correction_scale": args.correction_scale,
        "base_checkpoint": str(base_checkpoint),
        "base_checkpoint_sha256": sha256(base_checkpoint),
        "frozen_stop_adapter": str(stop_adapter_path),
        "frozen_stop_adapter_sha256": sha256(stop_adapter_path),
    }
    torch.save(artifact, output)
    metrics = {
        "base_checkpoint": str(base_checkpoint),
        "frozen_stop_adapter": str(stop_adapter_path),
        "long_hold_teacher": str(args.long_hold_teacher.resolve()),
        "positive_correction_rows": int(len(positive_obs)),
        "stable_zero_rows": int(len(stable_obs)),
        "output": str(output),
        "output_sha256": sha256(output),
        "steps": args.steps,
        "correction_scale": args.correction_scale,
        "positive_action_mse": float(positive_mse),
        "stable_zero_mse": float(stable_zero_mse),
        "loss_trace": trace,
        "ppo_started": False,
    }
    metrics_path = args.metrics.resolve()
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print("P1_P2_POST_STOP_CORRECTION=" + json.dumps(metrics, ensure_ascii=False))


if __name__ == "__main__":
    main()
