from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

from train_p1_p2_ring_only_bc_checkpoint import PolicyNet


LEARNED_INDICES = (8, 9, 10, 11, 1, 2, 3, 5, 6, 7, 17, 18, 19)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class TerminalStopAdapter(nn.Module):
    def __init__(self, hidden_size: int, residual_scale: float) -> None:
        super().__init__()
        self.trunk = nn.Sequential(nn.Linear(12, hidden_size), nn.Tanh())
        self.residual_head = nn.Linear(hidden_size, len(LEARNED_INDICES))
        self.stop_head = nn.Linear(hidden_size, 1)
        self.residual_scale = residual_scale

    def forward(self, condition: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self.trunk(condition)
        terminal = condition[:, :1]
        residual = terminal * self.residual_scale * torch.tanh(
            self.residual_head(hidden)
        )
        stop = terminal * torch.sigmoid(self.stop_head(hidden))
        return residual, stop


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a distilled terminal stop adapter.")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--base-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=10000)
    parser.add_argument("--batch-size", type=int, default=1536)
    parser.add_argument("--hidden-size", type=int, default=32)
    parser.add_argument("--residual-scale", type=float, default=1.0)
    parser.add_argument("--lr", type=float, default=6.0e-4)
    parser.add_argument("--seed", type=int, default=2821)
    parser.add_argument(
        "--initial-adapter",
        type=Path,
        default=None,
        help="Optional compatible stop adapter used as a warm start.",
    )
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    with np.load(args.dataset) as data:
        distill_obs_np = np.asarray(data["distill_observation"], dtype=np.float32)[:, 0]
        teacher_obs_np = np.asarray(data["teacher_observation"], dtype=np.float32)[:, 0]
        teacher_action_np = np.asarray(data["teacher_action"], dtype=np.float32)[:, 0]
        stop_target_np = np.asarray(data["stop_target"], dtype=np.float32)
        action_mask_np = np.asarray(data["action_loss_mask"], dtype=np.bool_)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    distill_obs = torch.as_tensor(distill_obs_np, device=device)
    teacher_obs = torch.as_tensor(teacher_obs_np, device=device)
    teacher_action = torch.as_tensor(teacher_action_np, device=device)
    stop_target = torch.as_tensor(stop_target_np, device=device)
    action_mask = torch.as_tensor(action_mask_np, device=device)

    checkpoint = torch.load(
        args.base_checkpoint.resolve(), map_location="cpu", weights_only=False
    )
    base = PolicyNet(teacher_obs.shape[-1]).to(device)
    base.load_skrl(checkpoint["policy"])
    base.eval()
    for parameter in base.parameters():
        parameter.requires_grad_(False)
    preprocessor = checkpoint["observation_preprocessor"]
    mean = preprocessor["running_mean"].to(device=device, dtype=teacher_obs.dtype)
    variance = preprocessor["running_variance"].to(
        device=device, dtype=teacher_obs.dtype
    )

    def base_action(observation: torch.Tensor) -> torch.Tensor:
        return base((observation - mean) / torch.sqrt(variance + 1.0e-8))

    with torch.no_grad():
        teacher_base = base_action(teacher_obs)
        distill_base = base_action(distill_obs)
    learned = torch.as_tensor(LEARNED_INDICES, dtype=torch.long, device=device)
    drive_rows = torch.nonzero(~teacher_obs[:, 197].bool(), as_tuple=False).squeeze(-1)
    stable_rows = torch.nonzero(stop_target > 0.5, as_tuple=False).squeeze(-1)
    failed_rows = torch.nonzero(
        teacher_obs[:, 197].bool() & (stop_target < 0.5), as_tuple=False
    ).squeeze(-1)
    pools = (drive_rows, stable_rows, failed_rows)
    if any(len(pool) == 0 for pool in pools):
        raise ValueError("Drive, stable-hold, and failed-hold pools must all be non-empty")

    adapter = TerminalStopAdapter(args.hidden_size, args.residual_scale).to(device)
    initial_adapter_path = (
        args.initial_adapter.resolve() if args.initial_adapter is not None else None
    )
    if initial_adapter_path is not None:
        initial = torch.load(initial_adapter_path, map_location="cpu", weights_only=False)
        if initial.get("base_checkpoint_sha256") != sha256(args.base_checkpoint.resolve()):
            raise ValueError("Initial adapter and base checkpoint SHA256 do not match")
        if int(initial["hidden_size"]) != int(args.hidden_size):
            raise ValueError("Initial adapter hidden size does not match")
        adapter.load_state_dict(initial["state_dict"])
    optimizer = torch.optim.AdamW(adapter.parameters(), lr=args.lr, weight_decay=1.0e-5)
    generator = torch.Generator(device=device).manual_seed(args.seed)
    loss_trace = []
    counts = (args.batch_size // 2, args.batch_size // 4, args.batch_size // 4)
    for step in range(args.steps):
        sampled = []
        for pool, count in zip(pools, counts, strict=True):
            sampled.append(
                pool[
                    torch.randint(
                        0, len(pool), (count,), generator=generator, device=device
                    )
                ]
            )
        rows = torch.cat(sampled)
        rows = rows[torch.randperm(len(rows), generator=generator, device=device)]
        condition = teacher_obs[rows, 186:198]
        residual, stop = adapter(condition)
        drive_action = teacher_base[rows][:, learned] + residual
        composite_action = (1.0 - stop) * drive_action
        selected_action_mask = action_mask[rows]
        action_loss = torch.mean(
            (
                composite_action[selected_action_mask]
                - teacher_action[rows][:, learned][selected_action_mask]
            )
            ** 2
        )
        stop_loss = torch.nn.functional.binary_cross_entropy(
            stop.squeeze(-1), stop_target[rows]
        )
        distill_rows = torch.randint(
            0,
            len(distill_obs),
            (args.batch_size // 2,),
            generator=generator,
            device=device,
        )
        distill_residual, distill_stop = adapter(distill_obs[distill_rows, 186:198])
        distill_action = (1.0 - distill_stop) * (
            distill_base[distill_rows][:, learned] + distill_residual
        )
        distill_loss = torch.mean(
            (distill_action - distill_base[distill_rows][:, learned]) ** 2
        )
        loss = action_loss + 0.35 * stop_loss + 1.5 * distill_loss
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(adapter.parameters(), 1.0)
        optimizer.step()
        if step % 250 == 0 or step == args.steps - 1:
            loss_trace.append(
                {
                    "step": step,
                    "loss": float(loss.detach()),
                    "action_mse": float(action_loss.detach()),
                    "stop_bce": float(stop_loss.detach()),
                    "distill_mse": float(distill_loss.detach()),
                }
            )

    with torch.no_grad():
        residual, stop = adapter(teacher_obs[:, 186:198])
        composite = (1.0 - stop) * (teacher_base[:, learned] + residual)
        final_action_mse = torch.mean(
            (composite[action_mask] - teacher_action[:, learned][action_mask]) ** 2
        )
        stop_prediction = stop.squeeze(-1) >= 0.5
        stop_accuracy = torch.mean(
            (stop_prediction == (stop_target > 0.5)).float()
        )
        stable_recall = torch.mean(stop_prediction[stable_rows].float())
        failed_reject = torch.mean((~stop_prediction[failed_rows]).float())

    output = args.output.resolve()
    metrics_path = args.metrics.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    artifact = {
        "state_dict": {key: value.detach().cpu() for key, value in adapter.state_dict().items()},
        "input_columns": tuple(range(186, 198)),
        "learned_action_indices": LEARNED_INDICES,
        "hidden_size": args.hidden_size,
        "residual_scale": args.residual_scale,
        "has_stop_head": True,
        "base_checkpoint": str(args.base_checkpoint.resolve()),
        "initial_adapter": str(initial_adapter_path) if initial_adapter_path else None,
        "initial_adapter_sha256": (
            sha256(initial_adapter_path) if initial_adapter_path else None
        ),
        "base_checkpoint_sha256": sha256(args.base_checkpoint.resolve()),
    }
    torch.save(artifact, output)
    metrics = {
        "dataset": str(args.dataset.resolve()),
        "dataset_sha256": sha256(args.dataset.resolve()),
        "base_checkpoint": str(args.base_checkpoint.resolve()),
        "output": str(output),
        "output_sha256": sha256(output),
        "steps": args.steps,
        "hidden_size": args.hidden_size,
        "action_mse": float(final_action_mse),
        "stop_accuracy": float(stop_accuracy),
        "stable_hold_recall": float(stable_recall),
        "failed_hold_rejection": float(failed_reject),
        "loss_trace": loss_trace,
        "ppo_started": False,
    }
    metrics_path.write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print("P1_P2_TERMINAL_STOP_ADAPTER=" + json.dumps(metrics, ensure_ascii=False))


if __name__ == "__main__":
    main()
