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


class TerminalResidualAdapter(nn.Module):
    def __init__(self, hidden_size: int, residual_scale: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(6, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, len(LEARNED_INDICES)),
        )
        self.residual_scale = residual_scale

    def forward(self, condition: torch.Tensor) -> torch.Tensor:
        terminal = condition[:, :1]
        return terminal * self.residual_scale * torch.tanh(self.net(condition))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train a nonlinear terminal residual adapter with base-policy distillation."
    )
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--base-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=8000)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--hidden-size", type=int, default=32)
    parser.add_argument("--residual-scale", type=float, default=1.0)
    parser.add_argument("--lr", type=float, default=8.0e-4)
    parser.add_argument("--seed", type=int, default=2821)
    parser.add_argument("--teacher-phase", type=int, default=6)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    with np.load(args.dataset) as data:
        observation = np.asarray(data["observation"], dtype=np.float32)[:, 0]
        teacher_action = np.asarray(data["action"], dtype=np.float32)[:, 0]
        phase_kind = np.asarray(data["phase_kind"], dtype=np.int64)
    if observation.shape[-1] != 192:
        raise ValueError(f"Expected 192-D observation, got {observation.shape}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    obs = torch.as_tensor(observation, device=device)
    teacher = torch.as_tensor(teacher_action, device=device)
    checkpoint = torch.load(
        args.base_checkpoint.resolve(), map_location="cpu", weights_only=False
    )
    base = PolicyNet(observation.shape[-1]).to(device)
    base.load_skrl(checkpoint["policy"])
    base.eval()
    for parameter in base.parameters():
        parameter.requires_grad_(False)
    preprocessor = checkpoint["observation_preprocessor"]
    mean = preprocessor["running_mean"].to(device=device, dtype=obs.dtype)
    variance = preprocessor["running_variance"].to(device=device, dtype=obs.dtype)
    obs_n = (obs - mean) / torch.sqrt(variance + 1.0e-8)
    with torch.no_grad():
        base_action = base(obs_n)

    learned = torch.as_tensor(LEARNED_INDICES, dtype=torch.long, device=device)
    condition = obs[:, 186:192]
    target_residual = teacher[:, learned] - base_action[:, learned]
    teacher_rows = torch.as_tensor(
        np.flatnonzero(phase_kind == args.teacher_phase),
        dtype=torch.long,
        device=device,
    )
    distill_rows = torch.as_tensor(
        np.flatnonzero(phase_kind != args.teacher_phase),
        dtype=torch.long,
        device=device,
    )
    if len(teacher_rows) == 0 or len(distill_rows) == 0:
        raise ValueError("Dataset must contain both paired teacher and distillation rows")

    adapter = TerminalResidualAdapter(args.hidden_size, args.residual_scale).to(device)
    optimizer = torch.optim.AdamW(adapter.parameters(), lr=args.lr, weight_decay=1.0e-5)
    generator = torch.Generator(device=device).manual_seed(args.seed)
    teacher_batch = args.batch_size // 2
    distill_batch = args.batch_size - teacher_batch
    loss_trace = []
    for step in range(args.steps):
        teacher_choice = teacher_rows[
            torch.randint(
                0, len(teacher_rows), (teacher_batch,), generator=generator, device=device
            )
        ]
        distill_choice = distill_rows[
            torch.randint(
                0, len(distill_rows), (distill_batch,), generator=generator, device=device
            )
        ]
        teacher_prediction = adapter(condition[teacher_choice])
        distill_prediction = adapter(condition[distill_choice])
        teacher_loss = torch.mean(
            (teacher_prediction - target_residual[teacher_choice]) ** 2
        )
        distill_loss = torch.mean(distill_prediction.square())
        smooth_loss = torch.mean(
            (adapter(condition[teacher_choice] + 0.002 * torch.randn_like(condition[teacher_choice]))
             - teacher_prediction.detach()) ** 2
        )
        loss = teacher_loss + 1.5 * distill_loss + 0.1 * smooth_loss
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(adapter.parameters(), 1.0)
        optimizer.step()
        if step % 250 == 0 or step == args.steps - 1:
            loss_trace.append(
                {
                    "step": step,
                    "loss": float(loss.detach()),
                    "teacher_mse": float(teacher_loss.detach()),
                    "distill_mse": float(distill_loss.detach()),
                }
            )

    with torch.no_grad():
        teacher_residual = adapter(condition[teacher_rows])
        distill_residual = adapter(condition[distill_rows])
        teacher_mse = torch.mean(
            (teacher_residual - target_residual[teacher_rows]) ** 2
        )
        distill_mse = torch.mean(distill_residual.square())
        terminal_off = condition[:, 0] < 0.5
        terminal_off_max = torch.max(torch.abs(adapter(condition[terminal_off])))

    output = args.output.resolve()
    metrics_path = args.metrics.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    artifact = {
        "state_dict": {key: value.detach().cpu() for key, value in adapter.state_dict().items()},
        "input_columns": (186, 187, 188, 189, 190, 191),
        "learned_action_indices": LEARNED_INDICES,
        "hidden_size": args.hidden_size,
        "residual_scale": args.residual_scale,
        "base_checkpoint": str(args.base_checkpoint.resolve()),
        "base_checkpoint_sha256": sha256(args.base_checkpoint.resolve()),
    }
    torch.save(artifact, output)
    metrics = {
        "dataset": str(args.dataset.resolve()),
        "dataset_sha256": sha256(args.dataset.resolve()),
        "base_checkpoint": str(args.base_checkpoint.resolve()),
        "output": str(output),
        "output_sha256": sha256(output),
        "teacher_rows": int(len(teacher_rows)),
        "distill_rows": int(len(distill_rows)),
        "hidden_size": args.hidden_size,
        "steps": args.steps,
        "teacher_mse": float(teacher_mse),
        "distill_mse": float(distill_mse),
        "terminal_off_max_residual": float(terminal_off_max),
        "loss_trace": loss_trace,
        "ppo_started": False,
    }
    metrics_path.write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print("P1_P2_TERMINAL_ADAPTER=" + json.dumps(metrics, ensure_ascii=False))


if __name__ == "__main__":
    main()
