from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn


class PolicyNet(nn.Module):
    def __init__(self, observation_dim: int) -> None:
        super().__init__()
        self.fc0 = nn.Linear(observation_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc4 = nn.Linear(256, 128)
        self.head = nn.Linear(128, 20)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        x = torch.nn.functional.elu(self.fc0(obs))
        x = torch.nn.functional.elu(self.fc2(x))
        x = torch.nn.functional.elu(self.fc4(x))
        return self.head(x)

    def load_skrl(self, state: dict[str, torch.Tensor]) -> None:
        self.fc0.load_state_dict({"weight": state["net_container.0.weight"], "bias": state["net_container.0.bias"]})
        self.fc2.load_state_dict({"weight": state["net_container.2.weight"], "bias": state["net_container.2.bias"]})
        self.fc4.load_state_dict({"weight": state["net_container.4.weight"], "bias": state["net_container.4.bias"]})
        self.head.load_state_dict({"weight": state["policy_layer.weight"], "bias": state["policy_layer.bias"]})

    def store_skrl(self, state: dict[str, torch.Tensor]) -> None:
        state["net_container.0.weight"] = self.fc0.weight.detach().cpu()
        state["net_container.0.bias"] = self.fc0.bias.detach().cpu()
        state["net_container.2.weight"] = self.fc2.weight.detach().cpu()
        state["net_container.2.bias"] = self.fc2.bias.detach().cpu()
        state["net_container.4.weight"] = self.fc4.weight.detach().cpu()
        state["net_container.4.bias"] = self.fc4.bias.detach().cpu()
        state["policy_layer.weight"] = self.head.weight.detach().cpu()
        state["policy_layer.bias"] = self.head.bias.detach().cpu()


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage-balanced BC distillation of the v9 expert cascade.")
    parser.add_argument("--dataset", type=Path, nargs="+", required=True)
    parser.add_argument("--base-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=6000)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--preserve-normalization",
        action="store_true",
        help="Keep the base checkpoint observation normalization while fine-tuning.",
    )
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    observation_sets = []
    action_sets = []
    stage_sets = []
    for dataset in args.dataset:
        with np.load(dataset) as data:
            obs_np = np.asarray(data["observation"])
            action_np = np.asarray(data["action"])
            stage_np = np.asarray(data["stage_id"])
            if obs_np.ndim == 3:
                lengths = np.asarray(data["lengths"], dtype=np.int64)
                valid = np.arange(obs_np.shape[1])[None, :] < lengths[:, None]
                obs_np = obs_np[valid]
                action_np = action_np[valid]
                stage_np = stage_np[valid]
            observation_sets.append(obs_np)
            action_sets.append(action_np)
            stage_sets.append(stage_np)
    obs_np = np.concatenate(observation_sets)
    action_np = np.concatenate(action_sets)
    stage_np = np.concatenate(stage_sets)
    del observation_sets, action_sets, stage_sets
    obs = torch.as_tensor(obs_np, dtype=torch.float32, device=device)
    action = torch.as_tensor(action_np, dtype=torch.float32, device=device)
    stage = torch.as_tensor(stage_np, dtype=torch.long, device=device)
    checkpoint = torch.load(args.base_checkpoint, map_location="cpu", weights_only=False)
    if args.preserve_normalization:
        preprocessor = checkpoint["observation_preprocessor"]
        mean = preprocessor["running_mean"].to(device=device, dtype=torch.float32)
        variance = preprocessor["running_variance"].to(
            device=device, dtype=torch.float32
        ).clamp_min(1e-8)
    else:
        mean = obs.mean(dim=0)
        variance = obs.var(dim=0).clamp_min(1e-8)
    obs_n = torch.clamp((obs - mean) / (torch.sqrt(variance) + 1e-8), -5.0, 5.0)
    model = PolicyNet(obs.shape[-1]).to(device)
    model.load_skrl(checkpoint["policy"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-6)
    stage_values = torch.unique(stage, sorted=True)
    stage_rows = [
        torch.nonzero(stage == value, as_tuple=False).squeeze(-1)
        for value in stage_values
    ]
    generator = torch.Generator(device=device); generator.manual_seed(args.seed)
    loss_trace = []
    for step in range(args.steps):
        sampled_stage = torch.randint(
            0, len(stage_rows), (args.batch_size,), generator=generator, device=device
        )
        ids = torch.empty(args.batch_size, dtype=torch.long, device=device)
        for stage_index, rows in enumerate(stage_rows):
            mask = sampled_stage == stage_index
            count = int(mask.sum())
            if count:
                ids[mask] = rows[torch.randint(0, len(rows), (count,), generator=generator, device=device)]
        prediction = model(obs_n[ids])
        loss = torch.mean((prediction - action[ids]) ** 2)
        optimizer.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step()
        if step % 250 == 0 or step == args.steps - 1:
            loss_trace.append({"step": step, "mse": float(loss.detach())})
    with torch.no_grad():
        prediction = model(obs_n)
        error = prediction - action
        per_stage = {}
        for value, rows in zip(stage_values.tolist(), stage_rows, strict=True):
            per_stage[str(value)] = {
                "rows": len(rows),
                "mse": float(torch.mean(error[rows] ** 2)),
                "mae": float(torch.mean(torch.abs(error[rows]))),
            }
    model.store_skrl(checkpoint["policy"])
    model.store_skrl(checkpoint["value"])
    if not args.preserve_normalization:
        checkpoint["observation_preprocessor"]["running_mean"] = mean.detach().cpu()
        checkpoint["observation_preprocessor"]["running_variance"] = variance.detach().cpu()
        checkpoint["observation_preprocessor"]["current_count"] = torch.tensor(float(len(obs)))
    checkpoint.pop("optimizer", None)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, args.output)
    metrics = {
        "dataset": [str(path) for path in args.dataset], "base_checkpoint": str(args.base_checkpoint),
        "output": str(args.output), "rows": len(obs), "steps": args.steps,
        "stage_values": stage_values.tolist(),
        "preserve_normalization": args.preserve_normalization,
        "batch_size": args.batch_size, "device": str(device), "loss_trace": loss_trace,
        "overall_mse": float(torch.mean(error ** 2)), "overall_mae": float(torch.mean(torch.abs(error))),
        "per_stage": per_stage,
    }
    args.metrics.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
