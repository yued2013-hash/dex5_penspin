from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn


RING = slice(8, 12)
NON_RING = [*range(0, 8), *range(12, 20)]
TIM_FLEX = [1, 2, 3, 5, 6, 7, 17, 18, 19]


class PolicyNet(nn.Module):
    def __init__(self, observation_dim: int) -> None:
        super().__init__()
        self.fc0 = nn.Linear(observation_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc4 = nn.Linear(256, 128)
        self.head = nn.Linear(128, 20)

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        hidden = torch.nn.functional.elu(self.fc0(observation))
        hidden = torch.nn.functional.elu(self.fc2(hidden))
        hidden = torch.nn.functional.elu(self.fc4(hidden))
        return self.head(hidden)

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
    parser = argparse.ArgumentParser(description="BC warm-start for a strict ring-only P1-to-P2 teacher.")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--base-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=6000)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--lr", type=float, default=3.0e-4)
    parser.add_argument("--seed", type=int, default=2821)
    parser.add_argument("--allow-tim-assist", action="store_true")
    parser.add_argument(
        "--preserve-forbidden-policy",
        action="store_true",
        help="Retain base outputs outside learned indices instead of zeroing their heads.",
    )
    parser.add_argument(
        "--preserve-preprocessor",
        action="store_true",
        help="Keep observation normalization statistics from the base checkpoint.",
    )
    parser.add_argument(
        "--balance-key",
        type=str,
        default="",
        help="Trajectory-level dataset key used for equal-mode minibatch sampling.",
    )
    parser.add_argument(
        "--preserve-base-groups",
        type=str,
        default="",
        help="Comma-separated balance groups distilled from the base policy instead of teacher labels.",
    )
    parser.add_argument(
        "--phase-input-column",
        type=int,
        default=-1,
        help="Only train one first-layer input column; useful as a phase-gated adapter.",
    )
    parser.add_argument(
        "--adapter-input-columns",
        type=str,
        default="",
        help=(
            "Comma-separated first-layer input columns to train while freezing the "
            "rest of the policy. Supersedes --phase-input-column when provided."
        ),
    )
    parser.add_argument(
        "--condition-mode-key",
        type=str,
        default="",
        help="Trajectory mode label appended as a six-way desired-mode one-hot.",
    )
    parser.add_argument(
        "--waypoint-manifold-bank",
        type=Path,
        default=None,
        help="Paired manifold bank used to reconstruct four waypoint phases.",
    )
    args = parser.parse_args()
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    with np.load(args.dataset) as data:
        observation = np.asarray(data["observation"], dtype=np.float32)
        action = np.asarray(data["action"], dtype=np.float32)
        lengths = np.asarray(data["lengths"], dtype=np.int64)
        trajectory_groups = (
            np.asarray(data[args.balance_key], dtype=np.int64)
            if args.balance_key
            else None
        )
        condition_modes = (
            np.asarray(data[args.condition_mode_key], dtype=np.int64)
            if args.condition_mode_key
            else None
        )
        source_archive_rows = (
            np.asarray(data["source_archive_row"], dtype=np.int64)
            if args.condition_mode_key
            else None
        )
    if condition_modes is not None:
        if args.waypoint_manifold_bank is None:
            raise ValueError("--waypoint-manifold-bank is required with mode conditioning")
        if np.any((condition_modes < 0) | (condition_modes >= 6)):
            raise ValueError("Condition modes must be in [0, 5]")
        with np.load(args.waypoint_manifold_bank) as manifold:
            manifold_rows = np.asarray(manifold["source_row_ids"], dtype=np.int64)
            route_parameters = np.asarray(manifold["search_parameters"], dtype=np.float32)
        parameter_by_row = {
            int(row): route_parameters[index] for index, row in enumerate(manifold_rows)
        }
        durations = np.stack(
            [parameter_by_row[int(row)][8:11] for row in source_archive_rows]
        )
        elapsed = np.arange(observation.shape[1], dtype=np.float32)[None, :] + 1.0
        d1 = durations[:, 0:1]
        d12 = d1 + durations[:, 1:2]
        d123 = d12 + durations[:, 2:3]
        phases = np.where(
            elapsed <= d1,
            0,
            np.where(elapsed <= d12, 1, np.where(elapsed <= d123, 2, 3)),
        )
        mode_one_hot = np.eye(6, dtype=np.float32)[condition_modes]
        mode_one_hot = np.broadcast_to(
            mode_one_hot[:, None, :], (*observation.shape[:2], 6)
        )
        phase_one_hot = np.eye(4, dtype=np.float32)[phases]
        observation = np.concatenate(
            (observation, mode_one_hot, phase_one_hot), axis=-1
        )
    valid = np.arange(observation.shape[1])[None, :] < lengths[:, None]
    row_groups = (
        np.broadcast_to(trajectory_groups[:, None], valid.shape)[valid]
        if trajectory_groups is not None
        else None
    )
    observation = observation[valid]
    action = action[valid]
    learned_indices = [8, 9, 10, 11]
    if args.allow_tim_assist:
        learned_indices.extend(TIM_FLEX)
    forbidden_indices = [index for index in range(20) if index not in learned_indices]
    forbidden_max = float(np.max(np.abs(action[:, forbidden_indices])))
    if forbidden_max > 1.0e-7:
        raise ValueError(f"Teacher contains forbidden action {forbidden_max}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    obs = torch.as_tensor(observation, device=device)
    target = torch.as_tensor(action, device=device)
    checkpoint = torch.load(args.base_checkpoint, map_location="cpu", weights_only=False)
    for model_key in ("policy", "value"):
        input_weight = checkpoint[model_key]["net_container.0.weight"]
        if input_weight.shape[1] > obs.shape[-1]:
            raise ValueError(
                f"Checkpoint {model_key} input {input_weight.shape[1]} exceeds dataset {obs.shape[-1]}"
            )
        if input_weight.shape[1] < obs.shape[-1]:
            expanded = torch.zeros(
                (input_weight.shape[0], obs.shape[-1]), dtype=input_weight.dtype
            )
            expanded[:, : input_weight.shape[1]] = input_weight
            checkpoint[model_key]["net_container.0.weight"] = expanded
    if args.preserve_preprocessor:
        preprocessor = checkpoint["observation_preprocessor"]
        mean = preprocessor["running_mean"].to(device=device, dtype=obs.dtype)
        variance = preprocessor["running_variance"].to(device=device, dtype=obs.dtype)
        if mean.shape != obs.shape[1:] or variance.shape != obs.shape[1:]:
            raise ValueError(
                "Base checkpoint preprocessor shape does not match dataset observations: "
                f"mean={tuple(mean.shape)}, variance={tuple(variance.shape)}, "
                f"observation={tuple(obs.shape[1:])}"
            )
        preprocessor_count = preprocessor["current_count"]
    else:
        mean = obs.mean(dim=0)
        variance = obs.var(dim=0).clamp_min(1.0e-8)
        preprocessor_count = torch.tensor(float(len(obs)))
    variance = variance.clamp_min(1.0e-8)
    obs_n = torch.clamp((obs - mean) / (torch.sqrt(variance) + 1.0e-8), -5.0, 5.0)
    model = PolicyNet(obs.shape[-1]).to(device)
    model.load_skrl(checkpoint["policy"])
    if not args.preserve_forbidden_policy:
        with torch.no_grad():
            model.head.weight[forbidden_indices] = 0.0
            model.head.bias[forbidden_indices] = 0.0
    preserve_base_groups = {
        int(value) for value in args.preserve_base_groups.split(",") if value.strip()
    }
    if preserve_base_groups:
        if row_groups is None:
            raise ValueError("--preserve-base-groups requires --balance-key")
        preserve_mask_np = np.isin(row_groups, list(preserve_base_groups))
        preserve_mask = torch.as_tensor(preserve_mask_np, device=device)
        preserve_rows = torch.nonzero(preserve_mask, as_tuple=False).squeeze(-1)
        learned_columns = torch.as_tensor(
            learned_indices, dtype=torch.long, device=device
        )
        with torch.no_grad():
            base_prediction = model(obs_n[preserve_rows])
            target[preserve_rows[:, None], learned_columns[None, :]] = base_prediction[
                :, learned_columns
            ]
    adapter_input_columns = [
        int(value) for value in args.adapter_input_columns.split(",") if value.strip()
    ]
    if not adapter_input_columns and args.phase_input_column >= 0:
        adapter_input_columns = [args.phase_input_column]
    adapter_input_columns = sorted(set(adapter_input_columns))
    invalid_adapter_columns = [
        column
        for column in adapter_input_columns
        if column < 0 or column >= obs.shape[-1]
    ]
    if invalid_adapter_columns:
        raise ValueError(
            f"Adapter input columns {invalid_adapter_columns} exceed observation size {obs.shape[-1]}"
        )
    if adapter_input_columns:
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        model.fc0.weight.requires_grad_(True)
        optimizer_parameters = [model.fc0.weight]
    else:
        optimizer_parameters = list(model.parameters())
    optimizer = torch.optim.AdamW(
        optimizer_parameters,
        lr=args.lr,
        weight_decay=0.0 if adapter_input_columns else 1.0e-6,
    )
    generator = torch.Generator(device=device); generator.manual_seed(args.seed)
    group_rows: dict[int, torch.Tensor] = {}
    if row_groups is not None:
        for group in np.unique(row_groups):
            group_rows[int(group)] = torch.as_tensor(
                np.flatnonzero(row_groups == group), dtype=torch.long, device=device
            )
        if len(group_rows) < 2:
            raise ValueError(f"Balance key {args.balance_key!r} has fewer than two groups")
    group_ids = sorted(group_rows)
    loss_trace = []
    for step in range(args.steps):
        if group_rows:
            counts = [args.batch_size // len(group_ids)] * len(group_ids)
            for index in range(args.batch_size % len(group_ids)):
                counts[index] += 1
            sampled = []
            for group, count in zip(group_ids, counts):
                pool = group_rows[group]
                choices = torch.randint(
                    0, len(pool), (count,), generator=generator, device=device
                )
                sampled.append(pool[choices])
            rows = torch.cat(sampled)
            rows = rows[torch.randperm(len(rows), generator=generator, device=device)]
        else:
            rows = torch.randint(
                0, len(obs), (args.batch_size,), generator=generator, device=device
            )
        prediction = model(obs_n[rows])
        loss = torch.mean(
            (prediction[:, learned_indices] - target[rows][:, learned_indices]) ** 2
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if adapter_input_columns:
            gradient = model.fc0.weight.grad
            frozen_columns = torch.ones(
                gradient.shape[1], dtype=torch.bool, device=gradient.device
            )
            frozen_columns[adapter_input_columns] = False
            gradient[:, frozen_columns] = 0.0
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if not args.preserve_forbidden_policy:
            with torch.no_grad():
                model.head.weight[forbidden_indices] = 0.0
                model.head.bias[forbidden_indices] = 0.0
        if step % 250 == 0 or step == args.steps - 1:
            loss_trace.append({"step": step, "learned_action_mse": float(loss.detach())})
    with torch.no_grad():
        prediction = model(obs_n)
        ring_error = prediction[:, RING] - target[:, RING]
        emitted_forbidden_max = float(
            torch.max(torch.abs(prediction[:, forbidden_indices]))
        )
        tim_error = prediction[:, TIM_FLEX] - target[:, TIM_FLEX]
    per_group_metrics = {}
    if row_groups is not None:
        for group in group_ids:
            mask = torch.as_tensor(row_groups == group, device=device)
            per_group_metrics[str(group)] = {
                "rows": int(mask.sum()),
                "ring_mse": float(torch.mean(ring_error[mask].square())),
                "ring_mae": float(torch.mean(torch.abs(ring_error[mask]))),
            }
    model.store_skrl(checkpoint["policy"])
    # The configured skrl model is shared (`separate: False`). Both checkpoint
    # entries contain the complete model and are loaded in sequence, so leaving
    # `value` stale would overwrite the BC-updated policy at restore time.
    for key, value in checkpoint["policy"].items():
        checkpoint["value"][key] = value.clone() if torch.is_tensor(value) else value
    checkpoint["observation_preprocessor"]["running_mean"] = mean.detach().cpu()
    checkpoint["observation_preprocessor"]["running_variance"] = variance.detach().cpu()
    checkpoint["observation_preprocessor"]["current_count"] = torch.as_tensor(
        preprocessor_count
    ).detach().cpu()
    checkpoint.pop("optimizer", None)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.metrics.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, args.output)
    metrics = {
        "dataset": str(args.dataset), "base_checkpoint": str(args.base_checkpoint),
        "output": str(args.output), "rows": len(obs), "steps": args.steps,
        "ring_mse": float(torch.mean(ring_error.square())),
        "ring_mae": float(torch.mean(torch.abs(ring_error))),
        "tim_assist_enabled": args.allow_tim_assist,
        "forbidden_policy_preserved": args.preserve_forbidden_policy,
        "preprocessor_preserved": args.preserve_preprocessor,
        "shared_policy_value_synced": True,
        "tim_mse": float(torch.mean(tim_error.square())),
        "balance_key": args.balance_key or None,
        "preserve_base_groups": sorted(preserve_base_groups),
        "phase_input_column": (
            args.phase_input_column if args.phase_input_column >= 0 else None
        ),
        "adapter_input_columns": adapter_input_columns,
        "condition_mode_key": args.condition_mode_key or None,
        "conditioned_observation_dim": int(obs.shape[-1]),
        "trajectory_group_counts": (
            {
                str(group): int(np.sum(trajectory_groups == group))
                for group in np.unique(trajectory_groups)
            }
            if trajectory_groups is not None
            else None
        ),
        "per_group_metrics": per_group_metrics,
        "teacher_forbidden_max_abs": forbidden_max,
        "policy_forbidden_max_abs": emitted_forbidden_max,
        "pinky_action_max_abs": float(torch.max(torch.abs(prediction[:, 12:16]))),
        "loss_trace": loss_trace, "device": str(device),
    }
    args.metrics.write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("P1_P2_RING_BC=" + json.dumps(metrics, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
