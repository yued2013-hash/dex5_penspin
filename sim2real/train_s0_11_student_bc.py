from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from s0_11_student_policy import Dex5StudentPolicy, checkpoint_payload, sha256


def _episode_rows(offsets: np.ndarray, episode_ids: np.ndarray) -> np.ndarray:
    return np.concatenate(
        [np.arange(offsets[index], offsets[index + 1]) for index in episode_ids]
    )


def _loader(
    observation: np.ndarray,
    action: np.ndarray,
    rows: np.ndarray,
    *,
    batch_size: int,
    shuffle: bool,
) -> DataLoader:
    dataset = TensorDataset(
        torch.from_numpy(observation[rows]), torch.from_numpy(action[rows])
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )


def _evaluate(
    model: nn.Module, loader: DataLoader, device: torch.device
) -> float:
    total = 0.0
    rows = 0
    model.eval()
    with torch.inference_mode():
        for observation, target in loader:
            observation = observation.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            loss = nn.functional.mse_loss(model(observation), target, reduction="sum")
            total += float(loss.item())
            rows += target.numel()
    return total / max(rows, 1)


def _train(
    model: Dex5StudentPolicy,
    train_loader: DataLoader,
    validation_loader: DataLoader,
    *,
    device: torch.device,
    epochs: int,
    learning_rate: float,
    patience: int,
) -> tuple[dict[str, object], dict[str, torch.Tensor]]:
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=1.0e-5
    )
    initial_train = _evaluate(model, train_loader, device)
    initial_validation = _evaluate(model, validation_loader, device)
    best_validation = float("inf")
    best_epoch = 0
    best_state: dict[str, torch.Tensor] = {}
    history: list[dict[str, float | int]] = []
    stale_epochs = 0
    started = time.monotonic()
    for epoch in range(1, epochs + 1):
        model.train()
        for observation, target in train_loader:
            observation = observation.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            loss = nn.functional.mse_loss(model(observation), target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
        train_mse = _evaluate(model, train_loader, device)
        validation_mse = _evaluate(model, validation_loader, device)
        history.append(
            {"epoch": epoch, "train_mse": train_mse, "validation_mse": validation_mse}
        )
        if validation_mse < best_validation - 1.0e-8:
            best_validation = validation_mse
            best_epoch = epoch
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            stale_epochs = 0
        else:
            stale_epochs += 1
        if stale_epochs >= patience:
            break
    model.load_state_dict(best_state)
    return (
        {
            "initial_train_mse": initial_train,
            "initial_validation_mse": initial_validation,
            "best_train_mse": min(float(row["train_mse"]) for row in history),
            "best_validation_mse": best_validation,
            "best_epoch": best_epoch,
            "epochs_run": len(history),
            "elapsed_s": time.monotonic() - started,
            "history": history,
        },
        best_state,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--report-json", type=Path, required=True)
    parser.add_argument("--report-md", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260802)
    parser.add_argument("--overfit-epochs", type=int, default=160)
    parser.add_argument("--full-epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--learning-rate", type=float, default=1.0e-3)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    dataset_hash = sha256(args.dataset)

    with np.load(args.dataset, allow_pickle=False) as data:
        observation = np.asarray(data["student_obs"], dtype=np.float32)
        action = np.asarray(data["action"], dtype=np.float32)
        offsets = np.asarray(data["episode_offsets"], dtype=np.int64)
        episode_split = np.asarray(data["episode_split"])
        mean = np.asarray(data["observation_mean_train"], dtype=np.float32)
        std = np.asarray(data["observation_std_train"], dtype=np.float32)

    train_episode_ids = np.flatnonzero(episode_split == "train")
    validation_episode_ids = np.flatnonzero(episode_split == "validation")
    test_episode_ids = np.flatnonzero(episode_split == "test")
    if (len(train_episode_ids), len(validation_episode_ids), len(test_episode_ids)) != (
        891,
        112,
        197,
    ):
        raise RuntimeError("frozen S0.10 episode split counts changed")

    rng = np.random.default_rng(args.seed)
    overfit_episode_ids = np.sort(
        rng.choice(train_episode_ids, size=100, replace=False)
    )
    overfit_rows = _episode_rows(offsets, overfit_episode_ids)
    train_rows = _episode_rows(offsets, train_episode_ids)
    validation_rows = _episode_rows(offsets, validation_episode_ids)
    test_rows = _episode_rows(offsets, test_episode_ids)

    overfit_model = Dex5StudentPolicy(
        torch.from_numpy(mean), torch.from_numpy(std)
    ).to(device)
    overfit_loader = _loader(
        observation, action, overfit_rows, batch_size=args.batch_size, shuffle=True
    )
    overfit_metrics, _ = _train(
        overfit_model,
        overfit_loader,
        overfit_loader,
        device=device,
        epochs=args.overfit_epochs,
        learning_rate=args.learning_rate,
        patience=args.overfit_epochs,
    )
    overfit_ratio = float(overfit_metrics["best_train_mse"]) / max(
        float(overfit_metrics["initial_train_mse"]), 1.0e-12
    )
    overfit_pass = bool(
        overfit_ratio <= 0.20 or float(overfit_metrics["best_train_mse"]) <= 0.01
    )
    if not overfit_pass:
        raise RuntimeError(f"100-episode BC overfit gate failed: {overfit_metrics}")

    torch.manual_seed(args.seed)
    full_model = Dex5StudentPolicy(
        torch.from_numpy(mean), torch.from_numpy(std)
    ).to(device)
    train_loader = _loader(
        observation, action, train_rows, batch_size=args.batch_size, shuffle=True
    )
    validation_loader = _loader(
        observation, action, validation_rows, batch_size=args.batch_size, shuffle=False
    )
    test_loader = _loader(
        observation, action, test_rows, batch_size=args.batch_size, shuffle=False
    )
    full_metrics, _ = _train(
        full_model,
        train_loader,
        validation_loader,
        device=device,
        epochs=args.full_epochs,
        learning_rate=args.learning_rate,
        patience=30,
    )
    offline_test_mse = _evaluate(full_model, test_loader, device)

    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        checkpoint_payload(
            full_model,
            dataset_path=args.dataset,
            dataset_sha256=dataset_hash,
            seed=args.seed,
            epoch=int(full_metrics["best_epoch"]),
            validation_mse=float(full_metrics["best_validation_mse"]),
        ),
        args.checkpoint,
    )
    report = {
        "schema": "dex5-s0.11-student-bc-training-v1",
        "status": "PASS",
        "dataset": str(args.dataset.resolve()),
        "dataset_sha256": dataset_hash,
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": sha256(args.checkpoint),
        "device": str(device),
        "seed": args.seed,
        "privileged_observation_used": False,
        "episode_counts": {"overfit": 100, "train": 891, "validation": 112, "test": 197},
        "row_counts": {
            "overfit": int(len(overfit_rows)),
            "train": int(len(train_rows)),
            "validation": int(len(validation_rows)),
            "test": int(len(test_rows)),
        },
        "overfit_smoke": {
            **overfit_metrics,
            "final_to_initial_ratio": overfit_ratio,
            "pass": overfit_pass,
        },
        "full_bc": full_metrics,
        "offline_test_mse": offline_test_mse,
        "hardware_commands_issued": False,
    }
    args.report_json.parent.mkdir(parents=True, exist_ok=True)
    args.report_json.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    args.report_md.write_text(
        "# Dex5 S0.11 Student BC Training Report\n\n"
        "## Status\n\n"
        "```text\nBC overfit smoke: PASS\nFull BC: PASS\n"
        "Simulation closed-loop: PENDING\nHardware control: NOT RUN\n```\n\n"
        "## Frozen inputs\n\n"
        f"- Dataset: `{args.dataset.resolve()}`\n"
        f"- Dataset SHA256: `{dataset_hash}`\n"
        "- Observation: deployable 340D student history only\n"
        "- Action: bounded 20D virtual normalized delta action\n"
        "- Privileged observation used: `false`\n\n"
        "## 100-episode overfit smoke\n\n"
        f"- Rows: `{len(overfit_rows)}`\n"
        f"- Initial MSE: `{float(overfit_metrics['initial_train_mse']):.8f}`\n"
        f"- Best MSE: `{float(overfit_metrics['best_train_mse']):.8f}`\n"
        f"- Final/initial ratio: `{overfit_ratio:.6f}`\n\n"
        "## Full BC\n\n"
        f"- Train/validation/test episodes: `891 / 112 / 197`\n"
        f"- Best epoch: `{full_metrics['best_epoch']}`\n"
        f"- Best validation MSE: `{float(full_metrics['best_validation_mse']):.8f}`\n"
        f"- Offline test MSE: `{offline_test_mse:.8f}`\n"
        f"- Checkpoint: `{args.checkpoint.resolve()}`\n"
        f"- Checkpoint SHA256: `{sha256(args.checkpoint)}`\n\n"
        "The offline test MSE is not a task-success result. Closed-loop PhysX "
        "evaluation is required before accepting the student policy.\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
