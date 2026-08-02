from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn


ROOT = Path(__file__).resolve().parents[2]
REPORT = ROOT / "dex5_penspin/reports/m3_2_30_index_near_surface"
BUNDLE = REPORT / "isaac_first_leg_bundle.npz"
OUT_POLICY = REPORT / "isaac_bc_warmstart_policy.pt"
OUT_METRICS = REPORT / "isaac_bc_warmstart_metrics.json"


class MlpPolicy(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)


def build_dataset(bundle_path: Path, augment_repeats: int, noise_std: float, seed: int) -> tuple[np.ndarray, np.ndarray, dict]:
    rng = np.random.default_rng(seed)
    data = np.load(bundle_path)
    observations = []
    actions = []
    rows = []

    for window_index, window in enumerate(("016_019", "019_022")):
        source_qpos = np.asarray(data[f"{window}_source_qpos"], dtype=np.float32)
        source_qvel = np.asarray(data[f"{window}_source_qvel"], dtype=np.float32)
        source_target_qpos = np.asarray(data[f"{window}_source_target_qpos"], dtype=np.float32)
        target_hand_qpos = np.asarray(data[f"{window}_target_hand_qpos"], dtype=np.float32)
        source_t = np.asarray(data[f"{window}_source_t"], dtype=np.float32)
        window_onehot = np.zeros((source_qpos.shape[0], 2), dtype=np.float32)
        window_onehot[:, window_index] = 1.0
        tau = ((source_t - source_t.min()) / max(float(source_t.max() - source_t.min()), 1e-6))[:, None]

        base_obs = np.concatenate((source_qpos, source_qvel, source_target_qpos, tau, window_onehot), axis=-1)
        base_action = target_hand_qpos[None, :] - source_qpos[:, : target_hand_qpos.shape[0]]

        for repeat in range(augment_repeats):
            obs = base_obs.copy()
            if repeat > 0:
                obs[:, : source_qpos.shape[1]] += rng.normal(0.0, noise_std, size=source_qpos.shape).astype(np.float32)
                obs[:, source_qpos.shape[1] : source_qpos.shape[1] + source_qvel.shape[1]] += rng.normal(
                    0.0, noise_std * 5.0, size=source_qvel.shape
                ).astype(np.float32)
            observations.append(obs)
            actions.append(base_action)

        rows.append(
            {
                "window": window,
                "source_states": int(source_qpos.shape[0]),
                "augmented_states": int(source_qpos.shape[0] * augment_repeats),
                "action_delta_abs_mean": float(np.mean(np.abs(base_action))),
                "action_delta_abs_max": float(np.max(np.abs(base_action))),
            }
        )

    obs_np = np.concatenate(observations, axis=0)
    act_np = np.concatenate(actions, axis=0)
    metadata = {"rows": rows, "obs_dim": int(obs_np.shape[1]), "action_dim": int(act_np.shape[1])}
    return obs_np, act_np, metadata


def main() -> None:
    parser = argparse.ArgumentParser(description="BC warm-start for M3.2.30 index-near-surface first leg.")
    parser.add_argument("--bundle", type=Path, default=BUNDLE)
    parser.add_argument("--out-policy", type=Path, default=OUT_POLICY)
    parser.add_argument("--out-metrics", type=Path, default=OUT_METRICS)
    parser.add_argument("--steps", type=int, default=1600)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--hidden-dim", type=int, default=192)
    parser.add_argument("--augment-repeats", type=int, default=1024)
    parser.add_argument("--noise-std", type=float, default=0.003)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    obs_np, act_np, metadata = build_dataset(args.bundle, args.augment_repeats, args.noise_std, args.seed)
    device = torch.device(args.device)
    obs = torch.as_tensor(obs_np, device=device)
    action = torch.as_tensor(act_np, device=device)

    obs_mean = obs.mean(dim=0, keepdim=True)
    obs_std = obs.std(dim=0, keepdim=True).clamp_min(1e-4)
    action_mean = action.mean(dim=0, keepdim=True)
    action_std = action.std(dim=0, keepdim=True).clamp_min(1e-4)
    obs_n = (obs - obs_mean) / obs_std
    action_n = (action - action_mean) / action_std

    policy = MlpPolicy(obs_n.shape[-1], action_n.shape[-1], args.hidden_dim).to(device)
    opt = torch.optim.AdamW(policy.parameters(), lr=args.lr, weight_decay=1e-5)

    generator = torch.Generator(device=device)
    generator.manual_seed(args.seed)
    losses = []
    for step in range(args.steps):
        ids = torch.randint(0, obs_n.shape[0], (args.batch_size,), generator=generator, device=device)
        pred = policy(obs_n[ids])
        loss = torch.mean((pred - action_n[ids]) ** 2)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
        opt.step()
        if step % 100 == 0 or step == args.steps - 1:
            losses.append(float(loss.detach().cpu()))

    with torch.no_grad():
        pred_n = policy(obs_n)
        pred = pred_n * action_std + action_mean
        err = pred - action
        mse = torch.mean(err**2).item()
        mae = torch.mean(torch.abs(err)).item()
        max_abs = torch.max(torch.abs(err)).item()

    checkpoint = {
        "policy_state_dict": policy.state_dict(),
        "obs_mean": obs_mean.detach().cpu(),
        "obs_std": obs_std.detach().cpu(),
        "action_mean": action_mean.detach().cpu(),
        "action_std": action_std.detach().cpu(),
        "metadata": metadata,
        "policy": {"type": "mlp_delta_qpos", "hidden_dim": args.hidden_dim},
    }
    args.out_policy.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, args.out_policy)

    metrics = {
        "bundle": str(args.bundle),
        "out_policy": str(args.out_policy),
        "device": str(device),
        "steps": args.steps,
        "batch_size": args.batch_size,
        "augment_repeats": args.augment_repeats,
        "noise_std": args.noise_std,
        "loss_trace": losses,
        "eval_mse_m2": mse,
        "eval_mae_m": mae,
        "eval_max_abs_m": max_abs,
        **metadata,
    }
    args.out_metrics.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(metrics, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
