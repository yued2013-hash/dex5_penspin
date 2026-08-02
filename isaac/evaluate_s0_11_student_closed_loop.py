from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from isaaclab.app import AppLauncher


PROJECT_ROOT = Path(__file__).resolve().parents[1]
for import_root in (PROJECT_ROOT, PROJECT_ROOT.parent):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))


parser = argparse.ArgumentParser()
parser.add_argument("--task", default="Dex5-S0-ShortRotation-16-MediumDR-Direct-v0")
parser.add_argument("--checkpoint", type=Path, required=True)
parser.add_argument("--source-rollout", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--episodes", type=int, default=197)
parser.add_argument("--seed", type=int, default=20260802)
AppLauncher.add_app_launcher_args(parser)
args, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0], *hydra_args]

launcher = AppLauncher(args)
simulation_app = launcher.app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

import dex5_penspin.isaac.dex5_penspin_isaaclab  # noqa: E402, F401
from isaaclab_tasks.utils.hydra import hydra_task_config  # noqa: E402
from sim2real.s0_11_student_policy import load_student_checkpoint, sha256  # noqa: E402


GATE_NAMES = (
    "target_band_axis",
    "target_speed",
    "wrong_axis",
    "height",
    "tim_all",
    "ring_pinky_clear",
    "no_failure",
    "all_instantaneous",
)


def _failure_reason(row: dict[str, object]) -> str:
    if bool(row["success"]):
        return "success"
    if float(row["max_penetration_mm"]) > 0.5:
        return "penetration"
    if bool(row["physical_failure_seen"]):
        return "physical_failure_or_drop"
    if float(row["peak_quaternion_rotation_deg"]) < 30.0:
        return "rotation_below_30deg"
    gates = row["gates_seen"]
    assert isinstance(gates, dict)
    if not bool(gates["target_band_axis"]):
        return "axis_or_target_band"
    if not bool(gates["tim_all"]):
        return "tim_contact"
    if not bool(gates["all_instantaneous"]):
        return "stability_gate"
    if int(row["max_hold_steps"]) < 9:
        return "hold_incomplete"
    if bool(row["timeout"]):
        return "timeout"
    return "other"


@hydra_task_config(args.task, "skrl_cfg_entry_point")
def main(env_cfg, _agent_cfg: dict) -> None:
    if args.episodes != 197:
        raise RuntimeError("S0.11 requires the frozen 197-episode test split")
    env_cfg.scene.num_envs = args.episodes
    env_cfg.sim.device = args.device
    env_cfg.seed = args.seed
    env_cfg.reset_split = "test"

    with np.load(args.source_rollout, allow_pickle=False) as source:
        split = np.asarray(source["episode_split"])
        test_rows = np.flatnonzero(split == "test")
        bank_ids = np.asarray(source["episode_bank_id"], dtype=np.int64)[test_rows]
        source_ids = np.asarray(
            source["episode_source_trajectory_id"], dtype=np.int64
        )[test_rows]
    if len(test_rows) != args.episodes:
        raise RuntimeError("source rollout no longer has exactly 197 test episodes")

    raw = gym.make(args.task, cfg=env_cfg)
    direct = raw.unwrapped
    device = torch.device(direct.device)
    model, checkpoint = load_student_checkpoint(args.checkpoint, device)
    if checkpoint["dataset_sha256"] != sha256(
        Path(checkpoint["dataset_path"])
    ):
        raise RuntimeError("student checkpoint dataset hash mismatch")
    if checkpoint.get("privileged_observation_used") is not False:
        raise RuntimeError("student checkpoint is not non-privileged")

    raw.reset()
    all_ids = direct.hand._ALL_INDICES
    direct.forced_bank_ids = torch.as_tensor(bank_ids, device=device, dtype=torch.long)
    direct._reset_idx(all_ids)
    del direct.forced_bank_ids
    initial_generation = direct.completed_s0_generation.detach().clone()
    completed = torch.zeros(args.episodes, dtype=torch.bool, device=device)
    tim_steps = torch.zeros(args.episodes, dtype=torch.long, device=device)
    active_steps = torch.zeros_like(tim_steps)
    max_steps = int(round(float(direct.cfg.episode_length_s) * direct.cfg.s0_policy_hz)) + 8
    records: list[dict[str, object] | None] = [None] * args.episodes

    for _ in range(max_steps):
        active = ~completed
        if not bool(active.any()):
            break
        direct._compute_intermediate_values()
        contacts = direct.contact_force_norm > direct.cfg.contact_force_threshold_n
        tim_steps[active] += contacts[active, :3].all(dim=-1).long()
        active_steps[active] += 1
        with torch.inference_mode():
            student_obs = direct.get_student_observation()
            if student_obs.shape != (args.episodes, 340):
                raise RuntimeError("student observation contract changed")
            actions = model(student_obs)
            actions[completed] = 0.0
        raw.step(actions)
        newly_completed = (~completed) & (
            direct.completed_s0_generation != initial_generation
        )
        for env_id in torch.nonzero(newly_completed, as_tuple=False).flatten().tolist():
            gates = direct.last_completed_s0_gate_seen[env_id].detach().cpu().tolist()
            record: dict[str, object] = {
                "episode_index": env_id,
                "source_rollout_episode_index": int(test_rows[env_id]),
                "source_trajectory_id": int(source_ids[env_id]),
                "bank_id": int(direct.last_completed_s0_bank_id[env_id].item()),
                "success": bool(direct.last_completed_s0_success[env_id].item()),
                "peak_quaternion_rotation_deg": float(
                    torch.rad2deg(direct.last_completed_s0_peak_angle[env_id]).item()
                ),
                "terminal_quaternion_rotation_deg": float(
                    torch.rad2deg(direct.last_completed_s0_actual_angle[env_id]).item()
                ),
                "terminal_axis_alignment": float(
                    direct.last_completed_s0_axis_alignment[env_id].item()
                ),
                "max_hold_steps": int(
                    direct.last_completed_s0_max_hold[env_id].item()
                ),
                "max_hold_ms": float(
                    direct.last_completed_s0_max_hold[env_id].item()
                    / direct.cfg.s0_policy_hz
                    * 1000.0
                ),
                "tim_contact_ratio": float(
                    tim_steps[env_id].float() / active_steps[env_id].clamp_min(1)
                ),
                "max_penetration_mm": float(
                    1000.0
                    * direct.last_completed_s0_max_penetration[env_id].item()
                ),
                "physical_failure_seen": bool(
                    direct.last_completed_s0_failure_seen[env_id].item()
                ),
                "timeout": bool(direct.last_completed_s0_timeout[env_id].item()),
                "forbidden_transition": bool(
                    direct.last_completed_s0_forbidden[env_id].item()
                ),
                "gates_seen": dict(zip(GATE_NAMES, map(bool, gates), strict=True)),
            }
            record["failure_reason"] = _failure_reason(record)
            records[env_id] = record
        completed |= newly_completed

    if not bool(completed.all()):
        missing = torch.nonzero(~completed, as_tuple=False).flatten().tolist()
        raise RuntimeError(f"closed-loop evaluation did not finish envs: {missing}")
    final_records = [record for record in records if record is not None]
    taxonomy: dict[str, int] = {}
    for record in final_records:
        reason = str(record["failure_reason"])
        taxonomy[reason] = taxonomy.get(reason, 0) + 1

    def rate(predicate) -> float:
        return sum(bool(predicate(row)) for row in final_records) / len(final_records)

    report = {
        "schema": "dex5-s0.11-student-closed-loop-eval-v1",
        "status": "PASS_EVALUATION_COMPLETE",
        "task": args.task,
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": sha256(args.checkpoint),
        "source_rollout": str(args.source_rollout.resolve()),
        "source_rollout_sha256": sha256(args.source_rollout),
        "seed": args.seed,
        "reset_split": "test",
        "episodes": len(final_records),
        "unique_source_trajectories": len(set(source_ids.tolist())),
        "policy_observation": "student_observation_340d_only",
        "oracle_observation_read_by_policy": False,
        "hardware_commands_issued": False,
        "successes": sum(bool(row["success"]) for row in final_records),
        "success_rate": rate(lambda row: row["success"]),
        "quaternion_rotation_30deg_rate": rate(
            lambda row: row["peak_quaternion_rotation_deg"] >= 30.0
        ),
        "axis_alignment_gate_rate": rate(
            lambda row: row["gates_seen"]["target_band_axis"]
        ),
        "hold_150ms_rate": rate(lambda row: row["max_hold_steps"] >= 9),
        "tim_contact_episode_mean": float(
            np.mean([row["tim_contact_ratio"] for row in final_records])
        ),
        "tim_contact_full_episode_rate": rate(
            lambda row: row["tim_contact_ratio"] >= 0.999
        ),
        "penetration": {
            "maximum_mm": max(float(row["max_penetration_mm"]) for row in final_records),
            "mean_mm": float(
                np.mean([row["max_penetration_mm"] for row in final_records])
            ),
            "over_0_5mm_rate": rate(
                lambda row: row["max_penetration_mm"] > 0.5
            ),
        },
        "failure_taxonomy": taxonomy,
        "episode_records": final_records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "episode_records"}, indent=2))
    raw.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
