from __future__ import annotations

import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
REPORT = ROOT / "dex5_penspin/reports/m3_2_30_index_near_surface"
CACHE = (
    ROOT
    / "dex5_penspin/reports/manus_mujoco_presets/short_primitive_reward_m3_2_23/dynamic_state_cache/"
    / "early_pre_ring_for_B12b_to_B23.npz"
)
TARGET_SEED = REPORT / "near_016_019/index_near_surface_seed.json"
OUTPUT = REPORT / "isaac_earlier_pre_ring_index_recapture_bundle.npz"


def mujoco_state_to_dex5_order(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values).copy()
    values[..., :20] = np.concatenate((values[..., 4:20], values[..., 0:4]), axis=-1)
    return values


def main() -> None:
    with np.load(CACHE, allow_pickle=False) as data:
        source = {key: np.asarray(data[key]) for key in data.files}
    seed = json.loads(TARGET_SEED.read_text(encoding="utf-8"))

    source_qpos = mujoco_state_to_dex5_order(source["qpos"].astype(np.float32))
    source_qvel = mujoco_state_to_dex5_order(source["qvel"].astype(np.float32))
    # The cache writer already stores target_qpos in the canonical Dex5 order;
    # only measured qpos/qvel use MuJoCo's thumb-leading order here.
    source_target = source["target_qpos"].astype(np.float32)
    target_hand = np.asarray(seed["hand_qpos"], dtype=np.float32)

    target_qpos = source_qpos.copy()
    target_qpos[:, :20] = target_hand[None, :]
    # This primitive is a contact recapture, so the object target is the
    # measured entry pose; rotation belongs to the later P3->P4 task.
    target_qvel = source_qvel.copy()
    source_policy_action = np.zeros((source_qpos.shape[0], 20), dtype=np.float32)
    bundle = {
        "source_qpos": source_qpos,
        "source_qvel": source_qvel,
        "source_target_qpos": source_target,
        "source_policy_action": source_policy_action,
        "source_episode_step": np.zeros(source_qpos.shape[0], dtype=np.int64),
        "source_teacher_actions": np.zeros((source_qpos.shape[0], 0, 20), dtype=np.float32),
        "target_qpos": target_qpos,
        "target_qvel": target_qvel,
        "phase_ids": np.zeros(source_qpos.shape[0], dtype=np.int64),
        "source_phase_offset_rad": source["rotation_phase"].astype(np.float32),
        "source_cache_t": source["t"].astype(np.float32),
        "source_contacts": source["contacts"].astype(str),
        "source_penetration_mm": source["max_penetration_mm"].astype(np.float32),
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUTPUT, **bundle)
    metadata = {
        "stage": "0.22-0.30s early pre-ring -> index/thumb recapture",
        "source_cache": str(CACHE.relative_to(ROOT)),
        "target_seed": str(TARGET_SEED.relative_to(ROOT)),
        "count": int(source_qpos.shape[0]),
        "source_time_s": [float(source["t"].min()), float(source["t"].max())],
        "source_phase_rad": [float(source["rotation_phase"].min()), float(source["rotation_phase"].max())],
        "source_contacts": sorted({str(value) for value in source["contacts"]}),
        "source_penetration_mm": {
            "min": float(source["max_penetration_mm"].min()),
            "mean": float(source["max_penetration_mm"].mean()),
            "max": float(source["max_penetration_mm"].max()),
        },
        "target_contacts": ["index", "thumb"],
        "rotation_policy": "hold measured pen pose; defer rotation to later primitive",
        "action_condition": "initial actuator action is zero; measured qpos/qvel and source target are preserved",
    }
    OUTPUT.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
