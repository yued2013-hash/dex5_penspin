from __future__ import annotations

import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
REPORT = ROOT / "dex5_penspin/reports/m3_2_30_index_near_surface"
OUT_NPZ = REPORT / "isaac_first_leg_bundle.npz"
OUT_JSON = REPORT / "isaac_first_leg_bundle.json"


WINDOWS = {
    "016_019": {
        "source_cache": ROOT
        / "dex5_penspin/reports/manus_mujoco_presets/short_primitive_reward_m3_2_23/dynamic_state_cache/"
        / "earlier_pre_ring_016_019_for_B12b_to_B23.npz",
        "near_cache": REPORT / "near_016_019/index_near_surface_cache.npz",
        "near_seed": REPORT / "near_016_019/index_near_surface_seed.json",
    },
    "019_022": {
        "source_cache": ROOT
        / "dex5_penspin/reports/manus_mujoco_presets/short_primitive_reward_m3_2_23/dynamic_state_cache/"
        / "earlier_pre_ring_019_022_for_B12b_to_B23.npz",
        "near_cache": REPORT / "near_019_022/index_near_surface_cache.npz",
        "near_seed": REPORT / "near_019_022/index_near_surface_seed.json",
    },
}


def load_npz(path: Path) -> dict[str, np.ndarray]:
    return dict(np.load(path, allow_pickle=True))


def mujoco_state_to_dex5_order(values: np.ndarray) -> np.ndarray:
    """Move MuJoCo's leading thumb joints behind the four long-finger groups."""
    values = np.asarray(values).copy()
    values[..., :20] = np.concatenate((values[..., 4:20], values[..., 0:4]), axis=-1)
    return values


def main() -> None:
    arrays: dict[str, np.ndarray] = {}
    metadata: dict[str, object] = {
        "stage": "M3.2.30 Isaac first-leg bundle",
        "thresholds": {
            "index_tip_dist_final_m": 0.014,
            "minimum_survival_s": 0.16,
        },
        "recommended_task": "approach-only index_near_surface",
        "notes": [
            "Use source cache as reset distribution.",
            "Use near cache/seed as teacher and target reference.",
            "Do not require index contact in this first primitive.",
            "Do not start with near->contact; MuJoCo active smoke still fails.",
        ],
        "windows": {},
    }

    for name, paths in WINDOWS.items():
        source = load_npz(paths["source_cache"])
        near = load_npz(paths["near_cache"])
        seed = json.loads(paths["near_seed"].read_text(encoding="utf-8"))

        arrays[f"{name}_source_qpos"] = mujoco_state_to_dex5_order(
            np.asarray(source["qpos"], dtype=np.float64)
        )
        arrays[f"{name}_source_qvel"] = mujoco_state_to_dex5_order(
            np.asarray(source["qvel"], dtype=np.float64)
        )
        arrays[f"{name}_source_target_qpos"] = np.asarray(source["target_qpos"], dtype=np.float64)
        arrays[f"{name}_source_rotation_phase"] = np.asarray(source.get("rotation_phase", np.zeros(source["qpos"].shape[0])), dtype=np.float64)
        arrays[f"{name}_source_t"] = np.asarray(source.get("t", source.get("cache_t", np.arange(source["qpos"].shape[0]))), dtype=np.float64)

        arrays[f"{name}_near_qpos"] = mujoco_state_to_dex5_order(
            np.asarray(near["qpos"], dtype=np.float64)
        )
        arrays[f"{name}_near_qvel"] = mujoco_state_to_dex5_order(
            np.asarray(near["qvel"], dtype=np.float64)
        )
        arrays[f"{name}_near_target_qpos"] = np.asarray(near["target_qpos"], dtype=np.float64)
        arrays[f"{name}_near_rotation_phase"] = np.asarray(near.get("rotation_phase", np.zeros(near["qpos"].shape[0])), dtype=np.float64)
        arrays[f"{name}_near_t"] = np.asarray(near.get("cache_t", np.arange(near["qpos"].shape[0])), dtype=np.float64)
        arrays[f"{name}_target_hand_qpos"] = np.asarray(seed["hand_qpos"], dtype=np.float64)
        arrays[f"{name}_target_pen_pos"] = np.asarray(seed["pen_pos"], dtype=np.float64)
        arrays[f"{name}_target_pen_quat"] = np.asarray(seed["pen_quat"], dtype=np.float64)

        metadata["windows"][name] = {
            "source_cache": str(paths["source_cache"].relative_to(ROOT)),
            "near_cache": str(paths["near_cache"].relative_to(ROOT)),
            "near_seed": str(paths["near_seed"].relative_to(ROOT)),
            "source_states": int(arrays[f"{name}_source_qpos"].shape[0]),
            "near_states": int(arrays[f"{name}_near_qpos"].shape[0]),
            "target_seed_summary": seed.get("summary", ""),
            "model_path": seed.get("model_path", ""),
        }

    np.savez_compressed(OUT_NPZ, **arrays)
    OUT_JSON.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"npz": str(OUT_NPZ), "json": str(OUT_JSON)}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
