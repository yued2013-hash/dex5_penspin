from __future__ import annotations

import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
REPORTS = ROOT / "dex5_penspin/reports"
OUTPUT = REPORTS / "m3_2_30_index_near_surface/isaac_recapture_p3_bundle.npz"
P3_PLANAR_TARGET_DEG = -10.0


def load(relative_path: str) -> dict[str, np.ndarray]:
    with np.load(REPORTS / relative_path, allow_pickle=True) as data:
        return {key: np.asarray(data[key]) for key in data.files}


def mujoco_state_to_dex5_order(values: np.ndarray) -> np.ndarray:
    """Move MuJoCo's leading thumb joints behind the four long-finger groups."""
    values = np.asarray(values).copy()
    values[..., :20] = np.concatenate((values[..., 4:20], values[..., 0:4]), axis=-1)
    return values


def nearest_rows(source_t: np.ndarray, target: dict[str, np.ndarray]) -> np.ndarray:
    target_t = np.asarray(target["cache_t"], dtype=np.float64)
    return np.asarray([np.argmin(np.abs(target_t - value)) for value in source_t], dtype=np.int64)


def quat_multiply(lhs: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    lw, lx, ly, lz = np.moveaxis(lhs, -1, 0)
    rw, rx, ry, rz = np.moveaxis(rhs, -1, 0)
    return np.stack(
        (
            lw * rw - lx * rx - ly * ry - lz * rz,
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
        ),
        axis=-1,
    )


def main() -> None:
    near = [
        load("m3_2_30_index_near_surface/near_016_019/index_near_surface_cache.npz"),
        load("m3_2_30_index_near_surface/near_019_022/index_near_surface_cache.npz"),
    ]
    contact = [
        load("m3_2_30_index_near_surface/contact_016_019/index_contact_cache.npz"),
        load("m3_2_30_index_near_surface/contact_019_022/index_contact_cache.npz"),
    ]
    recapture = [
        load(
            "m3_2_28_recapture_split_b23_entry/recapture_016_019_projected/"
            "index_thumb_recapture_projected_cache.npz"
        ),
        load(
            "m3_2_28_recapture_split_b23_entry/recapture_019_022_projected/"
            "index_thumb_recapture_projected_cache.npz"
        ),
    ]
    ring_touch = load(
        "m3_2_28_recapture_split_b23_entry/projected_to_b23_entry_ring_touch/"
        "b23_entry_ring_touch_cache.npz"
    )
    p3_entry = load(
        "manus_mujoco_presets/continuous_p1_p2_p3_m3_2_34/continuous_P3_entry_arrival.npz"
    )
    physical_recapture_path = REPORTS / "m3_2_30_index_near_surface/isaac_physics_recapture_states.npz"
    physical_recapture = None
    if physical_recapture_path.exists():
        with np.load(physical_recapture_path) as data:
            physical_recapture = {key: np.asarray(data[key]) for key in data.files}
    physical_ring_path = REPORTS / "m3_2_30_index_near_surface/isaac_physics_ring_touch_states.npz"
    physical_ring = None
    if physical_ring_path.exists():
        with np.load(physical_ring_path) as data:
            physical_ring = {key: np.asarray(data[key]) for key in data.files}

    source_qpos: list[np.ndarray] = []
    source_qvel: list[np.ndarray] = []
    source_target: list[np.ndarray] = []
    target_qpos: list[np.ndarray] = []
    target_qvel: list[np.ndarray] = []
    phase_ids: list[np.ndarray] = []

    def append_phase(
        source: dict[str, np.ndarray],
        target: dict[str, np.ndarray],
        target_ids: np.ndarray,
        phase: int,
        preserve_source_pen: bool = False,
    ):
        count = source["qpos"].shape[0]
        converted_source_qpos = mujoco_state_to_dex5_order(np.asarray(source["qpos"], dtype=np.float32))
        converted_target_qpos = mujoco_state_to_dex5_order(
            np.asarray(target["qpos"][target_ids], dtype=np.float32)
        )
        if preserve_source_pen:
            converted_target_qpos[:, 20:27] = converted_source_qpos[:, 20:27]
        source_qpos.append(converted_source_qpos)
        source_qvel.append(mujoco_state_to_dex5_order(np.asarray(source["qvel"], dtype=np.float32)))
        source_target.append(np.asarray(source["target_qpos"], dtype=np.float32))
        target_qpos.append(converted_target_qpos)
        target_qvel.append(
            mujoco_state_to_dex5_order(np.asarray(target["qvel"][target_ids], dtype=np.float32))
        )
        phase_ids.append(np.full(count, phase, dtype=np.int64))

    for source, target in zip(near, contact, strict=True):
        append_phase(source, target, np.arange(source["qpos"].shape[0]), 0)

    for source, target in zip(contact, recapture, strict=True):
        append_phase(
            source,
            target,
            np.arange(source["qpos"].shape[0]),
            1,
            preserve_source_pen=True,
        )

    if physical_recapture is None:
        for source in recapture:
            ids = nearest_rows(np.asarray(source["cache_t"]), ring_touch)
            append_phase(source, ring_touch, ids, 2)
    else:
        recapture_hand_rows: list[np.ndarray] = []
        ring_hand_rows: list[np.ndarray] = []
        for source in recapture:
            ids = nearest_rows(np.asarray(source["cache_t"]), ring_touch)
            recapture_hand_rows.append(
                mujoco_state_to_dex5_order(np.asarray(source["qpos"], dtype=np.float32))[:, :20]
            )
            ring_hand_rows.append(
                mujoco_state_to_dex5_order(np.asarray(ring_touch["qpos"][ids], dtype=np.float32))[:, :20]
            )
        ring_delta = np.mean(
            np.concatenate(ring_hand_rows) - np.concatenate(recapture_hand_rows), axis=0
        )
        physical_qpos = np.asarray(physical_recapture["qpos"], dtype=np.float32)
        physical_qvel = np.asarray(physical_recapture["qvel"], dtype=np.float32)
        physical_target = physical_qpos.copy()
        physical_target[:, :20] += ring_delta
        source_qpos.append(physical_qpos)
        source_qvel.append(physical_qvel)
        source_target.append(np.asarray(physical_recapture["target_qpos"], dtype=np.float32))
        target_qpos.append(physical_target)
        target_qvel.append(physical_qvel.copy())
        phase_ids.append(np.full(physical_qpos.shape[0], 2, dtype=np.int64))

    if physical_ring is None:
        p3_ids = np.zeros(ring_touch["qpos"].shape[0], dtype=np.int64)
        append_phase(ring_touch, p3_entry, p3_ids, 3)
    else:
        old_ring_qpos = mujoco_state_to_dex5_order(np.asarray(ring_touch["qpos"], dtype=np.float32))
        old_p3_qpos = mujoco_state_to_dex5_order(np.asarray(p3_entry["qpos"], dtype=np.float32))[0]
        hand_delta = old_p3_qpos[:20] - np.mean(old_ring_qpos[:, :20], axis=0)
        half_angle = np.deg2rad(P3_PLANAR_TARGET_DEG) / 2.0
        planar_quat_delta = np.asarray(
            (np.cos(half_angle), 0.0, 0.0, np.sin(half_angle)), dtype=np.float32
        )

        physical_qpos = np.asarray(physical_ring["qpos"], dtype=np.float32)
        physical_qvel = np.asarray(physical_ring["qvel"], dtype=np.float32)
        physical_target = physical_qpos.copy()
        physical_target[:, :20] += hand_delta
        # P2->P3 is a short dynamic deflection task. Keep the pen center near the
        # physical B23 reset and rotate its long axis by a measurable planar target.
        physical_target[:, 23:27] = quat_multiply(planar_quat_delta, physical_qpos[:, 23:27])
        physical_target[:, 23:27] /= np.linalg.norm(physical_target[:, 23:27], axis=-1, keepdims=True)
        source_qpos.append(physical_qpos)
        source_qvel.append(physical_qvel)
        source_target.append(np.asarray(physical_ring["target_qpos"], dtype=np.float32))
        target_qpos.append(physical_target)
        target_qvel.append(physical_qvel.copy())
        phase_ids.append(np.full(physical_qpos.shape[0], 3, dtype=np.int64))

    bundle = {
        "source_qpos": np.concatenate(source_qpos),
        "source_qvel": np.concatenate(source_qvel),
        "source_target_qpos": np.concatenate(source_target),
        "target_qpos": np.concatenate(target_qpos),
        "target_qvel": np.concatenate(target_qvel),
        "phase_ids": np.concatenate(phase_ids),
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUTPUT, **bundle)
    metadata = {
        "stage": "Isaac physical recapture to P3 curriculum",
        "bundle": str(OUTPUT.relative_to(ROOT)),
        "count": int(bundle["source_qpos"].shape[0]),
        "phases": {
            "0": "index_near_surface -> index_contact",
            "1": "index_contact -> projected_index_thumb_recapture",
            "2": "projected_recapture -> B23_entry/ring_touch",
            "3": "B23_entry/ring_touch -> dynamic_P3_entry",
        },
        "phase_counts": {
            str(phase): int(np.sum(bundle["phase_ids"] == phase)) for phase in range(4)
        },
        "p3_planar_target_deg": P3_PLANAR_TARGET_DEG,
    }
    OUTPUT.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
