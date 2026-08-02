from __future__ import annotations

import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "reports/m3_2_30_index_near_surface"
OUT = REPORT / "canonical_p1_p5_manifest_v2_candidate.json"


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    current = load(REPORT / "canonical_p1_p5_manifest.json")
    phases = {int(row["phase"]): row for row in current["phases"]}
    p1 = phases[1]
    p2_ring_under = load(
        ROOT
        / "reports/manus_mujoco_presets/p2_bottom_cradle_b23_thumb_pass_m3_2_31/"
        "P2_ring_bottom_thumb_cradle_entry_seed.json"
    )
    p5_pinch = load(
        ROOT
        / "reports/manus_mujoco_presets/rebuilt_p4_p5_transitions_m3_2_16/"
        "P5_rebuilt_thumb_index_capture_seed.json"
    )

    p1_hand = np.asarray(p1["hand_qpos"], dtype=np.float64)
    ring_under_hand = np.asarray(p2_ring_under["hand_qpos"], dtype=np.float64)
    p5_hand = np.asarray(p5_pinch["hand_qpos"], dtype=np.float64)

    p2_hand = p1_hand.copy()
    p2_hand[8:12] = ring_under_hand[8:12]
    p2_half = p1_hand.copy()
    p2_half[8:12] = 0.5 * (p1_hand[8:12] + p2_hand[8:12])

    p5_entry_hand = p5_hand.copy()
    p5_entry_hand[4:16] = p1_hand[4:16]

    phases[2] = {
        **phases[2],
        "name": "ring_under_transfer_with_p1_triad_hold",
        "summary": "Pen and thumb/index/middle remain at P1 while ring transfers below.",
        "hand_qpos": p2_hand.tolist(),
        "pen_pos": p1["pen_pos"],
        "pen_quat": p1["pen_quat"],
        "contact_set": ["thumb", "index", "middle"],
        "placement_support": ["thumb", "index", "middle"],
        "optional_contacts": ["ring"],
        "source_validation_status": "v2_candidate_not_hold_validated",
        "transition_keyframes": [
            {
                "name": "P2a_ring_transfer_half",
                "hand_qpos": p2_half.tolist(),
                "pen_pos": p1["pen_pos"],
                "pen_quat": p1["pen_quat"],
            },
            {
                "name": "P2b_ring_under_arrival",
                "hand_qpos": p2_hand.tolist(),
                "pen_pos": p1["pen_pos"],
                "pen_quat": p1["pen_quat"],
            },
        ],
    }

    p5_pen_pos = p5_pinch.get("pen_pos", phases[5]["pen_pos"])
    p5_pen_quat = p5_pinch.get("pen_quat", phases[5]["pen_quat"])
    phases[5] = {
        **phases[5],
        "name": "thumb_index_pinch_middle_ring_return",
        "summary": "Thumb/index pinch the pen while middle/ring retract from below toward P1 rear positions.",
        "hand_qpos": p5_hand.tolist(),
        "pen_pos": p5_pen_pos,
        "pen_quat": p5_pen_quat,
        "contact_set": ["thumb", "index"],
        "placement_support": ["thumb", "index"],
        "moving_fingers": ["middle", "ring"],
        "source_validation_status": "v2_candidate_not_hold_validated",
        "transition_keyframes": [
            {
                "name": "P5a_thumb_index_pinch",
                "hand_qpos": p5_hand.tolist(),
                "pen_pos": p5_pen_pos,
                "pen_quat": p5_pen_quat,
            },
            {
                "name": "P5b_middle_ring_rear_P1_entry",
                "hand_qpos": p5_entry_hand.tolist(),
                "pen_pos": p5_pen_pos,
                "pen_quat": p5_pen_quat,
            },
        ],
    }

    payload = {
        "stage": "canonical P1-P5 manifest v2 candidate",
        "source": str((REPORT / "canonical_p1_p5_manifest.json").relative_to(ROOT)),
        "note": "P2 and P5 semantics corrected from user photo review; visual confirmation required before promotion.",
        "phases": [phases[index] for index in range(1, 6)],
    }
    OUT.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(OUT)


if __name__ == "__main__":
    main()
