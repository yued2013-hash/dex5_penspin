from __future__ import annotations

import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "reports/m3_2_30_index_near_surface"
SOURCE = REPORT / "v9_backward_suffix35_to_braked50_physx_bundle.npz"
OUTPUT = REPORT / "v9_backward_suffix35_focusx3_to_braked50_physx_bundle.npz"


def main() -> None:
    with np.load(SOURCE) as data:
        payload = {key: np.asarray(data[key]).copy() for key in data.files}
    focus = payload["curriculum_stage_deg"] == 35.0
    for key, values in tuple(payload.items()):
        payload[key] = np.concatenate((values, values[focus], values[focus]), axis=0)
    np.savez_compressed(OUTPUT, **payload)
    stages, counts = np.unique(payload["curriculum_stage_deg"], return_counts=True)
    metadata = {
        "output": str(OUTPUT.relative_to(ROOT)),
        "source": str(SOURCE.relative_to(ROOT)),
        "total_rows": len(payload["source_qpos"]),
        "stage_counts": {str(int(stage)): int(count) for stage, count in zip(stages, counts)},
    }
    OUTPUT.with_suffix(".json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
