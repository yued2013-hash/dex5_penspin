from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = ROOT / "reports/m3_2_30_index_near_surface"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--bundle",
        type=Path,
        default=REPORT_DIR / "v9_current_approach49_prefill_to_braked50_bundle.npz",
    )
    parser.add_argument(
        "--audit",
        type=Path,
        default=REPORT_DIR / "v9_prefill_braked50_agent1536_bank4096.json",
    )
    parser.add_argument("--max-penetration-mm", type=float, default=0.325)
    parser.add_argument(
        "--output",
        type=Path,
        default=REPORT_DIR / "v9_approach49_prefill_braked50_lowpen325_bundle.npz",
    )
    args = parser.parse_args()

    audit = json.loads(args.audit.read_text(encoding="utf-8"))
    selected = np.asarray(
        [
            int(bank_id)
            for bank_id, metrics in audit["per_reset_bank"].items()
            if metrics["max_penetration_mm"] <= args.max_penetration_mm
        ],
        dtype=np.int64,
    )
    with np.load(args.bundle) as data:
        payload = {name: data[name][selected] for name in data.files}
    payload["original_bank_ids"] = selected
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **payload)

    episodes = sum(
        audit["per_reset_bank"][str(bank_id)]["episodes"] for bank_id in selected
    )
    successes = sum(
        audit["per_reset_bank"][str(bank_id)]["successes"] for bank_id in selected
    )
    print(f"output={args.output}")
    print(f"selected_banks={len(selected)}/{len(audit['per_reset_bank'])}")
    print(f"audit_episodes={episodes}")
    print(f"audit_success_rate={successes / max(episodes, 1):.6f}")


if __name__ == "__main__":
    main()
