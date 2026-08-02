from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BANK = (
    PROJECT_ROOT
    / "reports/m3_2_30_index_near_surface/p1_p2_reset_bank/"
    "P1_P2_reset_bank_phase1最终validated自复验_validated.npz"
)
DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT / "reports/m3_2_30_index_near_surface/p1_p2_reset_bank"
)


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def split_phase_ids(
    ids: np.ndarray, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    shuffled = ids.copy()
    rng.shuffle(shuffled)
    train_end = int(np.floor(0.80 * len(shuffled)))
    validation_end = train_end + int(np.floor(0.10 * len(shuffled)))
    return (
        np.sort(shuffled[:train_end]),
        np.sort(shuffled[train_end:validation_end]),
        np.sort(shuffled[validation_end:]),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Freeze and stratify the validated P1/P2 reset bank.")
    parser.add_argument("--bank", type=Path, default=DEFAULT_BANK)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--seed", type=int, default=20260722)
    args = parser.parse_args()

    bank = resolve(args.bank).resolve()
    output_dir = resolve(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    with np.load(bank) as data:
        phase_ids = np.asarray(data["phase_ids"], dtype=np.int64)

    rng = np.random.default_rng(args.seed)
    split_by_phase: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for phase in (1, 2):
        ids = np.flatnonzero(phase_ids == phase).astype(np.int64)
        if ids.size == 0:
            raise ValueError(f"Frozen bank has no phase {phase} rows")
        split_by_phase[phase] = split_phase_ids(ids, rng)

    split_names = ("train", "validation", "test")
    filenames = {
        "train": "P1_P2_train_ids.npy",
        "validation": "P1_P2_validation_ids.npy",
        "test": "P1_P2_test_ids.npy",
    }
    split_records: dict[str, dict[str, object]] = {}
    all_split_ids: list[np.ndarray] = []
    for split_index, split_name in enumerate(split_names):
        p1_ids = split_by_phase[1][split_index]
        p2_ids = split_by_phase[2][split_index]
        combined = np.sort(np.concatenate((p1_ids, p2_ids))).astype(np.int64)
        output = output_dir / filenames[split_name]
        np.save(output, combined, allow_pickle=False)
        all_split_ids.append(combined)
        split_records[split_name] = {
            "path": str(output.relative_to(PROJECT_ROOT)),
            "sha256": sha256(output),
            "rows": int(combined.size),
            "phase_counts": {"P1": int(p1_ids.size), "P2": int(p2_ids.size)},
            "bank_id_min": int(combined.min()),
            "bank_id_max": int(combined.max()),
        }

    concatenated = np.concatenate(all_split_ids)
    if np.unique(concatenated).size != phase_ids.size:
        raise RuntimeError("Train/validation/test IDs are not disjoint and exhaustive")
    if not np.array_equal(np.sort(concatenated), np.arange(phase_ids.size, dtype=np.int64)):
        raise RuntimeError("Train/validation/test IDs do not cover every bank row exactly once")

    manifest = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "frozen_bank": {
            "path": str(bank.relative_to(PROJECT_ROOT)),
            "sha256": sha256(bank),
            "rows": int(phase_ids.size),
            "phase_counts": {
                "P1": int(np.count_nonzero(phase_ids == 1)),
                "P2": int(np.count_nonzero(phase_ids == 2)),
            },
        },
        "bank_id_semantics": "zero-based row index in frozen_bank; IDs remain immutable",
        "split_seed": args.seed,
        "split_policy": "stratified by P1/P2, 80/10/10 using floor for train and validation",
        "leakage_policy": {
            "train": "PPO reset, reward tuning, BC and teacher generation allowed",
            "validation": "model selection and threshold diagnostics only",
            "test": "evaluation only; forbidden for PPO reset, reward tuning and teacher generation",
        },
        "splits": split_records,
    }
    manifest_path = output_dir / "P1_P2_reset_bank_phase1_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("P1_P2_FREEZE=" + json.dumps(manifest, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
