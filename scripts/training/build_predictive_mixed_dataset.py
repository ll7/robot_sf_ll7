#!/usr/bin/env python3
"""Build a mixed predictive-planner dataset with hard-case oversampling."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for dataset mixing."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-dataset", type=Path, required=True)
    parser.add_argument("--hardcase-dataset", type=Path, required=True)
    parser.add_argument("--hardcase-repeat", type=int, default=2)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("output/tmp/predictive_planner/datasets/predictive_rollouts_mixed_v1.npz"),
    )
    parser.add_argument("--shuffle-seed", type=int, default=42)
    return parser.parse_args()


def _load_npz(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load (state, target, mask) arrays from dataset npz."""
    raw = np.load(path)
    return (
        np.asarray(raw["state"], dtype=np.float32),
        np.asarray(raw["target"], dtype=np.float32),
        np.asarray(raw["mask"], dtype=np.float32),
    )


def main() -> int:
    """Create mixed dataset and write metadata sidecar."""
    args = parse_args()
    if int(args.hardcase_repeat) < 1:
        raise ValueError("--hardcase-repeat must be >= 1")

    base_state, base_target, base_mask = _load_npz(args.base_dataset)
    hard_state, hard_target, hard_mask = _load_npz(args.hardcase_dataset)

    for arr_base, arr_hard, name in [
        (base_state, hard_state, "state"),
        (base_target, hard_target, "target"),
        (base_mask, hard_mask, "mask"),
    ]:
        if arr_base.shape[1:] != arr_hard.shape[1:]:
            raise ValueError(
                f"Shape mismatch for {name}: base {arr_base.shape} vs hardcase {arr_hard.shape}"
            )

    hard_rep = int(args.hardcase_repeat)
    state = np.concatenate([base_state, np.repeat(hard_state, hard_rep, axis=0)], axis=0)
    target = np.concatenate([base_target, np.repeat(hard_target, hard_rep, axis=0)], axis=0)
    mask = np.concatenate([base_mask, np.repeat(hard_mask, hard_rep, axis=0)], axis=0)

    rng = np.random.default_rng(int(args.shuffle_seed))
    idx = np.arange(state.shape[0])
    rng.shuffle(idx)
    state = state[idx]
    target = target[idx]
    mask = mask[idx]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, state=state, target=target, mask=mask)

    summary = {
        "base_dataset": str(args.base_dataset),
        "hardcase_dataset": str(args.hardcase_dataset),
        "hardcase_repeat": hard_rep,
        "num_base_samples": int(base_state.shape[0]),
        "num_hardcase_samples": int(hard_state.shape[0]),
        "num_output_samples": int(state.shape[0]),
        "shuffle_seed": int(args.shuffle_seed),
        "output": str(args.output),
    }
    summary_path = args.output.with_suffix(".json")
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
