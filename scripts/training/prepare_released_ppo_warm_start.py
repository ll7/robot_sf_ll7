"""Prepare a fresh canonical-trainer resume ZIP from a released PPO (no training)."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from robot_sf.training.released_warm_start import load_warm_start, released_checkpoint


def main() -> None:
    """Materialize a weights-only warm start and its durable-source provenance."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-id",
        required=True,
        choices=["ppo_expert_issue_791_reward_curriculum_eval_aligned_large_capacity_20260417"],
    )
    parser.add_argument("--init-seed", type=int, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.is_absolute() or ".." in args.out.parts or args.out.suffix != ".zip":
        parser.error("--out must be a repository-relative ZIP path")
    if args.out.exists():
        parser.error("--out already exists; choose an isolated run path")
    source, provenance = released_checkpoint(args.model_id)
    model, _ = load_warm_start(source, init_seed=args.init_seed)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(args.out))
    provenance.update(
        {
            "prepared_artifact_path": args.out.as_posix(),
            "prepared_sha256": hashlib.sha256(args.out.read_bytes()).hexdigest(),
            "init_seed": args.init_seed,
            "observation_normalization": "none; source uses raw dict observations",
            "reset": ["num_timesteps", "updates", "episodes", "optimizer moments"],
        }
    )
    args.out.with_suffix(".json").write_text(json.dumps(provenance, indent=2) + "\n")


if __name__ == "__main__":
    main()
