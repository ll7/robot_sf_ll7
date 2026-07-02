"""Run or dry-run the issue #4018 density-curriculum comparator pair."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

from scripts.training.train_ppo import load_expert_training_config


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--curriculum-config", required=True, type=Path)
    parser.add_argument("--baseline-config", required=True, type=Path)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("output/issue_4018_density_curriculum")
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def _summary(config_path: Path) -> dict[str, Any]:
    cfg = load_expert_training_config(config_path)
    return {
        "path": str(config_path),
        "policy_id": cfg.policy_id,
        "total_timesteps": cfg.total_timesteps,
        "density_curriculum_enabled": bool(cfg.density_curriculum.get("enabled", False)),
    }


def main() -> int:
    """Validate or run the matched curriculum and fixed-density config pair.

    Returns:
        Process exit code.
    """
    args = _parse_args()
    curriculum = _summary(args.curriculum_config)
    baseline = _summary(args.baseline_config)
    if curriculum["total_timesteps"] != baseline["total_timesteps"]:
        raise ValueError("curriculum and fixed-density configs must use matched total_timesteps.")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema_version": "density_curriculum_comparison.v1",
        "issue": "ll7/robot_sf_ll7#4018",
        "claim_boundary": "diagnostic harness only; no benchmark or training-result claim",
        "dry_run": bool(args.dry_run),
        "curriculum": curriculum,
        "baseline": baseline,
    }
    manifest_path = args.output_dir / "comparison_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"wrote {manifest_path}")

    if args.dry_run:
        return 0

    for config_path in (args.curriculum_config, args.baseline_config):
        subprocess.run(
            [
                sys.executable,
                "scripts/training/train_ppo.py",
                "--config",
                str(config_path),
            ],
            check=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
