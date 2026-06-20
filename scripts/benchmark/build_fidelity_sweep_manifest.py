#!/usr/bin/env python3
"""Build an issue #3207 dry-run fidelity sweep manifest."""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

from robot_sf.benchmark.fidelity_sensitivity import load_fidelity_sensitivity_config
from robot_sf.benchmark.fidelity_sweep_manifest import (
    ManifestOptions,
    build_fidelity_sweep_manifest,
    write_fidelity_sweep_manifest,
)

REPO_ROOT = Path(__file__).resolve().parents[2]


def _git_head() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=False,
            timeout=5,
        )
    except (FileNotFoundError, subprocess.SubprocessError):
        return "unknown"
    return result.stdout.strip() if result.returncode == 0 else "unknown"


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="configs/research/fidelity_sensitivity_v1.yaml",
        help="Tracked fidelity-sensitivity config path.",
    )
    parser.add_argument(
        "--out",
        default="output/fidelity_sensitivity_manifest",
        help="Output directory for fidelity_sweep_manifest.json.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        required=True,
        help="Required acknowledgement that this only builds a manifest and runs no sweep.",
    )
    return parser.parse_args()


def main() -> int:
    """CLI entry point."""
    args = parse_args()
    config = load_fidelity_sensitivity_config(REPO_ROOT / args.config)
    manifest = build_fidelity_sweep_manifest(
        config,
        options=ManifestOptions(
            config_path=args.config,
            git_head=_git_head(),
            dry_run=args.dry_run,
        ),
    )
    manifest_path = write_fidelity_sweep_manifest(manifest, REPO_ROOT / args.out)
    print(f"wrote dry-run fidelity sweep manifest: {manifest_path.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
