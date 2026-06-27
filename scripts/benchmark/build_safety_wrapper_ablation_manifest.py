#!/usr/bin/env python3
"""Build an issue #3501 dry-run safety-wrapper factorial-ablation manifest.

This only enumerates and checks the ``planner x {wrapper off, wrapper on}`` design; it runs
no benchmark episodes, tunes no thresholds, and makes no mitigation-effectiveness claim.
"""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

from robot_sf.benchmark.safety_wrapper_ablation_manifest import (
    ManifestOptions,
    build_safety_wrapper_ablation_manifest,
    load_safety_wrapper_ablation_config,
    write_safety_wrapper_ablation_manifest,
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


def _display_path(path: Path) -> Path:
    """Return repo-relative path when possible, otherwise the original path.

    Returns:
        Repo-relative path or the original path.
    """
    try:
        return path.relative_to(REPO_ROOT)
    except ValueError:
        return path


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed argument namespace.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="configs/research/safety_wrapper_ablation_v1.yaml",
        help="Tracked safety-wrapper ablation config path.",
    )
    parser.add_argument(
        "--out",
        default="output/safety_wrapper_ablation_manifest",
        help="Output directory for safety_wrapper_ablation_manifest.json.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        required=True,
        help="Required acknowledgement that this only builds a manifest and runs no ablation.",
    )
    return parser.parse_args()


def main() -> int:
    """CLI entry point.

    Returns:
        Process exit code.
    """
    args = parse_args()
    config = load_safety_wrapper_ablation_config(REPO_ROOT / args.config)
    manifest = build_safety_wrapper_ablation_manifest(
        config,
        options=ManifestOptions(
            config_path=args.config,
            git_head=_git_head(),
            dry_run=args.dry_run,
        ),
    )
    manifest_path = write_safety_wrapper_ablation_manifest(manifest, REPO_ROOT / args.out)
    check = manifest["factorial_check"]
    print(f"wrote dry-run safety-wrapper ablation manifest: {_display_path(manifest_path)}")
    print(
        f"factorial check: complete={check['complete']} "
        f"cells={manifest['cell_count']} seeds_per_cell={check['seeds_per_cell']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
