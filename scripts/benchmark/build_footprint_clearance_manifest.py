#!/usr/bin/env python3
"""Build the issue #3207 footprint / clearance-semantics diagnostic manifest.

Diagnostic only: enumerates a bounded robot-proxy / pedestrian-radius sweep and a
collision/near-miss threshold-sensitivity table. It runs no benchmark episodes and
changes no frozen-release collision/near-miss metric semantics.
"""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

from robot_sf.benchmark.clearance_semantics import (
    build_footprint_clearance_manifest,
    write_footprint_clearance_manifest,
)
from robot_sf.benchmark.fidelity_sensitivity import load_fidelity_sensitivity_config

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
    """Return repo-relative path when possible, otherwise the original path."""
    try:
        return path.relative_to(REPO_ROOT)
    except ValueError:
        return path


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="configs/research/fidelity_sensitivity_v1.yaml",
        help="Tracked fidelity-sensitivity config path (must carry a footprint_semantics block).",
    )
    parser.add_argument(
        "--out",
        default="output/fidelity_sensitivity_footprint_clearance",
        help="Output directory for footprint_clearance_manifest.json.",
    )
    return parser.parse_args()


def main() -> int:
    """CLI entry point."""
    args = parse_args()
    config = load_fidelity_sensitivity_config(REPO_ROOT / args.config)
    manifest = build_footprint_clearance_manifest(
        config,
        config_path=args.config,
        git_head=_git_head(),
    )
    manifest_path = write_footprint_clearance_manifest(manifest, REPO_ROOT / args.out)
    print(f"wrote footprint clearance diagnostic manifest: {_display_path(manifest_path)}")
    print(
        f"cells={manifest['cell_count']} "
        f"proxy_radius_sensitive_rows={manifest['proxy_radius_sensitive_row_count']}"
    )
    ts_rows = manifest.get("threshold_sensitive_row_count")
    if ts_rows is not None:
        print(f"threshold_sensitive_rows={ts_rows}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
