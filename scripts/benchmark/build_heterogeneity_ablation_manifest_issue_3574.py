#!/usr/bin/env python3
"""Build the issue #3574 mean-matched heterogeneity dry-run manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import yaml

from robot_sf.benchmark.campaign_logging import (
    add_campaign_logging_argument,
    configure_campaign_logging,
)

REPO_ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="configs/benchmarks/issue_3574_mean_matched_harness_smoke.yaml",
        help="Tracked issue #3574 dry-run harness config.",
    )
    parser.add_argument(
        "--output",
        default="output/issue_3574_mean_matched_harness/manifest.json",
        help="Manifest JSON output path.",
    )
    parser.add_argument(
        "--legacy-map",
        help="Explicit map used to validate legacy inline scenarios without map_file fields.",
    )
    add_campaign_logging_argument(parser)
    return parser.parse_args()


def _load_yaml(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a YAML mapping")
    return payload


def _display_path(path: Path) -> Path:
    try:
        return path.relative_to(REPO_ROOT)
    except ValueError:
        return path


def main() -> int:
    """Build and write the manifest."""

    args = parse_args()
    configure_campaign_logging(debug=args.debug)
    config_path = REPO_ROOT / args.config
    output_path = REPO_ROOT / args.output
    from robot_sf.benchmark.heterogeneous_population_ablation import (
        build_mean_matched_harness_manifest,
    )
    from robot_sf.benchmark.heterogeneous_population_ablation_runner import (
        assert_manifest_spawn_realizable,
    )

    manifest = build_mean_matched_harness_manifest(
        _load_yaml(config_path),
        config_path=args.config,
    )
    legacy_map_path = None if args.legacy_map is None else REPO_ROOT / args.legacy_map
    assert_manifest_spawn_realizable(
        manifest["manifest_rows"],
        scenario_path=config_path,
        map_path=legacy_map_path,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(f"wrote issue #3574 mean-matched heterogeneity manifest: {_display_path(output_path)}")
    print(f"status: {manifest['status']}; rows: {manifest['row_count']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
