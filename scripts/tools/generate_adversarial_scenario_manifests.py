#!/usr/bin/env python3
"""Generate adversarial_scenario_manifest.v1 candidates from a search space.

Usage:
    uv run python scripts/tools/generate_adversarial_scenario_manifests.py \
        --search-space configs/adversarial/crossing_ttc_space.yaml \
        --scenario-template configs/scenarios/templates/crossing_ttc.yaml \
        --count 16 --seed 42 --output-dir output/adversarial_manifests
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

import yaml

from robot_sf.adversarial.config import SearchSpaceConfig
from robot_sf.adversarial.scenario_manifest import (
    SourceLineage,
    generate_manifests,
    write_manifest_yaml,
)


def _load_template_info(template_path: Path) -> SourceLineage:
    """Extract source lineage from a scenario template YAML."""
    raw = yaml.safe_load(template_path.read_text(encoding="utf-8")) or {}
    scenarios = raw.get("scenarios", [])
    map_id: str | None = None
    scenario_name: str | None = None
    if isinstance(scenarios, list) and scenarios:
        first = scenarios[0]
    elif isinstance(scenarios, dict) and scenarios:
        first = next(iter(scenarios.values()), None)
    else:
        first = None
    if isinstance(first, dict):
        map_id = str(first.get("map_id", "")) or None
        scenario_name = str(first.get("name", "")) or None
    return SourceLineage(
        scenario_template=template_path.name,
        config_path=str(template_path),
        search_space_path="",
        map_id=map_id,
        scenario_name=scenario_name,
    )


def main(argv: list[str] | None = None) -> int:
    """CLI entry point: generate and validate candidate manifests."""
    parser = argparse.ArgumentParser(
        description="Generate adversarial_scenario_manifest.v1 candidates"
    )
    parser.add_argument(
        "--search-space",
        type=Path,
        required=True,
        help="Path to search-space YAML config",
    )
    parser.add_argument(
        "--scenario-template",
        type=Path,
        required=True,
        help="Path to scenario template YAML",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=16,
        help="Number of candidates to generate (default: 16)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Generator seed (default: 42)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output/adversarial_manifests"),
        help="Output directory (default: output/adversarial_manifests)",
    )
    parser.add_argument(
        "--generator-family",
        type=str,
        default="random",
        help="Generator family label (default: random)",
    )
    args = parser.parse_args(argv)

    if not args.search_space.exists():
        print(f"Error: search space not found: {args.search_space}", file=sys.stderr)
        return 1
    if not args.scenario_template.exists():
        print(f"Error: scenario template not found: {args.scenario_template}", file=sys.stderr)
        return 1
    if args.count < 1:
        print("Error: count must be >= 1", file=sys.stderr)
        return 1

    search_space = SearchSpaceConfig.from_file(args.search_space)
    source = _load_template_info(args.scenario_template)
    source = replace(
        source, search_space=args.search_space.name, search_space_path=str(args.search_space)
    )

    manifests, summary = generate_manifests(
        search_space,
        seed=args.seed,
        count=args.count,
        source=source,
        generator_family=args.generator_family,
    )
    summary = {
        "schema_version": "adversarial_scenario_manifest_generation_summary.v1",
        "source": source.to_dict(),
        "generator": {
            "family": args.generator_family,
            "generator_id": "RandomCandidateSampler",
            "seed": int(args.seed),
        },
        **summary,
    }

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for manifest in manifests:
        index = manifest.generator.candidate_index if manifest.generator else 0
        path = output_dir / f"candidate_{index:04d}.yaml"
        write_manifest_yaml(manifest, path)

    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    print(f"Generated {summary['total_candidates']} manifests in {output_dir}")
    print(
        f"  valid: {summary['valid']}, invalid: {summary['invalid']}, "
        f"degenerate: {summary['degenerate']}"
    )
    if summary["rejection_reasons"]:
        print(f"  rejection reasons: {summary['rejection_reasons']}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
