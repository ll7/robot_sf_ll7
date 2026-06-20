#!/usr/bin/env python3
"""Build compact no-result reports for pedestrian archetype compositions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import yaml

from robot_sf.ped_npc.ped_archetypes import (
    build_archetype_population_report,
    load_archetypes,
)

REPO_ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="configs/research/pedestrian_archetypes_v1.yaml",
        help="Pedestrian archetype registry YAML.",
    )
    parser.add_argument(
        "--output-dir",
        default="docs/context/evidence/issue_3206_pedestrian_archetype_reporting_2026-06-20",
        help="Output directory for compact reports.",
    )
    parser.add_argument("--population-size", type=int, default=30)
    parser.add_argument("--initial-speed", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=3206)
    return parser.parse_args()


def _load_example_compositions(path: Path) -> dict[str, dict[str, float]]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    compositions = payload.get("example_compositions")
    if not isinstance(compositions, dict) or not compositions:
        raise ValueError(f"{path} must define non-empty example_compositions")
    out: dict[str, dict[str, float]] = {}
    for name, composition in compositions.items():
        if not isinstance(composition, dict):
            raise ValueError(f"example composition {name!r} must be a mapping")
        out[str(name)] = {str(k): float(v) for k, v in composition.items()}
    return out


def _format_markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# Issue #3206 Pedestrian Archetype Reporting Packet",
        "",
        f"- Status: `{summary['status']}`",
        f"- Config: `{summary['config_path']}`",
        f"- Population size per composition: `{summary['population_size']}`",
        f"- Claim boundary: {summary['claim_boundary']}",
        "",
        "## Composition Reports",
        "",
        "| Composition | Archetypes | Speed factor range | Assignment digest |",
        "|---|---:|---|---|",
    ]
    for name, report in summary["reports"].items():
        lines.append(
            f"| `{name}` | {len(report['archetypes'])} | "
            f"{report['speed_factor_min']:.3g}-{report['speed_factor_max']:.3g} | "
            f"`{report['assignment_order_sha1']}` |"
        )
    lines.extend(
        [
            "",
            "This packet records deterministic population-composition assumptions only.",
            "It is not a homogeneous-vs-heterogeneous benchmark smoke and does not report metric deltas.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    """CLI entry point."""
    args = parse_args()
    config_path = REPO_ROOT / args.config
    speed_factors = load_archetypes(config_path)
    compositions = _load_example_compositions(config_path)
    reports = {
        name: build_archetype_population_report(
            n=args.population_size,
            composition=composition,
            speed_factors=speed_factors,
            initial_speed=args.initial_speed,
            seed=args.seed,
        )
        for name, composition in compositions.items()
    }
    summary = {
        "schema_version": "pedestrian-archetype-reporting-packet.v1",
        "status": "composition_report_only",
        "config_path": args.config,
        "population_size": args.population_size,
        "claim_boundary": (
            "No benchmark, realism, or planner-ranking claim. This packet only "
            "records deterministic composition assumptions for the shipped speed-archetype MVP."
        ),
        "reports": reports,
    }
    out = REPO_ROOT / args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    (out / "README.md").write_text(_format_markdown(summary), encoding="utf-8")
    print(f"wrote {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
