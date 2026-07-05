#!/usr/bin/env python3
"""Stage SocNavBench ETH traversible data through SVG generation and smoke validation.

This command is a thin issue #4546 orchestration layer around the canonical
SocNavBench ETH owners:

* ``manage_external_data.py`` for license-safe local asset layout checks.
* ``generate_socnavbench_traversible.py`` for mesh-derived ``data.pkl`` creation.
* ``convert_socnavbench_traversible_to_svg.py`` for Robot SF SVG conversion.

It never downloads or commits SocNavBench/S3DIS bytes. Missing local assets fail closed with exit
code 2 and a machine-readable report.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from robot_sf.maps.verification.svg_inspection import inspect_svg
from robot_sf.nav.svg_map_parser import convert_map
from scripts.tools import convert_socnavbench_traversible_to_svg as converter
from scripts.tools import generate_socnavbench_traversible as generator
from scripts.tools.manage_external_data import EXTERNAL_DATA_ROOT_ENV, check_asset

EXIT_BLOCKED = 2
ASSET_ID = "socnavbench-s3dis-eth"
DEFAULT_REPORT = Path("output/maps/issue_4546_socnavbench_eth_stage_svg_smoke.json")


def _write_report(path: Path | None, report: dict[str, Any]) -> None:
    """Write optional JSON report with stable formatting."""
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _smoke_svg(path: Path) -> dict[str, Any]:
    """Run parser and capability smoke checks for the generated SVG."""
    map_def = convert_map(str(path))
    inspection = inspect_svg(path)
    capability = inspection.capability_metadata
    return {
        "parser_obstacle_count": len(map_def.obstacles),
        "parser_robot_spawn_zone_count": len(map_def.robot_spawn_zones),
        "parser_robot_goal_zone_count": len(map_def.robot_goal_zones),
        "parser_robot_route_count": len(map_def.robot_routes),
        "parser_ped_spawn_zone_count": len(map_def.ped_spawn_zones),
        "parser_ped_goal_zone_count": len(map_def.ped_goal_zones),
        "parser_ped_route_count": len(map_def.ped_routes),
        "has_robot_runtime_routes": capability.has_robot_runtime_routes,
        "has_pedestrian_runtime_routes": capability.has_pedestrian_runtime_routes,
        "has_explicit_robot_runtime_zones": capability.has_explicit_robot_runtime_zones,
        "has_explicit_pedestrian_runtime_zones": capability.has_explicit_pedestrian_runtime_zones,
    }


def _smoke_ok(smoke: dict[str, Any]) -> bool:
    """Return whether parser and capability metadata prove minimal runtime map wiring."""
    count_keys = (
        "parser_obstacle_count",
        "parser_robot_spawn_zone_count",
        "parser_robot_goal_zone_count",
        "parser_robot_route_count",
        "parser_ped_spawn_zone_count",
        "parser_ped_goal_zone_count",
        "parser_ped_route_count",
    )
    flag_keys = (
        "has_robot_runtime_routes",
        "has_pedestrian_runtime_routes",
        "has_explicit_robot_runtime_zones",
        "has_explicit_pedestrian_runtime_zones",
    )
    return all(smoke[key] > 0 for key in count_keys) and all(bool(smoke[key]) for key in flag_keys)


def _generator_root_from_socnav_root(socnav_root: Path | None) -> Path | None:
    """Adapt direct SocNavBench root input to the generator's external-data-root convention."""
    if socnav_root is None:
        return None
    resolved = socnav_root.expanduser().resolve()
    if resolved.name == "socnavbench":
        return resolved.parent
    return resolved


def stage_generate_convert_smoke(
    *,
    socnav_root: Path | None,
    output_svg: Path,
    report_json: Path | None,
    dry_run: bool,
    force_generate: bool,
) -> tuple[int, dict[str, Any]]:
    """Run the SocNavBench ETH staging pipeline and return ``(exit_code, report)``."""
    generator_root = _generator_root_from_socnav_root(socnav_root)
    asset_report = check_asset(ASSET_ID, source_path=socnav_root)
    preflight = generator.preflight("ETH", root=generator_root)
    report: dict[str, Any] = {
        "asset_id": ASSET_ID,
        "status": "started",
        "socnav_root": asset_report["source_path"],
        "external_data_root_env": EXTERNAL_DATA_ROOT_ENV,
        "asset_check": asset_report,
        "generation_preflight": preflight,
        "output_svg": str(output_svg),
        "dry_run": dry_run,
        "conversion_ready": False,
        "smoke_ready": False,
        "claim_boundary": (
            "Smoke evidence only: validates local staged SocNavBench ETH traversible conversion "
            "and SVG parser wiring. It is not benchmark evidence."
        ),
    }

    if preflight["status"] == generator.STATUS_BLOCKED_MISSING_MESH:
        report.update(
            {
                "status": "blocked_missing_mesh",
                "next_action": preflight["next_action"],
            }
        )
        _write_report(report_json, report)
        return EXIT_BLOCKED, report

    if preflight["status"] == generator.STATUS_READY:
        if dry_run:
            report.update(
                {
                    "status": "dry_run_generation_ready",
                    "next_action": "Run without --dry-run to generate traversibles/ETH/data.pkl.",
                }
            )
            _write_report(report_json, report)
            return 0, report
        try:
            report["generation"] = generator.build_traversible(
                "ETH",
                root=generator_root,
                force=force_generate,
            )
        except generator.TraversibleGenerationError as exc:
            report.update({"status": "blocked_generation_failed", "next_action": str(exc)})
            _write_report(report_json, report)
            return EXIT_BLOCKED, report
    else:
        report["generation"] = {
            "status": preflight["status"],
            "output_pkl": preflight["output_pkl"],
            "built": False,
        }

    converter_report = output_svg.with_suffix(output_svg.suffix + ".conversion.json")
    converter_args = [
        "--socnav-root",
        str(Path(report["socnav_root"])),
        "--output-svg",
        str(output_svg),
        "--report-json",
        str(converter_report),
    ]
    if dry_run:
        converter_args.append("--dry-run")
    conversion_exit = converter.main(converter_args)
    report["conversion_exit_code"] = conversion_exit
    report["conversion_report_json"] = str(converter_report)
    report["conversion_ready"] = conversion_exit == 0

    if conversion_exit != 0:
        report.update(
            {
                "status": "blocked_conversion_failed",
                "next_action": "Inspect conversion report and re-stage official ETH traversible data.",
            }
        )
        _write_report(report_json, report)
        return conversion_exit, report

    if dry_run:
        report.update(
            {
                "status": "dry_run_conversion_ready",
                "next_action": "Run without --dry-run to write SVG and execute parser smoke.",
            }
        )
        _write_report(report_json, report)
        return 0, report

    smoke = _smoke_svg(output_svg)
    report["smoke"] = smoke
    report["smoke_ready"] = _smoke_ok(smoke)
    report["status"] = "ready" if report["smoke_ready"] else "blocked_smoke_failed"
    report["next_action"] = (
        "Generated SVG passed parser smoke."
        if report["smoke_ready"]
        else "Generated SVG did not expose all required Robot SF runtime map primitives."
    )
    _write_report(report_json, report)
    return 0 if report["smoke_ready"] else EXIT_BLOCKED, report


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--socnav-root", type=Path, default=None)
    parser.add_argument("--output-svg", type=Path, default=converter.DEFAULT_OUTPUT)
    parser.add_argument("--report-json", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force-generate", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the CLI."""
    args = build_arg_parser().parse_args(argv)
    exit_code, report = stage_generate_convert_smoke(
        socnav_root=args.socnav_root,
        output_svg=args.output_svg,
        report_json=args.report_json,
        dry_run=args.dry_run,
        force_generate=args.force_generate,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
