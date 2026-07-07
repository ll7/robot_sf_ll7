#!/usr/bin/env python3
"""Dry-run Robot SF -> external-benchmark scenario converter CLI (issue #3285).

Reads Robot SF scenario-matrix files and emits deterministic, schema-validated
intermediate representation (IR) scenarios, explicit unsupported-field reports,
fail-closed target export manifests, and asset-free target-shaped preview files.
No command here produces runnable SocNavBench or HuNavSim assets.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import yaml

from robot_sf.benchmark.scenario_interop import (
    SUPPORTED_TARGETS,
    build_target_compatibility_report,
    build_target_export_manifest,
    build_target_export_preview,
    convert_scenario_to_ir,
    dump_ir,
)


def _load_scenarios(path: Path) -> list[dict[str, Any]]:
    """Load scenario dicts from a YAML/JSON matrix file."""

    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if isinstance(raw, dict) and "scenarios" in raw:
        raw = raw["scenarios"]
    if isinstance(raw, dict):
        raw = [raw]
    if not isinstance(raw, list) or not all(isinstance(item, dict) for item in raw):
        raise ValueError(f"{path} must contain a scenario dict or list of scenario dicts")
    return raw


def _write_outputs(
    *,
    ir: dict[str, Any],
    scenario_id: str,
    targets: tuple[str, ...] | list[str],
    out_dir: Path | None,
    target_out_dir: Path | None,
    target_preview_out_dir: Path | None,
) -> None:
    """Write requested IR, manifest, and preview outputs for one scenario."""

    if out_dir is not None:
        out_path = out_dir / f"{scenario_id}.ir.json"
        out_path.write_text(dump_ir(ir), encoding="utf-8")
    if target_out_dir is not None:
        for target in targets:
            manifest = build_target_export_manifest(ir, target=target)
            manifest_path = target_out_dir / f"{scenario_id}.{target}.export_manifest.json"
            manifest_path.write_text(dump_ir(manifest), encoding="utf-8")
    if target_preview_out_dir is not None:
        for target in targets:
            preview = build_target_export_preview(ir, target=target)
            preview_path = target_preview_out_dir / f"{scenario_id}.{target}.export_preview.json"
            preview_path.write_text(dump_ir(preview), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    """Run CLI and return ``0`` when every scenario IR validates."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--matrix",
        required=True,
        type=Path,
        help="Path to Robot SF scenario-matrix YAML/JSON file.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Optional directory to write one <scenario_id>.ir.json file per scenario.",
    )
    parser.add_argument(
        "--target-out-dir",
        type=Path,
        default=None,
        help=(
            "Optional directory to write one "
            "<scenario_id>.<target>.export_manifest.json fail-closed target export "
            "manifest per scenario and requested target."
        ),
    )
    parser.add_argument(
        "--target-preview-out-dir",
        type=Path,
        default=None,
        help=(
            "Optional directory to write one <scenario_id>.<target>.export_preview.json "
            "asset-free target-shaped preview per scenario and requested target."
        ),
    )
    parser.add_argument(
        "--target",
        action="append",
        choices=SUPPORTED_TARGETS,
        default=[],
        help=(
            "Include fail-closed compatibility report for a target exporter. "
            "May be passed more than once."
        ),
    )
    args = parser.parse_args(argv)

    scenarios = _load_scenarios(args.matrix)
    for directory in (args.out_dir, args.target_out_dir, args.target_preview_out_dir):
        if directory is not None:
            directory.mkdir(parents=True, exist_ok=True)

    exit_code = 0
    summary: list[dict[str, Any]] = []
    for scenario in scenarios:
        result = convert_scenario_to_ir(scenario, source_file=str(args.matrix))
        scenario_id = result.ir["provenance"]["source_scenario_id"]
        targets = args.target or SUPPORTED_TARGETS
        target_compatibility = build_target_compatibility_report(
            result.ir,
            targets=targets,
        )

        if not result.is_valid:
            exit_code = 1
        summary.append(
            {
                "scenario_id": scenario_id,
                "ir_valid": result.is_valid,
                "unsupported_field_count": len(result.unsupported_fields),
                "schema_errors": result.schema_errors,
                "target_compatibility": target_compatibility,
            }
        )

        _write_outputs(
            ir=result.ir,
            scenario_id=scenario_id,
            targets=targets,
            out_dir=args.out_dir,
            target_out_dir=args.target_out_dir,
            target_preview_out_dir=args.target_preview_out_dir,
        )

        if (
            args.out_dir is None
            and args.target_out_dir is None
            and args.target_preview_out_dir is None
        ):
            sys.stdout.write(dump_ir(result.ir))

    sys.stderr.write(json.dumps({"dry_run_summary": summary}, indent=2) + "\n")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
