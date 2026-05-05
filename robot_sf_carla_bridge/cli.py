"""Command-line wrappers for CARLA-free T0 export helpers."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from robot_sf_carla_bridge.availability import check_carla_availability, load_availability_schema
from robot_sf_carla_bridge.export import (
    BATCH_VALIDATION_SUMMARY_SCHEMA_VERSION,
    build_export_payloads_from_scenario_file,
    load_batch_validation_summary_schema,
    load_export_manifest_payloads,
    load_export_manifest_schema,
    load_export_schema,
    read_export_manifest,
    write_export_records,
)
from robot_sf_carla_bridge.schema_catalog import list_carla_bridge_schema_catalog


def export_t0_scenarios_main(argv: list[str] | None = None) -> int:
    """Export a scenario manifest to local CARLA T0 neutral JSON files.

    Returns:
        Process-style exit code.
    """

    parser = argparse.ArgumentParser(description="Export Robot-SF scenarios to CARLA T0 JSON.")
    parser.add_argument("--schema", action="store_true", help="Print the JSON Schema contract.")
    parser.add_argument("--scenario-file", help="Scenario manifest YAML path.")
    parser.add_argument("--output-dir", help="Directory for export JSON files.")
    parser.add_argument("--robot-sf-commit", help="Robot-SF commit/provenance id.")
    parser.add_argument("--created-by", default="robot_sf_carla_bridge.cli")
    parser.add_argument("--certificate-generator", default="scenario_cert.v1")
    args = parser.parse_args(argv)

    if args.schema:
        sys.stdout.write(f"{json.dumps(load_export_schema(), sort_keys=True)}\n")
        return 0

    missing_args = [
        option
        for option, value in (
            ("--scenario-file", args.scenario_file),
            ("--output-dir", args.output_dir),
            ("--robot-sf-commit", args.robot_sf_commit),
        )
        if value is None or not str(value).strip()
    ]
    if missing_args:
        parser.error(f"the following arguments are required: {', '.join(missing_args)}")

    scenario_file = Path(args.scenario_file)
    output_dir = Path(args.output_dir)
    provenance = {
        "robot_sf_commit": args.robot_sf_commit,
        "created_by": args.created_by,
        "certificate_generator": args.certificate_generator,
    }
    records = build_export_payloads_from_scenario_file(scenario_file, provenance=provenance)
    write_export_records(records, output_dir)
    sys.stdout.write(f"{(output_dir / 'manifest.json').as_posix()}\n")
    return 0


def validate_t0_manifest_main(argv: list[str] | None = None) -> int:
    """Validate a local CARLA T0 export manifest.

    Returns:
        Process-style exit code.
    """

    parser = argparse.ArgumentParser(description="Validate a CARLA T0 export manifest.")
    parser.add_argument("--manifest", help="Path to export manifest JSON.")
    parser.add_argument("--schema", action="store_true", help="Print the JSON Schema contract.")
    args = parser.parse_args(argv)

    if args.schema:
        sys.stdout.write(f"{json.dumps(load_export_manifest_schema(), sort_keys=True)}\n")
        return 0
    if args.manifest is None or not str(args.manifest).strip():
        parser.error("the following arguments are required: --manifest")

    manifest = read_export_manifest(Path(args.manifest))
    export_count = len(manifest["exports"])
    noun = "export" if export_count == 1 else "exports"
    sys.stdout.write(f"{Path(args.manifest).as_posix()}: {export_count} {noun}\n")
    return 0


def validate_t0_export_batch_main(argv: list[str] | None = None) -> int:
    """Validate a local CARLA T0 export manifest and every referenced payload.

    Returns:
        Process-style exit code.
    """

    parser = argparse.ArgumentParser(description="Validate a CARLA T0 export batch.")
    parser.add_argument("--manifest", help="Path to export manifest JSON.")
    parser.add_argument("--json", action="store_true", help="Print a machine-readable summary.")
    parser.add_argument("--schema", action="store_true", help="Print the JSON Schema contract.")
    args = parser.parse_args(argv)

    if args.schema:
        sys.stdout.write(f"{json.dumps(load_batch_validation_summary_schema(), sort_keys=True)}\n")
        return 0
    if args.manifest is None or not str(args.manifest).strip():
        parser.error("the following arguments are required: --manifest")

    manifest_path = Path(args.manifest)
    records = load_export_manifest_payloads(manifest_path)
    payload_count = len(records)
    if args.json:
        summary = {
            "manifest": manifest_path.as_posix(),
            "payload_count": payload_count,
            "scenario_ids": [str(record["scenario_id"]) for record in records],
            "schema_version": BATCH_VALIDATION_SUMMARY_SCHEMA_VERSION,
        }
        sys.stdout.write(f"{json.dumps(summary, sort_keys=True)}\n")
        return 0

    noun = "payload" if payload_count == 1 else "payloads"
    sys.stdout.write(f"{manifest_path.as_posix()}: {payload_count} {noun}\n")
    return 0


def check_carla_availability_main(argv: list[str] | None = None) -> int:
    """Report whether the optional CARLA Python API is importable.

    Returns:
        Process-style exit code.
    """

    parser = argparse.ArgumentParser(description="Check optional CARLA Python API availability.")
    parser.add_argument("--json", action="store_true", help="Print a machine-readable status.")
    parser.add_argument("--schema", action="store_true", help="Print the JSON Schema contract.")
    parser.add_argument(
        "--require",
        action="store_true",
        help="Exit nonzero when the CARLA Python API is not available.",
    )
    args = parser.parse_args(argv)

    if args.schema:
        sys.stdout.write(f"{json.dumps(load_availability_schema(), sort_keys=True)}\n")
        return 0

    status = check_carla_availability()
    exit_code = 0 if status["status"] == "available" or not args.require else 1
    if args.json:
        sys.stdout.write(f"{json.dumps(status, sort_keys=True)}\n")
        return exit_code

    sys.stdout.write(f"{status['dependency']}: {status['status']} - {status['reason']}\n")
    return exit_code


def catalog_carla_schemas_main(argv: list[str] | None = None) -> int:
    """Print import-safe CARLA bridge schema catalog metadata.

    Returns:
        Process-style exit code.
    """

    parser = argparse.ArgumentParser(description="Print CARLA bridge schema catalog metadata.")
    parser.parse_args(argv)

    sys.stdout.write(f"{json.dumps(list_carla_bridge_schema_catalog(), sort_keys=True)}\n")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(export_t0_scenarios_main())
