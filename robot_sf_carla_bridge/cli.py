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
    load_export_manifest_schema,
    load_export_schema,
    read_export_manifest,
    read_export_payload,
    resolve_export_manifest_payload_paths,
    write_export_records,
)


def _has_parent_reference(path_value: str) -> bool:
    return any(part == ".." for part in Path(path_value).parts)


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
    parser.add_argument(
        "--created-by",
        default="robot_sf_carla_bridge.cli",
        help="Provenance producer name recorded in the export payload.",
    )
    parser.add_argument(
        "--certificate-generator",
        default="scenario_cert.v1",
        help="Certificate generator id recorded in the export payload provenance.",
    )
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

    if _has_parent_reference(args.scenario_file):
        sys.stderr.write("Error: Invalid scenario file path.\n")
        return 1
    if _has_parent_reference(args.output_dir):
        sys.stderr.write("Error: Invalid output directory path.\n")
        return 1

    scenario_file = Path(args.scenario_file)
    output_dir = Path(args.output_dir)
    provenance = {
        "robot_sf_commit": args.robot_sf_commit,
        "created_by": args.created_by,
        "certificate_generator": args.certificate_generator,
    }
    try:
        records = build_export_payloads_from_scenario_file(scenario_file, provenance=provenance)
        write_export_records(records, output_dir)
    except (ValueError, OSError) as exc:
        sys.stderr.write(f"Error: {exc}\n")
        return 1
    sys.stdout.write(f"{(output_dir / 'manifest.json').as_posix()}\n")
    return 0


def validate_t0_manifest_main(argv: list[str] | None = None) -> int:
    """Validate a local CARLA T0 export manifest.

    Args:
        argv: Command-line arguments to parse; ``None`` reads from ``sys.argv``.

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

    manifest_path = Path(args.manifest)
    manifest = read_export_manifest(manifest_path)
    export_count = len(manifest["exports"])
    noun = "export" if export_count == 1 else "exports"
    sys.stdout.write(f"{manifest_path.as_posix()}: {export_count} {noun}\n")
    return 0


def validate_t0_export_batch_main(argv: list[str] | None = None) -> int:
    """Validate a local CARLA T0 export manifest and every referenced payload.

    Args:
        argv: Command-line arguments to parse; ``None`` reads from ``sys.argv``.

    Returns:
        Process-style exit code.
    """

    parser = argparse.ArgumentParser(description="Validate a CARLA T0 export batch.")
    parser.add_argument("--manifest", required=True, help="Path to export manifest JSON.")
    parser.add_argument("--json", action="store_true", help="Print a machine-readable summary.")
    args = parser.parse_args(argv)

    manifest_path = Path(args.manifest)
    payload_count = 0
    scenario_ids: list[str] = []
    for entry in resolve_export_manifest_payload_paths(manifest_path):
        read_export_payload(entry["path"])
        payload_count += 1
        scenario_ids.append(str(entry["scenario_id"]))

    if args.json:
        summary = {
            "manifest": manifest_path.as_posix(),
            "payload_count": payload_count,
            "scenario_ids": scenario_ids,
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


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(export_t0_scenarios_main())
