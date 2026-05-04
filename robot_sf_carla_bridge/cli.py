"""Command-line wrappers for CARLA-free T0 export helpers."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from robot_sf_carla_bridge.availability import check_carla_availability
from robot_sf_carla_bridge.export import (
    build_export_payloads_from_scenario_file,
    load_export_manifest_payloads,
    read_export_manifest,
    write_export_records,
)


def export_t0_scenarios_main(argv: list[str] | None = None) -> int:
    """Export a scenario manifest to local CARLA T0 neutral JSON files.

    Returns:
        Process-style exit code.
    """

    parser = argparse.ArgumentParser(description="Export Robot-SF scenarios to CARLA T0 JSON.")
    parser.add_argument("--scenario-file", required=True, help="Scenario manifest YAML path.")
    parser.add_argument("--output-dir", required=True, help="Directory for export JSON files.")
    parser.add_argument("--robot-sf-commit", required=True, help="Robot-SF commit/provenance id.")
    parser.add_argument("--created-by", default="robot_sf_carla_bridge.cli")
    parser.add_argument("--certificate-generator", default="scenario_cert.v1")
    args = parser.parse_args(argv)

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
    parser.add_argument("--manifest", required=True, help="Path to export manifest JSON.")
    args = parser.parse_args(argv)

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
    parser.add_argument("--manifest", required=True, help="Path to export manifest JSON.")
    parser.add_argument("--json", action="store_true", help="Print a machine-readable summary.")
    args = parser.parse_args(argv)

    manifest_path = Path(args.manifest)
    records = load_export_manifest_payloads(manifest_path)
    payload_count = len(records)
    if args.json:
        summary = {
            "manifest": manifest_path.as_posix(),
            "payload_count": payload_count,
            "scenario_ids": [str(record["scenario_id"]) for record in records],
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
    parser.add_argument(
        "--require",
        action="store_true",
        help="Exit nonzero when the CARLA Python API is not available.",
    )
    args = parser.parse_args(argv)

    status = check_carla_availability()
    exit_code = 0 if status["status"] == "available" or not args.require else 1
    if args.json:
        sys.stdout.write(f"{json.dumps(status, sort_keys=True)}\n")
        return exit_code

    sys.stdout.write(f"{status['dependency']}: {status['status']} - {status['reason']}\n")
    return exit_code


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(export_t0_scenarios_main())
