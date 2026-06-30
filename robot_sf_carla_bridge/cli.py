"""Command-line wrappers for CARLA bridge helpers."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import jsonschema

from robot_sf_carla_bridge.availability import (
    CarlaUnavailableError,
    check_carla_availability,
    load_availability_schema,
)
from robot_sf_carla_bridge.docker_runtime import (
    run_carla_docker_live_replay,
    run_carla_docker_preflight,
    run_carla_docker_runtime_smoke,
)
from robot_sf_carla_bridge.export import (
    BATCH_VALIDATION_SUMMARY_SCHEMA_VERSION,
    build_export_payloads_from_scenario_file,
    load_batch_validation_summary_schema,
    load_export_manifest_schema,
    load_export_schema,
    read_export_manifest,
    read_export_payload,
    resolve_export_manifest_payload_paths,
    write_export_records,
)
from robot_sf_carla_bridge.parity_bundle_preflight import (
    check_native_aligned_bundle_manifest_readiness,
    check_parity_bundle_readiness,
)
from robot_sf_carla_bridge.replay_smoke import (
    T1_ORACLE_REPLAY_SMOKE_SCHEMA_VERSION,
    build_t1_oracle_replay_smoke_setup,
)
from robot_sf_carla_bridge.schema_catalog import (
    list_carla_bridge_schema_catalog,
    load_schema_catalog_schema,
)


def _has_parent_reference(path_value: str) -> bool:
    return any(part == ".." for part in Path(path_value).parts)


def load_export_manifest_payloads(manifest_path: Path) -> list[dict[str, object]]:
    """Load and validate every payload referenced by a T0 export manifest.

    The CLI keeps this small wrapper so tests and callers can patch either the batch-level helper or
    the lower-level path resolution and payload-read boundaries.

    Returns:
        Ordered payload records with scenario id, path, and validated payload.
    """

    records: list[dict[str, object]] = []
    for entry in resolve_export_manifest_payload_paths(manifest_path):
        payload_path = Path(entry["path"])
        records.append(
            {
                "scenario_id": entry["scenario_id"],
                "path": payload_path,
                "payload": read_export_payload(payload_path),
            }
        )
    return records


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
    scenario_ids = [str(record["scenario_id"]) for record in records]

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


def catalog_carla_schemas_main(argv: list[str] | None = None) -> int:
    """Print import-safe CARLA bridge schema catalog metadata.

    Returns:
        Process-style exit code.
    """

    parser = argparse.ArgumentParser(description="Print CARLA bridge schema catalog metadata.")
    parser.add_argument("--schema", action="store_true", help="Print the JSON Schema contract.")
    args = parser.parse_args(argv)

    if args.schema:
        sys.stdout.write(f"{json.dumps(load_schema_catalog_schema(), sort_keys=True)}\n")
        return 0
    sys.stdout.write(f"{json.dumps(list_carla_bridge_schema_catalog(), sort_keys=True)}\n")
    return 0


def replay_t1_oracle_smoke_main(argv: list[str] | None = None) -> int:
    """Validate one T0 export and prepare a setup-only CARLA T1 oracle replay smoke.

    Returns:
        Process-style exit code.
    """

    parser = argparse.ArgumentParser(
        description="Prepare one CARLA T1 oracle replay smoke from a T0 export manifest."
    )
    parser.add_argument("--manifest", help="Path to a T0 export manifest JSON.")
    parser.add_argument(
        "--scenario-id",
        help="Optional scenario id to select from the manifest; defaults to the first payload.",
    )
    parser.add_argument("--json", action="store_true", help="Print a machine-readable summary.")
    args = parser.parse_args(argv)

    if args.manifest is None or not str(args.manifest).strip():
        parser.error("the following arguments are required: --manifest")

    manifest_path = Path(args.manifest)
    try:
        summary = build_t1_oracle_replay_smoke_setup(
            manifest_path,
            scenario_id=args.scenario_id,
        )
    except CarlaUnavailableError as exc:
        status = {
            "schema_version": T1_ORACLE_REPLAY_SMOKE_SCHEMA_VERSION,
            "status": "not-available",
            "mode": "not-available",
            "manifest": manifest_path.as_posix(),
            "dependency": "carla",
            "reason": str(exc),
            "action": (
                "Install CARLA and ensure its Python API is on PYTHONPATH before using "
                "CARLA replay entry points."
            ),
        }
        if args.json:
            sys.stdout.write(f"{json.dumps(status, sort_keys=True)}\n")
        else:
            sys.stderr.write(f"Error: {status['reason']} {status['action']}\n")
        return 1
    except (ValueError, OSError, json.JSONDecodeError, jsonschema.ValidationError) as exc:
        sys.stderr.write(f"Error: {exc}\n")
        return 1

    if args.json:
        sys.stdout.write(f"{json.dumps(summary, sort_keys=True)}\n")
        return 0

    sys.stdout.write(
        f"{manifest_path.as_posix()}: scenario "
        f"{summary['selected_payload']['scenario_id']} ready for CARLA T1 oracle replay setup\n"
    )
    return 0


def carla_docker_runtime_main(argv: list[str] | None = None) -> int:
    """Run the opt-in pinned CARLA Docker runtime preflight or smoke workflow.

    Returns:
        Process-style exit code.
    """

    parser = argparse.ArgumentParser(
        description="Preflight or smoke-test the pinned CARLA 0.9.16 Docker runtime."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    preflight = subparsers.add_parser("preflight", help="Check host prerequisites only.")
    preflight.add_argument("--json", action="store_true", help="Print a machine-readable status.")
    preflight.add_argument(
        "--pull",
        action="store_true",
        help="Pull carlasim/carla:0.9.16 when missing after disk-space checks pass.",
    )
    preflight.add_argument(
        "--skip-api-check",
        action="store_true",
        help="Skip host-side carla==0.9.16 import validation.",
    )

    smoke = subparsers.add_parser(
        "smoke", help="Start CARLA, connect a Python client, and stop it."
    )
    smoke.add_argument("--json", action="store_true", help="Print a machine-readable status.")
    smoke.add_argument(
        "--pull",
        action="store_true",
        help="Pull carlasim/carla:0.9.16 when missing after disk-space checks pass.",
    )
    smoke.add_argument(
        "--startup-timeout-s",
        type=float,
        default=120.0,
        help="Seconds to wait for Python-client connectivity.",
    )
    live_replay = subparsers.add_parser(
        "live-replay", help="Start CARLA, run one T1 oracle live replay, and stop it."
    )
    live_replay.add_argument("--manifest", required=True, help="Path to a T0 export manifest JSON.")
    live_replay.add_argument(
        "--scenario-id",
        help="Optional scenario id to select from the manifest; defaults to the first payload.",
    )
    live_replay.add_argument("--json", action="store_true", help="Print a machine-readable status.")
    live_replay.add_argument(
        "--pull",
        action="store_true",
        help="Pull carlasim/carla:0.9.16 when missing after disk-space checks pass.",
    )
    live_replay.add_argument(
        "--startup-timeout-s",
        type=float,
        default=120.0,
        help="Seconds to wait for Python-client connectivity before replay.",
    )
    live_replay.add_argument(
        "--max-steps",
        type=int,
        help="Maximum oracle replay steps to execute for a smoke run.",
    )
    args = parser.parse_args(argv)

    if args.command == "preflight":
        status = run_carla_docker_preflight(
            pull=args.pull,
            require_carla_api=not args.skip_api_check,
        )
    elif args.command == "smoke":
        status = run_carla_docker_runtime_smoke(
            pull=args.pull,
            startup_timeout_s=args.startup_timeout_s,
        )
    else:
        status = run_carla_docker_live_replay(
            manifest_path=args.manifest,
            scenario_id=args.scenario_id,
            pull=args.pull,
            startup_timeout_s=args.startup_timeout_s,
            max_steps=args.max_steps,
        )

    exit_code = 0 if status["status"] in {"available", "connected", "oracle-replay"} else 1
    if args.json:
        sys.stdout.write(f"{json.dumps(status, sort_keys=True)}\n")
    else:
        sys.stdout.write(f"{status['status']}: {status['reason']}\n")
    return exit_code


def preflight_carla_parity_bundle_main(argv: list[str] | None = None) -> int:
    """Run the read-only readiness preflight for the compact CARLA parity bundle.

    Checks T0 export manifests, scenario fixture/provenance metadata, and the intended output
    location without importing CARLA or running any simulation. Exits nonzero when the bundle is
    not ready so the gate fails closed.

    Returns:
        Process-style exit code (0 when ready, 1 when not ready).
    """

    parser = argparse.ArgumentParser(
        description=(
            "Read-only readiness preflight for the compact CARLA native/aligned parity bundle "
            "(#1510). Does not run CARLA or assert metric parity."
        )
    )
    parser.add_argument(
        "--manifest",
        action="append",
        default=[],
        metavar="PATH",
        help="T0 export manifest JSON path; repeat for multiple candidate scenarios.",
    )
    parser.add_argument(
        "--readiness-manifest",
        metavar="PATH",
        help=(
            "Issue #1491 native/aligned readiness manifest declaring the intended scenario "
            "export manifests."
        ),
    )
    parser.add_argument(
        "--output-dir",
        help="Intended parity-bundle output directory (checked for safety; never created).",
    )
    parser.add_argument("--json", action="store_true", help="Print a machine-readable report.")
    args = parser.parse_args(argv)

    if args.readiness_manifest and args.manifest:
        parser.error("--readiness-manifest cannot be combined with --manifest")
    if not args.readiness_manifest and not args.manifest:
        parser.error("at least one --manifest or --readiness-manifest is required")

    if args.readiness_manifest:
        report = check_native_aligned_bundle_manifest_readiness(
            args.readiness_manifest,
            output_dir=args.output_dir,
        )
    else:
        report = check_parity_bundle_readiness(args.manifest, output_dir=args.output_dir)
    exit_code = 0 if report["status"] == "ready" else 1
    if args.json:
        sys.stdout.write(f"{json.dumps(report, sort_keys=True)}\n")
        return exit_code

    sys.stdout.write(
        f"parity-bundle readiness: {report['status']} "
        f"({report['ready_count']}/{report['scenario_count']} scenarios ready)\n"
    )
    for reason in report["blocking_reasons"]:
        sys.stdout.write(f"  blocked: {reason}\n")
    return exit_code


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(export_t0_scenarios_main())
