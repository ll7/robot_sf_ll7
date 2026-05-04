"""Command-line wrappers for CARLA-free T0 export helpers."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from robot_sf_carla_bridge.export import (
    build_export_payloads_from_scenario_file,
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


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(export_t0_scenarios_main())
