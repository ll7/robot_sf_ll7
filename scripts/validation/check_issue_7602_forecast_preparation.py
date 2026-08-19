#!/usr/bin/env python3
"""Build or side-effect-free-check the issue #7602 preparation evidence packet."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from robot_sf.benchmark.forecast.forecast_preparation import (
    ForecastPreparationSourceSpec,
    build_forecast_preparation_packet,
    validate_forecast_preparation_packet,
)
from robot_sf.benchmark.identity.hash_utils import sha256_file
from robot_sf.evidence.writers import write_json, write_review_sidecar, write_text

REPO_ROOT = Path(__file__).resolve().parents[2]
EVIDENCE_DIR = REPO_ROOT / "docs/context/evidence/issue_7399_forecast_preparation"
DEFAULT_PACKET = EVIDENCE_DIR / "forecast_preparation_packet.json"
DEFAULT_MANIFEST = EVIDENCE_DIR / "checksums.sha256"

SOURCE_SPECS = (
    ForecastPreparationSourceSpec(
        path="tests/fixtures/analysis_workbench/simulation_trace_export_v1/issue_2937/"
        "bottleneck_motion_rich_fixture.json",
        scenario_family="bottleneck",
        cutoff_frame_step=5,
    ),
    ForecastPreparationSourceSpec(
        path="docs/context/evidence/issue_2667_trace_failure_predicate_tables_2026-06-12/"
        "inputs/synthetic_crossing_proxy_orca_111_trace_export.json",
        scenario_family="crossing_proxy",
        cutoff_frame_step=2,
    ),
    ForecastPreparationSourceSpec(
        path="docs/context/evidence/issue_2428_mechanism_trace_panels_2026-06-06/traces/"
        "ammv_social_force_trace_export.json",
        scenario_family="head_on_corridor",
        cutoff_frame_step=5,
    ),
)


def _checksum_paths() -> tuple[Path, ...]:
    return (
        DEFAULT_PACKET,
        DEFAULT_PACKET.parent / "README.md",
        REPO_ROOT / "robot_sf/benchmark/forecast/forecast_preparation.py",
        Path(__file__).resolve(),
        REPO_ROOT / "tests/benchmark/test_forecast_preparation.py",
        REPO_ROOT / SOURCE_SPECS[0].path,
        REPO_ROOT / SOURCE_SPECS[1].path,
        REPO_ROOT / SOURCE_SPECS[2].path,
        REPO_ROOT / "docs/context/evidence/issue_2667_trace_failure_predicate_tables_2026-06-12/"
        "trace_failure_predicate_tables.json",
        REPO_ROOT / "pyproject.toml",
        REPO_ROOT / "docs/context/dependency_license_inventory.md",
        REPO_ROOT / "fast-pysf/LICENSE",
        REPO_ROOT / "THIRD_PARTY_NOTICES.md",
        REPO_ROOT / "third_party/python-rvo2/LICENSE",
        REPO_ROOT / "docs/context/issue_653_social_navigation_pyenvs_socialforce_runtime.md",
    )


def _write_packet(packet: dict[str, object]) -> None:
    write_json(DEFAULT_PACKET, packet, catalog_area="benchmark_evidence")


def _write_checksums(paths: tuple[Path, ...]) -> None:
    lines = [
        f"{sha256_file(path)}  {path.resolve().relative_to(REPO_ROOT).as_posix()}"
        for path in sorted(
            set(paths), key=lambda item: item.resolve().relative_to(REPO_ROOT).as_posix()
        )
    ]
    write_text(
        DEFAULT_MANIFEST,
        "# AI-GENERATED NEEDS-REVIEW\n" + "\n".join(lines) + "\n",
    )


def _build() -> dict[str, object]:
    EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)
    # The packet covers its own final bytes; a placeholder makes that path resolvable
    # while the deterministic packet is being constructed.
    if not DEFAULT_PACKET.exists():
        write_json(DEFAULT_PACKET, {})
    checksum_paths = _checksum_paths()
    packet = build_forecast_preparation_packet(
        SOURCE_SPECS,
        repo_root=REPO_ROOT,
        checksum_paths=checksum_paths,
    )
    _write_packet(packet)
    _write_checksums(checksum_paths)
    for artifact_path in (EVIDENCE_DIR / "README.md", DEFAULT_PACKET, DEFAULT_MANIFEST):
        write_review_sidecar(artifact_path, repo_root=REPO_ROOT)
    return validate_forecast_preparation_packet(
        packet,
        repo_root=REPO_ROOT,
        verify_checksums=True,
    )


def _check(packet_path: Path) -> dict[str, object]:
    payload = json.loads(packet_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("packet must be a JSON object")
    return validate_forecast_preparation_packet(
        payload,
        repo_root=REPO_ROOT,
        verify_checksums=True,
    )


def main(argv: list[str] | None = None) -> int:
    """Run build or read-only check, returning nonzero on any violation."""
    parser = argparse.ArgumentParser(
        description="Validate issue #7602 matched forecast-preparation evidence."
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--build", action="store_true", help="Write the tracked evidence packet.")
    mode.add_argument("--check", action="store_true", help="Validate without writing files.")
    parser.add_argument(
        "--packet",
        type=Path,
        default=DEFAULT_PACKET,
        help="Packet path for --check (default: the issue #7602 packet).",
    )
    args = parser.parse_args(argv)
    try:
        result = _build() if args.build else _check(args.packet.resolve())
    except (OSError, TypeError, ValueError) as exc:
        print(f"FAIL: {exc}")
        return 2
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
