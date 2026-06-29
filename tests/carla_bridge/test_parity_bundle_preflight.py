"""Synthetic, simulator-free tests for the CARLA parity-bundle readiness preflight (#1510).

These tests never import CARLA or run a simulation. They construct schema-valid T0 export
manifests/payloads on disk (and hand-built payload mappings for the pure metadata checker) to prove
the preflight fails closed on missing scenario, certificate, provenance, and output prerequisites,
and reports ``ready`` only when every prerequisite is present.
"""

from __future__ import annotations

import copy
import json
from typing import TYPE_CHECKING

import pytest

from robot_sf_carla_bridge.cli import preflight_carla_parity_bundle_main
from robot_sf_carla_bridge.export import write_export_records
from robot_sf_carla_bridge.parity_bundle_preflight import (
    PREFLIGHT_SCHEMA_VERSION,
    READINESS_MANIFEST_SCHEMA_VERSION,
    check_native_aligned_bundle_manifest_readiness,
    check_parity_bundle_readiness,
    evaluate_payload_metadata,
)

if TYPE_CHECKING:
    from pathlib import Path


def _valid_payload(scenario_id: str = "unit_crossing") -> dict:
    """Return a minimal schema-valid T0 export payload for a certified scenario."""
    return {
        "schema_version": "carla-replay-export.v1",
        "mode": "neutral-export",
        "scenario": {
            "id": scenario_id,
            "source_config": f"configs/scenarios/{scenario_id}.yaml",
            "map_id": "unit_map",
            "certificate": {
                "schema_version": "scenario_cert.v1",
                "status": "valid",
                "source": f"output/scenario_cert/{scenario_id}.json",
            },
        },
        "robot": {
            "start": {"x": 0.0, "y": 0.0, "theta": 0.0},
            "goal": {"x": 4.0, "y": 0.0, "theta": 0.0},
            "footprint": {"radius_m": 0.3},
            "kinematics": {"model": "unicycle", "max_speed_mps": 1.0},
        },
        "pedestrians": [
            {
                "id": "ped_0",
                "start": {"x": 2.0, "y": -1.0, "theta": 1.57},
                "route": [{"x": 2.0, "y": 1.0}],
                "timing": {"start_delay_s": 0.0},
            }
        ],
        "static_geometry": {"obstacles": [], "route_topology_ref": "maps/svg_maps/unit_map.svg"},
        "simulation": {
            "dt_s": 0.1,
            "horizon_s": 10.0,
            "termination": ["success", "collision", "timeout"],
        },
        "metrics": {
            "trajectory_fields": ["success", "collision", "min_distance_m", "ttc_min_s"],
        },
        "provenance": {
            "robot_sf_commit": "abc123",
            "created_by": "unit-test",
            "certificate_generator": "scenario_cert.v1",
        },
    }


def _write_manifest(tmp_path: Path, *payloads: dict, subdir: str = "exports") -> Path:
    """Write payloads to a real T0 export manifest and return the manifest path."""
    records = [
        {"scenario_id": payload["scenario"]["id"], "payload": payload} for payload in payloads
    ]
    output_dir = tmp_path / subdir
    write_export_records(records, output_dir)
    return output_dir / "manifest.json"


def _write_readiness_manifest(tmp_path: Path, entries: list[dict]) -> Path:
    """Write issue #1491 readiness manifest beside synthetic export bundles."""

    manifest = tmp_path / "carla_parity_readiness.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": READINESS_MANIFEST_SCHEMA_VERSION,
                "bundle_id": "issue-1491-unit",
                "issue": 1491,
                "scenarios": entries,
            }
        ),
        encoding="utf-8",
    )
    return manifest


# --- pure metadata checker -------------------------------------------------------------------


def test_evaluate_payload_metadata_accepts_certified_payload() -> None:
    """A certified payload with provenance and trajectory fields yields no blocking reasons."""
    reasons, checks = evaluate_payload_metadata(_valid_payload())
    assert reasons == []
    assert checks["certificate_status"] == "valid"
    assert checks["robot_sf_commit"] == "abc123"
    assert "success" in checks["trajectory_fields"]


def test_evaluate_payload_metadata_flags_missing_certificate() -> None:
    """A payload without a certificate status must be blocked, not silently accepted."""
    payload = copy.deepcopy(_valid_payload())
    del payload["scenario"]["certificate"]["status"]
    reasons, checks = evaluate_payload_metadata(payload)
    assert any("certificate status is missing" in reason for reason in reasons)
    assert checks["certificate_status"] is None


def test_evaluate_payload_metadata_flags_ineligible_certificate() -> None:
    """A structurally present but non-eligible certificate status is rejected."""
    payload = copy.deepcopy(_valid_payload())
    payload["scenario"]["certificate"]["status"] = "excluded"
    reasons, _ = evaluate_payload_metadata(payload)
    assert any("is not eligible" in reason for reason in reasons)


def test_evaluate_payload_metadata_flags_missing_provenance_commit() -> None:
    """Missing provenance robot_sf_commit blocks readiness so the native side stays bindable."""
    payload = copy.deepcopy(_valid_payload())
    del payload["provenance"]["robot_sf_commit"]
    reasons, _ = evaluate_payload_metadata(payload)
    assert any("robot_sf_commit is missing" in reason for reason in reasons)


def test_evaluate_payload_metadata_flags_empty_trajectory_fields() -> None:
    """An empty trajectory-field list blocks readiness."""
    payload = copy.deepcopy(_valid_payload())
    payload["metrics"]["trajectory_fields"] = []
    reasons, _ = evaluate_payload_metadata(payload)
    assert any("trajectory_fields is missing or empty" in reason for reason in reasons)


def test_evaluate_payload_metadata_reports_missing_parity_metrics_without_blocking() -> None:
    """Parity metrics absent from trajectory_fields are reported but do not block readiness."""
    reasons, checks = evaluate_payload_metadata(_valid_payload(), metric_names=["success", "snqi"])
    assert reasons == []
    assert checks["missing_parity_metrics"] == ["snqi"]


# --- bundle-level readiness ------------------------------------------------------------------


def test_check_parity_bundle_readiness_ready_for_certified_bundle(tmp_path: Path) -> None:
    """A bundle of certified scenarios with a clean output dir reports ready."""
    manifest = _write_manifest(tmp_path, _valid_payload("scn_a"), _valid_payload("scn_b"))
    report = check_parity_bundle_readiness([manifest], output_dir=tmp_path / "parity_out")
    assert report["schema_version"] == PREFLIGHT_SCHEMA_VERSION
    assert report["status"] == "ready"
    assert report["scenario_count"] == 2
    assert report["ready_count"] == 2
    assert report["blocking_reasons"] == []
    assert report["output_location"]["status"] == "ready"


def test_check_parity_bundle_readiness_blocks_on_missing_manifest(tmp_path: Path) -> None:
    """A missing manifest file fails closed with an explicit blocked row."""
    report = check_parity_bundle_readiness([tmp_path / "nope" / "manifest.json"])
    assert report["status"] == "not-ready"
    assert report["scenario_count"] == 1
    assert report["scenarios"][0]["scenario_id"] is None
    assert any("not found" in reason for reason in report["blocking_reasons"])


def test_check_parity_bundle_readiness_blocks_on_missing_payload_file(tmp_path: Path) -> None:
    """A manifest that references a deleted payload file fails closed for that scenario."""
    manifest = _write_manifest(tmp_path, _valid_payload("scn_a"))
    # Remove the referenced payload to simulate an incomplete export bundle.
    for payload_file in manifest.parent.glob("*.json"):
        if payload_file.name != "manifest.json":
            payload_file.unlink()
    report = check_parity_bundle_readiness([manifest])
    assert report["status"] == "not-ready"
    assert report["scenarios"][0]["status"] == "blocked"
    assert any("payload file not found" in reason for reason in report["blocking_reasons"])


def test_check_parity_bundle_readiness_blocks_on_schema_invalid_payload(tmp_path: Path) -> None:
    """A referenced payload that is not schema-valid (missing provenance) fails closed.

    The export schema requires ``provenance.robot_sf_commit``, so a payload missing it is rejected
    at the file boundary. We write the bundle raw (bypassing the validating writer) to prove the
    preflight surfaces such a malformed export as a blocked row instead of trusting it.
    """
    payload = _valid_payload("scn_a")
    del payload["provenance"]["robot_sf_commit"]
    out_dir = tmp_path / "exports"
    out_dir.mkdir()
    (out_dir / "scn_a.json").write_text(json.dumps(payload), encoding="utf-8")
    manifest = out_dir / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": "carla-replay-export-manifest.v1",
                "exports": [{"scenario_id": "scn_a", "path": "scn_a.json"}],
            }
        ),
        encoding="utf-8",
    )
    report = check_parity_bundle_readiness([manifest])
    assert report["status"] == "not-ready"
    assert report["ready_count"] == 0
    assert report["scenarios"][0]["status"] == "blocked"
    assert any("invalid export payload" in reason for reason in report["blocking_reasons"])


def test_check_parity_bundle_readiness_no_manifests_is_not_ready() -> None:
    """An empty manifest list fails closed rather than vacuously passing."""
    report = check_parity_bundle_readiness([])
    assert report["status"] == "not-ready"
    assert report["scenario_count"] == 0
    assert "no scenario manifests were provided" in report["blocking_reasons"]


def test_check_parity_bundle_readiness_flags_unsafe_output_dir(tmp_path: Path) -> None:
    """A parent-relative output path is blocking even when scenarios are ready."""
    manifest = _write_manifest(tmp_path, _valid_payload("scn_a"))
    report = check_parity_bundle_readiness([manifest], output_dir="output/../escape")
    assert report["status"] == "not-ready"
    assert report["output_location"]["status"] == "unsafe"
    assert any("output location" in reason for reason in report["blocking_reasons"])


def test_check_parity_bundle_readiness_flags_non_empty_output_dir(tmp_path: Path) -> None:
    """A populated output dir is reported as overwrite risk but does not block on its own."""
    manifest = _write_manifest(tmp_path, _valid_payload("scn_a"))
    out_dir = tmp_path / "parity_out"
    out_dir.mkdir()
    (out_dir / "stale.json").write_text("{}", encoding="utf-8")
    report = check_parity_bundle_readiness([manifest], output_dir=out_dir)
    assert report["output_location"]["status"] == "not-empty"
    # Scenario is still ready; not-empty is informational, not a hard block.
    assert report["status"] == "ready"


def test_check_parity_bundle_readiness_never_claims_parity(tmp_path: Path) -> None:
    """The report must carry an explicit non-parity claim boundary."""
    manifest = _write_manifest(tmp_path, _valid_payload("scn_a"))
    report = check_parity_bundle_readiness([manifest])
    assert "does not assert native↔aligned metric parity" in report["claim_boundary"]


def test_native_aligned_readiness_manifest_reports_ready(tmp_path: Path) -> None:
    """Issue #1491 manifest ready only when native and aligned entries pass static checks."""

    native_manifest = _write_manifest(tmp_path, _valid_payload("native_a"), subdir="native")
    aligned_manifest = _write_manifest(tmp_path, _valid_payload("aligned_a"), subdir="aligned")
    readiness_manifest = _write_readiness_manifest(
        tmp_path,
        [
            {
                "scenario_id": "native_a",
                "mode": "native",
                "manifest": native_manifest.relative_to(tmp_path).as_posix(),
            },
            {
                "scenario_id": "aligned_a",
                "mode": "aligned",
                "manifest": aligned_manifest.relative_to(tmp_path).as_posix(),
            },
        ],
    )

    report = check_native_aligned_bundle_manifest_readiness(
        readiness_manifest,
        output_dir=tmp_path / "parity_out",
    )

    assert report["status"] == "ready"
    assert report["readiness_manifest_schema_version"] == READINESS_MANIFEST_SCHEMA_VERSION
    assert report["issue"] == 1491
    assert report["ready_count"] == 2
    assert {row["readiness_mode"] for row in report["scenarios"]} == {"native", "aligned"}
    assert "does not assert native↔aligned metric parity" in report["claim_boundary"]


def test_native_aligned_readiness_manifest_blocks_missing_mode(tmp_path: Path) -> None:
    """A manifest without both native and aligned coverage fails closed."""

    native_manifest = _write_manifest(tmp_path, _valid_payload("native_a"), subdir="native")
    readiness_manifest = _write_readiness_manifest(
        tmp_path,
        [
            {
                "scenario_id": "native_a",
                "mode": "native",
                "manifest": native_manifest.relative_to(tmp_path).as_posix(),
            }
        ],
    )

    report = check_native_aligned_bundle_manifest_readiness(readiness_manifest)

    assert report["status"] == "not-ready"
    assert report["ready_count"] == 1
    assert any("missing mode coverage: aligned" in reason for reason in report["blocking_reasons"])


def test_native_aligned_readiness_manifest_blocks_missing_export_manifest(tmp_path: Path) -> None:
    """Missing declared T0 export manifest is reported as a scenario blocker."""

    native_manifest = _write_manifest(tmp_path, _valid_payload("native_a"), subdir="native")
    readiness_manifest = _write_readiness_manifest(
        tmp_path,
        [
            {
                "scenario_id": "native_a",
                "mode": "native",
                "manifest": native_manifest.relative_to(tmp_path).as_posix(),
            },
            {"scenario_id": "aligned_a", "mode": "aligned", "manifest": "aligned/manifest.json"},
        ],
    )

    report = check_native_aligned_bundle_manifest_readiness(readiness_manifest)

    assert report["status"] == "not-ready"
    assert report["ready_count"] == 1
    assert any("export manifest file not found" in reason for reason in report["blocking_reasons"])


def test_native_aligned_readiness_manifest_blocks_missing_readiness_manifest(
    tmp_path: Path,
) -> None:
    """Absent issue #1491 readiness manifest produces explicit not-ready report."""

    report = check_native_aligned_bundle_manifest_readiness(tmp_path / "missing.json")

    assert report["status"] == "not-ready"
    assert report["scenario_count"] == 1
    assert any(
        "readiness manifest file not found" in reason for reason in report["blocking_reasons"]
    )


# --- CLI -------------------------------------------------------------------------------------


def test_cli_reports_ready_and_exits_zero(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    """The CLI exits 0 and emits a ready JSON report for a certified bundle."""
    manifest = _write_manifest(tmp_path, _valid_payload("scn_a"))
    exit_code = preflight_carla_parity_bundle_main(
        ["--manifest", str(manifest), "--output-dir", str(tmp_path / "out"), "--json"]
    )
    assert exit_code == 0
    report = json.loads(capsys.readouterr().out)
    assert report["status"] == "ready"


def test_cli_accepts_native_aligned_readiness_manifest(
    tmp_path: Path, capsys: pytest.CaptureFixture
) -> None:
    """The CLI checks an issue #1491 readiness manifest without CARLA."""
    native_manifest = _write_manifest(tmp_path, _valid_payload("native_a"), subdir="native")
    aligned_manifest = _write_manifest(tmp_path, _valid_payload("aligned_a"), subdir="aligned")
    readiness_manifest = _write_readiness_manifest(
        tmp_path,
        [
            {
                "scenario_id": "native_a",
                "mode": "native",
                "manifest": native_manifest.relative_to(tmp_path).as_posix(),
            },
            {
                "scenario_id": "aligned_a",
                "mode": "aligned",
                "manifest": aligned_manifest.relative_to(tmp_path).as_posix(),
            },
        ],
    )

    exit_code = preflight_carla_parity_bundle_main(
        ["--readiness-manifest", str(readiness_manifest), "--json"]
    )

    assert exit_code == 0
    report = json.loads(capsys.readouterr().out)
    assert report["readiness_manifest_schema_version"] == READINESS_MANIFEST_SCHEMA_VERSION
    assert report["status"] == "ready"


def test_cli_exits_nonzero_when_not_ready(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    """The CLI fails closed (exit 1) when prerequisites are missing."""
    exit_code = preflight_carla_parity_bundle_main(["--manifest", str(tmp_path / "missing.json")])
    assert exit_code == 1
    assert "not-ready" in capsys.readouterr().out


def test_cli_requires_manifest() -> None:
    """The CLI requires at least one --manifest argument."""
    with pytest.raises(SystemExit):
        preflight_carla_parity_bundle_main([])
