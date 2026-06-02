"""Tests for the hazard/ODD coverage rollup CLI."""

from __future__ import annotations

import csv
import json
from typing import TYPE_CHECKING

import yaml

from scripts.tools import hazard_odd_coverage_rollup as cli

if TYPE_CHECKING:
    from pathlib import Path


def test_hazard_odd_rollup_emits_status_tables_and_provenance(tmp_path: Path) -> None:
    """A fixture campaign should expose covered, partial, missing, and excluded cases."""

    campaign_root = _write_campaign_fixture(tmp_path / "campaign")
    hazard_path = _write_hazard_mapping(tmp_path / "hazards.yaml")
    odd_path = _write_odd_contract(tmp_path / "odd.yaml")
    scenario_contract_path = _write_scenario_contracts(tmp_path / "scenario_contracts.yaml")
    scenario_cert_path = _write_scenario_certs(tmp_path / "scenario_certs.yaml")
    output = tmp_path / "rollup"

    exit_code = cli.main(
        [
            "--campaign-root",
            str(campaign_root),
            "--output",
            str(output),
            "--report-id",
            "fixture_rollup",
            "--hazard-traceability",
            str(hazard_path),
            "--odd-contract",
            str(odd_path),
            "--scenario-contract",
            str(scenario_contract_path),
            "--scenario-cert",
            str(scenario_cert_path),
        ]
    )

    assert exit_code == 0
    summary = json.loads((output / "hazard_odd_coverage_summary.json").read_text())
    assert summary["schema_version"] == "hazard_odd_coverage_rollup.v1"
    assert summary["executed_evidence"]["row_count"] == 2
    assert "s_seed_only" not in summary["executed_evidence"]["scenario_ids"]
    assert summary["executed_evidence"]["caveated_row_count"] == 1
    assert summary["provenance"]["generation_commit"]
    assert summary["provenance"]["generated_artifacts"]
    assert "fallback" in (output / "hazard_odd_coverage_summary.md").read_text()

    hazard_statuses = {row["hazard_id"]: row["status"] for row in summary["hazards"]}
    assert hazard_statuses == {
        "h_full": "covered",
        "h_partial": "partial",
        "h_missing": "missing",
        "h_excluded": "excluded",
    }

    odd_rows = summary["odd_boundaries"]
    assert {
        row["boundary_id"]: row["status"]
        for row in odd_rows
        if row["boundary_type"] == "excluded_claim"
    }["safety_certification"] == "excluded"
    assert {
        row["boundary_id"]: row["status"]
        for row in odd_rows
        if row["boundary_type"] == "supported_claim"
    }["benchmark_evidence_boundary"] == "partial"

    hazard_table = (output / "hazard_coverage_table.csv").read_text(encoding="utf-8")
    odd_table = (output / "odd_boundary_table.csv").read_text(encoding="utf-8")
    assert "h_partial,partial" in hazard_table
    assert "safety_certification,excluded" in odd_table
    assert (output / "coverage_status_summary.png").exists()
    assert "coverage_status_summary.png" in (output / "checksums.sha256").read_text()


def test_hazard_odd_rollup_marks_missing_optional_inputs_unavailable(tmp_path: Path) -> None:
    """Unavailable inputs should be explicit instead of silently disappearing."""

    campaign_root = _write_campaign_fixture(tmp_path / "campaign")
    output = tmp_path / "rollup"

    assert cli.main(["--campaign-root", str(campaign_root), "--output", str(output)]) == 0

    summary = json.loads((output / "hazard_odd_coverage_summary.json").read_text())
    assert summary["hazards"][0]["status"] == "unavailable"
    assert summary["odd_boundaries"][0]["status"] == "unavailable"
    unavailable_inputs = [
        item for item in summary["metadata_inputs"] if item["status"] == "unavailable"
    ]
    assert any(
        item["reason"] == "hazard traceability file not supplied" for item in unavailable_inputs
    )
    assert summary["stress_uncertainty_coverage"]["status"] == "unavailable"


def _write_campaign_fixture(campaign_root: Path) -> Path:
    """Create a tiny campaign reports tree."""

    reports = campaign_root / "reports"
    reports.mkdir(parents=True)
    with (reports / "campaign_table.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["scenario_id", "scenario_family", "status"])
        writer.writeheader()
        writer.writerow(
            {"scenario_id": "s_full", "scenario_family": "family_full", "status": "native"}
        )
        writer.writerow(
            {
                "scenario_id": "s_partial",
                "scenario_family": "family_partial",
                "status": "fallback",
            }
        )
    with (reports / "seed_episode_rows.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["scenario_id", "scenario_family", "status"])
        writer.writeheader()
        writer.writerow(
            {
                "scenario_id": "s_seed_only",
                "scenario_family": "family_seed",
                "status": "native",
            }
        )
    return campaign_root


def _write_hazard_mapping(path: Path) -> Path:
    """Write a hazard traceability fixture."""

    payload = {
        "schema_version": "hazard_traceability.v1",
        "id": "fixture_hazards",
        "claim_boundary": "metadata only",
        "hazards": [
            _hazard("h_full"),
            _hazard("h_partial"),
            _hazard("h_missing"),
            _hazard("h_excluded"),
        ],
        "scenario_mappings": [
            _hazard_mapping("m_full", ["s_full"], ["h_full"]),
            _hazard_mapping("m_partial", ["s_partial"], ["h_partial"]),
            _hazard_mapping("m_missing", ["s_missing"], ["h_missing"]),
            _hazard_mapping("m_excluded", ["s_excluded"], ["h_excluded"]),
        ],
        "provenance": {
            "source_issue": "fixture",
            "authored_by": "test",
            "source_files": ["tests/tools/test_hazard_odd_coverage_rollup.py"],
            "notes": "fixture",
        },
    }
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return path


def _hazard(hazard_id: str) -> dict[str, object]:
    """Return a minimal hazard definition."""

    return {
        "id": hazard_id,
        "description": f"{hazard_id} description",
        "severity": "safety",
        "supporting_metrics": ["collision_rate"],
        "evidence_fields": ["metrics.collision_rate"],
    }


def _hazard_mapping(
    mapping_id: str, scenario_ids: list[str], hazards: list[str]
) -> dict[str, object]:
    """Return a minimal scenario-hazard mapping."""

    return {
        "id": mapping_id,
        "scenario_ids": scenario_ids,
        "scenario_families": [],
        "hazards": hazards,
        "notes": "fixture",
    }


def _write_odd_contract(path: Path) -> Path:
    """Write an ODD contract fixture."""

    payload = {
        "contracts": [
            {
                "schema_version": "odd_contract.v1",
                "id": "fixture_odd",
                "operating_context": {
                    "environment_types": ["synthetic_atomic"],
                    "map_families": ["family_full"],
                    "surface_conditions": ["dry"],
                    "visibility": ["clear"],
                    "semantic_features": ["pedestrian_flow"],
                },
                "agents": {
                    "actor_types": ["robot", "pedestrian"],
                    "pedestrian_motion_models": ["social_force"],
                    "robot_kinematics": ["differential_drive"],
                },
                "speed_limits": {
                    "max_robot_speed_mps": 2.0,
                    "max_pedestrian_speed_mps": 1.5,
                    "notes": "fixture",
                },
                "pedestrian_density": {
                    "density_bins": ["low"],
                    "max_pedestrians_per_scene": 2,
                    "notes": "fixture",
                },
                "observation": {
                    "observation_modes": ["socnav_state"],
                    "sensor_assumptions": ["simulator state"],
                },
                "exclusions": ["public-road autonomy"],
                "claim_boundaries": {
                    "evidence_status": "metadata_only",
                    "supported_claims": ["benchmark_evidence_boundary"],
                    "non_claims": ["safety_certification"],
                    "caveats": ["fixture caveat"],
                },
                "provenance": {
                    "source_issue": "fixture",
                    "authored_by": "test",
                    "source_files": ["tests/tools/test_hazard_odd_coverage_rollup.py"],
                    "notes": "fixture",
                },
            }
        ]
    }
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return path


def _write_scenario_contracts(path: Path) -> Path:
    """Write scenario contracts linked to the ODD fixture."""

    payload = {
        "contracts": [
            _scenario_contract("s_full", "family_full"),
            _scenario_contract("s_partial", "family_partial"),
        ]
    }
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return path


def _scenario_contract(scenario_name: str, scenario_family: str) -> dict[str, object]:
    """Return a minimal scenario contract fixture."""

    return {
        "schema_version": "scenario_contract.v1",
        "id": f"{scenario_name}_contract",
        "scenario_ref": {
            "source": "configs/scenarios/single/observation_visibility_blind_corner_smoke.yaml",
            "scenario_name": scenario_name,
            "scenario_family": scenario_family,
        },
        "odd": {
            "environment_type": "synthetic_atomic",
            "map_family": scenario_family,
            "density": "low",
            "flow": "crossing",
            "assumptions": ["fixture"],
        },
        "odd_contract_ref": {
            "source": "fixture_odd.yaml",
            "contract_id": "fixture_odd",
            "required_for_benchmark_claim": True,
        },
        "actors": [
            {
                "id": "robot",
                "kind": "robot",
                "count": {"minimum": 1, "maximum": 1},
                "motion_model": "configured_robot_policy",
                "assumptions": ["fixture"],
            }
        ],
        "invariants": [
            {
                "id": "intent_present",
                "scope": "authoring",
                "severity": "required",
                "description": "fixture",
            }
        ],
        "observables": [
            {
                "id": "collision_rate",
                "metric": "collision_rate",
                "source": "benchmark_episode_metrics",
                "required": True,
                "interpretation": "fixture",
            }
        ],
        "termination_conditions": [
            {
                "id": "route_success",
                "reason": "success",
                "source": "benchmark_runner",
                "description": "fixture",
            }
        ],
        "certification": {
            "schema_version": "scenario_cert.v1",
            "required_before_benchmark_claim": True,
            "expected_eligibility": "eligible",
            "notes": "fixture",
        },
        "benchmark_eligibility": {
            "intended_use": "exploratory",
            "requires_certification": True,
            "claim_boundary": "metadata only",
            "eligibility_hooks": ["fixture"],
        },
        "provenance": {
            "source_issue": "fixture",
            "authored_by": "test",
            "source_files": ["tests/tools/test_hazard_odd_coverage_rollup.py"],
            "notes": "fixture",
        },
    }


def _write_scenario_certs(path: Path) -> Path:
    """Write a scenario certificate fixture with one excluded scenario."""

    payload = {
        "certificates": [
            {
                "schema_version": "scenario_cert.v1",
                "scenario_id": "s_excluded",
                "source": "fixture",
                "classification": "invalid",
                "benchmark_eligibility": "ineligible",
                "reasons": ["fixture exclusion"],
                "checks": {},
                "route_certificates": [],
            }
        ]
    }
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return path
