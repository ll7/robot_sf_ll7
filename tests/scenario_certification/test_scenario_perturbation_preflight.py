"""Tests for scenario perturbation manifest preflight contracts."""

from __future__ import annotations

import json
from pathlib import Path

import jsonschema
import pytest
import yaml

from robot_sf.scenario_certification.perturbation_preflight import (
    PERTURBATION_MANIFEST_SCHEMA_VERSION,
    preflight_perturbation_manifest,
    preflight_to_dict,
)


def _write_manifest(path: Path, *, max_route_offset_m: float = 0.5) -> Path:
    """Write a tiny perturbation manifest that targets an existing simple scenario."""
    payload = {
        "schema_version": PERTURBATION_MANIFEST_SCHEMA_VERSION,
        "manifest_id": "test_perturbation_manifest",
        "scenario_config": "configs/scenarios/single/planner_sanity_simple.yaml",
        "seed_controls": {
            "baseline_seeds": [101],
            "replay_seed_policy": "explicit",
        },
        "validity": {
            "require_scenario_certification": True,
            "max_route_offset_m": max_route_offset_m,
            "invalid_variant_evidence_policy": "exclude_from_success_evidence",
        },
        "variants": [
            {
                "variant_id": "planner_sanity_simple_noop",
                "scenario_id": "planner_sanity_simple",
                "family": "noop",
                "seeds": [101],
            },
            {
                "variant_id": "planner_sanity_simple_route_offset",
                "scenario_id": "planner_sanity_simple",
                "family": "robot_route_offset",
                "seeds": [101],
                "parameters": {
                    "dx_m": 0.25,
                    "dy_m": 0.0,
                    "max_magnitude_m": max_route_offset_m,
                },
            },
        ],
    }
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return path


def test_manifest_schema_accepts_noop_and_bounded_route_offset(tmp_path: Path) -> None:
    """The public schema should document the first supported perturbation surface."""
    manifest_path = _write_manifest(tmp_path / "perturbations.yaml")
    manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    schema = json.loads(
        Path("robot_sf/benchmark/schemas/scenario_perturbation_manifest.v1.json").read_text(
            encoding="utf-8"
        )
    )

    jsonschema.validate(manifest, schema)


def test_preflight_rejects_manifest_that_violates_public_schema(tmp_path: Path) -> None:
    """Runtime preflight should enforce the same schema reviewers validate."""
    manifest_path = _write_manifest(tmp_path / "perturbations.yaml")
    manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    manifest["validity"]["require_scenario_certification"] = False
    manifest_path.write_text(yaml.safe_dump(manifest, sort_keys=False), encoding="utf-8")

    with pytest.raises(ValueError, match="validity.require_scenario_certification"):
        preflight_perturbation_manifest(manifest_path)


def test_preflight_certifies_noop_and_bounded_route_offset(tmp_path: Path) -> None:
    """Valid variants should carry scenario certificates and remain eligible evidence candidates."""
    manifest_path = _write_manifest(tmp_path / "perturbations.yaml")

    report = preflight_perturbation_manifest(manifest_path)

    assert report.schema_version == "scenario_perturbation_preflight.v1"
    assert [result.variant_id for result in report.results] == [
        "planner_sanity_simple_noop",
        "planner_sanity_simple_route_offset",
    ]
    assert {result.validity_status for result in report.results} == {"valid"}
    assert {result.benchmark_evidence_status for result in report.results} == {
        "eligible_success_evidence_candidate"
    }
    assert all(result.certificate is not None for result in report.results)
    assert report.results[1].perturbation_summary["magnitude_m"] == pytest.approx(0.25)


def test_preflight_excludes_unbounded_route_offset_before_certification(tmp_path: Path) -> None:
    """Out-of-bounds perturbations should fail closed without becoming benchmark evidence."""
    manifest_path = _write_manifest(tmp_path / "perturbations.yaml", max_route_offset_m=0.2)

    report = preflight_perturbation_manifest(manifest_path)

    excluded = report.results[1]
    assert excluded.validity_status == "invalid"
    assert excluded.benchmark_evidence_status == "excluded_from_success_evidence"
    assert excluded.certificate is None
    assert excluded.reasons == [
        "robot_route_offset magnitude 0.250000 m exceeds variant max_magnitude_m 0.200000 m"
    ]
    payload = preflight_to_dict(report)
    assert payload["results"][1]["certificate"] is None
