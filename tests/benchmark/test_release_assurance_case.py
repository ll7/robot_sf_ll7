"""Tests for release assurance case export and stale-reference checks."""

from __future__ import annotations

import json
from pathlib import Path

from robot_sf.benchmark.release_assurance_case import (
    build_release_assurance_case_from_paths,
    validate_release_assurance_case_references,
    validate_release_assurance_case_schema,
)

SMOKE_MANIFEST = Path(
    "configs/benchmarks/releases/paper_experiment_matrix_v1_release_smoke_v0_1.yaml"
)
GATE_SPEC = Path("configs/benchmarks/release_gates/default_safety_comfort_gates.yaml")


def test_release_assurance_case_exports_gate_claims_and_manifest_evidence() -> None:
    """Exporter maps gate specs and release manifest inputs to claim evidence."""

    payload = build_release_assurance_case_from_paths(
        manifest_path=SMOKE_MANIFEST,
        gate_spec_path=GATE_SPEC,
        generated_at_utc="2026-07-06T00:00:00Z",
    )

    validate_release_assurance_case_schema(payload)
    assert validate_release_assurance_case_references(payload) == []
    assert payload["schema_version"] == "benchmark_release_assurance_case.v1"
    assert payload["review_marker"] == "AI-GENERATED NEEDS-REVIEW"
    assert payload["release"]["release_id"] == "paper_experiment_matrix_v1_smoke_v0_1_0"
    assert "C_gate_collision_rate_zero" in {claim["id"] for claim in payload["claims"]}
    assert "E_release_gate_spec" in {leaf["id"] for leaf in payload["evidence"]}
    campaign_config = next(
        leaf for leaf in payload["evidence"] if leaf["id"] == "E_campaign_config"
    )
    assert campaign_config["sha256"] == campaign_config["expected_sha256"]


def test_release_assurance_case_reference_check_fails_on_stale_sha(tmp_path: Path) -> None:
    """Checker fails closed when a referenced evidence checksum goes stale."""

    payload = build_release_assurance_case_from_paths(
        manifest_path=SMOKE_MANIFEST,
        gate_spec_path=GATE_SPEC,
        generated_at_utc="2026-07-06T00:00:00Z",
    )
    stale_payload = json.loads(json.dumps(payload))
    stale_payload["evidence"][0]["sha256"] = "0" * 64

    problems = validate_release_assurance_case_references(stale_payload)

    assert problems
    assert problems[0].evidence_id == "E_release_manifest"
    assert "sha256 mismatch" in problems[0].reason


def test_checked_in_release_assurance_case_example_stays_fresh() -> None:
    """Worked example remains schema-valid and evidence-reference fresh."""

    payload = json.loads(
        Path("docs/context/evidence/issue_4683_release_assurance_case_example.json").read_text(
            encoding="utf-8"
        )
    )

    validate_release_assurance_case_schema(payload)
    assert validate_release_assurance_case_references(payload) == []
