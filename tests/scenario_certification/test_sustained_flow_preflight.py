"""Tests issue #3813 sustained-flow scenario variant preflight."""

from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import yaml

from robot_sf.scenario_certification import sustained_flow

REPO_ROOT = Path(__file__).resolve().parents[2]
SCENARIO_SET = REPO_ROOT / "configs/scenarios/sets/issue_3813_sustained_flow_scaffold_v0.yaml"
PREFLIGHT_SCRIPT = REPO_ROOT / "scripts/validation/preflight_sustained_flow_scenarios_issue_3813.py"


def test_expected_variants_enumerate_deterministically() -> None:
    """Generator emits light/medium/heavy in deterministic order."""
    generated = sustained_flow.generate_expected_sustained_flow_scenarios()
    assert len(generated) == 3

    tiers = [scenario["metadata"]["density"] for scenario in generated]
    assert tiers == ["light", "medium", "heavy"]

    spawn_rates = [
        scenario["metadata"]["continuous_spawn"]["spawn_rate_per_min"] for scenario in generated
    ]
    assert spawn_rates == [6.0, 12.0, 18.0]


def test_generated_variants_pass_generator_preflight() -> None:
    """Canonical generator rows validate before YAML materialization."""

    report = sustained_flow.preflight_generated_sustained_flow_scenarios()
    payload = sustained_flow.sustained_flow_preflight_to_dict(report)

    assert payload["scenario_set"] == "generated:issue_3813_sustained_flow_scaffold_v0"
    assert payload["conforms"] is True
    assert payload["benchmark_evidence"] is False
    assert payload["runtime_support"] == "metadata_only"
    assert payload["runtime_definition_readiness"] == {
        "status": sustained_flow.RUNTIME_DEFINITION_METADATA_ONLY_STATUS,
        "ready": False,
        "expected_runtime_support": sustained_flow.SUSTAINED_FLOW_RUNTIME_SUPPORTED_VALUE,
        "observed_runtime_support": "metadata_only",
        "benchmark_evidence": False,
    }
    assert payload["variant_count"] == 3
    assert [variant["density_tier"] for variant in payload["variants"]] == [
        "light",
        "medium",
        "heavy",
    ]
    assert payload["errors"] == []


def test_runtime_supported_variants_enumerate_deterministically() -> None:
    """Runtime-supported generator preserves deterministic sustained-flow tiers."""

    generated = sustained_flow.generate_runtime_supported_sustained_flow_scenarios()

    assert [scenario["metadata"]["density"] for scenario in generated] == [
        "light",
        "medium",
        "heavy",
    ]
    assert {
        scenario["metadata"]["continuous_spawn"]["current_runtime_support"]
        for scenario in generated
    } == {sustained_flow.SUSTAINED_FLOW_RUNTIME_SUPPORTED_VALUE}


def test_runtime_supported_generated_variants_pass_generator_preflight() -> None:
    """Runtime-supported generated rows validate without becoming benchmark evidence."""

    report = sustained_flow.preflight_runtime_supported_generated_sustained_flow_scenarios()
    payload = sustained_flow.sustained_flow_preflight_to_dict(report)

    assert payload["scenario_set"] == (
        "generated:issue_3813_sustained_flow_scaffold_v0:runtime_supported"
    )
    assert payload["conforms"] is True
    assert payload["benchmark_evidence"] is False
    assert payload["runtime_support"] == sustained_flow.SUSTAINED_FLOW_RUNTIME_SUPPORTED_VALUE
    assert payload["runtime_definition_readiness"] == {
        "status": sustained_flow.RUNTIME_DEFINITION_READY_STATUS,
        "ready": True,
        "expected_runtime_support": sustained_flow.SUSTAINED_FLOW_RUNTIME_SUPPORTED_VALUE,
        "observed_runtime_support": sustained_flow.SUSTAINED_FLOW_RUNTIME_SUPPORTED_VALUE,
        "benchmark_evidence": False,
    }
    assert payload["variant_count"] == 3
    assert [variant["density_tier"] for variant in payload["variants"]] == [
        "light",
        "medium",
        "heavy",
    ]
    assert payload["errors"] == []


def test_runtime_supported_invalid_definition_reports_invalid_status() -> None:
    """Runtime-supported rows that fail validation are invalid, not metadata-only."""

    report = sustained_flow.preflight_runtime_supported_generated_sustained_flow_scenarios()
    invalid_report = replace(
        report,
        conforms=False,
        errors=("runtime-supported definition failed scenario_cert.v1 validation",),
    )

    payload = sustained_flow.sustained_flow_preflight_to_dict(invalid_report)

    assert payload["conforms"] is False
    assert payload["runtime_support"] == sustained_flow.SUSTAINED_FLOW_RUNTIME_SUPPORTED_VALUE
    assert payload["runtime_definition_readiness"] == {
        "status": sustained_flow.RUNTIME_DEFINITION_INVALID_STATUS,
        "ready": False,
        "expected_runtime_support": sustained_flow.SUSTAINED_FLOW_RUNTIME_SUPPORTED_VALUE,
        "observed_runtime_support": sustained_flow.SUSTAINED_FLOW_RUNTIME_SUPPORTED_VALUE,
        "benchmark_evidence": False,
    }


def test_preflight_conforms_for_commit_scaffold() -> None:
    """Current scaffold set satisfies fail-closed sustained-flow preflight."""
    report = sustained_flow.preflight_sustained_flow_scenario_set(SCENARIO_SET)

    assert report.conforms
    assert len(report.variants) == 3
    assert report.runtime_support == "metadata_only"
    assert report.benchmark_evidence is False
    assert [variant.density_tier for variant in report.variants] == ["light", "medium", "heavy"]
    assert [variant.spawn_rate_per_min for variant in report.variants] == [6.0, 12.0, 18.0]


def test_preflight_rejects_truncated_variant_set(tmp_path: Path) -> None:
    """Truncating variants must fail closed with an expected-count error."""
    payload = yaml.safe_load(SCENARIO_SET.read_text(encoding="utf-8"))
    payload["scenarios"] = payload["scenarios"][:2]

    truncated = tmp_path / "issue_3813_sustained_flow_scaffold_truncated.yaml"
    truncated.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")

    report = sustained_flow.preflight_sustained_flow_scenario_set(truncated)
    assert not report.conforms
    assert any("expected 3 sustained-flow variants" in error for error in report.errors)


def test_preflight_cli_outputs_json_payload() -> None:
    """Validation script emits JSON payload for automation consumers."""
    command = [
        sys.executable,
        str(PREFLIGHT_SCRIPT),
        "--scenario-set",
        str(SCENARIO_SET),
        "--json",
    ]
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    assert result.returncode == 0

    payload = json.loads(result.stdout)
    assert payload["schema_version"] == sustained_flow.SUSTAINED_FLOW_PREFLIGHT_SCHEMA_VERSION
    assert payload["conforms"] is True
    assert payload["variant_count"] == 3
    assert (
        payload["scenario_set"]
        == "configs/scenarios/sets/issue_3813_sustained_flow_scaffold_v0.yaml"
    )


def test_preflight_cli_outputs_generated_json_payload() -> None:
    """Validation script can preflight generator rows directly."""

    command = [
        sys.executable,
        str(PREFLIGHT_SCRIPT),
        "--generated",
        "--json",
    ]

    result = subprocess.run(command, capture_output=True, text=True, check=False)

    assert result.returncode == 0
    payload = json.loads(result.stdout)
    assert payload["schema_version"] == sustained_flow.SUSTAINED_FLOW_PREFLIGHT_SCHEMA_VERSION
    assert payload["conforms"] is True
    assert payload["variant_count"] == 3
    assert payload["scenario_set"] == "generated:issue_3813_sustained_flow_scaffold_v0"


def test_preflight_cli_outputs_runtime_supported_generated_json_payload() -> None:
    """Validation script can preflight runtime-supported generator rows directly."""

    command = [
        sys.executable,
        str(PREFLIGHT_SCRIPT),
        "--runtime-supported-generated",
        "--json",
    ]

    result = subprocess.run(command, capture_output=True, text=True, check=False)

    assert result.returncode == 0
    payload = json.loads(result.stdout)
    assert payload["schema_version"] == sustained_flow.SUSTAINED_FLOW_PREFLIGHT_SCHEMA_VERSION
    assert payload["conforms"] is True
    assert payload["benchmark_evidence"] is False
    assert payload["runtime_support"] == sustained_flow.SUSTAINED_FLOW_RUNTIME_SUPPORTED_VALUE
    assert payload["variant_count"] == 3
    assert payload["scenario_set"] == (
        "generated:issue_3813_sustained_flow_scaffold_v0:runtime_supported"
    )


def test_runtime_supported_generated_cli_ignores_scenario_set_argument(tmp_path: Path) -> None:
    """Runtime-supported generator mode does not validate an overridden scenario-set path."""

    command = [
        sys.executable,
        str(PREFLIGHT_SCRIPT),
        "--scenario-set",
        str(tmp_path / "missing.yaml"),
        "--runtime-supported-generated",
        "--json",
    ]

    result = subprocess.run(command, capture_output=True, text=True, check=False)

    assert result.returncode == 0
    payload = json.loads(result.stdout)
    assert payload["conforms"] is True
    assert payload["errors"] == []
    assert payload["runtime_support"] == sustained_flow.SUSTAINED_FLOW_RUNTIME_SUPPORTED_VALUE
