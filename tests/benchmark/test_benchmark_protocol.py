"""Tests for the AMMV benchmark protocol manifest loader and validator."""

from __future__ import annotations

from pathlib import Path

import pytest

from robot_sf.benchmark.benchmark_protocol import (
    AMMV_BENCHMARK_PROTOCOL_PATH,
    BenchmarkProtocolError,
    load_benchmark_protocol,
    validate_benchmark_protocol_payload,
)


def test_load_checked_in_ammv_benchmark_protocol() -> None:
    """The checked-in protocol manifest should load with expected values."""
    manifest = load_benchmark_protocol()
    assert manifest.protocol_id == "ammv_benchmark_v0"
    assert "crossing" in manifest.scenario_classes
    assert "goal" in manifest.planner_panel
    assert "safety_gate" in manifest.metric_layers
    assert manifest.claim_rules.global_superiority_forbidden is True
    assert manifest.claim_rules.report_tradeoffs is True
    assert manifest.claim_rules.safety_gate_precedes_efficiency is True


def test_benchmark_protocol_rejects_missing_required_section(tmp_path: Path) -> None:
    """Manifest missing a required top-level section should be rejected."""
    path = tmp_path / "missing_claim_rules.yaml"
    path.write_text(
        "scenario_classes: [crossing]\nplanner_panel: [goal]\nmetric_layers: [safety_gate]\n",
        encoding="utf-8",
    )

    with pytest.raises(BenchmarkProtocolError, match="claim_rules"):
        load_benchmark_protocol(path)


def test_benchmark_protocol_rejects_missing_claim_rule(tmp_path: Path) -> None:
    """Manifest with missing required claim rule should be rejected."""
    path = tmp_path / "missing_rule.yaml"
    path.write_text(
        "scenario_classes: [crossing]\n"
        "planner_panel: [goal]\n"
        "metric_layers: [safety_gate]\n"
        "claim_rules:\n"
        "  global_superiority_forbidden: true\n"
        "  report_tradeoffs: true\n",
        encoding="utf-8",
    )

    with pytest.raises(BenchmarkProtocolError, match="safety_gate_precedes_efficiency"):
        load_benchmark_protocol(path)


def test_benchmark_protocol_rejects_non_bool_claim_rule(tmp_path: Path) -> None:
    """A claim rule with a non-bool value should be rejected."""
    path = tmp_path / "non_bool_rule.yaml"
    path.write_text(
        "scenario_classes: [crossing]\n"
        "planner_panel: [goal]\n"
        "metric_layers: [safety_gate]\n"
        "claim_rules:\n"
        '  global_superiority_forbidden: "true"\n'
        "  report_tradeoffs: true\n"
        "  safety_gate_precedes_efficiency: true\n",
        encoding="utf-8",
    )

    with pytest.raises(BenchmarkProtocolError, match="global_superiority_forbidden"):
        load_benchmark_protocol(path)


def test_benchmark_protocol_rejects_empty_section(tmp_path: Path) -> None:
    """A required list-typed section that is empty should be rejected."""
    path = tmp_path / "empty_scenarios.yaml"
    path.write_text(
        "scenario_classes: []\n"
        "planner_panel: [goal]\n"
        "metric_layers: [safety_gate]\n"
        "claim_rules:\n"
        "  global_superiority_forbidden: true\n"
        "  report_tradeoffs: true\n"
        "  safety_gate_precedes_efficiency: true\n",
        encoding="utf-8",
    )

    with pytest.raises(BenchmarkProtocolError, match="scenario_classes"):
        load_benchmark_protocol(path)


def test_benchmark_protocol_rejects_non_list_section(tmp_path: Path) -> None:
    """A section that should be a list but is not should be rejected."""
    path = tmp_path / "non_list_planners.yaml"
    path.write_text(
        "scenario_classes: [crossing]\n"
        "planner_panel: goal\n"
        "metric_layers: [safety_gate]\n"
        "claim_rules:\n"
        "  global_superiority_forbidden: true\n"
        "  report_tradeoffs: true\n"
        "  safety_gate_precedes_efficiency: true\n",
        encoding="utf-8",
    )

    with pytest.raises(BenchmarkProtocolError, match="planner_panel"):
        load_benchmark_protocol(path)


def test_benchmark_protocol_rejects_non_mapping_root(tmp_path: Path) -> None:
    """A YAML file that is not a mapping at the top level should be rejected."""
    path = tmp_path / "list_root.yaml"
    path.write_text("[crossing, goal]\n", encoding="utf-8")

    with pytest.raises(BenchmarkProtocolError, match="root"):
        load_benchmark_protocol(path)


def test_validate_payload_accepts_minimal_valid(tmp_path: Path) -> None:
    """A minimal valid payload should produce a typed manifest."""
    payload = {
        "scenario_classes": ["crossing"],
        "planner_panel": ["goal"],
        "metric_layers": ["safety_gate"],
        "claim_rules": {
            "global_superiority_forbidden": True,
            "report_tradeoffs": True,
            "safety_gate_precedes_efficiency": True,
        },
    }
    manifest = validate_benchmark_protocol_payload(payload, source_path=tmp_path / "test.yaml")
    assert manifest.protocol_id == "test"
    assert manifest.scenario_classes == ("crossing",)
    assert manifest.planner_panel == ("goal",)
    assert manifest.metric_layers == ("safety_gate",)
    assert manifest.claim_rules.global_superiority_forbidden is True


def test_benchmark_protocol_derives_protocol_id_from_stem(tmp_path: Path) -> None:
    """The protocol id should be derived from the YAML file stem."""
    path = tmp_path / "custom_protocol_v1.yaml"
    path.write_text(
        "scenario_classes: [crossing]\n"
        "planner_panel: [goal]\n"
        "metric_layers: [safety_gate]\n"
        "claim_rules:\n"
        "  global_superiority_forbidden: true\n"
        "  report_tradeoffs: true\n"
        "  safety_gate_precedes_efficiency: true\n",
        encoding="utf-8",
    )
    manifest = load_benchmark_protocol(path)
    assert manifest.protocol_id == "custom_protocol_v1"


def test_benchmark_protocol_default_path_points_to_checked_in_manifest() -> None:
    """The default path constant should point at the checked-in manifest file."""
    assert AMMV_BENCHMARK_PROTOCOL_PATH == Path("benchmarks/ammv_benchmark_v0.yaml")
