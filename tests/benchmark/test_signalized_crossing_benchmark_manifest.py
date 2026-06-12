"""Tests for the issue #2474 signalized crossing benchmark manifest."""

from __future__ import annotations

from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
MANIFEST_PATH = REPO_ROOT / "configs/benchmarks/signalized_pedestrian_crossing_issue_2474.yaml"


def _manifest() -> dict[str, object]:
    """Load the #2474 signalized crossing benchmark manifest."""
    payload = yaml.safe_load(MANIFEST_PATH.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return payload


def test_signalized_crossing_manifest_references_existing_surfaces() -> None:
    """Every tracked surface in the manifest should exist."""
    manifest = _manifest()
    surfaces = manifest["contract_surfaces"]
    assert isinstance(surfaces, dict)

    for rel_path in surfaces.values():
        assert (REPO_ROOT / str(rel_path)).exists(), rel_path
    assert surfaces["signal_state_proxy_scenario"] == (
        "configs/scenarios/single/issue_2527_waiting_then_crossing.yaml"
    )


def test_scenario_families_distinguish_proxy_and_future_signal_runtime() -> None:
    """The manifest should not mistake current wait_at proxies for signal runtime support."""
    manifest = _manifest()
    families = manifest["scenario_families"]
    assert isinstance(families, list)

    statuses = {family["required_runtime_status"] for family in families}
    assert "proxy_smoke_available" in statuses
    assert "future_signal_state_required" in statuses
    for family in families:
        assert family["initial_case_ids"], family["key"]
        assert (REPO_ROOT / str(family["nearest_existing_proxy"])).exists()


def test_signal_phase_contract_lists_future_fields_and_current_proxy_fields() -> None:
    """Signal phase semantics should name future explicit fields and current proxy controls."""
    manifest = _manifest()
    phase = manifest["signal_phase_semantics"]
    assert isinstance(phase, dict)

    future_fields = set(phase["future_explicit_fields"])
    proxy_fields = set(phase["proxy_fields_available_now"])
    assert {"phase", "phase_remaining_s", "legality_state"}.issubset(future_fields)
    assert {"wait_at", "signal_state", "single_pedestrian_wait_duration_offset"}.issubset(
        proxy_fields
    )
    assert "planner-observable" in phase["planner_observable_policy"]


def test_signal_state_promotion_contract_separates_proxy_observable_and_unavailable() -> None:
    """Signal-state rows need explicit promotion before entering benchmark evidence."""
    manifest = _manifest()
    contract = manifest["signal_state_promotion_contract"]
    assert isinstance(contract, dict)
    states = contract["states"]
    assert isinstance(states, dict)

    proxy = states["proxy_diagnostic"]
    observable = states["planner_observable"]
    unavailable = states["unavailable"]

    assert proxy["planner_consumed_fields"] == []
    assert "phase" in proxy["recorded_only_fields"]
    assert "traffic-light semantics" in proxy["fail_closed_reason"]

    assert observable["required_schema_version"] == "signal-state-observable.v1"
    assert observable["required_status"] == "planner_observable_signal_state"
    assert observable["required_observation_mode"] == "planner_observable"
    assert observable["required_benchmark_evidence"] is True
    assert "phase_remaining_s" in observable["planner_consumed_fields"]

    assert unavailable["planner_consumed_fields"] == []
    assert unavailable["fail_closed_reason"] == "signal_state_metadata_absent"
    assert "Proxy_diagnostic and unavailable rows are excluded" in contract["denominator_policy"]


def test_required_metrics_and_trace_fields_cover_signal_specific_contract() -> None:
    """The benchmark direction should specify existing and signal-specific evidence fields."""
    manifest = _manifest()
    metrics = manifest["required_metrics"]
    trace_fields = manifest["required_trace_fields"]
    assert isinstance(metrics, dict)
    assert isinstance(trace_fields, dict)

    assert "success" in metrics["existing_metrics"]
    assert "forced_wait_duration_s" in metrics["new_signal_metrics"]
    assert "signal_phase_violation_count" in metrics["new_signal_metrics"]
    assert "outcome.collision_event" in trace_fields["existing_trace_like_fields"]
    assert "signal_phase_timeline" in trace_fields["future_signal_trace_fields"]
    assert "forced_wait_intervals" in trace_fields["future_signal_trace_fields"]


def test_fallback_policy_and_claim_boundary_reject_benchmark_claims_for_proxy_smoke() -> None:
    """Proxy smoke rows should remain non-benchmark evidence until signal runtime exists."""
    manifest = _manifest()
    reporting = manifest["fallback_and_reporting_policy"]
    claim_boundary = str(manifest["claim_boundary"]).lower()
    first_spike = manifest["first_smoke_or_spike"]
    assert isinstance(reporting, dict)
    assert isinstance(first_spike, dict)

    assert manifest["benchmark_evidence"] is False
    assert "does not prove" in claim_boundary
    assert "forced_waiting_reasoning_claim" in reporting["proxy_smoke_not_allowed_for"]
    assert (
        "no fallback/degraded/not_available row counted as benchmark success"
        in (reporting["benchmark_success_requires"])
    )
    assert first_spike["proxy_config"] == (
        "configs/scenarios/perturbations/issue_1610_intersection_wait_phase_grid_v1.yaml"
    )
