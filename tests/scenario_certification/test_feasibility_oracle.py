"""Tests for the issue #5137 planner-free feasibility oracle + envelope-sensitivity axis.

These tests exercise the new capability introduced for issue #5137:
  * per-cell planner-free feasibility verdict with explicit margins
    (min corridor width vs envelope diameter, minimum completion steps vs horizon),
  * an envelope-sensitivity axis that re-runs the oracle at reduced envelope radius,
  * campaign-metadata annotation of zero-completion cells.

They do NOT duplicate the issue #3484 feasibility-diagnostics tests: #3484 has no
envelope axis and no margin assembly. The oracle reuses #3484's actor-free rollout
helper, so these tests inject deterministic certifier/runner stubs for unit coverage
plus one real end-to-end rollout for integration coverage.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from robot_sf.scenario_certification import feasibility_oracle
from robot_sf.scenario_certification.feasibility_oracle import (
    BLOCKED,
    CAMPAIGN_FEASIBILITY_ANNOTATION_SCHEMA,
    DEFAULT_ENVELOPE_RADII_M,
    ENVELOPE_SENSITIVE_HARD,
    ENVELOPE_SENSITIVITY_SCHEMA,
    FEASIBILITY_ORACLE_SCHEMA,
    FEASIBLE,
    INFEASIBLE_BY_CONSTRUCTION,
    PLANNER_LIMITED,
    TIME_TRUNCATED,
    CompletionMargin,
    EnvelopeSensitivityVerdict,
    FeasibilityOracleConfig,
    FeasibilityVerdict,
    GeometricMargin,
    annotate_zero_completion_cells,
    envelope_sensitivity_verdict_to_dict,
    feasibility_verdict_to_dict,
    make_envelope_scenario,
    run_envelope_sensitivity_sweep,
    run_feasibility_oracle,
)
from robot_sf.scenario_certification.v1 import (
    CERT_SCHEMA_VERSION,
    GEOMETRICALLY_INFEASIBLE,
    VALID,
    RouteCertificate,
    ScenarioCertificate,
)

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SCENARIO_PATH = _REPO_ROOT / "configs/scenarios/archetypes/classic_head_on_corridor.yaml"


def _route_certificate(
    *,
    classification: str = VALID,
    eligibility: str = "eligible",
    minimum_static_clearance_m: float | None = 0.5,
    shortest_path_length_m: float | None = 20.0,
    inflated_path: bool = True,
) -> RouteCertificate:
    """Build a route certificate with controlled geometric checks."""
    checks: dict[str, Any] = {"robot_radius_m": 1.0}
    if minimum_static_clearance_m is not None:
        checks["minimum_static_clearance_m"] = minimum_static_clearance_m
    if shortest_path_length_m is not None:
        checks["shortest_path_length_m"] = shortest_path_length_m
    checks["inflated_collision_free_path"] = inflated_path
    return RouteCertificate(
        route_id="route_0",
        spawn_id=0,
        goal_id=0,
        classification=classification,
        benchmark_eligibility=eligibility,
        reasons=[],
        checks=checks,
    )


def _certificate(
    classification: str = VALID,
    *,
    eligibility: str = "eligible",
    minimum_static_clearance_m: float | None = 0.5,
    shortest_path_length_m: float | None = 20.0,
    inflated_path: bool = True,
) -> ScenarioCertificate:
    """Build a scenario certificate wrapping one controlled route certificate."""
    return ScenarioCertificate(
        schema_version=CERT_SCHEMA_VERSION,
        scenario_id="classic_head_on_corridor_low",
        source="fixture",
        classification=classification,
        benchmark_eligibility=eligibility,
        reasons=[],
        checks={"route_count": 1},
        route_certificates=[
            _route_certificate(
                classification=classification,
                eligibility=eligibility,
                minimum_static_clearance_m=minimum_static_clearance_m,
                shortest_path_length_m=shortest_path_length_m,
                inflated_path=inflated_path,
            )
        ],
    )


def _scenario(
    *,
    name: str = "classic_head_on_corridor_low",
    archetype: str = "head_on_corridor",
    max_episode_steps: int = 500,
    robot_config: dict[str, Any] | None = None,
    seed: int = 111,
) -> dict[str, Any]:
    """Build a minimal scenario fixture."""
    return {
        "name": name,
        "simulation_config": {"max_episode_steps": max_episode_steps, "ped_density": 0.02},
        "robot_config": dict(robot_config or {}),
        "metadata": {"archetype": archetype},
        "seeds": [seed],
    }


def _oracle_config(radii: tuple[float, ...] = (1.0, 0.5)) -> FeasibilityOracleConfig:
    return FeasibilityOracleConfig(scenario_path=_SCENARIO_PATH, envelope_radii_m=radii)


# ---------------------------------------------------------------------------
# make_envelope_scenario
# ---------------------------------------------------------------------------


def test_make_envelope_scenario_overrides_radius_and_preserves_other_fields() -> None:
    """Envelope override injects radius while preserving robot type and route identity."""
    scenario = _scenario(robot_config={"type": "differential_drive", "max_linear_speed": 1.5})

    mutated = make_envelope_scenario(scenario, envelope_radius_m=0.5)

    assert mutated["robot_config"]["radius"] == 0.5
    assert mutated["robot_config"]["type"] == "differential_drive"
    assert mutated["robot_config"]["max_linear_speed"] == 1.5
    assert mutated["metadata"]["envelope_probe_radius_m"] == 0.5
    assert scenario["robot_config"] == {"type": "differential_drive", "max_linear_speed": 1.5}


def test_make_envelope_scenario_rejects_non_positive_radius() -> None:
    """A non-positive envelope radius is rejected."""
    with pytest.raises(ValueError, match="envelope_radius_m"):
        make_envelope_scenario(_scenario(), envelope_radius_m=0.0)


# ---------------------------------------------------------------------------
# run_feasibility_oracle - margin reporting
# ---------------------------------------------------------------------------


def test_oracle_reports_corridor_vs_envelope_margin_for_feasible_route() -> None:
    """A feasible route reports corridor width, envelope diameter, and their margin."""

    def runner(_s, _seed, _horizon, _algo):
        return {"steps": 200, "horizon": 500, "outcome": {"route_complete": True}}

    verdict = run_feasibility_oracle(
        _scenario(),
        config=_oracle_config(),
        envelope_radius_m=1.0,
        episode_runner=runner,
        certifier=lambda _s, _p: _certificate(VALID, minimum_static_clearance_m=1.5),
    )

    # minimum_static_clearance_m=1.5, radius=1.0 -> corridor width = 2*(1.5+1.0)=5.0
    # envelope diameter = 2.0; margin = 5.0 - 2.0 = 3.0
    assert verdict.feasible is True
    assert verdict.status == FEASIBLE
    assert verdict.geometric.route_geometrically_feasible is True
    assert verdict.geometric.envelope_diameter_m == pytest.approx(2.0)
    assert verdict.geometric.min_corridor_width_m == pytest.approx(5.0)
    assert verdict.geometric.corridor_envelope_margin_m == pytest.approx(3.0)
    assert verdict.completion.route_completion_feasible is True
    assert verdict.completion.min_completion_steps == 200
    assert verdict.completion.horizon_steps == 500
    assert verdict.completion.completion_horizon_margin_steps == 300


def test_oracle_reports_infeasible_by_construction_when_no_inflated_path() -> None:
    """A geometrically infeasible route is infeasible by construction and skips rollout."""

    def runner(_s, _seed, _horizon, _algo):  # pragma: no cover - must not be called
        pytest.fail("rollout must not run for a geometrically infeasible route")

    verdict = run_feasibility_oracle(
        _scenario(),
        config=_oracle_config(),
        envelope_radius_m=1.0,
        episode_runner=runner,
        certifier=lambda _s, _p: _certificate(GEOMETRICALLY_INFEASIBLE, eligibility="excluded"),
    )

    assert verdict.feasible is False
    assert verdict.status == INFEASIBLE_BY_CONSTRUCTION
    assert verdict.geometric.route_geometrically_feasible is False
    assert verdict.completion.route_completion_feasible is False
    assert verdict.completion.status == "failed"
    assert verdict.completion.blocker == "route_geometrically_infeasible_no_traversal_path"


def test_oracle_reports_time_truncated_when_geometric_ok_but_rollout_times_out() -> None:
    """A geometrically feasible route that times out is time-truncated."""

    def runner(_s, _seed, _horizon, _algo):
        return {"steps": 500, "horizon": 500, "termination_reason": "max_steps"}

    verdict = run_feasibility_oracle(
        _scenario(),
        config=_oracle_config(),
        envelope_radius_m=1.0,
        episode_runner=runner,
        certifier=lambda _s, _p: _certificate(VALID),
    )

    assert verdict.feasible is False
    assert verdict.status == TIME_TRUNCATED
    assert verdict.geometric.route_geometrically_feasible is True
    assert verdict.completion.route_completion_feasible is False
    assert verdict.completion.min_completion_steps is None


def test_oracle_corridor_margin_is_none_when_no_static_obstacles() -> None:
    """An empty obstacle set yields None corridor width (no clearance reported)."""

    def runner(_s, _seed, _horizon, _algo):
        return {"outcome": {"route_complete": True}, "steps": 100}

    verdict = run_feasibility_oracle(
        _scenario(),
        config=_oracle_config(),
        envelope_radius_m=1.0,
        episode_runner=runner,
        certifier=lambda _s, _p: _certificate(VALID, minimum_static_clearance_m=None),
    )

    assert verdict.geometric.min_corridor_width_m is None
    assert verdict.geometric.corridor_envelope_margin_m is None
    assert verdict.feasible is True


def test_oracle_fails_closed_when_certifier_raises() -> None:
    """A raising certifier produces a blocked geometric margin, not a crash."""

    def _raise(_s, _p):
        raise RuntimeError("boom")

    def runner(_s, _seed, _horizon, _algo):
        return {"outcome": {"route_complete": True}, "steps": 100}

    verdict = run_feasibility_oracle(
        _scenario(),
        config=_oracle_config(),
        envelope_radius_m=1.0,
        episode_runner=runner,
        certifier=_raise,
    )

    assert verdict.geometric.route_geometrically_feasible is None
    assert verdict.geometric.classification.startswith("blocked:")
    assert verdict.status == BLOCKED
    assert verdict.feasible is None


def test_oracle_fails_closed_when_rollout_raises() -> None:
    """A raising rollout produces a blocked completion margin, not a crash."""

    def _raise(_s, _seed, _horizon, _algo):
        raise RuntimeError("rollout boom")

    verdict = run_feasibility_oracle(
        _scenario(),
        config=_oracle_config(),
        envelope_radius_m=1.0,
        episode_runner=_raise,
        certifier=lambda _s, _p: _certificate(VALID),
    )

    assert verdict.completion.status == "blocked"
    assert verdict.completion.route_completion_feasible is None
    assert verdict.status == BLOCKED


# ---------------------------------------------------------------------------
# run_envelope_sensitivity_sweep
# ---------------------------------------------------------------------------


def test_envelope_sweep_classifies_feasible_when_nominal_envelope_feasible() -> None:
    """A nominal-feasible cell is classified feasible."""

    def runner(_s, _seed, _horizon, _algo):
        return {"outcome": {"route_complete": True}, "steps": 200}

    verdict = run_envelope_sensitivity_sweep(
        _scenario(),
        config=_oracle_config((1.0, 0.5)),
        episode_runner=runner,
        certifier=lambda _s, _p: _certificate(VALID),
    )

    assert verdict.category == FEASIBLE
    assert verdict.nominal_envelope_radius_m == 1.0
    assert len(verdict.reduced_verdicts) == 1
    assert verdict.reduced_verdicts[0].envelope_radius_m == 0.5


def test_envelope_sweep_classifies_infeasible_by_construction() -> None:
    """Infeasible at every envelope radius (including reduced) is infeasible by construction."""

    def runner(_s, _seed, _horizon, _algo):  # pragma: no cover - skipped when infeasible
        pytest.fail("rollout must not run for geometrically infeasible routes")

    verdict = run_envelope_sensitivity_sweep(
        _scenario(),
        config=_oracle_config((1.0, 0.5)),
        episode_runner=runner,
        certifier=lambda _s, _p: _certificate(GEOMETRICALLY_INFEASIBLE, eligibility="excluded"),
    )

    assert verdict.category == INFEASIBLE_BY_CONSTRUCTION
    assert verdict.nominal_verdict.feasible is False
    assert all(v.feasible is False for v in verdict.reduced_verdicts)


def test_envelope_sweep_classifies_envelope_sensitive_hard() -> None:
    """Infeasible at nominal envelope but feasible at reduced envelope is envelope-sensitive-hard."""

    # The stub certifier inspects the injected robot radius to flip feasibility.
    def certifier(scenario, _p):
        radius = float((scenario.get("robot_config") or {}).get("radius", 1.0))
        if radius > 0.6:
            return _certificate(GEOMETRICALLY_INFEASIBLE, eligibility="excluded")
        return _certificate(VALID)

    def runner(_s, _seed, _horizon, _algo):
        return {"outcome": {"route_complete": True}, "steps": 200}

    verdict = run_envelope_sensitivity_sweep(
        _scenario(),
        config=_oracle_config((1.0, 0.5)),
        episode_runner=runner,
        certifier=certifier,
    )

    assert verdict.category == ENVELOPE_SENSITIVE_HARD
    assert verdict.nominal_verdict.feasible is False
    assert verdict.reduced_verdicts[0].feasible is True


def test_envelope_sweep_blocks_when_nominal_verdict_is_blocked() -> None:
    """A blocked nominal verdict propagates a blocked category."""

    def _raise(_s, _p):
        raise RuntimeError("cert boom")

    def runner(_s, _seed, _horizon, _algo):
        return {"outcome": {"route_complete": True}, "steps": 200}

    verdict = run_envelope_sensitivity_sweep(
        _scenario(),
        config=_oracle_config((1.0, 0.5)),
        episode_runner=runner,
        certifier=_raise,
    )

    assert verdict.category == BLOCKED


def test_oracle_blocks_when_default_runner_setup_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    """Default-runner construction failures become blocked rather than escaping the oracle."""

    def _raise(_config: FeasibilityOracleConfig):
        raise ImportError("benchmark runner unavailable")

    monkeypatch.setattr(feasibility_oracle, "_default_actor_free_runner", _raise)
    verdict = run_feasibility_oracle(
        _scenario(),
        config=_oracle_config((1.0,)),
        envelope_radius_m=1.0,
        certifier=lambda _s, _p: _certificate(VALID),
    )

    assert verdict.status == BLOCKED
    assert verdict.completion.status == BLOCKED
    assert verdict.completion.blocker == "rollout_error: benchmark runner unavailable"


def test_envelope_sweep_blocks_when_a_reduced_probe_is_unobserved() -> None:
    """A blocked reduced probe cannot establish a map-artifact classification."""

    def certifier(scenario, _p):
        radius = float((scenario.get("robot_config") or {}).get("radius", 1.0))
        if radius < 0.6:
            raise RuntimeError("reduced probe unavailable")
        return _certificate(GEOMETRICALLY_INFEASIBLE, eligibility="excluded")

    verdict = run_envelope_sensitivity_sweep(
        _scenario(),
        config=_oracle_config((1.0, 0.5)),
        certifier=certifier,
    )

    assert verdict.nominal_verdict.status == INFEASIBLE_BY_CONSTRUCTION
    assert verdict.reduced_verdicts[0].status == BLOCKED
    assert verdict.category == BLOCKED


def test_envelope_sweep_rejects_empty_radii() -> None:
    """An empty envelope-radii configuration is rejected."""
    with pytest.raises(ValueError, match="envelope_radii_m"):
        FeasibilityOracleConfig(scenario_path=_SCENARIO_PATH, envelope_radii_m=())


def test_envelope_sweep_rejects_duplicate_radii_in_nominal_first_order() -> None:
    """Duplicate nominal-first probes are rejected instead of silently re-running a radius."""
    with pytest.raises(ValueError, match="envelope_radii_m"):
        FeasibilityOracleConfig(scenario_path=_SCENARIO_PATH, envelope_radii_m=(1.0, 1.0, 0.5))


# ---------------------------------------------------------------------------
# annotate_zero_completion_cells
# ---------------------------------------------------------------------------


def _envelope_verdict(
    scenario_id: str,
    *,
    category: str,
) -> EnvelopeSensitivityVerdict:
    """Build a minimal envelope-sensitivity verdict for annotation tests."""
    geom = GeometricMargin(
        envelope_radius_m=1.0,
        envelope_diameter_m=2.0,
        route_geometrically_feasible=True,
        min_corridor_width_m=4.0,
        corridor_envelope_margin_m=2.0,
        min_static_clearance_m=1.0,
        shortest_path_length_m=20.0,
        classification=VALID,
        benchmark_eligibility="eligible",
    )
    completion = CompletionMargin(
        route_completion_feasible=True,
        min_completion_steps=200,
        horizon_steps=500,
        completion_horizon_margin_steps=300,
        kinematic_min_steps_lower_bound=100.0,
        termination_reason="success",
        status="passed",
    )
    nominal = FeasibilityVerdict(
        scenario_id=scenario_id,
        family_id="head_on_corridor",
        envelope_radius_m=1.0,
        geometric=geom,
        completion=completion,
        feasible=category == FEASIBLE,
        status=FEASIBLE if category == FEASIBLE else INFEASIBLE_BY_CONSTRUCTION,
    )
    return EnvelopeSensitivityVerdict(
        scenario_id=scenario_id,
        family_id="head_on_corridor",
        nominal_envelope_radius_m=1.0,
        nominal_verdict=nominal,
        reduced_verdicts=(),
        category=category,
    )


def test_annotate_flags_zero_completion_infeasible_by_construction() -> None:
    """A zero-completion cell that is infeasible by construction is annotated as a map artifact."""
    completion = {
        "classic_bottleneck_low": {"completion_rate": 0.0},
        "classic_crossing_low": {"success_rate": 0.3},
    }
    verdicts = {
        "classic_bottleneck_low": _envelope_verdict(
            "classic_bottleneck_low", category=INFEASIBLE_BY_CONSTRUCTION
        )
    }

    payload = annotate_zero_completion_cells(completion, verdicts)

    assert payload["schema_version"] == CAMPAIGN_FEASIBILITY_ANNOTATION_SCHEMA
    assert payload["total_zero_completion_cells"] == 1
    assert payload["annotated_cell_count"] == 1
    entry = payload["annotations"][0]
    assert entry["scenario_id"] == "classic_bottleneck_low"
    assert entry["zero_completion"] is True
    assert entry["envelope_category"] == INFEASIBLE_BY_CONSTRUCTION
    assert "map_artifact" in entry["annotation"]


def test_annotate_flags_zero_completion_envelope_sensitive_hard() -> None:
    """A zero-completion cell that is envelope-sensitive-hard is annotated accordingly."""
    payload = annotate_zero_completion_cells(
        {"classic_cross_trap_high": {"completion": 0.0}},
        {
            "classic_cross_trap_high": _envelope_verdict(
                "classic_cross_trap_high", category=ENVELOPE_SENSITIVE_HARD
            )
        },
    )
    entry = payload["annotations"][0]
    assert entry["envelope_category"] == ENVELOPE_SENSITIVE_HARD
    assert "envelope_sensitive_hard" in entry["annotation"]


def test_annotate_marks_planner_limited_when_oracle_feasible() -> None:
    """A zero-completion cell the oracle finds feasible is planner-limited."""
    payload = annotate_zero_completion_cells(
        {"classic_head_on_low": {"success": 0.0}},
        {"classic_head_on_low": _envelope_verdict("classic_head_on_low", category=FEASIBLE)},
    )
    entry = payload["annotations"][0]
    assert entry["envelope_category"] == FEASIBLE
    assert "planner_limited" in entry["annotation"]


def test_annotate_marks_planner_limited_when_oracle_missing() -> None:
    """A zero-completion cell without an oracle verdict defaults to planner-limited (missing)."""
    payload = annotate_zero_completion_cells(
        {"classic_unknown": {"completion_rate": 0.0}},
        {},
    )
    entry = payload["annotations"][0]
    assert entry["oracle_status"] == "missing"
    assert entry["annotation"] == PLANNER_LIMITED
    assert payload["annotated_cell_count"] == 0


def test_annotate_skips_nonzero_completion_cells() -> None:
    """Cells above the zero-completion threshold are not annotated."""
    payload = annotate_zero_completion_cells(
        {"classic_ok": {"success_rate": 0.5}},
        {"classic_ok": _envelope_verdict("classic_ok", category=FEASIBLE)},
    )
    assert payload["total_zero_completion_cells"] == 0
    assert payload["annotations"] == []


def test_annotate_respects_custom_threshold() -> None:
    """A custom threshold annotates low-but-nonzero completion cells."""
    payload = annotate_zero_completion_cells(
        {"classic_low": {"success_rate": 0.1}},
        {"classic_low": _envelope_verdict("classic_low", category=FEASIBLE)},
        zero_completion_threshold=0.1,
    )
    assert payload["total_zero_completion_cells"] == 1


# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------


def test_feasibility_verdict_to_dict_carries_schema_and_claim_boundary() -> None:
    """Serialized verdicts carry the v1 schema and diagnostic claim boundary."""
    geom = GeometricMargin(
        envelope_radius_m=1.0,
        envelope_diameter_m=2.0,
        route_geometrically_feasible=True,
        min_corridor_width_m=4.0,
        corridor_envelope_margin_m=2.0,
        min_static_clearance_m=1.0,
        shortest_path_length_m=20.0,
        classification=VALID,
        benchmark_eligibility="eligible",
    )
    completion = CompletionMargin(
        route_completion_feasible=True,
        min_completion_steps=200,
        horizon_steps=500,
        completion_horizon_margin_steps=300,
        kinematic_min_steps_lower_bound=100.0,
        termination_reason="success",
        status="passed",
    )
    verdict = FeasibilityVerdict(
        scenario_id="classic_head_on_low",
        family_id="head_on_corridor",
        envelope_radius_m=1.0,
        geometric=geom,
        completion=completion,
        feasible=True,
        status=FEASIBLE,
    )

    payload = feasibility_verdict_to_dict(verdict)

    assert payload["schema_version"] == FEASIBILITY_ORACLE_SCHEMA
    assert payload["claim_boundary"] == "diagnostic_only_not_benchmark_evidence"
    assert payload["geometric"]["min_corridor_width_m"] == pytest.approx(4.0)
    assert payload["completion"]["completion_horizon_margin_steps"] == 300


def test_envelope_verdict_to_dict_round_trips_category() -> None:
    """The envelope-sensitivity payload preserves the category and nominal radius."""
    verdict = _envelope_verdict("classic_head_on_low", category=FEASIBLE)
    payload = envelope_sensitivity_verdict_to_dict(verdict)
    assert payload["schema_version"] == ENVELOPE_SENSITIVITY_SCHEMA
    assert payload["category"] == FEASIBLE
    assert payload["nominal_envelope_radius_m"] == pytest.approx(1.0)


def test_default_envelope_radii_match_issue_nominal_and_reduced() -> None:
    """Default envelope radii are the 1.0 m nominal and 0.5 m reduced probe from the issue."""
    assert DEFAULT_ENVELOPE_RADII_M == (1.0, 0.5)


# ---------------------------------------------------------------------------
# Real end-to-end integration (one actor-free rollout on a committed map)
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_oracle_end_to_end_on_committed_head_on_corridor_scenario() -> None:
    """The oracle produces a coherent verdict on a real committed scenario.

    This is an integration test: it loads the classic_head_on_corridor archetype,
    runs the real certifier and the real actor-free rollout at the nominal envelope,
    and checks the margins are internally consistent. It is marked slow because it
    builds the simulator; it does NOT make any benchmark claim.
    """
    pytest.importorskip("robot_sf.benchmark.map_runner")
    if not _SCENARIO_PATH.exists():
        pytest.skip("head-on-corridor archetype scenario not available in this checkout")

    from robot_sf.training.scenario_loader import load_scenarios

    scenarios = load_scenarios(_SCENARIO_PATH)
    assert scenarios, "head-on-corridor archetype must define at least one scenario"
    scenario = scenarios[0]

    verdict = run_feasibility_oracle(
        scenario,
        config=FeasibilityOracleConfig(
            scenario_path=_SCENARIO_PATH,
            envelope_radii_m=(1.0,),
            rollout_seed=int(scenario.get("seeds", [111])[0]),
        ),
        envelope_radius_m=1.0,
    )

    # The corridor width must exceed the envelope diameter when the route is feasible,
    # and the corridor-envelope margin must equal corridor_width - diameter.
    geom = verdict.geometric
    if geom.route_geometrically_feasible and geom.min_corridor_width_m is not None:
        assert geom.min_corridor_width_m > geom.envelope_diameter_m
        assert geom.corridor_envelope_margin_m == pytest.approx(
            geom.min_corridor_width_m - geom.envelope_diameter_m
        )
    # Completion margin is internally consistent when the rollout completed.
    comp = verdict.completion
    if comp.min_completion_steps is not None and comp.horizon_steps is not None:
        assert (
            comp.completion_horizon_margin_steps == comp.horizon_steps - comp.min_completion_steps
        )
    assert verdict.status in {FEASIBLE, INFEASIBLE_BY_CONSTRUCTION, TIME_TRUNCATED, BLOCKED}
