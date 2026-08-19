"""Capability-only tests for the issue #4360 search preparation slice."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import yaml

from robot_sf.adversarial.materialize import ImmutableScenarioOverlay
from robot_sf.adversarial.search_harness import (
    BASELINE_NAMES,
    CandidateSpecOverlayAdapter,
    FiniteSearchSpaceManifest,
    MappingOverlayAdapter,
    prepare_baseline,
    prepare_equal_budget_baselines,
)

MANIFEST_PATH = (
    Path(__file__).resolve().parents[2]
    / "configs"
    / "adversarial"
    / "issue_4360_search_harness_fixture.yaml"
)
SOURCE_PATH = (
    Path(__file__).resolve().parent
    / "fixtures"
    / "issue_4360_search_harness"
    / "source_scenario.yaml"
)


def _manifest() -> FiniteSearchSpaceManifest:
    """Load the tracked preparation-only manifest fixture."""
    return FiniteSearchSpaceManifest.from_file(MANIFEST_PATH)


def _source() -> dict[str, Any]:
    """Load the small source scenario mapping without invoking a loader or simulator."""
    payload = yaml.safe_load(SOURCE_PATH.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return payload


def _mapping_adapter(manifest: FiniteSearchSpaceManifest) -> MappingOverlayAdapter:
    """Map every manifest variable into an isolated candidate-values subtree."""
    return MappingOverlayAdapter(
        {name: f"candidate.values.{name}" for name in manifest.variable_names}
    )


def test_manifest_round_trip_freezes_typed_contract_and_digest() -> None:
    """The manifest keeps units, bounds, constraints, objectives, seeds, and budget typed."""
    manifest = _manifest()
    restored = FiniteSearchSpaceManifest.from_mapping(json.loads(manifest.to_json()))

    assert restored == manifest
    assert restored.variable_names[-1] == "scenario_seed"
    assert restored._variable("scenario_seed").bounds.kind == "integer"
    assert restored.constraints[0].referenced_variables == ("goal_x", "start_x")
    assert restored.seed_policy.held_out_replay_seeds == (843601, 843602)
    assert restored.rollout_budget.candidate_budget == 8
    assert restored.digest == manifest.digest


def test_manifest_rejects_unknown_constraint_variables() -> None:
    """A cross-variable predicate cannot silently introduce an undeclared dimension."""
    payload = _manifest().to_dict()
    payload["constraints"][0]["expression"] = "unknown_variable > 0.0"

    with pytest.raises(ValueError, match="unknown variables: unknown_variable"):
        FiniteSearchSpaceManifest.from_mapping(payload)


def test_random_and_quasi_random_baselines_are_deterministic_and_equal_budget() -> None:
    """Both preparation arms emit the same fixed count without sharing proposal points."""
    manifest = _manifest()
    first = prepare_equal_budget_baselines(manifest, _source(), _mapping_adapter(manifest))
    second = prepare_equal_budget_baselines(manifest, _source(), _mapping_adapter(manifest))

    assert tuple(first) == BASELINE_NAMES
    assert all(
        len(result.candidates) == manifest.rollout_budget.candidate_budget
        for result in first.values()
    )
    assert {result.provenance["candidate_budget"] for result in first.values()} == {
        manifest.rollout_budget.candidate_budget
    }
    assert [row.candidate.to_dict() for row in first["random"].candidates] == [
        row.candidate.to_dict() for row in second["random"].candidates
    ]
    assert [row.candidate.to_dict() for row in first["quasi_random"].candidates] == [
        row.candidate.to_dict() for row in second["quasi_random"].candidates
    ]
    assert [row.candidate.values for row in first["random"].candidates] != [
        row.candidate.values for row in first["quasi_random"].candidates
    ]
    assert all(result.provenance["simulation_executed"] is False for result in first.values())


def test_cross_variable_rejection_happens_before_adapter_or_simulation() -> None:
    """Infeasible proposals never reach the adapter seam."""
    manifest = FiniteSearchSpaceManifest.from_mapping(
        {
            "schema_version": "adversarial_search_harness.v1",
            "name": "impossible_fixture",
            "variables": {
                "x": {"unit": "m", "bounds": {"min": 0.0, "max": 1.0}},
                "y": {"unit": "m", "bounds": {"min": 0.0, "max": 1.0}},
            },
            "constraints": [
                {"name": "impossible", "expression": "y - x >= 2.0"},
            ],
            "objective_vector": {
                "components": [{"name": "diagnostic", "direction": "maximize", "unit": "score"}]
            },
            "seed_policy": {"search_seed": 1, "held_out_replay_seeds": [2]},
            "rollout_budget": {"candidate_budget": 4, "max_steps": 8},
        }
    )

    class CountingAdapter:
        """Adapter fixture that would expose an accidental pre-rejection call."""

        adapter_id = "counting_fixture.v1"

        def __init__(self) -> None:
            self.validated = 0
            self.materialized = 0

        def validate(self, source_scenario: dict[str, Any], candidate: Any) -> tuple[str, ...]:
            del source_scenario, candidate
            self.validated += 1
            return ()

        def materialize(self, source_scenario: dict[str, Any], candidate: Any) -> Any:
            del source_scenario, candidate
            self.materialized += 1
            raise AssertionError("infeasible candidates must not materialize")

    adapter = CountingAdapter()
    result = prepare_baseline(manifest, {}, adapter, baseline="random")

    assert adapter.validated == 0
    assert adapter.materialized == 0
    assert result.prepared_count == 0
    assert result.rejected_count == manifest.rollout_budget.candidate_budget
    assert all(row.rejection is not None for row in result.candidates)
    assert all(row.rejection.stage == "manifest" for row in result.candidates if row.rejection)
    assert all(
        row.rejection.to_dict()["simulation_executed"] is False
        for row in result.candidates
        if row.rejection
    )


def test_mapping_overlay_is_immutable_and_serializes_stably() -> None:
    """Materialized preparation data cannot mutate the source scenario in place."""
    manifest = _manifest()
    result = prepare_baseline(
        manifest, _source(), _mapping_adapter(manifest), baseline="quasi_random"
    )
    prepared = next(row for row in result.candidates if row.overlay is not None)
    overlay = prepared.overlay
    assert overlay is not None

    with pytest.raises(TypeError):
        overlay.source["mutated"] = True  # type: ignore[index]
    with pytest.raises(TypeError):
        overlay.materialized["mutated"] = True  # type: ignore[index]

    assert "mutated" not in _source()
    assert overlay.to_json() == overlay.to_json()
    assert overlay.source_digest
    assert overlay.patch_digest
    assert overlay.materialized_digest


def test_candidate_spec_adapter_reuses_existing_pure_materializer() -> None:
    """The existing CandidateSpec bundle seam can feed an immutable preparation overlay."""
    manifest = _manifest()
    adapter = CandidateSpecOverlayAdapter(pedestrian_id="p0")
    coordinates = (0.5, 0.5, 0.5, 0.5, 0.25, 0.5, 0.25, 0.5)
    candidate = manifest.build_candidate(
        baseline="random",
        index=0,
        unit_coordinates=coordinates,
    )
    overlay = adapter.materialize(_source(), candidate)

    assert isinstance(overlay, ImmutableScenarioOverlay)
    scenario = overlay.materialized["scenarios"][0]
    assert scenario["simulation_config"]["peds_speed_mult"] == pytest.approx(1.1)
    assert scenario["single_pedestrians"][0]["id"] == "p0"
    assert overlay.provenance["existing_materializer"].endswith("build_candidate_payload")
    assert overlay.materialized["route_overrides"]["robot_routes"]
