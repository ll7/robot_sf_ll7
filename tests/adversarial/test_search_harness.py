"""Capability-only tests for the issue #4360 search preparation slice."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
import yaml

from robot_sf.adversarial.config import (
    CandidateSpec,
    MultiPedAdversarialConfig,
    MultiPedCandidateSpec,
    Pose2D,
)
from robot_sf.adversarial.materialize import (
    ImmutableScenarioOverlay,
    materialize_manifest_route_overrides,
    materialize_manifest_scenario_payload,
    materialize_manifest_single_pedestrian_override,
    materialize_multi_ped_scenario_payload,
    materialize_multi_ped_single_pedestrian_overrides,
)
from robot_sf.adversarial.scenario_manifest import build_manifest
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


def _candidate_spec(seed: int = 23) -> CandidateSpec:
    """Build a valid single-pedestrian manifest candidate for materializer tests."""
    return CandidateSpec(
        start=Pose2D(1.0, 2.0),
        goal=Pose2D(5.0, 2.0),
        spawn_time_s=0.5,
        pedestrian_speed_mps=1.0,
        pedestrian_delay_s=0.25,
        scenario_seed=seed,
    )


def _scenario_template() -> dict[str, Any]:
    """Return a small scenario matrix that exercises nested merge behavior."""
    return {
        "scenarios": [
            {
                "scenario_id": "materializer_template",
                "map_id": "classic_cross_trap",
                "simulation_config": {"max_episode_steps": 30, "ped_density": 0.0},
                "metadata": {"archetype": "test"},
                "single_pedestrians": [
                    {"id": "probe", "role": "template", "note": "preserve me"},
                    {"not_an_override": True},
                    "ignore non-mappings",
                ],
            }
        ]
    }


def test_immutable_overlay_freezes_nested_values_and_round_trips_provenance() -> None:
    """Immutable overlays merge nested mappings without mutating source or patch inputs."""
    source = {"nested": {"keep": 1, "items": ["source"]}, "replace": {"old": True}}
    patch = {"nested": {"added": 2, "items": ["candidate"]}, "replace": "candidate"}
    overlay = ImmutableScenarioOverlay(
        source=source,
        patch=patch,
        candidate_id="candidate:0000",
        adapter_id="test_overlay.v1",
        provenance={"stage": "test", "nested": {"value": 1}},
    )

    materialized = overlay.materialized
    assert materialized["nested"]["keep"] == 1
    assert materialized["nested"]["added"] == 2
    assert materialized["nested"]["items"] == ("candidate",)
    assert materialized["replace"] == "candidate"
    assert overlay.to_mapping() == materialized
    with pytest.raises(TypeError):
        overlay.source["changed"] = True  # type: ignore[index]
    with pytest.raises(TypeError):
        overlay.materialized["nested"]["changed"] = True  # type: ignore[index]

    payload = json.loads(overlay.to_json(indent=2))
    assert payload["candidate_id"] == "candidate:0000"
    assert payload["provenance"]["nested"] == {"value": 1}
    assert overlay.source_digest != overlay.patch_digest
    assert overlay.materialized_digest


@pytest.mark.parametrize(
    ("source", "patch", "candidate_id", "adapter_id", "error"),
    [
        ([], {}, "candidate", "adapter", "source must be a mapping"),
        ({}, [], "candidate", "adapter", "patch must be a mapping"),
        ({}, {}, "", "adapter", "candidate_id must be non-empty"),
        ({}, {}, "candidate", "", "adapter_id must be non-empty"),
    ],
)
def test_immutable_overlay_rejects_invalid_identity_and_shapes(
    source: Any,
    patch: Any,
    candidate_id: str,
    adapter_id: str,
    error: str,
) -> None:
    """Overlay construction should fail closed before any candidate data is exposed."""
    with pytest.raises((TypeError, ValueError), match=error):
        ImmutableScenarioOverlay(
            source=source,
            patch=patch,
            candidate_id=candidate_id,
            adapter_id=adapter_id,
        )

    with pytest.raises(ValueError, match="must be finite"):
        ImmutableScenarioOverlay(
            source={"value": float("nan")},
            patch={},
            candidate_id="candidate",
            adapter_id="adapter",
        )
    with pytest.raises(TypeError, match="unsupported value"):
        ImmutableScenarioOverlay(
            source={"value": object()},
            patch={},
            candidate_id="candidate",
            adapter_id="adapter",
        )


def test_multi_ped_materializers_merge_existing_entries_and_preserve_runtime_boundary() -> None:
    """Multi-ped overlays preserve authored fields and mark output as uncertified smoke data."""
    config = MultiPedAdversarialConfig(
        family="group_squeeze",
        scenario_seed=41,
        pedestrians=[
            MultiPedCandidateSpec(
                id="probe",
                start=Pose2D(1.0, 2.0),
                goal=Pose2D(5.0, 2.0),
                spawn_time_s=0.5,
                speed_mps=1.1,
                delay_s=0.25,
                metadata={"role": "left"},
            ),
            MultiPedCandidateSpec(
                id="new_probe",
                start=Pose2D(1.0, 3.0),
                goal=Pose2D(5.0, 3.0),
                speed_mps=1.2,
            ),
        ],
    )

    overrides = materialize_multi_ped_single_pedestrian_overrides(config)
    assert overrides[0]["start_delay_s"] == pytest.approx(0.75)
    assert overrides[0]["metadata"]["pedestrian_metadata"] == {"role": "left"}

    payload = materialize_multi_ped_scenario_payload(config, _scenario_template())
    scenario = payload["scenarios"][0]
    assert scenario["name"] == "materializer_template_multi_ped_adversarial_0041"
    assert scenario["simulation_config"]["route_spawn_seed"] == 41
    assert scenario["metadata"]["archetype"] == "test"
    assert scenario["metadata"]["adversarial_multi_ped"]["family"] == "group_squeeze"
    assert scenario["metadata"]["adversarial_multi_ped_runtime"]["benchmark_frozen"] is False
    assert scenario["single_pedestrians"][0]["role"] == "template"
    assert scenario["single_pedestrians"][0]["start"] == [1.0, 2.0]
    assert scenario["single_pedestrians"][1]["id"] == "new_probe"
    assert len(scenario["single_pedestrians"]) == 2


@pytest.mark.parametrize("template", [{}, {"scenarios": []}, {"scenarios": ["invalid"]}])
def test_multi_ped_materializer_requires_a_scenario_mapping(template: dict[str, Any]) -> None:
    """The pure bridge should reject malformed scenario matrices before copying data."""
    config = MultiPedAdversarialConfig(
        family="late_stop",
        scenario_seed=7,
        pedestrians=[
            MultiPedCandidateSpec(
                id="stopper",
                start=Pose2D(0.0, 0.0),
                goal=Pose2D(1.0, 0.0),
                speed_mps=0.8,
            )
        ],
    )
    with pytest.raises(ValueError, match="non-empty scenarios list"):
        materialize_multi_ped_scenario_payload(config, template)


def test_manifest_materializers_cover_inline_and_route_file_payloads() -> None:
    """Valid manifests bridge to both inline pedestrian and explicit route-file payloads."""
    legacy_manifest = build_manifest(_candidate_spec(), generator=None)
    override = materialize_manifest_single_pedestrian_override(legacy_manifest)
    assert override["id"] == "manifest_candidate_0000"
    assert override["start_delay_s"] == pytest.approx(0.75)
    assert override["metadata"]["validation_status"] == "valid"

    inline = materialize_manifest_scenario_payload(legacy_manifest, _scenario_template())
    inline_scenario = inline["scenarios"][0]
    assert inline_scenario["name"] == "materializer_template_manifest_0000"
    assert inline_scenario["single_pedestrians"][0]["id"] == "probe"
    assert any(
        entry["id"] == "manifest_candidate_0000" for entry in inline_scenario["single_pedestrians"]
    )
    assert inline_scenario["simulation_config"]["peds_speed_mult"] == pytest.approx(1.0)

    route_payload = materialize_manifest_route_overrides(legacy_manifest, route_id=77)
    assert route_payload["robot_routes"][0]["spawn_id"] == 77
    assert route_payload["robot_routes"][0]["waypoints"] == [[1.0, 2.0], [5.0, 2.0]]

    route_file = materialize_manifest_scenario_payload(
        legacy_manifest,
        _scenario_template(),
        route_file_name="routes/candidate.yaml",
    )
    route_scenario = route_file["scenarios"][0]
    assert route_scenario["route_overrides_file"] == "routes/candidate.yaml"
    assert route_scenario["single_pedestrians"][0]["id"] == "probe"


def test_manifest_materializers_fail_closed_for_missing_validation_controls_and_poses() -> None:
    """Manifest materialization must reject incomplete or malformed preparation records."""
    manifest = build_manifest(_candidate_spec())
    with pytest.raises(ValueError, match="validation record is required"):
        materialize_manifest_route_overrides(replace(manifest, validation=None))

    assert manifest.candidate_controls is not None
    missing_controls = dict(manifest.candidate_controls)
    missing_controls.pop("goal")
    with pytest.raises(ValueError, match="candidate_controls.goal is required"):
        materialize_manifest_route_overrides(replace(manifest, candidate_controls=missing_controls))

    bad_pose = dict(manifest.candidate_controls)
    bad_pose["start"] = []
    with pytest.raises(ValueError, match="candidate_controls.start must be a mapping"):
        materialize_manifest_route_overrides(replace(manifest, candidate_controls=bad_pose))

    missing_coordinate = dict(manifest.candidate_controls)
    missing_coordinate["start"] = {"x": 1.0}
    with pytest.raises(ValueError, match="requires both 'x' and 'y' keys"):
        materialize_manifest_single_pedestrian_override(
            replace(manifest, candidate_controls=missing_coordinate)
        )

    with pytest.raises(ValueError, match="non-empty scenarios list"):
        materialize_manifest_scenario_payload(manifest, {})

    invalid = build_manifest(
        CandidateSpec(
            start=Pose2D(1.0, 1.0),
            goal=Pose2D(1.0, 1.0),
            spawn_time_s=0.0,
            pedestrian_speed_mps=0.0,
            pedestrian_delay_s=0.0,
            scenario_seed=1,
        )
    )
    with pytest.raises(ValueError, match="only valid manifests"):
        materialize_manifest_scenario_payload(invalid, _scenario_template())
