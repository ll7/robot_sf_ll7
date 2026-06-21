"""Tests for controlled counterfactual scenario-pair manifest creation."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
import yaml

from robot_sf.benchmark.manifest_lineage import validate_lineage_contract
from scripts.tools import create_counterfactual_scenario_pair

if TYPE_CHECKING:
    from pathlib import Path


def test_create_supported_robot_route_offset_pair(tmp_path: Path) -> None:
    """The CLI should write a preflight-validated route-offset counterfactual pair."""
    output_path = tmp_path / "pair.yaml"

    result = create_counterfactual_scenario_pair.main(
        [
            "--source",
            "planner_sanity_simple",
            "--feature",
            "robot_route_offset",
            "--magnitude",
            "0.25",
            "--seed",
            "111",
            "--scenario-config",
            "configs/scenarios/single/planner_sanity_simple.yaml",
            "--output",
            str(output_path),
        ]
    )

    assert result == 0
    output_text = output_path.read_text(encoding="utf-8")
    payload = yaml.safe_load(output_text)
    assert payload["changed_feature"] == "robot_route_offset"
    assert payload["source"] == {
        "scenario_config": "configs/scenarios/single/planner_sanity_simple.yaml",
        "source_scenario_id": "planner_sanity_simple",
    }
    assert payload["generator_id"] == "create_counterfactual_scenario_pair"
    assert payload["validator_version"] == "counterfactual_scenario_pair_validator.v1"
    assert payload["evidence_tier"] == "diagnostic-only"
    assert payload["denominator_policy"] == "counterfactual_pairs_not_benchmark_denominator"
    assert payload["execution_gate"] == "preflight_success_required_before_execution"
    assert validate_lineage_contract(payload) == []
    assert payload["changed_factor"] == "robot_route_offset"
    assert (
        payload["claim_boundary"] == "candidate mechanism-test inputs only; not benchmark evidence"
    )
    assert payload["validity_status"] == "valid"
    assert payload["mechanism_taxonomy"] == {
        "label": "clearance_pressure",
        "label_source": "counterfactual_mechanism_taxonomy.v1",
        "mechanism_hypothesis": (
            "A bounded robot-route offset changes clearance pressure while holding the scenario, "
            "planner, and seed fixed."
        ),
        "expected_metric_direction": {
            "clearance_min_distance_m": "direction_depends_on_offset_sign",
            "collision_or_near_miss_risk": "may_increase_when_offset_reduces_clearance",
            "success": "no_directional_claim_from_pair_manifest",
        },
        "validity_constraints": [
            "baseline and intervention must both pass perturbation preflight",
            "seed and source scenario must remain unchanged",
            "single pair is a mechanism hypothesis input, not causal evidence",
        ],
    }
    assert payload["pair_report"] == {
        "base_scenario_id": "planner_sanity_simple",
        "counterfactual_scenario_id": "planner_sanity_simple",
        "changed_factor": "robot_route_offset",
        "artifact_manifest_ref": "perturbation_manifest",
        "expected_vs_observed_metric_change": {
            "status": "not_available",
            "reason": "no smoke-run metrics were supplied to this pair manifest",
        },
    }
    assert payload["baseline"] == {
        "scenario_id": "planner_sanity_simple",
        "variant_id": "planner_sanity_simple_baseline_seed_111",
        "family": "noop",
        "seed": 111,
    }
    assert payload["intervention"]["variant_id"] == (
        "planner_sanity_simple_robot_route_offset_0p250_seed_111"
    )
    assert payload["intervention"]["parameters"] == {
        "dx_m": 0.25,
        "dy_m": 0.0,
        "max_magnitude_m": 0.25,
    }
    assert payload["unchanged_controls"]["scenario_config"] == (
        "configs/scenarios/single/planner_sanity_simple.yaml"
    )
    assert payload["unchanged_controls"]["seed"] == 111
    assert payload["perturbation_manifest"]["variants"] == [
        {
            "variant_id": "planner_sanity_simple_baseline_seed_111",
            "scenario_id": "planner_sanity_simple",
            "family": "noop",
            "seeds": [111],
        },
        {
            "variant_id": "planner_sanity_simple_robot_route_offset_0p250_seed_111",
            "scenario_id": "planner_sanity_simple",
            "family": "robot_route_offset",
            "seeds": [111],
            "parameters": {
                "dx_m": 0.25,
                "dy_m": 0.0,
                "max_magnitude_m": 0.25,
            },
        },
    ]
    assert [row["benchmark_evidence_status"] for row in payload["preflight"]["results"]] == [
        "eligible_success_evidence_candidate",
        "eligible_success_evidence_candidate",
    ]
    assert all(
        "route_certificates" not in row["certificate"] for row in payload["preflight"]["results"]
    )
    assert "&id" not in output_text
    assert "*id" not in output_text


@pytest.mark.parametrize(
    (
        "source",
        "feature",
        "scenario_config",
        "expected_parameters",
        "expected_taxonomy_label",
    ),
    [
        (
            "classic_group_crossing_high",
            "pedestrian_route_offset",
            "configs/scenarios/classic_interactions_francis2023.yaml",
            {"dx_m": 0.25, "dy_m": 0.0, "max_magnitude_m": 0.25},
            "clearance_pressure",
        ),
        (
            "francis2023_join_group",
            "single_pedestrian_start_delay_offset",
            "configs/scenarios/classic_interactions_francis2023.yaml",
            {"dt_s": 0.25, "max_abs_dt_s": 0.25},
            "bottleneck_negotiation",
        ),
        (
            "francis2023_join_group",
            "single_pedestrian_speed_offset",
            "configs/scenarios/classic_interactions_francis2023.yaml",
            {"speed_delta_m_s": 0.25, "max_abs_speed_delta_m_s": 0.25},
            "bottleneck_negotiation",
        ),
        (
            "francis2023_intersection_wait",
            "single_pedestrian_wait_duration_offset",
            "configs/scenarios/classic_interactions_francis2023.yaml",
            {"wait_delta_s": 0.25, "max_abs_wait_delta_s": 0.25},
            "signal_compliance",
        ),
    ],
)
def test_create_supported_non_robot_counterfactual_pairs(
    tmp_path: Path,
    source: str,
    feature: str,
    scenario_config: str,
    expected_parameters: dict[str, float],
    expected_taxonomy_label: str,
) -> None:
    """Registered non-robot perturbation families should produce preflight-valid pairs."""
    output_path = tmp_path / f"{feature}.yaml"

    result = create_counterfactual_scenario_pair.main(
        [
            "--source",
            source,
            "--feature",
            feature,
            "--magnitude",
            "0.25",
            "--seed",
            "111",
            "--scenario-config",
            scenario_config,
            "--output",
            str(output_path),
        ]
    )

    assert result == 0
    payload = yaml.safe_load(output_path.read_text(encoding="utf-8"))
    assert payload["changed_feature"] == feature
    assert payload["changed_factor"] == feature
    assert payload["source"] == {
        "scenario_config": scenario_config,
        "source_scenario_id": source,
    }
    assert validate_lineage_contract(payload) == []
    assert payload["intervention"]["family"] == feature
    assert payload["intervention"]["parameters"] == expected_parameters
    assert payload["mechanism_taxonomy"]["label"] == expected_taxonomy_label
    assert payload["pair_report"]["changed_factor"] == feature
    assert [row["benchmark_evidence_status"] for row in payload["preflight"]["results"]] == [
        "eligible_success_evidence_candidate",
        "eligible_success_evidence_candidate",
    ]


def test_unsupported_feature_fails_closed_without_output(tmp_path: Path, capsys) -> None:
    """Unsupported features should not emit candidate inputs."""
    output_path = tmp_path / "pair.yaml"

    result = create_counterfactual_scenario_pair.main(
        [
            "--source",
            "planner_sanity_simple",
            "--feature",
            "pedestrian_density_offset",
            "--magnitude",
            "0.25",
            "--seed",
            "111",
            "--scenario-config",
            "configs/scenarios/single/planner_sanity_simple.yaml",
            "--output",
            str(output_path),
        ]
    )

    captured = capsys.readouterr()
    assert result == 2
    assert "unsupported counterfactual feature" in captured.err
    assert not output_path.exists()
