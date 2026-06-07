"""Tests for controlled counterfactual scenario-pair manifest creation."""

from __future__ import annotations

from typing import TYPE_CHECKING

import yaml

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
    assert (
        payload["claim_boundary"] == "candidate mechanism-test inputs only; not benchmark evidence"
    )
    assert payload["validity_status"] == "valid"
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
