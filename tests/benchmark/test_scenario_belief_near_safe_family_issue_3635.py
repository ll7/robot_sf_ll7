"""Contract tests for the #3635 ScenarioBelief near-safe crossing slice."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from robot_sf.benchmark.runner import load_scenario_matrix

REPO_ROOT = Path(__file__).resolve().parents[2]
LAUNCH_PACKET = REPO_ROOT / "configs/benchmarks/scenario_belief_drop_vs_retain_issue_3556.yaml"
SCENARIO_SET = (
    REPO_ROOT / "configs/scenarios/sets/issue_3635_near_safe_occlusion_bearing_crossing.yaml"
)


def _load_yaml(path: Path) -> dict[str, Any]:
    """Load a committed YAML mapping for contract assertions."""
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert isinstance(data, dict)
    return data


def test_near_safe_occlusion_bearing_family_loads_as_single_contract_scenario() -> None:
    """The #3635 set manifest resolves to the intended near-safe crossing family."""
    scenarios = load_scenario_matrix(SCENARIO_SET)

    assert [scenario["name"] for scenario in scenarios] == [
        "issue_3635_near_safe_occlusion_bearing_crossing"
    ]
    scenario = scenarios[0]
    metadata = scenario["metadata"]

    assert scenario["scenario_family"] == "near_safe_occlusion_bearing_crossing"
    assert metadata["occlusion_bearing"] is True
    assert metadata["benchmark_evidence"] is False
    assert metadata["near_safe_precondition"] == (
        "oracle_clears_most_episodes_before_contrast_claim"
    )
    assert metadata["scenario_belief_contract"]["required_modes"] == [
        "oracle",
        "uncertain_retained",
        "uncertain_dropped",
    ]
    assert scenario["seeds"] == [363501, 363502, 363503]


def test_launch_packet_distinguishes_oracle_retained_and_dropped_arms() -> None:
    """The manifest distinguishes all three arms without running a benchmark campaign."""
    packet = _load_yaml(LAUNCH_PACKET)
    arms = packet["runner_arms"]
    by_id = {arm["arm_id"]: arm for arm in arms}

    assert list(by_id) == ["oracle", "uncertain_retained", "uncertain_dropped"]
    assert set(by_id) == set(packet["belief_modes"])

    assert {arm["scenario_family"] for arm in arms} == {packet["scenario_family"]}
    assert {arm["seed_set"] for arm in arms} == {packet["seed_set"]}
    assert {arm["algo"] for arm in arms} == {"stream_gap"}

    for arm_id, arm in by_id.items():
        mode_config = packet["belief_modes"][arm_id]
        assert arm["algo"] == mode_config["algo"]
        assert arm["algo_config"] == {
            "belief_mode": mode_config["belief_mode"],
            "belief_fov_degrees": mode_config["belief_fov_degrees"],
        }

    assert by_id["oracle"]["expected_gate_enabled"] is False
    assert by_id["oracle"]["expected_planner_visibility"] == "all_agents_retained"
    assert by_id["uncertain_retained"]["expected_gate_enabled"] is False
    assert by_id["uncertain_retained"]["expected_planner_visibility"] == (
        "low_confidence_agents_retained"
    )
    assert by_id["uncertain_dropped"]["expected_gate_enabled"] is True
    assert by_id["uncertain_dropped"]["expected_planner_visibility"] == (
        "low_confidence_agents_dropped"
    )


def test_launch_packet_remains_proposal_not_benchmark_evidence() -> None:
    """The #3635 slice must not self-promote to result or claim evidence."""
    packet = _load_yaml(LAUNCH_PACKET)

    assert packet["evidence_status"] == "proposal"
    assert packet["no_benchmark_result_claim"] is True
    assert packet["claim_gate"]["no_claim_until_run"] is True
    assert {"issue": 3635} in packet["follows"]
    assert packet["seed_sets"][packet["seed_set"]]["seeds"] == [363501, 363502, 363503]
    assert "fallback/degraded rows never count as success" in " ".join(packet["validation"])
