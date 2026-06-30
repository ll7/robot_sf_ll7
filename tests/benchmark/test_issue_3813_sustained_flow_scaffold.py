"""Issue #3813 sustained-flow scaffold contract tests."""

from __future__ import annotations

import json
from pathlib import Path

import yaml

from robot_sf.benchmark.cli import cli_main
from robot_sf.benchmark.scenario_contract import (
    load_scenario_contracts,
    validate_scenario_contract_references,
)
from robot_sf.benchmark.sustained_flow_preflight import (
    RUNTIME_SUPPORTED_VALUE,
    preflight_sustained_flow_matrix,
)
from robot_sf.training.scenario_loader import load_scenarios

REPO_ROOT = Path(__file__).resolve().parents[2]
SCENARIO_SET = REPO_ROOT / "configs/scenarios/sets/issue_3813_sustained_flow_scaffold_v0.yaml"
CONTRACTS = REPO_ROOT / "configs/scenarios/contracts/issue_3813_sustained_flow_contracts.yaml"
LAUNCH_PACKET = REPO_ROOT / "configs/benchmarks/issue_3813_sustained_flow_launch_packet.yaml"


def _load_yaml(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def test_sustained_flow_scenario_set_is_runner_loadable(capsys) -> None:
    """The opt-in family is a valid scenario matrix for existing runner tooling."""

    rc = cli_main(["validate-config", "--matrix", str(SCENARIO_SET)])
    captured = capsys.readouterr()

    assert rc == 0, captured.out + captured.err
    report = json.loads(captured.out)
    assert report["num_scenarios"] == 3
    assert report["source"]["schema_version"] == "robot_sf.scenario_matrix.v1"
    assert report["summary"]["archetypes"] == {"sustained_flow_t_intersection": 3}

    scenarios = load_scenarios(SCENARIO_SET)
    assert [scenario["name"] for scenario in scenarios] == [
        "issue_3813_sustained_flow_t_intersection_light",
        "issue_3813_sustained_flow_t_intersection_medium",
        "issue_3813_sustained_flow_t_intersection_heavy",
    ]
    assert [scenario["simulation_config"]["ped_density"] for scenario in scenarios] == [
        0.02,
        0.05,
        0.08,
    ]
    # spawn_rate_per_min is the demand-parameterization knob that spans the
    # light/medium/heavy tiers, so pin it alongside ped_density.
    assert [
        scenario["metadata"]["continuous_spawn"]["spawn_rate_per_min"] for scenario in scenarios
    ] == [6.0, 12.0, 18.0]

    for scenario in scenarios:
        metadata = scenario["metadata"]
        assert scenario["simulation_config"]["max_episode_steps"] == 600
        assert metadata["pack_id"] == "issue_3813_sustained_flow_scaffold_v0"
        assert metadata["status"] == "pre_benchmark_scaffold"
        assert metadata["enabled_by_default"] is False
        assert metadata["benchmark_evidence"] is False
        assert metadata["continuous_spawn"]["required_before_benchmark_use"] is True
        assert metadata["continuous_spawn"]["current_runtime_support"] == "metadata_only"
        assert metadata["success_metric"]["id"] == "sustained_progress_rate_m_per_s"
        assert metadata["termination"]["mode"] == "time_bounded"
        assert metadata["termination"]["goal_reach_is_not_primary_success"] is True


def test_sustained_flow_preflight_enumerates_variants_and_fails_closed() -> None:
    """The generator preflight enumerates variants without claiming benchmark eligibility."""

    preflight = preflight_sustained_flow_matrix(SCENARIO_SET)
    payload = preflight.to_payload()

    assert payload["schema_version"] == "sustained_flow_preflight.v1"
    assert payload["status"] == "not_available"
    assert payload["benchmark_eligible"] is False
    assert payload["variant_count"] == 3
    assert [variant["density_tier"] for variant in payload["variants"]] == [
        "light",
        "medium",
        "heavy",
    ]
    assert [variant["spawn_rate_per_min"] for variant in payload["variants"]] == [
        6.0,
        12.0,
        18.0,
    ]
    assert [variant["ped_density"] for variant in payload["variants"]] == [
        0.02,
        0.05,
        0.08,
    ]
    assert all(
        f"expected {RUNTIME_SUPPORTED_VALUE!r}" in reason for reason in payload["blocking_reasons"]
    )


def test_sustained_flow_contract_defines_progress_metric_and_reference() -> None:
    """The scenario contract points at the scaffold and keeps evidence boundaries explicit."""

    contracts = load_scenario_contracts(CONTRACTS)
    assert len(contracts) == 1
    contract = contracts[0]

    assert validate_scenario_contract_references(contract, repo_root=REPO_ROOT) == []
    assert contract.scenario_ref.source == (
        "configs/scenarios/sets/issue_3813_sustained_flow_scaffold_v0.yaml"
    )
    assert contract.scenario_ref.scenario_name == "issue_3813_sustained_flow_t_intersection_medium"
    assert contract.certification.required_before_benchmark_claim is True
    assert contract.certification.expected_eligibility == "unknown"
    assert contract.benchmark_eligibility.intended_use == "exploratory"
    assert "Do not cite as benchmark evidence" in contract.benchmark_eligibility.claim_boundary

    observable_metrics = {observable.metric for observable in contract.observables}
    assert {
        "sustained_progress_rate_m_per_s",
        "interaction_exposure_share",
        "min_clearance_m",
    } <= observable_metrics

    extension = contract.extensions["sustained_flow.v1"]
    assert extension["benchmark_evidence"] is False
    assert extension["current_runtime_support"] == "metadata_only"
    assert extension["wait_policy_expected_result"] == "zero_or_near_zero_progress"


def test_launch_packet_keeps_no_submit_and_fail_closed_boundaries() -> None:
    """The campaign packet is a no-submit scaffold, not a benchmark claim."""

    packet = _load_yaml(LAUNCH_PACKET)
    campaign = packet["campaign"]

    assert campaign["parent_issue"] == 3813
    assert campaign["evidence_tier"] == "launch-packet-only"
    assert campaign["no_submit"] is True
    assert campaign["scenario_suite"]["matrix_path"] == (
        "configs/scenarios/sets/issue_3813_sustained_flow_scaffold_v0.yaml"
    )
    assert campaign["scenario_suite"]["contract_path"] == (
        "configs/scenarios/contracts/issue_3813_sustained_flow_contracts.yaml"
    )
    assert campaign["scenario_suite"]["preflight_command"] == (
        "uv run python scripts/validation/preflight_sustained_flow_scenarios_issue_3813.py --json"
    )
    assert campaign["metrics"]["success_boundary"].startswith("Sustained progress")
    assert (
        "missing continuous-spawn runtime support"
        in campaign["row_status_policy"]["fallback_policy"]
    )
    assert {
        "full benchmark campaign run",
        "Slurm or GPU submission",
        "paper or dissertation claim edits",
    } <= set(campaign["out_of_scope"])
