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
from robot_sf.scenario_certification.sustained_flow import (
    generate_runtime_supported_sustained_flow_scenarios,
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
        spawn_definition = metadata["continuous_spawn"]["definition"]
        assert scenario["simulation_config"]["max_episode_steps"] == 600
        assert metadata["pack_id"] == "issue_3813_sustained_flow_scaffold_v0"
        assert metadata["status"] == "pre_benchmark_scaffold"
        assert metadata["enabled_by_default"] is False
        assert metadata["benchmark_evidence"] is False
        assert metadata["continuous_spawn"]["required_before_benchmark_use"] is True
        assert metadata["continuous_spawn"]["current_runtime_support"] == "metadata_only"
        assert spawn_definition["demand_model"] == "non_clearing_poisson_flow"
        assert spawn_definition["spawn_budget"] == "time_bounded_episode"
        assert spawn_definition["minimum_active_pedestrians"] == 1
        assert spawn_definition["clearing_policy"] == "disallow_empty_scene_wait_success"
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
    assert payload["runtime_readiness"] == {
        "status": "not_supported",
        "supported": False,
        "expected_runtime_support": RUNTIME_SUPPORTED_VALUE,
        "observed_runtime_support": ["metadata_only"],
    }
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


def test_sustained_flow_preflight_accepts_runtime_supported_matrix(tmp_path: Path) -> None:
    """Runtime-supported rows become eligible only when every variant opts in."""

    matrix = _load_yaml(SCENARIO_SET)
    matrix["scenarios"] = generate_runtime_supported_sustained_flow_scenarios()

    supported_matrix = tmp_path / "runtime_supported_sustained_flow.yaml"
    supported_matrix.write_text(yaml.safe_dump(matrix, sort_keys=False), encoding="utf-8")

    payload = preflight_sustained_flow_matrix(supported_matrix).to_payload()

    assert payload["status"] == "available"
    assert payload["benchmark_eligible"] is True
    assert payload["runtime_readiness"] == {
        "status": "supported",
        "supported": True,
        "expected_runtime_support": RUNTIME_SUPPORTED_VALUE,
        "observed_runtime_support": [RUNTIME_SUPPORTED_VALUE],
    }
    assert payload["blocking_reasons"] == []


def test_sustained_flow_preflight_rejects_generator_drift(tmp_path: Path) -> None:
    """Preflight fails closed if YAML rows stop matching the generator."""

    matrix = _load_yaml(SCENARIO_SET)
    for scenario in matrix["scenarios"]:
        scenario["metadata"]["continuous_spawn"]["current_runtime_support"] = (
            RUNTIME_SUPPORTED_VALUE
        )
    matrix["scenarios"][1]["metadata"]["continuous_spawn"]["spawn_rate_per_min"] = 13.0

    drifted_matrix = tmp_path / "drifted_sustained_flow.yaml"
    drifted_matrix.write_text(yaml.safe_dump(matrix, sort_keys=False), encoding="utf-8")

    payload = preflight_sustained_flow_matrix(drifted_matrix).to_payload()

    assert payload["status"] == "not_available"
    assert payload["benchmark_eligible"] is False
    assert payload["variant_count"] == 3
    assert (
        "sustained-flow matrix must match canonical generated variant definitions"
        in payload["blocking_reasons"]
    )


def test_sustained_flow_preflight_rejects_wrong_spawn_process(tmp_path: Path) -> None:
    """Continuous-spawn variants must name the supported respawn process."""

    matrix = _load_yaml(SCENARIO_SET)
    matrix["scenarios"][0]["metadata"]["continuous_spawn"]["intended_process"] = "finite_batch"

    wrong_process_matrix = tmp_path / "wrong_spawn_process_sustained_flow.yaml"
    wrong_process_matrix.write_text(yaml.safe_dump(matrix, sort_keys=False), encoding="utf-8")

    payload = preflight_sustained_flow_matrix(wrong_process_matrix).to_payload()

    assert payload["status"] == "not_available"
    assert payload["benchmark_eligible"] is False
    assert payload["variant_count"] == 0
    assert any(
        "continuous_spawn.intended_process must be 'poisson_respawn'" in reason
        for reason in payload["blocking_reasons"]
    )


def test_sustained_flow_preflight_rejects_wrong_spawn_definition(tmp_path: Path) -> None:
    """Continuous-spawn variants must keep the non-clearing demand definition."""
    matrix = _load_yaml(SCENARIO_SET)
    matrix["scenarios"][0]["metadata"]["continuous_spawn"]["definition"]["clearing_policy"] = (
        "allow_empty_scene_wait_success"
    )

    wrong_definition_matrix = tmp_path / "wrong_spawn_definition_sustained_flow.yaml"
    wrong_definition_matrix.write_text(yaml.safe_dump(matrix, sort_keys=False), encoding="utf-8")
    payload = preflight_sustained_flow_matrix(wrong_definition_matrix).to_payload()

    assert payload["status"] == "not_available"
    assert payload["benchmark_eligible"] is False
    assert payload["variant_count"] == 0
    assert any(
        "continuous_spawn.definition.clearing_policy must be "
        "'disallow_empty_scene_wait_success'" in reason
        for reason in payload["blocking_reasons"]
    )


def test_sustained_flow_preflight_rejects_waitable_spawn_budget(tmp_path: Path) -> None:
    """Continuous-spawn variants must not allow finite-batch waiting."""
    matrix = _load_yaml(SCENARIO_SET)
    matrix["scenarios"][0]["metadata"]["continuous_spawn"]["definition"]["spawn_budget"] = (
        "finite_batch"
    )

    wrong_budget_matrix = tmp_path / "wrong_spawn_budget_sustained_flow.yaml"
    wrong_budget_matrix.write_text(yaml.safe_dump(matrix, sort_keys=False), encoding="utf-8")
    payload = preflight_sustained_flow_matrix(wrong_budget_matrix).to_payload()

    assert payload["status"] == "not_available"
    assert payload["benchmark_eligible"] is False
    assert payload["variant_count"] == 0
    assert any(
        "continuous_spawn.definition.spawn_budget must be 'time_bounded_episode'" in reason
        for reason in payload["blocking_reasons"]
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
