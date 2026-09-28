"""Fake-runtime tests for the fail-closed #8872 admission/accounting adapter.

These tests never import a simulator, submit a scheduler job, or execute a
registered episode.  The fake executor only returns provenance-shaped rows so
that packet gates and complete row accounting can be tested locally.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any

import pytest

from scripts.benchmark import run_issue_8872_pedestrian_speed_campaign as campaign

SOURCE_COMMIT = "a" * 40
TOKEN = "private-ops-test-token-" + ("t" * 48)


@pytest.fixture(autouse=True)
def permit_uncommitted_test_checkout(monkeypatch: pytest.MonkeyPatch) -> None:
    """Unit tests use synthetic packet SHAs; no source execution is performed."""
    monkeypatch.setattr(campaign, "_validate_source_checkout", lambda _source: None)


def _binding() -> str:
    manifest = campaign._compiled_manifest()
    return campaign._packet_binding_hash(manifest, SOURCE_COMMIT)


def _receipts() -> dict[str, dict[str, Any]]:
    binding = _binding()
    return {
        "activation_receipt": {
            "schema_version": campaign.ACTIVATION_RECEIPT_SCHEMA_VERSION,
            "issue": 8871,
            "verdict": "activation_pass",
            "preserved": True,
            "current": True,
            "registered_rows_executed": False,
            "seed_disjoint": True,
            "protocol_semantic_hash": campaign.EXPECTED_PROTOCOL_SEMANTIC_HASH,
            "production_manifest_hash": campaign.PRODUCTION_MANIFEST_HASH,
            "source_commit": SOURCE_COMMIT,
            "packet_binding_hash": binding,
            "preservation": {
                "status": "preserved",
                "artifact_reference": "artifact://activation",
                "receipt_sha256": "1" * 64,
            },
        },
        "speed_integrity_receipt": {
            "schema_version": campaign.SPEED_INTEGRITY_RECEIPT_SCHEMA_VERSION,
            "issue": 6102,
            "status": "integrity_validated",
            "preserved": True,
            "manifest_hash": campaign.ROBOT_SPEED_MANIFEST_HASH,
            "expected_rows": 2160,
            "native_rows": 2160,
            "excluded_rows": 0,
            "fallback_rows": 0,
            "degraded_rows": 0,
            "missing_rows": 0,
            "duplicate_rows": 0,
            "provenance_invalid_rows": 0,
            "execution_mode": "native_only",
            "artifact_reference": "artifact://robot-speed",
            "artifact_digest": "2" * 64,
            "source_commit": SOURCE_COMMIT,
            "packet_binding_hash": binding,
        },
        "native_preflight": {
            "schema_version": campaign.NATIVE_PREFLIGHT_RECEIPT_SCHEMA_VERSION,
            "issue": 8872,
            "status": "pass",
            "native": True,
            "fallback": False,
            "degraded": False,
            "planner_ids": list(campaign.EXPECTED_PLANNERS),
            "checkpoint_provenance_complete": True,
            "protocol_semantic_hash": campaign.EXPECTED_PROTOCOL_SEMANTIC_HASH,
            "manifest_hash": campaign.PRODUCTION_MANIFEST_HASH,
            "source_commit": SOURCE_COMMIT,
            "packet_binding_hash": binding,
        },
        "private_admission": {
            "schema_version": campaign.PRIVATE_ADMISSION_RECEIPT_SCHEMA_VERSION,
            "issue": 8872,
            "wrapper_status": "pass",
            "predicates": dict.fromkeys(campaign.REQUIRED_PRIVATE_PREDICATES, True),
            "private_details_excluded": True,
            "source_commit": SOURCE_COMMIT,
            "packet_binding_hash": binding,
        },
        "production_authorization": {
            "schema_version": campaign.AUTHORIZATION_RECEIPT_SCHEMA_VERSION,
            "issue": 8872,
            "scope": "run-production",
            "issuer": "private-ops",
            "decision_id": "decision-test",
            "token_sha256": campaign.hashlib.sha256(TOKEN.encode()).hexdigest(),
            "token_binding_sha256": campaign._token_binding_digest(TOKEN, binding),
            "source_commit": SOURCE_COMMIT,
            "packet_binding_hash": binding,
        },
    }


def _packet() -> dict[str, Any]:
    return campaign.build_production_packet(source_commit=SOURCE_COMMIT, **_receipts())


def _protocol_metrics() -> dict[str, float]:
    return {
        "success_rate": 1.0,
        "collision_rate": 0.0,
        "near_miss_rate": 0.0,
        "time_to_goal_norm": 0.5,
        "total_exposure_seconds": 1.25,
        "travel_distance_m": 12.0,
        "mean_clearance_m": 1.5,
        "min_clearance_m": 0.8,
        "ped_collision_rate": 0.0,
        "obstacle_collision_rate": 0.0,
        "agent_collision_rate": 0.0,
        "unclassified_collision_rate": 0.0,
    }


def _native_outcome(identity: dict[str, Any], packet: dict[str, Any]) -> dict[str, Any]:
    metrics = _protocol_metrics()
    return {
        "identity_key": identity["identity_key"],
        "terminal_status": campaign.SUCCESS_STATUS,
        "metrics": metrics,
        "provenance": {
            "identity_key": identity["identity_key"],
            "scenario_id": identity["scenario_id"],
            "scenario_source_sha256": identity["scenario_source_sha256"],
            "regime_id": identity["regime_id"],
            "planner_id": identity["planner_id"],
            "planner_config_sha256": identity["planner_config_sha256"],
            "seed": identity["seed"],
            "horizon_steps": identity["horizon_steps"],
            "dt_seconds": identity["dt_seconds"],
            "robot_speed_cap_m_s": identity["robot_speed_cap_m_s"],
            "runtime_controls": identity["runtime_controls"],
            "protocol_semantic_hash": campaign.EXPECTED_PROTOCOL_SEMANTIC_HASH,
            "manifest_hash": packet["manifest_hash"],
            "source_commit": packet["source_commit"],
            "execution_mode": "native",
            "native": True,
            "fallback": False,
            "degraded": False,
            "intervention_status": (
                "not_applicable" if identity["regime_id"] == "legacy_default" else "activated"
            ),
            "metrics": metrics,
        },
    }


def test_packet_binds_exact_compiled_manifest_and_all_gates() -> None:
    packet = _packet()

    assert packet["expected_rows"] == campaign.EXPECTED_ROWS
    assert packet["identity_count"] == campaign.EXPECTED_ROWS
    assert packet["unique_identity_count"] == campaign.EXPECTED_ROWS
    assert packet["manifest_hash"] == campaign.PRODUCTION_MANIFEST_HASH
    assert packet["execution_boundary"]["public_repo_submits"] is False
    assert campaign.validate_production_packet(packet)["manifest_hash"] == packet["manifest_hash"]
    readiness = campaign.inspect_packet(packet)
    assert readiness["ready"] is False
    assert "authenticated authorization" in readiness["reason"]


def test_packet_rejects_non_passing_activation_receipt() -> None:
    packet = _packet()
    packet["activation_receipt"]["verdict"] = "invalid_transient"
    packet["packet_sha256"] = campaign._canonical_hash(campaign._packet_core(packet))

    with pytest.raises(campaign.CampaignAdapterError, match="activation_pass"):
        campaign.validate_production_packet(packet)


def test_gate_failure_prevents_any_executor_call() -> None:
    packet = _packet()
    packet["activation_receipt"]["verdict"] = "invalid_transient"
    packet["packet_sha256"] = campaign._canonical_hash(campaign._packet_core(packet))
    calls = 0

    with pytest.raises(campaign.CampaignAdapterError):
        campaign.run_production(
            packet,
            TOKEN,
            checkpoint_root="/tmp/checkpoints",
            output_path="/tmp/8872-test-receipt.json",
            journal_path="/tmp/8872-test-journal.jsonl",
            lock_path="/tmp/8872-test.lock",
        )
    assert calls == 0


def test_fake_native_runtime_accounts_all_rows_once() -> None:
    packet = _packet()
    outcomes = [_native_outcome(identity, packet) for identity in packet["identities"]]
    report = campaign.account_production_rows(
        packet, outcomes, source_commit=packet["source_commit"]
    )

    assert report["admissible"] is True
    assert report["complete_native"] is True
    assert report["terminal_status_counts"] == {campaign.SUCCESS_STATUS: campaign.EXPECTED_ROWS}
    assert len(report["rows"]) == campaign.EXPECTED_ROWS
    assert all(row["missingness"] is None for row in report["rows"])
    assert all("provenance" in row for row in report["rows"])
    assert all(row["metrics"] == _protocol_metrics() for row in report["rows"])


def test_fallback_row_is_recorded_but_never_admitted() -> None:
    packet = _packet()
    outcomes = [_native_outcome(identity, packet) for identity in packet["identities"]]
    outcomes[0] = {
        "identity_key": packet["identities"][0]["identity_key"],
        "terminal_status": "fallback",
        "reason": "fake fallback",
    }
    report = campaign.account_production_rows(
        packet, outcomes, source_commit=packet["source_commit"]
    )

    assert report["admissible"] is False
    assert report["terminal_status_counts"]["fallback"] == 1
    assert report["terminal_status_counts"][campaign.SUCCESS_STATUS] == campaign.EXPECTED_ROWS - 1


def test_duplicate_missing_and_unexpected_outcomes_are_accounted() -> None:
    manifest = campaign._compiled_manifest()
    first = manifest["identities"][0]
    outcome = {
        "identity_key": first["identity_key"],
        "terminal_status": "missing",
    }
    report = campaign.account_production_rows(
        manifest,
        [
            outcome,
            deepcopy(outcome),
            {"identity_key": "unexpected", "terminal_status": "failed"},
        ],
        source_commit=SOURCE_COMMIT,
    )

    assert report["accounted_rows"] == campaign.EXPECTED_ROWS
    assert report["unique_accounted_identities"] == campaign.EXPECTED_ROWS
    assert report["unexpected_outcome_count"] == 1
    assert report["terminal_status_counts"]["duplicate"] == 1
    assert report["terminal_status_counts"]["missing"] == campaign.EXPECTED_ROWS - 1
    assert report["admissible"] is False


def test_intervention_not_activated_cannot_be_native_success() -> None:
    packet = _packet()
    treated_identity = next(
        identity for identity in packet["identities"] if identity["regime_id"] != "legacy_default"
    )
    outcome = _native_outcome(treated_identity, packet)
    outcome["provenance"]["intervention_status"] = "not_activated"

    rows = campaign.account_production_rows(
        packet,
        [outcome],
        source_commit=packet["source_commit"],
    )["rows"]
    row = next(item for item in rows if item["identity_key"] == treated_identity["identity_key"])
    assert row["terminal_status"] == "intervention_not_activated"
    assert row["missingness"] == "intervention_not_activated"


def _fake_native_record(*, preferred_speed: float = 0.65) -> dict[str, Any]:
    return {
        "metrics": _protocol_metrics(),
        "algorithm_metadata": {
            "status": "ok",
            "planner_kinematics": {"execution_mode": "native"},
            "simulation_step_trace": {
                "dt": 0.1,
                "reset": {"pedestrians": [{"actor_id": "p0", "velocity": [0.5, 0.0]}]},
                "steps": [
                    {
                        "oracle_transition_trace": {
                            "transitions": [
                                {
                                    "simulator_pedestrian_id": "p0",
                                    "dynamics": {"preferred_speed_mps": preferred_speed},
                                    "post_integration": {"velocity_xy": [preferred_speed, 0.0]},
                                }
                            ]
                        }
                    }
                ],
            },
        },
    }


def test_fixed_native_record_adapter_uses_trace_not_executor_flag() -> None:
    packet = _packet()
    identity = next(
        item for item in packet["identities"] if item["regime_id"] == "slow_distributed"
    )
    outcome = campaign._native_outcome_from_record(
        identity,
        _fake_native_record(preferred_speed=0.65),
        source_commit=packet["source_commit"],
        manifest_hash=packet["manifest_hash"],
        planner_algorithm="fake-native-planner",
        robot_speed_cap_m_s=2.0,
        checkpoint_provenance=[],
    )
    assert outcome["terminal_status"] == campaign.SUCCESS_STATUS
    assert outcome["provenance"]["runtime_controls"] == identity["runtime_controls"]
    assert outcome["provenance"]["diagnostics"]["desired_speed_activation_fraction"] == 1.0
    assert outcome["metrics"] == _protocol_metrics()
    assert outcome["provenance"]["metrics"] == _protocol_metrics()
    assert "executor_flag" not in outcome["provenance"]


def test_native_record_projects_established_raw_metric_contract() -> None:
    packet = _packet()
    identity = next(
        item for item in packet["identities"] if item["regime_id"] == "slow_distributed"
    )
    record = _fake_native_record()
    record["metrics"] = {
        "success": True,
        "collisions": 0.0,
        "near_misses": 1.0,
        "ped_collision_count": 0.0,
        "obstacle_collision_count": 0.0,
        "agent_collision_count": 0.0,
        "time_to_goal_norm": 0.4,
        "socnavbench_path_length": 9.0,
        "mean_clearance": 1.2,
        "min_clearance": 0.7,
    }
    record["interaction_exposure"] = {
        "interaction_exposure_share": 0.25,
        "interaction_exposure_denominator_steps": 20.0,
    }

    outcome = campaign._native_outcome_from_record(
        identity,
        record,
        source_commit=packet["source_commit"],
        manifest_hash=packet["manifest_hash"],
        planner_algorithm="fake-native-planner",
        robot_speed_cap_m_s=2.0,
        checkpoint_provenance=[],
    )

    assert outcome["terminal_status"] == campaign.SUCCESS_STATUS
    assert outcome["metrics"] == {
        "success_rate": 1.0,
        "collision_rate": 0.0,
        "near_miss_rate": 1.0,
        "time_to_goal_norm": 0.4,
        "total_exposure_seconds": 0.5,
        "travel_distance_m": 9.0,
        "mean_clearance_m": 1.2,
        "min_clearance_m": 0.7,
        "ped_collision_rate": 0.0,
        "obstacle_collision_rate": 0.0,
        "agent_collision_rate": 0.0,
        "unclassified_collision_rate": 0.0,
    }


def test_native_record_rejects_non_finite_protocol_metric() -> None:
    packet = _packet()
    identity = next(
        item for item in packet["identities"] if item["regime_id"] == "slow_distributed"
    )
    record = _fake_native_record()
    record["metrics"]["collision_rate"] = float("nan")

    outcome = campaign._native_outcome_from_record(
        identity,
        record,
        source_commit=packet["source_commit"],
        manifest_hash=packet["manifest_hash"],
        planner_algorithm="fake-native-planner",
        robot_speed_cap_m_s=2.0,
        checkpoint_provenance=[],
    )

    assert outcome["terminal_status"] == "provenance_invalid"
    assert outcome["reason"] == "metric_contract_nonfinite:collision_rate"
    campaign.json.dumps(outcome, allow_nan=False)


def test_non_success_reason_redacts_private_exception_detail() -> None:
    identity = campaign._compiled_manifest()["identities"][0]
    row = campaign._normalize_outcome(
        identity,
        {
            "identity_key": identity["identity_key"],
            "terminal_status": "failed",
            "reason": "RuntimeError: /private/cluster/checkpoints/model.zip",
        },
        source_commit=SOURCE_COMMIT,
        manifest_hash=campaign.PRODUCTION_MANIFEST_HASH,
    )

    assert row["reason"] == "reason:unsafe_detail_redacted"
    assert "/private/cluster" not in str(row)
    assert campaign._exception_reason(RuntimeError("/private/cluster")) == (
        "exception:RuntimeError"
    )


def test_terminal_journal_preserves_metrics_and_reconcile_hides_path(tmp_path: Any) -> None:
    journal = tmp_path / "private" / "run.jsonl"
    journal.parent.mkdir()
    with journal.open("w", encoding="utf-8") as handle:
        campaign._append_journal_event(
            handle,
            "row_finished",
            **campaign._journal_row_payload(
                {
                    "identity_key": "identity-1",
                    "terminal_status": campaign.SUCCESS_STATUS,
                    "missingness": None,
                    "reason": None,
                    "metrics": _protocol_metrics(),
                    "provenance": {"source_commit": SOURCE_COMMIT},
                }
            ),
        )

    summary = campaign.reconcile_execution_journal(journal, expected_rows=1)

    assert summary["journal_path"] == "run.jsonl"
    assert summary["journal_reference"] == "run.jsonl"
    assert summary["rows"][0]["metrics"] == _protocol_metrics()
    assert str(journal) not in campaign.json.dumps(summary, sort_keys=True)


def test_self_minted_authorization_cannot_enter_production_runner(tmp_path: Any) -> None:
    packet = _packet()

    with pytest.raises(campaign.CampaignAdapterError, match="authenticated authorization"):
        campaign.run_production(
            packet,
            TOKEN,
            checkpoint_root=tmp_path / "checkpoints",
            output_path=tmp_path / "receipt.json",
            journal_path=tmp_path / "run.jsonl",
            lock_path=tmp_path / "run.lock",
        )
    assert not (tmp_path / "run.jsonl").exists()
    assert not (tmp_path / "receipt.json").exists()


def test_fixed_executor_fake_runner_receives_bound_native_identity(
    tmp_path: Any,
) -> None:
    protocol = campaign.load_protocol()
    from scripts.benchmark.run_issue_8871_pedestrian_speed_canary import (
        _load_planner_specs,
        _load_scenarios,
    )

    manifest = campaign._compiled_manifest()
    identity = next(
        item
        for item in manifest["identities"]
        if item["planner_id"] == "orca" and item["regime_id"] == "slow_distributed"
    )
    scenarios = _load_scenarios(protocol)
    planner_specs = {str(item["planner_id"]): item for item in _load_planner_specs(protocol)}
    observed: dict[str, Any] = {}

    def fake_runner(**kwargs: Any) -> dict[str, Any]:
        observed.update(kwargs)
        from robot_sf.benchmark.map_runner import map_runner_episode

        observed["runtime_config"] = map_runner_episode._build_env_config(
            kwargs["scenario"], scenario_path=kwargs["scenario_path"]
        )
        return _fake_native_record(preferred_speed=0.65)

    outcome = campaign._execute_native_identity(
        identity,
        source_commit=SOURCE_COMMIT,
        manifest_hash=campaign.PRODUCTION_MANIFEST_HASH,
        protocol=protocol,
        scenarios=scenarios,
        planner_specs=planner_specs,
        checkpoints={},
        policy_builder=lambda *args, **kwargs: (None, {}),
        checkpoint_root=tmp_path,
        episode_runner=fake_runner,
    )

    assert outcome["terminal_status"] == campaign.SUCCESS_STATUS
    assert observed["seed"] == identity["seed"]
    assert observed["horizon"] == identity["horizon_steps"]
    assert observed["dt"] == identity["dt_seconds"]
    assert observed["scenario"]["simulation_config"]["desired_speed_mean"] == 0.65
    assert observed["scenario"]["simulation_config"]["desired_speed_seed"] == "episode_seed"
    assert observed["runtime_config"].sim_config.desired_speed_seed == identity["seed"]


def test_receipt_source_and_artifact_refs_are_cross_bound_and_safe() -> None:
    packet = _packet()
    packet["native_preflight"]["source_commit"] = "b" * 40
    packet["packet_sha256"] = campaign._canonical_hash(campaign._packet_core(packet))
    with pytest.raises(campaign.CampaignAdapterError, match="does not match production packet"):
        campaign.validate_production_packet(packet)

    packet = _packet()
    packet["activation_receipt"]["preservation"]["artifact_reference"] = "/private/raw"
    packet["packet_sha256"] = campaign._canonical_hash(campaign._packet_core(packet))
    with pytest.raises(campaign.CampaignAdapterError, match="safe durable artifact"):
        campaign.validate_production_packet(packet)


def test_token_is_bound_to_packet_and_not_self_authenticated() -> None:
    packet = _packet()
    campaign._validate_token_binding(
        packet["production_authorization"], packet["packet_binding_hash"], TOKEN
    )
    with pytest.raises(campaign.CampaignAdapterError, match="bound to this exact packet"):
        campaign._validate_token_binding(packet["production_authorization"], "f" * 64, TOKEN)


def test_receipts_reject_raw_tokens_and_unknown_authorization_keys() -> None:
    for field, key in (
        ("production_authorization", "token"),
        ("speed_integrity_receipt", "execution_token"),
    ):
        packet = _packet()
        packet[field][key] = TOKEN
        packet["packet_sha256"] = campaign._canonical_hash(campaign._packet_core(packet))

        with pytest.raises(campaign.CampaignAdapterError, match="secret material"):
            campaign.validate_production_packet(packet)


def test_cli_summary_uses_safe_output_reference() -> None:
    private_output = "/private/cluster/receipts/issue-8872.json"
    summary = campaign._summary("smoke", {}, output=campaign.Path(private_output))

    rendered = campaign.json.dumps(summary, sort_keys=True)
    assert summary["output"] == "issue-8872.json"
    assert private_output not in rendered


def test_existing_journal_refuses_retry(tmp_path: Any) -> None:
    journal = tmp_path / "run.jsonl"
    journal.write_text('{"event":"row_started","identity_key":"x"}\n', encoding="utf-8")
    summary = campaign.reconcile_execution_journal(journal, expected_rows=2160)
    assert summary["retry_allowed"] is False
    assert summary["in_flight_identity_keys"] == ["x"]
    with pytest.raises(campaign.CampaignAdapterError, match="automatic retry is forbidden"):
        campaign._prepare_execution_paths(tmp_path / "receipt.json", journal, tmp_path / "run.lock")


def test_smoke_packet_is_tiny_disjoint_and_diagnostic() -> None:
    smoke = campaign.compile_smoke_manifest(source_commit="b" * 40)
    production_keys = {
        identity["identity_key"] for identity in campaign._compiled_manifest()["identities"]
    }

    assert smoke["expected_rows"] == 3
    assert smoke["diagnostic_only"] is True
    assert smoke["execution_allowed"] is False
    assert smoke["scientific_evidence"] is False
    assert all(identity["registered"] is False for identity in smoke["identities"])
    assert not production_keys.intersection(
        identity["identity_key"] for identity in smoke["identities"]
    )


def test_smoke_packet_cannot_enter_production_runner() -> None:
    smoke = campaign.compile_smoke_manifest(source_commit="b" * 40)
    calls = 0

    with pytest.raises(campaign.CampaignAdapterError, match="production packet schema"):
        campaign.run_production(
            smoke,
            TOKEN,
            checkpoint_root="/tmp/checkpoints",
            output_path="/tmp/8872-test-receipt.json",
            journal_path="/tmp/8872-test-journal.jsonl",
            lock_path="/tmp/8872-test.lock",
        )
    assert calls == 0


def test_validate_summary_does_not_print_identity_rows(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert campaign.main(["validate"]) == 0
    output = capsys.readouterr().out
    assert "identity_count" not in output
    assert "identities" not in output
