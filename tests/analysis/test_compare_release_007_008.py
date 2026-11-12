"""Synthetic contracts for the 0.0.7→0.0.8 comparison gate."""

from __future__ import annotations

import hashlib
import json
import subprocess
from typing import TYPE_CHECKING

import pytest
import yaml

from scripts.analysis import compare_release_007_008 as comparator
from scripts.analysis.compare_release_007_008 import (
    ATTRIBUTION_SCHEMA,
    HISTORICAL_BUNDLE_SHA256,
    RECEIPT_SCHEMA,
    _finding,
    compare_with_attribution,
    load_attribution_ledger,
    read_candidate_rows,
    read_historical_bundle,
)

if TYPE_CHECKING:
    from pathlib import Path


def _row(arm: str, *, success: bool, collision: bool, score: float) -> dict:
    return {
        "outcome": {
            "route_complete": success,
            "collision_event": collision,
            "timeout_event": False,
        },
        "metrics": {"snqi": score, "path_length": 3.0},
        "metric_values": {"clearance": 0.5},
        "status": "ok",
        "integrity": {"contradictions": []},
        "steps": 10,
        "algorithm_metadata": {
            "planner_kinematics": {"execution_mode": "native"},
            "algorithm": arm,
        },
    }


def _slot(old: str = "arm_a", new: str | None = None, replaced: bool = False) -> dict:
    return {"old_key": old, "new_key": new or old, "implementation_replaced": replaced}


def _compare(old, new, slots, ledger=None, *, scenarios=("s",), seeds=(111,)):
    findings = []
    summary = compare_with_attribution(
        old,
        new,
        slots,
        ledger or {},
        scenarios=scenarios,
        seeds=seeds,
        emit=findings.append,
    )
    return summary, findings


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_unexplained_outcome_and_metric_changes_block_at_1e12() -> None:
    old = {("arm_a", "s", 111): _row("arm_a", success=False, collision=True, score=0.2)}
    new = {("arm_a", "s", 111): _row("arm_a", success=True, collision=False, score=0.4)}
    summary, findings = _compare(old, new, [_slot()])
    assert {item["field"] for item in findings} == {
        "outcome.route_complete",
        "outcome.collision_event",
        "metrics.snqi",
    }
    assert len(summary["unexplained_findings"]) == 3
    assert not summary["comparison_passed"]
    assert summary["rate_impacts"]["success_rate"]["arm_a"]["delta"] == 1.0
    assert summary["rate_impacts"]["collision_rate"]["arm_a"]["delta"] == -1.0


def test_raw_candidate_degradation_and_contradiction_block_unchanged_pair() -> None:
    old_row = _row("arm_a", success=True, collision=False, score=1.0)
    new_row = _row("arm_a", success=True, collision=False, score=1.0)
    new_row.update(
        {
            "readiness_status": "degraded",
            "row_status": "degraded",
            "availability_status": "unavailable",
            "fallback_used": True,
            "planner_runtime": {"fallback_count": 1},
            "integrity": {"contradictions": ["invalid"]},
        }
    )
    summary, findings = _compare(
        {("arm_a", "s", 111): old_row},
        {("arm_a", "s", 111): new_row},
        [_slot()],
    )
    assert findings == []
    candidate_anomalies = [
        item for item in summary["row_anomalies"] if item["kind"] == "degraded_candidate_row"
    ]
    assert len(candidate_anomalies) == 1
    details = candidate_anomalies[0]["reasons"]
    assert any("readiness_status=degraded" in item for item in details)
    assert any("row_status=degraded" in item for item in details)
    assert any("availability_status=unavailable" in item for item in details)
    assert any("fallback_used=true" in item for item in details)
    assert any("fallback_count" in item for item in details)
    assert any("integrity.contradictions" in item for item in details)
    assert not summary["comparison_passed"]


def test_overlapping_guarded_ppo_status_checks_report_one_candidate_reason() -> None:
    row = _row("guarded_ppo", success=True, collision=False, score=1.0)
    row["algorithm_metadata"]["guard_stats"] = {"stop_best_effort": 1}
    summary, findings = _compare(
        {("guarded_ppo", "s", 111): row},
        {("guarded_ppo", "s", 111): row},
        [_slot("guarded_ppo")],
    )
    assert findings == []
    assert summary["row_anomalies"] == [
        {
            "kind": "degraded_candidate_row",
            "identity": ["guarded_ppo", "s", 111],
            "reasons": ["algorithm_metadata.guard_stats.stop_best_effort=1"],
        }
    ]
    assert not summary["comparison_passed"]


def test_tolerance_detects_only_change_exceeding_1e12() -> None:
    old = {("arm_a", "s", 111): _row("arm_a", success=True, collision=False, score=1.0)}
    near = {
        ("arm_a", "s", 111): _row("arm_a", success=True, collision=False, score=1.0000000000005)
    }
    far = {("arm_a", "s", 111): _row("arm_a", success=True, collision=False, score=1.000000000002)}
    assert _compare(old, near, [_slot()])[1] == []
    assert [item["field"] for item in _compare(old, far, [_slot()])[1]] == ["metrics.snqi"]


def test_legacy_nonfinite_sentinels_are_symbolic_and_changed_values_are_findings() -> None:
    old_row = _row("arm_a", success=True, collision=False, score=1.0)
    old_row["metrics"]["min_separation_corrupted_m"] = float("inf")
    old_row["metric_values"]["min_predicted_separation_m"] = float("nan")
    old = {("arm_a", "s", 111): old_row}
    same = {("arm_a", "s", 111): dict(old_row)}
    same_summary, same_findings = _compare(old, same, [_slot()])
    assert same_findings == []
    assert (
        same_summary["nonfinite_common_metric_counts"]["old:metrics.min_separation_corrupted_m"]
        == 1
    )
    changed_row = _row("arm_a", success=True, collision=False, score=1.0)
    changed_row["metrics"]["min_separation_corrupted_m"] = 2.0
    changed_row["metric_values"]["min_predicted_separation_m"] = 1.0
    _, findings = _compare(old, {("arm_a", "s", 111): changed_row}, [_slot()])
    assert {item["field"] for item in findings} == {
        "metrics.min_separation_corrupted_m",
        "metric_values.min_predicted_separation_m",
    }
    assert all(
        item["old"] in ({"nonfinite": "+Infinity"}, {"nonfinite": "NaN"}) for item in findings
    )


def test_v4_replacement_pairs_by_historical_slot_and_labels_finding() -> None:
    old = {("old_v3", "s", 111): _row("old_v3", success=True, collision=False, score=1.0)}
    new = {("new_v4", "s", 111): _row("new_v4", success=True, collision=False, score=2.0)}
    summary, findings = _compare(old, new, [_slot("old_v3", "new_v4", True)])
    assert summary["paired_rows"] == 1
    assert summary["inventory"]["missing_new"] == []
    assert findings[0]["identity"] == ["old_v3", "s", 111]
    assert findings[0]["comparison"] == "implementation replaced"


def test_missing_extra_and_missing_common_metric_block() -> None:
    old = {("arm_a", "s", 111): _row("arm_a", success=True, collision=False, score=1.0)}
    new_row = _row("arm_a", success=True, collision=False, score=1.0)
    del new_row["metrics"]["snqi"]
    new = {("arm_a", "s", 111): new_row, ("arm_a", "outside", 111): new_row}
    summary, _ = _compare(old, new, [_slot()], seeds=(111, 112))
    assert summary["inventory"]["missing_old"] == [["arm_a", "s", 112]]
    assert summary["inventory"]["missing_new"] == [["arm_a", "s", 112]]
    assert summary["inventory"]["extra_new"] == [["arm_a", "outside", 111]]
    assert any(item.get("field") == "metrics.snqi" for item in summary["row_anomalies"])
    assert not summary["comparison_passed"]


def test_available_mapped_extra_episode_is_diffed_but_excluded_from_rates() -> None:
    old = {
        ("arm_a", "s", 111): _row("arm_a", success=True, collision=False, score=1.0),
        ("arm_a", "outside", 111): _row("arm_a", success=False, collision=False, score=1.0),
    }
    new = {
        ("arm_a", "s", 111): _row("arm_a", success=True, collision=False, score=1.0),
        ("arm_a", "outside", 111): _row("arm_a", success=True, collision=False, score=2.0),
    }
    summary, findings = _compare(old, new, [_slot()])
    assert summary["paired_rows"] == 2
    assert summary["paired_contract_rows"] == 1
    assert {item["field"] for item in findings} == {"outcome.route_complete", "metrics.snqi"}
    assert summary["rate_impacts"]["success_rate"]["arm_a"]["old"] == 1.0
    assert not summary["comparison_passed"]


def test_explained_change_with_checked_receipt_can_pass(tmp_path: Path) -> None:
    old = {("arm_a", "s", 111): _row("arm_a", success=True, collision=False, score=1.0)}
    new = {("arm_a", "s", 111): _row("arm_a", success=True, collision=False, score=2.0)}
    finding = _finding("arm_a", "s", 111, "metrics.snqi", 1.0, 2.0, False)
    evidence = tmp_path / "evidence.json"
    evidence.write_text('{"trace": "reviewed synthetic example"}\n')
    receipt = tmp_path / "receipt.json"
    receipt.write_text(
        json.dumps(
            {
                "schema_version": RECEIPT_SCHEMA,
                "change_id": "planner-v4",
                "finding_ids": [finding["finding_id"]],
                "mechanism": "versioned planner correction changes path",
                "evidence_path": evidence.name,
                "evidence_sha256": _sha(evidence),
                "review": {
                    "decision": "accepted",
                    "reviewer": "example-reviewer",
                    "reviewed_at_utc": "2026-09-28T00:00:00Z",
                },
            }
        )
    )
    ledger_path = tmp_path / "ledger.json"
    ledger_path.write_text(
        json.dumps(
            {
                "schema_version": ATTRIBUTION_SCHEMA,
                "candidate_source_sha": "a" * 40,
                "entries": [
                    {
                        "finding_id": finding["finding_id"],
                        "change_id": "planner-v4",
                        "receipt_path": receipt.name,
                        "receipt_sha256": _sha(receipt),
                    }
                ],
            }
        )
    )
    identity = {"source_sha": "a" * 40, "versioned_changes": [{"id": "planner-v4"}]}
    ledger, anomalies, _ = load_attribution_ledger(ledger_path, identity)
    assert anomalies == []
    summary, findings = _compare(old, new, [_slot()], ledger)
    assert summary["comparison_passed"]
    assert findings[0]["attribution"]["change_id"] == "planner-v4"
    evidence.write_text("changed")
    _, anomalies, _ = load_attribution_ledger(ledger_path, identity)
    assert [item["kind"] for item in anomalies] == ["invalid_causal_receipt"]


def test_rate_ranking_moves_on_paired_rows() -> None:
    old = {
        ("arm_a", "s", 111): _row("arm_a", success=True, collision=False, score=1),
        ("arm_b", "s", 111): _row("arm_b", success=False, collision=False, score=1),
    }
    new = {
        ("arm_a", "s", 111): _row("arm_a", success=False, collision=False, score=1),
        ("arm_b", "s", 111): _row("arm_b", success=True, collision=False, score=1),
    }
    summary, _ = _compare(old, new, [_slot("arm_a"), _slot("arm_b")])
    assert summary["rate_impacts"]["success_rate"]["arm_a"]["rank_shift"] == 1
    assert summary["rate_impacts"]["success_rate"]["arm_b"]["rank_shift"] == -1


def test_wrong_historical_bundle_cannot_be_promoted(tmp_path: Path) -> None:
    fake = tmp_path / "historical.tar.gz"
    fake.write_bytes(b"not the accepted bundle")
    with pytest.raises(ValueError, match="historical bundle differs"):
        read_historical_bundle(fake)
    assert HISTORICAL_BUNDLE_SHA256 != _sha(fake)


def test_candidate_reader_reports_duplicate_and_nonfinite_rows(tmp_path: Path) -> None:
    run = tmp_path / "runs" / "arm_a__differential_drive"
    run.mkdir(parents=True)
    row = _row("arm_a", success=True, collision=False, score=1.0)
    row.update({"scenario_id": "s", "seed": 111, "algo": "arm_a", "git_hash": "a" * 40})
    episode_file = run / "episodes.jsonl"
    bad_algo = dict(row, seed=112, algo=["arm_a"])
    bad_status = dict(row, seed=113, readiness_status="degraded", fallback_used=True)
    bad_status["integrity"] = {"contradictions": ["bad"]}
    episode_file.write_text(
        json.dumps(row)
        + "\n"
        + json.dumps(row)
        + "\n"
        + "{invalid\n"
        + json.dumps(bad_algo)
        + "\n"
        + json.dumps(bad_status)
        + "\n"
    )
    identity = {
        "source_sha": "a" * 40,
        "arm_slots": [{"new_key": "arm_a", "row_algos": ["arm_a"]}],
        "_effective_algorithms": {"arm_a": {"s": "arm_a"}},
        "episode_files": {"runs/arm_a__differential_drive/episodes.jsonl": _sha(episode_file)},
    }
    rows, anomalies = read_candidate_rows(tmp_path, identity)
    assert len(rows) == 3
    assert {item["kind"] for item in anomalies} == {
        "duplicate_identity",
        "malformed_json",
        "arm_mismatch",
        "ineligible_candidate_row",
    }
    assert any(
        any("integrity.contradictions" in reason for reason in item["reasons"])
        for item in anomalies
        if item["kind"] == "ineligible_candidate_row"
    )


def test_candidate_identity_requires_exact_roster_and_v4_names(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(comparator, "_source_head", lambda _root: "a" * 40)
    monkeypatch.setattr(comparator, "_bound_file", lambda _root, _path, digest, **_kw: digest)
    monkeypatch.setattr(comparator, "_validate_effective_algorithms", lambda _identity, _root: {})
    monkeypatch.setattr(comparator, "_candidate_run_controls", lambda _identity, _root: {})
    slots = [
        {
            "old_key": old,
            "new_key": old,
            "implementation_replaced": False,
            "implementation_version": "0.0.8-v1",
            "row_algos": ["hybrid_rule_local_planner"] if old in comparator.HYBRID_SLOTS else [old],
            "config_path": "config.json",
            "config_sha256": "b" * 64,
        }
        for old in sorted(comparator.EXPECTED_ARM_KEYS)
    ]
    hybrid = next(slot for slot in slots if slot["old_key"] in comparator.HYBRID_SLOTS)
    hybrid["new_key"] = hybrid["old_key"].replace("v3", "v4").replace("v2", "v4")
    hybrid["implementation_replaced"] = True
    identity = {
        "schema_version": comparator.CANDIDATE_SCHEMA,
        "release": "0.0.8",
        "source_sha": "a" * 40,
        "effective_config_path": "config.json",
        "effective_config_sha256": "b" * 64,
        "scenario_matrix": {"path": "matrix.yaml", "sha256": "c" * 64},
        "arm_slots": slots,
        "scenario_ids": sorted(comparator.EXPECTED_SCENARIO_IDS),
        "seeds": list(comparator.EXPECTED_SEEDS),
        "episode_files": {"runs/example/episodes.jsonl": "d" * 64},
        "versioned_changes": [
            {
                "id": "source-v4",
                "kind": "source",
                "version": "v4",
                "old_identity": comparator.HISTORICAL_SOURCE_SHA,
                "new_identity": "a" * 40,
            }
        ],
    }
    path = tmp_path / "identity.json"
    path.write_text(json.dumps(identity))
    assert comparator.load_candidate_identity(path, tmp_path)["arm_slots"] == slots
    hybrid["new_key"] = hybrid["old_key"] + "_renamed"
    path.write_text(json.dumps(identity))
    with pytest.raises(ValueError, match="v4 replacement"):
        comparator.load_candidate_identity(path, tmp_path)


def _write_yaml(root: Path, relative: str, payload: dict) -> str:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=True))
    return _sha(path)


def _git_commit(root: Path) -> str:
    subprocess.run(["git", "-C", str(root), "add", "."], check=True, capture_output=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(root),
            "-c",
            "user.name=Fixture",
            "-c",
            "user.email=fixture@example.invalid",
            "commit",
            "-m",
            "fixture",
        ],
        check=True,
        capture_output=True,
    )
    return subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _versioned_v4_fixture(tmp_path: Path) -> tuple[Path, Path, dict, str]:
    source = tmp_path / "source"
    source.mkdir()
    subprocess.run(["git", "-C", str(source), "init", "-q"], check=True, capture_output=True)
    v4_base_sha = _write_yaml(
        source,
        comparator.V4_BASE_CONFIG_PATH,
        {
            "planner_variant": comparator.V4_PLANNER_VARIANT,
        },
    )
    v3_path = "configs/algos/hybrid_rule_v3_fixture.yaml"
    v3_base_sha = _write_yaml(source, v3_path, {"planner_variant": "hybrid_rule_v3"})
    orca_sha = _write_yaml(source, comparator.APPROVED_ORCA_CONFIG_PATH, {"orca_time_horizon": 5.0})
    matrix_sha = _write_yaml(source, "configs/matrix.yaml", {"fixture": True})
    slots = []
    planners = []
    target_old = "scenario_adaptive_hybrid_orca_v2_bottleneck_yield"
    for old in sorted(comparator.EXPECTED_ARM_KEYS):
        hybrid = old in comparator.HYBRID_SLOTS
        replaced = old == target_old
        new = old + "_v4" if replaced else old
        slot = {
            "old_key": old,
            "new_key": new,
            "implementation_replaced": replaced,
            "implementation_version": "v4" if replaced else "historical",
            "row_algos": (
                ["hybrid_rule_local_planner", "orca"]
                if old in comparator.ADAPTIVE_HYBRID_SLOTS
                else (["hybrid_rule_local_planner"] if hybrid else [new])
            ),
            "config_path": None,
            "config_sha256": None,
        }
        planner = {"key": new, "algo": "hybrid_rule_local_planner" if hybrid else new}
        if hybrid:
            config_path = f"configs/policy_search/candidates/{new}.yaml"
            config = {
                "name": new + "_s30_h600_release",
                "algo": "hybrid_rule_local_planner",
                "base_config_path": comparator.V4_BASE_CONFIG_PATH if replaced else v3_path,
                "params": {},
            }
            if old in comparator.ADAPTIVE_HYBRID_SLOTS:
                config["scenario_algo_overrides"] = {
                    comparator.APPROVED_ORCA_HANDOFF_SCENARIO: {
                        "algo": "orca",
                        "base_config_path": comparator.APPROVED_ORCA_CONFIG_PATH,
                        "params": {},
                    }
                }
                slot["handoff_config_sha256"] = orca_sha
            slot.update(
                {
                    "config_path": config_path,
                    "config_sha256": _write_yaml(source, config_path, config),
                    "base_config_sha256": v4_base_sha if replaced else v3_base_sha,
                }
            )
            planner["algo_config"] = config_path
        slots.append(slot)
        planners.append(planner)
    campaign_sha = _write_yaml(source, "configs/campaign.yaml", {"planners": planners})
    source_sha = _git_commit(source)
    identity = {
        "schema_version": comparator.CANDIDATE_SCHEMA,
        "release": "0.0.8",
        "source_sha": source_sha,
        "effective_config_path": "configs/campaign.yaml",
        "effective_config_sha256": campaign_sha,
        "scenario_matrix": {"path": "configs/matrix.yaml", "sha256": matrix_sha},
        "arm_slots": slots,
        "scenario_ids": sorted(comparator.EXPECTED_SCENARIO_IDS),
        "seeds": list(comparator.EXPECTED_SEEDS),
        "episode_files": {"runs/example/episodes.jsonl": "d" * 64},
        "versioned_changes": [
            {
                "id": "v4-source",
                "kind": "source",
                "version": "v4",
                "old_identity": comparator.HISTORICAL_SOURCE_SHA,
                "new_identity": source_sha,
            }
        ],
    }
    identity_path = tmp_path / "identity.json"
    identity_path.write_text(json.dumps(identity))
    return source, identity_path, identity, target_old


def _run_control_fixture(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict:
    source = tmp_path / "run-control-source"
    matrix = source / "configs/matrix.yaml"
    campaign = source / "configs/campaign.yaml"
    matrix.parent.mkdir(parents=True)
    matrix.write_text(
        yaml.safe_dump(
            {
                "scenarios": [
                    {
                        "name": "doorway",
                        "id": "doorway",
                        "seeds": list(comparator.EXPECTED_SEEDS),
                        "robot_config": {"type": "holonomic", "command_mode": "goal"},
                        "simulation_config": {
                            "route_spawn_seed": None,
                            "goal_completion_policy": "zone",
                        },
                    }
                ]
            },
            sort_keys=True,
        )
    )
    campaign.write_text(
        yaml.safe_dump(
            {
                "name": "run-control-fixture",
                "scenario_matrix": "matrix.yaml",
                "horizon": 600,
                "dt": 0.1,
                "kinematics_matrix": ["differential_drive"],
                "planners": [{"key": "fixture_arm", "algo": "goal"}],
            },
            sort_keys=True,
        )
    )
    identity = {
        "source_sha": "a" * 40,
        "effective_config_path": "configs/campaign.yaml",
        "scenario_matrix": {
            "path": "configs/matrix.yaml",
            "sha256": _sha(matrix),
        },
        "scenario_ids": ["doorway"],
        "_effective_algorithms": {"fixture_arm": {"doorway": "goal"}},
    }
    monkeypatch.setattr(
        comparator,
        "_runtime_successor_identity",
        lambda *_args: (
            "b" * 64,
            "c" * 64,
            {},
            {("fixture_arm", "differential_drive"): "d" * 64},
            set(),
        ),
    )
    return comparator._candidate_run_controls(identity, source)


@pytest.mark.parametrize(
    ("mutate", "reason"),
    [
        (lambda row: row.update({"horizon": 300}), "row.horizon"),
        (
            lambda row: row["result_provenance"]["simulator_settings"].update({"dt": 0.2}),
            "simulator_settings.dt",
        ),
        (
            lambda row: row["scenario_params"].update({"run_dt": 0.2}),
            "scenario_params differs from source-bound campaign, planner, and scenario",
        ),
        (
            lambda row: row["scenario_params"]["robot_config"].update({"type": "holonomic"}),
            "scenario_params differs from source-bound campaign, planner, and scenario",
        ),
        (
            lambda row: row["scenario_params"]["robot_config"].update({"command_mode": "vx_vy"}),
            "scenario_params differs from source-bound campaign, planner, and scenario",
        ),
        (
            lambda row: row["scenario_params"].update({"algo": "social_force"}),
            "source-bound campaign, planner, and scenario",
        ),
        (
            lambda row: row["provenance"]["config_identity"].update(
                {"scenario_matrix_hash": "e" * 64}
            ),
            "scenario_matrix_hash differs from pinned scoped runner",
        ),
        (
            lambda row: row["provenance"]["config_identity"].update(
                {"algo_config_path": "configs/algos/unpinned.yaml"}
            ),
            "algo_config_path differs from pinned source",
        ),
        (
            lambda row: row["provenance"].update({"commit_hash": "f" * 40}),
            "commit_hash differs from pinned source",
        ),
        (
            lambda row: row["provenance"]["config_identity"].update(
                {"campaign_config_hash": "e" * 64}
            ),
            "campaign_config_hash differs from pinned campaign config",
        ),
        (
            lambda row: row["algorithm_metadata"].update({"config_hash": "e" * 64}),
            "algorithm_metadata.config_hash differs from pinned planner config",
        ),
    ],
)
def test_candidate_run_controls_reject_rehashed_control_changes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mutate, reason: str
) -> None:
    controls = _run_control_fixture(tmp_path, monkeypatch)
    scenario = controls["scenarios"]["doorway"]
    scenario_with_seed = comparator._scenario_with_episode_seed_defaults(
        scenario["scenario"], seed=111
    )
    arm_control = controls["arms"]["fixture_arm"]
    policy = arm_control["policy_configs"]["doorway"]
    effective_policy_config = comparator._apply_planner_selector_v2_context(
        policy["algo"],
        dict(policy["policy_config"]),
        scenario=scenario_with_seed,
        seed=111,
    )
    effective_policy_config = comparator._apply_scenario_uncertainty_envelope_config(
        policy["algo"], effective_policy_config, scenario_with_seed
    )
    params = comparator._scenario_identity_payload(
        scenario_with_seed,
        algo=policy["algo"],
        algo_config=policy["policy_config"],
        horizon=arm_control["horizon"],
        dt=arm_control["dt"],
        record_forces=arm_control["record_forces"],
        observation_mode=arm_control["observation_mode"],
        observation_noise=arm_control["observation_noise"],
        synthetic_actuation_profile=arm_control["synthetic_actuation_profile"],
        latency_stress_profile=arm_control["latency_stress_profile"],
        safety_wrapper=arm_control["safety_wrapper"],
        record_planner_decision_trace=arm_control["record_planner_decision_trace"],
        record_simulation_step_trace=arm_control["record_simulation_step_trace"],
    )
    row = {
        "scenario_id": "doorway",
        "seed": 111,
        "algo": policy["algo"],
        "horizon": arm_control["horizon"],
        "config_hash": comparator._config_hash(params),
        "result_provenance": {
            "config_hash": comparator._config_hash(params),
            "simulator_settings": {"horizon": 600, "dt": 0.1},
        },
        "algorithm_metadata": {
            "algorithm": policy["algo"],
            "config": effective_policy_config,
            "config_hash": comparator._config_hash(effective_policy_config),
        },
        "provenance": {
            "commit_hash": controls["source_sha"],
            "config_identity": {
                "algo": policy["algo"],
                "algo_config_path": arm_control["planner_config_path"],
                "scenario_matrix_hash": controls["scoped_hashes"][
                    ("fixture_arm", "differential_drive")
                ],
                "campaign_config_hash": controls["campaign_config_hash"],
            },
        },
        "scenario_params": params,
    }
    assert (
        comparator._candidate_run_control_issues(
            row,
            arm="fixture_arm",
            expected_algo=policy["algo"],
            expected=controls,
        )
        == []
    )
    mutate(row)
    issues = comparator._candidate_run_control_issues(
        row,
        arm="fixture_arm",
        expected_algo=policy["algo"],
        expected=controls,
    )
    assert any(reason in issue for issue in issues)


def test_stage3_scaffold_binds_source_inputs_and_raw_rows_but_is_not_admissible(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    subprocess.run(["git", "-C", str(source), "init", "-q"], check=True, capture_output=True)
    config_sha = _write_yaml(source, "configs/campaign.yaml", {"scenario_matrix": "matrix.yaml"})
    matrix_sha = _write_yaml(
        source,
        "configs/matrix.yaml",
        {"scenarios": [{"name": item} for item in sorted(comparator.EXPECTED_SCENARIO_IDS)]},
    )
    source_sha = _git_commit(source)
    campaign_root = tmp_path / "campaign"
    episode_hashes = {}
    for index in range(14):
        path = campaign_root / f"runs/arm_{index}__differential_drive/episodes.jsonl"
        path.parent.mkdir(parents=True)
        path.write_text(json.dumps({"arm": index}) + "\n", encoding="utf-8")
        episode_hashes[path.relative_to(campaign_root).as_posix()] = _sha(path)
    scientific_identity = {
        "source_sha": source_sha,
        "campaign_template_path": "configs/campaign.yaml",
        "campaign_template_sha256": config_sha,
        "scenario_ids": sorted(comparator.EXPECTED_SCENARIO_IDS),
        "scientific_manifest": {
            "scenario": {"matrix_path": "configs/matrix.yaml", "matrix_sha256": matrix_sha},
            "seed_policy": {"resolved_seeds": list(comparator.EXPECTED_SEEDS)},
        },
    }

    scaffold = comparator.write_candidate_input_scaffolds(
        campaign_root,
        source_root=source,
        scientific_identity=scientific_identity,
    )

    identity_path = campaign_root / scaffold["candidate_identity_path"]
    ledger_path = campaign_root / scaffold["attribution_ledger_path"]
    identity = json.loads(identity_path.read_text(encoding="utf-8"))
    ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
    assert identity["source_sha"] == source_sha
    assert identity["effective_config_sha256"] == config_sha
    assert identity["scenario_matrix"] == {
        "path": "configs/matrix.yaml",
        "sha256": matrix_sha,
    }
    assert identity["episode_files"] == episode_hashes
    assert identity["scaffold_status"] == "requires_author_review"
    assert identity["arm_slots"] == []
    assert identity["versioned_changes"] == []
    assert ledger == {
        "schema_version": comparator.ATTRIBUTION_SCHEMA,
        "candidate_source_sha": source_sha,
        "entries": [],
    }
    assert scaffold["candidate_identity_sha256"] == _sha(identity_path)
    assert scaffold["attribution_ledger_sha256"] == _sha(ledger_path)
    with pytest.raises(ValueError, match="requires author/domain review"):
        comparator.load_candidate_identity(identity_path, source)


def test_hashed_v4_config_derives_per_scenario_runtime_and_rejects_row_switch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(comparator, "_candidate_run_controls", lambda _identity, _root: {})
    source, identity_path, identity, target_old = _versioned_v4_fixture(tmp_path)
    checked = comparator.load_candidate_identity(identity_path, source)
    target = next(slot for slot in identity["arm_slots"] if slot["old_key"] == target_old)
    new_key = target["new_key"]
    assert (
        checked["_effective_algorithms"][new_key]["classic_bottleneck_low"]
        == "hybrid_rule_local_planner"
    )
    assert checked["_effective_algorithms"][new_key]["francis2023_leave_group"] == "orca"
    run = tmp_path / "candidate" / "runs" / f"{new_key}__differential_drive"
    run.mkdir(parents=True)
    row = _row("orca", success=True, collision=False, score=1.0)
    row.update(
        {
            "scenario_id": "classic_bottleneck_low",
            "seed": 111,
            "algo": "orca",
            "git_hash": identity["source_sha"],
        }
    )
    episode_file = run / "episodes.jsonl"
    episode_file.write_text(json.dumps(row) + "\n")
    checked["episode_files"] = {
        f"runs/{new_key}__differential_drive/episodes.jsonl": _sha(episode_file)
    }
    _, anomalies = read_candidate_rows(tmp_path / "candidate", checked)
    assert any(item["kind"] == "arm_mismatch" for item in anomalies)


def test_joint_hybrid_config_and_row_algo_edits_cannot_make_all_orca_valid(tmp_path: Path) -> None:
    source, identity_path, identity, target_old = _versioned_v4_fixture(tmp_path)
    target = next(slot for slot in identity["arm_slots"] if slot["old_key"] == target_old)
    path = source / target["config_path"]
    config = yaml.safe_load(path.read_text())
    config["scenario_algo_overrides"] = {
        scenario: {"algo": "orca", "base_config_path": comparator.APPROVED_ORCA_CONFIG_PATH}
        for scenario in identity["scenario_ids"]
    }
    target["config_sha256"] = _write_yaml(source, target["config_path"], config)
    target["row_algos"] = ["orca"]
    identity["source_sha"] = _git_commit(source)
    identity["versioned_changes"][0]["new_identity"] = identity["source_sha"]
    identity_path.write_text(json.dumps(identity))
    with pytest.raises(ValueError, match="unapproved scenario algorithm override"):
        comparator.load_candidate_identity(identity_path, source)


def test_v4_key_cannot_bind_v3_base_even_with_updated_hashes(tmp_path: Path) -> None:
    source, identity_path, identity, target_old = _versioned_v4_fixture(tmp_path)
    target = next(slot for slot in identity["arm_slots"] if slot["old_key"] == target_old)
    path = source / target["config_path"]
    config = yaml.safe_load(path.read_text())
    config["base_config_path"] = "configs/algos/hybrid_rule_v3_fixture.yaml"
    target["config_sha256"] = _write_yaml(source, target["config_path"], config)
    target["base_config_sha256"] = _sha(source / config["base_config_path"])
    identity["source_sha"] = _git_commit(source)
    identity["versioned_changes"][0]["new_identity"] = identity["source_sha"]
    identity_path.write_text(json.dumps(identity))
    with pytest.raises(ValueError, match="approved v4 base"):
        comparator.load_candidate_identity(identity_path, source)


def test_v4_key_cannot_override_effective_variant_for_one_scenario(tmp_path: Path) -> None:
    source, identity_path, identity, target_old = _versioned_v4_fixture(tmp_path)
    target = next(slot for slot in identity["arm_slots"] if slot["old_key"] == target_old)
    path = source / target["config_path"]
    config = yaml.safe_load(path.read_text())
    config["scenario_overrides"] = {"classic_bottleneck_low": {"planner_variant": "hybrid_rule_v3"}}
    target["config_sha256"] = _write_yaml(source, target["config_path"], config)
    identity["source_sha"] = _git_commit(source)
    identity["versioned_changes"][0]["new_identity"] = identity["source_sha"]
    identity_path.write_text(json.dumps(identity))
    with pytest.raises(ValueError, match="effective planner variant is not v4"):
        comparator.load_candidate_identity(identity_path, source)
