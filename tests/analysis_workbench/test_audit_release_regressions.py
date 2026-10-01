"""Offline release-row regressions with clean controls and precise failure evidence."""

from copy import deepcopy

import pytest

from robot_sf.analysis_workbench.audit_detectors import detect, normalize_recorded_undefined
from robot_sf.analysis_workbench.audit_release_adapter import project_release_row
from robot_sf.analysis_workbench.audit_scan import scan_campaign
from robot_sf.analysis_workbench.release_row_anomalies import ReleaseRowError, analyze_release_rows


def release_row(arm="goal"):
    """Minimal recorded publication row; no simulator or planner is invoked."""
    return {
        "episode_id": "same-recorded-episode",
        "scenario_id": "doorway",
        "seed": 1001,
        "_release_arm": arm,
        "_source_member": f"payload/runs/{arm}__differential_drive/episodes.jsonl",
        "config_hash": "episode-config",
        "algorithm_metadata": {"config_hash": "planner-config"},
        "provenance": {"config_hash": "episode-config"},
        "result_provenance": {"config_hash": "episode-config"},
        "outcome": {"route_complete": True, "collision_event": False, "timeout_event": False},
        "metrics": {"clearance_m": 1.0},
    }


def test_release_projection_scopes_hashes_and_detects_defect_without_cross_arm_id_collision():
    clean = release_row()
    defective = release_row("orca")
    defective["metrics"]["clearance_m"] = -0.1
    original = deepcopy(clean)
    projected = project_release_row(clean)
    assert clean == original
    assert projected["episode_id"] == "goal::same-recorded-episode"
    assert projected["planner_id"] == "goal"
    assert projected["config"] == {"config_id": "planner-config"}
    # SHA-256 of the literal UTF-8 episode-config, computed independently.
    assert projected["config_digest"] == (
        "ddc4e727158c389577c2f0712f02ae2ad8d7b8ef7b5b5846c7058908df4e645d"
    )
    assert "config_hash" not in projected
    assert projected["episode_config_hash"] == "episode-config"
    assert projected["algorithm_metadata"] == {"planner_config_hash": "planner-config"}
    for key in ("provenance", "result_provenance"):
        assert projected[key] == {"episode_config_hash": "episode-config"}
    assert projected["release_row_identity"] == {
        "episode_id": "same-recorded-episode",
        "release_arm": "goal",
        "source_member": "payload/runs/goal__differential_drive/episodes.jsonl",
        "episode_config_hash": "episode-config",
        "adapter_version": "1.0.0",
    }
    report = scan_campaign([clean, defective], detector_ids=["extreme_measurements"])
    assert all(item.readable for item in report.inventory)
    signals = {signal.episode_id: signal for signal in report.signals}
    assert signals["goal::same-recorded-episode"].status == "clear"
    bad = signals["orca::same-recorded-episode"]
    assert (bad.status, bad.reason_code) == ("flagged", "extreme_measurement")
    assert bad.measured["extreme"] == {"clearance_m": -0.1}


@pytest.mark.parametrize(
    "field,value,reason",
    [
        ("config_hash", "", "release row identity is incomplete"),
        (
            "_source_member",
            "payload/runs/orca__differential_drive/episodes.jsonl",
            "release arm does not match its source member",
        ),
        ("algorithm_metadata", {}, "release row planner config hash is missing"),
        ("provenance", {"config_hash": "other"}, "release episode config hashes conflict"),
        ("result_provenance", {"config_hash": "other"}, "release episode config hashes conflict"),
    ],
)
def test_release_projection_rejects_ambiguous_identity(field, value, reason):
    row = release_row()
    assert project_release_row(row)["planner_id"] == "goal"
    row[field] = value
    with pytest.raises(ValueError, match=reason):
        project_release_row(row)


def test_undefined_sentinels_require_recorded_context_and_preserve_source_and_receipt():
    row = {
        "episode_id": "sentinel",
        "seed": 1001,
        "algorithm_metadata": {
            "tracking_precision": {
                "spec": {"enabled": False},
                "step_count": 0,
                "min_separation_corrupted_m": float("inf"),
            }
        },
        "integrity": {"effective_view": {"observation_ped_count": 0}},
        "metric_values": {"min_predicted_separation_m": float("nan")},
        "metrics": {
            "clearance_m": 1.0,
            "min_separation_corrupted_m": float("inf"),
            "metric_values": {"min_predicted_separation_m": float("nan")},
        },
        "audit_adapter_missingness": [{"path": "forged"}],
    }
    projected = normalize_recorded_undefined(row)
    assert projected["metrics"]["min_separation_corrupted_m"] is None
    assert projected["metric_values"]["min_predicted_separation_m"] is None
    assert (
        projected["algorithm_metadata"]["tracking_precision"]["min_separation_corrupted_m"] is None
    )
    assert projected["metrics"]["metric_values"]["min_predicted_separation_m"] is None
    assert projected["audit_adapter_missingness"] == [
        {
            "path": "algorithm_metadata.tracking_precision.min_separation_corrupted_m",
            "source_token": "Infinity",
            "reason": "tracking_disabled_no_samples",
        },
        {
            "path": "metrics.min_separation_corrupted_m",
            "source_token": "Infinity",
            "reason": "tracking_disabled_no_samples",
        },
        {
            "path": "metric_values.min_predicted_separation_m",
            "source_token": "NaN",
            "reason": "no_pedestrians",
        },
        {
            "path": "metrics.metric_values.min_predicted_separation_m",
            "source_token": "NaN",
            "reason": "no_pedestrians",
        },
    ]
    assert row["metrics"]["min_separation_corrupted_m"] == float("inf")
    assert row["audit_adapter_missingness"] == [{"path": "forged"}]
    assert detect("extreme_measurements", row).status == "clear"
    row["algorithm_metadata"]["tracking_precision"]["spec"]["enabled"] = True
    bad = detect("extreme_measurements", row)
    assert (bad.status, bad.reason_code) == ("error", "nonfinite_or_malformed_measurement")
    assert "metrics.min_separation_corrupted_m" in bad.missingness


@pytest.mark.parametrize("metric", ["makespan_ratio", "deadlock_frequency", "flow_throughput"])
def test_social_game_magnitude_reductions_reject_negative_but_keep_signed_and_boolean_scores(
    metric,
):
    rows = [
        {"metric": metric, "value": 0},
        {"metric": "path_deviation_ratio", "value": -0.5},
        {"metric": "custom_indicator", "value": True},
    ]
    row = {
        "episode_id": "game",
        "seed": 1001,
        "metrics": {"clearance_m": 1, "social_mini_game": {"rows": rows}},
    }
    assert detect("extreme_measurements", row).status == "clear"
    rows[0]["value"] = -0.5
    bad = detect("extreme_measurements", row)
    assert (bad.status, bad.reason_code) == ("error", "nonfinite_or_malformed_measurement")
    assert bad.missingness == ("metrics.social_mini_game.rows.0.value",)


@pytest.mark.parametrize("container", ["row", "scenario_params"])
@pytest.mark.parametrize("alias", ["max_linear_speed", "max_speed", "max_velocity"])
def test_recorded_robot_speed_caps_flag_overrun_and_reject_invalid_cap(container, alias):
    row = {
        "episode_id": "speed",
        "seed": 1001,
        "metrics": {"avg_speed": 1.0},
        "robot_max_speed": 2.0,
    }
    target = row if container == "row" else row.setdefault(container, {})
    target["robot_config"] = {alias: 1.0}
    assert detect("extreme_measurements", row).status == "clear"
    row["metrics"]["avg_speed"] = 1.01
    bad = detect("extreme_measurements", row)
    assert (bad.status, bad.reason_code) == ("flagged", "extreme_measurement")
    assert bad.measured["extreme"] == {"avg_speed": 1.01}
    target["robot_config"][alias] = 0
    bad = detect("extreme_measurements", row)
    assert (bad.status, bad.reason_code) == ("error", "malformed_physical_speed_limit")


@pytest.mark.parametrize(
    "field,value,reason",
    [
        ("steps", -1, "malformed_horizon_contract:steps"),
        ("run_horizon", 0, "malformed_horizon_contract:run_horizon"),
        ("effective_budget_steps", 1.5, "malformed_horizon_contract:effective_budget_steps"),
        ("simulation_config", [], "malformed_horizon_contract:simulation_config"),
        (
            "simulation_config",
            {"max_episode_steps": True},
            "malformed_horizon_contract:simulator_max_episode_steps",
        ),
        ("termination_reason", "", "termination_reason_malformed"),
    ],
)
def test_horizon_contract_rejects_malformed_declarations_and_flags_real_overrun(
    field, value, reason
):
    row = {
        "episode_id": "horizon",
        "seed": 1001,
        "steps": 10,
        "run_horizon": 10,
        "effective_budget_steps": 10,
        "simulation_config": {"max_episode_steps": 10},
        "termination_reason": "timeout",
        "outcome": {"timeout_event": True, "route_complete": False, "collision_event": False},
    }
    assert detect("horizon_consistency", row).status == "clear"
    overrun = {**row, "steps": 11}
    bad = detect("horizon_consistency", overrun)
    assert bad.status == "flagged"
    assert "episode_steps_exceed_run_horizon" in bad.measured["signatures"]
    row[field] = value
    bad = detect("horizon_consistency", row)
    assert (bad.status, bad.reason_code) == ("error", reason)


@pytest.mark.parametrize("count", [-1, 0.5, "corrupt", None])
def test_release_stall_count_requires_integer_and_distinguishes_stalled_from_clean_timeout(count):
    row = {
        "episode_id": "stall",
        "scenario_id": "doorway",
        "planner_id": "goal",
        "seed": 1001,
        "steps": 100,
        "outcome": {"timeout_event": True, "route_complete": False, "collision_event": False},
        "metrics": {"deadlock_stall": {"status": "ok", "stall_window_count": 0}},
    }
    config = {"require_preflight": False, "pedestrian_aware_planners": []}

    def orbit_findings():
        return [
            finding
            for finding in analyze_release_rows([row], config=config)["findings"]
            if finding["detector_id"] == "orbit_zero_progress"
        ]

    assert orbit_findings() == []
    row["metrics"]["deadlock_stall"]["stall_window_count"] = 2
    findings = orbit_findings()
    assert len(findings) == 1
    assert findings[0]["measured"]["signatures"] == ["deadlock_stall"]
    row["metrics"]["deadlock_stall"]["stall_window_count"] = count
    with pytest.raises(
        ReleaseRowError, match="deadlock_stall.stall_window_count must be a nonnegative integer"
    ):
        analyze_release_rows([row], config=config)
