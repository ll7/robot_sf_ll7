"""Offline regressions from SHA-pinned published 0.0.7 rows; never run episodes."""

import hashlib
import json
import math
from pathlib import Path

import pytest

from robot_sf.analysis_workbench.audit_detectors import detect

FIXTURE = Path(__file__).parents[1] / "fixtures/analysis_workbench/aud_release_0_0_7.json"


def release_row(kind):
    samples = json.loads(FIXTURE.read_text())["samples"]
    sample = next(item for item in samples if item["kind"] == kind)
    assert hashlib.sha256(sample["raw_line"].encode()).hexdigest() == sample["line_sha256"]
    return json.loads(sample["raw_line"])


def test_published_positive_ttc_uses_seconds_bound():
    recorded = release_row("ttc_positive")
    ttc = recorded["metrics"]["time_to_collision_min"]
    assert 1 < ttc < 17  # Published success with positive TTC, not a collision count.
    row = {"episode_id": recorded["episode_id"], "metrics": {"time_to_collision_min": ttc}}
    signal = detect("extreme_measurements", row)
    assert signal.status == "clear", signal.measured
    row["metrics"]["time_to_collision_min"] = -0.1
    assert detect("extreme_measurements", row).status == "flagged"
    row["metrics"] = {"total_collision_count": 2}
    assert detect("extreme_measurements", row).status == "flagged"


def test_orbit_uses_recorded_stall_windows_not_failure_boolean():
    from robot_sf.analysis_workbench.release_row_anomalies import analyze_release_rows

    row = release_row("timeout")
    assert row["metrics"]["deadlock"] is True
    # Counterfactual on a real timeout: isolate the stall channel from geometry.
    row["metrics"]["deadlock_stall"]["stall_window_count"] = 0
    config = {
        "min_orbit_curvature": 1e9,
        "min_orbit_path_length_m": 1e9,
        "require_preflight": False,
        "pedestrian_aware_planners": [],
    }
    report = analyze_release_rows([row], config=config)
    assert not [f for f in report["findings"] if f["detector_id"] == "orbit_zero_progress"]
    row["metrics"]["deadlock_stall"]["stall_window_count"] = 1
    row["metrics"]["deadlock"] = False
    report = analyze_release_rows([row], config=config)
    orbit = [f for f in report["findings"] if f["detector_id"] == "orbit_zero_progress"]
    assert len(orbit) == 1
    assert orbit[0]["measured"]["signatures"] == ["deadlock_stall"]
    assert orbit[0]["detector_version"] == "1.1.0"
    completed = release_row("success_stall")
    assert completed["outcome"]["route_complete"] is True
    assert completed["metrics"]["deadlock_stall"]["stall_window_count"] > 0
    report = analyze_release_rows([completed], config=config)
    assert not [f for f in report["findings"] if f["detector_id"] == "orbit_zero_progress"]


def cohort_row(episode, value, outcome="success"):
    return {
        "episode_id": episode,
        "planner_id": "goal",
        "scenario_id": "doorway",
        "config_id": "c",
        "seed": 1001,
        "outcome": {"label": outcome},
        "metrics": {"time_to_goal_norm": value},
    }


@pytest.mark.parametrize("detector", ["seed_outlier", "cohort_multivariate_outlier"])
def test_outlier_mad_floor_rejects_numerical_noise(detector):
    peers = [cohort_row(str(i), 0.0) for i in range(4)]
    target = cohort_row("target", 1e-6)
    signal = detect(detector, target, cohort=[*peers, target])
    assert signal.status == "clear", signal.measured
    target["metrics"]["time_to_goal_norm"] = 1.0
    assert detect(detector, target, cohort=[*peers, target]).status == "flagged"


@pytest.mark.parametrize("detector", ["seed_outlier", "cohort_multivariate_outlier"])
def test_outlier_cohorts_match_terminal_outcome(detector):
    target = cohort_row("target", 0.2)
    successes = [cohort_row(str(i), 0.2) for i in range(3)]
    timeouts = [cohort_row("timeout-" + str(i), 1.0, "timeout") for i in range(12)]
    signal = detect(detector, target, cohort=[target, *successes, *timeouts])
    assert signal.status == "clear", signal.measured
    assert signal.measured["cohort_size"] == 4


def test_release_failure_timeout_is_admitted_but_execution_failure_is_not():
    from robot_sf.analysis_workbench.audit_scan import scan_campaign

    recorded = release_row("timeout")
    row = {
        "episode_id": recorded["episode_id"],
        "status": recorded["status"],
        "outcome": recorded["outcome"],
        "metrics": {"clearance_m": 1.0},
    }
    assert detect("extreme_measurements", row).status == "clear"
    assert scan_campaign([row], detector_ids=["extreme_measurements"]).inventory[0].readable
    row["execution_status"] = "failure"
    assert detect("extreme_measurements", row).status == "unavailable"
    assert (
        scan_campaign([row], detector_ids=["extreme_measurements"]).inventory[0].status
        == "unsupported"
    )
    row.pop("execution_status")
    row["algorithm_metadata"] = {"status": "failure"}
    assert detect("extreme_measurements", row).status == "unavailable"


def test_disabled_tracking_infinity_is_missing_not_row_corruption():
    from robot_sf.analysis_workbench.audit_scan import scan_campaign

    recorded = release_row("success")
    row = {
        "episode_id": recorded["episode_id"],
        "outcome": recorded["outcome"],
        "integrity": recorded["integrity"],
        "metric_values": recorded["metric_values"],
        "algorithm_metadata": {
            "tracking_precision": recorded["algorithm_metadata"]["tracking_precision"]
        },
        "metrics": {
            "clearance_m": 1.0,
            "min_separation_corrupted_m": math.inf,
            "metric_values": recorded["metrics"]["metric_values"],
        },
    }
    report = scan_campaign([row], detector_ids=["extreme_measurements"])
    assert report.inventory[0].readable, report.inventory[0].reason
    retained = report.inventory[0].row
    assert retained["metrics"]["min_separation_corrupted_m"] is None
    assert (
        retained["algorithm_metadata"]["tracking_precision"]["min_separation_corrupted_m"] is None
    )
    assert retained["metrics"]["metric_values"]["min_predicted_separation_m"] is None
    assert len(retained["audit_adapter_missingness"]) == 4
    assert math.isinf(row["metrics"]["min_separation_corrupted_m"])  # Immutable source.
    assert detect("extreme_measurements", row).status == "clear"
    row["metrics"]["unexpected_corruption"] = math.inf
    assert (
        scan_campaign([row], detector_ids=["extreme_measurements"]).inventory[0].status == "invalid"
    )
    assert detect("extreme_measurements", row).status == "error"


def test_verified_release_arm_keeps_config_scopes_and_episode_identity():
    from robot_sf.analysis_workbench.audit_scan import scan_campaign

    row = release_row("success")
    row["_release_arm"] = "goal"
    row["_source_member"] = "payload/runs/goal__differential_drive/episodes.jsonl"
    report = scan_campaign([row], detector_ids=["extreme_measurements"])
    assert report.inventory[0].readable, report.inventory[0].reason
    retained = report.inventory[0].row
    assert retained["episode_id"] == "goal::" + row["episode_id"]
    assert retained["config"]["config_id"] == row["algorithm_metadata"]["config_hash"]
    assert (
        retained["algorithm_metadata"]["planner_config_hash"]
        == row["algorithm_metadata"]["config_hash"]
    )
    assert retained["release_row_identity"]["episode_id"] == row["episode_id"]
    assert report.signals[0].status == "clear"


def test_scan_shards_cohorts_without_changing_signals(monkeypatch):
    from robot_sf.analysis_workbench import audit_scan
    from robot_sf.analysis_workbench.audit_contracts import record_to_dict

    rows = []
    for scenario in range(8):
        for i in range(4):
            row = cohort_row(f"{scenario}-{i}", i / 10)
            row["scenario_id"] = str(scenario)
            rows.append(row)
    seen_sizes = []
    original = audit_scan.detect

    def observe(*args, **kwargs):
        seen_sizes.append(len(kwargs["cohort"]))
        return original(*args, **kwargs)

    monkeypatch.setattr(audit_scan, "detect", observe)
    report = audit_scan.scan_campaign(rows, detector_ids=["seed_outlier"])
    assert max(seen_sizes) == 4  # A linear partition, not every row in every call.
    admitted = [item.row for item in report.inventory]
    brute = [original("seed_outlier", row, cohort=admitted) for row in admitted]
    assert [record_to_dict(s) for s in report.signals] == [record_to_dict(s) for s in brute]


def test_scan_accepts_20160_rows_above_old_source_cap(tmp_path):
    from robot_sf.analysis_workbench.audit_scan import scan_campaign

    source = tmp_path / "episodes.jsonl"
    with source.open("w") as stream:
        for i in range(20160):
            stream.write(
                json.dumps(
                    {
                        "episode_id": str(i),
                        "metrics": {"clearance_m": 1.0},
                        "retained_note": "x" * 3350,
                    }
                )
                + "\n"
            )
    assert source.stat().st_size > 64 * 1024**2
    report = scan_campaign(source, root=tmp_path, detector_ids=["extreme_measurements"])
    assert report.counts["coverage"]["readable"] == 20160
    assert len(report.signals) == 20160


def queue_candidate(execution, planner="goal", seed=1001, reason="common_mode_failure"):
    from robot_sf.analysis_workbench.audit_contracts import EpisodeRef, Signal
    from robot_sf.analysis_workbench.audit_queue import QueueCandidate

    episode = EpisodeRef(
        campaign_digest="a" * 64,
        source_digest="b" * 64,
        execution_id=execution,
        planner_id=planner,
        scenario_id="doorway",
        seed=seed,
        config_digest="c" * 64,
    )
    candidate = QueueCandidate(episode)
    signal = Signal(
        signal_id="signal-" + execution,
        detector_id="common_mode_anomaly",
        detector_version="1.0.0",
        episode_id=candidate.episode_id,
        status="flagged",
        reason_code=reason,
    )
    return QueueCandidate(episode, signals=(signal,))


def test_common_mode_failure_has_benchmark_defect_priority():
    from robot_sf.analysis_workbench.audit_queue import PRIORITY_BENCHMARK_CONFIG, AuditQueue

    ranked = AuditQueue([queue_candidate("common")]).rank_candidates()
    assert ranked[0].priority_band == PRIORITY_BENCHMARK_CONFIG


def test_top_100_represent_distinct_cells_and_retain_all_episodes():
    from robot_sf.analysis_workbench.audit_queue import AuditQueue

    candidates = [
        queue_candidate(f"{seed}-{planner}", planner, seed)
        for seed in range(1001, 1111)
        for planner in ("goal", "orca")
    ]
    ranked = AuditQueue(candidates).rank_candidates()
    assert len({(r.candidate.scenario_id, r.candidate.episode.seed) for r in ranked[:100]}) == 100
    assert {r.candidate.episode_id for r in ranked} == {c.episode_id for c in candidates}


def test_queue_loads_20160_candidates_above_old_byte_and_node_caps(tmp_path):
    from robot_sf.analysis_workbench.audit_queue import load_queue_input

    candidate = queue_candidate("template").to_dict()
    candidate["signals"] = []
    candidate["episode"].pop("episode_id")
    candidate["metadata"] = {"retained_note": "x" * 500}
    candidates = []
    for i in range(20160):
        candidates.append(
            {**candidate, "episode": {**candidate["episode"], "execution_id": str(i)}}
        )
    source = tmp_path / "queue.json"
    source.write_text(
        json.dumps({"schema_version": "audit-queue-input.v1", "candidates": candidates})
    )
    assert source.stat().st_size > 8 * 1024**2
    loaded = load_queue_input(source)
    assert len(loaded.candidates) == 20160


def test_release_seed_cohorts_use_planner_config_not_seeded_episode_hash():
    from robot_sf.analysis_workbench.audit_scan import scan_campaign

    rows = []
    for i in range(4):
        row = release_row("success")
        row["_release_arm"] = "goal"
        row["_source_member"] = "payload/runs/goal__differential_drive/episodes.jsonl"
        row["episode_id"] = "dev-control-" + str(i)
        row["seed"] = 1001 + i
        row["config_hash"] = "seeded-config-" + str(i)
        row["provenance"]["config_hash"] = row["config_hash"]
        row["result_provenance"]["config_hash"] = row["config_hash"]
        rows.append(row)
    report = scan_campaign(rows, detector_ids=["seed_outlier"])
    assert all(signal.status == "clear" for signal in report.signals)
    assert all(signal.measured["cohort_size"] == 4 for signal in report.signals)
