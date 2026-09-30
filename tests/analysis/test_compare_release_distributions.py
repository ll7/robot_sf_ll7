"""Read-only release distributions: independent oracles, no environment steps."""

import copy
import json
from io import BytesIO

import numpy as np
import pytest

from robot_sf.benchmark.metric_definitions import CHANGED_METRICS
from scripts.analysis import compare_release_distributions as cmp
from scripts.analysis.compare_release_0_0_7_to_0_0_8 import _insert_rows


def rows(*, successes=15, release="0.0.7", arm="goal", seed_offset=1001, metrics=None):
    result = {}
    for index in range(30):
        result[arm, "differential_drive", "scenario_a", seed_offset + index, ""] = {
            "outcome": {
                "route_complete": index < successes,
                "collision_event": index >= successes,
                "timeout_event": False,
            },
            "termination_reason": "success" if index < successes else "collision",
            "metrics": {
                "avg_speed": 1.0,
                "jerk_mean": 0.5,
                "path_efficiency": 0.75,
                **(metrics or {}),
            },
            "metric_schema_version": "robot-sf-metrics.v1"
            if release == "0.0.7"
            else "robot-sf-metrics.v2",
            "_definition": {
                "scenario": {
                    "map_file": "old.svg" if release == "0.0.7" else "new.svg",
                    "run_horizon": 600,
                    "simulation_config": {"max_episode_steps": 500},
                },
                "algo_config_hash": "abc",
                "steps": 500,
            },
        }
    return result


def cell(report, metric, level="unit"):
    return next(c for c in report["cells"] if c["metric"] == metric and c["level"] == level)


def test_planted_shift_and_null_after_bh():
    old = rows(successes=0)
    # Disjoint row seed identifiers; no episodes are executed.
    new = rows(successes=30, release="0.0.8", seed_offset=2001)
    report = cmp.compare_samples(old, new)
    assert cell(report, "success")["difference"] == 1.0
    assert cell(report, "success")["changed"]
    assert cell(report, "success", "arm")["primary"]
    assert cell(report, "success", "arm")["changed"]
    assert cell(report, "success")["interpretation"] == "combined release difference"
    null = cmp.compare_samples(rows(), rows(release="0.0.8", seed_offset=2001))
    assert not any(c["changed"] for c in null["cells"])
    assert cell(null, "success")["q_value"] == 1.0


def test_wilson_newcombe_known_values():
    # Hand/reference score interval for 0/10; z^2/(10+z^2) = .2775327998628892.
    assert cmp.wilson(0, 10) == pytest.approx([0, 0.2775327998628892])
    assert cmp.wilson(5, 10) == pytest.approx([0.236593090512564, 0.763406909487436])
    assert cmp.newcombe(0, 10, 10, 10) == pytest.approx([0.607509350430, 1.0], abs=1e-10)
    assert cmp.newcombe(5, 10, 5, 10) == pytest.approx([-0.372513, 0.372513], abs=1e-6)


def test_definition_changed_never_differenced_and_success_support():
    metrics = {
        name: 2.0 for name in CHANGED_METRICS if name not in ("snqi_v2_terms", "deadlock_stall")
    }
    metrics.update(
        curvature_mean=2.0, snqi_v2_terms={"F": 3.0}, deadlock_stall={"stall_window_count": 4}
    )
    old_metrics = {
        key: value
        for key, value in metrics.items()
        if key != "path_efficiency_reference_violation" and not key.startswith("snqi_v2")
    }
    report = cmp.compare_samples(rows(metrics=old_metrics), rows(release="0.0.8", metrics=metrics))
    for c in report["cells"]:
        if c["definition_status"] != "same":
            assert c["difference"] is None and c["difference_ci"] is None
            assert c["p_value"] is None and not c["changed"]
    jerk = cell(report, "jerk_mean")
    assert jerk["name_0_0_7"] == "jerk_mean@v1"
    assert jerk["name_0_0_8"] == "jerk_mean@v2"
    assert cell(report, "snqi_v2")["release_0_0_7"]["estimate"] is None
    insufficient = cmp.compare_samples(rows(successes=4), rows(release="0.0.8"))
    assert not cell(insufficient, "path_efficiency")["release_0_0_7"]["sufficient"]
    assert cell(insufficient, "path_efficiency")["release_0_0_7"]["estimate"] is None


def test_null_nan_empty_jerk_and_episode_source():
    old, new = rows(), rows(release="0.0.8")
    values = list(old.values())
    values[0]["metrics"]["avg_speed"] = None
    values[1]["metrics"]["avg_speed"] = float("nan")
    values[2]["metrics"]["jerk_mean"] = ""  # #10044 empty cells are missing, never zero
    report = cmp.compare_samples(old, new)
    speed = cell(report, "avg_speed")["release_0_0_7"]
    assert speed["n_defined"] == 28 and speed["n_missing"] == 2 and speed["estimate"] == 1.0
    jerk = cell(report, "jerk_mean")["release_0_0_7"]
    assert jerk["n_defined"] == 29 and jerk["n_missing"] == 1 and jerk["estimate"] == 0.5


def test_schema_mixture_and_reversed_versions_refused():
    old = rows()
    next(iter(old.values()))["metric_schema_version"] = "robot-sf-metrics.v2"
    with pytest.raises(ValueError, match="incompatible metric definitions"):
        cmp.compare_samples(old, rows(release="0.0.8"))
    with pytest.raises(ValueError, match="release schemas must be v1"):
        cmp.compare_samples(rows(release="0.0.8"), rows())


def test_replaced_arms_and_registry():
    old = rows(arm="hybrid_rule_v3_fast_progress_static_escape")
    new = rows(release="0.0.8", arm="hybrid_rule_v4_fast_progress_static_escape")
    report = cmp.compare_samples(old, new)
    assert all(c["arm_replaced"] for c in report["cells"])
    assert len(cmp.ARM_SLOTS_0_0_7_TO_0_0_8) == 14
    assert sum(s.key_0_0_7 != s.key_0_0_8 for s in cmp.ARM_SLOTS_0_0_7_TO_0_0_8) == 4
    assert not set(cmp.registry()["allowlist"]) & CHANGED_METRICS
    assert "curvature_mean" not in cmp.registry()["allowlist"]


def test_bootstrap_and_outputs_byte_reproducible(tmp_path):
    old, new = rows(), rows(release="0.0.8")
    for index, row in enumerate(old.values()):
        row["metrics"]["avg_speed"] = index / 10
    r1 = cmp.compare_samples(old, new)
    r2 = cmp.compare_samples(old, new)
    cmp.write_report(r1, tmp_path / "a")
    cmp.write_report(r2, tmp_path / "b")
    for name in ("distribution.json", "distribution.csv", "distribution.md"):
        assert (tmp_path / "a" / name).read_bytes() == (tmp_path / "b" / name).read_bytes()
    assert cell(r1, "avg_speed")["difference"] == pytest.approx(-0.45)
    assert (
        cell(r1, "avg_speed")["difference_ci"][0]
        < -0.45
        < cell(r1, "avg_speed")["difference_ci"][1]
    )


def test_two_stage_bootstrap_retains_between_scenario_variation():
    groups = [np.zeros(30), np.ones(30)]
    draws = cmp._bootstrap(groups, np.random.default_rng(20260930), 10000, pooled=True)
    assert np.mean(draws) == pytest.approx(0.5, abs=0.02)
    # Pooling episodes directly would wrongly shrink scenario-level uncertainty.
    assert cmp._ci(draws) == [0.0, 1.0]


def test_bh_rejects_nominal_false_positive():
    cells = [
        {"p_value": p, "difference_ci": [0.01, 0.1], "changed": False}
        for p in [0.04, 0.2, 0.5, 0.9]
    ]
    cmp.benjamini_hochberg(cells)
    assert cells[0]["q_value"] == pytest.approx(0.16)
    assert not any(c["changed"] for c in cells)


def test_authored_limit_timeout_mapping():
    row = next(iter(rows(successes=0).values()))
    row["outcome"]["collision_event"] = False
    row["termination_reason"] = "terminated"
    assert cmp.outcomes(row, "0.0.7")["timeout"]
    assert not cmp.outcomes(row, "0.0.8")["timeout"]
    row["_definition"]["steps"] = 499
    assert not cmp.outcomes(row, "0.0.7")["timeout"]
    row["termination_reason"] = "max_steps"
    assert cmp.outcomes(row, "0.0.7")["timeout"]
    assert cmp.outcomes(row, "0.0.8")["timeout"]


def test_stream_reader_discards_traces_before_retaining():
    raw = next(iter(rows(release="0.0.8").values()))
    raw.update(scenario_id="scenario_a", seed=1001, git_hash="a" * 40)
    raw["algorithm_metadata"] = {
        "simulation_step_trace": {
            "schema_version": "simulation-step-trace.v2",
            "steps": list(range(1000)),
        },
        "planner_decision_trace": [1] * 1000,
    }
    raw["metrics"].update(
        robot_force_samples=[1] * 1000, metric_schema_version="robot-sf-metrics.v2"
    )
    compact = {}
    _insert_rows(
        compact,
        BytesIO((json.dumps(raw) + "\n").encode()),
        "goal__differential_drive",
        "synthetic",
        retain_provenance=True,
    )
    retained = next(iter(compact.values()))
    assert "robot_force_samples" not in retained["metrics"]
    assert "simulation_step_trace" not in retained["_provenance"]["algorithm_metadata"]
    assert "planner_decision_trace" not in retained["_provenance"]["algorithm_metadata"]


def test_exact_rectangular_slot_counts_and_diagnostic_flag():
    sample = {}
    for arm in cmp.ARM_SLOTS_0_0_7_TO_0_0_8:
        for scenario in range(48):
            for seed in range(1001, 1031):
                sample[arm.key_0_0_7, "differential_drive", f"s{scenario}", seed, ""] = {}
    cmp.check_slots(sample, release="0.0.7")
    partial = copy.copy(sample)
    partial.pop(next(iter(partial)))
    with pytest.raises(ValueError, match="14 x 48 x 30"):
        cmp.check_slots(partial, release="0.0.7")
    cmp.check_slots(partial, release="0.0.7", diagnostic_partial=True)


def test_probe_excluded_from_unit_and_arm_pooling():
    old, new = rows(), rows(release="0.0.8")
    probe_id = next(iter(cmp.PROBE_SCENARIO_IDS))
    for release in (old, new):
        for slot, row in list(release.items()):
            release[slot[0], slot[1], probe_id, slot[3], slot[4]] = copy.deepcopy(row)
    report = cmp.compare_samples(old, new)
    assert not any(c["scenario_id"] == probe_id for c in report["cells"])
    assert cell(report, "success", "arm")["release_0_0_7"]["n"] == 30
    assert len(report["probe_units"]) == 1
    assert report["probe_units"][0]["release_0_0_8"]["n"] == 30
    assert sum(report["probe_units"][0]["release_0_0_8"]["safe_failure_classes"].values()) == 30


def test_scenario_and_planner_definition_changes_are_combined():
    old, new = rows(), rows(release="0.0.8")
    for row in new.values():
        row["_definition"]["scenario"]["simulation_config"]["goal_completion_policy"] = "new_policy"
        row["_definition"]["scenario"]["simulation_config"]["ped_density"] = 0.5
        row["_definition"]["algo_config_hash"] = "changed"
    report = cmp.compare_samples(old, new)
    assert all(c["scenario_changed"] and c["planner_config_changed"] for c in report["cells"])
    assert all(c["interpretation"] == "combined release difference" for c in report["cells"])
    fingerprint = next(iter(report["fingerprints"].values()))
    assert fingerprint["0.0.7"]["algo_config_hashes"] == ["abc"]
    assert fingerprint["0.0.8"]["algo_config_hashes"] == ["changed"]


def test_diagnostic_output_label_and_success_only_cell_insufficient():
    report = cmp.compare_samples(rows(successes=4), rows(release="0.0.8"), diagnostic_partial=True)
    assert report["label"] == "diagnostic, not the release comparison"
    assert not cell(report, "path_efficiency")["sufficient"]
    with pytest.raises(ValueError, match="at least 10000"):
        cmp.compare_samples(rows(), rows(release="0.0.8"), resamples=9999)
