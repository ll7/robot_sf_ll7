"""Regression tests for the frozen outcome rule and recorded execution modes."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
import yaml

PACKET = Path(__file__).parent
spec = importlib.util.spec_from_file_location("xhet_analyze", PACKET / "analyze.py")
assert spec is not None and spec.loader is not None
analyzer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(analyzer)


@pytest.mark.parametrize(
    ("success_effect", "slow_interval", "expected_outcome", "expected_qualifier"),
    [
        ((-6.0, -8.0, -2.0), (0.01, 0.02), "B_ORDER_CHANGE_OR_SELECTIVE_DETERIORATION", None),
        (
            (0.0, -1.0, 1.0),
            (0.01, 0.02),
            "A_PERSIST_FINITE_SENSITIVITY",
            "C_SLOW_PED_DELAY_QUALIFIER",
        ),
        (
            (0.0, -11.0, 1.0),
            (0.01, 0.02),
            "D_INCONCLUSIVE_RETAIN_LIMITATION",
            "C_SLOW_PED_DELAY_QUALIFIER",
        ),
        ((0.0, -1.0, 1.0), (0.0, 0.02), "A_PERSIST_FINITE_SENSITIVITY", None),
    ],
)
def test_outcome_rule_applies_c_only_on_top_of_a_or_d(
    success_effect, slow_interval, expected_outcome, expected_qualifier
):
    """C is an observation for B but a formal qualifier only for A or D."""
    rule = yaml.safe_load((PACKET / "contract.yaml").read_text())["outcome_rule"]
    pooled = {}
    for planner in analyzer.PLANNERS:
        pooled[("success", planner)] = (0.0, -1.0, 1.0)
        pooled[("collision", planner)] = (0.0, -1.0, 1.0)
        pooled[("ped_near_delay_rate_slow", planner)] = (0.0, -0.01, 0.01)
    pooled[("success", "orca")] = success_effect
    pooled[("ped_near_delay_rate_slow", "orca")] = (0.015, *slow_interval)
    result = analyzer.decide(rule, pooled, [], {})
    assert result["outcome"] == expected_outcome
    assert result["qualifier"] == expected_qualifier
    assert result["secondary_observations"]["slow_ped_near_delay_increase_planners"] == (
        ["orca"] if slow_interval[0] > 0 else []
    )


@pytest.mark.parametrize(
    ("uniform", "interval", "mixed", "expected"),
    [
        (3.0, (1.0, 5.0), 0.0, False),
        (-3.0, (-5.0, -1.0), 0.0, False),
        (3.0, (1.0, 5.0), -2.0, True),
        (-3.0, (-5.0, -1.0), 2.0, True),
        (3.0, (-1.0, 5.0), -2.0, False),
    ],
)
def test_resolved_order_change_requires_opposite_nonzero_signs(uniform, interval, mixed, expected):
    """A tie or an unresolved original difference is not a change of sign."""
    rule = yaml.safe_load((PACKET / "contract.yaml").read_text())["outcome_rule"]
    pooled = {
        (endpoint, planner): (0.0, -1.0, 1.0)
        for endpoint in ("success", "collision", "ped_near_delay_rate_slow")
        for planner in analyzer.PLANNERS
    }
    interaction = {
        "endpoint": "success",
        "planner_a": "orca",
        "planner_b": "social_force",
        "diff_uniform_pp": uniform,
        "diff_uniform_lo": interval[0],
        "diff_uniform_hi": interval[1],
        "diff_mixed_pp": mixed,
        "interaction_pp": mixed - uniform,
        "interaction_lo": min(mixed - uniform, 0),
        "interaction_hi": max(mixed - uniform, 0),
    }
    result = analyzer.decide(rule, pooled, [interaction], {})
    assert result["triggers"] == (["order_change:orca>social_force"] if expected else [])
    assert result["outcome"] == (
        "B_ORDER_CHANGE_OR_SELECTIVE_DETERIORATION" if expected else "A_PERSIST_FINITE_SENSITIVITY"
    )


@pytest.mark.parametrize(
    "metadata",
    [
        {"status": "policy_step_error_fallback", "planner_runtime": {"fallback_count": 0}},
        {
            "status": "ok",
            "fallback_reason": "prediction_failed",
            "planner_runtime": {"fallback_count": 0},
        },
        {"status": "ok", "planner_runtime": {"fallback_count": 1}},
    ],
)
def test_mode_rejects_recorded_fallback(metadata):
    """Fallback indicators exclude a record even if another indicator is zero."""
    mode, reason = analyzer._mode({"status": "ok", "algorithm_metadata": metadata})
    assert mode == "fallback"
    assert reason


def test_mode_labels_unavailable_counter_without_inventing_zero():
    """A native classification with no counter preserves the telemetry limitation."""
    record = {"status": "ok", "algorithm_metadata": {"status": "ok", "planner_runtime": {}}}
    mode, reason = analyzer._mode(record)
    assert mode == "native"
    assert "planner_fallback_counter_unavailable" in reason
    record["algorithm_metadata"]["planner_runtime"]["fallback_count"] = 0
    mode, reason = analyzer._mode(record)
    assert mode == "native"
    assert "planner_fallback_counter_unavailable" not in reason


@pytest.mark.parametrize("value", [-1, "0", False])
def test_mode_marks_invalid_counter_unknown(value):
    """Invalid counter values are not evidence of a recorded zero."""
    record = {
        "status": "ok",
        "algorithm_metadata": {"status": "ok", "planner_runtime": {"fallback_count": value}},
    }
    mode, reason = analyzer._mode(record)
    assert mode == "native"
    assert "planner_fallback_counter_invalid" in reason


def test_mode_counts_separate_missing_from_zero_counter(tmp_path):
    """A configuration's telemetry counts cannot turn missing counters into zeroes."""
    import csv

    rows = [
        {
            "planner": "orca",
            "cell": "crossing",
            "condition": "uniform",
            "mode": "fallback" if count == 2 else "native",
            "planner_fallback_count": count,
        }
        for count in (None, 0, 2, -1)
    ]
    path = tmp_path / "mode_counts.csv"
    analyzer._write_mode_counts(path, rows)
    with path.open(newline="") as stream:
        result = next(csv.DictReader(stream))
    assert (result["counter_unknown"], result["counter_zero"], result["counter_positive"]) == (
        "2",
        "1",
        "1",
    )


def test_summary_keeps_checkpoint_and_missing_counter_evidence_separate():
    """A loaded checkpoint does not fill in an absent planner fallback counter."""
    rows = [
        {
            "planner": "ppo_release",
            "planner_fallback_count": "0",  # An invalid raw string can serialize like integer zero.
            "planner_fallback_counter_state": "unknown",
            "algorithm_status": "ok",
            "checkpoint_load_succeeded": "True",
            "checkpoint_fallback_triggered": "False",
        },
        {
            "planner": "hybrid_v4_continuous",
            "planner_fallback_count": "0",
            "planner_fallback_counter_state": "zero",
        },
    ]
    summary = analyzer._telemetry_summary(rows)
    assert summary["planner_fallback_counters"]["ppo_release"] == {
        "zero": 0,
        "positive": 0,
        "unknown": 1,
    }
    assert summary["planner_fallback_counters"]["hybrid_v4_continuous"]["zero"] == 1
    assert summary["ppo_checkpoint"]["loaded_without_checkpoint_fallback"] == 1
    assert summary["algorithm_status"] == {"ok": 1, "other": 0, "unknown": 1}


def test_summary_separates_recorded_view_status_from_probe():
    """An undegraded recorded status is not an independently probed view."""
    rows = [
        {
            "planner": "ppo_release",
            "effective_view_degraded": "False",
            "effective_view_probed": "False",
            "effective_view_blind_by_design": "True",
        },
        {
            "planner": "hybrid_v4_continuous",
            "effective_view_degraded": "False",
            "effective_view_probed": "True",
            "effective_view_blind_by_design": "False",
        },
        {"planner": "ppo_release"},
    ]
    views = analyzer._telemetry_summary(rows)["effective_view_by_planner"]
    assert views["ppo_release"] == {
        "recorded_degraded_false": 1,
        "recorded_degraded_true": 0,
        "degraded_unknown": 1,
        "probed": 0,
        "unprobed": 1,
        "probe_unknown": 1,
        "blind_by_design": 1,
        "not_blind_by_design": 0,
        "blind_design_unknown": 1,
    }
    assert views["hybrid_v4_continuous"]["probed"] == 1


def test_sensitivity_summary_separates_input_and_retained_populations(tmp_path):
    """Relocation exclusions are not missing or non-native episodes."""
    import csv
    import json

    rule = yaml.safe_load((PACKET / "contract.yaml").read_text())
    rule["seeds"]["main"] = [1001, 1002]
    rule["analysis"]["bootstrap_draws"] = 2
    contract = tmp_path / "contract.yaml"
    contract.write_text(yaml.safe_dump(rule))
    rows = []
    for planner in analyzer.PLANNERS:
        for cell in analyzer.CELLS:
            for seed in rule["seeds"]["main"]:
                for condition in analyzer.CONDS:
                    row: dict[str, object] = {col: 1.0 for _, col, _, _ in analyzer.ENDPOINTS}
                    row.update(
                        planner=planner,
                        cell=cell,
                        seed=seed,
                        condition=condition,
                        mode="native",
                        pair_relocation_differs=int(
                            planner == "orca" and cell == "crossing" and seed == 1001
                        ),
                        planner_fallback_counter_state="unknown",
                    )
                    rows.append(row)
    episodes = tmp_path / "episodes.csv"
    with episodes.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    analyzer.analyze(str(episodes), str(contract), str(tmp_path / "out"), True)
    summary = json.loads((tmp_path / "out" / "summary.json").read_text())
    population = summary["analysis_population"]
    assert population["input_main_episodes"] == 64
    assert population["excluded_relocation_episodes"] == 2
    assert population["retained_episodes"] == 62
    assert summary["counts"]["episodes_expected"] == 64
    assert summary["counts"]["episodes_present"] == 62
    assert summary["counts"]["episodes_non_native"] == 0
