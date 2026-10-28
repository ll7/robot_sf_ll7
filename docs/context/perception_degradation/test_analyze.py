"""Regression tests for the frozen perception-degradation analysis contract."""

from __future__ import annotations

import gzip
import importlib.util
import json
import subprocess
import sys
from copy import deepcopy
from pathlib import Path

import pytest
import yaml

PACKET = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("perception_analysis", PACKET / "analyze.py")
assert SPEC is not None and SPEC.loader is not None
analysis = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(analysis)


@pytest.fixture(scope="module")
def summary(tmp_path_factory: pytest.TempPathFactory) -> dict:
    """Analyze the archived compact rows through the public command line."""
    output = tmp_path_factory.mktemp("perception-analysis")
    subprocess.run(
        [
            sys.executable,
            str(PACKET / "analyze.py"),
            "--episodes",
            str(PACKET / "evidence/episodes.jsonl.gz"),
            "--output",
            str(output),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads((output / "summary.json").read_text())


def test_intervention_failures_are_reported_by_cell_and_condition() -> None:
    """Every undisturbed treatment row and unexercised cell triggers literal O5."""
    with gzip.open(PACKET / "evidence/episodes.jsonl.gz", "rt", encoding="utf-8") as fh:
        rows = [json.loads(line) for line in fh]
    contract = yaml.safe_load((PACKET / "contract.yaml").read_text())
    report = analysis.validity(rows, contract["cells"])
    failures = report.get("intervention_failures", {})

    expected_undisturbed = {
        "classic_cross_trap_medium|static_occlusion": 2,
        "classic_overtaking_medium|static_occlusion": 1,
        "classic_head_on_corridor_medium|static_occlusion": 200,
        "classic_bottleneck_high|static_occlusion": 200,
    }
    assert set(failures) == set(expected_undisturbed)
    for key, count in expected_undisturbed.items():
        assert failures[key]["treatment_never_differs_from_truth"] == count
    for cell in contract["occlusion_negative_control_cells"]:
        assert failures[f"{cell}|static_occlusion"]["occlusion_not_exercised"] is True


def test_missing_ppo_checkpoint_fallback_evidence_is_not_a_verified_false() -> None:
    """A missing checkpoint fallback flag cannot establish fallback-free loading."""
    with gzip.open(PACKET / "evidence/episodes.jsonl.gz", "rt", encoding="utf-8") as fh:
        row = next(r for r in map(json.loads, fh) if r["planner_key"] == "ppo")
    row = deepcopy(row)
    row["checkpoint_fallback_triggered"] = None
    report = analysis.validity([row], [row["scenario_id"]])
    group = report["by_planner_condition"][f"ppo|{row['condition']}"]
    assert group["checkpoint_not_loaded"] == 1


def test_timeout_follows_the_frozen_max_steps_definition() -> None:
    """Timeout requires max-steps termination, not merely a producer event flag."""
    with gzip.open(PACKET / "evidence/episodes.jsonl.gz", "rt", encoding="utf-8") as fh:
        row = next(r for r in map(json.loads, fh) if r["termination_reason"] == "max_steps")
    row = deepcopy(row)
    row["outcome"]["timeout_event"] = False
    assert analysis.endpoint_values(row)[analysis.ENDPOINTS.index("timeout")] == 1.0
    row["termination_reason"] = "other"
    row["outcome"]["timeout_event"] = True
    assert analysis.endpoint_values(row)[analysis.ENDPOINTS.index("timeout")] == 0.0


def test_o5_summary_separates_expected_controls_from_active_failures(summary: dict) -> None:
    """O5 must expose all four failed cell/conditions without treating controls as active."""
    failures = summary["decision_rule"]["O5_intervention_failed"]
    actual = {(entry["cell"], entry["condition"]): entry for entry in failures}
    assert set(actual) == {
        ("classic_cross_trap_medium", "static_occlusion"),
        ("classic_overtaking_medium", "static_occlusion"),
        ("classic_head_on_corridor_medium", "static_occlusion"),
        ("classic_bottleneck_high", "static_occlusion"),
    }
    assert actual[("classic_cross_trap_medium", "static_occlusion")]["negative_control"] is False
    assert actual[("classic_overtaking_medium", "static_occlusion")]["negative_control"] is False
    assert actual[("classic_head_on_corridor_medium", "static_occlusion")]["negative_control"]
    assert actual[("classic_bottleneck_high", "static_occlusion")]["negative_control"]


def test_o1_five_percentage_point_boundary_is_inclusive(summary: dict) -> None:
    """The independently counted ORCA delay effects are exactly +5 and -5 pp."""
    relevant_and_narrow = summary["decision_rule"]["O1_relevant_and_narrow"]
    assert "orca|delay|success" in relevant_and_narrow
    assert "orca|delay|collision" in relevant_and_narrow


def test_occlusion_rankings_compare_matched_populations(summary: dict) -> None:
    """The four-cell reversal and six-cell baseline must never be mixed."""
    populations = summary["occlusion_ranking_populations"]
    pair = "hybrid_v4_continuous_guard_off-ppo|success"
    four = populations["four_active_cells"]["comparisons"][pair]
    six = populations["six_cells"]["comparisons"][pair]
    assert four["oracle"] == [-0.18, -0.245, -0.115]
    assert four["static_occlusion"] == [0.1, 0.03, 0.17]
    assert four["ranking_change"] == [0.28, 0.235, 0.32]
    assert four["o2"] == "reversal"
    assert six["oracle"] == [0.006667, -0.04, 0.053333]
    assert six["o2"] == "not_triggered"
