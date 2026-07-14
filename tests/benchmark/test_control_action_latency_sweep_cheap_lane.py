"""Tests for the cheap-lane CPU control-action-latency sweep slice (issue #5034).

These tests verify the cheap-lane execution step that prior PRs (#5061, #5085,
#5536, #5620, #5629) left un-done: real native episode execution on the
``control_action_latency`` axis for the dependency-free planners, and durable
promotion of those rows. They do NOT require Slurm or the native ORCA/hybrid
planner set; they run a tiny in-process campaign.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from collections.abc import Mapping
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
SCENARIO_SET = REPO_ROOT / "configs" / "scenarios" / "sets" / "issue_5034_latency_sweep_cpu_v1.yaml"
CONFIG_PATH = REPO_ROOT / "configs" / "research" / "fidelity_sensitivity_v1.yaml"
RUNNER_PATH = REPO_ROOT / "scripts" / "benchmark" / "run_control_action_latency_sweep_cheap_lane.py"
PROMOTER_PATH = REPO_ROOT / "scripts" / "benchmark" / "promote_control_action_latency_evidence.py"

LATENCY_AXIS_KEY = "control_action_latency"
REQUIRED_LATENCY_STEPS = (0, 1, 3)
NATIVE_CPU_PLANNERS = ("goal_seek", "baseline_social_force")


def _load_module(name: str, path: Path) -> object:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def cheap_lane_runner() -> object:
    return _load_module("run_control_action_latency_sweep_cheap_lane", RUNNER_PATH)


def test_scenario_set_uses_pedestrian_bearing_scenarios() -> None:
    """The cheap-lane slice should exercise pedestrian-bearing scenarios, not empty maps."""
    data = yaml.safe_load(SCENARIO_SET.read_text(encoding="utf-8"))
    scenarios = data.get("scenarios", [])
    assert scenarios, "cheap-lane scenario set must define at least one scenario"
    assert any(
        isinstance(scenario.get("simulation_config"), Mapping)
        and float(scenario["simulation_config"].get("ped_density", 0.0)) > 0.0
        for scenario in scenarios
    )


def test_runner_exposes_native_cpu_planners_only(cheap_lane_runner: object) -> None:
    """The cheap-lane slice must restrict execution to dependency-free planners."""
    assert tuple(cheap_lane_runner.NATIVE_CPU_PLANNERS) == NATIVE_CPU_PLANNERS


def test_runner_executes_real_latency_axis_episodes(cheap_lane_runner: object) -> None:
    """Executing the slice must emit real latency-axis rows covering 0/1/3 steps."""
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    rows = cheap_lane_runner.run_sweep(
        config=config,
        scenario_path=SCENARIO_SET,
        planner_names=NATIVE_CPU_PLANNERS,
        horizon=20,
        seeds=[101, 102],
    )
    latency_rows = [row for row in rows if str(row.get("axis")) == LATENCY_AXIS_KEY]
    assert latency_rows, "runner must emit control_action_latency axis rows"
    observed_steps = sorted({int(row["action_latency"]["effective_steps"]) for row in latency_rows})
    assert observed_steps == list(REQUIRED_LATENCY_STEPS)
    # Every latency row must carry native availability metadata so the promoter
    # accepts it as a result cell (not an exclusion).
    for row in latency_rows:
        assert row.get("execution_mode", "native") == "native"
        assert row.get("availability_status", "available") == "available"


def test_runner_rejects_non_native_planner_request(cheap_lane_runner: object) -> None:
    """A request for a planner outside the CPU-native set must fail before execution.

    The cheap-lane slice restricts execution to dependency-free planners; a
    non-native planner such as ``orca`` must never be silently dropped or run.
    """
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    with pytest.raises(ValueError, match="unsupported planner request"):
        cheap_lane_runner.run_sweep(
            config=config,
            scenario_path=SCENARIO_SET,
            planner_names=["orca", "goal_seek"],
            horizon=20,
            seeds=[101],
        )


def test_promoter_accepts_cheap_lane_rows(cheap_lane_runner: object, tmp_path: Path) -> None:
    """Real cheap-lane rows must promote without exclusion or coverage failure."""
    promoter = _load_module("promote_control_action_latency_evidence", PROMOTER_PATH)
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    rows = cheap_lane_runner.run_sweep(
        config=config,
        scenario_path=SCENARIO_SET,
        planner_names=NATIVE_CPU_PLANNERS,
        horizon=20,
        seeds=[101, 102],
    )
    raw_rows_path = tmp_path / "episode_rows.jsonl"
    raw_rows_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8"
    )
    evidence_dir = tmp_path / "evidence"
    packet = promoter.build_latency_evidence(
        promoter.load_latency_rows(raw_rows_path),
        config=config,
        config_path="configs/research/fidelity_sensitivity_v1.yaml",
        git_head="deadbeef",
        date="2026-07-14",
        raw_rows_path="output/fidelity_latency_raw/episode_rows.jsonl",
    )
    assert packet["latency_coverage"]["coverage_complete"] is True
    assert packet["scope"]["excluded_row_count"] == 0
    assert packet["scope"]["result_row_count"] == len(rows)
    written = promoter.write_latency_evidence(packet, evidence_dir)
    assert (evidence_dir / "summary.json") in written
    assert (evidence_dir / "manifest.sha256") in written
