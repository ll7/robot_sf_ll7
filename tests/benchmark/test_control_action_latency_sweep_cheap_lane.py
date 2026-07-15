"""Tests for the cheap-lane CPU control-action-latency sweep slice (issue #5034).

These tests verify the cheap-lane execution step that prior PRs (#5061, #5085,
#5536, #5620, #5629, #5648) progressively built up: real native episode execution
on the ``control_action_latency`` axis for the native CPU-runnable planner groups,
and durable promotion of those rows. They do NOT require Slurm; they run a tiny
in-process campaign.

The ORCA (``rvo2``) and ``hybrid_rule_v0_minimal`` planner groups are native on a
CPU-capable host with their runtime present, so the slice now covers all four native
groups rather than only the two always-dependency-free planners from PR #5648.
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
#: All native CPU-runnable planner groups the slice may execute.
NATIVE_CPU_PLANNERS = ("goal_seek", "baseline_social_force", "orca", "hybrid_rule_v0_minimal")
#: Always-dependency-free subset (the PR #5648 baseline).
DEPENDENCY_FREE_PLANNERS = ("goal_seek", "baseline_social_force")


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


def test_runner_exposes_all_native_cpu_planner_groups(cheap_lane_runner: object) -> None:
    """The slice must cover every native CPU-runnable planner group, including ORCA/hybrid."""
    assert tuple(cheap_lane_runner.NATIVE_CPU_PLANNERS) == NATIVE_CPU_PLANNERS


def test_runner_executes_real_latency_axis_episodes(cheap_lane_runner: object) -> None:
    """Executing the dependency-free planners must emit latency rows covering 0/1/3 steps."""
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    rows = cheap_lane_runner.run_sweep(
        config=config,
        scenario_path=SCENARIO_SET,
        planner_names=DEPENDENCY_FREE_PLANNERS,
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


@pytest.mark.parametrize("planner", ["orca", "hybrid_rule_v0_minimal"])
def test_runner_executes_orca_and_hybrid_natively_when_available(
    cheap_lane_runner: object, planner: str
) -> None:
    """ORCA (rvo2) and hybrid planners run natively on CPU when their runtime is present.

    These two planner groups were previously treated as out of cheap-lane reach. On a
    CPU-capable host with ``rvo2`` installed they are native (no Slurm, no fallback),
    so they must emit real latency rows. The test skips cleanly on a host whose
    optional runtime is missing rather than failing.
    """
    if not cheap_lane_runner._native_runtime_available(planner):
        pytest.skip(f"{planner} native runtime not importable on this host")
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    rows = cheap_lane_runner.run_sweep(
        config=config,
        scenario_path=SCENARIO_SET,
        planner_names=[planner],
        horizon=20,
        seeds=[101],
    )
    latency_rows = [row for row in rows if str(row.get("axis")) == LATENCY_AXIS_KEY]
    assert latency_rows, f"runner must emit latency rows for native planner {planner}"
    observed_steps = sorted({int(row["action_latency"]["effective_steps"]) for row in latency_rows})
    assert observed_steps == list(REQUIRED_LATENCY_STEPS)
    for row in latency_rows:
        assert row.get("execution_mode", "native") == "native"
        assert row.get("availability_status", "available") == "available"


def test_runner_rejects_non_native_planner_request(cheap_lane_runner: object) -> None:
    """A planner name outside the native CPU set must fail before execution.

    The cheap-lane slice restricts execution to native CPU-runnable planner groups;
    an unknown planner must never be silently dropped or run.
    """
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    with pytest.raises(ValueError, match="unsupported planner request"):
        cheap_lane_runner.run_sweep(
            config=config,
            scenario_path=SCENARIO_SET,
            planner_names=["definitely_not_a_real_planner", "goal_seek"],
            horizon=20,
            seeds=[101],
        )


def test_runner_rejects_native_planner_with_missing_runtime(
    monkeypatch: pytest.MonkeyPatch, cheap_lane_runner: object
) -> None:
    """A native planner whose optional runtime is missing must fail closed.

    Simulates ``rvo2`` being absent: ``orca`` is in the native set but its runtime
    probe reports unavailable, so the slice must reject it rather than silently
    dropping it or falling back to a heuristic ORCA.
    """
    monkeypatch.setattr(cheap_lane_runner, "_native_runtime_available", lambda planner: False)
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    with pytest.raises(ValueError, match="native runtime not importable"):
        cheap_lane_runner.run_sweep(
            config=config,
            scenario_path=SCENARIO_SET,
            planner_names=["orca"],
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
        planner_names=DEPENDENCY_FREE_PLANNERS,
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
    assert packet["raw_rows_path"] == "ignored_output/fidelity_latency_raw/episode_rows.jsonl"
    assert packet["scope"]["excluded_row_count"] == 0
    assert packet["scope"]["result_row_count"] == len(rows)
    written = promoter.write_latency_evidence(packet, evidence_dir)
    assert (evidence_dir / "summary.json") in written
    assert (evidence_dir / "manifest.sha256") in written
    summary_text = (evidence_dir / "summary.json").read_text(encoding="utf-8")
    readme_text = (evidence_dir / "README.md").read_text(encoding="utf-8")
    csv_text = (evidence_dir / "per_cell_metrics.csv").read_text(encoding="utf-8")
    manifest_text = (evidence_dir / "manifest.sha256").read_text(encoding="utf-8")
    assert "AI-GENERATED" in summary_text and "NEEDS-REVIEW" in summary_text
    assert "AI-GENERATED" in readme_text and "NEEDS-REVIEW" in readme_text
    assert "AI-GENERATED" in csv_text and "NEEDS-REVIEW" in csv_text
    assert "# distance_convention: surface_clearance" in csv_text
    assert "AI-GENERATED" in manifest_text and "NEEDS-REVIEW" in manifest_text
