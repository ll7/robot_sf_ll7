"""Tests for the issue #2777 live observation-perturbation replay wrapper."""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = REPO_ROOT / "scripts/benchmark/run_observation_noise_live_replay_issue_2777.py"
_LOADED_MOD = None


def _load_script():
    """Load the issue script as a module."""
    global _LOADED_MOD
    if _LOADED_MOD is not None:
        return _LOADED_MOD
    spec = importlib.util.spec_from_file_location(
        "run_observation_noise_live_replay_issue_2777", SCRIPT_PATH
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules["run_observation_noise_live_replay_issue_2777"] = mod
    spec.loader.exec_module(mod)
    _LOADED_MOD = mod
    return mod


def test_condition_set_matches_issue_2755_contract() -> None:
    """The wrapper should preserve the seven perturbation family names."""
    mod = _load_script()

    assert tuple(condition.name for condition in mod.CONDITIONS) == mod.REQUIRED_CONDITIONS
    delay = next(condition for condition in mod.CONDITIONS if condition.name == "delay_only")
    assert "--observation-delay-steps" in delay.flags
    assert "2" in delay.flags


def test_default_strict_mode_fails_closed_without_occluded_fixture(tmp_path: Path) -> None:
    """A non-occluded matrix should not be promoted as #2777 live replay evidence."""
    mod = _load_script()
    matrix = tmp_path / "matrix.yaml"
    matrix.write_text(
        "schema_version: robot_sf.scenario_matrix.v1\n"
        "select_scenarios: [issue_3233_near_field_observation_noise]\n",
        encoding="utf-8",
    )
    output_dir = tmp_path / "out"
    args = mod.parse_args(["--scenario-matrix", str(matrix), "--output-dir", str(output_dir)])

    report = mod.run_live_batch(args)

    assert report["status"] == "fail_closed"
    assert report["classification"]["label"] == "blocked"
    assert report["fixture_contract"]["satisfied"] is False
    assert len(report["conditions"]) == 7
    assert {condition["status"] for condition in report["conditions"]} == {"blocked"}


def test_dry_run_plans_all_live_diagnostics_commands(tmp_path: Path) -> None:
    """Proxy dry-run mode should expose the exact subprocess plan without execution."""
    mod = _load_script()
    matrix = tmp_path / "matrix.yaml"
    matrix.write_text(
        "schema_version: robot_sf.scenario_matrix.v1\n"
        "select_scenarios: [issue_3233_near_field_observation_noise]\n",
        encoding="utf-8",
    )
    output_dir = tmp_path / "out"
    args = mod.parse_args(
        [
            "--scenario-matrix",
            str(matrix),
            "--output-dir",
            str(output_dir),
            "--allow-non-occluded-live-fixture",
            "--dry-run",
        ]
    )

    report = mod.run_live_batch(args)

    assert report["status"] == "diagnostic_only"
    assert [condition["name"] for condition in report["conditions"]] == list(
        mod.REQUIRED_CONDITIONS
    )
    commands = {condition["name"]: condition["command"] for condition in report["conditions"]}
    assert "--observation-delay-steps" in commands["delay_only"]
    assert "--occlusion-distance-m" in commands["occlusion_only"]
    assert "--missed-detection-probability" in commands["missed_detection_only"]
    assert (output_dir / "generated_policy_search_funnel.yaml").exists()


def test_scalar_includes_are_ignored_for_fixture_detection(tmp_path: Path) -> None:
    """Malformed scalar includes should not be iterated character-by-character."""
    mod = _load_script()
    matrix = tmp_path / "matrix.yaml"
    matrix.write_text(
        "schema_version: robot_sf.scenario_matrix.v1\n"
        "includes: ../single/issue_2756_occluded_emergence.yaml\n",
        encoding="utf-8",
    )

    contract = mod._fixture_contract(matrix)

    assert contract["satisfied"] is False
    assert "blocker" in contract


def test_live_condition_timeout_becomes_fail_closed_blocker(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """Subprocess timeouts should produce a compact blocker instead of crashing."""
    mod = _load_script()

    def _raise_timeout(*_args, **_kwargs):
        raise subprocess.TimeoutExpired(cmd=["diagnostics"], timeout=0.01, stderr="hung")

    monkeypatch.setattr(mod.subprocess, "run", _raise_timeout)
    args = mod.parse_args(
        [
            "--scenario-matrix",
            str(tmp_path / "matrix.yaml"),
            "--output-dir",
            str(tmp_path / "out"),
            "--allow-non-occluded-live-fixture",
        ]
    )
    funnel = mod._write_generated_funnel(
        output_dir=args.output_dir,
        scenario_matrix=args.scenario_matrix,
        stage=args.stage,
        horizon=args.horizon,
    )

    conditions, blockers = mod._execute_conditions(
        args=args,
        output_dir=args.output_dir,
        funnel_config=funnel,
    )

    assert len(conditions) == 7
    assert len(blockers) == 7
    assert all(condition["status"] == "blocked" for condition in conditions)
    assert "timed out" in blockers[0]


def _write_trace(path: Path, *, command: list[float], observed_count: int, closest: float) -> None:
    payload = {
        "candidate": "risk_surface_dwa_v0",
        "stage": "issue_2777_live_replay",
        "scenario_id": "issue_2756_occluded_emergence",
        "seed": 2756,
        "algo": "risk_surface_dwa",
        "progress_summary": {
            "net_goal_progress": 1.0,
            "best_goal_progress": 1.2,
            "closest_robot_ped_distance": closest,
            "closest_robot_ped_step": 3,
            "collision_flag_counts": {"pedestrian": 0, "obstacle": 0, "robot": 0},
            "progress_step_count": 1,
            "regression_step_count": 0,
            "stagnant_step_count": 0,
            "longest_stagnant_run": 0,
        },
        "steps": [
            {
                "policy_command": command,
                "observation_perturbation": {
                    "noise_profile": "bounded_gaussian",
                    "missed_actor_count": 0,
                    "occluded_actor_count": 0,
                    "observed_actor_count": observed_count,
                },
            }
        ],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_trace_comparison_names_scenario_seed_planner_and_policy_insensitive(
    tmp_path: Path,
) -> None:
    """Comparison payload should carry the benchmark-facing provenance fields."""
    mod = _load_script()
    noop = tmp_path / "noop.json"
    condition = tmp_path / "condition.json"
    _write_trace(noop, command=[1.0, 0.0], observed_count=1, closest=1.5)
    _write_trace(condition, command=[1.0, 0.0], observed_count=0, closest=1.5)

    comparison = mod._compare_condition(
        noop_trace_path=noop,
        condition_trace_path=condition,
        fixture_contract_satisfied=True,
    )

    assert comparison["scenario"]["same"] is True
    assert comparison["seed"]["same"] is True
    assert comparison["planner_mode"]["candidate"] == "risk_surface_dwa_v0"
    assert comparison["classification"]["label"] == "policy_insensitive"


def test_markdown_and_json_outputs_are_written(tmp_path: Path) -> None:
    """The wrapper writes compact durable artifacts."""
    mod = _load_script()
    matrix = tmp_path / "matrix.yaml"
    matrix.write_text("schema_version: robot_sf.scenario_matrix.v1\n", encoding="utf-8")
    output_dir = tmp_path / "out"

    exit_code = mod.main(["--scenario-matrix", str(matrix), "--output-dir", str(output_dir)])

    assert exit_code == 2
    summary = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["schema_version"] == mod.SCHEMA_VERSION
    assert "Issue #2777" in (output_dir / "README.md").read_text(encoding="utf-8")


def test_relative_output_dir_is_repo_relative_from_other_cwd(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """Repository-relative artifact paths should not depend on the launch cwd."""
    mod = _load_script()
    fake_repo = tmp_path / "repo"
    fake_repo.mkdir()
    matrix = fake_repo / "matrix.yaml"
    matrix.write_text("schema_version: robot_sf.scenario_matrix.v1\n", encoding="utf-8")
    outside_cwd = tmp_path / "outside"
    outside_cwd.mkdir()
    monkeypatch.setattr(mod, "REPO_ROOT", fake_repo)
    monkeypatch.chdir(outside_cwd)

    exit_code = mod.main(["--scenario-matrix", "matrix.yaml", "--output-dir", "relative-out"])

    assert exit_code == 2
    assert (fake_repo / "relative-out" / "summary.json").exists()
    assert not (outside_cwd / "relative-out").exists()
