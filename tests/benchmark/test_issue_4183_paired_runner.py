"""Tests for the issue #4183 paired diagnostic runner."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
import yaml

from robot_sf.benchmark.hybrid_global_rl_diagnostic import ROUTE_ARM


def _load_issue_4183_runner():
    script_path = Path("scripts/benchmark/run_hybrid_global_rl_diagnostic_issue_4183.py")
    spec = importlib.util.spec_from_file_location("issue_4183_runner", script_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_issue_4183_runner_honors_config_seed_policy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The paired runner overrides scenario-matrix seeds with benchmark-config seeds."""
    scenario_matrix = tmp_path / "scenarios.yaml"
    scenario_matrix.write_text(
        yaml.safe_dump(
            {
                "scenarios": [
                    {
                        "name": "francis2023_intersection_wait",
                        "seeds": [240],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    config_path = tmp_path / "arm.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "scenario_matrix": str(scenario_matrix),
                "seed_policy": {"mode": "fixed-list", "seeds": [4183, 4184]},
                "horizon": 120,
                "dt": 0.1,
                "record_forces": True,
                "workers": 1,
                "planners": [
                    {
                        "algo": "hybrid_global_rl",
                        "benchmark_profile": "diagnostic-only",
                        "config": {"local_policy_algo": "ppo"},
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    captured: dict[str, object] = {}
    issue_4183_runner = _load_issue_4183_runner()

    def fake_run_map_batch(
        scenarios_or_path: list[dict[str, object]],
        out_path: Path,
        *_args: object,
        **_kwargs: object,
    ) -> dict[str, object]:
        captured["scenario_seeds"] = scenarios_or_path[0]["seeds"]
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text("", encoding="utf-8")
        return {
            "failures": [
                {
                    "error": "RuntimeError('hybrid_global_rl route waypoint unavailable')",
                    "scenario_id": "francis2023_intersection_wait",
                    "seed": 4183,
                }
            ]
        }

    monkeypatch.setattr(issue_4183_runner, "run_map_batch", fake_run_map_batch)
    _episodes_path, _summary, failures = issue_4183_runner._run_arm(
        config_path=config_path,
        arm=ROUTE_ARM,
        work_dir=tmp_path / "run",
        schema_path=tmp_path / "schema.json",
        horizon_override=5,
    )
    assert captured["scenario_seeds"] == [4183, 4184]
    assert failures == [
        {
            "arm": ROUTE_ARM,
            "scenario_id": "francis2023_intersection_wait",
            "seed": 4183,
            "reason": "RuntimeError('hybrid_global_rl route waypoint unavailable')",
            "row_classification": "fail_closed_no_episode_row",
            "source": "issue_4183_paired_runner",
        }
    ]
