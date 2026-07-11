"""Production-call contracts for the issue #5331 DWA global-route trace runner."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "benchmark" / "trace_dwa_global_route_probe_issue_5331.py"


def _load_trace_module():
    spec = importlib.util.spec_from_file_location("_dwa_global_route_trace_issue_5331", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_load_scenario_resolves_relative_paths_from_matrix_parent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The scenario loader resolves relatives from the matrix directory."""
    trace = _load_trace_module()
    matrix = ROOT / "configs" / "scenarios" / "classic_interactions.yaml"
    captured: dict[str, Path] = {}

    def fake_load(path: Path, *, base_dir: Path):
        captured["path"] = path
        captured["base_dir"] = base_dir
        return [{"name": "target"}]

    monkeypatch.setattr(trace, "load_scenarios", fake_load)

    assert trace._load_scenario("target", 7, matrix) == {"name": "target", "seeds": [7]}
    assert captured == {"path": matrix, "base_dir": matrix.parent}


def test_run_trace_uses_current_map_runner_contract(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The trace runner passes an episode JSONL path to the production batch API."""
    trace = _load_trace_module()
    calls: list[tuple[list[dict[str, object]], Path, dict[str, object]]] = []

    def fake_load_scenario(name: str, seed: int, _matrix_path: Path) -> dict[str, object]:
        return {"name": name, "seeds": [seed]}

    def fake_run_map_batch(
        scenarios_or_path: list[dict[str, object]],
        out_path: Path,
        schema_path: Path,
        **kwargs: object,
    ) -> dict[str, object]:
        calls.append((scenarios_or_path, out_path, {"schema_path": schema_path, **kwargs}))
        out_path.parent.mkdir(parents=True, exist_ok=True)
        seed = scenarios_or_path[0]["seeds"][0]
        out_path.write_text(
            json.dumps(
                {
                    "scenario_id": scenarios_or_path[0]["name"],
                    "seed": seed,
                    "termination_reason": "max_steps",
                    "steps": 1,
                    "outcome": {
                        "route_complete": False,
                        "collision_event": False,
                        "timeout_event": True,
                    },
                    "algorithm_metadata": {
                        "planner_decision_trace": {
                            "steps": [
                                {
                                    "step": 0,
                                    "selected_command": [0.1, 0.0],
                                    "selected_score": 1.0,
                                    "constraint_reason": "best_feasible",
                                    "candidate_total": 1,
                                    "candidate_feasible": 1,
                                    "candidate_infeasible": 0,
                                    "dynamic_window": {},
                                    "target_goal": {},
                                    "distance_to_goal_m": 1.0,
                                    "route_progress_from_start_m": 0.0,
                                    "global_route_probe_activated": True,
                                }
                            ]
                        }
                    },
                }
            )
            + "\n",
            encoding="utf-8",
        )
        return {}

    monkeypatch.setattr(trace, "_load_scenario", fake_load_scenario)
    monkeypatch.setattr(trace, "run_map_batch", fake_run_map_batch)

    trace.run_trace(
        algo_config=trace.DEFAULT_ALGO_CONFIG,
        matrix_path=trace.DEFAULT_MATRIX,
        out_dir=tmp_path,
    )

    assert len(calls) == 2
    for scenarios, out_path, kwargs in calls:
        assert scenarios[0]["seeds"] in ([131], [161])
        assert out_path.suffix == ".jsonl"
        assert kwargs["schema_path"] == trace.SCHEMA_PATH
        assert kwargs["scenario_path"] == trace.DEFAULT_MATRIX
        assert kwargs["algo"] == "dwa"
        assert kwargs["algo_config_path"] == str(trace.DEFAULT_ALGO_CONFIG)
        assert kwargs["benchmark_profile"] == "experimental"
        assert kwargs["record_planner_decision_trace"] is True
