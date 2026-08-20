"""Focused contracts for the shared DWA diagnostic execution envelope."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest

from robot_sf.benchmark.dwa_diagnostic_harness import (
    DwaDiagnosticRequest,
    collect_episode,
    flatten_trace_step,
    load_scenario,
    write_json_atomic,
    write_markdown_atomic,
)

if TYPE_CHECKING:
    from pathlib import Path


def _record(*, scenario: str = "target", seed: int = 7) -> dict[str, object]:
    return {
        "scenario_id": scenario,
        "seed": seed,
        "termination_reason": "max_steps",
        "steps": 1,
        "outcome": {"route_complete": False, "collision_event": False, "timeout_event": True},
        "algorithm_metadata": {
            "planner_decision_trace": {
                "steps": [
                    {
                        "step": 0,
                        "selected_command": [0.5, 0.1],
                        "selected_score": 1.0,
                        "constraint_reason": "best_feasible",
                        "candidate_total": 1,
                        "candidate_feasible": 1,
                        "candidate_infeasible": 0,
                        "dynamic_window": {"v_min": 0.0, "v_max": 1.0},
                        "target_goal": {"kind": "goal", "x": 1.0, "y": 2.0},
                        "distance_to_goal_m": 2.0,
                        "route_progress_from_start_m": 0.0,
                    }
                ]
            }
        },
    }


def test_load_scenario_rejects_ambiguous_names(tmp_path: Path) -> None:
    matrix = tmp_path / "matrix.yaml"

    def fake_load(_path: Path, *, base_dir: Path) -> list[dict[str, object]]:
        assert base_dir == matrix.parent
        return [{"name": "target"}, {"name": "target"}]

    with pytest.raises(ValueError, match="ambiguous"):
        load_scenario("target", 7, matrix, load_scenarios_fn=fake_load)


def test_collect_episode_supports_existing_result_without_running_map_batch(tmp_path: Path) -> None:
    result = tmp_path / "existing.jsonl"
    result.write_text(json.dumps(_record()) + "\n", encoding="utf-8")
    request = DwaDiagnosticRequest(
        config_path=tmp_path / "algo.yaml",
        scenario="target",
        seed=7,
        algorithm="dwa",
        output_dir=tmp_path / "out",
        existing_result=result,
    )

    def unexpected_runner(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("existing-result mode must not invoke the map runner")

    episode = collect_episode(request, run_map_batch_fn=unexpected_runner)

    assert episode.episode_row["scenario_id"] == "target"
    assert len(episode.trace_steps) == 1
    assert episode.source_artifacts == {"episodes_jsonl": result}


def test_collect_episode_rejects_duplicate_provenance_rows(tmp_path: Path) -> None:
    result = tmp_path / "existing.jsonl"
    result.write_text(json.dumps(_record()) + "\n", encoding="utf-8")
    result.with_name(result.name + ".provenance.json").write_text(
        json.dumps(
            {
                "rows": [
                    {"scenario_id": "target", "seed": 7},
                    {"scenario_id": "target", "seed": 7},
                ]
            }
        ),
        encoding="utf-8",
    )
    request = DwaDiagnosticRequest(
        config_path=tmp_path / "algo.yaml",
        scenario="target",
        seed=7,
        algorithm="dwa",
        output_dir=tmp_path / "out",
        existing_result=result,
    )

    with pytest.raises(ValueError, match="exactly one provenance row"):
        collect_episode(request)


def test_collect_episode_preserves_map_runner_contract(tmp_path: Path) -> None:
    calls: list[dict[str, object]] = []
    matrix = tmp_path / "matrix.yaml"
    schema = tmp_path / "schema.json"

    def fake_load_scenario(name: str, seed: int, path: Path) -> dict[str, object]:
        assert (name, seed, path) == ("target", 7, matrix)
        return {"name": name, "seeds": [seed]}

    def fake_run(scenarios: list[dict[str, object]], out_path: Path, **kwargs: object) -> None:
        calls.append({"scenarios": scenarios, "out_path": out_path, **kwargs})
        out_path.write_text(json.dumps(_record()) + "\n", encoding="utf-8")

    request = DwaDiagnosticRequest(
        config_path=tmp_path / "algo.yaml",
        scenario="target",
        seed=7,
        algorithm="dwa",
        output_dir=tmp_path / "out",
        episode_id="target_episode",
    )
    collect_episode(
        request,
        matrix_path=matrix,
        schema_path=schema,
        run_map_batch_fn=fake_run,
        load_scenario_fn=fake_load_scenario,
    )

    assert len(calls) == 1
    call = calls[0]
    assert call["scenarios"] == [{"name": "target", "seeds": [7]}]
    assert call["out_path"] == tmp_path / "out" / "episodes_target_episode.jsonl"
    assert call["schema_path"] == schema
    assert call["scenario_path"] == matrix
    assert call["algo"] == "dwa"
    assert call["algo_config_path"] == str(tmp_path / "algo.yaml")
    assert call["benchmark_profile"] == "experimental"
    assert call["record_planner_decision_trace"] is True


def test_flatten_trace_step_rejects_malformed_nested_fields() -> None:
    with pytest.raises(ValueError, match="dynamic_window"):
        flatten_trace_step(
            {"selected_command": [0.1, 0.0], "dynamic_window": "invalid"},
            episode_id="episode",
            scenario_id="target",
            seed=7,
        )
    with pytest.raises(ValueError, match="selected_command"):
        flatten_trace_step(
            {"selected_command": "invalid"},
            episode_id="episode",
            scenario_id="target",
            seed=7,
        )


def test_atomic_publication_preserves_markers_and_cleans_temporary_files(tmp_path: Path) -> None:
    json_path = tmp_path / "nested" / "summary.json"
    markdown_path = tmp_path / "nested" / "README.md"

    write_json_atomic(json_path, {"value": 1}, review_marker=True)
    write_markdown_atomic(markdown_path, "<!-- AI-GENERATED (test) - NEEDS-REVIEW -->\n# Test\n")

    assert json.loads(json_path.read_text(encoding="utf-8"))["review_marker"] == (
        "AI-GENERATED NEEDS-REVIEW"
    )
    assert markdown_path.read_text(encoding="utf-8").startswith("<!-- AI-GENERATED")
    assert list(json_path.parent.glob("*.json.*")) == []
    assert list(markdown_path.parent.glob("README.md.*")) == []
