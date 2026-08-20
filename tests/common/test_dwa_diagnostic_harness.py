"""Focused contracts for the shared DWA diagnostic execution envelope."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

import robot_sf.benchmark.dwa_diagnostic_harness as harness
from robot_sf.benchmark.dwa_diagnostic_harness import (
    DwaDiagnosticRequest,
    _extract_trace_steps,
    _read_json_object,
    _validate_identity,
    collect_episode,
    constraint_reason_counts,
    first_infeasible_candidate_step,
    first_unrecoverable_step,
    flatten_trace_step,
    load_scenario,
    read_single_episode_record,
    repo_relative_path,
    route_progress_summary,
    sha256_file,
    summarize_episode,
    trace_commit,
    write_json_atomic,
    write_markdown_atomic,
    write_steps_csv,
)


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


def test_load_scenario_pins_one_seed_and_rejects_missing_or_lossy_seed(
    tmp_path: Path,
) -> None:
    matrix = tmp_path / "matrix.yaml"

    def fake_load(_path: Path, *, base_dir: Path) -> list[dict[str, object]]:
        assert base_dir == matrix.parent
        return [{"name": "target", "value": "preserved"}]

    assert load_scenario("target", 7, matrix, load_scenarios_fn=fake_load) == {
        "name": "target",
        "value": "preserved",
        "seeds": [7],
    }
    with pytest.raises(ValueError, match="integer"):
        load_scenario("target", True, matrix, load_scenarios_fn=fake_load)

    with pytest.raises(KeyError, match="absent"):
        load_scenario("missing", 7, matrix, load_scenarios_fn=lambda *_args, **_kwargs: [])


@pytest.mark.parametrize(
    ("content", "message"),
    [("\n", "exactly one"), ("{}\n{}\n", "exactly one"), ("[]\n", "JSON object")],
)
def test_read_single_episode_record_rejects_non_single_object(
    tmp_path: Path, content: str, message: str
) -> None:
    path = tmp_path / "episodes.jsonl"
    path.write_text(content, encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        read_single_episode_record(path)


def test_extract_trace_steps_and_json_object_reject_malformed_payloads(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="algorithm_metadata"):
        _extract_trace_steps({})
    with pytest.raises(ValueError, match="planner_decision_trace"):
        _extract_trace_steps({"algorithm_metadata": {}})
    with pytest.raises(ValueError, match="non-empty"):
        _extract_trace_steps({"algorithm_metadata": {"planner_decision_trace": {"steps": []}}})
    with pytest.raises(ValueError, match="contain objects"):
        _extract_trace_steps({"algorithm_metadata": {"planner_decision_trace": {"steps": [None]}}})

    provenance = tmp_path / "provenance.json"
    provenance.write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError, match="JSON object"):
        _read_json_object(provenance)


def test_validate_identity_rejects_mismatches_and_malformed_provenance() -> None:
    request = DwaDiagnosticRequest(
        config_path=Path("algo.yaml"),
        scenario="target",
        seed=7,
        algorithm="dwa",
        output_dir=Path("out"),
    )
    with pytest.raises(ValueError, match="scenario mismatch"):
        _validate_identity(_record(scenario="other"), request, {})
    with pytest.raises(ValueError, match="seed mismatch"):
        _validate_identity(_record(seed=8), request, {})
    with pytest.raises(ValueError, match="rows must be a list"):
        _validate_identity(_record(), request, {"rows": {}})
    with pytest.raises(ValueError, match="contain objects"):
        _validate_identity(_record(), request, {"rows": [None]})


def test_collect_episode_includes_valid_provenance_and_steps_property(tmp_path: Path) -> None:
    result = tmp_path / "existing.jsonl"
    result.write_text(json.dumps(_record()) + "\n", encoding="utf-8")
    provenance = result.with_name(result.name + ".provenance.json")
    provenance.write_text(
        json.dumps({"rows": [{"scenario_id": "target", "seed": 7}]}), encoding="utf-8"
    )
    request = DwaDiagnosticRequest(
        config_path=tmp_path / "algo.yaml",
        scenario="target",
        seed=7,
        algorithm="dwa",
        output_dir=tmp_path / "out",
        existing_result=result,
    )

    episode = collect_episode(request)

    assert episode.steps == episode.trace_steps
    assert episode.source_artifacts["provenance"] == provenance


def test_collect_episode_requires_matrix_and_schema_for_generated_result(tmp_path: Path) -> None:
    request = DwaDiagnosticRequest(
        config_path=tmp_path / "algo.yaml",
        scenario="target",
        seed=7,
        algorithm="dwa",
        output_dir=tmp_path / "out",
    )

    with pytest.raises(ValueError, match="matrix_path"):
        collect_episode(request)


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


@pytest.mark.parametrize("invalid_seed", [True, 7.0, "7"])
def test_collect_episode_rejects_lossy_identity_seed_types(
    tmp_path: Path, invalid_seed: object
) -> None:
    result = tmp_path / "existing.jsonl"
    record = _record()
    record["seed"] = invalid_seed
    result.write_text(json.dumps(record) + "\n", encoding="utf-8")
    request = DwaDiagnosticRequest(
        config_path=tmp_path / "algo.yaml",
        scenario="target",
        seed=7,
        algorithm="dwa",
        output_dir=tmp_path / "out",
        existing_result=result,
    )

    with pytest.raises(ValueError, match="integer"):
        collect_episode(request)


def test_collect_episode_rejects_non_integer_provenance_seed(tmp_path: Path) -> None:
    result = tmp_path / "existing.jsonl"
    result.write_text(json.dumps(_record()) + "\n", encoding="utf-8")
    result.with_name(result.name + ".provenance.json").write_text(
        json.dumps({"rows": [{"scenario_id": "target", "seed": True}]}),
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

    with pytest.raises(ValueError, match="integer"):
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
    with pytest.raises(ValueError, match="target_goal"):
        flatten_trace_step(
            {"selected_command": [0.1, 0.0], "target_goal": "invalid"},
            episode_id="episode",
            scenario_id="target",
            seed=7,
        )


def test_flatten_and_summarize_trace_fields() -> None:
    row = flatten_trace_step(
        {
            "step": 3,
            "selected_command": [0.5, -0.1],
            "selected_score": 2.0,
            "dynamic_window": {"v_min": 0.0},
            "target_goal": {"kind": "goal", "x": 1.0},
        },
        episode_id="episode",
        scenario_id="target",
        seed=7,
        extra_fields={"diagnostic_flag": True},
    )
    assert row["selected_v_mps"] == 0.5
    assert row["selected_w_radps"] == -0.1
    assert row["dynamic_window_v_min"] == 0.0
    assert row["diagnostic_flag"] is True

    rows = [
        {
            "step": 0,
            "candidate_feasible": 2,
            "candidate_infeasible": 0,
            "constraint_reason": "z_reason",
            "distance_to_goal_m": 3.0,
            "route_progress_from_start_m": 0.0,
            "selected_v_mps": 0.5,
            "selected_w_radps": 0.0,
            "selected_score": 1.0,
        },
        {
            "step": 1,
            "candidate_feasible": 0,
            "candidate_infeasible": 2,
            "constraint_reason": "a_reason",
            "distance_to_goal_m": None,
            "route_progress_from_start_m": "",
            "selected_v_mps": 0.0,
            "selected_w_radps": 0.0,
            "selected_score": 2.0,
        },
        {
            "step": 2,
            "candidate_feasible": 1,
            "candidate_infeasible": 1,
            "constraint_reason": "a_reason",
            "distance_to_goal_m": "nan",
            "route_progress_from_start_m": "inf",
            "selected_v_mps": 0.0,
            "selected_w_radps": 0.0,
            "selected_score": 3.0,
        },
        {
            "step": 3,
            "candidate_feasible": 1,
            "candidate_infeasible": 0,
            "constraint_reason": "z_reason",
            "distance_to_goal_m": 2.0,
            "route_progress_from_start_m": 1.0,
            "selected_v_mps": 0.0,
            "selected_w_radps": 0.0,
            "selected_score": 4.0,
        },
    ]
    assert first_unrecoverable_step(rows) == 1
    assert first_infeasible_candidate_step(rows) == 1
    assert constraint_reason_counts(rows) == {"a_reason": 2, "z_reason": 2}
    progress = route_progress_summary(rows)
    assert progress["initial_distance_to_goal_m"] == 3.0
    assert progress["final_distance_to_goal_m"] == 2.0
    assert progress["net_progress_m"] == 1.0
    assert progress["skipped_non_finite_rows"] == 1
    assert progress["skipped_non_finite_cells"] == 2
    assert route_progress_summary([]) == {"status": "no_steps"}

    summary = summarize_episode(
        episode_id="episode",
        record={
            "scenario_id": "target",
            "seed": 7,
            "termination_reason": "max_steps",
            "steps": 4,
            "outcome": {"route_complete": False, "collision_event": False, "timeout_event": True},
        },
        rows=rows,
        extra_fields={"diagnostic_status": "diagnostic-only"},
    )
    assert summary["timeout_event"] is True
    assert summary["first_all_infeasible_step"] == 1
    assert summary["last_selected_command"] == {"v_mps": 0.0, "w_radps": 0.0}
    assert summary["diagnostic_status"] == "diagnostic-only"


def test_write_steps_and_provenance_helpers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    csv_path = tmp_path / "steps.csv"
    write_steps_csv(csv_path, [{"step": 0, "distance": 1.0}], ("step", "distance"))
    assert "distance_convention: center_center" in csv_path.read_text(encoding="utf-8")
    with pytest.raises(ValueError, match="empty"):
        write_steps_csv(csv_path, [], ("step",))

    json_path = tmp_path / "plain.json"
    write_json_atomic(json_path, {"value": 2}, review_marker=False)
    assert json.loads(json_path.read_text(encoding="utf-8")) == {"value": 2}

    in_repo = Path(harness.__file__).resolve()
    assert repo_relative_path(in_repo) == "robot_sf/benchmark/dwa_diagnostic_harness.py"
    assert repo_relative_path(tmp_path / "outside.txt") == str(tmp_path / "outside.txt")
    digest_path = tmp_path / "digest.txt"
    digest_path.write_text("hello", encoding="utf-8")
    assert sha256_file(digest_path) == (
        "2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824"
    )
    assert len(trace_commit()) == 40
    monkeypatch.setattr(
        harness.subprocess,
        "check_output",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(subprocess.CalledProcessError(1, "git")),
    )
    assert trace_commit() == "unknown"


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
