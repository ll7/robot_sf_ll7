"""Focused tests for the generic fail-closed trace-series adapter (Issue #5883)."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

import pytest

if TYPE_CHECKING:
    from pathlib import Path

from robot_sf.benchmark import trace_scene_figure as tsf
from scripts.repro.trace_series_adapter import (
    TraceSeriesAdapterError,
    build_bundle,
    main,
    select_row,
)


def _frame(step: int, robot_x: float, ped_x: float, ped_id: int = 0) -> dict[str, Any]:
    return {
        "step": step,
        "time_s": round((step + 1) * 0.1, 3),
        "robot": {
            "position": [robot_x, 0.0],
            "velocity": [1.0, 0.0],
            "heading": 0.0,
        },
        "pedestrians": [{"id": ped_id, "position": [ped_x, 0.0], "velocity": [0.0, 0.0]}],
        "planner": {"selected_action": {"linear_velocity": 0.8, "angular_velocity": 0.0}},
        "rl": {"terminated": False, "truncated": False},
    }


def _trace(
    *, steps: int = 5, collision: bool = False, ped_ids: tuple[int, ...] = (0,)
) -> dict[str, Any]:
    frames = []
    for step in range(steps):
        ped_x = 4.5 if not collision else (3.0 if step == steps - 1 else 4.5)
        frames.append(_frame(step, float(step), float(ped_x), ped_id=ped_ids[0]))
    return {"schema_version": "simulation-step-trace.v1", "dt": 0.1, "steps": frames}


def _episode(
    arm: str,
    seed: int,
    outcome: str,
    *,
    opts: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build one synthetic benchmark episode row.

    ``opts`` may override: episode_id, scenario, git_commit, trace_steps,
    collision, ped_ids.
    """
    opts = opts or {}
    episode_id = opts.get("episode_id")
    scenario = opts.get("scenario", "classic_doorway_medium")
    git_commit = opts.get("git_commit", "abc123")
    trace_steps = opts.get("trace_steps", 5)
    collision = opts.get("collision", False)
    ped_ids = opts.get("ped_ids", (0,))
    metadata: dict[str, Any] = {"algorithm": arm}
    metadata["simulation_step_trace"] = _trace(
        steps=trace_steps, collision=collision, ped_ids=ped_ids
    )
    return {
        "algo": arm,
        "arm": arm,
        "episode_id": episode_id or f"{scenario}--{arm}--{seed}",
        "scenario_id": scenario,
        "seed": seed,
        "status": outcome,
        "termination_reason": outcome,
        "steps": trace_steps,
        "git_hash": git_commit,
        "result_provenance": {
            "repo_commit": git_commit,
            "scenario_id": scenario,
            "seed": seed,
            "config_hash": f"{arm}-config",
            "simulator_settings": {"dt": 0.1, "horizon": 600},
        },
        "algorithm_metadata": metadata,
        "metrics": {"near_misses": 3, "min_clearance": 0.2},
    }


def _write(rows: list[dict[str, Any]], tmp_path: Path, name: str = "episodes.jsonl") -> Path:
    path = tmp_path / name
    path.write_text("".join(f"{json.dumps(row)}\n" for row in rows), encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# Selection
# ---------------------------------------------------------------------------


def test_select_by_episode_id_is_unique(tmp_path: Path) -> None:
    rows = [
        _episode("ppo", 113, "success", opts={"episode_id": "case-1"}),
        _episode("ppo", 114, "collision", opts={"episode_id": "case-2"}),
    ]
    path = _write(rows, tmp_path)
    row = select_row(path, episode_id="case-2")
    assert row["seed"] == 114


def test_select_by_scenario_planner_seed_triple(tmp_path: Path) -> None:
    rows = [
        _episode("ppo", 113, "success"),
        _episode("orca", 113, "collision"),
    ]
    path = _write(rows, tmp_path)
    row = select_row(path, scenario="classic_doorway_medium", planner="orca", seed=113)
    assert row["algo"] == "orca"


def test_select_ambiguous_same_seed_fails_closed(tmp_path: Path) -> None:
    rows = [
        _episode("orca", 113, "success", opts={"scenario": "scene_a"}),
        _episode("orca", 113, "collision", opts={"scenario": "scene_b"}),
    ]
    path = _write(rows, tmp_path)
    with pytest.raises(TraceSeriesAdapterError, match="ambiguous"):
        select_row(path, scenario="classic_doorway_medium", planner="orca", seed=113)


def test_select_no_match_fails_closed(tmp_path: Path) -> None:
    path = _write([_episode("ppo", 113, "success")], tmp_path)
    with pytest.raises(TraceSeriesAdapterError, match="no episode"):
        select_row(path, scenario="nope", planner="ppo", seed=113)


def test_select_mutually_exclusive_selectors(tmp_path: Path) -> None:
    path = _write([_episode("ppo", 113, "success")], tmp_path)
    with pytest.raises(TraceSeriesAdapterError, match="mutually exclusive"):
        select_row(path, episode_id="x", scenario="y", planner="z", seed=1)


# ---------------------------------------------------------------------------
# Fail-closed conversions
# ---------------------------------------------------------------------------


def test_unknown_trace_schema_fails_closed(tmp_path: Path) -> None:
    row = _episode("ppo", 113, "success")
    row["algorithm_metadata"]["simulation_step_trace"]["schema_version"] = "v2"
    path = _write([row], tmp_path)
    with pytest.raises(TraceSeriesAdapterError, match="unsupported"):
        build_bundle(
            path, tmp_path / "out", scenario="classic_doorway_medium", planner="ppo", seed=113
        )


def test_missing_provenance_fails_closed(tmp_path: Path) -> None:
    row = _episode("ppo", 113, "success")
    row.pop("git_hash")
    row.pop("result_provenance")
    path = _write([row], tmp_path)
    with pytest.raises(TraceSeriesAdapterError, match="provenance"):
        build_bundle(
            path, tmp_path / "out", scenario="classic_doorway_medium", planner="ppo", seed=113
        )


def test_actor_set_change_fails_closed(tmp_path: Path) -> None:
    row = _episode("ppo", 113, "collision", opts={"trace_steps": 3, "ped_ids": (0,)})
    trace = row["algorithm_metadata"]["simulation_step_trace"]
    trace["steps"][-1]["pedestrians"].append(
        {"id": 1, "position": [2.0, 0.0], "velocity": [0.0, 0.0]}
    )
    path = _write([row], tmp_path)
    with pytest.raises(TraceSeriesAdapterError, match="actor-set"):
        build_bundle(
            path, tmp_path / "out", scenario="classic_doorway_medium", planner="ppo", seed=113
        )


def test_zero_pedestrian_frame_fails_closed(tmp_path: Path) -> None:
    row = _episode("ppo", 113, "collision", opts={"trace_steps": 1, "ped_ids": (0,)})
    trace = row["algorithm_metadata"]["simulation_step_trace"]
    trace["steps"][0]["pedestrians"] = []
    path = _write([row], tmp_path)
    with pytest.raises(TraceSeriesAdapterError, match="zero pedestrians"):
        build_bundle(
            path, tmp_path / "out", scenario="classic_doorway_medium", planner="ppo", seed=113
        )


def test_malformed_frame_fails_closed_before_writing(tmp_path: Path) -> None:
    row = _episode("ppo", 113, "collision")
    row["algorithm_metadata"]["simulation_step_trace"]["steps"][0]["robot"]["position"] = [
        "not-a-number",
        0.0,
    ]
    path = _write([row], tmp_path)
    with pytest.raises(TraceSeriesAdapterError, match="finite number"):
        build_bundle(
            path, tmp_path / "out", scenario="classic_doorway_medium", planner="ppo", seed=113
        )
    assert not (tmp_path / "out" / "trace_series.json").exists()


def test_cli_fails_closed_with_actionable_error(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    row = _episode("ppo", 113, "success")
    row["algorithm_metadata"]["simulation_step_trace"]["schema_version"] = "v2"
    path = _write([row], tmp_path)
    result = main(
        [
            "build-bundle",
            "--episodes-jsonl",
            str(path),
            "--scenario",
            "classic_doorway_medium",
            "--planner",
            "ppo",
            "--seed",
            "113",
            "--out-dir",
            str(tmp_path / "out"),
        ]
    )
    assert result == 1
    assert "failed closed" in capsys.readouterr().err


# ---------------------------------------------------------------------------
# Deterministic bundles from synthetic fixtures
# ---------------------------------------------------------------------------


def test_same_seed_flip_produces_deterministic_bundle(tmp_path: Path) -> None:
    rows = [
        _episode("ppo", 113, "success", opts={"episode_id": "flip-a"}),
        _episode("ppo", 113, "collision", opts={"episode_id": "flip-b", "collision": True}),
    ]
    path = _write(rows, tmp_path)
    out = tmp_path / "out"
    for eid in ("flip-a", "flip-b"):
        summary = build_bundle(path, out / eid, episode_id=eid)
        assert summary["n_steps"] == 5
        assert (out / eid / "trace_series.json").is_file()
        assert (out / eid / "metadata.json").is_file()


def test_cross_planner_same_seed_upset_produces_bundle(tmp_path: Path) -> None:
    rows = [
        _episode("ppo", 113, "success", opts={"episode_id": "up-a"}),
        _episode("orca", 113, "collision", opts={"episode_id": "up-b", "collision": True}),
    ]
    path = _write(rows, tmp_path)
    summary = build_bundle(path, tmp_path / "out" / "orca", episode_id="up-b")
    assert summary["planner"] == "orca"
    assert summary["episode_status"] == "collision"


def test_bundle_is_deterministic_and_preserves_source_provenance(tmp_path: Path) -> None:
    row = _episode("orca", 221, "collision", opts={"episode_id": "deterministic"})
    path = _write([row], tmp_path)
    out = tmp_path / "out" / "deterministic"
    build_bundle(path, out, episode_id="deterministic")
    first_trace = (out / "trace_series.json").read_bytes()
    first_metadata = (out / "metadata.json").read_bytes()
    build_bundle(path, out, episode_id="deterministic")
    assert (out / "trace_series.json").read_bytes() == first_trace
    assert (out / "metadata.json").read_bytes() == first_metadata
    metadata = json.loads(first_metadata)
    assert metadata["source_provenance"]["config_hash"] == "orca-config"
    assert metadata["source_provenance"]["simulator_settings"]["dt"] == 0.1


# ---------------------------------------------------------------------------
# Real-path renderer handoff (production serializer/renderer contract)
# ---------------------------------------------------------------------------


def test_bundle_loads_through_production_renderer(tmp_path: Path) -> None:
    row = _episode("ppo", 113, "success", opts={"episode_id": "real-1"})
    path = _write([row], tmp_path)
    out = tmp_path / "out" / "real-1"
    build_bundle(path, out, episode_id="real-1")
    episode = tsf.load_episode(out)  # must not raise TraceSchemaError
    assert episode.metadata["planner"] == "ppo"
    assert len(episode.steps) == 5
    assert len(episode.pedestrian_tracks) == 1
    assert all(d is not None for d in episode.min_robot_ped_distance_m)
    assert episode.metadata["summary"]["step_count"] == 5


def test_bundle_metadata_satisfies_renderer_contract(tmp_path: Path) -> None:
    row = _episode("orca", 221, "collision", opts={"episode_id": "real-2", "collision": True})
    path = _write([row], tmp_path)
    out = tmp_path / "out" / "real-2"
    build_bundle(path, out, episode_id="real-2")
    metadata = json.loads((out / "metadata.json").read_text(encoding="utf-8"))
    for key in ("planner", "scenario_id", "seed", "episode_status", "summary"):
        assert key in metadata
    summary = metadata["summary"]
    for key in (
        "global_min_robot_ped_distance_m",
        "global_min_distance_step",
        "step_count",
        "termination_reason",
    ):
        assert key in summary
    assert metadata["summary"]["termination_reason"] == "collision"
