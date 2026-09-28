"""Focused contracts for the issue #9759 mirror diagnostic runner."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from robot_sf.benchmark import map_runner_episode
from robot_sf.benchmark.release_map_mirror import reflect_map_definition
from robot_sf.nav.global_route import GlobalRoute
from robot_sf.nav.map_config import MapDefinition, MapDefinitionPool
from robot_sf.nav.obstacle import Obstacle
from scripts.benchmark import issue_9759_hybrid_mirror_diagnostic as runner


def _map_definition() -> MapDefinition:
    """Return a small valid post-loader map for builder-scope tests."""

    width = height = 10.0
    route = GlobalRoute(
        spawn_id=0,
        goal_id=0,
        waypoints=[(1.0, 2.0), (8.0, 2.0)],
        spawn_zone=((1.0, 2.0), (1.2, 2.0), (1.0, 2.2)),
        goal_zone=((8.0, 2.0), (8.2, 2.0), (8.0, 2.2)),
    )
    return MapDefinition(
        width=width,
        height=height,
        obstacles=[Obstacle([(4.0, 4.0), (5.0, 4.0), (5.0, 5.0), (4.0, 5.0)])],
        robot_spawn_zones=[route.spawn_zone],
        ped_spawn_zones=[],
        robot_goal_zones=[route.goal_zone],
        bounds=[
            ((0.0, 0.0), (width, 0.0)),
            ((width, 0.0), (width, height)),
            ((width, height), (0.0, height)),
            ((0.0, height), (0.0, 0.0)),
        ],
        robot_routes=[route],
        ped_goal_zones=[],
        ped_crowded_zones=[],
        ped_routes=[],
        single_pedestrians=[],
    )


def _episode_job(
    *,
    axis: runner.EpisodeAxis = "base",
    preflight_status: str = "ok",
    preflight_error: str | None = None,
) -> runner.EpisodeJob:
    return runner.EpisodeJob(
        scenario={"name": "fixture"},
        scenario_id="fixture",
        seed=111,
        arm="v4",
        config_path="configs/v4.yaml",
        config_digest="c" * 64,
        axis=axis,
        map_id="fixture-map",
        map_digest="a" * 64,
        source_map_digest="b" * 64,
        preflight_status=preflight_status,
        preflight_error=preflight_error,
        scenario_matrix_path="scenarios.yaml",
        horizon=600,
        input_digest="d" * 64,
    )


def test_release_inputs_resolve_manifest_relative_matrix() -> None:
    """The release manifest resolves the canonical 48-row matrix and seed set."""

    inputs = runner.load_release_inputs(runner.DEFAULT_MANIFEST)

    assert (
        inputs.scenario_matrix_path
        == (
            runner.DEFAULT_MANIFEST.parent / "../../scenarios/classic_interactions_francis2023.yaml"
        ).resolve()
    )
    assert len(inputs.scenarios) == 48
    assert inputs.seeds == tuple(range(111, 141))
    assert inputs.horizon == 600
    assert len(inputs.input_digest) == 64


def test_release_input_digest_is_bound_to_producing_head(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Episode resume provenance changes when the producing source revision changes."""
    original = runner.load_release_inputs()
    original_git_value = runner._git_value

    def changed_head(*arguments: str, repo_root: Path = runner._REPO_ROOT) -> str:
        if arguments == ("rev-parse", "HEAD"):
            return "f" * 40
        return original_git_value(*arguments, repo_root=repo_root)

    monkeypatch.setattr(runner, "_git_value", changed_head)
    changed = runner.load_release_inputs()

    assert changed.input_digest != original.input_digest


def test_scoped_reflected_builder_replaces_only_post_loader_map(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The private builder override transforms one map and always restores itself."""

    original_map = _map_definition()
    config = SimpleNamespace(
        map_id="fixture",
        map_pool=MapDefinitionPool(map_defs={"fixture": original_map}),
    )
    calls: list[list[dict[str, str]] | None] = []

    def fake_builder(
        scenario: dict[str, object],
        *,
        scenario_path: Path,
        runtime_input_records: list[dict[str, str]] | None = None,
    ) -> SimpleNamespace:
        del scenario, scenario_path
        calls.append(runtime_input_records)
        return SimpleNamespace(
            map_id=config.map_id,
            map_pool=MapDefinitionPool(map_defs={"fixture": original_map}),
        )

    monkeypatch.setattr(map_runner_episode, "_build_env_config", fake_builder)
    original_builder = map_runner_episode._build_env_config
    source_digests: list[str] = []
    with runner.scoped_reflected_builder(axis="y", captured_source_map_digests=source_digests):
        records: list[dict[str, str]] = []
        transformed = map_runner_episode._build_env_config(
            {},
            scenario_path=Path("fixture.yaml"),
            runtime_input_records=records,
        )
        expected = reflect_map_definition(original_map, "y")
        assert transformed.map_pool.map_defs["fixture"].robot_routes[0].waypoints == (
            expected.robot_routes[0].waypoints
        )
        assert calls == [records]
    assert map_runner_episode._build_env_config is original_builder
    assert source_digests == [runner._map_digest(original_map)]


def test_resume_digest_is_fail_closed(tmp_path: Path) -> None:
    """Resume rows from another input selection cannot be silently reused."""

    path = tmp_path / "episodes.jsonl"
    row = {
        "schema_version": runner.SCHEMA_VERSION,
        "input_digest": "a" * 64,
        "scenario_id": "fixture",
        "seed": 111,
        "arm": "v3",
        "axis": "base",
        "status": "ok",
        "config_sha256": "c" * 64,
        "source_map_sha256": "b" * 64,
        "map_sha256": "a" * 64,
        "preflight_source_map_sha256": "b" * 64,
        "preflight_map_sha256": "a" * 64,
        "preflight_status": "ok",
        "preflight_error": None,
    }
    path.write_text(runner._canonical_json(row) + "\n", encoding="utf-8")

    identity = ("fixture", 111, "v3", "base")
    expected_provenance = {
        identity: {
            "config_sha256": "c" * 64,
            "preflight_source_map_sha256": "b" * 64,
            "preflight_map_sha256": "a" * 64,
            "preflight_status": "ok",
            "preflight_error": None,
        }
    }
    loaded = runner.load_resume_records(
        path,
        input_digest="a" * 64,
        expected_provenance=expected_provenance,
    )
    assert identity in loaded
    stale_preflight = dict(expected_provenance[identity])
    stale_preflight["preflight_status"] = "failed"
    stale_preflight["preflight_error"] = "reset spawn overlap"
    with pytest.raises(ValueError, match="preflight_status mismatch"):
        runner.load_resume_records(
            path,
            input_digest="a" * 64,
            expected_provenance={identity: stale_preflight},
        )
    with pytest.raises(ValueError, match="digest mismatch"):
        runner.load_resume_records(path, input_digest="b" * 64)
    with pytest.raises(ValueError, match="schema mismatch"):
        stale = dict(row)
        stale["schema_version"] = "issue_9759_hybrid_mirror_diagnostic.v1"
        path.write_text(runner._canonical_json(stale) + "\n", encoding="utf-8")
        runner.load_resume_records(path, input_digest="a" * 64)


def test_summary_reports_pair_agreement_and_step_delta() -> None:
    """Pair summaries retain outcome agreement and reflected step differences."""

    records = []
    for axis, steps, route_complete in (
        ("base", 10, True),
        ("x", 12, True),
        ("y", 9, False),
    ):
        records.append(
            {
                "scenario_id": "fixture",
                "seed": 111,
                "arm": "v4",
                "axis": axis,
                "status": "success",
                "termination_reason": "success",
                "preflight_status": "ok",
                "preflight_error": None,
                "steps": steps,
                "config_sha256": "c" * 64,
                "source_map_sha256": "b" * 64,
                "map_sha256": "a" * 64,
                "preflight_source_map_sha256": "b" * 64,
                "preflight_map_sha256": "a" * 64,
                "outcome": {
                    "route_complete": route_complete,
                    "collision_event": False,
                    "timeout_event": not route_complete,
                },
            }
        )

    summary = runner.summarize_records(records, expected_jobs=3, input_digest="c" * 64)

    assert summary["evidence_status"] == "diagnostic-only"
    assert summary["coverage"]["status"] == "complete"
    axis_summary = summary["paired"]["by_arm_axis"]["v4"]
    assert axis_summary["x"]["outcome_agreement"] == 1
    assert axis_summary["x"]["step_delta_min"] == 2
    assert axis_summary["y"]["outcome_disagreement"] == 1
    assert axis_summary["y"]["step_delta_min"] == -1


def test_missing_rows_retain_preflight_error_and_digest_provenance() -> None:
    job = _episode_job(
        axis="x",
        preflight_status="failed",
        preflight_error="ValueError: reflected bounds exceed map limits",
    )

    row = runner._missing_record(job, job.preflight_error or "missing")

    assert row["status"] == "excluded"
    assert row["error"] == "ValueError: reflected bounds exceed map limits"
    assert row["preflight_error"] == row["error"]
    assert row["config_sha256"] == "c" * 64
    assert row["source_map_sha256"] is None
    assert row["map_sha256"] is None
    assert row["preflight_source_map_sha256"] == "b" * 64
    assert row["preflight_map_sha256"] == "a" * 64


def test_compact_record_keeps_actual_map_digests_and_canonical_failure() -> None:
    job = _episode_job()
    record = {
        "status": "failure",
        "termination_reason": "max_steps",
        "steps": 600,
        "outcome": {
            "route_complete": False,
            "collision_event": False,
            "timeout_event": True,
        },
        "integrity": {"contradictions": ["canonical contradiction"]},
    }

    compact = runner._compact_record(
        job,
        record,
        0.1,
        map_digest="e" * 64,
        source_map_digest="f" * 64,
    )

    assert compact["status"] == "max_steps"
    assert compact["map_sha256"] == "e" * 64
    assert compact["source_map_sha256"] == "f" * 64
    assert compact["preflight_map_sha256"] == "a" * 64
    assert compact["preflight_source_map_sha256"] == "b" * 64
    assert compact["error"] == "canonical contradiction"

    failure = runner._failure_record(
        job,
        RuntimeError("episode failed after map load"),
        map_digest="e" * 64,
        source_map_digest="f" * 64,
    )
    assert failure["map_sha256"] == "e" * 64
    assert failure["source_map_sha256"] == "f" * 64
    assert failure["preflight_map_sha256"] == "a" * 64
    assert failure["preflight_source_map_sha256"] == "b" * 64
    assert "episode failed after map load" in failure["error"]


def test_summary_is_partial_for_failed_missing_excluded_and_unknown_rows() -> None:
    def row(status: str, *, error: str | None = None) -> dict[str, object]:
        return {
            "schema_version": runner.SCHEMA_VERSION,
            "scenario_id": status,
            "seed": 111,
            "arm": "v4",
            "axis": "base",
            "status": status,
            "termination_reason": status if status in runner.VALID_TERMINAL_STATUSES else "error",
            "preflight_status": "ok",
            "preflight_error": None,
            "steps": 1,
            "config_sha256": "c" * 64,
            "source_map_sha256": "b" * 64,
            "map_sha256": "a" * 64,
            "preflight_source_map_sha256": "b" * 64,
            "preflight_map_sha256": "a" * 64,
            "error": error,
            "outcome": {
                "route_complete": False,
                "collision_event": False,
                "timeout_event": True,
            },
        }

    summary = runner.summarize_records(
        [
            row("success"),
            row("failed", error="worker failed"),
            row("missing", error="not produced"),
            row("unknown"),
        ],
        expected_jobs=4,
        eligible_jobs=3,
        excluded_jobs=1,
        input_digest="d" * 64,
    )

    coverage = summary["coverage"]
    assert coverage["status"] == "partial"
    assert coverage["valid_terminal_jobs"] == 1
    assert coverage["incomplete_jobs"] == 3
    assert coverage["excluded_jobs"] == 1


def test_unknown_status_or_error_is_not_comparable() -> None:
    def terminal(
        axis: str, *, status: str = "success", error: str | None = None
    ) -> dict[str, object]:
        return {
            "scenario_id": "fixture",
            "seed": 111,
            "arm": "v4",
            "axis": axis,
            "status": status,
            "termination_reason": "success",
            "preflight_status": "ok",
            "preflight_error": None,
            "steps": 10,
            "config_sha256": "c" * 64,
            "source_map_sha256": "b" * 64,
            "map_sha256": "a" * 64,
            "preflight_source_map_sha256": "b" * 64,
            "preflight_map_sha256": "a" * 64,
            "error": error,
            "outcome": {
                "route_complete": True,
                "collision_event": False,
                "timeout_event": False,
            },
        }

    pairs = runner._paired_rows(
        [terminal("base"), terminal("x", status="mystery"), terminal("y", error="late error")]
    )
    assert pairs[0]["pairs"]["x"]["status"] == "incomplete"
    assert pairs[0]["pairs"]["x"]["outcome_agreement"] is None
    assert pairs[0]["pairs"]["x"]["step_delta"] is None
    assert pairs[0]["pairs"]["y"]["status"] == "incomplete"


def test_terminal_validity_rejects_malformed_status_and_raw_outcome() -> None:
    job = _episode_job()
    malformed = {
        "status": "success",
        "termination_reason": "max_steps",
        "preflight_status": "ok",
        "preflight_error": None,
        "steps": 1,
        "config_sha256": job.config_digest,
        "source_map_sha256": job.source_map_digest,
        "map_sha256": job.map_digest,
        "outcome": {"route_complete": 1, "collision_event": False, "timeout_event": False},
    }
    assert runner._record_is_terminal_valid(malformed) is False


def test_preflight_transform_failure_retains_source_digest(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _map_definition()
    config = SimpleNamespace(
        map_id="fixture",
        map_pool=MapDefinitionPool(map_defs={"fixture": source}),
    )
    monkeypatch.setattr(map_runner_episode, "_build_env_config", lambda *_args, **_kwargs: config)
    monkeypatch.setattr(
        runner,
        "reflect_map_definition",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("unsupported geometry")),
    )
    inputs = runner.ReleaseInputs(
        manifest_path=Path("manifest.yaml"),
        scenario_matrix_path=Path("matrix.yaml"),
        scenarios=({"name": "fixture"},),
        seeds=(111,),
        horizon=600,
        manifest_digest="a" * 64,
        scenario_matrix_digest="b" * 64,
        input_digest="d" * 64,
    )

    rows = runner.preflight_maps(inputs, scenarios=inputs.scenarios, axes=("x",), seed=111)

    assert rows[0]["status"] == "failed"
    assert rows[0]["source_map_digest"] == runner._map_digest(source)
    assert "unsupported geometry" in rows[0]["error"]


def test_output_path_is_confined_to_common_git_diagnostic_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The runner refuses worktree-local or arbitrary output custody."""

    allowed = tmp_path / "common-git" / "codex-agent-runs" / "active" / "issue-9759" / "diagnostic"
    monkeypatch.setattr(runner, "default_output_dir", lambda _repo_root: allowed)

    accepted = runner.validate_output_dir(allowed / "slice", repo_root=tmp_path)
    assert accepted.is_dir()
    with pytest.raises(ValueError, match="common Git directory"):
        runner.validate_output_dir(tmp_path / "worktree-output", repo_root=tmp_path)
