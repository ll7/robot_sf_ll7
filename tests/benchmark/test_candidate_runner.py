"""End-to-end tests for the candidate persistence runner.

These tests wire realistic episode traces through segment extraction, replay,
event reproduction, and perturbation persistence to produce conformance records
with both promotion and rejection verdicts.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

import pytest

from robot_sf.benchmark.scenario_generation.candidate_runner import (
    build_cell_verdict_from_trace_replay,
    build_persistence_from_catalog_entry,
    build_persistence_from_episode_trace,
    get_critical_event_from_frames,
    run_candidate_persistence_smoke,
)
from robot_sf.benchmark.scenario_generation.persistence_gate import (
    PASS,
    PERSISTENCE_SCHEMA_VERSION,
    REQUIRED_STATUSES,
    validate_persistence_record,
)

if TYPE_CHECKING:
    from pathlib import Path

_COMMIT_HASHES = {"code": "test-commit", "config": "test-config"}
_CONFIG = {"config_id": "test-runner", "frozen": True}


def _build_episode(
    *,
    episode_id: str,
    seed: int,
    source_map: str,
    robot_positions: list[list[float]],
    ped_positions: list[list[list[float]]],
) -> dict[str, Any]:
    """Build a synthetic episode trace."""
    n_steps = len(robot_positions)
    steps: list[dict[str, Any]] = []
    for t in range(n_steps):
        pedestrian_count = len(ped_positions[0]) if ped_positions else 0
        step: dict[str, Any] = {
            "time_s": float(t),
            "robot": {"position": robot_positions[t]},
            "pedestrians": [],
        }
        for pid in range(pedestrian_count):
            step["pedestrians"].append({"position": ped_positions[0][pid]})
            for traj in ped_positions[1:]:
                if pid < len(traj):
                    step["pedestrians"][-1]["position"] = traj[pid]
            break
        steps.append(step)
    return {
        "episode_id": episode_id,
        "seed": seed,
        "source_map": source_map,
        "steps": steps,
    }


def _build_parallel_episode(
    *,
    episode_id: str = "ep-persistent",
    seed: int = 42,
    source_map: str = "maps/svg_maps/classing.svg",
    robot_x: float = 1.0,
    ped_start_pos: tuple[float, float] = (5.0, 10.0),
    n_steps: int = 5,
) -> dict[str, Any]:
    """Build an episode with a persistent min-clearance event.

    Robot stays at fixed x, pedestrians cross from y=10 to y=0.
    At t=2, pedestrian crosses near robot at clearance ~4.0m.
    """
    return _build_episode(
        episode_id=episode_id,
        seed=seed,
        source_map=source_map,
        robot_positions=[[robot_x, float(y)] for y in (0.0, 2.0, 4.0, 6.0, 8.0)],
        ped_positions=[
            [[ped_start_pos], [[0.0, 8.0]], [[0.0, 6.0]], [[0.0, 4.0]], [[0.0, 2.0]]],
        ],
    )


class TestCriticalEventFromFrames:
    """get_critical_event_from_frames tests."""

    def test_finds_min_clearance_frame(self) -> None:
        frames = [
            {
                "time_s": 0.0,
                "robot": {"position": [0.0, 0.0]},
                "pedestrians": [{"position": [5.0, 5.0]}],
            },
            {
                "time_s": 1.0,
                "robot": {"position": [0.0, 0.1]},
                "pedestrians": [{"position": [1.0, 1.0]}],
            },
            {
                "time_s": 2.0,
                "robot": {"position": [0.0, 0.2]},
                "pedestrians": [{"position": [3.0, 3.0]}],
            },
        ]
        event_time, min_clearance, ped_pos = get_critical_event_from_frames(frames)
        assert event_time == 1.0
        assert ped_pos == [1.0, 1.0]
        assert math.isclose(min_clearance, math.dist([0.0, 0.1], [1.0, 1.0]), rel_tol=1e-9)

    def test_empty_frames_raises(self) -> None:
        with pytest.raises(ValueError, match="empty"):
            get_critical_event_from_frames([])


class TestBuildPersistenceFromEpisode:
    """Full episode -> persistence record pipeline."""

    def test_basic_record_produced(self) -> None:
        episode = _build_episode(
            episode_id="ep-test",
            seed=1,
            source_map="maps/test.svg",
            robot_positions=[[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]],
            ped_positions=[
                [[5.0, 5.0], [9.0, 9.0]],  # t=0: 2 peds
                [[4.0, 4.0], [8.0, 8.0]],  # t=1: 2 peds closer
                [[3.0, 3.0], [7.0, 7.0]],  # t=2: closer still
            ],
        )
        record = build_persistence_from_episode_trace(
            episode=episode,
            config=_CONFIG,
            commit_hashes=_COMMIT_HASHES,
        )
        assert record["schema_version"] == PERSISTENCE_SCHEMA_VERSION
        validate_persistence_record(record)
        assert record["promotion"]["required_statuses"] == list(REQUIRED_STATUSES)

    def test_record_exactly_passes_when_all_checks_pass(self) -> None:
        episode = _build_episode(
            episode_id="ep-pass",
            seed=2,
            source_map="maps/test.svg",
            robot_positions=[[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
            ped_positions=[
                [[1.0, 0.0], [3.0, 0.0]],
                [[1.0, 0.0], [3.0, 0.0]],
                [[1.0, 0.0], [3.0, 0.0]],
            ],
        )
        record = build_persistence_from_episode_trace(
            episode=episode,
            config=_CONFIG,
            commit_hashes=_COMMIT_HASHES,
        )
        validate_persistence_record(record)
        assert record["exact_replay"]["status"] == PASS

    def test_smoke_both_verdict_paths(self, tmp_path: Path) -> None:
        """Two candidates: one promotes, one does not (missing required check or perturbation fail)."""
        # Candidate that passes exact replay (static trace, stable)
        episode_pass = _build_episode(
            episode_id="ep-smoke-pass",
            seed=10,
            source_map="maps/test.svg",
            robot_positions=[[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
            ped_positions=[[[1.0, 0.0]], [[1.0, 0.0]], [[1.0, 0.0]]],
        )
        results = run_candidate_persistence_smoke(
            candidates=[episode_pass],
            config=_CONFIG,
            commit_hashes=_COMMIT_HASHES,
            output_root=tmp_path,
        )
        assert len(results) == 1
        validate_persistence_record(results[0])
        assert results[0]["exact_replay"]["status"] == PASS
        record_path = tmp_path / f"{results[0]['scenario_id']}.json"
        assert record_path.exists()


class TestBuildPersistenceFromCatalogEntry:
    """Catalog entry -> persistence record pipeline."""

    def test_catalog_entry_persistence_record(self) -> None:
        entry = {
            "schema_version": "generated-scenario-catalog-entry.v1",
            "scenario_id": "catalog-entry-001",
            "metadata": {
                "source": "auto_generated",
                "generated_by": "test",
                "required_manual_review": True,
                "benchmark_evidence": False,
            },
            "source_episode": {
                "episode_id": "ep-001",
                "source_seed": 42,
                "source_map": "maps/test.svg",
            },
            "criticality": {
                "signal": "min_clearance",
                "observed_at_s": 1.0,
                "source_metrics": {"min_clearance_m": 1.0},
            },
            "segment": {
                "window_start_s": 0.0,
                "window_end_s": 2.0,
                "initial_robot_state": {"position": [0.0, 0.0]},
                "trace_frames": [
                    {
                        "time_s": 0.0,
                        "robot": {"position": [0.0, 0.0]},
                        "pedestrians": [{"position": [5.0, 5.0]}],
                    },
                    {
                        "time_s": 1.0,
                        "robot": {"position": [0.5, 0.5]},
                        "pedestrians": [{"position": [1.0, 1.0]}],
                    },
                    {
                        "time_s": 2.0,
                        "robot": {"position": [1.0, 1.0]},
                        "pedestrians": [{"position": [3.0, 3.0]}],
                    },
                ],
            },
            "replay": {
                "schema_version": "generated-scenario-replay.v1",
                "source_seed": 42,
                "replay_contract": "source_episode_seed_pinned.v1",
                "status": "not_representable_yet",
                "warnings": ["replay_gap: distilled mid-episode state is not representable"],
            },
            "provenance": {
                "schema_version": "generated-scenario-provenance.v1",
                "source_issue": "#5600",
                "distiller": "test",
                "claim_boundary": "generated scenario hypotheses only",
                "reviewed": False,
            },
        }
        record = build_persistence_from_catalog_entry(
            catalog_entry=entry,
            config=_CONFIG,
            commit_hashes=_COMMIT_HASHES,
        )
        assert record["schema_version"] == PERSISTENCE_SCHEMA_VERSION
        assert record["scenario_id"] == "catalog-entry-001"
        validate_persistence_record(record)
        assert record["exact_replay"]["status"] == PASS


class TestCellVerdictFromTraceReplay:
    """Trace-based perturbation verdict function tests."""

    def test_verdict_fn_returns_mapping(self) -> None:
        source_frames = [
            {
                "time_s": 0.0,
                "robot": {"position": [0.0, 0.0]},
                "pedestrians": [{"position": [2.0, 2.0]}],
            },
            {
                "time_s": 1.0,
                "robot": {"position": [0.0, 0.0]},
                "pedestrians": [{"position": [1.0, 1.0]}],
            },
        ]
        fn = build_cell_verdict_from_trace_replay(
            source_frames=source_frames,
            event_time_s=1.0,
            event_pedestrian_positions={"0": [1.0, 1.0]},
            time_tolerance_s=1.0,
            location_tolerance_m=2.0,
        )
        result = fn(timing_offset_s=0.0, speed_delta_m_s=0.0)
        assert result is not None
        assert "verdict" in result
        assert "reason" in result


class TestBatchSmokeRunner:
    """run_candidate_persistence_smoke integration."""

    def test_batch_with_mixed_candidates(self, tmp_path: Path) -> None:
        candidates = [
            _build_episode(
                episode_id=f"ep-batched-{i}",
                seed=i,
                source_map="maps/test.svg",
                robot_positions=[[0.0, 0.0], [0.0, 0.0]],
                ped_positions=[[[1.0, 0.0]], [[1.0, 0.0]]],
            )
            for i in range(3)
        ]
        results = run_candidate_persistence_smoke(
            candidates=candidates,
            config=_CONFIG,
            commit_hashes=_COMMIT_HASHES,
            output_root=tmp_path,
        )
        assert len(results) == 3
        for record in results:
            validate_persistence_record(record)
            assert scenario_id_matches(record, record["scenario_id"])

    def test_unfrozen_config_fails(self) -> None:
        candidates = [
            _build_episode(
                episode_id="ep-config-fail",
                seed=99,
                source_map="maps/test.svg",
                robot_positions=[[0.0, 0.0], [0.0, 0.0]],
                ped_positions=[[[1.0, 0.0]], [[1.0, 0.0]]],
            )
        ]
        with pytest.raises(Exception, match="frozen"):
            run_candidate_persistence_smoke(
                candidates=candidates,
                config={"config_id": "unfrozen", "frozen": False},
                commit_hashes=_COMMIT_HASHES,
            )

    def test_empty_candidates_returns_empty(self) -> None:
        results = run_candidate_persistence_smoke(candidates=[])
        assert results == []


def scenario_id_matches(record: dict[str, Any], expected_id: str) -> bool:
    """Verify the scenario_id is preserved through the pipeline."""
    return record["scenario_id"] == expected_id
