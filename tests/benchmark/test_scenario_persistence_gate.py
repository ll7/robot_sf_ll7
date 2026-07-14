"""CPU-only tests for issue #5600's persistence promotion gate."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

import pytest

if TYPE_CHECKING:
    from collections.abc import Mapping
    from pathlib import Path

from robot_sf.benchmark.scenario_generation.persistence_gate import (
    FAIL,
    PASS,
    PERSISTENCE_SCHEMA_VERSION,
    ScenarioPersistenceValidationError,
    assess_critical_event_reproduction,
    assess_exact_replay,
    compute_persistence_record,
    evaluate_perturbation_grid,
    validate_persistence_record,
)

_SOURCE_EPISODE = {
    "episode_id": "ep-4932",
    "source_seed": 4932,
    "source_map": "maps/svg_maps/classic_crossing.svg",
    "replay_digest": "abc",
}
_GENERATED = {
    "catalog_schema_version": "generated-scenario-catalog-entry.v1",
    "scenario_id": "generated-0001",
    "catalog_entry_digest": "def",
}
_CONFIG = {"config_id": "issue-5600-persistence-gate", "frozen": True, "config_hash": "cfg1"}
_COMMITS = {"code": "deadbee", "config": "c0ffee"}


def _grid_block(
    *,
    timing_offsets_s: list[float],
    speed_deltas_m_s: list[float],
    cell_results: list[Mapping[str, Any]],
    missing_cell_reasons: list[Mapping[str, Any]] | None = None,
) -> tuple[Mapping[str, Any], list[Mapping[str, Any]], list[Mapping[str, Any]]]:
    grid = {"timing_offsets_s": timing_offsets_s, "speed_deltas_m_s": speed_deltas_m_s}

    def verdict_fn(*args: Any, **kwargs: Any) -> Mapping[str, Any] | None:
        return None if not cell_results else dict(cell_results.pop(0))

    cells, missing = evaluate_perturbation_grid(
        timing_offsets_s=timing_offsets_s,
        speed_deltas_m_s=speed_deltas_m_s,
        cell_verdict_fn=verdict_fn,
    )
    missing.extend(missing_cell_reasons or [])
    return grid, cells, missing


def _all_pass_persistence() -> dict[str, Any]:
    grid, cells, missing = _grid_block(
        timing_offsets_s=[-0.25, 0.0, 0.25],
        speed_deltas_m_s=[-0.2, 0.0, 0.2],
        cell_results=[{"verdict": PASS, "reason": "event reproduced"} for _ in range(9)],
    )
    return compute_persistence_record(
        scenario_id="generated-0001",
        source_episode=_SOURCE_EPISODE,
        generated_scenario=_GENERATED,
        planner="goal",
        seed=4932,
        config=_CONFIG,
        commit_hashes=_COMMITS,
        exact_replay=assess_exact_replay(
            _SOURCE_EPISODE,
            replayed_episode=dict(_SOURCE_EPISODE),
        ),
        critical_event_reproduced=assess_critical_event_reproduction(
            event_type="min_clearance",
            source_event_time_s=1.0,
            source_event_location=[2.0, 0.0],
            replayed_event_time_s=1.0,
            replayed_event_location=[2.0, 0.0],
            time_tolerance_s=0.5,
            location_tolerance_m=0.75,
        ),
        perturbation_grid=grid,
        cell_verdicts=cells,
        missing_cell_reasons=missing,
    )


def test_schema_version_constant_is_stable() -> None:
    assert PERSISTENCE_SCHEMA_VERSION == "generated_scenario_persistence.v1"


def test_positive_promotion_path() -> None:
    """All three independent checks pass -> promote."""

    record = _all_pass_persistence()
    assert record["exact_replay"]["status"] == PASS
    assert record["critical_event_reproduced"]["status"] == PASS
    assert record["perturbation_persistence"]["persistence_rate"] == 1.0
    assert record["promotion"]["verdict"] == "promote"
    assert record["promotion"]["exclusion_reason"] == "all three independent status checks passed"
    validate_persistence_record(record)


def test_replay_failure_blocks_promotion() -> None:
    grid, cells, missing = _grid_block(
        timing_offsets_s=[-0.25, 0.0, 0.25],
        speed_deltas_m_s=[-0.2, 0.0, 0.2],
        cell_results=[{"verdict": PASS, "reason": "ok"} for _ in range(9)],
    )
    record = compute_persistence_record(
        scenario_id="generated-0002",
        source_episode=_SOURCE_EPISODE,
        generated_scenario=_GENERATED,
        planner="goal",
        seed=4932,
        config=_CONFIG,
        commit_hashes=_COMMITS,
        exact_replay=assess_exact_replay(_SOURCE_EPISODE, replay_error="replay crashed"),
        critical_event_reproduced=assess_critical_event_reproduction(
            event_type="min_clearance",
            source_event_time_s=1.0,
            source_event_location=[2.0, 0.0],
            replayed_event_time_s=1.0,
            replayed_event_location=[2.0, 0.0],
            time_tolerance_s=0.5,
            location_tolerance_m=0.75,
        ),
        perturbation_grid=grid,
        cell_verdicts=cells,
        missing_cell_reasons=missing,
    )
    assert record["exact_replay"]["status"] == FAIL
    assert record["promotion"]["verdict"] == "reject"
    assert record["promotion"]["exclusion_reason"].startswith("exact_replay:fail")
    validate_persistence_record(record)


def test_replay_divergence_blocks_promotion() -> None:
    """A digest mismatch is a concrete divergence fail, not an unknown."""

    replayed = {**_SOURCE_EPISODE, "source_seed": 9999}
    block = assess_exact_replay(_SOURCE_EPISODE, replayed_episode=replayed)
    assert block["status"] == FAIL
    assert "digest mismatch" in block["divergence_reason"]


def test_critical_event_non_reproduction_blocks_promotion() -> None:
    """A source event that does not recur under the source planner fails closed."""

    grid, cells, missing = _grid_block(
        timing_offsets_s=[-0.25, 0.0, 0.25],
        speed_deltas_m_s=[-0.2, 0.0, 0.2],
        cell_results=[{"verdict": PASS, "reason": "ok"} for _ in range(9)],
    )
    record = compute_persistence_record(
        scenario_id="generated-0003",
        source_episode=_SOURCE_EPISODE,
        generated_scenario=_GENERATED,
        planner="goal",
        seed=4932,
        config=_CONFIG,
        commit_hashes=_COMMITS,
        exact_replay=assess_exact_replay(_SOURCE_EPISODE, replayed_episode=dict(_SOURCE_EPISODE)),
        critical_event_reproduced=assess_critical_event_reproduction(
            event_type="min_clearance",
            source_event_time_s=1.0,
            source_event_location=[2.0, 0.0],
            replayed_event_time_s=2.5,
            replayed_event_location=[5.0, 0.0],
            time_tolerance_s=0.5,
            location_tolerance_m=0.75,
        ),
        perturbation_grid=grid,
        cell_verdicts=cells,
        missing_cell_reasons=missing,
    )
    assert record["critical_event_reproduced"]["status"] == FAIL
    assert record["promotion"]["verdict"] == "reject"
    assert "critical_event_reproduced:fail" in record["promotion"]["exclusion_reason"]


def test_deliberately_non_persistent_candidate_rejected() -> None:
    """A candidate whose event dies under perturbation fails the grid."""

    grid, cells, missing = _grid_block(
        timing_offsets_s=[-0.25, 0.0, 0.25],
        speed_deltas_m_s=[-0.2, 0.0, 0.2],
        cell_results=(
            [{"verdict": PASS, "reason": "ok"} for _ in range(6)]
            + [{"verdict": FAIL, "reason": "event not reproduced"} for _ in range(3)]
        ),
    )
    record = compute_persistence_record(
        scenario_id="generated-0004",
        source_episode=_SOURCE_EPISODE,
        generated_scenario=_GENERATED,
        planner="goal",
        seed=4932,
        config=_CONFIG,
        commit_hashes=_COMMITS,
        exact_replay=assess_exact_replay(_SOURCE_EPISODE, replayed_episode=dict(_SOURCE_EPISODE)),
        critical_event_reproduced=assess_critical_event_reproduction(
            event_type="min_clearance",
            source_event_time_s=1.0,
            source_event_location=[2.0, 0.0],
            replayed_event_time_s=1.0,
            replayed_event_location=[2.0, 0.0],
            time_tolerance_s=0.5,
            location_tolerance_m=0.75,
        ),
        perturbation_grid=grid,
        cell_verdicts=cells,
        missing_cell_reasons=missing,
    )
    assert record["perturbation_persistence"]["persistence_rate"] == pytest.approx(6 / 9)
    assert record["promotion"]["verdict"] == "reject"
    assert "perturbation_cell:" in record["promotion"]["exclusion_reason"]


def test_missing_cell_reasons_block_promotion() -> None:
    """An un-evaluable grid cell is recorded and fails closed."""

    grid = {"timing_offsets_s": [-0.25, 0.0], "speed_deltas_m_s": [0.0, 0.2]}

    def verdict_fn(*_: Any, **__: Any) -> Mapping[str, Any] | None:
        return None

    cells, missing = evaluate_perturbation_grid(
        timing_offsets_s=grid["timing_offsets_s"],
        speed_deltas_m_s=grid["speed_deltas_m_s"],
        cell_verdict_fn=verdict_fn,
    )
    assert len(cells) == 0
    assert len(missing) == 4

    record = compute_persistence_record(
        scenario_id="generated-0005",
        source_episode=_SOURCE_EPISODE,
        generated_scenario=_GENERATED,
        planner="goal",
        seed=4932,
        config=_CONFIG,
        commit_hashes=_COMMITS,
        exact_replay=assess_exact_replay(_SOURCE_EPISODE, replayed_episode=dict(_SOURCE_EPISODE)),
        critical_event_reproduced=assess_critical_event_reproduction(
            event_type="min_clearance",
            source_event_time_s=1.0,
            source_event_location=[2.0, 0.0],
            replayed_event_time_s=1.0,
            replayed_event_location=[2.0, 0.0],
            time_tolerance_s=0.5,
            location_tolerance_m=0.75,
        ),
        perturbation_grid=grid,
        cell_verdicts=cells,
        missing_cell_reasons=missing,
    )
    assert record["promotion"]["verdict"] == "reject"
    assert "missing_cells:4" in record["promotion"]["exclusion_reason"]


def test_unknown_critical_event_blocks_promotion() -> None:
    """A not-observed critical event is unknown and must not promote."""

    grid, cells, missing = _grid_block(
        timing_offsets_s=[0.0],
        speed_deltas_m_s=[0.0],
        cell_results=[
            {"verdict": PASS, "reason": "ok", "timing_offset_s": 0.0, "speed_delta_m_s": 0.0}
        ],
    )
    record = compute_persistence_record(
        scenario_id="generated-0006",
        source_episode=_SOURCE_EPISODE,
        generated_scenario=_GENERATED,
        planner="goal",
        seed=4932,
        config=_CONFIG,
        commit_hashes=_COMMITS,
        exact_replay=assess_exact_replay(_SOURCE_EPISODE, replayed_episode=dict(_SOURCE_EPISODE)),
        critical_event_reproduced=assess_critical_event_reproduction(
            event_type="min_clearance",
            source_event_time_s=1.0,
            source_event_location=[2.0, 0.0],
            not_observed_reason="event not observed in replay",
        ),
        perturbation_grid=grid,
        cell_verdicts=cells,
        missing_cell_reasons=missing,
    )
    assert record["critical_event_reproduced"]["status"] == FAIL
    assert record["promotion"]["verdict"] == "reject"


def test_unfrozen_config_fails_closed() -> None:
    """A non-frozen perturbation config cannot gate promotion."""

    with pytest.raises(ScenarioPersistenceValidationError, match="frozen"):
        compute_persistence_record(
            scenario_id="generated-0007",
            source_episode=_SOURCE_EPISODE,
            generated_scenario=_GENERATED,
            planner="goal",
            seed=4932,
            config={"config_id": "x", "frozen": False},
            commit_hashes=_COMMITS,
            exact_replay=assess_exact_replay(
                _SOURCE_EPISODE, replayed_episode=dict(_SOURCE_EPISODE)
            ),
            critical_event_reproduced=assess_critical_event_reproduction(
                event_type="min_clearance",
                source_event_time_s=1.0,
                source_event_location=[2.0, 0.0],
                replayed_event_time_s=1.0,
                replayed_event_location=[2.0, 0.0],
            ),
            perturbation_grid={"timing_offsets_s": [0.0], "speed_deltas_m_s": [0.0]},
            cell_verdicts=[
                {"verdict": PASS, "reason": "ok", "timing_offset_s": 0.0, "speed_delta_m_s": 0.0}
            ],
        )


def test_record_is_schema_valid() -> None:
    """The produced record validates against the versioned JSON Schema."""

    record = _all_pass_persistence()
    assert record["schema_version"] == PERSISTENCE_SCHEMA_VERSION
    validate_persistence_record(record)


def test_identical_inputs_produce_checksum_identical_output(tmp_path: Path) -> None:
    """Two identical gate runs yield byte-identical JSON."""

    def build() -> str:
        record = _all_pass_persistence()
        return json.dumps(record, sort_keys=True, indent=2)

    first = build()
    second = build()
    assert first == second
    out_path = tmp_path / "record.json"
    out_path.write_text(first, encoding="utf-8")
    reparsed = json.loads(out_path.read_text())
    validate_persistence_record(reparsed)


def test_two_candidate_smoke_shows_promote_and_reject(tmp_path: Path) -> None:
    """A real two-candidate smoke demonstrates both verdict paths."""

    promoted = _all_pass_persistence()
    grid, cells, missing = _grid_block(
        timing_offsets_s=[-0.25, 0.0, 0.25],
        speed_deltas_m_s=[-0.2, 0.0, 0.2],
        cell_results=(
            [{"verdict": PASS, "reason": "ok"} for _ in range(6)]
            + [{"verdict": FAIL, "reason": "event not reproduced"} for _ in range(3)]
        ),
    )
    rejected = compute_persistence_record(
        scenario_id="generated-0008",
        source_episode=_SOURCE_EPISODE,
        generated_scenario=_GENERATED,
        planner="goal",
        seed=4932,
        config=_CONFIG,
        commit_hashes=_COMMITS,
        exact_replay=assess_exact_replay(_SOURCE_EPISODE, replayed_episode=dict(_SOURCE_EPISODE)),
        critical_event_reproduced=assess_critical_event_reproduction(
            event_type="min_clearance",
            source_event_time_s=1.0,
            source_event_location=[2.0, 0.0],
            replayed_event_time_s=1.0,
            replayed_event_location=[2.0, 0.0],
            time_tolerance_s=0.5,
            location_tolerance_m=0.75,
        ),
        perturbation_grid=grid,
        cell_verdicts=cells,
        missing_cell_reasons=missing,
    )

    promote_path = tmp_path / "promote.json"
    reject_path = tmp_path / "reject.json"
    promote_path.write_text(json.dumps(promoted, sort_keys=True), encoding="utf-8")
    reject_path.write_text(json.dumps(rejected, sort_keys=True), encoding="utf-8")

    validate_persistence_record(json.loads(promote_path.read_text()))
    validate_persistence_record(json.loads(reject_path.read_text()))
    assert promoted["promotion"]["verdict"] == "promote"
    assert rejected["promotion"]["verdict"] == "reject"


def test_criticality_reproduction_is_separate_from_exact_replay() -> None:
    """A positive replay does not imply event reproduction and vice versa."""

    replay = assess_exact_replay(_SOURCE_EPISODE, replayed_episode=dict(_SOURCE_EPISODE))
    event = assess_critical_event_reproduction(
        event_type="min_clearance",
        source_event_time_s=1.0,
        source_event_location=[2.0, 0.0],
        replayed_event_time_s=2.5,
        replayed_event_location=[5.0, 0.0],
    )
    assert replay["status"] == PASS
    assert event["status"] == FAIL
    assert replay["status"] != event["status"]
