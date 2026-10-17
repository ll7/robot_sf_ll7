"""Spawn-overlap rows stay out of every release rate path (issue #9725)."""

from __future__ import annotations

from typing import Any

import pytest

from robot_sf.benchmark.camera_ready._reporting import _resolve_planner_metrics
from robot_sf.benchmark.collision.collision_pressure_report import _ledger_exclusion_reason
from robot_sf.benchmark.event_ledger import (
    EPISODE_EVENT_LEDGER_SCHEMA_VERSION,
    build_event_ledger,
    reconcile_event_ledger,
)
from robot_sf.benchmark.hierarchical_paired_release_analysis import (
    build_matched_cells_from_ledger_rows,
)
from robot_sf.benchmark.parquet_export import (
    _cell_record_eligible,
    _comparison_record_eligible,
)
from robot_sf.benchmark.seed_variance import build_seed_variability_rows
from robot_sf.benchmark.spawn_validity import (
    build_spawn_validity,
    record_has_invalid_spawn,
    record_has_spawn_overlap,
)

_OVERLAP = build_spawn_validity({"overlap": True}, [])
_CLEAN = build_spawn_validity({"overlap": False}, [])


def _record(*, seed: int, collision: bool, spawn: dict[str, Any]) -> dict[str, Any]:
    """Build a minimal episode record with a spawn-validity block."""
    return {
        "episode_id": f"ep-{seed}-{collision}",
        "scenario_id": "scenario",
        "seed": seed,
        "algo": "orca",
        "status": "collision" if collision else "success",
        "termination_reason": "collision" if collision else "success",
        "outcome": {
            "collision_event": collision,
            "route_complete": not collision,
            "timeout_event": False,
        },
        "metrics": {
            "success": 0.0 if collision else 1.0,
            "collisions": 1.0 if collision else 0.0,
        },
        "spawn_validity": spawn,
    }


def test_parquet_cell_and_comparison_eligibility_exclude_spawn_overlap() -> None:
    """Parquet cell aggregates and comparisons drop spawn-overlap rows only."""
    invalid = _record(seed=1, collision=True, spawn=_OVERLAP)
    valid = _record(seed=1, collision=True, spawn=_CLEAN)
    assert _cell_record_eligible(invalid) is False
    assert _comparison_record_eligible(invalid) is False
    assert _cell_record_eligible(valid) == _cell_record_eligible(
        {k: v for k, v in valid.items() if k != "spawn_validity"}
    )
    assert _comparison_record_eligible(valid) == _comparison_record_eligible(
        {k: v for k, v in valid.items() if k != "spawn_validity"}
    )


def test_seed_variability_rows_skip_spawn_overlap() -> None:
    """Seed-variability rows count only valid episodes."""
    rows = build_seed_variability_rows(
        [
            _record(seed=1, collision=True, spawn=_OVERLAP),
            _record(seed=2, collision=False, spawn=_CLEAN),
        ],
        metrics=["success", "collisions"],
        campaign_id="c",
        config_hash="h",
        git_hash="g",
    )
    assert len(rows) == 1
    assert rows[0]["episode_count"] == 1
    assert rows[0]["seed_list"] == [2]


def test_camera_ready_planner_metrics_skip_spawn_overlap() -> None:
    """The camera-ready planner row recomputes rates without spawn-overlap rows."""
    metrics, *_ = _resolve_planner_metrics(
        {},
        [
            _record(seed=1, collision=True, spawn=_OVERLAP),
            _record(seed=2, collision=False, spawn=_CLEAN),
        ],
        (0.0, 0.0),
        (0.0, 0.0),
        (0.0, 0.0),
    )
    assert metrics["success_mean"] == pytest.approx(1.0)
    assert metrics["collisions_mean"] == pytest.approx(0.0)


def _ledger_row(seed: int, planner: str, *, collision: bool, invalid: bool) -> dict[str, Any]:
    return {
        "schema_version": EPISODE_EVENT_LEDGER_SCHEMA_VERSION,
        "scenario_id": "scenario",
        "seed": seed,
        "planner": planner,
        "exact_events": {
            "collision": collision,
            "goal_reached": not collision,
            "timeout": False,
            "invalid_run": invalid,
        },
        "surrogate_events": {"near_miss": False},
        "provenance": {
            "completion_time": 1.0,
            "near_miss_count": 0,
            "exposure": {"time": 1.0, "distance": 1.0, "opportunity": 1.0},
            "interaction_exposure": {
                "schema_version": "interaction_exposure.v1",
                "status": "computed",
                "source_steps": 1,
                "denominator_steps": 1,
            },
        },
    }


def test_hierarchical_pairing_drops_invalid_run_cells() -> None:
    """A pair with an invalid arm is dropped whole; the other pairs remain."""
    rows = [
        _ledger_row(1, "a", collision=True, invalid=True),
        _ledger_row(1, "b", collision=False, invalid=False),
        _ledger_row(2, "a", collision=False, invalid=False),
        _ledger_row(2, "b", collision=False, invalid=False),
    ]
    cells = build_matched_cells_from_ledger_rows(rows, planner_pair=("a", "b"))
    assert [cell.seed for cell in cells] == [2]


def test_collision_pressure_report_excludes_invalid_run_ledgers() -> None:
    """The collision-pressure report names invalid runs as an explicit exclusion."""
    record = _record(seed=1, collision=True, spawn=_OVERLAP)
    record["metrics"]["total_collision_count"] = 1.0
    collision_event = {
        "collision_partner_type": "pedestrian",
        "collision_partner_id": "0",
        "collision_time": 0.1,
        "relative_speed_at_contact": 0.0,
        "clearance_series_source": "test",
        "exact_event_source": "test",
    }
    ledger = build_event_ledger(record, collision_events=[collision_event])
    assert ledger["exact_events"]["invalid_run"] is True
    assert _ledger_exclusion_reason(ledger) == "invalid_run"


def test_reset_overlap_with_completed_route_is_invalid_and_reconciles() -> None:
    """Observed success remains visible but an invalid reset cannot enter rates."""
    spawn = build_spawn_validity({"overlap": True}, [], route_complete=True)
    assert spawn["schema_version"] == "spawn_validity.v2"
    assert spawn["reset_overlap"] is True
    assert spawn["invalid_run"] is True
    assert spawn["invalid_reason"] == "spawn_overlap"
    record = _record(seed=1, collision=False, spawn=spawn)
    ledger = build_event_ledger(record)
    assert ledger["exact_events"]["goal_reached"] is True
    assert ledger["exact_events"]["invalid_run"] is True
    assert ledger["provenance"]["invalid_reason"] == "spawn_overlap"
    assert reconcile_event_ledger(ledger) == []
    assert _cell_record_eligible(record) is False
    assert _comparison_record_eligible(record) is False
    collided = _record(seed=2, collision=True, spawn=_OVERLAP)
    collided["metrics"]["total_collision_count"] = 1.0
    collided_ledger = build_event_ledger(collided)
    assert collided_ledger["exact_events"]["invalid_run"] is True
    assert reconcile_event_ledger(collided_ledger) == []


def test_respawn_collision_needs_same_pedestrian_and_time_window() -> None:
    """Only a collision with the respawned pedestrian shortly after respawn is invalid."""
    event = {"group_id": 0, "ped_rows": [3, 4], "step": 10}

    def collision(partner: str, time_s: float) -> dict[str, Any]:
        return {
            "collision_partner_type": "pedestrian",
            "collision_partner_id": partner,
            "collision_time": time_s,
        }

    same_soon = build_spawn_validity(
        {"overlap": False}, [event], collision_events=[collision("3", 1.1)]
    )
    assert same_soon["invalid_run"] is True
    assert record_has_spawn_overlap({"spawn_validity": same_soon}) is True
    other_ped = build_spawn_validity(
        {"overlap": False}, [event], collision_events=[collision("7", 1.1)]
    )
    assert other_ped["invalid_run"] is False
    too_late = build_spawn_validity(
        {"overlap": False}, [event], collision_events=[collision("4", 5.0)]
    )
    assert too_late["invalid_run"] is False


def test_seed_episode_rows_mark_invalid_and_readers_skip_them(tmp_path) -> None:
    """seed_episode_rows.csv lists spawn-overlap rows with invalid_run; rate readers drop them."""
    import csv

    from robot_sf.benchmark.seed_variance import build_seed_episode_rows, seed_episode_row_is_valid
    from scripts.tools.analyze_scenario_seed_sensitivity import load_selected_episode_rows

    rows = build_seed_episode_rows(
        [
            _record(seed=1, collision=True, spawn=_OVERLAP),
            _record(seed=2, collision=False, spawn=_CLEAN),
        ]
    )
    by_seed = {row["seed"]: row for row in rows}
    assert by_seed[1]["invalid_run"] is True
    assert by_seed[1]["invalid_reason"] == "spawn_overlap"
    assert by_seed[2]["invalid_run"] is False
    csv_path = tmp_path / "seed_episode_rows.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=sorted({k for r in rows for k in r}))
        writer.writeheader()
        writer.writerows(rows)
    with csv_path.open(newline="", encoding="utf-8") as handle:
        csv_rows = list(csv.DictReader(handle))
    assert [seed_episode_row_is_valid(r) for r in csv_rows] == [False, True]
    assert seed_episode_row_is_valid({"success": "1"})  # older CSVs without the column
    loaded = load_selected_episode_rows(csv_path, {"orca"})
    assert [(row.seed, row.success) for row in loaded] == [(2, 1.0)]


def test_respawn_attribution_uses_every_contact_partner_and_lower_tolerance() -> None:
    """A respawned pedestrian in contact counts even when another one is nearest."""
    event = {"group_id": 0, "ped_rows": [5], "step": 10}
    nearest_other = {
        "collision_partner_type": "pedestrian",
        "collision_partner_id": "2",
        "contact_partner_ids": ["2", "5"],
        "collision_time": 1.0,
    }
    assert build_spawn_validity({"overlap": False}, [event], collision_events=[nearest_other])[
        "invalid_run"
    ]
    boundary = {
        "collision_partner_type": "pedestrian",
        "collision_partner_id": "5",
        "collision_time": 0.9,
    }
    # elapsed == -dt exactly (0.9 - 1.0): inside the tolerant lower bound.
    assert build_spawn_validity({"overlap": False}, [event], collision_events=[boundary])[
        "invalid_run"
    ]


def test_aggregate_meta_counts_unmeasured_reset_clearance() -> None:
    """Unmeasured resets are visible in metadata and excluded from rates."""
    from robot_sf.benchmark.aggregate import compute_aggregates
    from robot_sf.benchmark.seed_variance import build_seed_episode_rows

    unmeasured = build_spawn_validity(None, [], reset_clearance_error="AttributeError: x")
    assert unmeasured["reset_clearance_status"] == "unavailable"
    assert unmeasured["reset_clearance_error"] == "AttributeError: x"
    assert unmeasured["invalid_run"] is True
    assert unmeasured["invalid_reason"] == "reset_clearance_unavailable"
    records = [
        _record(seed=1, collision=False, spawn=unmeasured),
        _record(seed=2, collision=True, spawn=_OVERLAP),
        _record(seed=3, collision=False, spawn=_CLEAN),
    ]
    aggregate = compute_aggregates(records, group_by="algo")
    meta = aggregate["_meta"]["spawn_validity"]
    assert meta == {
        "records_with_spawn_validity": 3,
        "spawn_overlap_excluded_count": 1,
        "reset_clearance_unavailable_count": 1,
    }
    assert aggregate["_meta"]["evidence_eligibility"]["excluded_record_count"] == 2
    assert aggregate["orca"]["success"]["mean"] == pytest.approx(1.0)
    assert _cell_record_eligible(records[0]) is False
    assert _comparison_record_eligible(records[0]) is False
    unknown_ledger = build_event_ledger(records[0])
    assert unknown_ledger["exact_events"]["goal_reached"] is True
    assert unknown_ledger["exact_events"]["invalid_run"] is True
    assert unknown_ledger["provenance"]["invalid_reason"] == "reset_clearance_unavailable"
    assert reconcile_event_ledger(unknown_ledger) == []
    seed_rows = build_seed_variability_rows(
        records, metrics=["success"], campaign_id="c", config_hash="h", git_hash="g"
    )
    assert len(seed_rows) == 1
    assert seed_rows[0]["seed_list"] == [3]
    episode_rows = {row["seed"]: row for row in build_seed_episode_rows(records)}
    assert episode_rows[1]["invalid_run"] is True
    assert episode_rows[1]["invalid_reason"] == "reset_clearance_unavailable"
    metrics, *_ = _resolve_planner_metrics({}, records, (0.0, 0.0), (0.0, 0.0), (0.0, 0.0))
    assert metrics["success_mean"] == pytest.approx(1.0)


@pytest.mark.parametrize(
    "mutation",
    ["reported_overlap", "raw_overlap", "unavailable", "missing_status"],
)
def test_v2_forged_valid_reset_telemetry_stays_out_of_rates(mutation: str) -> None:
    """Rate readers inspect v2 reset telemetry, not only its invalid flag."""
    from robot_sf.benchmark.aggregate import compute_aggregates
    from robot_sf.benchmark.seed_variance import build_seed_episode_rows

    forged = build_spawn_validity({"overlap": False}, [])
    if mutation == "reported_overlap":
        forged["reset_overlap"] = True
    elif mutation == "raw_overlap":
        forged["reset_clearance"]["overlap"] = True
    elif mutation == "unavailable":
        forged["reset_clearance_status"] = "unavailable"
    else:
        del forged["reset_clearance_status"]
    assert forged["invalid_run"] is False
    bad = _record(seed=1, collision=True, spawn=forged)
    good = _record(seed=2, collision=False, spawn=_CLEAN)
    assert record_has_invalid_spawn(bad) is True
    assert _cell_record_eligible(bad) is False
    assert _comparison_record_eligible(bad) is False
    assert compute_aggregates([bad, good], group_by="algo")["orca"]["success"][
        "mean"
    ] == pytest.approx(1.0)
    metrics, *_ = _resolve_planner_metrics({}, [bad, good], (0.0, 0.0), (0.0, 0.0), (0.0, 0.0))
    assert metrics["success_mean"] == pytest.approx(1.0)
    seed_rows = build_seed_variability_rows(
        [bad, good], metrics=["success"], campaign_id="c", config_hash="h", git_hash="g"
    )
    assert seed_rows[0]["seed_list"] == [2]
    rows = {row["seed"]: row for row in build_seed_episode_rows([bad, good])}
    assert rows[1]["invalid_run"] is True
    assert rows[1]["invalid_reason"] == "spawn_validity_inconsistent"
    legacy = dict(forged, schema_version="spawn_validity.v1")
    assert record_has_invalid_spawn(_record(seed=1, collision=True, spawn=legacy)) is False


def test_v1_non_overlap_invalid_reason_keeps_historical_rate_classification() -> None:
    """Legacy rows were excluded only for the explicit spawn-overlap reason."""
    from robot_sf.benchmark.aggregate import compute_aggregates

    historical = dict(
        build_spawn_validity({"overlap": False}, []),
        schema_version="spawn_validity.v1",
        invalid_run=True,
        invalid_reason="other_diagnostic_reason",
    )
    row = _record(seed=1, collision=True, spawn=historical)
    assert record_has_invalid_spawn(row) is False
    assert _cell_record_eligible(row) == _cell_record_eligible(
        {key: value for key, value in row.items() if key != "spawn_validity"}
    )
    assert _comparison_record_eligible(row) is True
    result = compute_aggregates(
        [row, _record(seed=2, collision=False, spawn=_CLEAN)], group_by="algo"
    )
    assert result["orca"]["success"]["mean"] == pytest.approx(0.5)
    historical["invalid_reason"] = "spawn_overlap"
    assert record_has_invalid_spawn(row) is True


def test_map_inventory_includes_successor_maps() -> None:
    """The default map inventory verifies successor maps kept outside the pinned registry."""
    from robot_sf.maps.verification.map_inventory import MapInventory

    inventory = MapInventory()
    ids = {record.map_id for record in inventory.get_all_maps()}
    assert "classic_station_platform_v2" in ids
    assert "classic_station_platform" in ids
