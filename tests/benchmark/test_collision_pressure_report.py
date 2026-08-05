"""Tests for the typed-ledger collision-pressure report."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from pathlib import Path

from robot_sf.benchmark.collision_pressure_report import (
    CollisionPressureReportError,
    build_collision_pressure_report,
    main,
    write_collision_pressure_report,
)
from robot_sf.benchmark.event_ledger import build_event_ledger

CHECKSUMS = {"episodes.jsonl": "a" * 64}


def _row(
    episode_id: str,
    *,
    family: str = "family_a",
    collision: bool = False,
    partner_events: list[dict[str, object]] | None = None,
) -> dict[str, object]:
    """Build an episode row with a reconciled EpisodeEventLedger.v2 payload."""
    events = partner_events or []
    record: dict[str, object] = {
        "episode_id": episode_id,
        "scenario_family": family,
        "scenario_id": f"scenario-{episode_id}",
        "seed": 1,
        "algo": "goal",
        "metrics": {"collisions": 1.0 if collision else 0.0},
        "outcome": {
            "collision_event": collision,
            "route_complete": not collision,
        },
        "termination_reason": "collision" if collision else "success",
        "git_hash": "commit-6708",
    }
    record["event_ledger"] = build_event_ledger(record, collision_events=events)
    return record


def _event(partner_type: str, *, partner_id: str | None = "actor-1") -> dict[str, object]:
    """Build one typed collision event."""
    return {
        "collision_partner_type": partner_type,
        "collision_partner_id": partner_id,
        "collision_time": 1.0,
        "relative_speed_at_contact": 0.2,
        "clearance_series_source": "trace.clearance",
        "exact_event_source": "trace.contact",
    }


def _report(rows: list[dict[str, object]]) -> dict[str, object]:
    """Build a report with fixed provenance for tests."""
    return build_collision_pressure_report(
        rows,
        eligible_families=["family_a"],
        source_commit="commit-6708",
        release_id="release-test",
        bundle_id="bundle-test",
        input_checksums=CHECKSUMS,
    )


def test_report_counts_contact_episodes_once_and_preserves_partner_overlap() -> None:
    """Simultaneous pedestrian/obstacle events do not double-count contacts."""
    rows = [
        _row(
            "episode-1",
            collision=True,
            partner_events=[
                _event("pedestrian"),
                _event("static_geometry"),
                _event("boundary"),
            ],
        ),
        _row("episode-2", collision=True, partner_events=[_event("boundary")]),
        _row("episode-3", collision=False),
    ]

    report = _report(rows)
    counts = report["counts"]

    assert report["denominator"]["eligible_episode_count"] == 3
    assert counts["contact_episode_count"] == 2
    assert counts["collision_event_count"] == 4
    assert counts["partner_type_episode_counts"] == {
        "pedestrian": 1,
        "static_geometry": 1,
        "boundary": 2,
        "goal_artifact": 0,
    }
    assert counts["obstacle_rollup_episode_count"] == 2
    assert counts["pedestrian_obstacle_overlap_episode_counts"] == {
        "pedestrian_only": 0,
        "obstacle_only": 1,
        "pedestrian_and_obstacle": 1,
    }


def test_missing_ledger_is_an_explicit_exclusion() -> None:
    """A missing ledger cannot silently enter the denominator."""
    rows = [_row("episode-1"), {"episode_id": "episode-missing", "scenario_family": "family_a"}]

    report = _report(rows)

    assert report["denominator"]["eligible_episode_count"] == 1
    assert report["selection"]["excluded_row_count"] == 1
    assert report["selection"]["exclusion_counts"] == {"missing_event_ledger": 1}


def test_missing_family_is_an_explicit_exclusion() -> None:
    """Rows without the declared family field are excluded with a reason."""
    rows = [_row("episode-1"), {"episode_id": "episode-missing"}]

    report = _report(rows)

    assert report["selection"]["exclusion_counts"] == {"missing_scenario_family": 1}


def test_collision_without_typed_event_records_is_excluded() -> None:
    """An exact collision without typed event records is not reportable."""
    rows = [_row("episode-1", collision=True)]

    with pytest.raises(CollisionPressureReportError, match="collision_event_records_missing"):
        _report(rows)


def test_duplicate_episode_keys_fail_closed() -> None:
    """Duplicate eligible identities make the denominator unknowable."""
    rows = [_row("episode-1"), _row("episode-1")]

    with pytest.raises(CollisionPressureReportError, match="duplicate eligible episode keys"):
        _report(rows)


def test_missing_requested_family_fails_closed() -> None:
    """A requested family with no auditable rows cannot produce a report."""
    with pytest.raises(CollisionPressureReportError, match="no auditable eligible episodes"):
        build_collision_pressure_report(
            [_row("episode-1", family="other")],
            eligible_families=["family_a"],
            source_commit="commit-6708",
            release_id="release-test",
            bundle_id="bundle-test",
            input_checksums=CHECKSUMS,
        )


def test_report_outputs_are_deterministic(tmp_path: Path) -> None:
    """Repeated JSON and CSV generation is byte-identical."""
    report = _report([_row("episode-1", collision=True, partner_events=[_event("goal_artifact")])])
    first = write_collision_pressure_report(
        report,
        json_path=tmp_path / "first.json",
        csv_path=tmp_path / "first.csv",
    )
    second = write_collision_pressure_report(
        report,
        json_path=tmp_path / "second.json",
        csv_path=tmp_path / "second.csv",
    )

    assert first["json"].read_bytes() == second["json"].read_bytes()
    assert first["csv"].read_bytes() == second["csv"].read_bytes()
    assert json.loads(first["json"].read_text(encoding="utf-8"))["schema_version"] == (
        "collision_pressure_report.v1"
    )


def test_cli_reads_jsonl_and_writes_both_outputs(tmp_path: Path) -> None:
    """The CLI preserves the same declared-family and provenance contract."""
    rows_path = tmp_path / "episodes.jsonl"
    rows_path.write_text(
        json.dumps(_row("episode-1", collision=True, partner_events=[_event("pedestrian")])) + "\n",
        encoding="utf-8",
    )
    json_path = tmp_path / "report.json"
    csv_path = tmp_path / "report.csv"

    exit_code = main(
        [
            "--rows",
            str(rows_path),
            "--eligible-family",
            "family_a",
            "--source-commit",
            "commit-6708",
            "--release-id",
            "release-test",
            "--bundle-id",
            "bundle-test",
            "--input-checksum",
            f"episodes.jsonl={'a' * 64}",
            "--json-out",
            str(json_path),
            "--csv-out",
            str(csv_path),
        ]
    )

    assert exit_code == 0
    assert json_path.exists()
    assert csv_path.exists()
    assert "contact_episode_count" in csv_path.read_text(encoding="utf-8")
