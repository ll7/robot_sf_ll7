"""Focused fast-lane coverage for the published scenario archetype column (#9107).

Symbol-level tests keep the changed camera-ready reporting lines inside the exact-head
fast lane: they exercise archetype extraction, the scenario lookup, breakdown-row
emission, and one bounded artifact write without running a campaign.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest

from robot_sf.benchmark.camera_ready._reporting import (
    _build_breakdown_rows,
    _build_scenario_archetype_lookup,
)
from robot_sf.benchmark.camera_ready._summaries import _extract_archetype
from robot_sf.benchmark.camera_ready.campaign import (
    _FAMILY_BREAKDOWN_HEADERS,
    _SCENARIO_BREAKDOWN_HEADERS,
    _write_breakdown_and_parity_artifacts,
)

if TYPE_CHECKING:
    from pathlib import Path


def _episodes(tmp_path: Path, name: str, records: list[dict]) -> str:
    path = tmp_path / name
    path.write_text("\n".join(json.dumps(record) for record in records) + "\n", encoding="utf-8")
    return str(path)


def test_extract_archetype_prefers_metadata_and_strips() -> None:
    assert _extract_archetype({"metadata": {"archetype": " bottleneck "}}) == "bottleneck"
    assert _extract_archetype({"archetype": "crossing"}) == "crossing"
    assert _extract_archetype({"metadata": {"archetype": "  "}}) == ""
    assert _extract_archetype({}) == ""


@pytest.mark.parametrize("tag", ["crossing;legacy", "Crossing", "crossing legacy"])
def test_extract_archetype_rejects_ambiguous_or_noncanonical_tags(tag: str) -> None:
    with pytest.raises(ValueError, match="archetype"):
        _extract_archetype({"metadata": {"archetype": tag}})


def test_build_scenario_archetype_lookup_retains_undeclared_scenarios() -> None:
    lookup = _build_scenario_archetype_lookup(
        [
            {"name": "classic_bottleneck", "metadata": {"archetype": "bottleneck"}},
            {"name": "classic_crossing", "metadata": {"archetype": "crossing"}},
            {"name": "no_tag"},
        ]
    )
    assert lookup == {
        "classic_bottleneck": "bottleneck",
        "classic_crossing": "crossing",
        "no_tag": "",
    }


def test_build_scenario_archetype_lookup_rejects_conflicts_regardless_of_order() -> None:
    scenarios = [
        {"name": " shared ", "metadata": {"archetype": "bottleneck"}},
        {"name": "shared", "metadata": {"archetype": "crossing"}},
    ]

    with pytest.raises(ValueError, match="conflicting archetypes") as exc_info:
        _build_scenario_archetype_lookup(scenarios)
    first_message = str(exc_info.value)
    assert "shared" in first_message
    assert "bottleneck" in first_message
    assert "crossing" in first_message

    with pytest.raises(ValueError, match="conflicting archetypes") as exc_info:
        _build_scenario_archetype_lookup(list(reversed(scenarios)))
    assert str(exc_info.value) == first_message


def test_build_scenario_archetype_lookup_rejects_tagged_and_untagged_duplicates() -> None:
    scenarios = [
        {"name": "shared", "metadata": {"archetype": "bottleneck"}},
        {"name": " shared "},
    ]

    with pytest.raises(ValueError, match="conflicting archetypes"):
        _build_scenario_archetype_lookup(scenarios)


def test_breakdown_headers_place_archetype_beside_scenario_family() -> None:
    for headers in (_FAMILY_BREAKDOWN_HEADERS, _SCENARIO_BREAKDOWN_HEADERS):
        assert headers.index("archetype") == headers.index("scenario_family") + 1


def test_breakdown_rows_emit_config_archetype_and_empty_placeholder(tmp_path: Path) -> None:
    episodes_path = _episodes(
        tmp_path, "ep.jsonl", [{"scenario_id": "classic_bottleneck", "ped_collision_count": 0}]
    )
    run_entries = [
        {"planner": {"key": "orca", "algo": "orca"}, "status": "ok", "episodes_path": episodes_path}
    ]
    scenario_rows, family_rows = _build_breakdown_rows(
        run_entries, scenario_archetype_lookup={"classic_bottleneck": "bottleneck"}
    )
    assert scenario_rows[0]["archetype"] == "bottleneck"
    assert family_rows[0]["archetype"] == "bottleneck"

    plain_rows, plain_families = _build_breakdown_rows(run_entries)
    assert plain_rows[0]["archetype"] == ""  # no lookup and no record metadata
    assert plain_families[0]["archetype"] == ""

    embedded = _episodes(
        tmp_path,
        "embedded.jsonl",
        [
            {
                "scenario_id": "classic_bottleneck",
                "scenario_params": {"metadata": {"archetype": "bottleneck"}},
                "ped_collision_count": 0,
            }
        ],
    )
    fallback_rows, _ = _build_breakdown_rows(
        [{"planner": {"key": "orca", "algo": "orca"}, "status": "ok", "episodes_path": embedded}]
    )
    assert fallback_rows[0]["archetype"] == "bottleneck"  # record metadata fallback

    untagged = _episodes(
        tmp_path, "untagged.jsonl", [{"scenario_id": "corridor_low", "ped_collision_count": 0}]
    )
    empty_rows, empty_families = _build_breakdown_rows(
        [{"planner": {"key": "orca", "algo": "orca"}, "status": "ok", "episodes_path": untagged}]
    )
    assert empty_rows[0]["archetype"] == ""
    assert empty_families[0]["archetype"] == ""

    known_untagged = _episodes(
        tmp_path,
        "known-untagged.jsonl",
        [
            {
                "scenario_id": "no_tag",
                "scenario_params": {"metadata": {"archetype": "stale_record_tag"}},
                "ped_collision_count": 0,
            }
        ],
    )
    config_absence_rows, _ = _build_breakdown_rows(
        [{"planner": {"key": "orca", "algo": "orca"}, "episodes_path": known_untagged}],
        scenario_archetype_lookup={"no_tag": ""},
    )
    assert config_absence_rows[0]["archetype"] == ""


def test_breakdown_rows_join_whitespace_bearing_record_to_config_id(tmp_path: Path) -> None:
    """Record and config scenario IDs use the same normalized join key."""
    episodes_path = _episodes(
        tmp_path,
        "whitespace.jsonl",
        [
            {
                "scenario_id": " classic_bottleneck ",
                "scenario_params": {"metadata": {"scenario_family": "bottleneck"}},
                "ped_collision_count": 0,
            }
        ],
    )
    scenario_rows, family_rows = _build_breakdown_rows(
        [{"planner": {"key": "orca", "algo": "orca"}, "episodes_path": episodes_path}],
        scenario_archetype_lookup=_build_scenario_archetype_lookup(
            [{"name": " classic_bottleneck ", "metadata": {"archetype": "bottleneck"}}]
        ),
    )

    assert scenario_rows[0]["scenario_id"] == "classic_bottleneck"
    assert scenario_rows[0]["archetype"] == "bottleneck"
    assert family_rows[0]["scenario_family"] == "bottleneck"
    assert family_rows[0]["archetype"] == "bottleneck"

    padded_path = _episodes(tmp_path, "padded-id.jsonl", [{"scenario_id": " classic_crossing "}])
    padded_rows, _ = _build_breakdown_rows(
        [{"planner": {"key": "orca", "algo": "orca"}, "episodes_path": padded_path}],
        scenario_archetype_lookup={"classic_crossing": "crossing"},
    )
    assert padded_rows[0]["scenario_family"] == "classic"


def test_breakdown_rows_reject_record_family_conflict_with_config_archetype(
    tmp_path: Path,
) -> None:
    episodes_path = _episodes(
        tmp_path,
        "conflict.jsonl",
        [
            {
                "scenario_id": "classic_bottleneck",
                "scenario_params": {"metadata": {"scenario_family": "crossing"}},
            }
        ],
    )

    with pytest.raises(ValueError, match="scenario_family.*config archetype"):
        _build_breakdown_rows(
            [{"planner": {"key": "orca", "algo": "orca"}, "episodes_path": episodes_path}],
            scenario_archetype_lookup={"classic_bottleneck": "bottleneck"},
        )


def test_breakdown_rows_reject_delimiter_bearing_lookup_archetype(tmp_path: Path) -> None:
    episodes_path = _episodes(
        tmp_path, "invalid-lookup.jsonl", [{"scenario_id": "classic_bottleneck"}]
    )

    with pytest.raises(ValueError, match="reserved.*delimiter"):
        _build_breakdown_rows(
            [{"planner": {"key": "orca", "algo": "orca"}, "episodes_path": episodes_path}],
            scenario_archetype_lookup={"classic_bottleneck": "bottleneck;crossing"},
        )


def test_breakdown_rows_reject_delimiter_bearing_record_archetype(tmp_path: Path) -> None:
    episodes_path = _episodes(
        tmp_path,
        "invalid-record.jsonl",
        [
            {
                "scenario_id": "classic_bottleneck",
                "scenario_params": {"metadata": {"archetype": "bottleneck;crossing"}},
            }
        ],
    )

    with pytest.raises(ValueError, match="reserved.*delimiter"):
        _build_breakdown_rows(
            [{"planner": {"key": "orca", "algo": "orca"}, "episodes_path": episodes_path}]
        )


def test_breakdown_rows_collapse_shared_archetype_and_join_distinct_tags(tmp_path: Path) -> None:
    first = _episodes(tmp_path, "a.jsonl", [{"scenario_id": "classic_bottleneck"}])
    second = _episodes(tmp_path, "b.jsonl", [{"scenario_id": "classic_realworld_bottleneck"}])
    entries = [
        {"planner": {"key": "orca", "algo": "orca"}, "status": "ok", "episodes_path": first},
        {"planner": {"key": "orca", "algo": "orca"}, "status": "ok", "episodes_path": second},
    ]
    scenario_rows, family_rows = _build_breakdown_rows(
        entries,
        scenario_archetype_lookup={
            "classic_bottleneck": "bottleneck",
            "classic_realworld_bottleneck": "bottleneck",
        },
    )
    assert {row["archetype"] for row in scenario_rows} == {"bottleneck"}
    assert family_rows[0]["archetype"] == "bottleneck"

    third = _episodes(tmp_path, "c.jsonl", [{"scenario_id": "classic_crossing"}])
    _, mixed_families = _build_breakdown_rows(
        [
            *entries,
            {"planner": {"key": "orca", "algo": "orca"}, "status": "ok", "episodes_path": third},
        ],
        scenario_archetype_lookup={
            "classic_bottleneck": "bottleneck",
            "classic_realworld_bottleneck": "bottleneck",
            "classic_crossing": "crossing",
        },
    )
    assert mixed_families[0]["archetype"] == "bottleneck;crossing"


def test_write_breakdown_artifacts_publish_archetype_column(tmp_path: Path) -> None:
    episodes_path = _episodes(
        tmp_path, "ep.jsonl", [{"scenario_id": "classic_bottleneck", "ped_collision_count": 0}]
    )
    reports_dir = tmp_path / "reports"
    _write_breakdown_and_parity_artifacts(
        reports_dir,
        scenarios=[{"name": "classic_bottleneck", "metadata": {"archetype": "bottleneck"}}],
        run_entries=[
            {
                "planner": {"key": "orca", "algo": "orca"},
                "status": "ok",
                "episodes_path": episodes_path,
            }
        ],
        planner_rows=[],
    )
    for name in ("scenario_breakdown.csv", "scenario_family_breakdown.csv"):
        header = (reports_dir / name).read_text(encoding="utf-8").splitlines()[0].split(",")
        assert header[:4] == ["planner_key", "algo", "scenario_family", "archetype"]
    assert "bottleneck" in (reports_dir / "scenario_family_breakdown.csv").read_text(
        encoding="utf-8"
    )
