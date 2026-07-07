"""Tests for the footprint-orientation diagnostic (issue #4762)."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest

from robot_sf.nav.footprint_diagnostic import (
    FOOTPRINT_ORIENTATION_SCHEMA_VERSION,
    CircularFootprint,
    FootprintOrientationConfigError,
    RectangularFootprint,
    build_diagnostic_report,
    build_diagnostic_scenarios,
    centerline_clearance_m,
    footprint_aware_clearance_m,
    load_footprint_orientation_config,
    parse_diagnostic_parameters,
    parse_footprints,
    run_footprint_diagnostic,
    validate_footprint_orientation_config,
)

CONFIG_PATH = (
    Path(__file__).resolve().parents[2]
    / "configs"
    / "diagnostics"
    / "footprint_orientation_v1.yaml"
)


@pytest.fixture
def valid_payload() -> dict:
    """Return the repository footprint-orientation diagnostic config."""

    return load_footprint_orientation_config(CONFIG_PATH)


@pytest.fixture
def footprints(valid_payload: dict) -> list:
    """Return parsed footprint models from the checked-in config."""

    return parse_footprints(valid_payload)


@pytest.fixture
def params(valid_payload: dict) -> dict:
    """Return parsed diagnostic parameters from the checked-in config."""

    return parse_diagnostic_parameters(valid_payload)


def _result_by_footprint(results: list, footprint_id: str):
    """Return the clearance result row for one footprint id."""

    for result in results:
        if result.footprint_id == footprint_id:
            return result
    raise AssertionError(f"missing result for footprint {footprint_id!r}")


# --------------------------------------------------------------------------- #
# Config contract
# --------------------------------------------------------------------------- #


def test_config_loads_with_required_contract(valid_payload: dict) -> None:
    """The checked-in config carries the v1 schema, boundary language, and footprints."""

    assert valid_payload["schema_version"] == FOOTPRINT_ORIENTATION_SCHEMA_VERSION
    claim = valid_payload["claim_boundary"].casefold()
    assert "diagnostic" in claim
    assert "not a full se(2) planner" in claim
    assert "not calibrated" in claim
    kinds = {footprint["kind"] for footprint in valid_payload["footprints"]}
    assert "circular" in kinds
    assert "rectangular" in kinds
    family_ids = {family["id"] for family in valid_payload["scenario_families"]}
    assert {
        "narrow_passage",
        "pedestrian_crossing",
        "occluded_corner",
        "recovery_after_avoidance",
        "blocked_path_turn_around",
    }.issubset(family_ids)
    roles = {entry["role"] for entry in valid_payload["source_literature"]}
    assert roles == {"motivation_only"}


def test_parse_footprints_returns_circular_and_rectangular(footprints: list) -> None:
    """Parsing yields the four issue #4762 footprint models with correct kinds."""

    by_id = {footprint.id: footprint for footprint in footprints}
    assert set(by_id) == {"circular", "scooter_like", "cargo_bike_like", "shuttle_pod_like"}
    assert isinstance(by_id["circular"], CircularFootprint)
    assert by_id["circular"].radius_m == pytest.approx(0.35)
    assert isinstance(by_id["scooter_like"], RectangularFootprint)
    assert by_id["scooter_like"].length_m == pytest.approx(1.3)
    assert by_id["scooter_like"].width_m == pytest.approx(0.55)
    assert isinstance(by_id["shuttle_pod_like"], RectangularFootprint)
    assert by_id["shuttle_pod_like"].length_m == pytest.approx(3.0)


def test_duplicate_footprint_ids_fail_closed(valid_payload: dict) -> None:
    """Duplicate footprint IDs are rejected."""

    payload = deepcopy(valid_payload)
    payload["footprints"].append(deepcopy(payload["footprints"][0]))

    with pytest.raises(FootprintOrientationConfigError, match="duplicate footprint ids"):
        validate_footprint_orientation_config(payload)


def test_missing_rectangular_footprint_fails_closed(valid_payload: dict) -> None:
    """At least one rectangular footprint is required (AC1)."""

    payload = deepcopy(valid_payload)
    payload["footprints"] = [
        footprint for footprint in payload["footprints"] if footprint["kind"] != "rectangular"
    ]

    with pytest.raises(FootprintOrientationConfigError, match="at least one rectangular"):
        validate_footprint_orientation_config(payload)


def test_invalid_footprint_kind_fails_closed(valid_payload: dict) -> None:
    """Unknown footprint kinds are rejected."""

    payload = deepcopy(valid_payload)
    payload["footprints"][1]["kind"] = "elliptical"

    with pytest.raises(
        FootprintOrientationConfigError, match="kind must be 'circular' or 'rectangular'"
    ):
        validate_footprint_orientation_config(payload)


def test_missing_claim_boundary_language_fails_closed(valid_payload: dict) -> None:
    """The claim boundary must keep diagnostic, not-a-full-SE(2)-planner, and not-calibrated language."""

    payload = deepcopy(valid_payload)
    payload["claim_boundary"] = "a helpful clearance report"

    with pytest.raises(FootprintOrientationConfigError, match="claim_boundary"):
        validate_footprint_orientation_config(payload)


def test_non_motivation_source_role_fails_closed(valid_payload: dict) -> None:
    """Source literature cannot claim implemented-method status."""

    payload = deepcopy(valid_payload)
    payload["source_literature"][0]["role"] = "implemented_method"

    with pytest.raises(FootprintOrientationConfigError, match="motivation_only"):
        validate_footprint_orientation_config(payload)


def test_missing_required_scenario_family_fails_closed(valid_payload: dict) -> None:
    """All five candidate scenario families must be declared."""

    payload = deepcopy(valid_payload)
    payload["scenario_families"] = [
        family for family in payload["scenario_families"] if family["id"] != "occluded_corner"
    ]

    with pytest.raises(FootprintOrientationConfigError, match="missing required scenario_family"):
        validate_footprint_orientation_config(payload)


def test_nonpositive_sample_step_fails_closed(valid_payload: dict) -> None:
    """A non-positive sample step is rejected."""

    payload = deepcopy(valid_payload)
    payload["diagnostic_parameters"]["sample_step_m"] = 0.0

    with pytest.raises(FootprintOrientationConfigError, match="sample_step_m"):
        validate_footprint_orientation_config(payload)


# --------------------------------------------------------------------------- #
# Geometry: centerline vs footprint-aware clearance
# --------------------------------------------------------------------------- #


def test_centerline_clearance_matches_known_values() -> None:
    """Centerline clearance is the bare route-to-obstacle distance, ignoring footprint."""

    scenarios = {s.id: s for s in build_diagnostic_scenarios()}
    assert centerline_clearance_m(
        scenarios["narrow_passage_v1"].route, scenarios["narrow_passage_v1"].obstacles
    ) == pytest.approx(0.45)
    assert centerline_clearance_m(
        scenarios["occluded_corner_v1"].route, scenarios["occluded_corner_v1"].obstacles
    ) == pytest.approx(0.7)
    assert centerline_clearance_m(
        scenarios["blocked_path_turn_around_v1"].route,
        scenarios["blocked_path_turn_around_v1"].obstacles,
    ) == pytest.approx(1.0)


def test_circular_footprint_aware_equals_centerline_minus_radius(
    footprints: list, params: dict
) -> None:
    """For a circular footprint the analytic margin equals centerline minus radius (AC3)."""

    scenarios = {s.id: s for s in build_diagnostic_scenarios()}
    narrow = scenarios["narrow_passage_v1"]
    circular = next(fp for fp in footprints if isinstance(fp, CircularFootprint))
    clearance, collision, _ = footprint_aware_clearance_m(
        narrow.route, circular, narrow.obstacles, params["sample_step_m"], params["max_samples"]
    )
    assert clearance == pytest.approx(0.45 - 0.35)
    assert collision is False


def test_shuttle_collides_in_narrow_passage_where_circular_clears(
    footprints: list, params: dict
) -> None:
    """AC2: the same scenario produces contrasting pass/fail outcomes across footprints."""

    scenarios = {s.id: s for s in build_diagnostic_scenarios()}
    narrow = scenarios["narrow_passage_v1"]
    results = {
        r.footprint_id: r
        for r in run_footprint_diagnostic(
            narrow, footprints, params["sample_step_m"], params["max_samples"]
        )
    }
    assert results["circular"].status == "clear"
    assert results["scooter_like"].status == "clear"
    assert results["cargo_bike_like"].status == "clear"
    assert results["shuttle_pod_like"].status == "collision"
    # Centerline clearance is identical across footprints (same route/obstacles),
    # while footprint-aware clearance diverges: this is the AC3 distinction.
    assert (
        results["circular"].centerline_clearance_m
        == results["shuttle_pod_like"].centerline_clearance_m
    )
    assert results["circular"].footprint_aware_clearance_m == pytest.approx(0.10)
    assert results["shuttle_pod_like"].footprint_aware_clearance_m == pytest.approx(0.0)


def test_occluded_corner_surfaces_turn_overrun_for_elongated_bodies(
    footprints: list, params: dict
) -> None:
    """An elongated body oriented along the incoming leg overruns the turn and collides."""

    scenarios = {s.id: s for s in build_diagnostic_scenarios()}
    corner = scenarios["occluded_corner_v1"]
    results = {
        r.footprint_id: r
        for r in run_footprint_diagnostic(
            corner, footprints, params["sample_step_m"], params["max_samples"]
        )
    }
    assert results["circular"].status == "clear"
    assert results["scooter_like"].status == "clear"
    assert results["cargo_bike_like"].status == "collision"
    assert results["shuttle_pod_like"].status == "collision"
    assert results["cargo_bike_like"].method == "oriented_rectangle_sampling"
    assert results["circular"].method == "analytic_margin"


def test_blocked_path_surfaces_forward_reach_for_elongated_bodies(
    footprints: list, params: dict
) -> None:
    """An elongated body's forward reach pokes a dead-end wall a circular body clears."""

    scenarios = {s.id: s for s in build_diagnostic_scenarios()}
    blocked = scenarios["blocked_path_turn_around_v1"]
    results = {
        r.footprint_id: r
        for r in run_footprint_diagnostic(
            blocked, footprints, params["sample_step_m"], params["max_samples"]
        )
    }
    assert results["circular"].status == "clear"
    assert results["scooter_like"].status == "clear"
    assert results["cargo_bike_like"].status == "collision"
    assert results["shuttle_pod_like"].status == "collision"


def test_footprint_aware_returns_none_without_obstacles(footprints: list, params: dict) -> None:
    """Missing obstacles produce None, not a zero-like clearance."""

    circular = next(fp for fp in footprints if isinstance(fp, CircularFootprint))
    clearance, collision, samples = footprint_aware_clearance_m(
        [(0.0, 0.0), (5.0, 0.0)], circular, [], params["sample_step_m"], params["max_samples"]
    )
    assert clearance is None
    assert collision is False
    assert samples == 0


# --------------------------------------------------------------------------- #
# Report shape
# --------------------------------------------------------------------------- #


def test_report_distinguishes_centerline_from_footprint_aware_clearance(
    footprints: list, params: dict, valid_payload: dict
) -> None:
    """AC3: the report carries both clearance fields and a per-footprint status."""

    scenarios = build_diagnostic_scenarios()
    report = build_diagnostic_report(
        scenarios,
        footprints,
        params["sample_step_m"],
        params["max_samples"],
        profile_id=valid_payload["profile_id"],
    )
    assert report["schema_version"] == FOOTPRINT_ORIENTATION_SCHEMA_VERSION
    assert "not a full se(2) planner" in report["claim_boundary_note"].casefold()
    assert len(report["scenarios"]) == 5
    narrow = next(sc for sc in report["scenarios"] if sc["scenario_id"] == "narrow_passage_v1")
    assert narrow["collision_footprint_ids"] == ["shuttle_pod_like"]
    assert set(narrow["clear_footprint_ids"]) == {"circular", "scooter_like", "cargo_bike_like"}
    for row in narrow["results"]:
        assert "centerline_clearance_m" in row
        assert "footprint_aware_clearance_m" in row
        assert row["status"] in ("clear", "collision")


# --------------------------------------------------------------------------- #
# Runner
# --------------------------------------------------------------------------- #


def test_runner_writes_markdown_and_json_reports(tmp_path: Path) -> None:
    """The runner produces both a JSON and a Markdown artifact from the checked-in config."""

    from scripts.diagnostics.run_footprint_orientation_diagnostic import (
        build_markdown_report,
        main,
    )

    json_path = tmp_path / "report.json"
    md_path = tmp_path / "report.md"

    assert main(["--config", str(CONFIG_PATH), "--format", "json", "--output", str(json_path)]) == 0
    assert (
        main(["--config", str(CONFIG_PATH), "--format", "markdown", "--output", str(md_path)]) == 0
    )

    import json

    report = json.loads(json_path.read_text(encoding="utf-8"))
    assert report["profile_id"] == "footprint_orientation_diagnostic_v1"
    assert len(report["scenarios"]) == 5

    markdown = md_path.read_text(encoding="utf-8")
    assert "Footprint-orientation diagnostic" in markdown
    assert "Centerline (m)" in markdown
    assert "collision" in markdown
    # build_markdown_report must render a self-contained table from a report dict
    assert "shuttle_pod_like" in build_markdown_report(report)
