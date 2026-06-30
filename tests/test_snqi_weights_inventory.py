"""Tests for the SNQI weight-set provenance inventory (issue #3723).

These tests prove that conflicting "canonical" SNQI weight sources are
discovered and reported **fail-closed**, while asserting that the diagnostic
does not change any current scoring behavior.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from robot_sf.benchmark.snqi.compute import (
    WEIGHT_NAMES,
    compute_snqi,
    recompute_snqi_weights,
)
from robot_sf.benchmark.snqi.weights_inventory import (
    SNQIWeightProvenanceError,
    build_inventory_report,
    compare_code_default_to_shipped_sources,
    detect_conflicts,
    inventory_weight_sets,
    preflight_snqi_weight_sets,
)

# Code-default ("canonical" method) weights, pinned here as a regression guard.
_CODE_DEFAULT = {
    "w_success": 1.0,
    "w_time": 0.8,
    "w_collisions": 2.0,
    "w_near": 1.0,
    "w_comfort": 0.5,
    "w_force_exceed": 1.5,
    "w_jerk": 0.3,
}

REPO_ROOT = Path(__file__).parents[1]


def _write_weights(path: Path, weights: dict[str, float], *, nested: bool = False) -> None:
    """Write a weights JSON file in either flat or model-artifact (nested) form."""
    payload = {"weights": weights} if nested else dict(weights)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _make_fixture_repo(tmp_path: Path, *, model_jerk_dominant: bool = True) -> Path:
    """Create a minimal repo layout with the registered SNQI weight files."""
    jerk = dict.fromkeys(WEIGHT_NAMES, 0.1)
    jerk["w_jerk"] = 3.0  # jerk-dominant, raw scale (matches shipped v1)

    # If model file should AGREE with the code default, ship the collision set.
    model_weights = jerk if model_jerk_dominant else dict(_CODE_DEFAULT)

    _write_weights(tmp_path / "model/snqi_canonical_weights_v1.json", model_weights, nested=True)
    _write_weights(tmp_path / "configs/benchmarks/snqi_weights_camera_ready_v1.json", jerk)
    _write_weights(
        tmp_path / "configs/benchmarks/snqi_weights_camera_ready_v2.json",
        {
            "w_success": 0.2,
            "w_time": 0.28,
            "w_collisions": 0.05,
            "w_near": 0.18,
            "w_comfort": 0.15,
            "w_force_exceed": 0.08,
            "w_jerk": 0.06,
        },
    )
    _write_weights(
        tmp_path / "configs/benchmarks/snqi_weights_camera_ready_v3.json",
        {
            "w_success": 0.19,
            "w_time": 0.095,
            "w_collisions": 0.105,
            "w_near": 0.31,
            "w_comfort": 0.18,
            "w_force_exceed": 0.069,
            "w_jerk": 0.052,
        },
    )
    return tmp_path


def test_inventory_discovers_all_registered_sources(tmp_path):
    """Every registered weight source is reported, code default included."""
    repo = _make_fixture_repo(tmp_path)
    records = inventory_weight_sets(repo)
    names = {r.name for r in records}
    assert {
        "code_default",
        "model_canonical_v1",
        "camera_ready_v1",
        "camera_ready_v2",
        "camera_ready_v3",
    } <= names

    code = next(r for r in records if r.name == "code_default")
    assert code.available
    assert code.weights == _CODE_DEFAULT
    assert code.dominant_term == "w_collisions"
    assert code.scale_class == "raw"
    assert (
        code.content_sha256
        == hashlib.sha256(
            json.dumps(_CODE_DEFAULT, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
    )

    available_records = [r for r in records if r.available]
    assert all(r.content_sha256 for r in available_records)
    model = next(r for r in records if r.name == "model_canonical_v1")
    assert (
        model.content_sha256
        == hashlib.sha256((repo / "model/snqi_canonical_weights_v1.json").read_bytes()).hexdigest()
    )


def test_code_default_matches_recompute_canonical(tmp_path):
    """The inventory's code-default record mirrors recompute_snqi_weights('canonical')."""
    repo = _make_fixture_repo(tmp_path)
    code = next(r for r in inventory_weight_sets(repo) if r.name == "code_default")
    canonical = dict(recompute_snqi_weights({}, method="canonical").weights)
    assert code.weights == {k: float(canonical[k]) for k in WEIGHT_NAMES}


def test_repo_code_default_is_reported_distinct_from_shipped_canonical_model():
    """Guard the issue #3723 decision state without choosing canonical weights."""
    report = build_inventory_report(REPO_ROOT)
    records = {record.name: record for record in report.records}
    code_default = records["code_default"]
    model_canonical = records["model_canonical_v1"]

    assert code_default.available
    assert model_canonical.available
    assert code_default.declares_canonical
    assert model_canonical.declares_canonical
    assert code_default.weights == _CODE_DEFAULT
    assert code_default.dominant_term == "w_collisions"
    assert model_canonical.dominant_term == "w_jerk"
    assert code_default.normalized_direction() != model_canonical.normalized_direction()

    canonical_conflicts = [
        conflict for conflict in report.conflicts if conflict.kind == "canonical_direction_conflict"
    ]
    assert any(
        conflict.severity == "error" and conflict.sources == ["code_default", "model_canonical_v1"]
        for conflict in canonical_conflicts
    )
    assert report.has_blocking_conflict


def test_conflicting_canonical_sources_detected_fail_closed(tmp_path):
    """Code default (collision) vs model canonical (jerk) raises fail-closed preflight."""
    repo = _make_fixture_repo(tmp_path, model_jerk_dominant=True)
    report = build_inventory_report(repo)

    summary = report.to_dict()["source_summary"]
    assert summary["registered_source_count"] == 5
    assert summary["discovered_shipped_source_count"] == 4
    assert summary["unregistered_shipped_source_count"] == 0
    assert summary["canonical_declaring_sources"] == ["code_default", "model_canonical_v1"]
    assert summary["blocking_conflict_kinds"] == ["canonical_direction_conflict"]

    direction_conflicts = [c for c in report.conflicts if c.kind == "canonical_direction_conflict"]
    assert direction_conflicts, "expected a canonical direction conflict"
    assert set(direction_conflicts[0].sources) == {"code_default", "model_canonical_v1"}
    assert report.has_blocking_conflict

    with pytest.raises(SNQIWeightProvenanceError, match="fail-closed"):
        preflight_snqi_weight_sets(repo, strict=True)


def test_scale_split_reported_across_sources(tmp_path):
    """Raw (code default / model v1) vs normalized (camera-ready) surfaces as warning."""
    repo = _make_fixture_repo(tmp_path)
    report = build_inventory_report(repo)
    assert any(c.kind == "scale_split" for c in report.conflicts)


def test_duplicate_weights_distinct_label_reported(tmp_path):
    """camera_ready_v1 duplicating the jerk-dominant model set is flagged as info."""
    repo = _make_fixture_repo(tmp_path)
    report = build_inventory_report(repo)
    dups = [c for c in report.conflicts if c.kind == "duplicate_weights_distinct_label"]
    assert any(set(c.sources) == {"model_canonical_v1", "camera_ready_v1"} for c in dups)


def test_code_default_vs_shipped_direction_matrix(tmp_path):
    """Every shipped source is compared against the code default."""
    repo = _make_fixture_repo(tmp_path)
    report = build_inventory_report(repo)

    comparisons = compare_code_default_to_shipped_sources(report.records)
    by_source = {comparison.source: comparison for comparison in comparisons}

    assert set(by_source) == {
        "model_canonical_v1",
        "camera_ready_v1",
        "camera_ready_v2",
        "camera_ready_v3",
    }
    assert by_source["model_canonical_v1"].relationship == "different_direction"
    assert by_source["model_canonical_v1"].source_dominant_term == "w_jerk"
    assert by_source["camera_ready_v1"].relationship == "different_direction"
    assert by_source["camera_ready_v2"].relationship == "different_direction"
    assert by_source["camera_ready_v3"].relationship == "different_direction"

    summary_comparisons = report.to_dict()["source_summary"][
        "code_default_shipped_direction_comparisons"
    ]
    assert [entry["source"] for entry in summary_comparisons] == [
        "camera_ready_v1",
        "camera_ready_v2",
        "camera_ready_v3",
        "model_canonical_v1",
    ]


def test_no_blocking_conflict_when_canonical_sources_agree(tmp_path):
    """When the model canonical file matches the code default, preflight passes."""
    repo = _make_fixture_repo(tmp_path, model_jerk_dominant=False)
    report = preflight_snqi_weight_sets(repo, strict=True)
    assert not report.has_blocking_conflict
    assert not any(c.kind == "canonical_direction_conflict" for c in report.conflicts)


def test_missing_canonical_file_fails_closed(tmp_path):
    """A missing canonical-declaring file is surfaced as a blocking load error."""
    repo = _make_fixture_repo(tmp_path)
    (repo / "model/snqi_canonical_weights_v1.json").unlink()
    report = build_inventory_report(repo)
    load_errors = [c for c in report.conflicts if c.kind == "canonical_load_error"]
    assert load_errors and "model_canonical_v1" in load_errors[0].sources
    assert report.has_blocking_conflict

    # Inspection mode (strict=False) must not raise even with a blocking conflict.
    inspected = preflight_snqi_weight_sets(repo, strict=False)
    assert inspected.has_blocking_conflict


def test_non_strict_preflight_does_not_raise(tmp_path):
    """strict=False returns the report regardless of blocking conflicts."""
    repo = _make_fixture_repo(tmp_path, model_jerk_dominant=True)
    report = preflight_snqi_weight_sets(repo, strict=False)
    assert report.has_blocking_conflict  # conflict present...
    # ...but no exception was raised.


def test_inventory_does_not_change_scoring():
    """Diagnostic import/use must not perturb compute_snqi or canonical weights."""
    # Canonical method still returns the pinned code-default values.
    canonical = dict(recompute_snqi_weights({}, method="canonical").weights)
    assert canonical == _CODE_DEFAULT

    # A representative score is unchanged by the presence of the inventory module.
    metrics = {
        "success": 1.0,
        "time_to_goal_norm": 0.5,
        "collisions": 2.0,
        "near_misses": 1.0,
        "comfort_exposure": 0.2,
        "force_exceed_events": 1.0,
        "jerk_mean": 0.5,
    }
    baseline = {
        "collisions": {"med": 0.0, "p95": 4.0},
        "near_misses": {"med": 0.0, "p95": 2.0},
        "force_exceed_events": {"med": 0.0, "p95": 2.0},
        "jerk_mean": {"med": 0.0, "p95": 1.0},
    }
    score = compute_snqi(metrics, _CODE_DEFAULT, baseline)
    # success(1.0)*1 - time(0.8)*0.5 - coll(2.0)*0.5 - near(1.0)*0.5
    #   - comfort(0.5)*0.2 - force(1.5)*0.5 - jerk(0.3)*0.5
    expected = 1.0 - 0.4 - 1.0 - 0.5 - 0.1 - 0.75 - 0.15
    assert score == pytest.approx(expected)


def test_detect_conflicts_is_deterministic(tmp_path):
    """Conflict ordering is stable (errors first) across repeated runs."""
    repo = _make_fixture_repo(tmp_path)
    records = inventory_weight_sets(repo)
    first = [(c.severity, c.kind, tuple(c.sources)) for c in detect_conflicts(records)]
    second = [(c.severity, c.kind, tuple(c.sources)) for c in detect_conflicts(records)]
    assert first == second
    severities = [c.severity for c in detect_conflicts(records)]
    # errors must precede warnings/info
    assert severities == sorted(severities, key={"error": 0, "warning": 1, "info": 2}.get)


def test_unregistered_shipped_weight_file_reported_fail_closed(tmp_path):
    """Bounded discovery reports shipped SNQI weight files missing from the registry."""
    repo = _make_fixture_repo(tmp_path)
    _write_weights(
        tmp_path / "configs/benchmarks/snqi_weights_experimental_v9.json",
        _CODE_DEFAULT,
    )

    report = build_inventory_report(repo)
    unregistered = [r for r in report.records if r.kind == "unregistered_shipped_json"]
    assert len(unregistered) == 1
    assert unregistered[0].relpath == "configs/benchmarks/snqi_weights_experimental_v9.json"
    assert unregistered[0].available
    assert (
        unregistered[0].content_sha256
        == hashlib.sha256(
            (repo / "configs/benchmarks/snqi_weights_experimental_v9.json").read_bytes()
        ).hexdigest()
    )

    summary = report.to_dict()["source_summary"]
    assert summary["registered_source_count"] == 5
    assert summary["discovered_shipped_source_count"] == 5
    assert summary["unregistered_shipped_source_count"] == 1
    assert summary["unregistered_shipped_sources"] == [unregistered[0].name]
    assert summary["blocking_conflict_kinds"] == [
        "canonical_direction_conflict",
        "unregistered_shipped_weight_source",
    ]

    conflicts = [c for c in report.conflicts if c.kind == "unregistered_shipped_weight_source"]
    assert conflicts
    assert conflicts[0].severity == "error"
    assert conflicts[0].sources == [unregistered[0].name]
    assert report.has_blocking_conflict
    with pytest.raises(SNQIWeightProvenanceError, match="unregistered_shipped_weight_source"):
        preflight_snqi_weight_sets(repo, strict=True)
