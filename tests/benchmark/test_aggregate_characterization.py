"""Characterization baseline tests for ``robot_sf/benchmark/aggregate.py``.

These tests pin the *current observable behavior* of the benchmark aggregation
helpers on small synthetic inputs. They are intentionally table-driven and
assert exact golden values (including the documented edge cases: empty inputs,
single episodes, and NaN values where finite-guards exist).

Purpose (issue #4874, Refs #4770): lock a behavioral baseline so the
post-submission refactor wave can prove behavior-preservation by re-running
these tests and observing identical results. If a test reveals a genuine bug,
do NOT fix it here — document it and file a separate fix issue.

These tests are additive: they do not duplicate the SNQI-recompute behavior
coverage in ``test_aggregate_snqi_recompute.py``.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

import pytest

from robot_sf.benchmark import aggregate
from robot_sf.benchmark.aggregate import (
    build_observation_track_meta,
    compute_aggregates,
    flatten_metrics,
    normalize_observation_track_mode,
    read_jsonl,
    resolve_benchmark_track,
    write_episode_csv,
)
from robot_sf.benchmark.errors import (
    AggregationMetadataError,
    EpisodeRecordInputError,
)

if TYPE_CHECKING:
    from pathlib import Path

# Radius-aware clearance uses ``robot_radius + ped_radius`` (defaults 1.0 + 0.4).


# ---------------------------------------------------------------------------
# flatten_metrics
# ---------------------------------------------------------------------------


def test_flatten_metrics_extracts_id_fields_and_flat_numbers() -> None:
    """Top-level id fields and scalar metrics pass through unchanged."""
    rec = {
        "episode_id": "ep1",
        "scenario_id": "sc1",
        "seed": 7,
        "metrics": {"success": 1.0, "collisions": 0.0, "avg_speed": 0.9},
    }
    flat = flatten_metrics(rec)
    assert flat["episode_id"] == "ep1"
    assert flat["scenario_id"] == "sc1"
    assert flat["seed"] == 7
    assert flat["success"] == 1.0
    assert flat["collisions"] == 0.0
    assert flat["avg_speed"] == 0.9


def test_flatten_metrics_promotes_force_quantiles_to_top_level_keys() -> None:
    """The nested ``force_quantiles`` block is flattened to ``force_q*`` keys."""
    rec = {
        "episode_id": "ep1",
        "scenario_id": "sc1",
        "seed": 1,
        "metrics": {"force_quantiles": {"q50": 1.1, "q90": 2.2, "q95": 3.3}},
    }
    flat = flatten_metrics(rec)
    assert flat["force_q50"] == 1.1
    assert flat["force_q90"] == 2.2
    assert flat["force_q95"] == 3.3
    # The original nested block key is not retained as a column.
    assert "force_quantiles" not in flat


def test_flatten_metrics_drops_distributional_disruption_block() -> None:
    """The ``distributional_disruption`` nested block is intentionally dropped."""
    rec = {"metrics": {"distributional_disruption": {"foo": 1.0}, "avg_speed": 0.5}}
    flat = flatten_metrics(rec)
    assert "distributional_disruption" not in flat
    assert flat["avg_speed"] == 0.5


def test_flatten_metrics_maps_schema_blocks_to_prefixed_columns() -> None:
    """Each schema-backed nested block maps to a documented prefix family."""
    rec = {
        "episode_id": "ep1",
        "scenario_id": "sc1",
        "seed": 7,
        "metrics": {
            "pedestrian_impact": {
                "canonical_reductions": {"accel_delta_mean": 0.5, "turn_rate_delta_mean": 0.2},
                "sample_counts": {
                    "pedestrians": 3,
                    "near_samples": 10,
                    "far_samples": 5,
                    "near_sample_frac": 0.666,
                },
            },
            "social_acceptability": {
                "available": 1.0,
                "parameters": {"proxemic_radius_m": 1.2},
                "sample_counts": {"pedestrians": 3, "timesteps": 100},
                "proxemic": {"intrusion_frac": 0.1, "min_clearance_m": 0.3},
            },
            "human_interaction_proxy": {
                "available": 1.0,
                "parameters": {"proxemic_radius_m": 1.2, "yield_speed_mps": 0.15},
                "sample_counts": {"pedestrians": 3, "timesteps": 100},
                "canonical_reductions": {"time_to_yield_s": 1.5},
            },
            "social_mini_game": {
                "status": "available",
                "rows": [
                    {
                        "metric": "cooperation",
                        "status": "available",
                        "value": 0.42,
                        "support_count": 2,
                    }
                ],
            },
            "clear_tracking_uncertainty": {
                "enabled": True,
                "mota": 0.8,
                "motp_m": 0.1,
                "counts": {"ground_truth": 5, "detections": 4},
            },
            "inter_robot": {"inter_robot_min_distance": 2.0},
        },
    }
    flat = flatten_metrics(rec)
    # pedestrian-impact reductions
    assert flat["ped_impact_accel_delta_mean"] == 0.5
    assert flat["ped_impact_near_sample_frac"] == 0.666
    assert flat["ped_impact_ped_count"] == 3
    # social-acceptability proxemic
    assert flat["social_proxemic_radius_m"] == 1.2
    assert flat["social_proxemic_intrusion_frac"] == 0.1
    assert flat["social_proxemic_ped_count"] == 3
    # human-interaction proxy reductions (keys carried verbatim when present)
    assert flat["human_proxy_available"] == 1.0
    assert flat["human_proxy_proxemic_radius_m"] == 1.2
    assert flat["human_proxy_ped_count"] == 3
    assert flat["time_to_yield_s"] == 1.5
    # social mini-game rows get a per-metric prefix
    assert flat["social_mini_game_cooperation"] == 0.42
    assert flat["social_mini_game_cooperation_support_count"] == 2
    assert flat["social_mini_game_cooperation_status"] == "available"
    # CLEAR tracking
    assert flat["clear_tracking_enabled"] is True
    assert flat["clear_mota"] == 0.8
    assert flat["clear_ground_truth_count"] == 5
    assert flat["clear_detection_count"] == 4
    # inter-robot block is flattened key-for-key
    assert flat["inter_robot_min_distance"] == 2.0


def test_flatten_metrics_empty_and_none_metrics_yield_all_none_columns() -> None:
    """Empty records and ``metrics=None`` flatten to all-None column rows.

    The pedestrian-impact / social-acceptability / human-proxy / force-quantile
    prefixes are always emitted (so CSV headers stay stable), while CLEAR and
    Social-Mini-Game columns are conditional and absent from an empty row.
    """
    for rec in ({}, {"metrics": None}):
        flat = flatten_metrics(rec)
        assert flat["episode_id"] is None
        assert flat["scenario_id"] is None
        assert flat["seed"] is None
        assert flat["force_q50"] is None
        assert flat["ped_impact_accel_delta_mean"] is None
        assert flat["social_proxemic_radius_m"] is None
        assert flat["human_proxy_available"] is None
        # CLEAR and Social-Mini-Game columns are conditional on the block content.
        assert "clear_mota" not in flat
        assert "social_mini_game_status" not in flat


# ---------------------------------------------------------------------------
# compute_aggregates
# ---------------------------------------------------------------------------

# Records exercising bool->float success coercion, two groups, one metric.
_TWO_GROUP_RECORDS: list[dict[str, Any]] = [
    {
        "episode_id": "a1",
        "scenario_id": "s1",
        "seed": 1,
        "algo": "A",
        "metrics": {"success": True, "collisions": 0.0, "avg_speed": 1.0},
    },
    {
        "episode_id": "a2",
        "scenario_id": "s1",
        "seed": 2,
        "algo": "A",
        "metrics": {"success": False, "collisions": 1.0, "avg_speed": 3.0},
    },
    {
        "episode_id": "b1",
        "scenario_id": "s1",
        "seed": 1,
        "algo": "B",
        "metrics": {"success": True, "collisions": 0.0, "avg_speed": 2.0},
    },
]


def test_compute_aggregates_coerces_bool_success_and_computes_exact_stats() -> None:
    """``success`` bools are coerced to 0/1 and aggregated with mean/median/p95."""
    agg = compute_aggregates(_TWO_GROUP_RECORDS, group_by="algo", fallback_group_by="scenario_id")
    # Group A: success {1.0, 0.0}; avg_speed {1.0, 3.0}; collisions {0.0, 1.0}
    assert agg["A"]["success"]["mean"] == pytest.approx(0.5)
    # numpy linear-interp p95 of [0.0, 1.0] == 0.95
    assert agg["A"]["success"]["p95"] == pytest.approx(0.95)
    assert agg["A"]["success"]["median"] == pytest.approx(0.5)
    assert agg["A"]["avg_speed"]["mean"] == pytest.approx(2.0)
    assert agg["A"]["avg_speed"]["median"] == pytest.approx(2.0)
    assert agg["A"]["collisions"]["mean"] == pytest.approx(0.5)
    # Group B: single successful episode.
    assert agg["B"]["success"]["mean"] == pytest.approx(1.0)
    assert agg["B"]["success"]["p95"] == pytest.approx(1.0)


def test_compute_aggregates_meta_block_shape_and_defaults() -> None:
    """The additive ``_meta`` block carries grouping, threshold, and track info."""
    agg = compute_aggregates(_TWO_GROUP_RECORDS, group_by="algo")
    meta = agg["_meta"]
    assert set(meta) >= {
        "group_by",
        "effective_group_key",
        "missing_algorithms",
        "warnings",
        "metric_parameters",
        "observation_tracks",
    }
    assert meta["group_by"] == "algo"
    assert "scenario_params.algo" in meta["effective_group_key"]
    assert meta["missing_algorithms"] == []
    assert meta["warnings"] == []
    assert meta["observation_tracks"]["mode"] == "strict"
    assert meta["observation_tracks"]["selected_track"] == "unspecified"


def test_compute_aggregates_reports_missing_expected_algorithms() -> None:
    """``expected_algorithms`` not present in records surface as missing + warning."""
    agg = compute_aggregates(
        _TWO_GROUP_RECORDS,
        group_by="algo",
        expected_algorithms={"A", "B", "C"},
    )
    assert agg["_meta"]["missing_algorithms"] == ["C"]
    assert any("C" in w for w in agg["_meta"]["warnings"])


def test_compute_aggregates_empty_records_returns_meta_only() -> None:
    """An empty record set aggregates to just the ``_meta`` block (no groups)."""
    agg = compute_aggregates([])
    assert list(agg.keys()) == ["_meta"]
    groups = [k for k in agg if k != "_meta"]
    assert groups == []


def test_compute_aggregates_falls_back_to_scenario_id_when_algo_missing() -> None:
    """Records lacking algorithm metadata group by the fallback key."""
    records = [
        {"episode_id": "e1", "scenario_id": "scX", "seed": 1, "metrics": {"avg_speed": 1.0}},
        {"episode_id": "e2", "scenario_id": "scY", "seed": 1, "metrics": {"avg_speed": 2.0}},
    ]
    agg = compute_aggregates(records, group_by="algo", fallback_group_by="scenario_id")
    assert "scX" in agg and "scY" in agg


def test_compute_aggregates_ignores_nan_metric_values_via_nanmean() -> None:
    """NaN metric values are ignored by the NaN-aware mean/median aggregators."""
    records = [
        {
            "episode_id": "a1",
            "scenario_id": "s1",
            "seed": 1,
            "algo": "A",
            "metrics": {"avg_speed": 2.0},
        },
        {
            "episode_id": "a2",
            "scenario_id": "s1",
            "seed": 2,
            "algo": "A",
            "metrics": {"avg_speed": float("nan")},
        },
        {
            "episode_id": "a3",
            "scenario_id": "s1",
            "seed": 3,
            "algo": "A",
            "metrics": {"avg_speed": 4.0},
        },
    ]
    agg = compute_aggregates(records, group_by="algo")
    # nan ignored -> mean of {2.0, 4.0}
    assert agg["A"]["avg_speed"]["mean"] == pytest.approx(3.0)
    assert agg["A"]["avg_speed"]["median"] == pytest.approx(3.0)


def test_compute_aggregates_single_episode_statistics_equal_the_value() -> None:
    """A single-episode group has mean == median == p95 == the value."""
    records = [
        {
            "episode_id": "a1",
            "scenario_id": "s1",
            "seed": 1,
            "algo": "A",
            "metrics": {"avg_speed": 1.5},
        },
    ]
    agg = compute_aggregates(records, group_by="algo")
    stats = agg["A"]["avg_speed"]
    assert stats["mean"] == pytest.approx(1.5)
    assert stats["median"] == pytest.approx(1.5)
    assert stats["p95"] == pytest.approx(1.5)


# ---------------------------------------------------------------------------
# observation-track helpers
# ---------------------------------------------------------------------------


def test_resolve_benchmark_track_falls_back_to_unspecified() -> None:
    """Records without any track declaration resolve to ``"unspecified"``."""
    assert resolve_benchmark_track({"episode_id": "x"}) == "unspecified"
    assert (
        resolve_benchmark_track({"scenario_params": {"benchmark_track": "  odometry  "}})
        == "odometry"
    )


def test_normalize_observation_track_mode_canonicalizes_inputs() -> None:
    """Mode strings are lowercased and dash-normalized; invalid modes raise."""
    assert normalize_observation_track_mode("strict") == "strict"
    assert normalize_observation_track_mode("Diagnostic-Cross-Track") == "diagnostic_cross_track"
    with pytest.raises(ValueError):
        normalize_observation_track_mode("bogus")


def test_build_observation_track_meta_single_track_no_mixed_caveat() -> None:
    """A single declared track yields no mixed-track caveat."""
    meta = build_observation_track_meta([{"benchmark_track": "A"}, {"benchmark_track": "A"}])
    assert meta["tracks"] == {"A": 2}
    assert meta["mixed_tracks"] is False
    assert meta["selected_track"] == "A"
    assert meta["caveat_record_count"] == 0
    assert "cross_track_caveat" not in meta


def test_build_observation_track_meta_counts_caveat_statuses() -> None:
    """Fallback/degraded/failed statuses are counted as caveats."""
    meta = build_observation_track_meta(
        [
            {"benchmark_track": "A", "availability_status": "fallback"},
            {"benchmark_track": "A", "availability_status": "ok"},
            {"benchmark_track": "A", "algorithm_metadata": {"execution_mode": "degraded"}},
        ]
    )
    assert meta["caveat_record_count"] == 2


def test_build_observation_track_meta_mixed_tracks_adds_cross_track_caveat() -> None:
    """Mixed tracks set ``mixed_tracks`` and add a cross-track caveat."""
    meta = build_observation_track_meta([{"benchmark_track": "A"}, {"benchmark_track": "B"}])
    assert meta["mixed_tracks"] is True
    assert meta["selected_track"] is None
    assert "cross_track_caveat" in meta


def test_compute_aggregates_strict_mode_rejects_mixed_tracks() -> None:
    """Mixed benchmark tracks cannot be pooled under the default strict policy."""
    mixed = [
        {"episode_id": "m1", "benchmark_track": "A", "algo": "X", "metrics": {}},
        {"episode_id": "m2", "benchmark_track": "B", "algo": "X", "metrics": {}},
    ]
    with pytest.raises(AggregationMetadataError):
        compute_aggregates(mixed, group_by="algo")


def test_compute_aggregates_diagnostic_cross_track_keeps_groups_separate() -> None:
    """Diagnostic mode pools mixed tracks but labels each group with its track."""
    mixed = [
        {"episode_id": "m1", "benchmark_track": "A", "algo": "X", "metrics": {}},
        {"episode_id": "m2", "benchmark_track": "B", "algo": "X", "metrics": {}},
    ]
    agg = compute_aggregates(
        mixed, group_by="algo", observation_track_mode="diagnostic-cross-track"
    )
    groups = sorted(k for k in agg if k != "_meta")
    assert groups == ["A :: X", "B :: X"]


# ---------------------------------------------------------------------------
# write_episode_csv
# ---------------------------------------------------------------------------


def test_write_episode_csv_header_orders_id_fields_first(tmp_path: Path) -> None:
    """CSV header is ``[episode_id, scenario_id, seed]`` then sorted metric keys."""
    out = tmp_path / "out.csv"
    records = [
        {
            "episode_id": "a1",
            "scenario_id": "s1",
            "seed": 1,
            "algo": "A",
            "metrics": {"success": 1.0, "collisions": 2},
        },
        {
            "episode_id": "b1",
            "scenario_id": "s2",
            "seed": 9,
            "algo": "B",
            "metrics": {"success": 0.0, "near_misses": 1},
        },
    ]
    returned = write_episode_csv(records, str(out))
    assert returned == str(out)
    text = out.read_text(encoding="utf-8").splitlines()
    header = text[0].split(",")
    assert header[:3] == ["episode_id", "scenario_id", "seed"]
    # All non-id keys are present and sorted.
    metric_keys = header[3:]
    assert metric_keys == sorted(metric_keys)
    # The union of metrics across both rows is represented.
    assert "success" in metric_keys
    assert "collisions" in metric_keys
    assert "near_misses" in metric_keys


def test_write_episode_csv_rows_carry_values_and_blank_for_missing(tmp_path: Path) -> None:
    """CSV rows carry present values and leave absent metrics as empty cells."""
    out = tmp_path / "out.csv"
    write_episode_csv(
        [
            {
                "episode_id": "a1",
                "scenario_id": "s1",
                "seed": 1,
                "metrics": {"collisions": 2, "success": 1.0},
            },
            {
                "episode_id": "b1",
                "scenario_id": "s2",
                "seed": 2,
                "metrics": {"success": 0.0, "near_misses": 1},
            },
        ],
        str(out),
    )
    rows = list(out.read_text(encoding="utf-8").splitlines())
    header = {col: i for i, col in enumerate(rows[0].split(","))}
    a_row = rows[1].split(",")
    assert a_row[header["episode_id"]] == "a1"
    assert a_row[header["collisions"]] == "2"
    assert a_row[header["success"]] == "1.0"
    # ``near_misses`` is absent from the first record -> empty cell.
    assert a_row[header["near_misses"]] == ""


# ---------------------------------------------------------------------------
# read_jsonl
# ---------------------------------------------------------------------------


def _write_jsonl(path: Path, lines: list[str]) -> Path:
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def test_read_jsonl_strict_skips_blank_lines_and_parses_records(tmp_path: Path) -> None:
    """Blank and whitespace-only lines are skipped; valid records parse in strict mode."""
    path = _write_jsonl(
        tmp_path / "r.jsonl",
        ['{"episode_id": "x"}', "", "   ", '{"episode_id": "y"}'],
    )
    records = read_jsonl(path, strict=True)
    assert [r["episode_id"] for r in records] == ["x", "y"]


def test_read_jsonl_strict_raises_on_malformed_line(tmp_path: Path) -> None:
    """A malformed JSONL line raises ``EpisodeRecordInputError`` in strict mode."""
    path = _write_jsonl(tmp_path / "r.jsonl", ['{"episode_id": "x"}', '{"bad json'])
    with pytest.raises(EpisodeRecordInputError):
        read_jsonl(path, strict=True)


def test_read_jsonl_non_strict_skips_malformed_lines(tmp_path: Path) -> None:
    """Non-strict mode skips malformed lines and returns only parsed records."""
    path = _write_jsonl(
        tmp_path / "r.jsonl", ['{"episode_id": "x"}', '{"bad json', '{"episode_id": "y"}']
    )
    records = read_jsonl(path, strict=False)
    assert [r["episode_id"] for r in records] == ["x", "y"]


def test_read_jsonl_strict_raises_on_missing_path(tmp_path: Path) -> None:
    """A missing file raises ``EpisodeRecordInputError`` in strict mode."""
    with pytest.raises(EpisodeRecordInputError):
        read_jsonl(tmp_path / "nope.jsonl", strict=True)


def test_read_jsonl_non_strict_silently_skips_missing_path(tmp_path: Path) -> None:
    """Non-strict mode silently skips a missing path and returns an empty list."""
    assert read_jsonl(tmp_path / "nope.jsonl", strict=False) == []


def test_read_jsonl_accepts_single_path_argument(tmp_path: Path) -> None:
    """A single path argument is accepted, not only sequences of paths."""
    path = _write_jsonl(tmp_path / "r.jsonl", ['{"episode_id": "x"}'])
    assert read_jsonl(path, strict=True)[0]["episode_id"] == "x"


def test_aggregate_module_exports_stable_public_surface() -> None:
    """The ``__all__`` exports are the locked public surface of aggregate.py."""
    expected = {
        "build_observation_track_meta",
        "compute_aggregates",
        "compute_aggregates_with_ci",
        "ensure_observation_track_policy",
        "flatten_metrics",
        "normalize_observation_track_mode",
        "observation_track_group_label",
        "read_jsonl",
        "resolve_benchmark_track",
        "resolve_report_group_key",
        "write_episode_csv",
    }
    assert expected <= set(aggregate.__all__)


# Keep json imported symbol referenced for static analyzers in case of expansion.
_ = json
