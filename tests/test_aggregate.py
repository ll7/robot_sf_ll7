"""TODO docstring. Document this module."""

from __future__ import annotations

from pathlib import Path

import pytest

from robot_sf.benchmark.aggregate import (
    _numeric_items,
    compute_aggregates,
    compute_aggregates_with_ci,
    flatten_metrics,
    read_jsonl,
    write_episode_csv,
)
from robot_sf.benchmark.errors import AggregationMetadataError, EpisodeRecordInputError
from robot_sf.benchmark.runner import run_batch

SCHEMA_PATH = "robot_sf/benchmark/schemas/episode.schema.v1.json"


def _make_sample_jsonl(tmp_path: Path) -> Path:
    # Use run_batch to generate 3 episodes across 2 algos (via scenario_params.algo)
    """TODO docstring. Document this function.

    Args:
        tmp_path: TODO docstring.

    Returns:
        TODO docstring.
    """
    scenarios = [
        {
            "id": "agg-uni-low-open-A",
            "density": "low",
            "flow": "uni",
            "obstacle": "open",
            "groups": 0.0,
            "speed_var": "low",
            "goal_topology": "point",
            "robot_context": "embedded",
            "repeats": 2,
            "algo": "A",
        },
        {
            "id": "agg-uni-low-open-B",
            "density": "low",
            "flow": "uni",
            "obstacle": "open",
            "groups": 0.0,
            "speed_var": "low",
            "goal_topology": "point",
            "robot_context": "embedded",
            "repeats": 1,
            "algo": "B",
        },
    ]
    out_file = tmp_path / "episodes.jsonl"
    # run without forces for speed
    run_batch(
        scenarios,
        out_path=out_file,
        schema_path=SCHEMA_PATH,
        base_seed=10,
        horizon=8,
        dt=0.1,
        record_forces=False,
        append=False,
    )
    return out_file


def test_read_and_flatten_and_write_csv(tmp_path: Path):
    """TODO docstring. Document this function.

    Args:
        tmp_path: TODO docstring.
    """
    jsonl_path = _make_sample_jsonl(tmp_path)
    recs = read_jsonl(jsonl_path)
    assert len(recs) == 3
    rows = [flatten_metrics(r) for r in recs]
    assert all("episode_id" in r for r in rows)
    # Write CSV
    csv_path = tmp_path / "episodes.csv"
    out = write_episode_csv(recs, csv_path)
    assert Path(out).exists()
    text = Path(out).read_text(encoding="utf-8")
    assert text.splitlines()[0].startswith("episode_id,scenario_id,seed")


def test_read_jsonl_fails_closed_on_malformed_line(tmp_path: Path) -> None:
    """Aggregate input loading should not silently drop malformed benchmark records."""
    path = tmp_path / "episodes_bad.jsonl"
    path.write_text('{"episode_id":"ok","metrics":{}}\n{bad json\n', encoding="utf-8")

    with pytest.raises(EpisodeRecordInputError) as excinfo:
        read_jsonl(path)

    message = str(excinfo.value)
    assert "malformed_lines=1" in message
    assert f"{path}:2" in message


def test_read_jsonl_fails_closed_on_missing_path(tmp_path: Path) -> None:
    """Aggregate input loading should report missing benchmark inputs by default."""
    path = tmp_path / "missing.jsonl"

    with pytest.raises(EpisodeRecordInputError) as excinfo:
        read_jsonl(path)

    message = str(excinfo.value)
    assert "missing_paths=1" in message
    assert str(path) in message


def test_read_jsonl_best_effort_mode_is_explicit(tmp_path: Path) -> None:
    """Exploratory aggregate callers can opt into partial parsing explicitly."""
    path = tmp_path / "episodes_bad.jsonl"
    path.write_text('{"episode_id":"ok","metrics":{}}\n{bad json\n', encoding="utf-8")

    assert [record["episode_id"] for record in read_jsonl(path, strict=False)] == ["ok"]


def test_aggregate_cli_reports_malformed_input(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The benchmark-facing aggregate command should fail closed with diagnostics."""
    from robot_sf.benchmark import cli

    src = tmp_path / "episodes_bad.jsonl"
    src.write_text('{"episode_id":"ok","metrics":{}}\n{bad json\n', encoding="utf-8")
    errors: list[str] = []

    def _record_exception(message: str, *args: object, **_kwargs: object) -> None:
        errors.append(message % args)

    monkeypatch.setattr(cli.logging, "exception", _record_exception)

    exit_code = cli.cli_main(
        [
            "aggregate",
            "--in",
            str(src),
            "--out",
            str(tmp_path / "summary.json"),
        ]
    )

    assert exit_code == 2
    assert errors
    assert "malformed_lines=1" in errors[0]
    assert f"{src}:2" in errors[0]


def test_aggregation_metadata_error_to_dict_includes_optional_context() -> None:
    """Structured aggregation errors should expose optional context fields."""
    error = AggregationMetadataError(
        "missing metadata",
        episode_id="episode-1",
        missing_fields=["scenario_params.algo", "algo"],
        advice="add algorithm metadata",
    )

    assert error.to_dict() == {
        "message": "missing metadata",
        "episode_id": "episode-1",
        "missing_fields": ["scenario_params.algo", "algo"],
        "advice": "add algorithm metadata",
    }


def test_compute_aggregates_group_by_algo(tmp_path: Path):
    """TODO docstring. Document this function.

    Args:
        tmp_path: TODO docstring.
    """
    jsonl_path = _make_sample_jsonl(tmp_path)
    recs = read_jsonl(jsonl_path)
    # We stored algo at the top-level of scenario params; group path is scenario_params.algo
    summary = compute_aggregates(recs, group_by="scenario_params.algo")
    # Should contain two groups A and B plus metadata
    algorithm_groups = {k for k in summary.keys() if k != "_meta"}
    assert algorithm_groups == {"A", "B"}
    # Should have _meta section
    assert "_meta" in summary
    # Each group should have numeric aggregates present for some core metric
    for group_name, metrics in summary.items():
        if group_name == "_meta":
            continue  # Skip metadata section
        assert "time_to_goal_norm" in metrics
        assert set(metrics["time_to_goal_norm"].keys()) == {"mean", "median", "p95"}


def test_compute_aggregates_with_ci_shape_and_determinism(tmp_path: Path):
    """TODO docstring. Document this function.

    Args:
        tmp_path: TODO docstring.
    """
    jsonl_path = _make_sample_jsonl(tmp_path)
    recs = read_jsonl(jsonl_path)
    # Compute with CIs
    summary_ci = compute_aggregates_with_ci(
        recs,
        group_by="scenario_params.algo",
        bootstrap_samples=200,
        bootstrap_confidence=0.90,
        bootstrap_seed=123,
    )
    # Basic shape: groups and keys
    algorithm_groups = {k for k in summary_ci.keys() if k != "_meta"}
    assert algorithm_groups == {"A", "B"}
    # Should have _meta section
    assert "_meta" in summary_ci
    any_group = next(iter(g for k, g in summary_ci.items() if k != "_meta"))
    # Ensure a known metric exists
    assert "time_to_goal_norm" in any_group
    m = any_group["time_to_goal_norm"]
    # Base stats still present
    assert {"mean", "median", "p95"}.issubset(set(m.keys()))
    # CI keys present and are [low, high]
    for key in ("mean_ci", "median_ci", "p95_ci"):
        assert key in m
        ci = m[key]
        assert isinstance(ci, list) and len(ci) == 2
        assert all(isinstance(v, float) for v in ci)
        assert ci[0] <= ci[1]

    # Determinism with same seed
    summary_ci_2 = compute_aggregates_with_ci(
        recs,
        group_by="scenario_params.algo",
        bootstrap_samples=200,
        bootstrap_confidence=0.90,
        bootstrap_seed=123,
    )
    assert summary_ci == summary_ci_2


def test_compute_aggregates_flattens_pedestrian_impact_block() -> None:
    """Schema-backed pedestrian-impact reductions should aggregate without custom parsing."""
    records = [
        {
            "episode_id": "ep-1",
            "scenario_id": "sc-1",
            "seed": 1,
            "algo": "planner-a",
            "metric_parameters": {
                "threshold_signature": "default",
                "threshold_profile": {
                    "profile_id": "default",
                    "collision_distance_m": 0.3,
                    "near_miss_distance_m": 0.6,
                    "comfort_force_threshold": 2.0,
                },
            },
            "metrics": {
                "success": True,
                "pedestrian_impact": {
                    "schema_version": "pedestrian-impact.v1",
                    "parameters": {"near_radius_m": 2.0, "window_steps": 1},
                    "units": {
                        "accel": "m/s^2",
                        "turn_rate": "rad/s",
                        "near_radius": "m",
                        "sample_counts": "samples",
                        "sample_fraction": "fraction",
                    },
                    "sample_counts": {
                        "pedestrians": 1,
                        "near_samples": 4,
                        "far_samples": 5,
                        "near_sample_frac": 4.0 / 9.0,
                    },
                    "canonical_reductions": {
                        "accel_delta_mean": 0.5,
                        "accel_delta_median": 0.4,
                        "accel_delta_valid_pedestrians": 1,
                        "turn_rate_delta_mean": 0.2,
                        "turn_rate_delta_median": 0.1,
                        "turn_rate_delta_valid_pedestrians": 1,
                    },
                },
            },
        },
        {
            "episode_id": "ep-2",
            "scenario_id": "sc-1",
            "seed": 2,
            "algo": "planner-a",
            "metric_parameters": {
                "threshold_signature": "default",
                "threshold_profile": {
                    "profile_id": "default",
                    "collision_distance_m": 0.3,
                    "near_miss_distance_m": 0.6,
                    "comfort_force_threshold": 2.0,
                },
            },
            "metrics": {
                "success": True,
                "pedestrian_impact": {
                    "schema_version": "pedestrian-impact.v1",
                    "parameters": {"near_radius_m": 2.0, "window_steps": 1},
                    "units": {
                        "accel": "m/s^2",
                        "turn_rate": "rad/s",
                        "near_radius": "m",
                        "sample_counts": "samples",
                        "sample_fraction": "fraction",
                    },
                    "sample_counts": {
                        "pedestrians": 1,
                        "near_samples": 6,
                        "far_samples": 6,
                        "near_sample_frac": 0.5,
                    },
                    "canonical_reductions": {
                        "accel_delta_mean": 1.5,
                        "accel_delta_median": 1.4,
                        "accel_delta_valid_pedestrians": 1,
                        "turn_rate_delta_mean": 0.6,
                        "turn_rate_delta_median": 0.5,
                        "turn_rate_delta_valid_pedestrians": 1,
                    },
                },
            },
        },
    ]

    summary = compute_aggregates(records, group_by="algo")

    metrics = summary["planner-a"]
    assert metrics["ped_impact_accel_delta_mean"]["mean"] == 1.0
    assert metrics["ped_impact_turn_rate_delta_mean"]["mean"] == 0.4
    assert metrics["ped_impact_near_samples"]["mean"] == 5.0


def test_compute_aggregates_flattens_social_acceptability_block() -> None:
    """Schema-backed social-acceptability pilot reductions should aggregate as scalars."""
    records = [
        {
            "episode_id": "ep-1",
            "scenario_id": "sc-1",
            "seed": 1,
            "algo": "planner-a",
            "metric_parameters": {
                "threshold_signature": "default",
                "threshold_profile": {
                    "profile_id": "default",
                    "collision_distance_m": 0.3,
                    "near_miss_distance_m": 0.6,
                    "comfort_force_threshold": 2.0,
                },
            },
            "metrics": {
                "success": True,
                "social_acceptability": {
                    "schema_version": "social-acceptability-pilot.v1",
                    "status": "exploratory",
                    "parameters": {"proxemic_radius_m": 1.2},
                    "available": True,
                    "sample_counts": {"pedestrians": 1, "timesteps": 2},
                    "proxemic": {
                        "intrusion_steps": 2,
                        "intrusion_frac": 0.5,
                        "intrusion_area_m_s": 0.9,
                        "min_clearance_m": 0.1,
                    },
                },
            },
        },
        {
            "episode_id": "ep-2",
            "scenario_id": "sc-1",
            "seed": 2,
            "algo": "planner-a",
            "metric_parameters": {
                "threshold_signature": "default",
                "threshold_profile": {
                    "profile_id": "default",
                    "collision_distance_m": 0.3,
                    "near_miss_distance_m": 0.6,
                    "comfort_force_threshold": 2.0,
                },
            },
            "metrics": {
                "success": True,
                "social_acceptability": {
                    "schema_version": "social-acceptability-pilot.v1",
                    "status": "exploratory",
                    "parameters": {"proxemic_radius_m": 1.2},
                    "available": True,
                    "sample_counts": {"pedestrians": 1, "timesteps": 1},
                    "proxemic": {
                        "intrusion_steps": 1,
                        "intrusion_frac": 0.25,
                        "intrusion_area_m_s": 0.3,
                        "min_clearance_m": 0.4,
                    },
                },
            },
        },
    ]

    flat = flatten_metrics(records[0])
    assert "social_acceptability" not in flat
    assert flat["social_proxemic_intrusion_area_m_s"] == 0.9

    summary = compute_aggregates(records, group_by="algo")

    metrics = summary["planner-a"]
    assert metrics["social_proxemic_intrusion_steps"]["mean"] == 1.5
    assert metrics["social_proxemic_intrusion_frac"]["mean"] == 0.375
    assert metrics["social_proxemic_intrusion_area_m_s"]["mean"] == 0.6


def test_compute_aggregates_flattens_human_interaction_proxy_block() -> None:
    """Schema-backed human-interaction proxy reductions should aggregate as scalars."""
    records = [
        {
            "episode_id": "ep-1",
            "scenario_id": "sc-1",
            "seed": 1,
            "algo": "planner-a",
            "metric_parameters": {
                "threshold_signature": "default",
                "threshold_profile": {
                    "profile_id": "default",
                    "collision_distance_m": 0.3,
                    "near_miss_distance_m": 0.6,
                    "comfort_force_threshold": 2.0,
                },
            },
            "metrics": {
                "success": True,
                "human_interaction_proxy": {
                    "schema_version": "human-interaction-proxy.v1",
                    "status": "simulation_proxy",
                    "parameters": {"proxemic_radius_m": 1.2, "yield_speed_mps": 0.1},
                    "available": True,
                    "sample_counts": {"pedestrians": 1, "timesteps": 4},
                    "canonical_reductions": {
                        "human_discomfort_exposure_m_s": 0.9,
                        "intrusion_duration_s": 1.0,
                        "time_to_yield_s": 0.5,
                        "robot_yield_distance_m": 1.5,
                        "pedestrian_path_deviation_proxy_m": 0.2,
                        "group_split_intrusion_available": False,
                    },
                },
            },
        },
        {
            "episode_id": "ep-2",
            "scenario_id": "sc-1",
            "seed": 2,
            "algo": "planner-a",
            "metric_parameters": {
                "threshold_signature": "default",
                "threshold_profile": {
                    "profile_id": "default",
                    "collision_distance_m": 0.3,
                    "near_miss_distance_m": 0.6,
                    "comfort_force_threshold": 2.0,
                },
            },
            "metrics": {
                "success": True,
                "human_interaction_proxy": {
                    "schema_version": "human-interaction-proxy.v1",
                    "status": "simulation_proxy",
                    "parameters": {"proxemic_radius_m": 1.2, "yield_speed_mps": 0.1},
                    "available": True,
                    "sample_counts": {"pedestrians": 1, "timesteps": 4},
                    "canonical_reductions": {
                        "human_discomfort_exposure_m_s": 0.3,
                        "intrusion_duration_s": 0.5,
                        "time_to_yield_s": 0.0,
                        "robot_yield_distance_m": 2.0,
                        "pedestrian_path_deviation_proxy_m": 0.4,
                        "group_split_intrusion_available": False,
                    },
                },
            },
        },
    ]

    flat = flatten_metrics(records[0])
    assert "human_interaction_proxy" not in flat
    assert flat["human_discomfort_exposure_m_s"] == 0.9
    assert flat["human_proxy_yield_speed_mps"] == 0.1

    summary = compute_aggregates(records, group_by="algo")

    metrics = summary["planner-a"]
    assert metrics["human_discomfort_exposure_m_s"]["mean"] == 0.6
    assert metrics["intrusion_duration_s"]["mean"] == 0.75
    assert metrics["time_to_yield_s"]["mean"] == 0.25
    assert metrics["pedestrian_path_deviation_proxy_m"]["mean"] == pytest.approx(0.3)


def test_compute_aggregates_flattens_social_mini_game_available_rows() -> None:
    """Available Social Mini-Game row values should aggregate without nested payload leakage."""
    records = [
        {
            "episode_id": "ep-1",
            "scenario_id": "sc-1",
            "seed": 1,
            "algo": "planner-a",
            "metrics": {
                "success": True,
                "social_mini_game": {
                    "schema_version": "social-mini-game-metrics.v1",
                    "status": "diagnostic",
                    "mechanism_family": "doorway",
                    "rows": [
                        {
                            "metric": "deadlock_frequency",
                            "status": "available",
                            "unit": "events_per_episode",
                            "denominator": "one episode",
                            "value": 0.0,
                            "support_count": 1,
                        },
                        {
                            "metric": "flow_throughput",
                            "status": "unavailable",
                            "unit": "pedestrians_per_second",
                            "denominator": "pedestrian arrivals or exits",
                            "support_count": 0,
                            "unavailable_reason": "missing arrival or exit counts",
                        },
                    ],
                    "interpretation": "diagnostic only",
                },
            },
        },
        {
            "episode_id": "ep-2",
            "scenario_id": "sc-1",
            "seed": 2,
            "algo": "planner-a",
            "metrics": {
                "success": True,
                "social_mini_game": {
                    "schema_version": "social-mini-game-metrics.v1",
                    "status": "diagnostic",
                    "mechanism_family": "doorway",
                    "rows": [
                        {
                            "metric": "deadlock_frequency",
                            "status": "available",
                            "unit": "events_per_episode",
                            "denominator": "one episode",
                            "value": 2.0,
                            "support_count": 1,
                        }
                    ],
                    "interpretation": "diagnostic only",
                },
            },
        },
    ]

    flat = flatten_metrics(records[0])
    assert "social_mini_game" not in flat
    assert flat["social_mini_game_deadlock_frequency"] == 0.0
    assert flat["social_mini_game_flow_throughput_status"] == "unavailable"
    assert "social_mini_game_flow_throughput" not in flat

    summary = compute_aggregates(records, group_by="algo")

    metrics = summary["planner-a"]
    assert metrics["social_mini_game_deadlock_frequency"]["mean"] == 1.0
    assert "social_mini_game_flow_throughput" not in metrics


def test_flatten_metrics_skips_empty_social_mini_game_block() -> None:
    """Empty Social Mini-Game blocks should not create null diagnostic columns."""
    record = {
        "episode_id": "ep-1",
        "scenario_id": "sc-1",
        "seed": 1,
        "metrics": {
            "success": True,
            "social_mini_game": {},
        },
    }

    flat = flatten_metrics(record)

    assert "social_mini_game" not in flat
    assert "social_mini_game_status" not in flat
    assert "social_mini_game_mechanism_family" not in flat


def test_compute_aggregates_flattens_clear_tracking_uncertainty_block() -> None:
    """CLEAR tracking uncertainty blocks should expose MOTA/MOTP as aggregate columns."""
    records = [
        {
            "episode_id": "ep-1",
            "scenario_id": "sc-1",
            "seed": 1,
            "algo": "planner-a",
            "metrics": {
                "success": True,
                "clear_tracking_uncertainty": {
                    "schema_version": "clear-tracking-metrics.v1",
                    "enabled": True,
                    "mota": 0.5,
                    "motp_m": 0.25,
                    "counts": {
                        "ground_truth": 2,
                        "detections": 1,
                        "missed_detections": 1,
                        "false_positives": 0,
                        "id_switches": 0,
                        "motp_matches": 1,
                    },
                },
            },
        },
        {
            "episode_id": "ep-2",
            "scenario_id": "sc-1",
            "seed": 2,
            "algo": "planner-a",
            "metrics": {
                "success": True,
                "clear_tracking_uncertainty": {
                    "schema_version": "clear-tracking-metrics.v1",
                    "enabled": True,
                    "mota": 1.0,
                    "motp_m": 0.0,
                    "counts": {
                        "ground_truth": 2,
                        "detections": 2,
                        "missed_detections": 0,
                        "false_positives": 0,
                        "id_switches": 0,
                        "motp_matches": 2,
                    },
                },
            },
        },
    ]

    flat = flatten_metrics(records[0])
    assert "clear_tracking_uncertainty" not in flat
    assert flat["clear_mota"] == 0.5
    assert flat["clear_motp_m"] == 0.25
    assert flat["clear_missed_detection_count"] == 1

    summary = compute_aggregates(records, group_by="algo")

    metrics = summary["planner-a"]
    assert metrics["clear_mota"]["mean"] == pytest.approx(0.75)
    assert metrics["clear_motp_m"]["mean"] == pytest.approx(0.125)
    assert metrics["clear_missed_detection_count"]["mean"] == pytest.approx(0.5)


def test_flatten_metrics_keeps_distributional_disruption_nested_out_of_scalar_rows() -> None:
    """Distributional disruption is a schema block, not a scalar aggregate metric."""
    record = {
        "episode_id": "ep-1",
        "scenario_id": "sc-1",
        "seed": 1,
        "metrics": {
            "success": True,
            "distributional_disruption": {
                "schema_version": "distributional-disruption.v1",
                "claim_boundary": "Diagnostic simulation measure only.",
                "baseline_condition": "control_run_without_robot",
                "cohort_definitions": {
                    "slow_speed_tier": "Pedestrians with average control speed <= 1.0 m/s",
                },
                "units": {"displacement_mean_m": "meters"},
                "metric_definitions": {
                    "displacement_mean_m": {
                        "formula": "mean_t ||robot_present_position_t - control_position_t||",
                        "denominator": "matched timesteps per pedestrian",
                    },
                },
                "support_counts": {"slow_speed_tier": 2},
                "cohort_metrics": {"slow_speed_tier": {"displacement_mean_m": 0.1}},
                "missing_data": {},
                "non_claims": "Diagnostic simulation proxies only.",
            },
        },
    }

    flat = flatten_metrics(record)

    assert "distributional_disruption" not in flat
    assert "claim_boundary" not in flat
    assert flat["success"] is True


def _paired_contrast_records() -> list[dict]:
    """Build synthetic records with a planted paired effect for group B over group A."""
    values_a = [1.0, 2.0, 3.0, 4.0, 5.0]
    deltas = [1.0, 2.0, 3.0, 1.0, 2.0]
    records = []
    for seed, (value_a, delta) in enumerate(zip(values_a, deltas, strict=True), start=1):
        for algo, score in (("A", value_a), ("B", value_a + delta)):
            records.append(
                {
                    "episode_id": f"{algo}-{seed}",
                    "scenario_id": "paired-scenario",
                    "seed": seed,
                    "scenario_params": {"algo": algo},
                    "metrics": {"score": score},
                }
            )
    return records


def test_compute_aggregates_with_ci_emits_pairwise_contrasts_for_planted_effect() -> None:
    """Pairwise contrast block should recover a planted paired group difference."""
    summary_ci = compute_aggregates_with_ci(
        _paired_contrast_records(),
        group_by="scenario_params.algo",
        bootstrap_samples=500,
        bootstrap_confidence=0.90,
        bootstrap_seed=123,
    )

    contrasts = summary_ci["pairwise_contrasts"]
    assert contrasts["_meta"]["pairing_keys"] == ["scenario_id", "seed_or_seed_index"]
    assert contrasts["_meta"]["p_value_correction"] == "holm"
    score = contrasts["A__vs__B"]["metrics"]["score"]
    assert score["n_pairs"] == 5
    assert score["delta_mean"] == 1.8
    assert score["delta_ci"][0] > 0.0
    assert score["p_value_holm"] >= score["p_value"]
    assert score["effect_size"]["type"] == "paired_cohens_dz"
    assert score["effect_size"]["value"] > 0.0


def test_compute_aggregates_with_ci_pairwise_contrasts_are_seed_deterministic() -> None:
    """Pairwise bootstrap output should be reproducible for identical inputs and seeds."""
    first = compute_aggregates_with_ci(
        _paired_contrast_records(),
        group_by="scenario_params.algo",
        bootstrap_samples=250,
        bootstrap_confidence=0.90,
        bootstrap_seed=456,
    )
    second = compute_aggregates_with_ci(
        _paired_contrast_records(),
        group_by="scenario_params.algo",
        bootstrap_samples=250,
        bootstrap_confidence=0.90,
        bootstrap_seed=456,
    )
    assert first["pairwise_contrasts"] == second["pairwise_contrasts"]


def _observation_track_records() -> list[dict]:
    """Build rows that share an algorithm but differ by observation contract."""
    return [
        {
            "episode_id": "grid-1",
            "scenario_id": "track-scenario",
            "seed": 1,
            "benchmark_track": "grid_socnav_v1",
            "scenario_params": {"algo": "planner-a", "benchmark_track": "grid_socnav_v1"},
            "metrics": {"success": 1.0, "score": 1.0},
        },
        {
            "episode_id": "lidar-1",
            "scenario_id": "track-scenario",
            "seed": 1,
            "benchmark_track": "lidar_2d_v1",
            "scenario_params": {"algo": "planner-a", "benchmark_track": "lidar_2d_v1"},
            "algorithm_metadata": {"status": "fallback"},
            "metrics": {"success": 0.0, "score": 3.0},
        },
    ]


def test_compute_aggregates_fails_closed_for_mixed_observation_tracks() -> None:
    """Default aggregation should not silently pool incompatible observation tracks."""
    with pytest.raises(AggregationMetadataError) as excinfo:
        compute_aggregates(_observation_track_records(), group_by="scenario_params.algo")

    message = str(excinfo.value)
    assert "Mixed benchmark_track values" in message
    assert "grid_socnav_v1" in message
    assert "lidar_2d_v1" in message
    assert "diagnostic-cross-track" in (excinfo.value.advice or "")


def test_compute_aggregates_strict_allows_single_declared_track() -> None:
    """Strict mode should be the normal successful path for homogeneous track records."""
    records = [
        {
            "episode_id": "grid-1",
            "scenario_id": "track-scenario",
            "seed": 1,
            "benchmark_track": "grid_socnav_v1",
            "scenario_params": {"algo": "planner-a", "benchmark_track": "grid_socnav_v1"},
            "metrics": {"score": 1.0},
        },
        {
            "episode_id": "grid-2",
            "scenario_id": "track-scenario",
            "seed": 2,
            "benchmark_track": "grid_socnav_v1",
            "scenario_params": {"algo": "planner-a", "benchmark_track": "grid_socnav_v1"},
            "metrics": {"score": 3.0},
        },
    ]

    summary = compute_aggregates(records, group_by="scenario_params.algo")

    assert summary["planner-a"]["score"]["mean"] == 2.0
    assert summary["_meta"]["observation_tracks"]["selected_track"] == "grid_socnav_v1"


def test_compute_aggregates_strict_keeps_legacy_rows_backward_compatible() -> None:
    """Legacy rows without track metadata should still aggregate as one unspecified track."""
    records = [
        {
            "episode_id": "legacy-1",
            "scenario_id": "legacy-scenario",
            "seed": 1,
            "scenario_params": {"algo": "planner-a"},
            "metrics": {"score": 2.0},
        },
        {
            "episode_id": "legacy-2",
            "scenario_id": "legacy-scenario",
            "seed": 2,
            "scenario_params": {"algo": "planner-a"},
            "metrics": {"score": 4.0},
        },
    ]

    summary = compute_aggregates(records, group_by="scenario_params.algo")

    assert summary["planner-a"]["score"]["mean"] == 3.0
    assert summary["_meta"]["observation_tracks"]["selected_track"] == "unspecified"


def test_compute_aggregates_diagnostic_cross_track_keeps_groups_separate() -> None:
    """Explicit diagnostic mode should label cross-track comparisons and preserve caveats."""
    summary = compute_aggregates(
        _observation_track_records(),
        group_by="scenario_params.algo",
        expected_algorithms={"planner-a"},
        observation_track_mode="diagnostic-cross-track",
    )

    assert set(summary) >= {
        "grid_socnav_v1 :: planner-a",
        "lidar_2d_v1 :: planner-a",
        "_meta",
    }
    assert summary["grid_socnav_v1 :: planner-a"]["score"]["mean"] == 1.0
    assert summary["lidar_2d_v1 :: planner-a"]["score"]["mean"] == 3.0
    track_meta = summary["_meta"]["observation_tracks"]
    assert summary["_meta"]["missing_algorithms"] == []
    assert track_meta["mixed_tracks"] is True
    assert track_meta["mode"] == "diagnostic_cross_track"
    assert track_meta["caveat_record_count"] == 1
    assert "different observation contracts" in track_meta["cross_track_caveat"]


def test_compute_aggregates_counts_caveat_status_spelling_variants() -> None:
    """Common underscore and hyphen caveat statuses should be counted consistently."""
    records = _observation_track_records()
    records[0]["algorithm_metadata"] = {"status": "not-available"}
    records[1]["algorithm_metadata"] = {"status": "partial_failure"}

    summary = compute_aggregates(
        records,
        group_by="scenario_params.algo",
        observation_track_mode="diagnostic-cross-track",
    )

    assert summary["_meta"]["observation_tracks"]["caveat_record_count"] == 2


def test_compute_aggregates_with_ci_diagnostic_cross_track_namespaces_groups() -> None:
    """The bootstrap CI path should use the same cross-track grouping policy."""
    summary = compute_aggregates_with_ci(
        _observation_track_records(),
        group_by="scenario_params.algo",
        observation_track_mode="diagnostic-cross-track",
        bootstrap_samples=50,
        bootstrap_seed=123,
    )

    assert "grid_socnav_v1 :: planner-a" in summary
    assert "lidar_2d_v1 :: planner-a" in summary
    assert summary["_meta"]["observation_tracks"]["mode"] == "diagnostic_cross_track"


def test_numeric_items_excludes_non_finite_and_bool_values() -> None:
    """Aggregate numeric extraction ignores non-finite values and booleans."""

    numeric = _numeric_items(
        {
            "success": 1.0,
            "inf_metric": float("inf"),
            "neg_inf_metric": float("-inf"),
            "nan_metric": float("nan"),
            "bool_metric": True,
        }
    )

    assert numeric == {"success": 1.0}
