"""Contract tests for the episode JSON schema (T010)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from robot_sf.benchmark.schema_validator import validate_episode

try:
    import jsonschema  # type: ignore
except Exception as e:  # pragma: no cover
    raise RuntimeError("jsonschema dependency required for contract tests") from e


SCHEMA_PATH = (
    Path(__file__).resolve().parents[2]
    / "robot_sf"
    / "benchmark"
    / "schemas"
    / "episode.schema.v1.json"
)


def _load_schema() -> dict:
    """Load the canonical episode schema from the repository."""
    text = SCHEMA_PATH.read_text()
    return json.loads(text)


def test_episode_schema_invalid_sample_fails():
    """A partial episode record should fail the v1 schema."""
    schema = _load_schema()
    invalid_record = {  # missing many required keys by design
        "episode_id": "abc123",
        "metrics": {"collisions": 0},
    }
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(instance=invalid_record, schema=schema)


def test_episode_schema_minimal_valid_passes_when_ready():
    """A minimal v1 episode record should validate against the canonical schema."""
    schema = _load_schema()
    minimal = {
        "episode_id": "e_000",
        "version": "v1",
        "scenario_id": "sc_basic",
        "seed": 123,
        "metrics": {"collisions": 0, "near_misses": 0},
        "termination_reason": "max_steps",
        "outcome": {
            "route_complete": False,
            "collision_event": False,
            "timeout_event": True,
        },
        "integrity": {"contradictions": []},
    }
    jsonschema.validate(instance=minimal, schema=schema)


def test_episode_schema_validates_pedestrian_impact_block() -> None:
    """The pedestrian-impact metric block should be schema-backed when present."""
    schema = _load_schema()
    record = {
        "episode_id": "e_ped_impact",
        "version": "v1",
        "scenario_id": "sc_ped_impact",
        "seed": 123,
        "metrics": {
            "collisions": 0,
            "near_misses": 0,
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
                    "accel_delta_mean": 0.75,
                    "accel_delta_median": 0.70,
                    "accel_delta_valid_pedestrians": 1,
                    "turn_rate_delta_mean": 0.20,
                    "turn_rate_delta_median": 0.18,
                    "turn_rate_delta_valid_pedestrians": 1,
                },
            },
        },
        "termination_reason": "max_steps",
        "outcome": {
            "route_complete": False,
            "collision_event": False,
            "timeout_event": True,
        },
        "integrity": {"contradictions": []},
    }

    jsonschema.validate(instance=record, schema=schema)
    record["metrics"]["pedestrian_impact"]["schema_version"] = "wrong"
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(instance=record, schema=schema)

    record["metrics"]["pedestrian_impact"]["schema_version"] = "pedestrian-impact.v1"
    record["metrics"]["pedestrian_impact"]["sample_counts"]["pedestrians"] = 1.5
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(instance=record, schema=schema)


def test_episode_schema_validates_social_acceptability_block() -> None:
    """The social-acceptability pilot block should be schema-backed when present."""
    schema = _load_schema()
    record = {
        "episode_id": "e_social_acceptability",
        "version": "v1",
        "scenario_id": "sc_social_acceptability",
        "seed": 123,
        "metrics": {
            "collisions": 0,
            "near_misses": 0,
            "social_acceptability": {
                "schema_version": "social-acceptability-pilot.v1",
                "status": "exploratory",
                "parameters": {"proxemic_radius_m": 1.2},
                "units": {
                    "clearance": "m",
                    "intrusion_area": "m*s",
                    "intrusion_fraction": "fraction",
                    "sample_counts": "count",
                },
                "available": True,
                "sample_counts": {"pedestrians": 1, "timesteps": 2},
                "proxemic": {
                    "intrusion_steps": 2,
                    "intrusion_frac": 0.5,
                    "intrusion_area_m_s": 0.9,
                    "min_clearance_m": 0.1,
                },
                "interpretation": "Exploratory trajectory-only proxemic proxy.",
            },
        },
        "termination_reason": "max_steps",
        "outcome": {
            "route_complete": False,
            "collision_event": False,
            "timeout_event": True,
        },
        "integrity": {"contradictions": []},
    }

    jsonschema.validate(instance=record, schema=schema)
    record["metrics"]["social_acceptability"]["schema_version"] = "wrong"
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(instance=record, schema=schema)

    record["metrics"]["social_acceptability"]["schema_version"] = "social-acceptability-pilot.v1"
    record["metrics"]["social_acceptability"]["sample_counts"]["pedestrians"] = 1.5
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(instance=record, schema=schema)


def test_episode_schema_validates_human_interaction_proxy_block() -> None:
    """The human-interaction proxy block should be schema-backed when present."""
    schema = _load_schema()
    record = {
        "episode_id": "e_human_interaction_proxy",
        "version": "v1",
        "scenario_id": "sc_human_interaction_proxy",
        "seed": 123,
        "metrics": {
            "collisions": 0,
            "near_misses": 0,
            "human_interaction_proxy": {
                "schema_version": "human-interaction-proxy.v1",
                "status": "simulation_proxy",
                "parameters": {"proxemic_radius_m": 1.2, "yield_speed_mps": 0.1},
                "units": {
                    "discomfort_exposure": "m*s",
                    "duration": "s",
                    "time_to_yield": "s",
                    "distance": "m",
                    "path_deviation": "m",
                    "sample_counts": "count",
                },
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
                "exclusions": {
                    "group_split_intrusion": "No group-membership labels in EpisodeData.",
                },
                "interpretation": "Diagnostic simulation-proxy metrics only.",
            },
        },
        "termination_reason": "max_steps",
        "outcome": {
            "route_complete": False,
            "collision_event": False,
            "timeout_event": True,
        },
        "integrity": {"contradictions": []},
    }

    jsonschema.validate(instance=record, schema=schema)
    record["metrics"]["human_interaction_proxy"]["schema_version"] = "wrong"
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(instance=record, schema=schema)

    record["metrics"]["human_interaction_proxy"]["schema_version"] = "human-interaction-proxy.v1"
    record["metrics"]["human_interaction_proxy"]["sample_counts"]["pedestrians"] = 1.5
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(instance=record, schema=schema)


def test_episode_schema_validates_social_mini_game_block() -> None:
    """The Social Mini-Game metric block should validate row availability contracts."""
    schema = _load_schema()
    record = {
        "episode_id": "e_social_mini_game",
        "version": "v1",
        "scenario_id": "sc_social_mini_game",
        "seed": 123,
        "metrics": {
            "collisions": 0,
            "near_misses": 0,
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
                "interpretation": "Diagnostic Social Mini-Game mechanism metrics.",
            },
        },
        "termination_reason": "max_steps",
        "outcome": {
            "route_complete": False,
            "collision_event": False,
            "timeout_event": True,
        },
        "integrity": {"contradictions": []},
    }

    jsonschema.validate(instance=record, schema=schema)
    record["metrics"]["social_mini_game"]["schema_version"] = "wrong"
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(instance=record, schema=schema)

    record["metrics"]["social_mini_game"]["schema_version"] = "social-mini-game-metrics.v1"
    record["metrics"]["social_mini_game"]["rows"][0]["support_count"] = -1
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(instance=record, schema=schema)


def test_episode_schema_rejects_collision_event_without_collision_metric() -> None:
    """New v1 records should not report collision_event=true with zero collision count."""
    schema = _load_schema()
    record = {
        "episode_id": "e_collision",
        "version": "v1",
        "scenario_id": "sc_collision",
        "seed": 123,
        "metrics": {"success": 0.0, "collisions": 0.0},
        "termination_reason": "collision",
        "outcome": {
            "route_complete": False,
            "collision_event": True,
            "timeout_event": False,
        },
        "integrity": {"contradictions": []},
    }
    with pytest.raises(jsonschema.ValidationError, match="collisions"):
        jsonschema.validate(instance=record, schema=schema)


def test_episode_schema_rejects_collision_metric_without_collision_event() -> None:
    """New v1 records should not carry a positive collision count for non-collision episodes."""
    schema = _load_schema()
    record = {
        "episode_id": "e_stale_collision",
        "version": "v1",
        "scenario_id": "sc_collision",
        "seed": 124,
        "metrics": {"success": 0.0, "collisions": 1.0},
        "termination_reason": "max_steps",
        "outcome": {
            "route_complete": False,
            "collision_event": False,
            "timeout_event": True,
        },
        "integrity": {"contradictions": []},
    }
    with pytest.raises(jsonschema.ValidationError, match="collisions"):
        jsonschema.validate(instance=record, schema=schema)


def test_episode_validator_rejects_legacy_collision_alias_drift() -> None:
    """Semantic validation should also catch legacy collision aliases in schema-valid records."""
    schema = _load_schema()
    record = {
        "episode_id": "e_alias_collision",
        "version": "v1",
        "scenario_id": "sc_collision",
        "seed": 125,
        "metrics": {"success": 0.0, "collisions": 0.0, "collision_rate": 1.0},
        "termination_reason": "max_steps",
        "outcome": {
            "route_complete": False,
            "collision_event": False,
            "timeout_event": True,
        },
        "integrity": {"contradictions": []},
    }
    with pytest.raises(jsonschema.ValidationError, match="collision_event=false"):
        validate_episode(record, schema)


def test_episode_schema_validates_distributional_disruption_block() -> None:
    """The distributional disruption metric block should validate successfully."""
    schema = _load_schema()
    record = {
        "episode_id": "e_dist_disruption",
        "version": "v1",
        "scenario_id": "sc_dist_disruption",
        "seed": 123,
        "metrics": {
            "collisions": 0,
            "near_misses": 0,
            "distributional_disruption": {
                "schema_version": "distributional-disruption.v1",
                "claim_boundary": (
                    "These metrics are diagnostic simulation measures for analyzing per-subgroup "
                    "displacement and inconvenience distribution in controlled settings. "
                    "They do not represent real-world ethical outcomes."
                ),
                "baseline_condition": "control_run_without_robot",
                "cohort_definitions": {
                    "slow_speed_tier": "Pedestrians with average control speed <= 1.0 m/s",
                    "fast_speed_tier": "Pedestrians with average control speed > 1.0 m/s and <= 1.8 m/s",
                    "extreme_speed_tier": "Pedestrians with average control speed > 1.8 m/s",
                },
                "units": {"displacement_mean_m": "meters", "delay_mean_s": "seconds"},
                "metric_definitions": {
                    "displacement_mean_m": {
                        "formula": "mean_t ||robot_present_position_t - control_position_t||",
                        "denominator": (
                            "matched timesteps per pedestrian, then supported pedestrians per cohort"
                        ),
                    },
                    "delay_mean_s": {
                        "formula": (
                            "max(0, robot_present_path_length - control_path_length) / "
                            "max(control_mean_speed, 0.1)"
                        ),
                        "denominator": "supported pedestrians per cohort",
                    },
                },
                "support_counts": {
                    "slow_speed_tier": 2,
                    "fast_speed_tier": 3,
                    "extreme_speed_tier": 0,
                },
                "cohort_metrics": {
                    "slow_speed_tier": {"displacement_mean_m": 0.15, "delay_mean_s": 0.5},
                    "fast_speed_tier": {"displacement_mean_m": 0.08, "delay_mean_s": 0.2},
                },
                "missing_data": {
                    "extreme_speed_tier": {"status": "missing", "reason": "No samples available"}
                },
                "non_claims": (
                    "We make no claims regarding real-world fairness, equity, bias, "
                    "protected attributes, demographic groups, or disparate impact. "
                    "These measures are diagnostic simulation proxies only."
                ),
            },
        },
        "termination_reason": "max_steps",
        "outcome": {
            "route_complete": False,
            "collision_event": False,
            "timeout_event": True,
        },
        "integrity": {"contradictions": []},
    }
    jsonschema.validate(instance=record, schema=schema)
