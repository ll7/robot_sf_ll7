"""Tests for the issue #7318 CALF/LegNav comparator diagnostic."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import yaml
from jsonschema import Draft202012Validator

from robot_sf.benchmark.calf_legnav_comparator import (
    build_calf_legnav_comparator_report,
    canonical_config_digest,
)

REPO_ROOT = Path(__file__).parents[2]
SCHEMA_PATH = REPO_ROOT / "robot_sf/benchmark/schemas/calf_legnav_comparator.v1.json"


def _trace(
    evidence_class: str,
    distances: list[float],
    *,
    fallback: bool = False,
) -> dict[str, Any]:
    """Build a compact paired trace fixture with the fields used by the report."""
    rows = []
    for step, distance in enumerate(distances):
        rows.append(
            {
                "step": step,
                "env_action": [1.0 + step * 0.1, 0.1 + step * 0.05],
                "is_success": False,
                "is_pedestrian_collision": False,
                "is_obstacle_collision": False,
                "is_robot_collision": False,
                "truncated": False,
                "min_robot_ped_distance": distance,
                "post_step_min_robot_ped_distance": distance,
                "observed_observation": {
                    "evidence_class": evidence_class,
                    "noise_profile": "none"
                    if evidence_class == "ideal_state"
                    else "fixture_visibility",
                },
                "observation_perturbation": {"observed_actor_count": 1},
            }
        )
    return {
        "candidate": "ppo_fixture",
        "scenario_id": "fixture_scenario",
        "seed": 111,
        "horizon": len(rows),
        "algo": "PPO",
        "planner_execution_mode": "command_adapter",
        "fallback_degraded_status": {"reported_fallback_or_degraded": fallback},
        "observation_perturbation_config": {"seed": 7318},
        "done_info": {"success": True, "truncated": False},
        "steps": rows,
    }


def _config() -> dict[str, Any]:
    """Return the minimal config needed for the report contract."""
    return {
        "schema_version": "calf_legnav_comparator_config.v1",
        "issue": 7318,
        "candidate": "ppo_fixture",
        "scenario_name": "fixture_scenario",
        "seed": 111,
        "horizon": 3,
        "dt_s": 0.1,
        "personal_space_radius_m": 1.5,
    }


def test_paired_report_preserves_observation_and_proxy_boundaries() -> None:
    """A valid paired fixture produces local metrics without transfer claims."""
    config = _config()
    report = build_calf_legnav_comparator_report(
        _trace("ideal_state", [2.0, 2.0, 2.0]),
        _trace("perception_limited", [1.0, 1.0, 2.0]),
        config=config,
        input_refs={"fixture": "tests/benchmark/test_calf_legnav_comparator.py"},
    )

    assert report["status"] == "available"
    assert report["evidence_status"] == "diagnostic-only"
    assert report["conditions"]["perfect_perception"]["observation_contract"]["condition"] == (
        "perfect_perception"
    )
    assert report["conditions"]["sensor_limited"]["observation_contract"]["condition"] == (
        "sensor_limited"
    )
    assert report["paired_metrics"]["minimum_human_distance_m"]["sensor_minus_perfect"] == -1.0
    assert report["paired_metrics"]["personal_space_compliance_rate"][
        "sensor_minus_perfect"
    ] == pytest.approx(-2 / 3)
    assert report["paired_metrics"]["angular_jerk_rad_s3"]["delta_status"] == "available"
    assert report["zero_shot_transfer"]["status"] == "unavailable"
    assert report["provenance"]["uncertainty"]["status"] == "unavailable"
    assert report["provenance"]["config_digest"] == canonical_config_digest(config)


def test_durable_smoke_config_matches_config_schema() -> None:
    """The checked-in fixture config is itself schema-valid and reproducible."""
    config_path = REPO_ROOT / "configs/benchmarks/issue_7318_calf_legnav_comparator_smoke.yaml"
    schema_path = REPO_ROOT / "robot_sf/benchmark/schemas/calf_legnav_comparator_config.v1.json"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    schema = json.loads(schema_path.read_text(encoding="utf-8"))

    Draft202012Validator.check_schema(schema)
    assert list(Draft202012Validator(schema).iter_errors(config)) == []


def test_report_matches_schema_and_records_runner_errors() -> None:
    """The schema accepts a blocked report with compact runner provenance."""
    report = build_calf_legnav_comparator_report(
        _trace("ideal_state", [2.0, 2.0, 2.0]),
        _trace("perception_limited", [1.0, 1.0, 2.0], fallback=True),
        config=_config(),
    )
    report["runner_errors"] = [
        {
            "condition": "sensor_limited",
            "status": "blocked",
            "reason": "runner failed",
            "stderr_excerpt": "error",
            "command": ["python", "runner.py"],
        }
    ]

    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    Draft202012Validator.check_schema(schema)
    errors = list(Draft202012Validator(schema).iter_errors(report))

    assert not errors
    assert report["status"] == "blocked"
    assert report["conditions"]["sensor_limited"]["status"] == "blocked"
    assert report["conditions"]["sensor_limited"]["metrics"]["success_rate"]["status"] == "blocked"


def test_horizon_exhaustion_is_recorded_as_timeout() -> None:
    """A fixed-horizon trace without success is not silently reported as zero timeout."""
    perfect = _trace("ideal_state", [2.0, 2.0, 2.0])
    sensor = _trace("perception_limited", [1.0, 1.0, 2.0])
    perfect["done_info"] = {}
    sensor["done_info"] = {}

    report = build_calf_legnav_comparator_report(perfect, sensor, config=_config())

    assert report["paired_metrics"]["success_rate"]["perfect_perception"]["value"] == 0.0
    assert report["paired_metrics"]["timeout_rate"]["perfect_perception"]["value"] == 1.0
    assert report["paired_metrics"]["timeout_rate"]["sensor_limited"]["value"] == 1.0


def test_mismatched_pair_is_rejected() -> None:
    """A paired comparison cannot mix scenario or seed provenance."""
    sensor = _trace("perception_limited", [1.0, 1.0, 2.0])
    sensor["seed"] = 112

    with pytest.raises(ValueError, match="seed"):
        build_calf_legnav_comparator_report(
            _trace("ideal_state", [2.0, 2.0, 2.0]),
            sensor,
            config=_config(),
        )


def test_mismatched_horizon_is_rejected() -> None:
    """A paired comparison cannot mix replay horizons."""
    sensor = _trace("perception_limited", [1.0, 1.0, 2.0])
    sensor["horizon"] = 99

    with pytest.raises(ValueError, match="horizon"):
        build_calf_legnav_comparator_report(
            _trace("ideal_state", [2.0, 2.0, 2.0]),
            sensor,
            config=_config(),
        )


def test_swapped_observation_contract_fails_closed() -> None:
    """A trace whose evidence class contradicts its paired slot blocks the report."""
    report = build_calf_legnav_comparator_report(
        _trace("perception_limited", [2.0, 2.0, 2.0]),
        _trace("perception_limited", [1.0, 1.0, 2.0]),
        config=_config(),
    )

    ideal = report["conditions"]["perfect_perception"]
    assert ideal["observation_contract"]["condition_binding"] == "unavailable"
    assert ideal["observation_contract"]["status"] == "unavailable"
    assert ideal["status"] == "blocked"
    assert report["status"] == "blocked"
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    Draft202012Validator(schema).validate(report)


def test_matching_observation_contract_records_the_binding() -> None:
    """A correctly paired report records the matched condition binding."""
    report = build_calf_legnav_comparator_report(
        _trace("ideal_state", [2.0, 2.0, 2.0]),
        _trace("perception_limited", [1.0, 1.0, 2.0]),
        config=_config(),
    )

    for condition in ("perfect_perception", "sensor_limited"):
        binding = report["conditions"][condition]["observation_contract"]
        assert binding["condition_binding"] == "matched"
        assert binding["expected_condition"] == condition
    assert report["status"] == "available"
