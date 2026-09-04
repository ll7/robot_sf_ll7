"""Tests for the issue #7318 CALF/LegNav comparator diagnostic."""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any

import pytest
import yaml
from jsonschema import Draft202012Validator

import scripts.benchmark.run_calf_legnav_comparator_issue_7318 as comparator_runner
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


def test_distance_metrics_use_one_conservative_sample_per_action() -> None:
    """Pre/post distances in one row do not double-weight a control interval."""
    perfect = _trace("ideal_state", [2.0, 2.0, 2.0])
    sensor = _trace("perception_limited", [2.0, 2.0, 2.0])
    for trace in (perfect, sensor):
        for row in trace["steps"]:
            row["min_robot_ped_distance"] = 2.0
            row["post_step_min_robot_ped_distance"] = 1.0

    report = build_calf_legnav_comparator_report(perfect, sensor, config=_config())
    metrics = report["conditions"]["perfect_perception"]["metrics"]

    assert metrics["minimum_human_distance_m"]["value"] == 1.0
    assert metrics["personal_space_compliance_rate"]["value"] == 0.0


def test_malformed_outcome_flags_are_unavailable_not_truthy() -> None:
    """Stringified booleans cannot silently become positive outcome metrics."""
    perfect = _trace("ideal_state", [2.0, 2.0, 2.0])
    sensor = _trace("perception_limited", [2.0, 2.0, 2.0])
    perfect["steps"][0]["is_success"] = "false"
    perfect["steps"][0]["is_pedestrian_collision"] = "false"
    perfect["done_info"]["truncated"] = "false"

    report = build_calf_legnav_comparator_report(perfect, sensor, config=_config())
    metrics = report["conditions"]["perfect_perception"]["metrics"]

    for name in ("success_rate", "collision_rate", "timeout_rate"):
        assert metrics[name]["status"] == "unavailable"
        assert metrics[name]["value"] is None
        assert metrics[name]["reason"] == "outcome flags must be booleans when present"


def test_missing_outcome_flags_are_unavailable_not_zeroes() -> None:
    """Missing outcome fields cannot silently become fabricated zero metrics."""
    perfect = _trace("ideal_state", [2.0, 2.0, 2.0])
    sensor = _trace("perception_limited", [2.0, 2.0, 2.0])
    for trace in (perfect, sensor):
        trace["done_info"].pop("success")
        for row in trace["steps"]:
            row.pop("is_success")
            row.pop("is_pedestrian_collision")
            row.pop("is_obstacle_collision")
            row.pop("is_robot_collision")

    report = build_calf_legnav_comparator_report(perfect, sensor, config=_config())
    metrics = report["conditions"]["perfect_perception"]["metrics"]

    for name in ("success_rate", "collision_rate", "timeout_rate"):
        assert metrics[name]["status"] == "unavailable"
        assert metrics[name]["value"] is None


def test_missing_observation_contract_row_blocks_the_condition() -> None:
    """A partial evidence-class trace cannot become an available paired condition."""
    perfect = _trace("ideal_state", [2.0, 2.0, 2.0])
    sensor = _trace("perception_limited", [2.0, 2.0, 2.0])
    del sensor["steps"][1]["observed_observation"]

    report = build_calf_legnav_comparator_report(perfect, sensor, config=_config())

    condition = report["conditions"]["sensor_limited"]
    assert condition["status"] == "blocked"
    assert condition["observation_contract"]["status"] == "unavailable"
    assert "typed observed observation" in condition["observation_contract"]["reason"]
    assert report["status"] == "blocked"


@pytest.mark.parametrize("bad_count", [1.5, "1", "1.5", -1, True, math.nan])
def test_malformed_observed_actor_count_blocks_the_condition(bad_count: Any) -> None:
    """Actor-count provenance cannot be coerced or crash the comparator."""
    perfect = _trace("ideal_state", [2.0, 2.0, 2.0])
    sensor = _trace("perception_limited", [2.0, 2.0, 2.0])
    perfect["steps"][1]["observation_perturbation"]["observed_actor_count"] = bad_count

    report = build_calf_legnav_comparator_report(perfect, sensor, config=_config())

    condition = report["conditions"]["perfect_perception"]
    assert condition["status"] == "blocked"
    assert condition["observation_contract"]["status"] == "unavailable"
    assert (
        "non-negative integer observed_actor_count" in condition["observation_contract"]["reason"]
    )


def test_incomplete_trace_blocks_the_condition() -> None:
    """A partial trace cannot become an available fixed-horizon episode."""
    perfect = _trace("ideal_state", [2.0])
    sensor = _trace("perception_limited", [2.0])
    for trace in (perfect, sensor):
        trace["horizon"] = 12
        trace["done_info"] = {}
    config = _config()
    config["horizon"] = 12

    report = build_calf_legnav_comparator_report(perfect, sensor, config=config)

    condition = report["conditions"]["perfect_perception"]
    assert condition["status"] == "blocked"
    assert condition["execution"]["reason"] == (
        "trace ended before its horizon without a terminal done_info verdict"
    )
    assert condition["metrics"]["timeout_rate"]["status"] == "blocked"
    assert report["status"] == "blocked"


def test_malformed_metric_fields_do_not_produce_partial_values() -> None:
    """A malformed distance or action row invalidates the affected metric."""
    perfect = _trace("ideal_state", [2.0, 2.0, 2.0])
    sensor = _trace("perception_limited", [2.0, 2.0, 2.0])
    perfect["steps"][1]["min_robot_ped_distance"] = "2.0"
    sensor["steps"][1]["env_action"] = ["0.1", 0.2]

    report = build_calf_legnav_comparator_report(perfect, sensor, config=_config())

    perfect_metrics = report["conditions"]["perfect_perception"]["metrics"]
    sensor_metrics = report["conditions"]["sensor_limited"]["metrics"]
    for name in ("minimum_human_distance_m", "personal_space_compliance_rate"):
        assert perfect_metrics[name]["status"] == "unavailable"
        assert perfect_metrics[name]["value"] is None
        assert perfect_metrics[name]["reason"] == (
            "distance fields must be finite non-negative numbers when present"
        )
    for name in ("angular_jerk_rad_s3", "action_smoothness_l2"):
        assert sensor_metrics[name]["status"] == "unavailable"
        assert sensor_metrics[name]["value"] is None
        assert sensor_metrics[name]["reason"] == (
            "action fields must contain at least two finite numeric channels"
        )


def test_incomplete_distance_trace_does_not_report_partial_metrics() -> None:
    """Distance metrics must not silently drop an action with no distance field."""
    perfect = _trace("ideal_state", [2.0, 2.0, 2.0])
    sensor = _trace("perception_limited", [1.0, 1.0, 2.0])
    for trace in (perfect, sensor):
        trace["steps"][1].pop("min_robot_ped_distance")
        trace["steps"][1].pop("post_step_min_robot_ped_distance")

    report = build_calf_legnav_comparator_report(perfect, sensor, config=_config())
    metrics = report["conditions"]["perfect_perception"]["metrics"]

    for name in ("minimum_human_distance_m", "personal_space_compliance_rate"):
        assert metrics[name]["status"] == "unavailable"
        assert metrics[name]["value"] is None
        assert metrics[name]["reason"] == (
            "each executed action must expose at least one ground-truth distance field"
        )


def test_contradictory_success_fields_are_unavailable() -> None:
    """A terminal false outcome must not be overridden by a row success flag."""
    perfect = _trace("ideal_state", [2.0, 2.0, 2.0])
    sensor = _trace("perception_limited", [1.0, 1.0, 2.0])
    perfect["done_info"]["success"] = False
    perfect["steps"][1]["is_success"] = True

    report = build_calf_legnav_comparator_report(perfect, sensor, config=_config())
    success = report["conditions"]["perfect_perception"]["metrics"]["success_rate"]

    assert success["status"] == "unavailable"
    assert success["value"] is None
    assert success["reason"] == ("trace.done_info.success contradicts trace.is_success")


def test_runner_materializes_blocked_report_for_malformed_trace(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A malformed runner trace becomes a schema-valid blocked handoff."""
    config_path = REPO_ROOT / "configs/benchmarks/issue_7318_calf_legnav_comparator_smoke.yaml"
    monkeypatch.setattr(
        comparator_runner,
        "_run_condition",
        lambda *args, **kwargs: ({"candidate": "malformed"}, None),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_calf_legnav_comparator_issue_7318.py",
            "--config",
            str(config_path),
            "--output-dir",
            str(tmp_path),
        ],
    )

    assert comparator_runner.main() == 2
    report = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    assert report["status"] == "blocked"
    assert report["runner_errors"][0]["condition"] == "paired"


def test_runner_materializes_blocked_report_for_schema_invalid_trace(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A trace with invalid output metadata becomes a blocked schema-valid handoff."""
    config_path = REPO_ROOT / "configs/benchmarks/issue_7318_calf_legnav_comparator_smoke.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    malformed = {
        "candidate": config["candidate"],
        "scenario_id": config["scenario_name"],
        "seed": config["seed"],
        "horizon": config["horizon"],
        "algo": None,
        "planner_execution_mode": "command_adapter",
        "fallback_degraded_status": {"reported_fallback_or_degraded": False},
        "done_info": {},
        "steps": [],
    }
    monkeypatch.setattr(
        comparator_runner,
        "_run_condition",
        lambda *args, **kwargs: (dict(malformed), None),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_calf_legnav_comparator_issue_7318.py",
            "--config",
            str(config_path),
            "--output-dir",
            str(tmp_path),
        ],
    )

    assert comparator_runner.main() == 2
    report = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    assert report["status"] == "blocked"
    assert "schema validation" in report["runner_errors"][0]["reason"]


def test_runtime_checkpoint_refs_require_a_paired_registry_match() -> None:
    """Runtime checkpoint provenance must be present and match the declared digest."""
    digest = "a" * 64
    traces = {
        condition: {
            "planner_summary": {
                "checkpoint_provenance": {
                    "checkpoint_sha256": digest,
                    "hash_source": "computed_resolved_file",
                    "load_succeeded": True,
                }
            }
        }
        for condition in ("perfect_perception", "sensor_limited")
    }

    refs = comparator_runner._runtime_checkpoint_refs(traces, expected_sha256=digest)

    assert refs["checkpoint_sha256_runtime"] == digest
    assert refs["checkpoint_sha256_matches_declared"] == "true"

    refs = comparator_runner._runtime_checkpoint_refs(traces, expected_sha256="b" * 64)
    assert refs["checkpoint_sha256_matches_declared"] == "false"


def test_input_refs_bind_enabled_predictive_checkpoint_to_registry() -> None:
    """Enabled predictive foresight contributes its own declared registry identity."""
    config_path = REPO_ROOT / "configs/benchmarks/issue_7318_calf_legnav_comparator_smoke.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))

    refs = comparator_runner._input_refs(config_path, config)

    assert refs["predictive_checkpoint_model_id"] == "predictive_proxy_selected_v2_full"
    assert refs["predictive_checkpoint_sha256_declared"] == (
        "a28aed6d6ad7e1ebf597277ade1cf908efa6da038d0a9fcfdf80c7c31d8d1be1"
    )


def _checkpoint_traces(
    requested: str | None,
    observed: str | None,
    *,
    requested_model_id: str | None = "predictive_proxy_selected_v2_full",
    load_status: str = "loaded",
    fallback_used: bool = False,
) -> dict[str, dict[str, Any]]:
    """Build paired trace summaries with outer and nested checkpoint provenance."""
    return {
        condition: {
            "planner_summary": {
                "checkpoint_provenance": {
                    "checkpoint_sha256": "a" * 64,
                    "hash_source": "computed_resolved_file",
                    "load_succeeded": True,
                },
                "foresight_prediction": {
                    "requested_model_id": requested_model_id,
                    "load_status": load_status,
                    "fallback_used": fallback_used,
                    "requested_checkpoint_sha256": requested,
                    "observed_checkpoint_sha256": observed,
                },
            }
        }
        for condition in ("perfect_perception", "sensor_limited")
    }


def test_runtime_predictive_checkpoint_refs_require_paired_registry_match() -> None:
    """Loaded nested foresight digests must match the declared registry digest in both arms."""
    nested_digest = "b" * 64
    traces = _checkpoint_traces(nested_digest.upper(), nested_digest)

    refs = comparator_runner._runtime_checkpoint_refs(
        traces,
        expected_sha256="a" * 64,
        expected_predictive_sha256=nested_digest,
        expected_predictive_model_id="predictive_proxy_selected_v2_full",
    )

    assert refs["predictive_checkpoint_sha256_requested_perfect_perception"] == nested_digest
    assert refs["predictive_checkpoint_sha256_observed_sensor_limited"] == nested_digest
    assert refs["predictive_checkpoint_sha256_runtime"] == nested_digest
    assert refs["predictive_checkpoint_sha256_matches_declared"] == "true"
    assert refs["predictive_checkpoint_model_id_runtime"] == "predictive_proxy_selected_v2_full"
    assert refs["predictive_checkpoint_model_id_matches_declared"] == "true"
    assert comparator_runner._runtime_provenance_error(refs) is None


def test_runtime_predictive_checkpoint_digest_mismatch_blocks() -> None:
    """A nested requested/observed digest mismatch blocks the comparator report."""
    requested = "b" * 64
    observed = "c" * 64
    refs = comparator_runner._runtime_checkpoint_refs(
        _checkpoint_traces(requested, observed),
        expected_sha256="a" * 64,
        expected_predictive_sha256=requested,
        expected_predictive_model_id="predictive_proxy_selected_v2_full",
    )

    error = comparator_runner._runtime_provenance_error(
        {
            **refs,
            "predictive_checkpoint_sha256_declared": requested,
            "predictive_checkpoint_model_id": "predictive_proxy_selected_v2_full",
        }
    )

    assert refs["predictive_checkpoint_sha256_runtime"] == observed
    assert refs["predictive_checkpoint_sha256_matches_declared"] == "false"
    assert error is not None
    assert error["status"] == "blocked"
    assert "predictive foresight checkpoint digest" in error["reason"]


def test_runtime_predictive_checkpoint_model_id_mismatch_blocks() -> None:
    """Matching checkpoint bytes cannot mask a mismatched requested model identity."""
    digest = "b" * 64
    expected_model_id = "predictive_proxy_selected_v2_full"
    refs = comparator_runner._runtime_checkpoint_refs(
        _checkpoint_traces(digest, digest, requested_model_id="different_model"),
        expected_sha256="a" * 64,
        expected_predictive_sha256=digest,
        expected_predictive_model_id=expected_model_id,
    )

    error = comparator_runner._runtime_provenance_error(
        {
            **refs,
            "predictive_checkpoint_sha256_declared": digest,
            "predictive_checkpoint_model_id": expected_model_id,
        }
    )

    assert refs["predictive_checkpoint_sha256_matches_declared"] == "true"
    assert refs["predictive_checkpoint_model_id_runtime"] == "different_model"
    assert refs["predictive_checkpoint_model_id_matches_declared"] == "false"
    assert error is not None
    assert error["status"] == "blocked"
    assert "model identity" in error["reason"]


@pytest.mark.parametrize("requested_model_id", [None, 42])
def test_runtime_predictive_checkpoint_malformed_model_id_blocks_without_schema_error(
    requested_model_id: Any,
) -> None:
    """Missing or non-string model IDs become an explicit blocked provenance result."""
    digest = "b" * 64
    expected_model_id = "predictive_proxy_selected_v2_full"
    refs = comparator_runner._runtime_checkpoint_refs(
        _checkpoint_traces(digest, digest, requested_model_id=requested_model_id),
        expected_sha256="a" * 64,
        expected_predictive_sha256=digest,
        expected_predictive_model_id=expected_model_id,
    )

    error = comparator_runner._runtime_provenance_error(
        {
            **refs,
            "predictive_checkpoint_sha256_declared": digest,
            "predictive_checkpoint_model_id": expected_model_id,
        }
    )

    assert all(isinstance(value, str) for value in refs.values())
    assert refs["predictive_checkpoint_model_id_runtime"] == "unavailable"
    assert refs["predictive_checkpoint_model_id_matches_declared"] == "false"
    assert error is not None
    assert error["status"] == "blocked"
    assert "model identity" in error["reason"]


@pytest.mark.parametrize(
    ("load_status", "fallback_used"),
    [("not_attempted", False), ("loaded", True)],
)
def test_runtime_predictive_checkpoint_unavailable_blocks(
    load_status: str, fallback_used: bool
) -> None:
    """Predictive foresight must be loaded without fallback before it can admit a report."""
    digest = "b" * 64
    refs = comparator_runner._runtime_checkpoint_refs(
        _checkpoint_traces(
            digest,
            digest,
            load_status=load_status,
            fallback_used=fallback_used,
        ),
        expected_sha256="a" * 64,
        expected_predictive_sha256=digest,
        expected_predictive_model_id="predictive_proxy_selected_v2_full",
    )

    error = comparator_runner._runtime_provenance_error(
        {
            **refs,
            "predictive_checkpoint_sha256_declared": digest,
            "predictive_checkpoint_model_id": "predictive_proxy_selected_v2_full",
        }
    )

    assert refs["predictive_checkpoint_sha256_runtime"] == "unavailable"
    assert error is not None
    assert error["status"] == "blocked"
    assert "predictive foresight checkpoint provenance" in error["reason"]


def test_non_finite_config_is_rejected_before_execution(tmp_path: Path) -> None:
    """YAML NaN values cannot enter config provenance or generated commands."""
    config_path = REPO_ROOT / "configs/benchmarks/issue_7318_calf_legnav_comparator_smoke.yaml"
    malformed_path = tmp_path / "config.yaml"
    malformed_path.write_text(
        config_path.read_text(encoding="utf-8").replace("dt_s: 0.1", "dt_s: .nan"),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="finite JSON-compatible"):
        comparator_runner._load_config(malformed_path)


def test_config_digest_rejects_non_finite_values() -> None:
    """The provenance digest cannot normalize non-standard JSON numbers."""
    with pytest.raises(ValueError, match="Out of range float values"):
        canonical_config_digest({"dt_s": math.nan})


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


def test_missing_fallback_verdict_blocks_metrics() -> None:
    """A trace without an explicit fallback verdict is not execution evidence."""
    perfect = _trace("ideal_state", [2.0, 2.0, 2.0])
    sensor = _trace("perception_limited", [1.0, 1.0, 2.0])
    perfect.pop("fallback_degraded_status")

    report = build_calf_legnav_comparator_report(perfect, sensor, config=_config())

    condition = report["conditions"]["perfect_perception"]
    assert condition["status"] == "blocked"
    assert condition["execution"]["reason"] == (
        "trace lacks an explicit fallback_or_degraded verdict"
    )
    assert condition["metrics"]["success_rate"]["status"] == "blocked"
    assert report["status"] == "blocked"


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


def test_terminal_horizon_episode_is_not_counted_as_timeout() -> None:
    """A terminated full-horizon episode is distinct from horizon truncation."""
    perfect = _trace("ideal_state", [2.0, 2.0, 2.0])
    sensor = _trace("perception_limited", [1.0, 1.0, 2.0])
    perfect["done_info"] = {"success": False, "terminated": True, "truncated": False}
    perfect["steps"][-1]["terminated"] = True

    report = build_calf_legnav_comparator_report(perfect, sensor, config=_config())

    timeout = report["conditions"]["perfect_perception"]["metrics"]["timeout_rate"]
    assert timeout["status"] == "available"
    assert timeout["value"] == 0.0


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


def test_mismatched_algorithm_is_rejected() -> None:
    """A paired report cannot mix different effective planner algorithms."""
    perfect = _trace("ideal_state", [2.0, 2.0, 2.0])
    sensor = _trace("perception_limited", [1.0, 1.0, 2.0])
    sensor["algo"] = "different_algo"

    with pytest.raises(ValueError, match="algo"):
        build_calf_legnav_comparator_report(perfect, sensor, config=_config())


def test_trace_identity_must_match_config() -> None:
    """A pair from another candidate or scenario cannot borrow this config provenance."""
    perfect = _trace("ideal_state", [2.0, 2.0, 2.0])
    sensor = _trace("perception_limited", [1.0, 1.0, 2.0])
    sensor["candidate"] = perfect["candidate"]
    perfect["scenario_id"] = "other_scenario"
    sensor["scenario_id"] = "other_scenario"

    with pytest.raises(ValueError, match="trace identity disagrees with config: scenario_name"):
        build_calf_legnav_comparator_report(perfect, sensor, config=_config())


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
    for row in report["paired_metrics"].values():
        assert row["delta_status"] == "unavailable"
        assert row["perfect_perception"]["status"] == "blocked"
        assert row["sensor_limited"]["status"] == "available"
        assert row["sensor_minus_perfect"] is None
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
