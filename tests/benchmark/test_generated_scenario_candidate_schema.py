"""Tests for the ``generated_scenario_candidate.v1`` schema."""

from __future__ import annotations

import json
from pathlib import Path

import jsonschema
import pytest

SCHEMA_PATH = (
    Path(__file__).resolve().parents[2]
    / "robot_sf"
    / "benchmark"
    / "schemas"
    / "generated_scenario_candidate.v1.json"
)

SCHEMA = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))


def _heuristic_candidate() -> dict:
    """Return a valid heuristic-perturbation candidate."""
    return {
        "schema_version": "generated_scenario_candidate.v1",
        "candidate_id": "heuristic-candidate-001",
        "generator_family": "heuristic_perturbation",
        "generator_run_id": "run-crossing-ttc-001",
        "generator_config_ref": {
            "config_path": "configs/adversarial/crossing_ttc_space.yaml",
            "config_hash": "sha256-abc123",
        },
        "source_scenario_ref": {
            "scenario_name": "crossing_ttc_low",
            "config_path": "configs/scenarios/crossing_ttc.yaml",
            "config_hash": "sha256-def456",
            "map_id": "classic_crossing",
        },
        "source_split": "train",
        "perturbations": [
            {
                "family": "robot_route_offset",
                "parameters": {
                    "dx_m": 1.2,
                    "dy_m": 0.5,
                    "max_magnitude_m": 2.0,
                },
                "rationale": "shift robot to increase crossing exposure",
            }
        ],
        "validity": {
            "preflight_passed": True,
            "scenario_certified": True,
            "map_exists": True,
            "spawn_collision_free": True,
            "goal_reachable": True,
        },
        "metrics_summary": {
            "severity": {
                "ttc_min_s": 1.5,
                "min_clearance_m": 0.8,
                "comfort_force_max_N": 45.0,
                "near_miss_count": 2,
                "collision_count": 0,
                "timeout_risk": 0.1,
                "objective_value": 7.5,
            },
            "diversity": {
                "param_distance_min_m": 1.0,
                "param_distance_mean_m": 2.3,
                "unique_scenario_families": 3,
                "coverage_fraction": 0.6,
                "dedup_rate": 0.05,
            },
        },
        "trace_lineage": {
            "seed": 42,
            "generated_at_utc": "2026-06-13T10:00:00+00:00",
            "git_hash": "abc1234",
            "provenance": {
                "source_issue": "#2725",
            },
        },
        "promotion_status": "not_promoted",
    }


def _naturalistic_prior(
    *,
    passed: bool = True,
    field: str = "pedestrian_speed_mps",
) -> dict:
    """Return a generated-candidate naturalistic-prior payload."""
    return {
        "schema_version": "naturalistic_vru_prior.v1",
        "profile": "urban_vru_default_v1",
        "constraints": [
            {
                "field": field,
                "min": 0.4,
                "max": 2.2,
                "observed": 1.2 if passed else 3.5,
                "passed": passed,
                "description": "bounded walking-to-running VRU speed for plausible hard cases",
            }
        ],
        "passed": passed,
        "violation_flags": [] if passed else ["pedestrian_speed_mps_outside_urban_vru_default_v1"],
    }


def _rl_adversary_candidate() -> dict:
    """Return a valid RL-adversary candidate."""
    base = _heuristic_candidate()
    base["candidate_id"] = "rl-adversary-candidate-001"
    base["generator_family"] = "rl_adversary"
    base["adversary_checkpoint_ref"] = {
        "checkpoint_path": "model/adversary/ckpt-0100.pt",
        "checkpoint_hash": "sha256-rl-ckpt",
    }
    return base


def _diffusion_candidate() -> dict:
    """Return a valid diffusion-prior candidate."""
    base = _heuristic_candidate()
    base["candidate_id"] = "diffusion-candidate-001"
    base["generator_family"] = "diffusion_prior"
    base["model_checkpoint_ref"] = {
        "checkpoint_path": "model/diffusion/ckpt-0200.pt",
        "checkpoint_hash": "sha256-diff-ckpt",
    }
    base["sample_params"] = {
        "temperature": 0.8,
        "guidance_scale": 1.5,
        "num_samples": 4,
    }
    return base


def _noop_candidate() -> dict:
    """Return a valid noop-perturbation candidate."""
    base = _heuristic_candidate()
    base["candidate_id"] = "noop-candidate-001"
    base["perturbations"] = [{"family": "noop", "parameters": {}}]
    return base


class TestSchemaLoads:
    """Schema file must be valid JSON and parseable."""

    def test_schema_file_is_valid_json(self) -> None:
        data = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
        assert data["$id"].endswith("generated_scenario_candidate.v1.json")

    def test_schema_has_additional_properties_false(self) -> None:
        assert SCHEMA.get("additionalProperties") is False


class TestHeuristicCandidate:
    """Representative heuristic candidate must validate."""

    def test_valid_heuristic_candidate(self) -> None:
        payload = _heuristic_candidate()
        jsonschema.validate(instance=payload, schema=SCHEMA)

    def test_noop_candidate_valid(self) -> None:
        payload = _noop_candidate()
        jsonschema.validate(instance=payload, schema=SCHEMA)

    def test_candidate_with_naturalistic_prior_valid(self) -> None:
        payload = _heuristic_candidate()
        payload["naturalistic_prior"] = _naturalistic_prior()
        jsonschema.validate(instance=payload, schema=SCHEMA)

    def test_candidate_with_naturalistic_prior_violation_valid(self) -> None:
        payload = _heuristic_candidate()
        payload["naturalistic_prior"] = _naturalistic_prior(passed=False)
        jsonschema.validate(instance=payload, schema=SCHEMA)

    def test_passed_naturalistic_prior_rejects_violation_flags(self) -> None:
        payload = _heuristic_candidate()
        payload["naturalistic_prior"] = _naturalistic_prior()
        payload["naturalistic_prior"]["violation_flags"] = ["inconsistent"]
        with pytest.raises(jsonschema.ValidationError, match="is expected to be empty"):
            jsonschema.validate(instance=payload, schema=SCHEMA)

    def test_naturalistic_prior_rejects_unsupported_field(self) -> None:
        payload = _heuristic_candidate()
        payload["naturalistic_prior"] = _naturalistic_prior()
        payload["naturalistic_prior"]["constraints"][0]["field"] = "runtime_acceleration_mps2"
        with pytest.raises(jsonschema.ValidationError, match="runtime_acceleration_mps2"):
            jsonschema.validate(instance=payload, schema=SCHEMA)

    @pytest.mark.parametrize("field", ["pedestrian_acceleration_mps2", "group_size"])
    def test_naturalistic_prior_accepts_explicit_optional_control_fields(self, field: str) -> None:
        payload = _heuristic_candidate()
        payload["naturalistic_prior"] = _naturalistic_prior(field=field)
        jsonschema.validate(instance=payload, schema=SCHEMA)


class TestRLAdversaryCandidate:
    """RL adversary candidate must validate with extra required fields."""

    def test_valid_rl_adversary_candidate(self) -> None:
        payload = _rl_adversary_candidate()
        jsonschema.validate(instance=payload, schema=SCHEMA)

    def test_rl_adversary_missing_checkpoint_rejects(self) -> None:
        payload = _rl_adversary_candidate()
        del payload["adversary_checkpoint_ref"]
        with pytest.raises(jsonschema.ValidationError, match="adversary_checkpoint_ref"):
            jsonschema.validate(instance=payload, schema=SCHEMA)


class TestDiffusionCandidate:
    """Diffusion prior candidate must validate with model checkpoint and sample params."""

    def test_valid_diffusion_candidate(self) -> None:
        payload = _diffusion_candidate()
        jsonschema.validate(instance=payload, schema=SCHEMA)

    def test_diffusion_missing_model_checkpoint_rejects(self) -> None:
        payload = _diffusion_candidate()
        del payload["model_checkpoint_ref"]
        with pytest.raises(jsonschema.ValidationError, match="model_checkpoint_ref"):
            jsonschema.validate(instance=payload, schema=SCHEMA)

    def test_diffusion_missing_sample_params_rejects(self) -> None:
        payload = _diffusion_candidate()
        del payload["sample_params"]
        with pytest.raises(jsonschema.ValidationError, match="sample_params"):
            jsonschema.validate(instance=payload, schema=SCHEMA)


class TestUnsupportedPerturbationFamily:
    """Unsupported perturbation families must be rejected."""

    def test_rejects_unsupported_perturbation_family(self) -> None:
        payload = _heuristic_candidate()
        payload["perturbations"] = [{"family": "illegal_family", "parameters": {}}]
        with pytest.raises(jsonschema.ValidationError, match="illegal_family"):
            jsonschema.validate(instance=payload, schema=SCHEMA)

    def test_rejects_extra_perturbation_properties(self) -> None:
        payload = _heuristic_candidate()
        payload["perturbations"][0]["extra_field"] = "not_allowed"
        with pytest.raises(jsonschema.ValidationError, match="extra_field"):
            jsonschema.validate(instance=payload, schema=SCHEMA)


class TestPromotionClaimRejection:
    """Promotion or training-ready claims must be rejected when prerequisites are missing."""

    def test_promotion_status_is_required(self) -> None:
        """promotion_status is required to prevent implicit training-ready assumptions."""
        payload = _heuristic_candidate()
        del payload["promotion_status"]
        with pytest.raises(jsonschema.ValidationError, match="promotion_status"):
            jsonschema.validate(instance=payload, schema=SCHEMA)

    def test_rejects_bogus_promotion_status(self) -> None:
        payload = _heuristic_candidate()
        payload["promotion_status"] = "training_ready"
        with pytest.raises(jsonschema.ValidationError, match="training_ready"):
            jsonschema.validate(instance=payload, schema=SCHEMA)

    def test_rejects_promoted_with_preflight_failed(self) -> None:
        """promoted status must fail closed when validity.preflight_passed is false."""
        payload = _heuristic_candidate()
        payload["promotion_status"] = "promoted"
        payload["validity"]["preflight_passed"] = False
        with pytest.raises(jsonschema.ValidationError, match="True was expected"):
            jsonschema.validate(instance=payload, schema=SCHEMA)

    @pytest.mark.parametrize(
        "missing_key",
        [
            "preflight_passed",
            "scenario_certified",
            "map_exists",
            "spawn_collision_free",
            "goal_reachable",
        ],
    )
    def test_rejects_promoted_with_missing_validity_prerequisite(self, missing_key: str) -> None:
        """promoted status must require all schema-level validity gates."""
        payload = _heuristic_candidate()
        payload["promotion_status"] = "promoted"
        del payload["validity"][missing_key]
        with pytest.raises(jsonschema.ValidationError, match=missing_key):
            jsonschema.validate(instance=payload, schema=SCHEMA)

    @pytest.mark.parametrize(
        "false_key",
        ["scenario_certified", "map_exists", "spawn_collision_free", "goal_reachable"],
    )
    def test_rejects_promoted_with_false_validity_prerequisite(self, false_key: str) -> None:
        """promoted status must reject false prerequisite gates."""
        payload = _heuristic_candidate()
        payload["promotion_status"] = "promoted"
        payload["validity"][false_key] = False
        with pytest.raises(jsonschema.ValidationError, match="True was expected"):
            jsonschema.validate(instance=payload, schema=SCHEMA)

    def test_not_promoted_is_default_safe_status(self) -> None:
        """not_promoted is the fail-closed default; promoted requires explicit claim."""
        payload = _heuristic_candidate()
        payload["promotion_status"] = "not_promoted"
        jsonschema.validate(instance=payload, schema=SCHEMA)


class TestTraceLineage:
    """Missing trace lineage must be rejected."""

    def test_rejects_missing_generator_config_ref(self) -> None:
        payload = _heuristic_candidate()
        del payload["generator_config_ref"]
        with pytest.raises(jsonschema.ValidationError, match="generator_config_ref"):
            jsonschema.validate(instance=payload, schema=SCHEMA)

    def test_rejects_missing_source_split(self) -> None:
        payload = _heuristic_candidate()
        del payload["source_split"]
        with pytest.raises(jsonschema.ValidationError, match="source_split"):
            jsonschema.validate(instance=payload, schema=SCHEMA)

    def test_rejects_invalid_source_split(self) -> None:
        payload = _heuristic_candidate()
        payload["source_split"] = "development"
        with pytest.raises(jsonschema.ValidationError, match="development"):
            jsonschema.validate(instance=payload, schema=SCHEMA)

    def test_rejects_missing_seed(self) -> None:
        payload = _heuristic_candidate()
        del payload["trace_lineage"]["seed"]
        with pytest.raises(jsonschema.ValidationError, match="seed"):
            jsonschema.validate(instance=payload, schema=SCHEMA)

    def test_rejects_missing_generated_at_utc(self) -> None:
        payload = _heuristic_candidate()
        del payload["trace_lineage"]["generated_at_utc"]
        with pytest.raises(jsonschema.ValidationError, match="generated_at_utc"):
            jsonschema.validate(instance=payload, schema=SCHEMA)

    def test_rejects_missing_git_hash(self) -> None:
        payload = _heuristic_candidate()
        del payload["trace_lineage"]["git_hash"]
        with pytest.raises(jsonschema.ValidationError, match="git_hash"):
            jsonschema.validate(instance=payload, schema=SCHEMA)

    def test_rejects_missing_provenance(self) -> None:
        payload = _heuristic_candidate()
        del payload["trace_lineage"]["provenance"]
        with pytest.raises(jsonschema.ValidationError, match="provenance"):
            jsonschema.validate(instance=payload, schema=SCHEMA)

    def test_rejects_missing_source_issue_in_provenance(self) -> None:
        payload = _heuristic_candidate()
        del payload["trace_lineage"]["provenance"]["source_issue"]
        with pytest.raises(jsonschema.ValidationError, match="source_issue"):
            jsonschema.validate(instance=payload, schema=SCHEMA)

    def test_rejects_empty_perturbations(self) -> None:
        payload = _heuristic_candidate()
        payload["perturbations"] = []
        with pytest.raises(jsonschema.ValidationError, match="minItems"):
            jsonschema.validate(instance=payload, schema=SCHEMA)


class TestSchemaStrictness:
    """Schema must reject extra top-level and nested properties."""

    def test_rejects_extra_top_level_property(self) -> None:
        payload = _heuristic_candidate()
        payload["unexpected_field"] = "nope"
        with pytest.raises(jsonschema.ValidationError, match="unexpected_field"):
            jsonschema.validate(instance=payload, schema=SCHEMA)

    def test_rejects_wrong_schema_version(self) -> None:
        payload = _heuristic_candidate()
        payload["schema_version"] = "generated_scenario_candidate.v0"
        with pytest.raises(jsonschema.ValidationError, match="generated_scenario_candidate.v1"):
            jsonschema.validate(instance=payload, schema=SCHEMA)

    def test_rejects_noop_with_parameters(self) -> None:
        payload = _noop_candidate()
        payload["perturbations"][0]["parameters"] = {"dx_m": 1.0}
        with pytest.raises(jsonschema.ValidationError, match="should not be valid"):
            jsonschema.validate(instance=payload, schema=SCHEMA)

    def test_rejects_extra_validity_property(self) -> None:
        payload = _heuristic_candidate()
        payload["validity"]["bogus_check"] = True
        with pytest.raises(jsonschema.ValidationError, match="bogus_check"):
            jsonschema.validate(instance=payload, schema=SCHEMA)

    def test_rejects_extra_metrics_summary_property(self) -> None:
        payload = _heuristic_candidate()
        payload["metrics_summary"]["bogus_metric"] = 1.0
        with pytest.raises(jsonschema.ValidationError, match="bogus_metric"):
            jsonschema.validate(instance=payload, schema=SCHEMA)

    def test_rejects_unsupported_generator_family(self) -> None:
        payload = _heuristic_candidate()
        payload["generator_family"] = "unknown_family"
        with pytest.raises(jsonschema.ValidationError, match="unknown_family"):
            jsonschema.validate(instance=payload, schema=SCHEMA)
