"""Tests for the horizon ablation pair validation script."""

from __future__ import annotations

import importlib.util
import pathlib
import sys

import pytest
import yaml

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
_SCRIPT_PATH = REPO_ROOT / "scripts/benchmark/validate_horizon_ablation_pair.py"


def _load_script_module():
    """Load the validation script as a module for testing."""
    spec = importlib.util.spec_from_file_location("validate_horizon_ablation_pair", _SCRIPT_PATH)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["validate_horizon_ablation_pair"] = mod
    spec.loader.exec_module(mod)
    return mod


_mod = _load_script_module()

_load_raw_config = _mod._load_raw_config
_optional_int = _mod._optional_int
_optional_seed_id = _mod._optional_seed_id
_planner_roster = _mod._planner_roster
_seed_policy_signature = _mod._seed_policy_signature
_compare_planner_rosters = _mod._compare_planner_rosters
_check_horizon_differs = _mod._check_horizon_differs
_check_field_parity = _mod._check_field_parity
validate_horizon_ablation_pair = _mod.validate_horizon_ablation_pair
HorizonAblationPairResult = _mod.HorizonAblationPairResult


def _write_yaml(tmp_path: pathlib.Path, name: str, payload: dict) -> pathlib.Path:
    """Write a YAML config to a temp file and return its path."""
    path = tmp_path / name
    path.write_text(yaml.dump(payload, default_flow_style=False), encoding="utf-8")
    return path


def _make_base_payload(horizon: int = 500) -> dict:
    """Return a minimal valid campaign config payload."""
    return {
        "name": f"test_h{horizon}",
        "scenario_matrix": "configs/scenarios/classic_interactions_francis2023.yaml",
        "horizon": horizon,
        "dt": 0.1,
        "workers": 2,
        "kinematics_matrix": ["differential_drive"],
        "seed_policy": {
            "mode": "seed-set",
            "seed_set": "eval",
            "seed_sets_path": "configs/benchmarks/seed_sets_v1.yaml",
        },
        "planners": [
            {
                "key": "goal",
                "algo": "goal",
                "planner_group": "core",
                "benchmark_profile": "baseline-safe",
            },
            {
                "key": "social_force",
                "algo": "social_force",
                "planner_group": "core",
                "benchmark_profile": "baseline-safe",
            },
        ],
        "snqi_contract": {"enabled": True, "enforcement": "warn"},
        "amv_profile": {"name": "amv-paper-v1", "contract_version": "1"},
    }


class TestPlannerRosterExtraction:
    """Planner roster extraction from raw config payloads."""

    def test_extracts_keys_and_algos(self, tmp_path: pathlib.Path) -> None:
        payload = _make_base_payload()
        roster = _planner_roster(payload)
        assert len(roster) == 2
        assert roster[0]["key"] == "goal"
        assert roster[0]["algo"] == "goal"

    def test_raises_on_missing_planners(self) -> None:
        with pytest.raises(ValueError, match="no 'planners' list"):
            _planner_roster({})

    def test_raises_on_empty_planners(self) -> None:
        with pytest.raises(ValueError, match="no 'planners' list"):
            _planner_roster({"planners": []})


class TestComparePlannerRosters:
    """Planner roster comparison logic."""

    def test_identical_rosters_match(self) -> None:
        roster = [{"key": "goal", "algo": "goal", "algo_config": ""}]
        assert _compare_planner_rosters(roster, roster) == []

    def test_different_keys_detected(self) -> None:
        a = [{"key": "goal", "algo": "goal", "algo_config": ""}]
        b = [{"key": "orca", "algo": "orca", "algo_config": ""}]
        errors = _compare_planner_rosters(a, b)
        assert any("only in config A" in e for e in errors)
        assert any("only in config B" in e for e in errors)

    def test_different_algo_config_detected(self) -> None:
        a = [{"key": "ppo", "algo": "ppo", "algo_config": "configs/a.yaml"}]
        b = [{"key": "ppo", "algo": "ppo", "algo_config": "configs/b.yaml"}]
        errors = _compare_planner_rosters(a, b)
        assert any("algo_config differs" in e for e in errors)

    def test_different_order_detected(self) -> None:
        a = [
            {"key": "goal", "algo": "goal", "algo_config": ""},
            {"key": "orca", "algo": "orca", "algo_config": ""},
        ]
        b = [
            {"key": "orca", "algo": "orca", "algo_config": ""},
            {"key": "goal", "algo": "goal", "algo_config": ""},
        ]
        errors = _compare_planner_rosters(a, b)
        assert any("order differs" in e for e in errors)


class TestHorizonDiffers:
    """Horizon-difference checks."""

    def test_optional_int_rejects_non_integer_values(self) -> None:
        with pytest.raises(ValueError):
            _optional_int(500.5)
        with pytest.raises(ValueError):
            _optional_int(True)

    def test_missing_horizons_report_actionable_errors(self) -> None:
        h_a, h_b, errors = _check_horizon_differs({}, {})
        assert h_a is None
        assert h_b is None
        assert any("Config A is missing" in error for error in errors)
        assert any("Config B is missing" in error for error in errors)

    def test_invalid_horizons_report_actionable_errors(self) -> None:
        _, _, errors = _check_horizon_differs({"horizon": "bad"}, {"horizon": 600})
        assert any("Config A horizon is not a valid integer" in error for error in errors)

    def test_different_horizons_pass(self) -> None:
        a = {"horizon": 500}
        b = {"horizon": 600}
        h_a, h_b, errors = _check_horizon_differs(a, b)
        assert h_a == 500
        assert h_b == 600
        assert errors == []

    def test_identical_horizons_fail(self) -> None:
        a = {"horizon": 500}
        b = {"horizon": 500}
        _, _, errors = _check_horizon_differs(a, b)
        assert any("identical" in e for e in errors)

    def test_scenario_horizons_flagged(self) -> None:
        a = {"scenario_horizons": "configs/policy_search/scenario_horizons_h500.yaml"}
        b = {"horizon": 600}
        _, _, errors = _check_horizon_differs(a, b)
        assert any("scenario_horizons" in e for e in errors)


class TestSeedPolicyComparison:
    """Seed policy signature extraction and comparison."""

    def test_optional_seed_id_preserves_zero_and_normalizes_numeric_strings(self) -> None:
        assert _optional_seed_id(0) == 0
        assert _optional_seed_id(" 7 ") == 7
        assert _seed_policy_signature({"seed_policy": {"seeds": 0}})["seeds"] == [0]

    def test_seed_policy_signature_handles_null_and_scalar_values(self) -> None:
        assert _seed_policy_signature({"seed_policy": {"seeds": None}})["seeds"] == []
        assert _seed_policy_signature({"seed_policy": {"seeds": "eval-1"}})["seeds"] == ["eval-1"]

    def test_identical_policies_match(self) -> None:
        payload = _make_base_payload()
        sig_a = _seed_policy_signature(payload)
        sig_b = _seed_policy_signature(payload)
        assert sig_a == sig_b

    def test_different_seed_sets_detected(self) -> None:
        a = _make_base_payload()
        b = _make_base_payload()
        b["seed_policy"]["seed_set"] = "paper_eval_s30"
        assert _seed_policy_signature(a) != _seed_policy_signature(b)


class TestFieldParity:
    """Execution-relevant field parity checks."""

    def test_identical_payloads_match(self) -> None:
        a = _make_base_payload(500)
        b = _make_base_payload(600)
        errors = _check_field_parity(a, b)
        assert errors == []

    def test_different_dt_detected(self) -> None:
        a = _make_base_payload(500)
        b = _make_base_payload(600)
        b["dt"] = 0.2
        errors = _check_field_parity(a, b)
        assert any("dt" in e for e in errors)

    def test_different_scenario_matrix_detected(self) -> None:
        a = _make_base_payload(500)
        b = _make_base_payload(600)
        b["scenario_matrix"] = "configs/scenarios/other.yaml"
        errors = _check_field_parity(a, b)
        assert any("scenario_matrix" in e for e in errors)

    def test_different_publication_overwrite_policy_detected(self) -> None:
        a = _make_base_payload(500)
        b = _make_base_payload(600)
        a["overwrite_publication_bundle"] = False
        b["overwrite_publication_bundle"] = True
        errors = _check_field_parity(a, b)
        assert any("overwrite_publication_bundle" in e for e in errors)

    def test_non_mapping_contract_fields_fail_closed(self) -> None:
        payload = _make_base_payload()
        payload["snqi_contract"] = []
        with pytest.raises(ValueError, match="snqi_contract must be a mapping"):
            _check_field_parity(payload, _make_base_payload())

        payload = _make_base_payload()
        payload["amv_profile"] = "invalid"
        with pytest.raises(ValueError, match="amv_profile must be a mapping"):
            _check_field_parity(payload, _make_base_payload())


class TestValidateHorizonAblationPair:
    """End-to-end validation of config pairs."""

    def test_valid_pair_passes(self, tmp_path: pathlib.Path) -> None:
        path_a = _write_yaml(tmp_path, "h500.yaml", _make_base_payload(500))
        path_b = _write_yaml(tmp_path, "h600.yaml", _make_base_payload(600))
        result = validate_horizon_ablation_pair(path_a, path_b)
        assert result.is_valid
        assert result.horizon_a == 500
        assert result.horizon_b == 600
        assert result.mismatches == []

    def test_mismatched_roster_fails(self, tmp_path: pathlib.Path) -> None:
        a = _make_base_payload(500)
        b = _make_base_payload(600)
        b["planners"].append(
            {
                "key": "ppo",
                "algo": "ppo",
                "planner_group": "experimental",
                "benchmark_profile": "experimental",
            }
        )
        path_a = _write_yaml(tmp_path, "h500.yaml", a)
        path_b = _write_yaml(tmp_path, "h600.yaml", b)
        result = validate_horizon_ablation_pair(path_a, path_b)
        assert not result.is_valid
        assert any("Planners only" in m for m in result.mismatches)

    def test_mismatched_seed_fails(self, tmp_path: pathlib.Path) -> None:
        a = _make_base_payload(500)
        b = _make_base_payload(600)
        b["seed_policy"]["seed_set"] = "paper_eval_s30"
        path_a = _write_yaml(tmp_path, "h500.yaml", a)
        path_b = _write_yaml(tmp_path, "h600.yaml", b)
        result = validate_horizon_ablation_pair(path_a, path_b)
        assert not result.is_valid
        assert any("Seed policy" in m for m in result.mismatches)

    def test_same_horizon_fails(self, tmp_path: pathlib.Path) -> None:
        a = _make_base_payload(500)
        b = _make_base_payload(500)
        path_a = _write_yaml(tmp_path, "h500a.yaml", a)
        path_b = _write_yaml(tmp_path, "h500b.yaml", b)
        result = validate_horizon_ablation_pair(path_a, path_b)
        assert not result.is_valid
        assert any("identical" in m for m in result.mismatches)

    def test_invalid_horizon_returns_invalid_result(self, tmp_path: pathlib.Path) -> None:
        path_a = _write_yaml(tmp_path, "invalid.yaml", {"horizon": "bad"})
        path_b = _write_yaml(tmp_path, "valid.yaml", {"horizon": 600})
        result = validate_horizon_ablation_pair(path_a, path_b)
        assert not result.is_valid
        assert any("not a valid integer" in mismatch for mismatch in result.mismatches)

    def test_non_mapping_contract_returns_invalid_result(self, tmp_path: pathlib.Path) -> None:
        a = _make_base_payload(500)
        b = _make_base_payload(600)
        b["amv_profile"] = []
        path_a = _write_yaml(tmp_path, "h500.yaml", a)
        path_b = _write_yaml(tmp_path, "h600.yaml", b)
        result = validate_horizon_ablation_pair(path_a, path_b)
        assert not result.is_valid
        assert any("Contract field validation error" in mismatch for mismatch in result.mismatches)

    def test_missing_file_returns_loading_error(self, tmp_path: pathlib.Path) -> None:
        result = validate_horizon_ablation_pair(
            tmp_path / "missing.yaml", tmp_path / "also_missing.yaml"
        )
        assert not result.is_valid
        assert any("loading error" in m.lower() for m in result.mismatches)

    def test_to_payload_roundtrip(self, tmp_path: pathlib.Path) -> None:
        path_a = _write_yaml(tmp_path, "h500.yaml", _make_base_payload(500))
        path_b = _write_yaml(tmp_path, "h600.yaml", _make_base_payload(600))
        result = validate_horizon_ablation_pair(path_a, path_b)
        payload = result.to_payload()
        assert payload["schema_version"] == "horizon_ablation_pair_validation.v1"
        assert payload["is_valid"] is True
        assert payload["horizon_a"] == 500
        assert payload["horizon_b"] == 600


class TestActualConfigPair:
    """Validate the actual issue #5409 configs against each other."""

    def test_issue_5409_configs_are_valid_pair(self) -> None:
        h500 = REPO_ROOT / "configs/benchmarks/issue_5409_horizon_ablation_h500.yaml"
        h600 = REPO_ROOT / "configs/benchmarks/issue_5409_horizon_ablation_h600.yaml"
        if not h500.exists() or not h600.exists():
            pytest.skip("Issue #5409 configs not present in this checkout")
        result = validate_horizon_ablation_pair(h500, h600)
        assert result.is_valid, f"Mismatches: {result.mismatches}"

    def test_issue_5409_configs_have_correct_horizons(self) -> None:
        h500 = REPO_ROOT / "configs/benchmarks/issue_5409_horizon_ablation_h500.yaml"
        h600 = REPO_ROOT / "configs/benchmarks/issue_5409_horizon_ablation_h600.yaml"
        if not h500.exists() or not h600.exists():
            pytest.skip("Issue #5409 configs not present in this checkout")
        result = validate_horizon_ablation_pair(h500, h600)
        assert result.horizon_a == 500
        assert result.horizon_b == 600

    def test_issue_5409_configs_have_12_planners(self) -> None:
        h500 = REPO_ROOT / "configs/benchmarks/issue_5409_horizon_ablation_h500.yaml"
        if not h500.exists():
            pytest.skip("Issue #5409 config not present in this checkout")
        payload = _load_raw_config(h500)
        roster = _planner_roster(payload)
        assert len(roster) == 12

    def test_issue_5409_configs_use_eval_seeds(self) -> None:
        h500 = REPO_ROOT / "configs/benchmarks/issue_5409_horizon_ablation_h500.yaml"
        if not h500.exists():
            pytest.skip("Issue #5409 config not present in this checkout")
        payload = _load_raw_config(h500)
        seed_sig = _seed_policy_signature(payload)
        assert seed_sig["mode"] == "seed-set"
        assert seed_sig["seed_set"] == "eval"
