"""Tests for the issue #5409 guarded-PPO availability preflight.

CPU-only and network-free. The preflight reuses the repository's existing
checkpoint-resolution and observation-contract machinery, so these tests verify
the *wiring* (a guarded_ppo arm is detected, its two declared dependencies are
resolved, and the available/not-available verdict is derived correctly) without
running any campaign or touching the network.

The real issue #5409 configs are exercised where present; synthetic fixtures
cover the missing-dependency and absent-arm branches.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
_SCRIPT_PATH = REPO_ROOT / "scripts/benchmark/check_guarded_ppo_availability_issue_5409.py"
_H500_CONFIG = REPO_ROOT / "configs/benchmarks/issue_5409_horizon_ablation_h500.yaml"
_H600_CONFIG = REPO_ROOT / "configs/benchmarks/issue_5409_horizon_ablation_h600.yaml"


def _load_script_module():
    """Load the preflight script as a module for testing."""
    spec = importlib.util.spec_from_file_location(
        "check_guarded_ppo_availability_issue_5409", _SCRIPT_PATH
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules["check_guarded_ppo_availability_issue_5409"] = mod
    spec.loader.exec_module(mod)
    return mod


_mod = _load_script_module()
check_guarded_ppo_availability = _mod.check_guarded_ppo_availability
EXIT_AVAILABLE = _mod.EXIT_AVAILABLE
EXIT_NOT_AVAILABLE = _mod.EXIT_NOT_AVAILABLE
EXIT_CONFIG_ERROR = _mod.EXIT_CONFIG_ERROR


def _write_guarded_ppo_config(
    tmp_path: Path, *, with_checkpoint: bool, with_contract: bool
) -> Path:
    """Write a minimal campaign config with a guarded_ppo arm and dependencies toggled."""
    registry_path = tmp_path / "registry.yaml"
    if with_checkpoint:
        registry_payload = {
            "version": 1,
            "models": [
                {
                    "model_id": "guarded_ckpt",
                    "local_path": str(tmp_path / "guarded_ckpt-model.zip"),
                }
            ],
        }
    else:
        registry_payload = {"version": 1, "models": []}
    registry_path.write_text(yaml.safe_dump(registry_payload), encoding="utf-8")

    algo_cfg = tmp_path / "guarded_ppo.yaml"
    if with_contract:
        algo_payload = {
            "model_id": "guarded_ckpt",
            "benchmark_promotion": {
                "observation_level": "tracked_agents_no_noise",
                "observation_mode": "dict",
            },
        }
    else:
        algo_payload = {"model_id": "guarded_ckpt"}
    algo_cfg.write_text(yaml.safe_dump(algo_payload), encoding="utf-8")

    (tmp_path / "guarded_ckpt-model.zip").write_text("dummy", encoding="utf-8")

    config = tmp_path / "campaign.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "name": "guarded_test",
                "scenario_matrix": "configs/scenarios/classic_interactions_francis2023.yaml",
                "horizon": 500,
                "planners": [
                    {
                        "key": "goal",
                        "algo": "goal",
                        "planner_group": "core",
                        "benchmark_profile": "baseline-safe",
                    },
                    {
                        "key": "guarded_ppo",
                        "algo": "guarded_ppo",
                        "planner_group": "experimental",
                        "benchmark_profile": "experimental",
                        "algo_config": str(algo_cfg),
                        "availability_gate": "dependency_gated",
                        "fail_closed_reason": (
                            "guarded_ppo_checkpoint_observation_contract_missing"
                        ),
                    },
                ],
                "seed_policy": {
                    "mode": "seed-set",
                    "seed_set": "eval",
                    "seed_sets_path": "configs/benchmarks/seed_sets_v1.yaml",
                },
            }
        ),
        encoding="utf-8",
    )
    return config


class TestGuardedPpoAvailability:
    """Verdict derivation for the guarded_ppo arm."""

    def test_arm_present_and_available_when_both_deps_resolve(self, tmp_path: Path) -> None:
        config = _write_guarded_ppo_config(tmp_path, with_checkpoint=True, with_contract=True)
        verdict = check_guarded_ppo_availability(config, registry_path=tmp_path / "registry.yaml")
        assert verdict["present"] is True
        assert verdict["available"] is True
        assert verdict["remaining_missing_dependencies"] == []
        assert verdict["verdict"] == "available"

    def test_not_available_when_checkpoint_missing(self, tmp_path: Path) -> None:
        registry_path = tmp_path / "registry.yaml"
        registry_path.write_text(yaml.safe_dump({"version": 1, "models": []}), encoding="utf-8")
        algo_cfg = tmp_path / "guarded_ppo.yaml"
        algo_cfg.write_text(
            yaml.safe_dump(
                {
                    "model_id": "guarded_ckpt",
                    "benchmark_promotion": {
                        "observation_level": "tracked_agents_no_noise",
                        "observation_mode": "dict",
                    },
                }
            ),
            encoding="utf-8",
        )
        config = tmp_path / "campaign.yaml"
        config.write_text(
            yaml.safe_dump(
                {
                    "name": "guarded_test",
                    "scenario_matrix": "configs/scenarios/classic_interactions_francis2023.yaml",
                    "horizon": 500,
                    "planners": [
                        {
                            "key": "guarded_ppo",
                            "algo": "guarded_ppo",
                            "planner_group": "experimental",
                            "benchmark_profile": "experimental",
                            "algo_config": str(algo_cfg),
                            "availability_gate": "dependency_gated",
                            "fail_closed_reason": (
                                "guarded_ppo_checkpoint_observation_contract_missing"
                            ),
                        },
                    ],
                    "seed_policy": {
                        "mode": "seed-set",
                        "seed_set": "eval",
                        "seed_sets_path": "configs/benchmarks/seed_sets_v1.yaml",
                    },
                }
            ),
            encoding="utf-8",
        )
        verdict = check_guarded_ppo_availability(config, registry_path=registry_path)
        assert verdict["available"] is False
        assert "checkpoint" in verdict["remaining_missing_dependencies"]
        assert verdict["declared_fail_closed_reason"] == (
            "guarded_ppo_checkpoint_observation_contract_missing"
        )

    def test_absent_arm_reports_present_false(self, tmp_path: Path) -> None:
        config = tmp_path / "campaign.yaml"
        config.write_text(
            yaml.safe_dump(
                {
                    "name": "no_guarded",
                    "scenario_matrix": "configs/scenarios/classic_interactions_francis2023.yaml",
                    "horizon": 500,
                    "planners": [
                        {
                            "key": "goal",
                            "algo": "goal",
                            "planner_group": "core",
                            "benchmark_profile": "baseline-safe",
                        },
                    ],
                    "seed_policy": {
                        "mode": "seed-set",
                        "seed_set": "eval",
                        "seed_sets_path": "configs/benchmarks/seed_sets_v1.yaml",
                    },
                }
            ),
            encoding="utf-8",
        )
        verdict = check_guarded_ppo_availability(config)
        assert verdict["present"] is False
        assert verdict["available"] is False


class TestActualIssue5409Configs:
    """Exercise the real issue #5409 ablation configs where present."""

    def test_h500_guarded_ppo_arm_resolves_available(self) -> None:
        if not _H500_CONFIG.exists():
            pytest.skip("issue #5409 h500 config not present in this checkout")
        verdict = check_guarded_ppo_availability(_H500_CONFIG)
        assert verdict["present"] is True
        assert verdict["planner_key"] == "guarded_ppo"
        assert verdict["available"] is True
        assert verdict["checkpoint"]["resolvable"] is True
        assert verdict["observation_contract"]["resolvable"] is True

    def test_h600_guarded_ppo_arm_resolves_available(self) -> None:
        if not _H600_CONFIG.exists():
            pytest.skip("issue #5409 h600 config not present in this checkout")
        verdict = check_guarded_ppo_availability(_H600_CONFIG)
        assert verdict["present"] is True
        assert verdict["available"] is True


class TestCliExitCodes:
    """Exit-code contract for the submit wrapper."""

    def test_available_exit_zero(self, tmp_path: Path) -> None:
        config = _write_guarded_ppo_config(tmp_path, with_checkpoint=True, with_contract=True)
        exit_code = _mod.main(
            ["--config", str(config), "--registry-path", str(tmp_path / "registry.yaml")]
        )
        assert exit_code == EXIT_AVAILABLE

    def test_not_available_exit_one(self, tmp_path: Path) -> None:
        config = _write_guarded_ppo_config(tmp_path, with_checkpoint=False, with_contract=True)
        exit_code = _mod.main(
            [
                "--config",
                str(config),
                "--registry-path",
                str(tmp_path / "registry.yaml"),
            ]
        )
        assert exit_code == EXIT_NOT_AVAILABLE

    def test_missing_config_exit_two(self, tmp_path: Path) -> None:
        exit_code = _mod.main(["--config", str(tmp_path / "missing.yaml")])
        assert exit_code == EXIT_CONFIG_ERROR
