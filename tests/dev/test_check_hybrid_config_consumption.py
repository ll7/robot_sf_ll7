"""Reject unreachable static-clearance tuning before development runs."""

from __future__ import annotations

import subprocess
import sys
from typing import TYPE_CHECKING

import pytest
import yaml

from scripts.validation.check_hybrid_config_consumption import check_config

if TYPE_CHECKING:
    from pathlib import Path


@pytest.mark.parametrize(
    "key", ["static_clearance_escape_enabled", "static_corridor_transit_enabled"]
)
def test_continuous_check_rejects_unreachable_enabled_rules(tmp_path: Path, key: str) -> None:
    path = tmp_path / "planner.yaml"
    path.write_text(yaml.safe_dump({"continuous_static_clearance_enabled": True, key: True}))
    problems = check_config(path)
    assert any(key in problem and "unreachable" in problem for problem in problems)


def test_scenario_override_is_checked_after_real_manifest_resolution(tmp_path: Path) -> None:
    base = tmp_path / "base.yaml"
    base.write_text("continuous_static_clearance_enabled: true\n")
    path = tmp_path / "candidate.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "base_config_path": str(base),
                "params": {},
                "scenario_overrides": {
                    "classic_join_group": {"static_clearance_escape_enabled": True}
                },
            }
        )
    )
    assert any(
        "classic_join_group" in p and "static_clearance_escape_enabled" in p
        for p in check_config(path)
    )


@pytest.mark.parametrize(
    "cfg",
    [
        {"continuous_static_clearance_enabled": False, "static_clearance_escape_enabled": True},
        {"continuous_static_clearance_enabled": True, "static_clearance_escape_enabled": False},
        {
            "continuous_static_clearance_enabled": True,
            "corridor_subgoal_use_continuous_static_check": False,
            "static_corridor_transit_enabled": True,
        },
    ],
)
def test_reachable_or_disabled_rules_are_accepted(tmp_path: Path, cfg: dict) -> None:
    path = tmp_path / "planner.yaml"
    path.write_text(yaml.safe_dump(cfg))
    assert check_config(path) == []


def test_cli_fails_closed_on_invalid_yaml_shape(tmp_path: Path) -> None:
    path = tmp_path / "bad.yaml"
    path.write_text("- not-a-config\n")
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "scripts.validation.check_hybrid_config_consumption",
            str(path),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode != 0
    assert "mapping" in result.stderr


def test_cli_checks_versioned_development_manifest() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "scripts.validation.check_hybrid_config_consumption",
            "configs/algos/hybrid_rule_v4_predictive_braking_dev_v1.yaml",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 1, result.stderr
    assert "francis2023_perpendicular_traffic" in result.stdout
    assert "static_clearance_escape_enabled" in result.stdout
    assert "static_corridor_transit_enabled" in result.stdout


def test_family_override_uses_runtime_boolean_coercion(tmp_path: Path) -> None:
    path = tmp_path / "family.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "params": {"continuous_static_clearance_enabled": "true"},
                "family_overrides": {"classic": {"static_corridor_transit_enabled": "yes"}},
            }
        )
    )
    assert any(
        "family:classic" in p and "static_corridor_transit_enabled" in p for p in check_config(path)
    )


@pytest.mark.parametrize(
    ("manifest", "scenario"),
    [
        (
            {
                "params": {
                    "continuous_static_clearance_enabled": True,
                    "static_corridor_transit_enabled": True,
                },
                "scenario_overrides": {"default": {"static_corridor_transit_enabled": False}},
            },
            {"name": "classic_join_group"},
        ),
        (
            {
                "params": {},
                "family_overrides": {"custom": {"continuous_static_clearance_enabled": True}},
                "scenario_overrides": {"custom_case": {"static_corridor_transit_enabled": True}},
            },
            {"name": "custom_case", "family": "custom"},
        ),
    ],
    ids=["default-name-collision", "family-scenario-composition"],
)
def test_reviewer_runtime_masked_contexts_are_rejected(
    tmp_path: Path, manifest: dict, scenario: dict
) -> None:
    """Reject reviewer fixtures whose real runtime configuration masks the corridor rule."""
    from robot_sf.benchmark.policy_search_manifest import resolve_candidate_manifest_runtime
    from robot_sf.planner.hybrid_rule_local_planner import build_hybrid_rule_local_planner_config

    _, effective = resolve_candidate_manifest_runtime(
        default_algo="hybrid_rule_local_planner",
        manifest=manifest,
        scenario=scenario,
        load_config=lambda _: {},
    )
    config = build_hybrid_rule_local_planner_config(effective)
    assert config.continuous_static_clearance_enabled is True
    assert config.corridor_subgoal_use_continuous_static_check is True
    assert config.static_corridor_transit_enabled is True

    path = tmp_path / "reviewer.yaml"
    path.write_text(yaml.safe_dump(manifest))
    assert any("static_corridor_transit_enabled" in p for p in check_config(path))
    result = subprocess.run(
        [sys.executable, "-m", "scripts.validation.check_hybrid_config_consumption", str(path)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 1, result.stderr
    assert "static_corridor_transit_enabled" in result.stdout
