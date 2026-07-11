"""Regression tests for camera-ready runtime dependency and map preflight (issue #5300)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

import robot_sf.benchmark.camera_ready._preflight as preflight_module
from robot_sf.benchmark.camera_ready._config_types import CampaignConfig, PlannerSpec, SeedPolicy
from robot_sf.benchmark.camera_ready._preflight import prepare_campaign_preflight
from robot_sf.benchmark.campaign_runtime_preflight import (
    CampaignPolicyDependencyPreflightError,
    CampaignScenarioMapPreflightError,
    check_campaign_arm_policy_dependencies_preflight,
    check_campaign_scenario_maps_preflight,
)

if TYPE_CHECKING:
    from pathlib import Path


def _campaign(tmp_path: Path, planners: tuple[PlannerSpec, ...]) -> CampaignConfig:
    scenario_path = tmp_path / "scenarios.yaml"
    scenario_path.write_text("scenarios: []\n", encoding="utf-8")
    return CampaignConfig(
        name="runtime_preflight_test",
        scenario_matrix_path=scenario_path,
        planners=planners,
        seed_policy=SeedPolicy(),
    )


def test_policy_dependency_preflight_names_missing_arm_and_remediation(
    tmp_path: Path,
) -> None:
    """A missing PPO dependency fails before a campaign run with its exact remediation."""
    cfg = _campaign(tmp_path, (PlannerSpec(key="ppo_arm", algo="ppo"),))
    imported: list[str] = []

    def fail_import(module: str):
        imported.append(module)
        raise ModuleNotFoundError("No module named 'stable_baselines3'")

    with pytest.raises(CampaignPolicyDependencyPreflightError) as excinfo:
        check_campaign_arm_policy_dependencies_preflight(cfg, import_module=fail_import)

    assert imported == ["stable_baselines3"]
    assert excinfo.value.arms == ("ppo_arm",)
    assert "uv sync --all-extras" in str(excinfo.value)
    assert "stable_baselines3" in str(excinfo.value)


def test_policy_dependency_preflight_skips_disabled_arm(tmp_path: Path) -> None:
    """Disabled learned-policy arms do not block the campaign dependency gate."""
    cfg = _campaign(tmp_path, (PlannerSpec(key="ppo_off", algo="ppo", enabled=False),))
    assert check_campaign_arm_policy_dependencies_preflight(cfg) == {"checked": 0, "arms": []}


def test_map_resolvability_preflight_aggregates_every_bad_prepared_scenario() -> None:
    """Every bad normalized map reference is reported together before a run starts."""
    scenarios = [
        {"name": "bad_a", "map_file": "maps/missing_a.svg"},
        {"name": "valid", "map_file": "maps/valid.svg"},
        {"name": "bad_b", "map_file": "maps/missing_b.svg"},
    ]

    def resolve_map(map_file: str):
        if map_file.endswith("missing_a.svg"):
            raise RuntimeError("malformed SVG")
        return None if "missing" in map_file else object()

    with pytest.raises(CampaignScenarioMapPreflightError) as excinfo:
        check_campaign_scenario_maps_preflight(scenarios, resolve_map=resolve_map)

    assert excinfo.value.scenarios == ("bad_a", "bad_b")
    assert "maps/missing_a.svg" in str(excinfo.value)
    assert "maps/missing_b.svg" in str(excinfo.value)
    assert "RuntimeError: malformed SVG" in str(excinfo.value)
    assert "FileNotFoundError: map file could not be resolved or loaded" in str(excinfo.value)


def test_map_resolvability_preflight_accepts_valid_prepared_scenarios() -> None:
    """A fully resolvable prepared scenario list remains preflight-ready."""
    scenarios = [{"name": "valid", "map_file": "maps/valid.svg"}, {"name": "no_map"}]
    assert check_campaign_scenario_maps_preflight(
        scenarios,
        resolve_map=lambda _map_file: object(),
    ) == {"checked": 1, "scenario_count": 2}


def test_campaign_preflight_rejects_bad_map_from_prepared_scenarios(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The camera-ready entry point sweeps the post-normalization scenario list before a run."""
    cfg = _campaign(tmp_path, (PlannerSpec(key="goal", algo="goal"),))
    monkeypatch.setattr(
        preflight_module,
        "_load_campaign_scenarios",
        lambda _cfg: [{"name": "bad_prepared", "map_file": "maps/does_not_exist.svg"}],
    )

    with pytest.raises(CampaignScenarioMapPreflightError, match="bad_prepared"):
        prepare_campaign_preflight(cfg, output_root=tmp_path / "out", label="bad-map")
