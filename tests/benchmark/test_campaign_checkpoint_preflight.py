"""Tests for the campaign arm checkpoint preflight (issue #4613).

These exercise the fail-closed submit-time guard that verifies every enabled arm's checkpoint is
resolvable before a benchmark campaign runs, so a missing/corrupt checkpoint fails in seconds
rather than ~14h into compute. All tests are CPU-only and network-free: the ``model_id`` mode uses
a temporary registry fixture (present ``local_path`` => resolvable; absent/unknown => rejected).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
import yaml

import robot_sf.benchmark.camera_ready._preflight as preflight_module
from robot_sf.benchmark.camera_ready._config_types import (
    CampaignConfig,
    PlannerSpec,
    SeedPolicy,
)
from robot_sf.benchmark.camera_ready._preflight import prepare_campaign_preflight
from robot_sf.benchmark.campaign_checkpoint_preflight import (
    CampaignCheckpointPreflightError,
    check_campaign_arm_checkpoints_preflight,
    iter_campaign_arm_checkpoint_references,
)

if TYPE_CHECKING:
    from pathlib import Path


def _write_registry(tmp_path: Path, models: list[dict]) -> Path:
    """Write a minimal model registry YAML fixture and return its path."""
    registry_path = tmp_path / "registry.yaml"
    registry_path.write_text(
        yaml.safe_dump({"version": 1, "models": models}, sort_keys=False),
        encoding="utf-8",
    )
    return registry_path


def _write_algo_config(tmp_path: Path, name: str, payload: dict) -> Path:
    """Write an arm algo_config YAML and return its path."""
    path = tmp_path / name
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return path


def _campaign(planners: tuple[PlannerSpec, ...], *, tmp_path: Path) -> CampaignConfig:
    """Build a minimal campaign config wrapping the given planner arms."""
    scenario_path = tmp_path / "scenarios.yaml"
    if not scenario_path.exists():
        scenario_path.write_text("scenarios: []\n", encoding="utf-8")
    return CampaignConfig(
        name="checkpoint_preflight_test",
        scenario_matrix_path=scenario_path,
        planners=planners,
        seed_policy=SeedPolicy(),
    )


# --- reference extraction --------------------------------------------------


def test_iter_references_extracts_model_id_from_enabled_arm(tmp_path: Path) -> None:
    """An enabled arm's top-level model_id becomes a single checkpoint reference."""
    algo_config = _write_algo_config(tmp_path, "ppo.yaml", {"algo": "ppo", "model_id": "m1"})
    cfg = _campaign(
        (PlannerSpec(key="ppo", algo="ppo", algo_config_path=algo_config),),
        tmp_path=tmp_path,
    )
    refs = iter_campaign_arm_checkpoint_references(cfg)
    assert [(r.planner_key, r.kind, r.value) for r in refs] == [("ppo", "model_id", "m1")]


def test_iter_references_covers_nested_prior_checkpoint(tmp_path: Path) -> None:
    """A nested prior-policy model_id is still surfaced (recursive walk)."""
    algo_config = _write_algo_config(
        tmp_path,
        "guarded.yaml",
        {"algo": "guarded_ppo", "model_id": "policy", "prior": {"model_id": "prior_policy"}},
    )
    cfg = _campaign(
        (PlannerSpec(key="guarded", algo="guarded_ppo", algo_config_path=algo_config),),
        tmp_path=tmp_path,
    )
    values = sorted(r.value for r in iter_campaign_arm_checkpoint_references(cfg))
    assert values == ["policy", "prior_policy"]


def test_iter_references_skips_disabled_and_configless_arms(tmp_path: Path) -> None:
    """Disabled arms and arms without an algo_config contribute no references."""
    algo_config = _write_algo_config(tmp_path, "ppo.yaml", {"algo": "ppo", "model_id": "m1"})
    cfg = _campaign(
        (
            PlannerSpec(key="orca", algo="orca"),  # no algo_config
            PlannerSpec(
                key="ppo_off",
                algo="ppo",
                algo_config_path=algo_config,
                enabled=False,
            ),
        ),
        tmp_path=tmp_path,
    )
    assert iter_campaign_arm_checkpoint_references(cfg) == []


# --- cheap (network-free) resolvability ------------------------------------


def test_cheap_check_passes_when_local_path_present(tmp_path: Path) -> None:
    """A model_id whose registry local_path exists resolves without network access."""
    local_model = tmp_path / "weights" / "model.zip"
    local_model.parent.mkdir(parents=True)
    local_model.write_text("checkpoint", encoding="utf-8")
    registry_path = _write_registry(
        tmp_path, [{"model_id": "present", "local_path": str(local_model)}]
    )
    algo_config = _write_algo_config(tmp_path, "ppo.yaml", {"algo": "ppo", "model_id": "present"})
    cfg = _campaign(
        (PlannerSpec(key="ppo", algo="ppo", algo_config_path=algo_config),),
        tmp_path=tmp_path,
    )
    summary = check_campaign_arm_checkpoints_preflight(cfg, registry_path=registry_path)
    assert summary["resolved"] == 1
    assert summary["arms"][0]["status"] == "present_local"


def test_cheap_check_passes_when_remote_source_declared(tmp_path: Path) -> None:
    """A not-yet-cached model_id with a durable github_release source is 'stageable_remote'."""
    registry_path = _write_registry(
        tmp_path,
        [
            {
                "model_id": "remote",
                "local_path": "output/model_cache/remote/model.zip",  # absent
                "github_release": {
                    "asset_name": "remote-model.zip",
                    "url": "https://example.invalid/remote-model.zip",
                    "sha256": "0" * 64,
                },
            }
        ],
    )
    algo_config = _write_algo_config(tmp_path, "ppo.yaml", {"algo": "ppo", "model_id": "remote"})
    cfg = _campaign(
        (PlannerSpec(key="ppo", algo="ppo", algo_config_path=algo_config),),
        tmp_path=tmp_path,
    )
    summary = check_campaign_arm_checkpoints_preflight(cfg, registry_path=registry_path)
    assert summary["arms"][0]["status"] == "stageable_remote"


def test_cheap_check_rejects_unknown_model_id(tmp_path: Path) -> None:
    """An unknown/mistyped model_id fails closed and names the arm."""
    registry_path = _write_registry(
        tmp_path, [{"model_id": "known", "local_path": "output/model_cache/known/model.zip"}]
    )
    algo_config = _write_algo_config(tmp_path, "ppo.yaml", {"algo": "ppo", "model_id": "typo"})
    cfg = _campaign(
        (PlannerSpec(key="ppo", algo="ppo", algo_config_path=algo_config),),
        tmp_path=tmp_path,
    )
    with pytest.raises(CampaignCheckpointPreflightError) as excinfo:
        check_campaign_arm_checkpoints_preflight(cfg, registry_path=registry_path)
    assert excinfo.value.arms == ("ppo",)
    assert "typo" in str(excinfo.value)
    assert "unknown_model_id" in str(excinfo.value)


def test_cheap_check_rejects_local_only_missing(tmp_path: Path) -> None:
    """A local_only model with a missing local_path cannot be staged and fails closed."""
    registry_path = _write_registry(
        tmp_path,
        [
            {
                "model_id": "retired",
                "local_only": True,
                "local_path": str(tmp_path / "gone" / "model.zip"),
            }
        ],
    )
    algo_config = _write_algo_config(tmp_path, "ppo.yaml", {"algo": "ppo", "model_id": "retired"})
    cfg = _campaign(
        (PlannerSpec(key="ppo", algo="ppo", algo_config_path=algo_config),),
        tmp_path=tmp_path,
    )
    with pytest.raises(CampaignCheckpointPreflightError, match="local_only_missing"):
        check_campaign_arm_checkpoints_preflight(cfg, registry_path=registry_path)


def test_cheap_check_model_path_present_and_missing(tmp_path: Path) -> None:
    """A direct model_path reference passes when the file exists and fails when absent."""
    present = tmp_path / "model" / "policy.zip"
    present.parent.mkdir(parents=True)
    present.write_text("weights", encoding="utf-8")
    ok_config = _write_algo_config(tmp_path, "ok.yaml", {"algo": "ppo", "model_path": str(present)})
    cfg_ok = _campaign(
        (PlannerSpec(key="ppo", algo="ppo", algo_config_path=ok_config),),
        tmp_path=tmp_path,
    )
    assert check_campaign_arm_checkpoints_preflight(cfg_ok)["resolved"] == 1

    missing_config = _write_algo_config(
        tmp_path, "missing.yaml", {"algo": "ppo", "model_path": str(tmp_path / "nope.zip")}
    )
    cfg_missing = _campaign(
        (PlannerSpec(key="ppo", algo="ppo", algo_config_path=missing_config),),
        tmp_path=tmp_path,
    )
    with pytest.raises(CampaignCheckpointPreflightError, match="model_path_missing"):
        check_campaign_arm_checkpoints_preflight(cfg_missing)


def test_empty_campaign_returns_zero_checked(tmp_path: Path) -> None:
    """A config with no checkpoint-bearing arms is a no-op that never raises."""
    cfg = _campaign((PlannerSpec(key="orca", algo="orca"),), tmp_path=tmp_path)
    summary = check_campaign_arm_checkpoints_preflight(cfg)
    assert summary == {
        "checked": 0,
        "resolved": 0,
        "stage": False,
        "submit_safe": True,
        "arms": [],
    }


# --- staging mode ----------------------------------------------------------


def test_stage_mode_resolves_present_local_without_download(tmp_path: Path) -> None:
    """stage=True returns the present local_path (no download needed) and reports 'staged'."""
    local_model = tmp_path / "weights" / "model.zip"
    local_model.parent.mkdir(parents=True)
    local_model.write_text("checkpoint", encoding="utf-8")
    registry_path = _write_registry(
        tmp_path, [{"model_id": "present", "local_path": str(local_model)}]
    )
    algo_config = _write_algo_config(tmp_path, "ppo.yaml", {"algo": "ppo", "model_id": "present"})
    cfg = _campaign(
        (PlannerSpec(key="ppo", algo="ppo", algo_config_path=algo_config),),
        tmp_path=tmp_path,
    )
    summary = check_campaign_arm_checkpoints_preflight(cfg, stage=True, registry_path=registry_path)
    assert summary["stage"] is True
    assert summary["arms"][0]["status"] == "staged"


def test_stage_mode_rejects_unknown_model_id(tmp_path: Path) -> None:
    """stage=True fails closed for an unknown model_id (no silent fallback)."""
    registry_path = _write_registry(tmp_path, [])
    algo_config = _write_algo_config(tmp_path, "ppo.yaml", {"algo": "ppo", "model_id": "missing"})
    cfg = _campaign(
        (PlannerSpec(key="ppo", algo="ppo", algo_config_path=algo_config),),
        tmp_path=tmp_path,
    )
    with pytest.raises(CampaignCheckpointPreflightError, match="stage_failed"):
        check_campaign_arm_checkpoints_preflight(cfg, stage=True, registry_path=registry_path)


# --- wiring: fail fast before scenarios load -------------------------------


def test_prepare_campaign_preflight_rejects_missing_checkpoint_before_scenarios(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The campaign preflight raises for an unresolvable arm checkpoint before scenarios load."""
    algo_config = _write_algo_config(
        tmp_path,
        "ppo.yaml",
        {"algo": "ppo", "model_id": "definitely_absent_model_id_issue_4613"},
    )
    cfg = _campaign(
        (PlannerSpec(key="ppo", algo="ppo", algo_config_path=algo_config),),
        tmp_path=tmp_path,
    )

    def fail_if_scenarios_load(*_args, **_kwargs):
        raise AssertionError("_load_campaign_scenarios ran before the checkpoint preflight")

    monkeypatch.setattr(preflight_module, "_load_campaign_scenarios", fail_if_scenarios_load)

    with pytest.raises(CampaignCheckpointPreflightError, match="unknown_model_id"):
        prepare_campaign_preflight(cfg, output_root=tmp_path / "out", label="ckpt")
