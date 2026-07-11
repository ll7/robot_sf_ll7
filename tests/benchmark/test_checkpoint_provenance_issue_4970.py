"""Checkpoint provenance contract tests for issue #4970."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

import robot_sf.benchmark.camera_ready._preflight as preflight_module
import robot_sf.benchmark.campaign_checkpoint_preflight as checkpoint_module
from robot_sf.benchmark import map_runner
from robot_sf.benchmark.camera_ready._config import _validate_campaign_config
from robot_sf.benchmark.camera_ready._config_types import CampaignConfig, PlannerSpec
from robot_sf.benchmark.camera_ready._preflight import prepare_campaign_preflight
from robot_sf.benchmark.camera_ready.campaign import (
    _checkpoint_fallback_detected,
    _finalize_checkpoint_provenance,
)
from robot_sf.benchmark.campaign_checkpoint_preflight import (
    CampaignCheckpointPreflightError,
    check_campaign_arm_checkpoints_preflight,
)
from robot_sf.benchmark.map_runner_batch_summary import merge_runtime_algorithm_contract
from robot_sf.planner.socnav import SACADRLPlannerAdapter, SocNavPlannerConfig

if TYPE_CHECKING:
    from pathlib import Path


def _campaign(
    tmp_path: Path,
    *,
    enforcement: str = "off",
    planner: PlannerSpec | None = None,
) -> CampaignConfig:
    scenario_path = tmp_path / "scenarios.yaml"
    scenario_path.write_text("scenarios: []\n", encoding="utf-8")
    return CampaignConfig(
        name="issue_4970",
        scenario_matrix_path=scenario_path,
        planners=(planner or PlannerSpec(key="sacadrl", algo="sacadrl"),),
        checkpoint_provenance_enforcement=enforcement,
    )


def test_default_sacadrl_checkpoint_is_hashed_in_preflight_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The implicit registry checkpoint is visible even without an algo_config block."""
    monkeypatch.setattr(preflight_module, "_load_campaign_scenarios", lambda *_a, **_k: [])
    prepared = prepare_campaign_preflight(_campaign(tmp_path), output_root=tmp_path / "out")

    provenance = prepared["manifest_payload"]["planners"][0]["checkpoint_provenance"]
    assert provenance["status"] == "not_run"
    assert provenance["model_id"] == "ga3c_cadrl_iros18"
    assert len(provenance["checkpoint_sha256"]) == 64
    assert provenance["load_succeeded"] is None
    assert provenance["fallback_triggered"] is None
    assert provenance["references"][0]["implicit"] is True


def test_implicit_checkpoint_is_audit_only_by_default_but_strict_mode_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Default behavior stays compatible while the campaign flag makes absence blocking."""

    def missing_entry(_model_id: str, _registry_path: str | Path | None = None) -> dict:
        return {"model_id": "ga3c_cadrl_iros18", "local_path": str(tmp_path / "missing.meta")}

    monkeypatch.setattr(checkpoint_module, "get_registry_entry", missing_entry)
    summary = check_campaign_arm_checkpoints_preflight(_campaign(tmp_path))
    assert summary["resolved"] == 0
    assert summary["arms"][0]["status"] == "no_resolvable_source"

    with pytest.raises(CampaignCheckpointPreflightError, match="ga3c_cadrl_iros18"):
        check_campaign_arm_checkpoints_preflight(
            _campaign(tmp_path, enforcement="error"), fail_closed_implicit=True
        )


def test_checkpoint_provenance_enforcement_rejects_unknown_value(tmp_path: Path) -> None:
    """The campaign flag is a closed two-value vocabulary."""
    with pytest.raises(ValueError, match="checkpoint_provenance_enforcement"):
        _validate_campaign_config(_campaign(tmp_path, enforcement="sometimes"))


def test_sacadrl_runtime_diagnostics_record_load_and_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Planner initialization exposes the actual load outcome instead of an inferred label."""
    loaded = SACADRLPlannerAdapter()
    monkeypatch.setattr(loaded, "_build_model", object)
    assert loaded._ensure_model() is not None
    assert loaded.diagnostics()["checkpoint_provenance"]["load_succeeded"] is True
    assert loaded.diagnostics()["checkpoint_provenance"]["fallback_triggered"] is False

    fallback = SACADRLPlannerAdapter(allow_fallback=True)

    def fail_load() -> object:
        raise FileNotFoundError("checkpoint missing")

    monkeypatch.setattr(fallback, "_build_model", fail_load)
    assert fallback._ensure_model() is None
    provenance = fallback.diagnostics()["checkpoint_provenance"]
    assert provenance["load_succeeded"] is False
    assert provenance["fallback_triggered"] is True
    assert provenance["load_status"] == "fallback"
    assert "checkpoint missing" in provenance["load_error"]


def test_sacadrl_tensorflow_bundle_hash_covers_all_checkpoint_files(tmp_path: Path) -> None:
    """The runtime digest covers meta, index, and data files, not metadata alone."""
    prefix = tmp_path / "network"
    prefix.with_suffix(".meta").write_bytes(b"meta")
    prefix.with_suffix(".index").write_bytes(b"index")
    (tmp_path / "network.data-00000-of-00001").write_bytes(b"weights")
    adapter = SACADRLPlannerAdapter(
        SocNavPlannerConfig(sacadrl_checkpoint_path=str(prefix.with_suffix(".meta")))
    )
    assert adapter._resolve_checkpoint_prefix() == prefix
    first_hash = adapter.diagnostics()["checkpoint_provenance"]["checkpoint_sha256"]
    assert len(first_hash) == 64

    (tmp_path / "network.data-00000-of-00001").write_bytes(b"changed")
    adapter._resolve_checkpoint_prefix()
    assert adapter.diagnostics()["checkpoint_provenance"]["checkpoint_sha256"] != first_hash


def test_fail_fast_policy_disables_configured_planner_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Strict campaign execution cannot retain a hidden allow_fallback escape hatch."""
    seen: list[bool] = []

    def build_policy(_algo: str, config: dict, **_kwargs):
        seen.append(bool(config.get("allow_fallback")))
        return (lambda _obs: (0.0, 0.0)), {"status": "ok"}

    monkeypatch.setattr(map_runner, "_build_policy", build_policy)
    effective, preflight = map_runner._preflight_policy(
        algo="sacadrl",
        algo_config={"allow_fallback": True},
        benchmark_profile="baseline-safe",
        missing_prereq_policy="fail-fast",
    )
    assert seen == [False]
    assert effective["allow_fallback"] is False
    assert preflight["status"] == "ok"


def test_generic_learned_planner_stats_capture_runtime_fallback() -> None:
    """PPO/SAC/DRL-style metadata is normalized after runtime, not only at construction."""

    class Planner:
        def get_metadata(self) -> dict:
            return {"status": "fallback", "fallback_reason": "prediction_failed"}

    policy = lambda _obs: (0.0, 0.0)  # noqa: E731
    map_runner._attach_checkpoint_runtime_stats(policy, Planner(), {"model_id": "ppo_demo"})
    provenance = policy._planner_stats()["checkpoint_provenance"]
    assert provenance == {
        "model_id": "ppo_demo",
        "checkpoint_sha256": None,
        "hash_source": None,
        "load_succeeded": False,
        "fallback_triggered": True,
        "load_status": "fallback",
        "load_error": "prediction_failed",
    }
    assert _checkpoint_fallback_detected(
        {"algorithm_metadata_contract": {"checkpoint_provenance": provenance}}
    )


def test_batch_summary_bridges_runtime_checkpoint_provenance() -> None:
    """Per-episode planner diagnostics reach the batch contract consumed by campaigns."""
    runtime = {
        "planner_runtime": {
            "checkpoint_provenance": {
                "model_id": "ppo_demo",
                "load_succeeded": True,
                "fallback_triggered": False,
            }
        }
    }
    merged = merge_runtime_algorithm_contract({}, runtime)
    assert merged["checkpoint_provenance"] == {
        "model_id": "ppo_demo",
        "load_succeeded": True,
        "fallback_triggered": False,
    }

    existing = {"checkpoint_provenance": {"model_id": "ppo_demo", "load_succeeded": None}}
    merge_runtime_algorithm_contract(existing, runtime)
    assert existing["checkpoint_provenance"]["load_succeeded"] is True

    unchanged = {"checkpoint_provenance": "malformed"}
    merge_runtime_algorithm_contract(unchanged, runtime)
    assert unchanged["checkpoint_provenance"]["model_id"] == "ppo_demo"


def test_campaign_manifest_folds_runtime_checkpoint_status_per_kinematics() -> None:
    """Completed campaign arms expose runtime load/fallback status in the final manifest."""
    manifest = {
        "planners": [
            {
                "key": "sacadrl",
                "checkpoint_provenance": {
                    "status": "not_run",
                    "model_id": "ga3c_cadrl_iros18",
                    "checkpoint_sha256": "a" * 64,
                    "load_succeeded": None,
                    "fallback_triggered": None,
                    "runtime": [],
                },
            }
        ]
    }
    runs = [
        {
            "planner": {"key": "sacadrl", "kinematics": "differential_drive"},
            "status": "ok",
            "summary": {
                "algorithm_metadata_contract": {
                    "checkpoint_provenance": {
                        "model_id": "ga3c_cadrl_iros18",
                        "checkpoint_sha256": "b" * 64,
                        "load_succeeded": True,
                        "fallback_triggered": False,
                        "load_status": "loaded",
                    }
                }
            },
        }
    ]

    _finalize_checkpoint_provenance(manifest, runs)

    provenance = manifest["planners"][0]["checkpoint_provenance"]
    assert provenance["status"] == "loaded"
    assert provenance["load_succeeded"] is True
    assert provenance["fallback_triggered"] is False
    assert provenance["checkpoint_sha256"] == "b" * 64
    assert provenance["runtime"][0]["kinematics"] == "differential_drive"
    assert runs[0]["planner"]["checkpoint_provenance"]["load_status"] == "loaded"
