"""Tests for the predictive-checkpoint schema audit + config filter (issue #5241).

CPU-only, network-free. Synthetic checkpoint-metadata fixtures (one schema-compatible, one not)
prove the audit classifies both correctly and the filtered config drops exactly the incompatible
arm with provenance.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import yaml

from robot_sf.benchmark.camera_ready._config_types import (
    CampaignConfig,
    PlannerSpec,
    SeedPolicy,
)
from robot_sf.benchmark.predictive_checkpoint_schema_audit import (
    STATUS_COMPAT,
    STATUS_INCOMPAT,
    STATUS_NOT_PREDICTIVE,
    audit_predictive_checkpoint_schema,
    emit_schema_filtered_config,
    format_schema_audit_table,
)
from robot_sf.planner.obstacle_features import (
    PREDICTIVE_EGO_FEATURE_DIM,
    PREDICTIVE_EGO_FEATURE_SCHEMA,
    PREDICTIVE_LEGACY_FEATURE_DIM,
    PREDICTIVE_LEGACY_FEATURE_SCHEMA,
    predictive_feature_schema_metadata,
)
from robot_sf.planner.predictive_model import (
    PredictiveModelConfig,
    PredictiveTrajectoryModel,
    save_predictive_checkpoint,
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
        name="schema_audit_test",
        scenario_matrix_path=scenario_path,
        planners=planners,
        seed_policy=SeedPolicy(),
    )


def _save_legacy_checkpoint(tmp_path: Path, name: str) -> Path:
    """Write a schema-compatible (legacy_v1, input_dim=4) predictive checkpoint."""
    model = PredictiveTrajectoryModel(
        PredictiveModelConfig(
            input_dim=PREDICTIVE_LEGACY_FEATURE_DIM,
            feature_schema_name=PREDICTIVE_LEGACY_FEATURE_SCHEMA,
        )
    )
    path = tmp_path / "ckpt" / f"{name}.pt"
    save_predictive_checkpoint(
        path,
        model=model,
        optimizer=None,
        epoch=1,
        feature_schema_metadata=predictive_feature_schema_metadata(
            model_family=PREDICTIVE_LEGACY_FEATURE_SCHEMA
        ),
    )
    return path


def _save_xl_ego_checkpoint(tmp_path: Path, name: str) -> Path:
    """Write an ego-schema (predictive_ego_v1, input_dim=9) predictive checkpoint.

    This mirrors the real ``predictive_proxy_selected_v2_xl_ego`` defect from job 13194: the
    checkpoint declares the ego schema while the runtime planner config expects legacy_v1.
    """
    schema = predictive_feature_schema_metadata(
        model_family=PREDICTIVE_EGO_FEATURE_SCHEMA,
    )
    model = PredictiveTrajectoryModel(
        PredictiveModelConfig(
            input_dim=PREDICTIVE_EGO_FEATURE_DIM,
            feature_schema_name=PREDICTIVE_EGO_FEATURE_SCHEMA,
        )
    )
    path = tmp_path / "ckpt" / f"{name}.pt"
    save_predictive_checkpoint(
        path,
        model=model,
        optimizer=None,
        epoch=1,
        feature_schema_metadata=schema,
    )
    return path


# --- schema classification -------------------------------------------------


def test_audit_classifies_compatible_and_incompatible_arms(tmp_path: Path) -> None:
    """A legacy checkpoint is COMPAT; an ego checkpoint vs a legacy-expecting arm is INCOMPAT."""
    compat_ckpt = _save_legacy_checkpoint(tmp_path, "compat")
    incompat_ckpt = _save_xl_ego_checkpoint(tmp_path, "incompat")
    registry = _write_registry(
        tmp_path,
        [
            {"model_id": "compat_model", "local_path": str(compat_ckpt)},
            {"model_id": "incompat_model", "local_path": str(incompat_ckpt)},
        ],
    )
    compat_cfg = _write_algo_config(
        tmp_path,
        "compat.yaml",
        {"predictive_model_id": "compat_model"},
    )
    incompat_cfg = _write_algo_config(
        tmp_path,
        "incompat.yaml",
        # No explicit predictive_feature_schema_name -> runtime default legacy_v1 (the defect).
        {"predictive_model_id": "incompat_model", "predictive_ego_conditioning": True},
    )

    campaign = _campaign(
        (
            PlannerSpec(
                key="prediction_full", algo="prediction_planner", algo_config_path=compat_cfg
            ),
            PlannerSpec(
                key="prediction_xl_ego", algo="prediction_planner", algo_config_path=incompat_cfg
            ),
        ),
        tmp_path=tmp_path,
    )

    result = audit_predictive_checkpoint_schema(campaign, registry_path=registry)

    by_key = {arm.arm_key: arm for arm in result.arms}
    assert by_key["prediction_full"].status == STATUS_COMPAT
    assert by_key["prediction_full"].checkpoint_schema == PREDICTIVE_LEGACY_FEATURE_SCHEMA
    assert by_key["prediction_xl_ego"].status == STATUS_INCOMPAT
    assert by_key["prediction_xl_ego"].checkpoint_schema == "predictive_ego_v1"
    assert by_key["prediction_xl_ego"].expected_schema == PREDICTIVE_LEGACY_FEATURE_SCHEMA
    assert [arm.arm_key for arm in result.incompatible_arms] == ["prediction_xl_ego"]


def test_audit_explicit_expected_schema_can_match_ego_checkpoint(tmp_path: Path) -> None:
    """An ego checkpoint is COMPAT when the algo config declares the ego expected schema."""
    ego_ckpt = _save_xl_ego_checkpoint(tmp_path, "ego_ok")
    registry = _write_registry(
        tmp_path,
        [{"model_id": "ego_model", "local_path": str(ego_ckpt)}],
    )
    algo_cfg = _write_algo_config(
        tmp_path,
        "ego.yaml",
        {
            "predictive_model_id": "ego_model",
            "predictive_feature_schema_name": PREDICTIVE_EGO_FEATURE_SCHEMA,
        },
    )
    campaign = _campaign(
        (PlannerSpec(key="ego_arm", algo="prediction_planner", algo_config_path=algo_cfg),),
        tmp_path=tmp_path,
    )

    result = audit_predictive_checkpoint_schema(campaign, registry_path=registry)
    arm = result.arms[0]
    assert arm.status == STATUS_COMPAT
    assert arm.expected_schema == PREDICTIVE_EGO_FEATURE_SCHEMA


def test_audit_skips_non_predictive_arms(tmp_path: Path) -> None:
    """A non-predictive arm (no checkpoint) is reported NOT_PREDICTIVE, not audited."""
    algo_cfg = _write_algo_config(
        tmp_path,
        "mppi.yaml",
        {"algo": "mppi_social", "horizon_steps": 10},  # no predictive checkpoint
    )
    campaign = _campaign(
        (PlannerSpec(key="mppi_social", algo="mppi_social", algo_config_path=algo_cfg),),
        tmp_path=tmp_path,
    )

    result = audit_predictive_checkpoint_schema(campaign)
    assert result.arms[0].status == STATUS_NOT_PREDICTIVE
    assert result.incompatible_arms == []
    assert result.actionable_arms == []


# --- filtered config emission ----------------------------------------------


def test_filtered_config_drops_exactly_the_incompatible_arm(tmp_path: Path) -> None:
    """The filtered config removes only the INCOMPAT arm and records provenance."""
    compat_ckpt = _save_legacy_checkpoint(tmp_path, "compat2")
    incompat_ckpt = _save_xl_ego_checkpoint(tmp_path, "incompat2")
    registry = _write_registry(
        tmp_path,
        [
            {"model_id": "compat_model", "local_path": str(compat_ckpt)},
            {"model_id": "incompat_model", "local_path": str(incompat_ckpt)},
        ],
    )
    compat_cfg = _write_algo_config(
        tmp_path,
        "compat2.yaml",
        {"predictive_model_id": "compat_model"},
    )
    incompat_cfg = _write_algo_config(
        tmp_path,
        "incompat2.yaml",
        {"predictive_model_id": "incompat_model", "predictive_ego_conditioning": True},
    )

    # Source campaign config YAML (mirrors the real gap-prediction compare layout).
    scenario_matrix = tmp_path / "scenarios.yaml"
    scenario_matrix.write_text("scenarios: []\n", encoding="utf-8")
    source_config = tmp_path / "campaign.yaml"
    source_config.write_text(
        yaml.safe_dump(
            {
                "name": "schema_filter_test",
                "scenario_matrix": str(scenario_matrix),
                "planners": [
                    {
                        "key": "prediction_full",
                        "algo": "prediction_planner",
                        "algo_config": str(compat_cfg),
                    },
                    {
                        "key": "prediction_xl_ego",
                        "algo": "prediction_planner",
                        "algo_config": str(incompat_cfg),
                    },
                    {"key": "orca", "algo": "orca"},
                ],
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    from robot_sf.benchmark.camera_ready_campaign import load_campaign_config

    cfg = load_campaign_config(source_config)
    result = audit_predictive_checkpoint_schema(cfg, registry_path=registry)
    assert [arm.arm_key for arm in result.incompatible_arms] == ["prediction_xl_ego"]

    out_path = tmp_path / "campaign_schema_filtered.yaml"
    written = emit_schema_filtered_config(source_config, result, out_path)
    assert written == out_path.resolve()

    filtered = yaml.safe_load(out_path.read_text(encoding="utf-8"))
    remaining_keys = [p["key"] for p in filtered["planners"]]
    assert remaining_keys == ["prediction_full", "orca"]  # exactly the incompatible arm dropped

    excluded = filtered["schema_excluded_arms"]
    assert len(excluded) == 1
    assert excluded[0]["key"] == "prediction_xl_ego"
    assert excluded[0]["checkpoint_ref"] == "incompat_model"
    assert excluded[0]["expected_schema"] == PREDICTIVE_LEGACY_FEATURE_SCHEMA
    assert excluded[0]["checkpoint_schema"] == "predictive_ego_v1"
    assert filtered["schema_audit"]["excluded_arm_count"] == 1
    assert "issue #5241" in filtered["schema_audit"]["note"]


def test_filtered_config_leaves_uninspected_arms_in_place(tmp_path: Path) -> None:
    """An unstaged arm is NOT dropped -- it is recorded as uninspected for a human to decide."""
    compat_ckpt = _save_legacy_checkpoint(tmp_path, "compat3")
    # Registry with a local-only, absent path -> resolves as UNSTAGED (no --stage).
    registry = _write_registry(
        tmp_path,
        [
            {"model_id": "compat_model", "local_path": str(compat_ckpt)},
            {
                "model_id": "absent_model",
                "local_path": str(tmp_path / "does_not_exist.pt"),
                "local_only": True,
            },
        ],
    )
    compat_cfg = _write_algo_config(
        tmp_path,
        "compat3.yaml",
        {"predictive_model_id": "compat_model"},
    )
    absent_cfg = _write_algo_config(
        tmp_path,
        "absent.yaml",
        {"predictive_model_id": "absent_model"},
    )

    scenario_matrix = tmp_path / "scenarios2.yaml"
    scenario_matrix.write_text("scenarios: []\n", encoding="utf-8")
    source_config = tmp_path / "campaign2.yaml"
    source_config.write_text(
        yaml.safe_dump(
            {
                "name": "schema_unstaged_test",
                "scenario_matrix": str(scenario_matrix),
                "planners": [
                    {"key": "ok", "algo": "prediction_planner", "algo_config": str(compat_cfg)},
                    {
                        "key": "missing",
                        "algo": "prediction_planner",
                        "algo_config": str(absent_cfg),
                    },
                ],
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    from robot_sf.benchmark.camera_ready_campaign import load_campaign_config

    cfg = load_campaign_config(source_config)
    result = audit_predictive_checkpoint_schema(cfg, registry_path=registry)
    by_key = {arm.arm_key: arm for arm in result.arms}
    assert by_key["ok"].status == STATUS_COMPAT
    assert by_key["missing"].status == "UNSTAGED"
    assert result.incompatible_arms == []

    out_path = tmp_path / "campaign2_schema_filtered.yaml"
    emit_schema_filtered_config(source_config, result, out_path)
    filtered = yaml.safe_load(out_path.read_text(encoding="utf-8"))
    # The unstaged arm is kept (the tool never silently drops an arm it could not inspect).
    assert [p["key"] for p in filtered["planners"]] == ["ok", "missing"]
    assert filtered["schema_excluded_arms"] == []
    assert [a["key"] for a in filtered["schema_audit_uninspected_arms"]] == ["missing"]


def test_format_table_lists_all_arms_and_marks_incompat(tmp_path: Path) -> None:
    """The human-readable table lists every arm and flags INCOMPAT rows in the summary."""
    compat_ckpt = _save_legacy_checkpoint(tmp_path, "compat4")
    incompat_ckpt = _save_xl_ego_checkpoint(tmp_path, "incompat4")
    registry = _write_registry(
        tmp_path,
        [
            {"model_id": "compat_model", "local_path": str(compat_ckpt)},
            {"model_id": "incompat_model", "local_path": str(incompat_ckpt)},
        ],
    )
    compat_cfg = _write_algo_config(
        tmp_path, "compat4.yaml", {"predictive_model_id": "compat_model"}
    )
    incompat_cfg = _write_algo_config(
        tmp_path,
        "incompat4.yaml",
        {"predictive_model_id": "incompat_model"},
    )
    campaign = _campaign(
        (
            PlannerSpec(key="good", algo="prediction_planner", algo_config_path=compat_cfg),
            PlannerSpec(key="bad", algo="prediction_planner", algo_config_path=incompat_cfg),
        ),
        tmp_path=tmp_path,
    )
    result = audit_predictive_checkpoint_schema(campaign, registry_path=registry)

    table = format_schema_audit_table(result)
    assert "arm" in table and "checkpoint" in table
    assert "expected-schema" in table and "current-schema" in table and "status" in table
    assert "good" in table and "bad" in table
    assert "1 INCOMPAT" in table
    assert "INCOMPAT bad" in table
