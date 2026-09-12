"""Fast unit coverage for camera-ready arm-identity resolvers (issue #9106).

The resolver fallback branches live in ``test_camera_ready_campaign.py``,
which the PR fast-feedback shards deselect as auto-marked slow. These
deterministic unit tests keep the exact-head changed-line coverage for the
arm-identity producers in the fast lane.
"""

from pathlib import Path
from typing import Any

from robot_sf.benchmark.camera_ready._config_types import CampaignConfig, PlannerSpec
from robot_sf.benchmark.camera_ready._reporting import (
    _model_id_from_config_file,
    _model_id_from_summary,
    _resolve_arm_action_adapter,
    _resolve_arm_config_path,
    _resolve_arm_model_id,
    _resolve_arm_policy_source,
)
from robot_sf.benchmark.camera_ready.campaign import _write_campaign_table_artifacts


def _spec(key: str = "goal", algo: str = "goal", **kwargs: Any) -> PlannerSpec:
    return PlannerSpec(key=key, algo=algo, **kwargs)


def test_config_path_empty_without_config() -> None:
    assert _resolve_arm_config_path(_spec(), {}) == ""
    assert (
        _resolve_arm_config_path(
            PlannerSpec(key="ppo", algo="ppo", algo_config_path=Path("configs/x.yaml")),
            {},
        )
        == "configs/x.yaml"
    )


def test_model_id_from_summary_containers() -> None:
    assert _model_id_from_summary({"checkpoint_provenance": {"checkpoint": "ckpt-x"}}) == "ckpt-x"
    assert (
        _model_id_from_summary(
            {"algorithm_metadata_contract": {"checkpoint_provenance": {"model_id": "contract-m"}}}
        )
        == "contract-m"
    )
    assert _model_id_from_summary({"model_id": "top-m"}) == "top-m"
    assert _model_id_from_summary({}) == ""


def test_model_id_from_config_file_missing_unreadable_and_non_mapping(
    tmp_path: Path,
) -> None:
    assert _model_id_from_config_file(None) == ""
    assert _model_id_from_config_file("   ") == ""
    assert _model_id_from_config_file("no/such-config.yaml") == ""
    assert _model_id_from_config_file(tmp_path) == ""
    list_cfg = tmp_path / "list.yaml"
    list_cfg.write_text("- a\n- b\n", encoding="utf-8")
    assert _model_id_from_config_file(list_cfg) == ""
    nul_cfg = tmp_path / "nul.yaml"
    nul_cfg.write_bytes(b"\x00not-yaml")
    assert _model_id_from_config_file(nul_cfg) == ""


def test_model_id_from_config_file_id_keys_and_nested_fallback(
    tmp_path: Path,
) -> None:
    direct = tmp_path / "direct.yaml"
    direct.write_text("model_id: file-m\n", encoding="utf-8")
    assert _model_id_from_config_file(direct) == "file-m"
    nested = tmp_path / "nested.yaml"
    nested.write_text("outer:\n  inner:\n    model_path: deep/nested.ckpt\n", encoding="utf-8")
    assert _model_id_from_config_file(nested) == "deep/nested.ckpt"
    empty = tmp_path / "empty.yaml"
    empty.write_text("unrelated_key: 1\n", encoding="utf-8")
    assert _model_id_from_config_file(empty) == ""


def test_resolve_arm_model_id_summary_config_and_sacadrl_default(
    tmp_path: Path,
) -> None:
    summary_hit = _resolve_arm_model_id(_spec(), {"checkpoint_provenance": {"model_id": "sum-m"}})
    assert summary_hit == "sum-m"
    direct = tmp_path / "direct.yaml"
    direct.write_text("checkpoint_id: cfg-ckpt\n", encoding="utf-8")
    spec = PlannerSpec(key="ppo", algo="ppo", algo_config_path=direct)
    assert _resolve_arm_model_id(spec, {}) == "cfg-ckpt"
    assert _resolve_arm_model_id(_spec(key="sacadrl", algo="sacadrl"), {}) != ""
    assert _resolve_arm_model_id(_spec(), {}) == ""


def test_action_adapter_explicit_kinematics_default_and_native() -> None:
    assert _resolve_arm_action_adapter(_spec(), {"adapter_name": "MyAdapter"}) == "MyAdapter"
    assert _resolve_arm_action_adapter(_spec(key="ppo", algo="ppo"), {}) != ""
    assert _resolve_arm_action_adapter(_spec(), {}) == ""
    assert _resolve_arm_action_adapter(_spec(), {"adapter_name": "none"}) == ""


def test_policy_source_explicit_rule_based_and_trained() -> None:
    assert (
        _resolve_arm_policy_source(_spec(), "m", {"policy_source": "literature-pretrained"})
        == "literature-pretrained"
    )
    assert _resolve_arm_policy_source(_spec(), "") == "rule-based"
    assert (
        _resolve_arm_policy_source(_spec(key="sacadrl", algo="sacadrl"), "m")
        == "literature-pretrained"
    )
    assert _resolve_arm_policy_source(_spec(key="ppo", algo="ppo"), "m") == "trained-here"


def test_write_campaign_table_artifacts_publishes_arm_identity(
    tmp_path: Path,
) -> None:
    cfg = CampaignConfig(
        name="arm_identity_unit",
        scenario_matrix_path=tmp_path / "matrix.yaml",
        planners=(),
    )
    rows = [
        {
            "planner_key": "ppo",
            "planner_group": "core",
            "readiness_tier": "baseline-ready",
            "config_path": "configs/baselines/x.yaml",
            "model_id": "m",
            "action_adapter": "a",
            "policy_source": "trained-here",
        },
        {
            "planner_key": "goal",
            "planner_group": "experimental",
            "readiness_tier": "exploratory",
            "config_path": "",
            "model_id": "",
            "action_adapter": "",
            "policy_source": "rule-based",
        },
    ]
    reports_dir = tmp_path / "reports"
    reports_dir.mkdir()
    paths = _write_campaign_table_artifacts(cfg, reports_dir, rows)
    assert len(paths) == 8
    assert (reports_dir / "arm_identity.csv").is_file()
    assert (reports_dir / "arm_identity.md").is_file()


def test_build_planner_row_metadata_binds_arm_identity() -> None:
    from types import SimpleNamespace

    from robot_sf.benchmark.camera_ready._reporting import _build_planner_row_metadata

    availability = SimpleNamespace(
        availability_status="available",
        benchmark_success=True,
        availability_reason="",
    )
    meta = _build_planner_row_metadata(
        _spec(key="ppo", algo="ppo"),
        "unicycle",
        "ok",
        3,
        {"model_id": "m"},
        "native",
        {"adapter_name": "MyAdapter"},
        "ready",
        availability,
        "ok",
        "ok",
        0,
        0,
    )
    assert meta["planner_key"] == "ppo"
    assert meta["model_id"] == "m"
    assert meta["action_adapter"] == "MyAdapter"
    assert meta["episodes"] == 3
