"""Unit tests for predictive training pipeline helper utilities."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import yaml

from robot_sf.planner.obstacle_features import (
    PREDICTIVE_EGO_FEATURE_SCHEMA,
    PREDICTIVE_EGO_MOTION_PRODUCER_RUNTIME,
    PREDICTIVE_EGO_MOTION_PRODUCER_STANDALONE,
    PREDICTIVE_OBSTACLE_FEATURE_DIM,
    PREDICTIVE_OBSTACLE_FEATURE_SCHEMA,
    predictive_feature_schema_metadata,
)
from scripts.training import run_predictive_training_pipeline as pipeline


def _predictive_feature_schema_json(
    *,
    model_family: str = "predictive_legacy_v1",
    ego_conditioning: bool = False,
    producer: str | None = None,
) -> str:
    """Return predictive feature-schema metadata for pipeline NPZ fixtures."""
    return json.dumps(
        predictive_feature_schema_metadata(
            model_family=model_family,
            ego_conditioning=ego_conditioning,
            ego_motion_channel_producer=producer,
        ),
        sort_keys=True,
    )


def _obstacle_feature_schema_json(base_dim: int = 4) -> str:
    """Return obstacle-feature schema metadata for predictive NPZ fixtures."""
    return json.dumps(
        {
            "name": PREDICTIVE_OBSTACLE_FEATURE_SCHEMA,
            "base_schema": "predictive_legacy_v1",
            "base_feature_dim": base_dim,
            "obstacle_feature_schema": {
                "name": PREDICTIVE_OBSTACLE_FEATURE_SCHEMA,
                "feature_dim": PREDICTIVE_OBSTACLE_FEATURE_DIM,
            },
            "input_dim": base_dim + PREDICTIVE_OBSTACLE_FEATURE_DIM,
        }
    )


def _write_seed_manifest(output_path: Path) -> Path:
    """Write a minimal seed manifest used by pipeline smoke tests."""
    output_path.write_text("scenario: [1]\n", encoding="utf-8")
    return output_path


def _write_predictive_dataset_fixture(
    path: Path,
    *,
    state_dim: int,
    feature_schema_json: str | None = None,
    summary: dict[str, object] | None = None,
) -> None:
    """Write a predictive dataset fixture with optional embedded schema metadata."""
    payload: dict[str, object] = {
        "state": np.zeros((2, 3, state_dim), dtype=np.float32),
        "target": np.zeros((2, 3, 5, 2), dtype=np.float32),
        "mask": np.ones((2, 3), dtype=np.float32),
        "target_mask": np.ones((2, 3, 5), dtype=np.float32),
    }
    if feature_schema_json is not None:
        payload["feature_schema_json"] = np.asarray(feature_schema_json)
    np.savez(path, **payload)
    path.with_suffix(".json").write_text(
        json.dumps(summary or {"num_samples": 2}),
        encoding="utf-8",
    )


def _pipeline_test_output_root(tmp_path: Path) -> str:
    """Return a per-test predictive pipeline output root safe for xdist workers."""
    return str(tmp_path / "predictive_pipeline_output")


def _make_ego_pipeline_run_stub(invoked: list[list[str]]):
    """Return a pipeline stage stub that materializes ego-conditioned dataset artifacts."""
    ego_schema_json = _predictive_feature_schema_json(
        model_family=PREDICTIVE_EGO_FEATURE_SCHEMA,
        ego_conditioning=True,
        producer=PREDICTIVE_EGO_MOTION_PRODUCER_RUNTIME,
    )

    def _fake_run(cmd: list[str], *, log_level: str) -> None:
        del log_level
        invoked.append(list(cmd))
        if any("collect_predictive_hardcase_data.py" in part for part in cmd):
            output = Path(cmd[cmd.index("--output") + 1])
            output.parent.mkdir(parents=True, exist_ok=True)
            _write_predictive_dataset_fixture(
                output,
                state_dim=9 if "--ego-conditioning" in cmd else 4,
                feature_schema_json=ego_schema_json if "--ego-conditioning" in cmd else None,
            )
            return
        if any("build_predictive_mixed_dataset.py" in part for part in cmd):
            output = Path(cmd[cmd.index("--output") + 1])
            output.parent.mkdir(parents=True, exist_ok=True)
            _write_predictive_dataset_fixture(
                output,
                state_dim=9,
                feature_schema_json=ego_schema_json,
            )
            return
        if any("train_predictive_planner.py" in part for part in cmd):
            out_dir = Path(cmd[cmd.index("--output-dir") + 1])
            out_dir.mkdir(parents=True, exist_ok=True)
            (out_dir / "training_summary.json").write_text(
                json.dumps(
                    {
                        "best_checkpoint": str(out_dir / "predictive_model.pt"),
                        "selection": {"selected_epoch": 1},
                        "selected_checkpoint_reason": "proxy",
                        "source_dataset_ids": ["a", "b"],
                    }
                ),
                encoding="utf-8",
            )
            (out_dir / "predictive_model.pt").write_text("stub", encoding="utf-8")
            return
        if any("evaluate_predictive_planner.py" in part for part in cmd):
            output_dir = Path(cmd[cmd.index("--output-dir") + 1])
            output_dir.mkdir(parents=True, exist_ok=True)
            (output_dir / "final_eval_summary.json").write_text(
                json.dumps({"quality_gates": {"pass_all": True}, "integrity": {"pass": True}}),
                encoding="utf-8",
            )
            return
        if any("run_predictive_hard_seed_diagnostics.py" in part for part in cmd):
            output_dir = Path(cmd[cmd.index("--output-dir") + 1])
            output_dir.mkdir(parents=True, exist_ok=True)
            (output_dir / "summary.json").write_text(json.dumps({"ok": True}), encoding="utf-8")
            return
        if any("run_predictive_success_campaign.py" in part for part in cmd):
            output_dir = Path(cmd[cmd.index("--output-dir") + 1])
            output_dir.mkdir(parents=True, exist_ok=True)
            (output_dir / "campaign_summary.json").write_text(
                json.dumps({"integrity": {"pass": True}}),
                encoding="utf-8",
            )
            return
        raise AssertionError(f"Unexpected command: {cmd}")

    return _fake_run


def test_paths_from_config_resolves_root_relative_to_output_base(tmp_path: Path) -> None:
    """Resolve output root relative to output base directory and run id."""
    cfg = {"output": {"root": "output/tmp/predictive_planner/pipeline"}}
    run_id = "predictive_test_run"

    paths = pipeline._paths_from_config(
        cfg,
        run_id=run_id,
        base_dir=tmp_path,
        output_base_dir=tmp_path,
    )
    assert paths.root == (tmp_path / "output/tmp/predictive_planner/pipeline" / run_id).resolve()
    assert paths.base_dataset.name == "predictive_rollouts_base.npz"
    assert paths.checkpoint.name == "predictive_model.pt"
    assert paths.final_summary.name == "final_performance_summary.json"


def test_build_random_seed_manifest_generates_all_scenarios(monkeypatch, tmp_path: Path) -> None:
    """Generate deterministic per-scenario random seed lists for all scenarios."""
    scenario_matrix = tmp_path / "scenarios.yaml"
    scenario_matrix.write_text("[]\n", encoding="utf-8")

    monkeypatch.setattr(
        pipeline,
        "load_scenarios",
        lambda _path: [
            {"name": "classic_cross_trap_low"},
            {"name": "classic_doorway_high"},
        ],
    )

    out_path = tmp_path / "manifest.yaml"
    manifest_path = pipeline._build_random_seed_manifest(
        scenario_matrix=scenario_matrix,
        seeds_per_scenario=3,
        random_seed_base=100,
        output_path=out_path,
    )

    assert manifest_path == out_path
    payload = yaml.safe_load(out_path.read_text(encoding="utf-8"))
    assert payload["classic_cross_trap_low"] == [100, 101, 102]
    assert payload["classic_doorway_high"] == [100100, 100101, 100102]


def test_run_capture_json_anchors_subprocess_to_repo_root(monkeypatch) -> None:
    """Subprocess helpers should execute repo-relative commands from the repository root."""
    called: dict[str, object] = {}

    def _fake_run(cmd, **kwargs):
        """Capture subprocess invocation details and return empty JSON output."""
        called["cmd"] = cmd
        called["cwd"] = kwargs.get("cwd")
        called["env"] = kwargs.get("env")
        return type("Result", (), {"returncode": 0, "stdout": "{}", "stderr": ""})()

    monkeypatch.setattr(pipeline.subprocess, "run", _fake_run)
    payload = pipeline._run_capture_json(["python", "scripts/example.py"])
    assert payload["status"] == "ok"
    assert called["cwd"] == pipeline._REPO_ROOT


def test_write_dataset_manifest_records_contract_and_hashes(tmp_path: Path) -> None:
    """Dataset manifests should capture reset-v2 provenance and dataset digest."""
    dataset_path = tmp_path / "predictive_rollouts_base.npz"
    np.savez(
        dataset_path,
        state=np.zeros((2, 3, 4), dtype=np.float32),
        target=np.zeros((2, 3, 5, 2), dtype=np.float32),
        mask=np.ones((2, 3), dtype=np.float32),
        target_mask=np.ones((2, 3, 5), dtype=np.float32),
    )
    summary_path = dataset_path.with_suffix(".json")
    summary_path.write_text(json.dumps({"num_samples": 2}), encoding="utf-8")

    manifest_path = pipeline._write_dataset_manifest(
        dataset_path=dataset_path,
        summary_path=summary_path,
        role="predictive_base_dataset",
        run_id="run_123",
        config_path=tmp_path / "config.yaml",
        config_hash="abc123",
        git_commit="deadbeef",
        extra={"seed_manifest": "manifest.yaml"},
    )

    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert payload["contract_version"] == "benchmark-reset-v2"
    assert payload["training_family"] == "prediction_planner"
    assert payload["dataset_sha1"]
    assert payload["diagnostics"]["num_samples"] == 2


def test_dataset_npz_diagnostics_counts_empty_rows(tmp_path: Path) -> None:
    """NPZ diagnostics should surface empty agent and target rows."""
    dataset_path = tmp_path / "predictive_rollouts_mixed.npz"
    mask = np.array([[1.0, 0.0], [0.0, 0.0]], dtype=np.float32)
    target_mask = np.zeros((2, 2, 3), dtype=np.float32)
    target_mask[0, 0, :] = 1.0
    np.savez(
        dataset_path,
        state=np.zeros((2, 2, 4), dtype=np.float32),
        target=np.zeros((2, 2, 3, 2), dtype=np.float32),
        mask=mask,
        target_mask=target_mask,
    )

    payload = pipeline._dataset_npz_diagnostics(dataset_path)
    assert payload["empty_agent_rows"] == 1
    assert payload["empty_target_rows"] == 1


def test_pipeline_collection_commands_pass_ego_conditioning(monkeypatch, tmp_path: Path) -> None:
    """Pipeline should pass ego-conditioning to both base and hardcase collectors when enabled."""
    config_path = tmp_path / "predictive.yaml"
    output_root = _pipeline_test_output_root(tmp_path)
    config_path.write_text(
        yaml.safe_dump(
            {
                "experiment": {"run_id": "predictive_ego_smoke"},
                "output": {"root": output_root},
                "scenarios": {
                    "scenario_matrix": "scenarios.yaml",
                    "hard_seed_manifest": "hard.yaml",
                    "planner_grid": "grid.yaml",
                },
                "base_collection": {
                    "seeds_per_scenario": 1,
                    "random_seed_base": 7,
                    "ego_conditioning": True,
                },
                "hardcase_collection": {
                    "ego_conditioning": True,
                },
                "mixing": {},
                "training": {},
                "wandb": {"enabled": False},
                "evaluation": {},
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    (tmp_path / "scenarios.yaml").write_text("[]\n", encoding="utf-8")
    (tmp_path / "hard.yaml").write_text("{}\n", encoding="utf-8")
    (tmp_path / "grid.yaml").write_text("{}\n", encoding="utf-8")

    monkeypatch.setattr(pipeline, "_sha1_file", lambda _path: "cfghash")
    monkeypatch.setattr(pipeline, "_git_hash", lambda: "deadbeef")

    invoked: list[list[str]] = []
    monkeypatch.setattr(
        pipeline,
        "_build_random_seed_manifest",
        lambda **kwargs: _write_seed_manifest(Path(kwargs["output_path"])),
    )
    monkeypatch.setattr(pipeline, "_run", _make_ego_pipeline_run_stub(invoked))
    monkeypatch.setattr(
        pipeline,
        "_run_capture_json",
        lambda _cmd, **_kwargs: {"status": "ok", "return_code": 0},
    )
    monkeypatch.setattr(
        pipeline,
        "_parse_args",
        lambda: pipeline.argparse.Namespace(
            config=config_path,
            run_id="predictive_promotion_smoke",
            log_level="INFO",
        ),
    )

    resolved_paths = pipeline._paths_from_config(
        yaml.safe_load(config_path.read_text(encoding="utf-8")),
        run_id="predictive_promotion_smoke",
        base_dir=config_path.parent,
        output_base_dir=Path.cwd().resolve(),
    )
    assert resolved_paths.root.is_relative_to(tmp_path)
    code = pipeline.main()
    assert code == 0
    collector_cmds = [
        cmd for cmd in invoked if any("collect_predictive_hardcase_data.py" in part for part in cmd)
    ]
    assert len(collector_cmds) == 2
    assert all("--ego-conditioning" in cmd for cmd in collector_cmds)
    final_summary = json.loads(resolved_paths.final_summary.read_text(encoding="utf-8"))
    assert final_summary["producer_metadata_preflight"]["status"] == "ok"
    mixed_manifest = json.loads(
        Path(final_summary["dataset_manifests"]["mixed"]).read_text(encoding="utf-8")
    )
    assert mixed_manifest["extra"]["producer_metadata_preflight_status"] == "ok"


def test_pipeline_passes_resolved_weighting_spec_to_mixed_builder(
    monkeypatch,
    tmp_path: Path,
) -> None:
    """Pipeline mixing config should pass a resolved hard-case weighting spec path."""
    config_path = tmp_path / "predictive_weighting.yaml"
    weighting_spec = tmp_path / "configs" / "crossing_conflict_weighting.yaml"
    weighting_spec.parent.mkdir(parents=True)
    weighting_spec.write_text(
        yaml.safe_dump(
            {
                "profile_id": "crossing_conflict_hardcase_repeat_test",
                "weighting": {
                    "rule": "repeat_hardcase_rows",
                    "hardcase_family": "crossing_conflict",
                    "hardcase_repeat": 3,
                    "shuffle_seed": 3214,
                },
            }
        ),
        encoding="utf-8",
    )
    output_root = _pipeline_test_output_root(tmp_path)
    config_path.write_text(
        yaml.safe_dump(
            {
                "experiment": {"run_id": "predictive_weighting_smoke"},
                "output": {"root": output_root},
                "scenarios": {
                    "scenario_matrix": "scenarios.yaml",
                    "hard_seed_manifest": "hard.yaml",
                    "planner_grid": "grid.yaml",
                },
                "base_collection": {"seeds_per_scenario": 1, "ego_conditioning": True},
                "hardcase_collection": {"ego_conditioning": True},
                "mixing": {"weighting_spec": "configs/crossing_conflict_weighting.yaml"},
                "training": {},
                "wandb": {"enabled": False},
                "evaluation": {},
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    (tmp_path / "scenarios.yaml").write_text("[]\n", encoding="utf-8")
    (tmp_path / "hard.yaml").write_text("{}\n", encoding="utf-8")
    (tmp_path / "grid.yaml").write_text("{}\n", encoding="utf-8")

    monkeypatch.setattr(pipeline, "_sha1_file", lambda _path: "cfghash")
    monkeypatch.setattr(pipeline, "_git_hash", lambda: "deadbeef")
    monkeypatch.setattr(
        pipeline,
        "_build_random_seed_manifest",
        lambda **kwargs: _write_seed_manifest(Path(kwargs["output_path"])),
    )
    invoked: list[list[str]] = []
    monkeypatch.setattr(pipeline, "_run", _make_ego_pipeline_run_stub(invoked))
    monkeypatch.setattr(
        pipeline,
        "_run_capture_json",
        lambda _cmd, **_kwargs: {"status": "ok", "return_code": 0},
    )
    monkeypatch.setattr(
        pipeline,
        "_parse_args",
        lambda: pipeline.argparse.Namespace(
            config=config_path,
            run_id="predictive_weighting_smoke",
            log_level="INFO",
        ),
    )

    assert pipeline.main() == 0
    mixed_builder_cmd = next(
        cmd for cmd in invoked if any("build_predictive_mixed_dataset.py" in part for part in cmd)
    )
    assert mixed_builder_cmd[mixed_builder_cmd.index("--weighting-spec") + 1] == str(
        weighting_spec.resolve()
    )
    assert "--hardcase-repeat" not in mixed_builder_cmd
    assert "--shuffle-seed" not in mixed_builder_cmd


def test_issue_3254_config_targets_crossing_conflict_weighting_spec() -> None:
    """The real #3254 config should use the #3214 spec without CLI-style overrides."""
    repo = Path(__file__).resolve().parents[2]
    config_path = (
        repo
        / "configs"
        / "training"
        / "predictive"
        / "predictive_crossing_conflict_weighted_issue_3254.yaml"
    )

    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    mixing = cfg["mixing"]
    weighting_spec = config_path.parent / mixing["weighting_spec"]

    assert weighting_spec.name == "predictive_crossing_conflict_hardcase_mixing_issue_3214.yaml"
    assert weighting_spec.is_file()
    assert "hardcase_repeat" not in mixing
    assert "shuffle_seed" not in mixing
    assert cfg["base_collection"]["ego_conditioning"] is True
    assert cfg["hardcase_collection"]["ego_conditioning"] is True
    assert cfg["model_family"] == PREDICTIVE_EGO_FEATURE_SCHEMA
    assert cfg["training"]["model_id"] == "predictive_crossing_conflict_weighted_issue_3254_xl_ego"
    assert cfg["training"]["model_family"] == PREDICTIVE_EGO_FEATURE_SCHEMA
    assert "prepare-only" in cfg["claim_boundary"].lower()


def test_pipeline_passes_obstacle_model_family_to_collectors_and_training(
    monkeypatch,
    tmp_path: Path,
) -> None:
    """Obstacle-feature pipeline configs should thread model family through all stages."""
    config_path = tmp_path / "predictive_obstacle.yaml"
    output_root = _pipeline_test_output_root(tmp_path)
    config_path.write_text(
        yaml.safe_dump(
            {
                "experiment": {"run_id": "predictive_obstacle_smoke"},
                "model_family": PREDICTIVE_OBSTACLE_FEATURE_SCHEMA,
                "output": {"root": output_root},
                "scenarios": {
                    "scenario_matrix": "scenarios.yaml",
                    "hard_seed_manifest": "hard.yaml",
                    "planner_grid": "grid.yaml",
                },
                "base_collection": {"seeds_per_scenario": 1},
                "hardcase_collection": {},
                "mixing": {},
                "training": {"model_id": "predictive_obstacle_smoke"},
                "wandb": {"enabled": False},
                "evaluation": {},
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    (tmp_path / "scenarios.yaml").write_text("[]\n", encoding="utf-8")
    (tmp_path / "hard.yaml").write_text("{}\n", encoding="utf-8")
    (tmp_path / "grid.yaml").write_text("{}\n", encoding="utf-8")

    monkeypatch.setattr(pipeline, "_sha1_file", lambda _path: "cfghash")
    monkeypatch.setattr(pipeline, "_git_hash", lambda: "deadbeef")
    monkeypatch.setattr(
        pipeline,
        "_build_random_seed_manifest",
        lambda **kwargs: Path(kwargs["output_path"]),
    )

    invoked: list[list[str]] = []

    def _fake_dataset(path: Path) -> None:
        """Write an obstacle-schema predictive dataset fixture."""
        state = np.zeros((2, 3, 10), dtype=np.float32)
        state[:, :, -1] = 1.0
        np.savez(
            path,
            state=state,
            target=np.zeros((2, 3, 5, 2), dtype=np.float32),
            mask=np.ones((2, 3), dtype=np.float32),
            target_mask=np.ones((2, 3, 5), dtype=np.float32),
            feature_schema_json=np.asarray(_obstacle_feature_schema_json()),
        )
        path.with_suffix(".json").write_text(
            json.dumps(
                {
                    "num_samples": 2,
                    "obstacle_feature_source": "map_geometry",
                    "obstacle_line_count": 3,
                }
            ),
            encoding="utf-8",
        )

    def _fake_run(cmd: list[str], *, log_level: str) -> None:
        """Simulate pipeline commands and record schema-routing flags."""
        invoked.append(list(cmd))
        if any("collect_predictive_hardcase_data.py" in part for part in cmd):
            output = Path(cmd[cmd.index("--output") + 1])
            output.parent.mkdir(parents=True, exist_ok=True)
            _fake_dataset(output)
            return
        if any("build_predictive_mixed_dataset.py" in part for part in cmd):
            output = Path(cmd[cmd.index("--output") + 1])
            output.parent.mkdir(parents=True, exist_ok=True)
            _fake_dataset(output)
            return
        if any("train_predictive_planner.py" in part for part in cmd):
            out_dir = Path(cmd[cmd.index("--output-dir") + 1])
            out_dir.mkdir(parents=True, exist_ok=True)
            (out_dir / "training_summary.json").write_text(
                json.dumps(
                    {"selection": {"selected_epoch": 1}, "selected_checkpoint_reason": "proxy"}
                ),
                encoding="utf-8",
            )
            (out_dir / "predictive_model.pt").write_text("stub", encoding="utf-8")
            return
        if any("run_predictive_success_campaign.py" in part for part in cmd):
            output_dir = Path(cmd[cmd.index("--output-dir") + 1])
            output_dir.mkdir(parents=True, exist_ok=True)
            (output_dir / "campaign_summary.json").write_text(
                json.dumps({"run_id": "predictive_obstacle_smoke", "status": "success"}),
                encoding="utf-8",
            )
            return
        raise AssertionError(f"Unexpected command: {cmd}")

    monkeypatch.setattr(pipeline, "_run", _fake_run)
    monkeypatch.setattr(
        pipeline,
        "_run_capture_json",
        lambda _cmd, **_kwargs: {"status": "ok", "return_code": 0},
    )
    monkeypatch.setattr(
        pipeline,
        "_parse_args",
        lambda: pipeline.argparse.Namespace(
            config=config_path,
            run_id="predictive_obstacle_smoke",
            log_level="INFO",
        ),
    )

    assert pipeline.main() == 0
    collector_cmds = [
        cmd for cmd in invoked if any("collect_predictive_hardcase_data.py" in part for part in cmd)
    ]
    train_cmd = next(
        cmd for cmd in invoked if any("train_predictive_planner.py" in part for part in cmd)
    )
    assert len(collector_cmds) == 2
    assert all(
        cmd[cmd.index("--model-family") + 1] == PREDICTIVE_OBSTACLE_FEATURE_SCHEMA
        for cmd in collector_cmds
    )
    assert train_cmd[train_cmd.index("--model-family") + 1] == PREDICTIVE_OBSTACLE_FEATURE_SCHEMA


def test_obstacle_feature_preflight_reports_active_non_sentinel_rows(tmp_path: Path) -> None:
    """Obstacle preflight should pass when active rows contain map-derived features."""
    dataset_path = tmp_path / "predictive_rollouts_base.npz"
    state = np.zeros((2, 2, 10), dtype=np.float32)
    state[0, 0, 4 : 4 + PREDICTIVE_OBSTACLE_FEATURE_DIM] = np.asarray(
        [2.5, 1.0, 0.0, 0.0, 1.0, 1.0],
        dtype=np.float32,
    )
    np.savez(
        dataset_path,
        state=state,
        target=np.zeros((2, 2, 5, 2), dtype=np.float32),
        mask=np.ones((2, 2), dtype=np.float32),
        target_mask=np.ones((2, 2, 5), dtype=np.float32),
        feature_schema_json=np.asarray(_obstacle_feature_schema_json()),
    )
    dataset_path.with_suffix(".json").write_text(
        json.dumps({"obstacle_feature_source": "map_geometry", "obstacle_line_count": 2}),
        encoding="utf-8",
    )

    report = pipeline._obstacle_feature_preflight(dataset_path)

    assert report["status"] == "ok"
    assert report["active_agent_rows"] == 4
    assert report["active_valid_obstacle_rows"] == 1
    assert report["active_invalid_obstacle_rows"] == 3


def test_obstacle_feature_preflight_fails_sentinel_only_rows(tmp_path: Path) -> None:
    """Obstacle preflight should fail before training on sentinel-only obstacle rows."""
    dataset_path = tmp_path / "predictive_rollouts_base.npz"
    state = np.zeros((1, 3, 10), dtype=np.float32)
    state[:, :, 4 : 4 + PREDICTIVE_OBSTACLE_FEATURE_DIM] = np.asarray(
        [50.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        dtype=np.float32,
    )
    np.savez(
        dataset_path,
        state=state,
        target=np.zeros((1, 3, 5, 2), dtype=np.float32),
        mask=np.ones((1, 3), dtype=np.float32),
        target_mask=np.ones((1, 3, 5), dtype=np.float32),
        feature_schema_json=np.asarray(_obstacle_feature_schema_json()),
    )
    dataset_path.with_suffix(".json").write_text(
        json.dumps({"obstacle_feature_source": "map_geometry", "obstacle_line_count": 0}),
        encoding="utf-8",
    )

    report = pipeline._obstacle_feature_preflight(dataset_path)

    assert report["status"] == "failed"
    assert report["active_valid_obstacle_rows"] == 0
    assert report["active_exact_sentinel_rows"] == 3
    assert report["failure_reason"] == "Dataset contains no active non-sentinel obstacle rows."


def test_obstacle_feature_preflight_rejects_non_map_source_with_valid_rows(tmp_path: Path) -> None:
    """Obstacle preflight should fail closed when the source is not explicitly map-derived."""
    dataset_path = tmp_path / "predictive_rollouts_base.npz"
    state = np.zeros((1, 2, 10), dtype=np.float32)
    state[0, 0, 4 : 4 + PREDICTIVE_OBSTACLE_FEATURE_DIM] = np.asarray(
        [2.5, 1.0, 0.0, 0.0, 1.0, 1.0],
        dtype=np.float32,
    )
    np.savez(
        dataset_path,
        state=state,
        target=np.zeros((1, 2, 5, 2), dtype=np.float32),
        mask=np.ones((1, 2), dtype=np.float32),
        target_mask=np.ones((1, 2, 5), dtype=np.float32),
        feature_schema_json=np.asarray(_obstacle_feature_schema_json()),
    )
    dataset_path.with_suffix(".json").write_text(
        json.dumps({"obstacle_feature_source": "not_available", "obstacle_line_count": 2}),
        encoding="utf-8",
    )

    report = pipeline._obstacle_feature_preflight(dataset_path)

    assert report["status"] == "failed"
    assert report["obstacle_feature_source"] == "not_available"
    assert report["failure_reason"] == (
        "Dataset obstacle_feature_source must be an explicit map-derived source; got "
        "'not_available'."
    )


def test_producer_metadata_preflight_rejects_mismatched_ego_producers(tmp_path: Path) -> None:
    """Producer preflight must fail before mixing incompatible ego-conditioned datasets."""
    base_path = tmp_path / "predictive_rollouts_base.npz"
    hardcase_path = tmp_path / "predictive_rollouts_hardcase.npz"
    for path, producer in [
        (base_path, PREDICTIVE_EGO_MOTION_PRODUCER_RUNTIME),
        (hardcase_path, PREDICTIVE_EGO_MOTION_PRODUCER_STANDALONE),
    ]:
        np.savez(
            path,
            state=np.zeros((2, 3, 9), dtype=np.float32),
            target=np.zeros((2, 3, 5, 2), dtype=np.float32),
            mask=np.ones((2, 3), dtype=np.float32),
            target_mask=np.ones((2, 3, 5), dtype=np.float32),
            feature_schema_json=np.asarray(
                _predictive_feature_schema_json(
                    model_family=PREDICTIVE_EGO_FEATURE_SCHEMA,
                    ego_conditioning=True,
                    producer=producer,
                )
            ),
        )

    report_path = tmp_path / "producer_metadata_preflight.json"
    payload = pipeline._write_producer_metadata_preflight_report(
        report_path=report_path,
        run_id="predictive_ego_producer_mismatch",
        config_path=tmp_path / "config.yaml",
        config_hash="cfghash",
        git_commit="deadbeef",
        model_family=PREDICTIVE_EGO_FEATURE_SCHEMA,
        datasets={
            "base": pipeline._predictive_feature_dataset_report(base_path),
            "hardcase": pipeline._predictive_feature_dataset_report(hardcase_path),
        },
    )

    assert payload["status"] == "failed"
    assert payload["failure_reason"] == (
        "Predictive ego motion producer keys do not match between datasets."
    )
    assert json.loads(report_path.read_text(encoding="utf-8"))["status"] == "failed"


def test_producer_metadata_preflight_rejects_schema_presence_mismatch(
    tmp_path: Path,
) -> None:
    """Producer preflight must fail when only one dataset has parsed schema metadata."""
    base_path = tmp_path / "predictive_rollouts_base.npz"
    hardcase_path = tmp_path / "predictive_rollouts_hardcase.npz"
    np.savez(
        base_path,
        state=np.zeros((2, 3, 4), dtype=np.float32),
        target=np.zeros((2, 3, 5, 2), dtype=np.float32),
        mask=np.ones((2, 3), dtype=np.float32),
        target_mask=np.ones((2, 3, 5), dtype=np.float32),
        feature_schema_json=np.asarray(
            _predictive_feature_schema_json(model_family="predictive_legacy_v1")
        ),
    )
    np.savez(
        hardcase_path,
        state=np.zeros((2, 3, 4), dtype=np.float32),
        target=np.zeros((2, 3, 5, 2), dtype=np.float32),
        mask=np.ones((2, 3), dtype=np.float32),
        target_mask=np.ones((2, 3, 5), dtype=np.float32),
    )

    payload = pipeline._write_producer_metadata_preflight_report(
        report_path=tmp_path / "producer_metadata_preflight.json",
        run_id="predictive_schema_presence_mismatch",
        config_path=tmp_path / "config.yaml",
        config_hash="cfghash",
        git_commit="deadbeef",
        model_family="predictive_legacy_v1",
        datasets={
            "base": pipeline._predictive_feature_dataset_report(base_path),
            "hardcase": pipeline._predictive_feature_dataset_report(hardcase_path),
        },
    )

    assert payload["status"] == "failed"
    assert payload["failure_reason"] == (
        "Predictive feature schema presence mismatch between datasets "
        "(one dataset has parsed feature_schema while the other does not)."
    )


def test_predictive_feature_dataset_report_rejects_missing_ego_producer(tmp_path: Path) -> None:
    """Ego-conditioned datasets without producer metadata must fail preflight."""
    dataset_path = tmp_path / "predictive_rollouts_base.npz"
    np.savez(
        dataset_path,
        state=np.zeros((2, 3, 9), dtype=np.float32),
        target=np.zeros((2, 3, 5, 2), dtype=np.float32),
        mask=np.ones((2, 3), dtype=np.float32),
        target_mask=np.ones((2, 3, 5), dtype=np.float32),
        feature_schema_json=np.asarray(
            _predictive_feature_schema_json(
                model_family=PREDICTIVE_EGO_FEATURE_SCHEMA,
                ego_conditioning=True,
                producer=None,
            )
        ),
    )

    report = pipeline._predictive_feature_dataset_report(dataset_path)

    assert report["status"] == "failed"
    assert report["failure_reason"] == (
        "Ego-conditioned dataset is missing ego_motion_channel_producer metadata."
    )


def test_pipeline_rejects_mismatched_collection_model_families(
    monkeypatch,
    tmp_path: Path,
) -> None:
    """Base and hardcase datasets must share the same predictive feature schema."""
    config_path = tmp_path / "predictive_bad_schema.yaml"
    output_root = _pipeline_test_output_root(tmp_path)
    config_path.write_text(
        yaml.safe_dump(
            {
                "experiment": {"run_id": "predictive_bad_schema"},
                "model_family": PREDICTIVE_OBSTACLE_FEATURE_SCHEMA,
                "output": {"root": output_root},
                "scenarios": {
                    "scenario_matrix": "scenarios.yaml",
                    "hard_seed_manifest": "hard.yaml",
                    "planner_grid": "grid.yaml",
                },
                "base_collection": {},
                "hardcase_collection": {"model_family": "predictive_legacy_v1"},
                "mixing": {},
                "training": {},
                "wandb": {"enabled": False},
                "evaluation": {},
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    (tmp_path / "scenarios.yaml").write_text("[]\n", encoding="utf-8")
    (tmp_path / "hard.yaml").write_text("{}\n", encoding="utf-8")
    (tmp_path / "grid.yaml").write_text("{}\n", encoding="utf-8")
    monkeypatch.setattr(
        pipeline,
        "_parse_args",
        lambda: pipeline.argparse.Namespace(
            config=config_path,
            run_id="predictive_bad_schema",
            log_level="INFO",
        ),
    )
    monkeypatch.setattr(
        pipeline,
        "_build_random_seed_manifest",
        lambda **kwargs: Path(kwargs["output_path"]),
    )

    with pytest.raises(ValueError, match="model families"):
        pipeline.main()


def test_pipeline_uses_resolved_model_id_and_fails_when_promotion_fails(
    monkeypatch,
    tmp_path: Path,
) -> None:
    """Pipeline should reuse one resolved model id and mark all_ok false on promotion failure."""
    config_path = tmp_path / "predictive.yaml"
    output_root = _pipeline_test_output_root(tmp_path)
    config_path.write_text(
        yaml.safe_dump(
            {
                "experiment": {"run_id": "predictive_promotion_smoke"},
                "output": {"root": output_root},
                "scenarios": {
                    "scenario_matrix": "scenarios.yaml",
                    "hard_seed_manifest": "hard.yaml",
                    "planner_grid": "grid.yaml",
                },
                "base_collection": {},
                "hardcase_collection": {},
                "mixing": {},
                "training": {"model_id": "predictive_explicit_model"},
                "wandb": {"enabled": False},
                "evaluation": {},
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    (tmp_path / "scenarios.yaml").write_text("[]\n", encoding="utf-8")
    (tmp_path / "hard.yaml").write_text("{}\n", encoding="utf-8")
    (tmp_path / "grid.yaml").write_text("{}\n", encoding="utf-8")

    monkeypatch.setattr(pipeline, "_sha1_file", lambda _path: "cfghash")
    monkeypatch.setattr(pipeline, "_git_hash", lambda: "deadbeef")

    def _fake_build_seed_manifest(**kwargs) -> Path:
        """Write the seed manifest path requested by the promotion pipeline."""
        output_path = Path(kwargs["output_path"])
        output_path.write_text("scenario: [1]\n", encoding="utf-8")
        return output_path

    monkeypatch.setattr(pipeline, "_build_random_seed_manifest", _fake_build_seed_manifest)

    def _fake_dataset(path: Path) -> None:
        """Write a default predictive dataset fixture."""
        np.savez(
            path,
            state=np.zeros((2, 3, 4), dtype=np.float32),
            target=np.zeros((2, 3, 5, 2), dtype=np.float32),
            mask=np.ones((2, 3), dtype=np.float32),
            target_mask=np.ones((2, 3, 5), dtype=np.float32),
        )
        path.with_suffix(".json").write_text(json.dumps({"num_samples": 2}), encoding="utf-8")

    invoked: list[list[str]] = []

    def _fake_run(cmd: list[str], *, log_level: str) -> None:
        """Simulate promotion pipeline commands and materialize expected artifacts."""
        invoked.append(list(cmd))
        if any("collect_predictive_hardcase_data.py" in part for part in cmd):
            output = Path(cmd[cmd.index("--output") + 1])
            output.parent.mkdir(parents=True, exist_ok=True)
            _fake_dataset(output)
            return
        if any("build_predictive_mixed_dataset.py" in part for part in cmd):
            output = Path(cmd[cmd.index("--output") + 1])
            output.parent.mkdir(parents=True, exist_ok=True)
            _fake_dataset(output)
            return
        if any("train_predictive_planner.py" in part for part in cmd):
            out_dir = Path(cmd[cmd.index("--output-dir") + 1])
            out_dir.mkdir(parents=True, exist_ok=True)
            (out_dir / "training_summary.json").write_text(
                json.dumps(
                    {"selection": {"selected_epoch": 1}, "selected_checkpoint_reason": "proxy"}
                ),
                encoding="utf-8",
            )
            (out_dir / "predictive_model.pt").write_text("stub", encoding="utf-8")
            return
        if any("run_predictive_success_campaign.py" in part for part in cmd):
            output_dir = Path(cmd[cmd.index("--output-dir") + 1])
            output_dir.mkdir(parents=True, exist_ok=True)
            (output_dir / "campaign_summary.json").write_text(
                json.dumps({"run_id": "predictive_promotion_smoke", "status": "success"}),
                encoding="utf-8",
            )
            return
        raise AssertionError(f"Unexpected command: {cmd}")

    def _fake_run_capture_json(cmd: list[str], **_kwargs):
        """Return a failed registry response only for checkpoint registration."""
        if "--checkpoint-only-register" in cmd:
            return {"status": "failed", "return_code": 2}
        return {"status": "ok", "return_code": 0}

    monkeypatch.setattr(pipeline, "_run", _fake_run)
    monkeypatch.setattr(pipeline, "_run_capture_json", _fake_run_capture_json)
    monkeypatch.setattr(
        pipeline,
        "_parse_args",
        lambda: pipeline.argparse.Namespace(
            config=config_path,
            run_id="predictive_promotion_smoke",
            log_level="INFO",
        ),
    )

    resolved_paths = pipeline._paths_from_config(
        yaml.safe_load(config_path.read_text(encoding="utf-8")),
        run_id="predictive_promotion_smoke",
        base_dir=config_path.parent,
        output_base_dir=Path.cwd().resolve(),
    )
    code = pipeline.main()
    assert code == 2
    train_cmd = next(
        cmd for cmd in invoked if any("train_predictive_planner.py" in part for part in cmd)
    )
    assert train_cmd[train_cmd.index("--model-id") + 1] == "predictive_explicit_model"
    final_summary = json.loads(resolved_paths.final_summary.read_text(encoding="utf-8"))
    assert final_summary["promoted_model"]["model_id"] == "predictive_explicit_model"
    assert final_summary["stage_status"]["promotion_ok"] is False
    assert final_summary["final_gate_results"]["all_ok"] is False


def test_pipeline_uses_committed_base_seed_manifest_without_generating(
    monkeypatch,
    tmp_path: Path,
) -> None:
    """Pipeline should reuse a configured base seed manifest and record it as pre-existing."""
    config_path = tmp_path / "predictive_existing_manifest.yaml"
    output_root = _pipeline_test_output_root(tmp_path)
    config_path.write_text(
        yaml.safe_dump(
            {
                "experiment": {"run_id": "predictive_existing_manifest"},
                "output": {"root": output_root},
                "scenarios": {
                    "scenario_matrix": "scenarios.yaml",
                    "hard_seed_manifest": "hard.yaml",
                    "planner_grid": "grid.yaml",
                },
                "base_collection": {"seed_manifest": "base_manifest.yaml"},
                "hardcase_collection": {},
                "mixing": {},
                "training": {"model_id": "predictive_existing_manifest"},
                "wandb": {"enabled": False},
                "evaluation": {},
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    (tmp_path / "scenarios.yaml").write_text("[]\n", encoding="utf-8")
    (tmp_path / "hard.yaml").write_text("{}\n", encoding="utf-8")
    (tmp_path / "grid.yaml").write_text("{}\n", encoding="utf-8")
    base_manifest = tmp_path / "base_manifest.yaml"
    base_manifest.write_text("scenario: [7]\n", encoding="utf-8")

    monkeypatch.setattr(pipeline, "_sha1_file", lambda _path: "cfghash")
    monkeypatch.setattr(pipeline, "_git_hash", lambda: "deadbeef")

    build_calls: list[dict[str, object]] = []

    def _fake_build_seed_manifest(**kwargs) -> Path:
        """Capture unexpected seed-manifest generation requests."""
        build_calls.append(kwargs)
        output_path = Path(kwargs["output_path"])
        output_path.write_text("scenario: [1]\n", encoding="utf-8")
        return output_path

    monkeypatch.setattr(pipeline, "_build_random_seed_manifest", _fake_build_seed_manifest)

    def _fake_dataset(path: Path) -> None:
        """Write a default predictive dataset fixture."""
        np.savez(
            path,
            state=np.zeros((2, 3, 4), dtype=np.float32),
            target=np.zeros((2, 3, 5, 2), dtype=np.float32),
            mask=np.ones((2, 3), dtype=np.float32),
            target_mask=np.ones((2, 3, 5), dtype=np.float32),
        )
        path.with_suffix(".json").write_text(json.dumps({"num_samples": 2}), encoding="utf-8")

    def _fake_run(cmd: list[str], *, log_level: str) -> None:
        """Simulate pipeline commands and materialize expected artifacts."""
        if any("collect_predictive_hardcase_data.py" in part for part in cmd):
            output = Path(cmd[cmd.index("--output") + 1])
            output.parent.mkdir(parents=True, exist_ok=True)
            _fake_dataset(output)
            return
        if any("build_predictive_mixed_dataset.py" in part for part in cmd):
            output = Path(cmd[cmd.index("--output") + 1])
            output.parent.mkdir(parents=True, exist_ok=True)
            _fake_dataset(output)
            return
        if any("train_predictive_planner.py" in part for part in cmd):
            out_dir = Path(cmd[cmd.index("--output-dir") + 1])
            out_dir.mkdir(parents=True, exist_ok=True)
            (out_dir / "training_summary.json").write_text(
                json.dumps(
                    {"selection": {"selected_epoch": 1}, "selected_checkpoint_reason": "proxy"}
                ),
                encoding="utf-8",
            )
            (out_dir / "predictive_model.pt").write_text("stub", encoding="utf-8")
            return
        if any("run_predictive_success_campaign.py" in part for part in cmd):
            output_dir = Path(cmd[cmd.index("--output-dir") + 1])
            output_dir.mkdir(parents=True, exist_ok=True)
            (output_dir / "campaign_summary.json").write_text(
                json.dumps({"run_id": "predictive_existing_manifest", "status": "success"}),
                encoding="utf-8",
            )
            return
        raise AssertionError(f"Unexpected command: {cmd}")

    monkeypatch.setattr(pipeline, "_run", _fake_run)
    monkeypatch.setattr(
        pipeline,
        "_run_capture_json",
        lambda _cmd, **_kwargs: {"status": "ok", "return_code": 0},
    )
    monkeypatch.setattr(
        pipeline,
        "_parse_args",
        lambda: pipeline.argparse.Namespace(
            config=config_path,
            run_id="predictive_existing_manifest",
            log_level="INFO",
        ),
    )

    resolved_paths = pipeline._paths_from_config(
        yaml.safe_load(config_path.read_text(encoding="utf-8")),
        run_id="predictive_existing_manifest",
        base_dir=config_path.parent,
        output_base_dir=Path.cwd().resolve(),
    )

    assert pipeline.main() == 0
    assert build_calls == []

    final_summary = json.loads(resolved_paths.final_summary.read_text(encoding="utf-8"))
    assert final_summary["base_seed_manifest"] == str(base_manifest)
    assert final_summary["base_seed_manifest_generated"] is False

    base_dataset_manifest = json.loads(
        Path(final_summary["dataset_manifests"]["base"]).read_text(encoding="utf-8")
    )
    assert base_dataset_manifest["extra"]["seed_manifest"] == str(base_manifest)
    assert base_dataset_manifest["extra"]["seed_manifest_generated"] is False
