"""Unit tests for predictive training pipeline helper utilities."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import yaml

from scripts.training import run_predictive_training_pipeline as pipeline


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
            {"name": "classic_crossing_low"},
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
    assert payload["classic_crossing_low"] == [100, 101, 102]
    assert payload["classic_doorway_high"] == [100100, 100101, 100102]


def test_run_capture_json_anchors_subprocess_to_repo_root(monkeypatch) -> None:
    """Subprocess helpers should execute repo-relative commands from the repository root."""
    called: dict[str, object] = {}

    def _fake_run(cmd, **kwargs):
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
    config_path.write_text(
        yaml.safe_dump(
            {
                "experiment": {"run_id": "predictive_ego_smoke"},
                "output": {"root": "output/tmp/predictive_planner/pipeline"},
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

    def _fake_build_random_seed_manifest(**kwargs) -> Path:
        output_path = Path(kwargs["output_path"])
        output_path.write_text("scenario: [1]\n", encoding="utf-8")
        return output_path

    monkeypatch.setattr(pipeline, "_build_random_seed_manifest", _fake_build_random_seed_manifest)

    def _fake_dataset(path: Path, *, state_dim: int) -> None:
        np.savez(
            path,
            state=np.zeros((2, 3, state_dim), dtype=np.float32),
            target=np.zeros((2, 3, 5, 2), dtype=np.float32),
            mask=np.ones((2, 3), dtype=np.float32),
            target_mask=np.ones((2, 3, 5), dtype=np.float32),
        )
        path.with_suffix(".json").write_text(json.dumps({"num_samples": 2}), encoding="utf-8")

    invoked: list[list[str]] = []

    def _fake_run(cmd: list[str], *, log_level: str) -> None:
        invoked.append(list(cmd))
        if any("collect_predictive_hardcase_data.py" in part for part in cmd):
            output = Path(cmd[cmd.index("--output") + 1])
            state_dim = 9 if "--ego-conditioning" in cmd else 4
            output.parent.mkdir(parents=True, exist_ok=True)
            _fake_dataset(output, state_dim=state_dim)
            return
        if any("build_predictive_mixed_dataset.py" in part for part in cmd):
            output = Path(cmd[cmd.index("--output") + 1])
            output.parent.mkdir(parents=True, exist_ok=True)
            _fake_dataset(output, state_dim=9)
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

    monkeypatch.setattr(pipeline, "_run", _fake_run)
    monkeypatch.setattr(
        pipeline,
        "_run_capture_json",
        lambda _cmd, **_kwargs: {"status": "ok", "returncode": 0},
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

    code = pipeline.main()
    assert code in {0, 2}
    collector_cmds = [
        cmd for cmd in invoked if any("collect_predictive_hardcase_data.py" in part for part in cmd)
    ]
    assert len(collector_cmds) == 2
    assert all("--ego-conditioning" in cmd for cmd in collector_cmds)


def test_pipeline_uses_resolved_model_id_and_fails_when_promotion_fails(
    monkeypatch,
    tmp_path: Path,
) -> None:
    """Pipeline should reuse one resolved model id and mark all_ok false on promotion failure."""
    config_path = tmp_path / "predictive.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "experiment": {"run_id": "predictive_promotion_smoke"},
                "output": {"root": "output/tmp/predictive_planner/pipeline"},
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
        output_path = Path(kwargs["output_path"])
        output_path.write_text("scenario: [1]\n", encoding="utf-8")
        return output_path

    monkeypatch.setattr(pipeline, "_build_random_seed_manifest", _fake_build_seed_manifest)

    def _fake_dataset(path: Path) -> None:
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
        output_base_dir=pipeline._REPO_ROOT,
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
