"""Module test_train_expert_ppo auto-generated docstring."""

from __future__ import annotations

import json
from pathlib import Path

from robot_sf import common
from scripts.training.train_expert_ppo import (
    load_expert_training_config,
    run_expert_training,
)


def test_expert_training_dry_run(tmp_path, monkeypatch):
    """Test expert training dry run.

    Args:
        tmp_path: Auto-generated placeholder description.
        monkeypatch: Auto-generated placeholder description.

    Returns:
        Any: Auto-generated placeholder description.
    """
    monkeypatch.setenv("ROBOT_SF_ARTIFACT_ROOT", str(tmp_path))
    config_path = Path("configs/training/ppo_imitation/expert_ppo.yaml").resolve()
    config = load_expert_training_config(config_path)

    result = run_expert_training(config, config_path=config_path, dry_run=True)

    manifest_path = result.expert_manifest_path
    assert manifest_path.exists()

    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert payload["policy_id"] == config.policy_id
    assert set(payload["metrics"].keys()) >= {
        "success_rate",
        "collision_rate",
        "path_efficiency",
        "comfort_exposure",
        "snqi",
    }

    checkpoint = result.checkpoint_path
    assert checkpoint.exists() and checkpoint.read_text(encoding="utf-8").startswith("dry-run")

    run_manifest_path = result.training_run_manifest_path
    assert run_manifest_path.exists()
    training_payload = json.loads(run_manifest_path.read_text(encoding="utf-8"))
    assert training_payload["run_type"] == common.TrainingRunType.EXPERT_TRAINING.value

    log_dir = common.get_imitation_report_dir()
    assert any(log_dir.glob("episodes/*.jsonl"))
