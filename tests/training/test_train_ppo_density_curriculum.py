"""PPO config validation tests for issue #4018 density curriculum."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from scripts.training.train_ppo import load_expert_training_config

REPO_ROOT = Path(__file__).resolve().parents[2]
CURRICULUM_CONFIG = (
    REPO_ROOT / "configs/training/ppo/ablations/issue_4018_density_curriculum_smoke.yaml"
)
BASELINE_CONFIG = REPO_ROOT / "configs/training/ppo/ablations/issue_4018_fixed_density_smoke.yaml"


def test_issue_4018_smoke_configs_load_with_matched_budget() -> None:
    """Curriculum and fixed-density smoke configs parse and keep matched budget."""
    curriculum = load_expert_training_config(CURRICULUM_CONFIG)
    baseline = load_expert_training_config(BASELINE_CONFIG)

    assert curriculum.density_curriculum["enabled"] is True
    assert baseline.density_curriculum["enabled"] is False
    assert curriculum.total_timesteps == baseline.total_timesteps == 96


def test_invalid_density_curriculum_config_fails_closed(tmp_path: Path) -> None:
    """Config loading rejects invalid enabled schedules before training starts."""
    payload = yaml.safe_load(CURRICULUM_CONFIG.read_text(encoding="utf-8"))
    payload["density_curriculum"]["stages"][1]["density_m2"] = 0.01
    config_path = tmp_path / "bad_density_curriculum.yaml"
    config_path.write_text(yaml.safe_dump(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="non-decreasing"):
        load_expert_training_config(config_path)
