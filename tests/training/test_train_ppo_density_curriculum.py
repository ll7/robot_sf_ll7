"""PPO config validation tests for issue #4018 density curriculum."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from robot_sf.training.density_curriculum import build_density_curriculum_schedule
from scripts.training.train_ppo import _DensityCurriculumCallback, load_expert_training_config

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


def test_density_curriculum_callback_publishes_only_on_stage_change() -> None:
    """Callback avoids per-step environment IPC when the active stage is unchanged."""

    class _CountingCallback(_DensityCurriculumCallback):
        def __init__(self) -> None:
            super().__init__(schedule)
            self.published_timesteps: list[int] = []

        def _publish_timestep(self) -> None:
            self.published_timesteps.append(int(self.num_timesteps))

    schedule = build_density_curriculum_schedule(
        {
            "enabled": True,
            "stages": [
                {"id": "sparse", "until_timesteps": 10, "density_m2": 0.04},
                {"id": "dense", "until_timesteps": None, "density_m2": 0.12},
            ],
        }
    )
    callback = _CountingCallback()

    callback.num_timesteps = 0
    callback._on_training_start()
    callback.num_timesteps = 5
    assert callback._on_step() is True
    callback.num_timesteps = 10
    assert callback._on_step() is True

    assert callback.published_timesteps == [0, 10]
