"""Training config tests for the issue #4014 PPO-Mamba smoke lane."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

from robot_sf.feature_extractors.mamba_extractor import MambaFeatureExtractor
from scripts.training import train_ppo

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = REPO_ROOT / "configs" / "training" / "ppo" / "issue_4014_ppo_mamba_smoke.yaml"


def test_ppo_mamba_smoke_config_loads_cpu_safe_extractor_contract() -> None:
    """The PPO-Mamba smoke config registers the CPU-safe Mamba extractor."""
    config = train_ppo.load_expert_training_config(CONFIG_PATH)

    assert config.policy_id == "ppo_mamba_issue_4014_smoke"
    assert config.feature_extractor == "mamba"
    assert config.feature_extractor_kwargs == {
        "backend": "torch_ssm_lite",
        "d_model": 64,
        "d_state": 16,
        "d_conv": 4,
        "expand": 2,
        "num_layers": 1,
        "dropout_rate": 0.0,
        "sequence_source": "rays",
        "drive_hidden_dims": [32, 16],
    }
    assert config.total_timesteps == 2048
    assert config.seeds == (4014,)


def test_ppo_policy_selection_routes_mamba_extractor() -> None:
    """PPO policy kwargs should instantiate the Mamba extractor instead of SB3 default features."""
    config = train_ppo.load_expert_training_config(CONFIG_PATH)

    policy, policy_kwargs, critic_profile = train_ppo._resolve_policy_selection(config)

    assert policy == "MultiInputPolicy"
    assert policy_kwargs["features_extractor_class"] is MambaFeatureExtractor
    assert policy_kwargs["features_extractor_kwargs"] == config.feature_extractor_kwargs
    assert critic_profile == "standard"


def test_ppo_mamba_dry_run_emits_parameter_summary_placeholder(tmp_path: Path) -> None:
    """Dry-run PPO-Mamba perf summary records the parameter-count contract."""
    env = {
        **os.environ,
        "ROBOT_SF_ARTIFACT_ROOT": str(tmp_path),
    }
    result = subprocess.run(
        [
            sys.executable,
            "scripts/training/train_ppo.py",
            "--config",
            str(CONFIG_PATH),
            "--dry-run",
            "--log-level",
            "WARNING",
        ],
        cwd=REPO_ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr

    perf_paths = sorted((tmp_path / "benchmarks" / "ppo_imitation" / "perf").glob("*.json"))
    assert len(perf_paths) == 1
    payload = json.loads(perf_paths[0].read_text(encoding="utf-8"))
    assert payload["parameter_summary"] == {
        "available": False,
        "policy_parameter_count": None,
        "policy_trainable_parameter_count": None,
        "model_parameter_count": None,
        "model_trainable_parameter_count": None,
    }
