"""Contract tests for canonical PPO benchmark baseline configs."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

PROMOTED_PPO_MODEL_ID = "ppo_expert_br06_v3_15m_all_maps_randomized_20260304T075200"


def _load_yaml(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def repo_root() -> Path:
    """Return repository root for benchmark config contract tests."""
    return Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def registry(repo_root: Path) -> dict:
    """Load model registry used as source-of-truth for promoted artifacts."""
    return _load_yaml(repo_root / "model" / "registry.yaml")


@pytest.fixture(scope="module")
def promoted_entry(registry: dict) -> dict:
    """Return the promoted PPO registry entry referenced by benchmark configs."""
    return next(entry for entry in registry["models"] if entry["model_id"] == PROMOTED_PPO_MODEL_ID)


def test_canonical_ppo_baseline_points_at_promoted_artifact_path(
    repo_root: Path, promoted_entry: dict
) -> None:
    """Benchmark-facing PPO baseline should stay path-based and offline-safe."""
    baseline_config = _load_yaml(repo_root / "configs" / "baselines" / "ppo_15m_grid_socnav.yaml")

    assert "model_id" not in baseline_config
    assert baseline_config["model_path"] == promoted_entry["local_path"]


def test_issue_576_closeout_config_has_resolved_provenance(
    repo_root: Path, promoted_entry: dict
) -> None:
    """Issue-576 closeout config should no longer carry placeholder provenance fields."""
    issue_config = _load_yaml(
        repo_root / "configs" / "baselines" / "ppo_issue_576_br06_v2_15m.yaml"
    )

    assert issue_config["model_id"] == PROMOTED_PPO_MODEL_ID
    assert issue_config["provenance"]["training_config"] == promoted_entry["config_path"]
    assert issue_config["provenance"]["training_commit"] == promoted_entry["commit"]
    assert issue_config["provenance"]["checkpoint_id"] == PROMOTED_PPO_MODEL_ID
    assert issue_config["provenance"]["normalization_id"] == "route_completion_v3"
    assert issue_config["quality_gate"]["measured_success_rate"] > 0.0
