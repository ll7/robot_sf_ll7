"""Contract tests for canonical PPO benchmark baseline configs."""

from __future__ import annotations

from pathlib import Path

import yaml

PROMOTED_PPO_MODEL_ID = "ppo_expert_br06_v3_15m_all_maps_randomized_20260304T075200"


def _load_yaml(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def test_canonical_ppo_baseline_points_at_promoted_registry_model() -> None:
    """Benchmark-facing PPO baseline should follow the promoted BR-06 registry entry."""
    repo_root = Path(__file__).resolve().parents[2]
    baseline_config = _load_yaml(repo_root / "configs" / "baselines" / "ppo_15m_grid_socnav.yaml")
    registry = _load_yaml(repo_root / "model" / "registry.yaml")

    assert baseline_config["model_id"] == PROMOTED_PPO_MODEL_ID

    promoted_entry = next(
        entry for entry in registry["models"] if entry["model_id"] == PROMOTED_PPO_MODEL_ID
    )
    assert baseline_config["model_path"] == promoted_entry["local_path"]


def test_issue_576_closeout_config_has_resolved_provenance() -> None:
    """Issue-576 closeout config should no longer carry placeholder provenance fields."""
    repo_root = Path(__file__).resolve().parents[2]
    issue_config = _load_yaml(
        repo_root / "configs" / "baselines" / "ppo_issue_576_br06_v2_15m.yaml"
    )

    assert issue_config["model_id"] == PROMOTED_PPO_MODEL_ID
    assert issue_config["provenance"]["training_config"] == (
        "configs/training/ppo/expert_ppo_issue_576_br06_v3_15m_all_maps_randomized.yaml"
    )
    assert (
        issue_config["provenance"]["training_commit"] == "9fb131b6d7ff062887caab33ca8713ae05167ebe"
    )
    assert issue_config["provenance"]["checkpoint_id"] == PROMOTED_PPO_MODEL_ID
    assert issue_config["provenance"]["normalization_id"] == "route_completion_v3"
    assert issue_config["quality_gate"]["measured_success_rate"] > 0.0
