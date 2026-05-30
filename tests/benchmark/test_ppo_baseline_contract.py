"""Contract tests for canonical PPO benchmark baseline configs."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from robot_sf.models.registry import load_registry

CANONICAL_PPO_MODEL_ID = (
    "ppo_expert_issue_791_reward_curriculum_eval_aligned_large_capacity_20260417"
)
ISSUE_576_PPO_MODEL_ID = "ppo_expert_br06_v3_15m_all_maps_randomized_20260304T075200"


def _load_yaml(path: Path) -> dict:
    """Load YAML from a repository fixture path."""
    return yaml.safe_load(path.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def repo_root() -> Path:
    """Return repository root for benchmark config contract tests."""
    return Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def raw_registry(repo_root: Path) -> dict:
    """Load raw model registry YAML used as source-of-truth for promoted artifacts."""
    return _load_yaml(repo_root / "model" / "registry.yaml")


@pytest.fixture(scope="module")
def canonical_entry(raw_registry: dict) -> dict:
    """Return the canonical PPO registry entry referenced by configs/baselines/ppo_15m_grid_socnav.yaml."""
    matching_entries = [
        entry for entry in raw_registry["models"] if entry["model_id"] == CANONICAL_PPO_MODEL_ID
    ]

    assert len(matching_entries) == 1
    return matching_entries[0]


@pytest.fixture(scope="module")
def issue_576_entry(raw_registry: dict) -> dict:
    """Return the issue-576 closeout PPO registry entry."""
    matching_entries = [
        entry for entry in raw_registry["models"] if entry["model_id"] == ISSUE_576_PPO_MODEL_ID
    ]

    assert len(matching_entries) == 1
    return matching_entries[0]


def test_model_registry_loads_without_duplicate_model_ids(repo_root: Path) -> None:
    """Registry loader should accept the promoted PPO registry without duplicate model IDs."""
    registry = load_registry(repo_root / "model" / "registry.yaml")

    assert CANONICAL_PPO_MODEL_ID in registry
    assert ISSUE_576_PPO_MODEL_ID in registry


def test_canonical_ppo_baseline_points_at_promoted_artifact_path(
    repo_root: Path, canonical_entry: dict
) -> None:
    """Benchmark-facing PPO baseline should resolve through the durable registry entry."""
    baseline_config = _load_yaml(repo_root / "configs" / "baselines" / "ppo_15m_grid_socnav.yaml")

    assert "model_path" not in baseline_config
    assert baseline_config["model_id"] == canonical_entry["model_id"]
    assert canonical_entry["wandb_artifact_path"].endswith("-best-success:v9")


def test_canonical_ppo_baseline_declares_observation_track_metadata(canonical_entry: dict) -> None:
    """The benchmark-promoted PPO checkpoint should declare its track-level input contract."""
    promotion = canonical_entry["benchmark_promotion"]

    assert promotion["claim_boundary"] == "benchmark_promoted"
    assert promotion["benchmark_track"] == "grid_socnav_v1"
    assert promotion["observation_level"] == "tracked_agents_no_noise"
    assert promotion["observation_mode"] == "dict"
    assert "occupancy_grid" in promotion["allowed_observation_keys"]
    assert "predictive_min_clearance" in promotion["allowed_observation_keys"]
    assert promotion["privileged_input_status"] == "no evaluation-time privileged inputs"
    assert "docs/context/issue_1612_observation_track_architecture.md" in promotion["reference"]


def test_issue_576_closeout_config_has_resolved_provenance(
    repo_root: Path, issue_576_entry: dict
) -> None:
    """Issue-576 closeout config should no longer carry placeholder provenance fields."""
    issue_config = _load_yaml(
        repo_root / "configs" / "baselines" / "ppo_issue_576_br06_v2_15m.yaml"
    )

    assert issue_config["model_id"] == ISSUE_576_PPO_MODEL_ID
    assert issue_config["provenance"]["training_config"] == issue_576_entry["config_path"]
    assert issue_config["provenance"]["training_commit"] == issue_576_entry["commit"]
    assert issue_config["provenance"]["checkpoint_id"] == ISSUE_576_PPO_MODEL_ID
    assert issue_config["provenance"]["normalization_id"] == "route_completion_v3"
    assert issue_config["quality_gate"]["measured_success_rate"] > 0.0
