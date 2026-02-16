"""Contract tests for orchestrator algorithm metadata injection."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from loguru import logger

from robot_sf.benchmark.errors import AggregationMetadataError
from robot_sf.benchmark.full_classic import orchestrator

if TYPE_CHECKING:
    from pathlib import Path


def test_injects_nested_algo_metadata():
    """Ensures that missing nested algorithm metadata is injected."""

    record: dict[str, object] = {
        "episode_id": "ep-001",
        "scenario_params": {"density": "high"},
    }

    updated = orchestrator._ensure_algo_metadata(
        record,
        algo="sf",
        episode_id="ep-001",
    )

    assert updated["algo"] == "sf"
    assert updated["scenario_params"]["algo"] == "sf"
    algo_meta = updated["algorithm_metadata"]
    assert algo_meta["baseline_category"] == "classical"
    assert algo_meta["canonical_algorithm"] == "social_force"


def test_raises_on_missing_algo():
    """Missing algorithm identifiers should raise a metadata error."""

    record: dict[str, object] = {
        "episode_id": "ep-002",
        "scenario_params": {},
    }

    with pytest.raises(AggregationMetadataError):
        orchestrator._ensure_algo_metadata(record, algo=None, episode_id="ep-002")


def test_logs_warning_on_mismatch():
    """A conflicting nested algorithm value should trigger a warning and correction."""

    captured: list = []

    def capture_message(message):
        captured.append(message)

    handle = logger.add(capture_message, level="WARNING")
    try:
        record: dict[str, object] = {
            "episode_id": "ep-003",
            "scenario_params": {"algo": "random"},
        }

        updated = orchestrator._ensure_algo_metadata(
            record,
            algo="ppo",
            episode_id="ep-003",
        )
    finally:
        logger.remove(handle)

    assert updated["scenario_params"]["algo"] == "ppo"
    assert updated["algo"] == "ppo"
    assert any(msg.record["extra"].get("event") == "episode_metadata_mismatch" for msg in captured)


def test_find_repo_root_walks_up_to_git_metadata(tmp_path: Path) -> None:
    """Repo root discovery should walk parents until `.git` metadata is found."""

    repo_root = tmp_path / "repo"
    nested_dir = repo_root / "robot_sf" / "benchmark" / "full_classic"
    nested_dir.mkdir(parents=True)
    (repo_root / ".git").mkdir()

    found = orchestrator._find_repo_root(nested_dir / "orchestrator.py")

    assert found == repo_root
