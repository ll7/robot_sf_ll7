"""Focused tests for the issue #9431 release comparator."""

from __future__ import annotations

from io import BytesIO
import tarfile

import pytest

from scripts.analysis.compare_issue_9431_release import (
    _execution_audit,
    _markdown,
    _validate_successor_bundle_root,
)


def test_execution_audit_admits_mixed_mode_and_outcome_failures() -> None:
    row = {
        "status": "collision",
        "algorithm_metadata": {
            "planner_kinematics": {"execution_mode": "mixed"},
            "fallback_used": False,
            "degraded": False,
            "unavailable": False,
        },
    }

    assert _execution_audit(row) == []


def test_execution_audit_rejects_fallback_and_runtime_error() -> None:
    row = {
        "status": "runtime_error",
        "algorithm_metadata": {
            "planner_kinematics": {"execution_mode": "native"},
            "fallback_used": True,
        },
    }

    assert _execution_audit(row) == ["fallback_used=true", "status=runtime_error"]


def test_markdown_records_exact_source_range_and_execution_modes() -> None:
    report = {
        "predecessor": {
            "archive_sha256": "old-archive",
            "source_commit": "old-source",
        },
        "successor": {
            "source_commit": "new-source",
            "bundle_sha256": "new-bundle",
        },
        "paired_rows": 0,
        "changed_episode_count": 0,
        "execution_audit": {"admitted": True},
        "arms": {},
        "seed_block_bootstrap": {},
    }

    rendered = _markdown(report)

    assert "`old-source..new-source`" in rendered
    assert "native, adapter, or mixed mode" in rendered


def test_markdown_keeps_missing_goal_adjacent_labels_explicitly_unavailable() -> None:
    report = {
        "predecessor": {"archive_sha256": "old-archive", "source_commit": "old-source"},
        "successor": {
            "source_commit": "new-source",
            "bundle_sha256": "new-bundle",
        },
        "paired_rows": 1,
        "changed_episode_count": 0,
        "execution_audit": {"admitted": True},
        "arms": {
            "goal": {
                "paired_rows": 1,
                "predecessor_outcomes": {"success": 1},
                "successor_outcomes": {"success": 1},
                "predecessor_goal_adjacent_timeout": {"unavailable": 1},
                "successor_goal_adjacent_timeout": {"unavailable": 1},
            }
        },
        "seed_block_bootstrap": {},
    }

    rendered = _markdown(report)

    assert "old `unavailable=1`; new `unavailable=1`" in rendered
    assert "{'unavailable': 1}" not in rendered


def test_successor_root_must_match_pinned_bundle_episode_bytes(tmp_path) -> None:
    payload = b'{"scenario_id":"s1","seed":1}\n'
    bundle = tmp_path / "bundle.tar.gz"
    with tarfile.open(bundle, "w:gz") as archive:
        info = tarfile.TarInfo(
            "release/payload/runs/goal__differential_drive/episodes.jsonl"
        )
        info.size = len(payload)
        archive.addfile(info, BytesIO(payload))

    root = tmp_path / "root"
    run_dir = root / "runs/goal__differential_drive"
    run_dir.mkdir(parents=True)
    episode_file = run_dir / "episodes.jsonl"
    episode_file.write_bytes(payload)

    _validate_successor_bundle_root(bundle, root)

    episode_file.write_bytes(b'{"scenario_id":"other","seed":1}\n')
    with pytest.raises(ValueError, match="episode bytes do not match"):
        _validate_successor_bundle_root(bundle, root)
