"""Tests for routed worker artifact manifests."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest

from scripts.dev import routed_worker_manifest as manifest

if TYPE_CHECKING:
    from pathlib import Path


def _write_artifacts(run_dir: Path, filenames: list[str]) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    for filename in filenames:
        (run_dir / filename).write_text(f"{filename}\n", encoding="utf-8")


def test_scan_artifact_presence_reports_success_and_missing(tmp_path: Path) -> None:
    """Presence scan should include paths and missing reasons for all artifacts."""
    run_dir = tmp_path / "repo" / ".git" / "codex-agent-runs" / "run-1"
    _write_artifacts(run_dir, ["result.json", "RESULT.md", "status.txt"])

    presence = manifest.scan_artifact_presence(
        ".git/codex-agent-runs/run-1",
        target_repo=tmp_path / "repo",
    )

    assert presence["result_json"].present is True
    assert presence["result_json"].path == ".git/codex-agent-runs/run-1/result.json"
    assert presence["result_json"].size_bytes is not None
    assert presence["diffstat"].present is False
    assert presence["diffstat"].path == ".git/codex-agent-runs/run-1/diffstat.txt"
    assert presence["diffstat"].reason == "missing"
    assert presence["validation"].reason == "missing"


def test_build_manifest_includes_attempts_chosen_route_and_warning(tmp_path: Path) -> None:
    """Manifest should distinguish route evidence from task acceptance."""
    repo = tmp_path / "repo"
    run_dir = repo / ".git" / "codex-agent-runs" / "run-2"
    _write_artifacts(
        run_dir,
        ["result.json", "RESULT.md", "diffstat.txt", "status.txt", "validation.txt"],
    )
    attempts = [
        {
            "route": {"provider": "gemini"},
            "returncode": 2,
            "failure_class": "route-collapse",
            "run_dir": None,
        },
        {
            "route": {"provider": "qwen"},
            "returncode": 0,
            "failure_class": "none",
            "run_dir": ".git/codex-agent-runs/run-2",
        },
    ]

    data = manifest.build_routing_manifest(
        attempts,
        chosen_index=1,
        target_repo=repo,
        task_class="mechanical_code_edit",
    )

    assert data["schema"] == "routed_worker_manifest.v1"
    assert data["route_evidence_only"] is True
    assert "not task acceptance" in data["warning"]
    assert len(data["attempted_routes"]) == 2
    assert data["attempted_routes"][0]["compact_artifacts"]["validation"]["reason"] == "not-run"
    assert data["chosen_route"] == {"provider": "qwen"}
    assert data["compact_artifacts"]["validation"]["present"] is True


def test_write_manifest_uses_target_repository_run_directory(tmp_path: Path) -> None:
    """Relative run dirs must resolve under target_repo, not the caller cwd."""
    target_repo = tmp_path / "target-repo"
    orchestrator_repo = tmp_path / "orchestrator"
    chosen_run_dir = ".git/codex-agent-runs/run-3"
    _write_artifacts(
        target_repo / chosen_run_dir,
        ["result.json", "RESULT.md", "diffstat.txt", "status.txt"],
    )
    orchestrator_repo.mkdir()
    attempts = [
        {
            "route": {"provider": "qwen"},
            "returncode": 0,
            "failure_class": "none",
            "run_dir": chosen_run_dir,
        }
    ]

    output_path = manifest.write_routing_manifest(
        attempts,
        chosen_index=0,
        target_repo=target_repo,
        task_class="mechanical_code_edit",
    )

    assert output_path == target_repo / chosen_run_dir / "routing_manifest.json"
    assert not (orchestrator_repo / chosen_run_dir / "routing_manifest.json").exists()
    data = json.loads(output_path.read_text(encoding="utf-8"))
    assert data["compact_artifacts"]["validation"]["present"] is False
    assert data["compact_artifacts"]["validation"]["reason"] == "missing"


def test_write_manifest_rejects_run_dir_outside_target_repository(tmp_path: Path) -> None:
    """Traversal-style run dirs should not write manifests outside target_repo."""
    target_repo = tmp_path / "target-repo"
    target_repo.mkdir()
    attempts = [
        {
            "route": {"provider": "qwen"},
            "returncode": 0,
            "failure_class": "none",
            "run_dir": "../outside-run",
        }
    ]

    with pytest.raises(ValueError, match="inside target_repo"):
        manifest.write_routing_manifest(attempts, chosen_index=0, target_repo=target_repo)

    assert not (tmp_path / "outside-run" / "routing_manifest.json").exists()


def test_scan_artifact_presence_rejects_symlink_run_directory(tmp_path: Path) -> None:
    """Symlink-valued run dirs should fail closed before path resolution."""
    target_repo = tmp_path / "target-repo"
    outside_run = tmp_path / "outside-run"
    target_repo.mkdir()
    outside_run.mkdir()
    (target_repo / "run-link").symlink_to(outside_run, target_is_directory=True)

    with pytest.raises(ValueError, match="must not be a symlink"):
        manifest.scan_artifact_presence("run-link", target_repo=target_repo)


def test_build_manifest_rejects_invalid_chosen_index() -> None:
    """A malformed wrapper should fail before writing misleading route evidence."""
    with pytest.raises(IndexError):
        manifest.build_routing_manifest(
            [{"route": {"provider": "qwen"}, "run_dir": "run"}],
            chosen_index=3,
        )
