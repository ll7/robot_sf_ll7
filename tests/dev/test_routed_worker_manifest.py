"""Tests for routed worker artifact manifests."""

from __future__ import annotations

import json
import subprocess
from typing import TYPE_CHECKING

import pytest

from scripts.dev import routed_worker_manifest as manifest

if TYPE_CHECKING:
    from pathlib import Path


def _write_artifacts(run_dir: Path, filenames: list[str]) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    for filename in filenames:
        (run_dir / filename).write_text(f"{filename}\n", encoding="utf-8")


def _init_repo(repo: Path) -> Path:
    """Create a temporary Git checkout for target-worktree validation tests."""
    repo.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "init", "--quiet", str(repo)], check=True)
    return repo


ALL_ARTIFACTS = ["result.json", "RESULT.md", "diffstat.txt", "status.txt", "validation.txt"]


def test_scan_artifact_presence_reports_success_and_missing(tmp_path: Path) -> None:
    """Presence scan should include paths and missing reasons for all artifacts."""
    repo = _init_repo(tmp_path / "repo")
    run_dir = repo / ".git" / "codex-agent-runs" / "run-1"
    _write_artifacts(run_dir, ["result.json", "RESULT.md", "status.txt"])

    presence = manifest.scan_artifact_presence(
        ".git/codex-agent-runs/run-1",
        target_repo=repo,
    )

    assert presence["result_json"].present is True
    assert presence["result_json"].path == ".git/codex-agent-runs/run-1/result.json"
    assert presence["result_json"].size_bytes is not None
    assert presence["diffstat"].present is False
    assert presence["diffstat"].path == ".git/codex-agent-runs/run-1/diffstat.txt"
    assert presence["diffstat"].reason == "missing"
    assert presence["validation"].reason == "missing"


def test_validate_target_worktree_records_canonical_git_paths(tmp_path: Path) -> None:
    """Target validation should record the absolute worktree and common Git directory."""
    repo = _init_repo(tmp_path / "repo")

    check = manifest.validate_target_worktree(repo)

    assert check.ok is True
    assert check.requested_worktree == str(repo.resolve())
    assert check.resolved_worktree == str(repo.resolve())
    assert check.git_top_level == str(repo.resolve())
    assert check.common_git_dir == str((repo / ".git").resolve())


def test_validate_target_worktree_rejects_nested_checkout_path(tmp_path: Path) -> None:
    """A nested path must not be accepted when Git reports a different top-level."""
    repo = _init_repo(tmp_path / "repo")
    nested = repo / "nested"
    nested.mkdir()

    check = manifest.validate_target_worktree(nested)

    assert check.ok is False
    assert check.git_top_level == str(repo.resolve())
    assert "target worktree mismatch" in check.failure


def test_build_manifest_includes_attempts_chosen_route_and_warning(tmp_path: Path) -> None:
    """Manifest should distinguish route evidence from task acceptance."""
    repo = _init_repo(tmp_path / "repo")
    run_dir = repo / ".git" / "codex-agent-runs" / "run-2"
    _write_artifacts(run_dir, ALL_ARTIFACTS)
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

    assert data["schema"] == "routed_worker_manifest.v2"
    assert data["route_evidence_only"] is True
    assert "not task acceptance" in data["warning"]
    assert len(data["attempted_routes"]) == 2
    assert data["attempted_routes"][0]["compact_artifacts"]["validation"]["reason"] == "not-run"
    assert data["chosen_route"] == {"provider": "qwen"}
    assert data["compact_artifacts"]["validation"]["present"] is True


def test_write_manifest_uses_target_repository_run_directory(tmp_path: Path) -> None:
    """Relative run dirs must resolve under target_repo, not the caller cwd."""
    target_repo = _init_repo(tmp_path / "target-repo")
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
    _init_repo(target_repo)
    attempts = [
        {
            "route": {"provider": "qwen"},
            "returncode": 0,
            "failure_class": "none",
            "run_dir": "../outside-run",
        }
    ]

    with pytest.raises(ValueError, match="inside target_repo or the shared Git"):
        manifest.write_routing_manifest(attempts, chosen_index=0, target_repo=target_repo)

    assert not (tmp_path / "outside-run" / "routing_manifest.json").exists()


def test_scan_artifact_presence_rejects_symlink_run_directory(tmp_path: Path) -> None:
    """Symlink-valued run dirs should fail closed before path resolution."""
    target_repo = _init_repo(tmp_path / "target-repo")
    outside_run = tmp_path / "outside-run"
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


# --- Terminal failure classification tests ---


def test_classify_terminal_state_timeout_on_returncode_124() -> None:
    """Returncode 124 is classified as timeout."""
    state = manifest.classify_terminal_state(returncode=124)
    assert state == manifest.TerminalFailure.TIMEOUT


def test_classify_terminal_state_timeout_on_failure_class() -> None:
    """failure_class 'timeout' maps to timeout terminal state."""
    state = manifest.classify_terminal_state(failure_class="timeout")
    assert state == manifest.TerminalFailure.TIMEOUT


def test_classify_terminal_state_exception_on_negative_returncode() -> None:
    """Negative returncode (signal) maps to exception."""
    state = manifest.classify_terminal_state(returncode=-9)
    assert state == manifest.TerminalFailure.EXCEPTION


def test_classify_terminal_state_exception_on_failure_class() -> None:
    """failure_class 'exception' maps to exception terminal state."""
    state = manifest.classify_terminal_state(failure_class="exception")
    assert state == manifest.TerminalFailure.EXCEPTION


def test_classify_terminal_state_non_zero_exit() -> None:
    """Non-zero positive returncode maps to non_zero_exit."""
    state = manifest.classify_terminal_state(returncode=1)
    assert state == manifest.TerminalFailure.NON_ZERO_EXIT


def test_classify_terminal_state_route_not_started() -> None:
    """No run_dir means route was never started."""
    state = manifest.classify_terminal_state(has_run_dir=False)
    assert state == manifest.TerminalFailure.ROUTE_NOT_STARTED


def test_classify_terminal_state_unavailable_when_details_are_absent() -> None:
    """Missing terminal metadata must not be inferred as success."""
    state = manifest.classify_terminal_state()
    assert state == manifest.TerminalFailure.UNAVAILABLE


def test_classify_terminal_state_missing_artifact() -> None:
    """Missing required artifact when run exists maps to missing_artifact."""
    presence = {
        "result_json": manifest.ArtifactPresence(
            present=False, path="result.json", reason="missing", size_bytes=None
        ),
        "result_md": manifest.ArtifactPresence(
            present=True, path="RESULT.md", reason=None, size_bytes=10
        ),
        "diffstat": manifest.ArtifactPresence(
            present=False, path="diffstat.txt", reason="missing", size_bytes=None
        ),
        "status": manifest.ArtifactPresence(
            present=True, path="status.txt", reason=None, size_bytes=5
        ),
        "validation": manifest.ArtifactPresence(
            present=True, path="validation.txt", reason=None, size_bytes=5
        ),
    }
    state = manifest.classify_terminal_state(
        returncode=0, artifact_presence=presence, has_run_dir=True
    )
    assert state == manifest.TerminalFailure.MISSING_ARTIFACT


def test_classify_terminal_state_success() -> None:
    """Zero returncode with all artifacts present maps to none (success)."""
    presence = {
        key: manifest.ArtifactPresence(present=True, path=filename, reason=None, size_bytes=10)
        for key, filename in manifest.REQUIRED_ARTIFACTS.items()
    }
    state = manifest.classify_terminal_state(
        returncode=0, artifact_presence=presence, has_run_dir=True
    )
    assert state == manifest.TerminalFailure.NONE


def test_build_manifest_records_terminal_state_per_attempt(tmp_path: Path) -> None:
    """Each attempt in the manifest carries its own terminal_state."""
    repo = _init_repo(tmp_path / "repo")
    run_dir = repo / ".git" / "codex-agent-runs" / "run-ts"
    _write_artifacts(run_dir, ALL_ARTIFACTS)
    timeout_run_dir = repo / ".git" / "codex-agent-runs" / "run-timeout"
    _write_artifacts(timeout_run_dir, ["result.json", "RESULT.md", "status.txt"])
    attempts = [
        {
            "route": {"provider": "gemini"},
            "returncode": 124,
            "failure_class": "timeout",
            "run_dir": ".git/codex-agent-runs/run-timeout",
        },
        {
            "route": {"provider": "qwen"},
            "returncode": 2,
            "failure_class": "route-collapse",
            "run_dir": None,
        },
        {
            "route": {"provider": "local"},
            "returncode": 0,
            "failure_class": "none",
            "run_dir": ".git/codex-agent-runs/run-ts",
        },
    ]

    data = manifest.build_routing_manifest(attempts, chosen_index=2, target_repo=repo)

    assert data["attempted_routes"][0]["terminal_state"] == "timeout"
    assert data["attempted_routes"][1]["terminal_state"] == "route_not_started"
    assert data["attempted_routes"][2]["terminal_state"] == "none"


# --- Scope and spill detection tests ---


def test_validate_run_dir_scope_ok(tmp_path: Path) -> None:
    """Run dir inside repo resolves with ok=True and no spill."""
    repo = _init_repo(tmp_path / "repo")
    (repo / ".git" / "codex-agent-runs" / "run-1").mkdir(parents=True)
    check = manifest.validate_run_dir_scope(".git/codex-agent-runs/run-1", target_repo=repo)
    assert check.ok is True
    assert check.spill_detected is False
    assert check.failure is None


def test_validate_run_dir_scope_rejects_outside(tmp_path: Path) -> None:
    """Run dir outside repo is detected as scope violation."""
    repo = _init_repo(tmp_path / "repo")
    check = manifest.validate_run_dir_scope("../outside-run", target_repo=repo)
    assert check.ok is False
    assert check.spill_detected is True
    assert "authorized route scope" in check.failure


def test_validate_run_dir_scope_rejects_symlink(tmp_path: Path) -> None:
    """Symlink run dir is fail-closed as scope violation."""
    repo = _init_repo(tmp_path / "repo")
    outside = tmp_path / "outside"
    outside.mkdir()
    (repo / "link").symlink_to(outside, target_is_directory=True)
    check = manifest.validate_run_dir_scope("link", target_repo=repo)
    assert check.ok is False
    assert check.spill_detected is True
    assert "symlink" in check.failure


def test_validate_run_dir_scope_common_root_spill(tmp_path: Path) -> None:
    """Run dir that shares a common root but resolves outside triggers spill."""
    parent = tmp_path / "parent"
    repo = _init_repo(parent / "repo")
    shared = parent / "shared"
    shared.mkdir()
    check = manifest.validate_run_dir_scope("../shared", target_repo=repo)
    assert check.ok is False
    assert check.spill_detected is True


def test_build_manifest_includes_scope_check(tmp_path: Path) -> None:
    """Manifest attempts include scope_check for run_dir attempts."""
    repo = _init_repo(tmp_path / "repo")
    run_dir = repo / ".git" / "codex-agent-runs" / "run-sc"
    _write_artifacts(run_dir, ALL_ARTIFACTS)
    attempts = [
        {
            "route": {"provider": "qwen"},
            "returncode": 0,
            "failure_class": "none",
            "run_dir": ".git/codex-agent-runs/run-sc",
        }
    ]

    data = manifest.build_routing_manifest(attempts, chosen_index=0, target_repo=repo)

    scope = data["attempted_routes"][0]["scope_check"]
    assert scope is not None
    assert scope["ok"] is True
    assert scope["spill_detected"] is False
    assert scope["authorized_root"] == "shared_git_artifacts"
    assert data["target_worktree"]["ok"] is True


def test_validate_run_dir_scope_detects_shared_root_file_spill(tmp_path: Path) -> None:
    """A bundle file beside the assigned run directory is a scope failure."""
    repo = _init_repo(tmp_path / "repo")
    run_dir = repo / ".git" / "codex-agent-runs" / "run-spill"
    _write_artifacts(run_dir, ALL_ARTIFACTS)
    spilled = repo / ".git" / "codex-agent-runs" / "RESULT.md"
    spilled.write_text("wrong bundle\n", encoding="utf-8")

    check = manifest.validate_run_dir_scope(run_dir, target_repo=repo)

    assert check.ok is False
    assert check.spill_detected is True
    assert str(spilled.resolve()) in check.spill_paths
    assert "artifact spill" in check.failure


def test_validate_run_dir_scope_detects_reported_artifact_escape(tmp_path: Path) -> None:
    """Reported artifact paths must remain inside the assigned run bundle."""
    repo = _init_repo(tmp_path / "repo")
    run_dir = repo / ".git" / "codex-agent-runs" / "run-reported"
    run_dir.mkdir(parents=True)

    check = manifest.validate_run_dir_scope(
        run_dir,
        target_repo=repo,
        artifact_paths=["../RESULT.md"],
    )

    assert check.ok is False
    assert check.spill_detected is True
    assert check.spill_paths == (str((run_dir.parent / "RESULT.md").resolve()),)


def test_build_manifest_marks_reported_artifact_spill_as_scope_failure(tmp_path: Path) -> None:
    """Manifest spill evidence must stay failed and route-evidence-only."""
    repo = _init_repo(tmp_path / "repo")
    run_dir = repo / ".git" / "codex-agent-runs" / "run-manifest-spill"
    _write_artifacts(run_dir, ALL_ARTIFACTS)
    data = manifest.build_routing_manifest(
        [
            {
                "route": {"provider": "qwen"},
                "returncode": 0,
                "failure_class": "none",
                "run_dir": str(run_dir),
                "artifact_paths": ["../RESULT.md"],
            }
        ],
        chosen_index=0,
        target_repo=repo,
    )

    attempt = data["attempted_routes"][0]
    assert attempt["terminal_state"] == "scope_violation"
    assert attempt["scope_check"]["spill_paths"]
    assert data["chosen_terminal_state"] == "scope_violation"
    assert data["route_evidence_only"] is True
    json.dumps(data)


def test_build_manifest_scope_check_none_when_no_run_dir(tmp_path: Path) -> None:
    """Attempts without run_dir have scope_check=None."""
    repo = _init_repo(tmp_path / "repo")
    attempts = [
        {
            "route": {"provider": "qwen"},
            "returncode": None,
            "failure_class": "route_not_started",
            "run_dir": None,
        }
    ]

    data = manifest.build_routing_manifest(attempts, chosen_index=0, target_repo=repo)

    assert data["attempted_routes"][0]["scope_check"] is None


def test_build_manifest_scope_violation_on_outside_run_dir(tmp_path: Path) -> None:
    """Out-of-scope run_dir produces scope_check with ok=False."""
    repo = _init_repo(tmp_path / "repo")
    attempts = [
        {
            "route": {"provider": "qwen"},
            "returncode": 0,
            "failure_class": "none",
            "run_dir": "../escape-run",
        }
    ]

    data = manifest.build_routing_manifest(attempts, chosen_index=0, target_repo=repo)

    scope = data["attempted_routes"][0]["scope_check"]
    assert scope is not None
    assert scope["ok"] is False
    assert scope["spill_detected"] is True
    assert data["attempted_routes"][0]["terminal_state"] == "scope_violation"
    for entry in data["attempted_routes"][0]["compact_artifacts"].values():
        assert entry["present"] is False
        assert entry["reason"] == "scope_violation"


# --- Route evidence vs acceptance boundary ---


def test_manifest_route_evidence_only_flag(tmp_path: Path) -> None:
    """Manifest must carry route_evidence_only=True and explicit warning."""
    repo = _init_repo(tmp_path / "repo")
    run_dir = repo / ".git" / "codex-agent-runs" / "run-eva"
    _write_artifacts(run_dir, ALL_ARTIFACTS)
    attempts = [
        {
            "route": {"provider": "qwen"},
            "returncode": 0,
            "failure_class": "none",
            "run_dir": ".git/codex-agent-runs/run-eva",
        }
    ]

    data = manifest.build_routing_manifest(attempts, chosen_index=0, target_repo=repo)

    assert data["route_evidence_only"] is True
    assert "not task acceptance" in data["warning"]


def test_manifest_success_does_not_imply_acceptance(tmp_path: Path) -> None:
    """Successful terminal_state='none' must still carry route_evidence_only."""
    repo = _init_repo(tmp_path / "repo")
    run_dir = repo / ".git" / "codex-agent-runs" / "run-ok"
    _write_artifacts(run_dir, ALL_ARTIFACTS)
    attempts = [
        {
            "route": {"provider": "local"},
            "returncode": 0,
            "failure_class": "none",
            "run_dir": ".git/codex-agent-runs/run-ok",
        }
    ]

    data = manifest.build_routing_manifest(attempts, chosen_index=0, target_repo=repo)

    assert data["attempted_routes"][0]["terminal_state"] == "none"
    assert data["route_evidence_only"] is True
    assert data["compact_artifacts"]["result_json"]["present"] is True
    assert data["aggregation"] == "confirmed"
    assert data["chosen_output_contract"]["status"] == "usable"


def test_manifest_empty_output_and_permission_denial_stay_inconclusive(tmp_path: Path) -> None:
    """A clean exit cannot confirm a route when output is empty or permission-denied."""
    repo = _init_repo(tmp_path / "repo")
    run_dir = repo / ".git" / "codex-agent-runs" / "run-empty"
    _write_artifacts(run_dir, ALL_ARTIFACTS)
    data = manifest.build_routing_manifest(
        [
            {
                "route": {"provider": "antigravity"},
                "returncode": 0,
                "failure_class": "none",
                "run_dir": ".git/codex-agent-runs/run-empty",
                "stdout": "",
                "stderr": "headless command permission denied",
            }
        ],
        chosen_index=0,
        target_repo=repo,
    )

    assert data["attempted_routes"][0]["terminal_state"] == "none"
    assert data["aggregation"] == "inconclusive"
    assert data["aggregation_reason"] == "worker_output_empty"
    assert data["chosen_output_contract"]["missing_evidence"] == [
        "worker_output_empty",
        "permission_denied",
    ]


def test_manifest_explicit_no_findings_stays_inconclusive(tmp_path: Path) -> None:
    """An explicit producer no-findings signal cannot be promoted by artifact presence."""
    repo = _init_repo(tmp_path / "repo")
    run_dir = repo / ".git" / "codex-agent-runs" / "run-no-findings"
    _write_artifacts(run_dir, ALL_ARTIFACTS)
    data = manifest.build_routing_manifest(
        [
            {
                "route": {"provider": "qwen"},
                "returncode": 0,
                "failure_class": "none",
                "run_dir": ".git/codex-agent-runs/run-no-findings",
                "useful_findings": False,
            }
        ],
        chosen_index=0,
        target_repo=repo,
    )

    assert data["aggregation"] == "inconclusive"
    assert data["aggregation_reason"] == "useful_findings_absent"
