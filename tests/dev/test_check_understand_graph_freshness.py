"""Behavior tests for the read-only Understand-Anything graph freshness check."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from tests.support.environment_guards import configure_git_identity

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "scripts" / "dev" / "check_understand_graph_freshness.py"


def _git(repo: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")


def _graph_repo(tmp_path: Path) -> tuple[Path, str]:
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q")
    configure_git_identity(repo, name="Test Agent", email="test@example.invalid")
    (repo / "source.py").write_text("VALUE = 1\n", encoding="utf-8")
    _git(repo, "add", "source.py")
    _git(repo, "commit", "-q", "-m", "source")
    source_commit = _git(repo, "rev-parse", "HEAD")

    graph_dir = repo / ".understand-anything"
    _write_json(graph_dir / "meta.json", {"gitCommitHash": source_commit})
    _write_json(
        graph_dir / "knowledge-graph.json",
        {"version": "1.0.0", "project": {"gitCommitHash": source_commit}},
    )
    _write_json(
        graph_dir / "fingerprints.json",
        {"version": "1.0.0", "gitCommitHash": source_commit, "files": {}},
    )
    _git(repo, "add", ".understand-anything")
    _git(repo, "commit", "-q", "-m", "graph artifacts")
    return repo, source_commit


def _run_check(repo: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(SCRIPT), "--repo-root", str(repo)],
        check=False,
        capture_output=True,
        text=True,
    )


def test_artifact_only_commit_is_authoritative_when_source_tree_is_equivalent(
    tmp_path: Path,
) -> None:
    """A graph committed after its source commit stays usable when source content is unchanged."""
    repo, source_commit = _graph_repo(tmp_path)

    result = _run_check(repo)

    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["schema_version"] == "robot_sf.understand_graph_freshness.v1"
    assert payload["authoritativeness"] == "AUTHORITATIVE"
    assert payload["actionability"] == "ACTIONABLE"
    assert payload["reason_codes"] == []
    assert payload["graph"]["recorded_commit"] == source_commit
    assert payload["graph"]["source_tree_equivalent"] is True
    assert _git(repo, "status", "--porcelain") == ""


def test_source_change_is_explicitly_non_authoritative_and_non_actionable(
    tmp_path: Path,
) -> None:
    """A stale graph must fail closed so agents cannot treat old structure as current proof."""
    repo, source_commit = _graph_repo(tmp_path)
    (repo / "source.py").write_text("VALUE = 2\n", encoding="utf-8")
    _git(repo, "add", "source.py")
    _git(repo, "commit", "-q", "-m", "change source")

    result = _run_check(repo)

    assert result.returncode == 1
    payload = json.loads(result.stdout)
    assert payload["authoritativeness"] == "NON-AUTHORITATIVE"
    assert payload["actionability"] == "NON-ACTIONABLE"
    assert payload["reason_codes"] == ["source_tree_mismatch"]
    assert payload["graph"]["recorded_commit"] == source_commit
    assert payload["graph"]["source_tree_equivalent"] is False


def test_uncommitted_source_change_is_non_authoritative(tmp_path: Path) -> None:
    """A graph cannot authorize work against source edits that no inspected commit represents."""
    repo, source_commit = _graph_repo(tmp_path)
    (repo / "source.py").write_text("VALUE = 2\n", encoding="utf-8")

    result = _run_check(repo)

    assert result.returncode == 1
    payload = json.loads(result.stdout)
    assert payload["authoritativeness"] == "NON-AUTHORITATIVE"
    assert payload["actionability"] == "NON-ACTIONABLE"
    assert payload["reason_codes"] == ["working_tree_dirty"]
    assert payload["graph"]["recorded_commit"] == source_commit
    assert payload["graph"]["source_tree_equivalent"] is None


def test_uncommitted_graph_change_is_non_authoritative(tmp_path: Path) -> None:
    """Locally altered graph bytes must never be trusted as the tracked shared artifact."""
    repo, _ = _graph_repo(tmp_path)
    graph_path = repo / ".understand-anything" / "knowledge-graph.json"
    graph = json.loads(graph_path.read_text(encoding="utf-8"))
    graph["nodes"] = [{"id": "uncommitted"}]
    _write_json(graph_path, graph)

    result = _run_check(repo)

    assert result.returncode == 1
    payload = json.loads(result.stdout)
    assert payload["authoritativeness"] == "NON-AUTHORITATIVE"
    assert payload["actionability"] == "NON-ACTIONABLE"
    assert payload["reason_codes"] == ["graph_artifacts_dirty"]
    assert payload["graph"]["recorded_commit"] is None
    assert payload["graph"]["source_tree_equivalent"] is None


def test_missing_knowledge_graph_is_non_authoritative(tmp_path: Path) -> None:
    """A missing graph must produce a machine-readable denial instead of a traceback."""
    repo, _ = _graph_repo(tmp_path)
    (repo / ".understand-anything" / "knowledge-graph.json").unlink()
    _git(repo, "add", ".understand-anything/knowledge-graph.json")
    _git(repo, "commit", "-q", "-m", "remove graph")

    result = _run_check(repo)

    assert result.returncode == 1
    payload = json.loads(result.stdout)
    assert payload["authoritativeness"] == "NON-AUTHORITATIVE"
    assert payload["actionability"] == "NON-ACTIONABLE"
    assert payload["reason_codes"] == ["knowledge_graph_missing"]
    assert payload["graph"]["source_tree_equivalent"] is None


def test_malformed_knowledge_graph_is_non_authoritative(tmp_path: Path) -> None:
    """Malformed graph JSON must be denied with a stable code that automation can gate on."""
    repo, _ = _graph_repo(tmp_path)
    graph_path = repo / ".understand-anything" / "knowledge-graph.json"
    graph_path.write_text("{not json\n", encoding="utf-8")
    _git(repo, "add", ".understand-anything/knowledge-graph.json")
    _git(repo, "commit", "-q", "-m", "malform graph")

    result = _run_check(repo)

    assert result.returncode == 1
    payload = json.loads(result.stdout)
    assert payload["authoritativeness"] == "NON-AUTHORITATIVE"
    assert payload["actionability"] == "NON-ACTIONABLE"
    assert payload["reason_codes"] == ["knowledge_graph_malformed"]
    assert payload["graph"]["source_tree_equivalent"] is None


def test_metadata_without_source_commit_is_non_authoritative(tmp_path: Path) -> None:
    """Metadata without a source identity cannot authorize graph-guided action."""
    repo, _ = _graph_repo(tmp_path)
    _write_json(repo / ".understand-anything" / "meta.json", {"version": "1.0.0"})
    _git(repo, "add", ".understand-anything/meta.json")
    _git(repo, "commit", "-q", "-m", "drop graph source commit")

    result = _run_check(repo)

    assert result.returncode == 1
    payload = json.loads(result.stdout)
    assert payload["authoritativeness"] == "NON-AUTHORITATIVE"
    assert payload["actionability"] == "NON-ACTIONABLE"
    assert payload["reason_codes"] == ["meta_commit_missing"]
    assert payload["graph"]["recorded_commit"] is None


def test_malformed_source_commit_is_non_authoritative(tmp_path: Path) -> None:
    """A non-identity metadata value must fail before any Git tree comparison is trusted."""
    repo, _ = _graph_repo(tmp_path)
    _write_json(repo / ".understand-anything" / "meta.json", {"gitCommitHash": "not-a-sha"})
    _git(repo, "add", ".understand-anything/meta.json")
    _git(repo, "commit", "-q", "-m", "malform graph source commit")

    result = _run_check(repo)

    assert result.returncode == 1
    payload = json.loads(result.stdout)
    assert payload["authoritativeness"] == "NON-AUTHORITATIVE"
    assert payload["actionability"] == "NON-ACTIONABLE"
    assert payload["reason_codes"] == ["meta_commit_malformed"]
    assert payload["graph"]["recorded_commit"] == "not-a-sha"


def test_unresolvable_source_commit_is_non_authoritative(tmp_path: Path) -> None:
    """A well-shaped but unavailable source commit cannot support a freshness comparison."""
    repo, _ = _graph_repo(tmp_path)
    unavailable_commit = "0" * 40
    graph_dir = repo / ".understand-anything"
    _write_json(graph_dir / "meta.json", {"gitCommitHash": unavailable_commit})
    _write_json(
        graph_dir / "knowledge-graph.json",
        {"version": "1.0.0", "project": {"gitCommitHash": unavailable_commit}},
    )
    _write_json(
        graph_dir / "fingerprints.json",
        {"version": "1.0.0", "gitCommitHash": unavailable_commit, "files": {}},
    )
    _git(repo, "add", ".understand-anything")
    _git(repo, "commit", "-q", "-m", "record unavailable source commit")

    result = _run_check(repo)

    assert result.returncode == 1
    payload = json.loads(result.stdout)
    assert payload["authoritativeness"] == "NON-AUTHORITATIVE"
    assert payload["actionability"] == "NON-ACTIONABLE"
    assert payload["reason_codes"] == ["recorded_commit_unresolvable"]
    assert payload["graph"]["recorded_commit"] == unavailable_commit
    assert payload["graph"]["source_tree_equivalent"] is None


def test_disagreeing_artifact_commit_identities_are_non_authoritative(tmp_path: Path) -> None:
    """All tracked graph companions must bind to one source commit before use."""
    repo, source_commit = _graph_repo(tmp_path)
    _write_json(
        repo / ".understand-anything" / "fingerprints.json",
        {"gitCommitHash": "f" * 40, "files": {}},
    )
    _git(repo, "add", ".understand-anything/fingerprints.json")
    _git(repo, "commit", "-q", "-m", "desynchronize graph companions")

    result = _run_check(repo)

    assert result.returncode == 1
    payload = json.loads(result.stdout)
    assert payload["authoritativeness"] == "NON-AUTHORITATIVE"
    assert payload["actionability"] == "NON-ACTIONABLE"
    assert payload["reason_codes"] == ["artifact_commit_mismatch"]
    assert payload["graph"]["recorded_commit"] == source_commit
    assert payload["graph"]["source_tree_equivalent"] is None


def test_graph_without_embedded_source_commit_is_non_authoritative(tmp_path: Path) -> None:
    """A parseable graph without its source binding is semantically malformed and unusable."""
    repo, source_commit = _graph_repo(tmp_path)
    _write_json(
        repo / ".understand-anything" / "knowledge-graph.json",
        {"version": "1.0.0", "project": {}},
    )
    _git(repo, "add", ".understand-anything/knowledge-graph.json")
    _git(repo, "commit", "-q", "-m", "drop graph source binding")

    result = _run_check(repo)

    assert result.returncode == 1
    payload = json.loads(result.stdout)
    assert payload["authoritativeness"] == "NON-AUTHORITATIVE"
    assert payload["actionability"] == "NON-ACTIONABLE"
    assert payload["reason_codes"] == ["knowledge_graph_commit_missing"]
    assert payload["graph"]["recorded_commit"] == source_commit


def test_non_repository_path_fails_closed_with_json(tmp_path: Path) -> None:
    """Automation must receive a parseable denial when no repository commit can be inspected."""
    result = _run_check(tmp_path)

    assert result.returncode == 1
    payload = json.loads(result.stdout)
    assert payload["authoritativeness"] == "NON-AUTHORITATIVE"
    assert payload["actionability"] == "NON-ACTIONABLE"
    assert payload["reason_codes"] == ["repository_unavailable"]
    assert payload["repository"]["inspected_commit"] is None
