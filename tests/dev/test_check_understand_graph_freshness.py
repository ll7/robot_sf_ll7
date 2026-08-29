"""Behavior tests for the read-only Understand-Anything graph freshness check."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

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


def _run_check(
    repo: Path, *, env: dict[str, str] | None = None
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(SCRIPT), "--repo-root", str(repo)],
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )


def _object_db_snapshot(repo: Path) -> dict[str, str]:
    git_dir = Path(_git(repo, "rev-parse", "--absolute-git-dir"))
    object_dir = git_dir / "objects"
    return {
        str(path.relative_to(object_dir)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(object_dir.rglob("*"))
        if path.is_file()
    }


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


@pytest.mark.parametrize(
    ("relative_path", "content"),
    [
        (".understand-anything/config.json", '{"autoUpdate": false}\n'),
        (".understand-anything/.understandignore", "tmp/\n"),
        (".understand-anything/unexpected.json", '{"unexpected": true}\n'),
    ],
)
def test_committed_graph_control_change_is_source_bound(
    tmp_path: Path, relative_path: str, content: str
) -> None:
    """Graph control and unexpected tracked files remain part of the inspected source tree."""
    repo, source_commit = _graph_repo(tmp_path)
    changed_path = repo / relative_path
    changed_path.write_text(content, encoding="utf-8")
    _git(repo, "add", relative_path)
    _git(repo, "commit", "-q", "-m", "change graph control input")

    result = _run_check(repo)

    assert result.returncode == 1
    payload = json.loads(result.stdout)
    assert payload["authoritativeness"] == "NON-AUTHORITATIVE"
    assert payload["actionability"] == "NON-ACTIONABLE"
    assert payload["reason_codes"] == ["source_tree_mismatch"]
    assert payload["graph"]["recorded_commit"] == source_commit
    assert payload["graph"]["source_tree_equivalent"] is False


def test_git_comparison_error_is_not_reported_as_source_mismatch(tmp_path: Path) -> None:
    """A Git execution error must deny use without claiming that source content was compared."""
    repo, source_commit = _graph_repo(tmp_path)
    real_git = shutil.which("git")
    assert real_git is not None
    wrapper_dir = tmp_path / "bin"
    wrapper_dir.mkdir()
    git_wrapper = wrapper_dir / "git"
    git_wrapper.write_text(
        "#!/bin/sh\n"
        'if [ "$1" = "diff" ] && [ "$2" = "--quiet" ]; then\n'
        "  exit 2\n"
        "fi\n"
        'exec "$REAL_GIT" "$@"\n',
        encoding="utf-8",
    )
    git_wrapper.chmod(0o755)
    check_env = os.environ.copy()
    check_env.update(
        {
            "PATH": f"{wrapper_dir}{os.pathsep}{check_env['PATH']}",
            "REAL_GIT": real_git,
        }
    )

    result = _run_check(repo, env=check_env)

    assert result.returncode == 1
    payload = json.loads(result.stdout)
    assert payload["authoritativeness"] == "NON-AUTHORITATIVE"
    assert payload["actionability"] == "NON-ACTIONABLE"
    assert payload["reason_codes"] == ["git_comparison_failed"]
    assert payload["graph"]["recorded_commit"] == source_commit
    assert payload["graph"]["source_tree_equivalent"] is None


def test_every_git_subprocess_disables_lazy_fetch_and_optional_locks(
    tmp_path: Path,
) -> None:
    """Every checker Git call must inherit the same read-only, non-interactive guardrails."""
    repo, _ = _graph_repo(tmp_path)
    real_git = shutil.which("git")
    assert real_git is not None
    wrapper_dir = tmp_path / "bin"
    wrapper_dir.mkdir()
    git_wrapper = wrapper_dir / "git"
    git_wrapper.write_text(
        "#!/bin/sh\n"
        "printf '%s\\t%s\\t%s\\t%s\\n' "
        '"$GIT_NO_LAZY_FETCH" "$GIT_OPTIONAL_LOCKS" '
        '"$GIT_TERMINAL_PROMPT" "$*" >> "$GIT_ENV_LOG"\n'
        'exec "$REAL_GIT" "$@"\n',
        encoding="utf-8",
    )
    git_wrapper.chmod(0o755)
    git_env_log = tmp_path / "git-env.log"
    check_env = os.environ.copy()
    check_env.update(
        {
            "PATH": f"{wrapper_dir}{os.pathsep}{check_env['PATH']}",
            "REAL_GIT": real_git,
            "GIT_ENV_LOG": str(git_env_log),
            "GIT_NO_LAZY_FETCH": "0",
            "GIT_OPTIONAL_LOCKS": "1",
            "GIT_TERMINAL_PROMPT": "1",
        }
    )

    result = _run_check(repo, env=check_env)

    assert result.returncode == 0, result.stderr
    invocations = [line.split("\t", maxsplit=3) for line in git_env_log.read_text().splitlines()]
    assert invocations
    assert {tuple(fields[:3]) for fields in invocations} == {("1", "0", "0")}
    assert {fields[3].split(maxsplit=1)[0] for fields in invocations} == {
        "diff",
        "ls-files",
        "rev-parse",
        "status",
    }


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


@pytest.mark.parametrize(
    ("artifact_name", "artifact_relative"),
    [
        ("meta", ".understand-anything/meta.json"),
        ("knowledge_graph", ".understand-anything/knowledge-graph.json"),
        ("fingerprints", ".understand-anything/fingerprints.json"),
    ],
)
def test_ignored_untracked_graph_artifact_is_non_authoritative(
    tmp_path: Path, artifact_name: str, artifact_relative: str
) -> None:
    """A valid-looking ignored file cannot substitute for any tracked shared graph artifact."""
    repo, _ = _graph_repo(tmp_path)
    _git(repo, "rm", "--cached", artifact_relative)
    _git(repo, "commit", "-q", "-m", "stop tracking graph")
    git_dir = Path(_git(repo, "rev-parse", "--absolute-git-dir"))
    info_exclude = git_dir / "info" / "exclude"
    with info_exclude.open("a", encoding="utf-8") as handle:
        handle.write(f"\n/{artifact_relative}\n")
    assert _git(repo, "status", "--porcelain", "--untracked-files=all") == ""

    result = _run_check(repo)

    assert result.returncode == 1
    payload = json.loads(result.stdout)
    assert payload["authoritativeness"] == "NON-AUTHORITATIVE"
    assert payload["actionability"] == "NON-ACTIONABLE"
    assert payload["reason_codes"] == [f"{artifact_name}_not_tracked"]
    assert payload["graph"]["recorded_commit"] is None


@pytest.mark.parametrize(
    ("artifact_name", "artifact_relative", "external_payload"),
    [
        ("meta", ".understand-anything/meta.json", "meta"),
        ("knowledge_graph", ".understand-anything/knowledge-graph.json", "graph"),
        ("fingerprints", ".understand-anything/fingerprints.json", "fingerprints"),
    ],
)
def test_tracked_symlink_graph_artifact_is_non_authoritative(
    tmp_path: Path,
    artifact_name: str,
    artifact_relative: str,
    external_payload: str,
) -> None:
    """A tracked symlink cannot substitute external bytes for a shared graph artifact."""
    repo, source_commit = _graph_repo(tmp_path)
    payloads = {
        "meta": {"gitCommitHash": source_commit},
        "graph": {"version": "1.0.0", "project": {"gitCommitHash": source_commit}},
        "fingerprints": {"gitCommitHash": source_commit, "files": {}},
    }
    external_graph = tmp_path / f"external-{artifact_name}.json"
    _write_json(external_graph, payloads[external_payload])
    artifact_path = repo / artifact_relative
    artifact_path.unlink()
    artifact_path.symlink_to(external_graph)
    _git(repo, "add", artifact_relative)
    _git(repo, "commit", "-q", "-m", "replace graph with symlink")
    assert _git(repo, "status", "--porcelain", "--untracked-files=all") == ""

    result = _run_check(repo)

    assert result.returncode == 1
    payload = json.loads(result.stdout)
    assert payload["authoritativeness"] == "NON-AUTHORITATIVE"
    assert payload["actionability"] == "NON-ACTIONABLE"
    assert payload["reason_codes"] == [f"{artifact_name}_not_regular"]
    assert payload["graph"]["recorded_commit"] is None


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


def test_partial_clone_does_not_fetch_or_write_missing_recorded_commit(tmp_path: Path) -> None:
    """A missing promisor object must deny locally without transport use or object-db mutation."""
    source_repo, source_commit = _graph_repo(tmp_path)
    _git(source_repo, "config", "uploadpack.allowFilter", "true")
    partial_repo = tmp_path / "partial"
    clone_env = os.environ.copy()
    # The fixture intentionally transfers objects from a local file:// source. An outer
    # fail-closed validation environment must not disable construction of the promisor clone.
    clone_env.pop("GIT_NO_LAZY_FETCH", None)
    subprocess.run(
        [
            "git",
            "clone",
            "--quiet",
            "--depth=1",
            "--filter=blob:none",
            source_repo.as_uri(),
            str(partial_repo),
        ],
        check=True,
        capture_output=True,
        text=True,
        env=clone_env,
    )
    assert _git(partial_repo, "config", "--get", "remote.origin.promisor") == "true"

    transport_marker = tmp_path / "transport-invoked"
    ssh_sentinel = tmp_path / "deny-transport.sh"
    ssh_sentinel.write_text(
        '#!/bin/sh\n: > "$NETWORK_SENTINEL"\nexit 97\n',
        encoding="utf-8",
    )
    ssh_sentinel.chmod(0o755)
    _git(partial_repo, "remote", "set-url", "origin", "ssh://example.invalid/repo")
    check_env = os.environ.copy()
    check_env.update(
        {
            "GIT_SSH_COMMAND": str(ssh_sentinel),
            "GIT_TERMINAL_PROMPT": "0",
            "NETWORK_SENTINEL": str(transport_marker),
        }
    )
    before_objects = _object_db_snapshot(partial_repo)

    result = _run_check(partial_repo, env=check_env)

    assert result.returncode == 1
    payload = json.loads(result.stdout)
    assert payload["authoritativeness"] == "NON-AUTHORITATIVE"
    assert payload["actionability"] == "NON-ACTIONABLE"
    assert payload["reason_codes"] == ["recorded_commit_unresolvable"]
    assert payload["graph"]["recorded_commit"] == source_commit
    assert not transport_marker.exists()
    assert _object_db_snapshot(partial_repo) == before_objects


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
