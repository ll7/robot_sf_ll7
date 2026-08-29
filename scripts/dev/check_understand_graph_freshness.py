#!/usr/bin/env python3
"""Report whether the tracked Understand-Anything graph matches a repository commit."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
from pathlib import Path
from typing import Any

SCHEMA_VERSION = "robot_sf.understand_graph_freshness.v1"
GRAPH_DIR = Path(".understand-anything")
EXPECTED_GRAPH_ARTIFACTS = {
    "meta": GRAPH_DIR / "meta.json",
    "knowledge_graph": GRAPH_DIR / "knowledge-graph.json",
    "fingerprints": GRAPH_DIR / "fingerprints.json",
}
GENERATED_GRAPH_ARTIFACTS = tuple(EXPECTED_GRAPH_ARTIFACTS.values())
GENERATED_GRAPH_EXCLUSIONS = tuple(
    f":(exclude){path.as_posix()}" for path in GENERATED_GRAPH_ARTIFACTS
)


class _GraphCheckError(Exception):
    def __init__(
        self,
        reason_code: str,
        *,
        recorded_commit: str | None = None,
        source_tree_equivalent: bool | None = None,
    ) -> None:
        super().__init__(reason_code)
        self.reason_code = reason_code
        self.recorded_commit = recorded_commit
        self.source_tree_equivalent = source_tree_equivalent


def _git(repo_root: Path, *args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env.update(
        {
            "GIT_NO_LAZY_FETCH": "1",
            "GIT_OPTIONAL_LOCKS": "0",
            "GIT_TERMINAL_PROMPT": "0",
        }
    )
    return subprocess.run(
        ["git", *args],
        cwd=repo_root,
        check=check,
        capture_output=True,
        text=True,
        env=env,
    )


def _read_json_object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def _result(
    *,
    repo_root: Path,
    inspected_commit: str | None,
    authoritative: bool,
    reason_codes: list[str],
    recorded_commit: str | None = None,
    source_tree_equivalent: bool | None = None,
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "authoritativeness": "AUTHORITATIVE" if authoritative else "NON-AUTHORITATIVE",
        "actionability": "ACTIONABLE" if authoritative else "NON-ACTIONABLE",
        "reason_codes": reason_codes,
        "repository": {
            "root": str(repo_root),
            "inspected_commit": inspected_commit,
        },
        "graph": {
            "directory": str(GRAPH_DIR),
            "recorded_commit": recorded_commit,
            "source_tree_equivalent": source_tree_equivalent,
        },
    }


def _require_clean_paths(
    repo_root: Path,
    *,
    pathspecs: tuple[str, ...],
    reason_code: str,
    recorded_commit: str | None = None,
) -> None:
    status = _git(
        repo_root,
        "status",
        "--porcelain",
        "--untracked-files=all",
        "--",
        *pathspecs,
    )
    if status.stdout.strip():
        raise _GraphCheckError(reason_code, recorded_commit=recorded_commit)


def _load_artifacts(repo_root: Path) -> dict[str, dict[str, Any]]:
    for artifact_name, relative_path in EXPECTED_GRAPH_ARTIFACTS.items():
        path = repo_root / relative_path
        if path.is_symlink():
            raise _GraphCheckError(f"{artifact_name}_not_regular")
        if not path.is_file():
            raise _GraphCheckError(f"{artifact_name}_missing")
        index_entry = _git(
            repo_root,
            "ls-files",
            "--stage",
            "--",
            relative_path.as_posix(),
            check=False,
        )
        if index_entry.returncode != 0:
            raise _GraphCheckError("git_index_check_failed")
        entries = index_entry.stdout.splitlines()
        if len(entries) != 1:
            raise _GraphCheckError(f"{artifact_name}_not_tracked")
        entry_metadata, _, listed_path = entries[0].partition("\t")
        metadata_fields = entry_metadata.split()
        if (
            listed_path != relative_path.as_posix()
            or len(metadata_fields) != 3
            or metadata_fields[0] not in {"100644", "100755"}
            or metadata_fields[2] != "0"
        ):
            raise _GraphCheckError(f"{artifact_name}_not_regular")

    artifacts: dict[str, dict[str, Any]] = {}
    for artifact_name, relative_path in EXPECTED_GRAPH_ARTIFACTS.items():
        path = repo_root / relative_path
        try:
            artifacts[artifact_name] = _read_json_object(path)
        except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError):
            raise _GraphCheckError(f"{artifact_name}_malformed") from None
    return artifacts


def _recorded_commit(artifacts: dict[str, dict[str, Any]], inspected_commit: str) -> str:
    meta = artifacts["meta"]
    raw_recorded_commit = meta.get("gitCommitHash")
    if not isinstance(raw_recorded_commit, str) or not raw_recorded_commit.strip():
        raise _GraphCheckError("meta_commit_missing")
    recorded_commit = raw_recorded_commit.strip()
    if (
        len(recorded_commit) != len(inspected_commit)
        or re.fullmatch(r"[0-9a-f]+", recorded_commit) is None
    ):
        raise _GraphCheckError(
            "meta_commit_malformed",
            recorded_commit=recorded_commit,
        )
    return recorded_commit


def _require_consistent_artifact_commits(
    artifacts: dict[str, dict[str, Any]], recorded_commit: str
) -> None:
    graph = artifacts["knowledge_graph"]
    fingerprints = artifacts["fingerprints"]
    project = graph.get("project")
    raw_graph_commit = project.get("gitCommitHash") if isinstance(project, dict) else None
    if not isinstance(raw_graph_commit, str) or not raw_graph_commit.strip():
        raise _GraphCheckError(
            "knowledge_graph_commit_missing",
            recorded_commit=recorded_commit,
        )
    raw_fingerprints_commit = fingerprints.get("gitCommitHash")
    if not isinstance(raw_fingerprints_commit, str) or not raw_fingerprints_commit.strip():
        raise _GraphCheckError(
            "fingerprints_commit_missing",
            recorded_commit=recorded_commit,
        )
    graph_commit = raw_graph_commit.strip()
    fingerprints_commit = raw_fingerprints_commit.strip()
    if graph_commit != recorded_commit or fingerprints_commit != recorded_commit:
        raise _GraphCheckError(
            "artifact_commit_mismatch",
            recorded_commit=recorded_commit,
        )


def _require_resolvable_commit(repo_root: Path, recorded_commit: str) -> None:
    check = _git(
        repo_root,
        "rev-parse",
        "--verify",
        "--quiet",
        f"{recorded_commit}^{{commit}}",
        check=False,
    )
    if check.returncode != 0 or check.stdout.strip() != recorded_commit:
        raise _GraphCheckError(
            "recorded_commit_unresolvable",
            recorded_commit=recorded_commit,
        )


def _source_tree_matches(repo_root: Path, *, recorded_commit: str, inspected_commit: str) -> bool:
    comparison = _git(
        repo_root,
        "diff",
        "--quiet",
        recorded_commit,
        inspected_commit,
        "--",
        ".",
        *GENERATED_GRAPH_EXCLUSIONS,
        check=False,
    )
    if comparison.returncode == 0:
        return True
    if comparison.returncode == 1:
        return False
    raise _GraphCheckError("git_comparison_failed", recorded_commit=recorded_commit)


def _authoritative_result(repo_root: Path) -> dict[str, Any]:
    inspected_commit = _git(repo_root, "rev-parse", "HEAD^{commit}").stdout.strip()
    recorded_commit: str | None = None
    try:
        _require_clean_paths(
            repo_root,
            pathspecs=(str(GRAPH_DIR),),
            reason_code="graph_artifacts_dirty",
        )
        artifacts = _load_artifacts(repo_root)
        recorded_commit = _recorded_commit(artifacts, inspected_commit)
        _require_consistent_artifact_commits(artifacts, recorded_commit)
        _require_clean_paths(
            repo_root,
            pathspecs=(".", *GENERATED_GRAPH_EXCLUSIONS),
            reason_code="working_tree_dirty",
            recorded_commit=recorded_commit,
        )
        _require_resolvable_commit(repo_root, recorded_commit)
        if not _source_tree_matches(
            repo_root,
            recorded_commit=recorded_commit,
            inspected_commit=inspected_commit,
        ):
            raise _GraphCheckError(
                "source_tree_mismatch",
                recorded_commit=recorded_commit,
                source_tree_equivalent=False,
            )
    except _GraphCheckError as exc:
        return _result(
            repo_root=repo_root,
            inspected_commit=inspected_commit,
            authoritative=False,
            reason_codes=[exc.reason_code],
            recorded_commit=exc.recorded_commit,
            source_tree_equivalent=exc.source_tree_equivalent,
        )

    return _result(
        repo_root=repo_root,
        inspected_commit=inspected_commit,
        authoritative=True,
        reason_codes=[],
        recorded_commit=recorded_commit,
        source_tree_equivalent=True,
    )


def main() -> int:
    """Run the graph freshness check and return its fail-closed exit status."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path.cwd(),
        help="Git repository to inspect (default: current directory)",
    )
    args = parser.parse_args()
    repo_root = args.repo_root.resolve()
    try:
        payload = _authoritative_result(repo_root)
    except (OSError, subprocess.CalledProcessError):
        payload = _result(
            repo_root=repo_root,
            inspected_commit=None,
            authoritative=False,
            reason_codes=["repository_unavailable"],
        )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if payload["authoritativeness"] == "AUTHORITATIVE" else 1


if __name__ == "__main__":
    raise SystemExit(main())
