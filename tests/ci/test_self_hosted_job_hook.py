"""Execute the runner's pre-job trust boundary against event payloads."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[2]
HOOK = ROOT / "scripts/ci/self_hosted/job_started_hook.sh"
BASE = "1" * 40
OTHER = "2" * 40
HEAD = "3" * 40
COMPARE = f"compare/{BASE}...{HEAD}?per_page=100&page=1"


def _commit(sha: str = HEAD, *, author: str = "ll7", committer: str = "ll7") -> dict:
    """REST account identities, independent of webhook commit names/emails."""
    return {
        "sha": sha,
        "author": {"login": author, "type": "Bot" if "[bot]" in author else "User"},
        "committer": {"login": committer, "type": "User"},
    }


def _comparison(commits: list[dict] | None = None, *, total: int | None = None) -> dict:
    """A complete REST comparison of fixed immutable endpoints."""
    commits = [_commit()] if commits is None else commits
    return {
        "base_commit": {"sha": BASE},
        "status": "ahead",
        "total_commits": len(commits) if total is None else total,
        "commits": commits,
    }


def _run_hook(
    tmp_path: Path,
    *,
    event_name: str = "pull_request",
    payload: dict | None = None,
    responses: dict[str, Any] | None = None,
    route: bool = False,
    changes: dict[str, str] | None = None,
) -> subprocess.CompletedProcess:
    """Execute the production boundary with an offline REST transport."""
    changes = changes or {}
    event_path = tmp_path / "event.json"
    event_path.write_text(json.dumps(_event() if payload is None else payload), encoding="utf-8")
    fixtures = tmp_path / "responses.json"
    fixtures.write_text(
        json.dumps({COMPARE: _comparison()} if responses is None else responses), encoding="utf-8"
    )
    curl = tmp_path / "curl"
    curl.write_text(
        f"#!{sys.executable}\n"
        + r"""import json, os, sys
from pathlib import Path
url = sys.argv[-1]
prefix = "https://api.github.com/repos/ll7/robot_sf_ll7/"
if not url.startswith(prefix):
    sys.exit(99)
endpoint = url[len(prefix):]
with open(os.environ["API_CALLS"], "a") as log:
    log.write(endpoint + "\n")
responses = json.loads(Path(os.environ["API_FIXTURES"]).read_text())
if endpoint not in responses:
    sys.exit(22)
value = responses[endpoint]
if isinstance(value, dict) and "_sequence" in value:
    counter_path = Path(os.environ["API_FIXTURES"] + ".counts")
    counts = json.loads(counter_path.read_text()) if counter_path.exists() else {}
    count = counts.get(endpoint, 0)
    counts[endpoint] = count + 1
    counter_path.write_text(json.dumps(counts))
    value = value["_sequence"][count % len(value["_sequence"])]
if isinstance(value, dict) and "_raw" in value:
    print(value["_raw"])
else:
    print(json.dumps(value))
""",
        encoding="utf-8",
    )
    curl.chmod(0o755)
    environment = {
        "PATH": f"{tmp_path}:{os.environ['PATH']}",
        "API_FIXTURES": str(fixtures),
        "API_CALLS": str(tmp_path / "api-calls.txt"),
        "GITHUB_REPOSITORY": changes.get("repository", "ll7/robot_sf_ll7"),
        "GITHUB_ACTOR": changes.get("actor", "ll7"),
        "GITHUB_TRIGGERING_ACTOR": changes.get("triggering_actor", "ll7"),
        "GITHUB_EVENT_NAME": event_name,
        "GITHUB_EVENT_PATH": str(event_path),
        "GITHUB_SHA": changes.get("sha", HEAD),
    }
    return subprocess.run(
        ["bash", str(HOOK), *(["--route"] if route else [])],
        env=environment,
        capture_output=True,
        text=True,
        check=False,
        timeout=10,
    )


def _event(
    *,
    event_repo: str = "ll7/robot_sf_ll7",
    head_repo: str = "ll7/robot_sf_ll7",
    author: str = "ll7",
) -> dict:
    return {
        "repository": {"full_name": event_repo},
        "before": BASE,
        "after": HEAD,
        "pull_request": {
            "number": 42,
            "base": {"sha": BASE},
            "head": {"sha": HEAD, "repo": {"full_name": head_repo}},
            "user": {"login": author},
        },
    }


@pytest.mark.parametrize(
    ("event_name", "changes", "expected"),
    [
        ("push", {}, 0),
        ("push", {"repository": "other/repo"}, 1),
        ("push", {"event_repo": "other/repo"}, 1),
        ("pull_request", {}, 0),
        ("pull_request", {"head_repo": "outsider/robot_sf_ll7"}, 1),
        ("pull_request", {"author": "outsider"}, 1),
        ("pull_request", {"actor": "dependabot[bot]"}, 1),
        pytest.param(
            "pull_request",
            {"triggering_actor": "maintainer"},
            1,
            id="third-party-rerun",
        ),
        ("pull_request_target", {}, 1),
        ("workflow_dispatch", {}, 1),
        ("issue_comment", {}, 1),
    ],
)
def test_job_hook_rejects_untrusted_events(
    tmp_path: Path, event_name: str, changes: dict[str, str], expected: int
) -> None:
    payload = _event(
        event_repo=changes.get("event_repo", "ll7/robot_sf_ll7"),
        head_repo=changes.get("head_repo", "ll7/robot_sf_ll7"),
        author=changes.get("author", "ll7"),
    )
    completed = _run_hook(tmp_path, event_name=event_name, payload=payload, changes=changes)
    assert completed.returncode == expected, completed.stderr


def test_job_hook_rejects_missing_or_malformed_event(tmp_path: Path) -> None:
    event_path = tmp_path / "event.json"
    environment = os.environ.copy()
    environment.update(
        GITHUB_REPOSITORY="ll7/robot_sf_ll7",
        GITHUB_ACTOR="ll7",
        GITHUB_TRIGGERING_ACTOR="ll7",
        GITHUB_EVENT_NAME="push",
        GITHUB_EVENT_PATH=str(event_path),
    )
    for contents in (None, "not json"):
        if contents is not None:
            event_path.write_text(contents, encoding="utf-8")
        completed = subprocess.run(
            ["bash", str(HOOK)], env=environment, capture_output=True, check=False
        )
        assert completed.returncode == 1
