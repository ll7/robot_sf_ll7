"""Execute the runner's pre-job trust boundary against event payloads."""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
HOOK = ROOT / "scripts/ci/self_hosted/job_started_hook.sh"


def _event(
    *,
    event_repo: str = "ll7/robot_sf_ll7",
    head_repo: str = "ll7/robot_sf_ll7",
    author: str = "ll7",
) -> dict:
    return {
        "repository": {"full_name": event_repo},
        "pull_request": {
            "head": {"repo": {"full_name": head_repo}},
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
    event_path = tmp_path / "event.json"
    event_path.write_text(json.dumps(payload), encoding="utf-8")
    environment = os.environ.copy()
    environment.update(
        GITHUB_REPOSITORY=changes.get("repository", "ll7/robot_sf_ll7"),
        GITHUB_ACTOR=changes.get("actor", "ll7"),
        GITHUB_TRIGGERING_ACTOR=changes.get("triggering_actor", "ll7"),
        GITHUB_EVENT_NAME=event_name,
        GITHUB_EVENT_PATH=str(event_path),
    )
    completed = subprocess.run(
        ["bash", str(HOOK)], env=environment, capture_output=True, check=False
    )
    assert completed.returncode == expected, completed.stderr.decode()


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
