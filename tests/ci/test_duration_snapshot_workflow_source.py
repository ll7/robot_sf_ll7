"""Run the duration-freezing workflow step with an older PR checkout."""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize("helper_available", [True, False])
def test_duration_snapshot_uses_workflow_revision_on_older_pr(
    tmp_path: Path, helper_available: bool
) -> None:
    """New workflow arguments must use their matching helper, even on an old PR."""
    workflow = yaml.safe_load((ROOT / ".github/workflows/ci.yml").read_text())
    step = next(
        step
        for step in workflow["jobs"]["dispatch-ownership"]["steps"]
        if step.get("name") == "Freeze matrix duration input"
    )
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    runner_temp = tmp_path / "runner"
    runner_temp.mkdir()
    env = dict(os.environ)
    env.update(
        GIT_AUTHOR_NAME="Test",
        GIT_AUTHOR_EMAIL="test@example.invalid",
        GIT_COMMITTER_NAME="Test",
        GIT_COMMITTER_EMAIL="test@example.invalid",
        RUNNER_TEMP=str(runner_temp),
    )

    def git(*args: str) -> str:
        result = subprocess.run(
            ["git", *args], cwd=checkout, env=env, capture_output=True, text=True, check=True
        )
        return result.stdout.strip()

    git("init", "-q")
    helper = checkout / "scripts/dev/merge_test_durations.py"
    helper.parent.mkdir(parents=True)
    legacy_source = (
        "import argparse\n"
        "parser = argparse.ArgumentParser()\n"
        "parser.add_argument('--artifact-dir')\n"
        "parser.add_argument('--output')\n"
        "parser.parse_args()\n"
    )
    helper.write_text(legacy_source)
    git("add", ".")
    git("-c", "core.hooksPath=/dev/null", "commit", "-qm", "Older PR helper")
    pr_sha = git("rev-parse", "HEAD")
    git("branch", "legacy-pr")
    helper.write_text((ROOT / "scripts/dev/merge_test_durations.py").read_text())
    git("add", ".")
    git("-c", "core.hooksPath=/dev/null", "commit", "-qm", "Workflow helper")
    env["WORKFLOW_SHA"] = git("rev-parse", "HEAD")
    if helper_available:
        git("checkout", "-q", "--detach", pr_sha)
    else:
        # A single-branch clone does not contain the newer workflow object.
        cloned_checkout = tmp_path / "older-pr"
        git(
            "clone",
            "-q",
            "--no-local",
            "--single-branch",
            "--branch",
            "legacy-pr",
            str(checkout),
            str(cloned_checkout),
        )
        checkout = cloned_checkout
        helper = checkout / "scripts/dev/merge_test_durations.py"
    durations = checkout / ".test_durations"
    durations.write_text('{"tests/example.py::test_example": 2.5}\n')

    result = subprocess.run(
        ["bash", "-e", "-c", step["run"]],
        cwd=checkout,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert json.loads(durations.read_text()) == {"tests/example.py::test_example": 2.5}
    assert helper.read_text() == legacy_source
    assert git("rev-parse", "HEAD") == pr_sha
    assert step["env"]["WORKFLOW_SHA"] == "${{ github.workflow_sha }}"
