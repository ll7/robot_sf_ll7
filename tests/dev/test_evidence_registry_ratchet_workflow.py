"""Contract tests for the evidence-registry ratchet pull-request workflow.

Issue #6740: the blocking PR gate for ``docs/context/evidence/**`` changes ran
``evidence_registry_ratchet.py --check | tee <log>`` in a step whose shell had
no pipefail, so the log pipe masked the ratchet's non-zero exit code. PR #6733
therefore merged a baseline drift that reddened main CI. These tests lock the
gate's fail-closed contract.
"""

from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
WORKFLOW_PATH = REPO_ROOT / ".github/workflows/evidence-registry-ratchet.yml"


def _ratchet_job() -> dict:
    """Return the evidence-registry ratchet job from the workflow definition."""
    workflow = yaml.load(WORKFLOW_PATH.read_text(encoding="utf-8"), Loader=yaml.BaseLoader)
    return workflow["jobs"]["evidence-registry-ratchet"]


def _ratchet_step() -> dict:
    """Return the blocking ratchet step from the workflow definition."""
    for step in _ratchet_job()["steps"]:
        if step.get("id") == "ratchet":
            return step
    raise AssertionError("evidence-registry-ratchet.yml has no step with id 'ratchet'")


def test_ratchet_step_is_a_blocking_gate() -> None:
    """The PR ratchet job and step must stay blocking, not advisory."""
    job = _ratchet_job()
    assert job.get("continue-on-error") in (None, "false"), (
        "the evidence-registry ratchet job must fail the check on net-new findings"
    )
    step = _ratchet_step()
    assert step.get("continue-on-error") in (None, "false"), (
        "the evidence-registry ratchet step must fail the job on net-new findings"
    )


def test_ratchet_step_propagates_check_exit_code() -> None:
    """A piped ``--check`` invocation must not mask the ratchet exit code."""
    run_lines = [line.strip() for line in _ratchet_step()["run"].splitlines() if line.strip()]
    piped_check_indices = [
        index
        for index, line in enumerate(run_lines)
        if "evidence_registry_ratchet.py --check" in line and "|" in line
    ]
    assert len(piped_check_indices) == 1, (
        "the ratchet step must contain exactly one piped `--check` invocation so the "
        "fail-closed exit-code contract is exercised"
    )
    index = piped_check_indices[0]
    assert any(earlier == "set -o pipefail" for earlier in run_lines[:index]), (
        "the piped --check invocation must run after `set -o pipefail`; otherwise "
        "the log pipe swallows the ratchet's non-zero exit code and the gate "
        "reports success while FAILED (issue #6740)"
    )


def test_ratchet_runs_when_release_assurance_inputs_change() -> None:
    """Pinned release-assurance inputs must trigger the pull-request ratchet."""
    workflow = yaml.load(
        WORKFLOW_PATH.read_text(encoding="utf-8"),
        Loader=yaml.BaseLoader,
    )
    paths = workflow["on"]["pull_request"]["paths"]

    assert "docs/RELEASE.md" in paths
    assert "CITATION.cff" in paths
