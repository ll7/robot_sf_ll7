"""The hosted PR event must activate affected-slow selection on its exact diff."""

from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]


def test_pr_shards_bind_affected_selection_to_event_base():
    """Catch the missing event binding that allowed shell-contract regressions."""
    workflow = yaml.safe_load((ROOT / ".github/workflows/ci.yml").read_text())
    job = workflow["jobs"]["fast-feedback"]
    binding = job["env"].get("ROBOT_SF_AFFECTED_BASE_REF", "")
    assert "github.event_name == 'pull_request'" in binding
    assert "github.event.pull_request.base.sha" in binding
    assert "|| ''" in binding
    checkout = next(step for step in job["steps"] if step.get("name") == "Checkout")
    assert checkout["with"]["fetch-depth"] == 0
    assert "github.event.pull_request.head.sha" in checkout["with"]["ref"]
