"""The intermediate CI PR stays conservative before default admission lands."""

from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]


def test_pr_shards_do_not_enable_partial_selection_before_default_admission():
    """The intermediate stack must run every slow test while activation is dormant."""
    workflow = yaml.safe_load((ROOT / ".github/workflows/ci.yml").read_text())
    job = workflow["jobs"]["fast-feedback"]
    assert str(job["env"]["ROBOT_SF_SHARD_INCLUDE_SLOW"]) == "1"
    # The top PR can consume the artifact only after fast-by-default admission.
    conftest = (ROOT / "tests/conftest.py").read_text()
    binding = job["env"].get("ROBOT_SF_AFFECTED_BASE_REF", "")
    assert not binding or "_FAST_FILES = {" not in conftest
    checkout = next(step for step in job["steps"] if step.get("name") == "Checkout")
    assert checkout["with"]["fetch-depth"] == 0
    assert "github.event.pull_request.head.sha" in checkout["with"]["ref"]


def test_selector_is_prepared_once_at_the_exact_matrix_head():
    """Avoid six repeated scans and a stale implicit PR merge checkout."""
    workflow = yaml.safe_load((ROOT / ".github/workflows/ci.yml").read_text())
    setup = workflow["jobs"]["dispatch-ownership"]
    prepare = next(
        (s for s in setup["steps"] if s.get("name") == "Prepare conservative test admission"), None
    )
    assert prepare is not None
    assert '--base "$BASE_SHA"' in prepare["run"] and '--head "$HEAD_SHA"' in prepare["run"]
    assert "--json-output output/ci/affected-selection.json" in prepare["run"]
    runs = "\n".join(
        str(s.get("run", "")) for j in workflow["jobs"].values() for s in j.get("steps", [])
    )
    assert runs.count("python scripts/dev/affected_test_selection.py") == 1
    checkout = setup["steps"][0]["with"]
    assert "github.event.pull_request.head.sha" in checkout["ref"] and checkout["fetch-depth"] == 0
    upload = next(
        s for s in setup["steps"] if s.get("name") == "Upload conservative test admission"
    )
    assert upload["with"]["if-no-files-found"] == "error"
