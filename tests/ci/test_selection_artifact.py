"""Reusable admission cannot falsely claim an unchanged source tree."""

import json
import subprocess

import pytest

from scripts.dev.affected_test_selection import read_report, selection_report
from tests.support.ci_wrapper_fixture import wrapper_repository


def test_unchanged_artifact_cannot_hide_a_changed_tree(tmp_path):
    """A false unchanged decision must fail even if its identities are otherwise current."""
    repo, _ = wrapper_repository(tmp_path)
    base = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()
    (repo / "tests/test_one.py").write_text("def test_one(): assert 1 == 1\n")
    subprocess.run(["git", "add", "tests/test_one.py"], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-qm", "change input"], cwd=repo, check=True)
    report = selection_report(repo, base)
    report.update(mode="unchanged", changed_paths=[])
    path = repo / "output/decision.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(report))
    with pytest.raises(ValueError, match="stale or invalid"):
        read_report(repo, path, base, "HEAD")
