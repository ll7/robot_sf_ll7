"""Actual Git histories and LFS objects protect post-run receipt admission."""

import hashlib
import json
import subprocess
from pathlib import Path

import pytest

from scripts.ci import behaviour_receipt as adapter
from tests.validation.test_behaviour_receipt_gate import _commit_receipt, _receipt
from tests.validation.test_behaviour_receipt_transport import _git_repo

ROOT = Path(__file__).resolve().parents[2]


def _run_receipt(monkeypatch, root):
    """Commit rows after the recorded run/audit commit and return their compact header."""
    body = _commit_receipt(monkeypatch, root, _receipt())
    header = json.loads(body.split("\n", 1)[1].rsplit("\n", 1)[0])
    scope = {
        "arms": ["goal", "orca"],
        "maps": ["open", "door"],
        "vehicle_id": "t60",
        "exceptions": [],
    }
    policy = root / "scope.json"
    policy.write_text(json.dumps(scope))
    monkeypatch.setattr(adapter, "SCOPE_PATH", policy)
    monkeypatch.setattr(adapter, "latest_release", lambda repo: "0.0.7")
    monkeypatch.setattr(adapter, "release_source", lambda release: "c" * 40)

    def git(*args):
        return subprocess.check_output(["git", *args], cwd=root, text=True).strip()

    return git, header, scope


def _body(header):
    return "<!-- behaviour-change-receipt:v2\n" + json.dumps(header) + "\n-->"


def _bind_final_head(monkeypatch, header, head):
    header["head_sha"] = header["refute_review"]["head_sha"] = head
    monkeypatch.setenv("BEHAVIOUR_PR_HEAD_SHA", head)


@pytest.mark.parametrize("field", ["scheduler", "interaction_audit"])
def test_rows_committed_after_execution_are_accepted(monkeypatch, tmp_path, field):
    """Run/audit source may precede the rows commit while review covers the final head."""
    git, header, _ = _run_receipt(monkeypatch, tmp_path)
    run = header[field]["source_sha"]
    head = header["head_sha"]
    assert run != head
    git("merge-base", "--is-ancestor", run, head)
    assert git("diff", "--name-only", run, head) == "receipts/behaviour/sweep.json"
    assert header["refute_review"]["head_sha"] == head
    # Isolate each execution identity; the shared transport fixtures also exercise both.
    other = "interaction_audit" if field == "scheduler" else "scheduler"
    header[other]["source_sha"] = head
    assert (
        adapter.check_receipt(_body(header), ["robot_sf/planner/policy.py"], "ll7/robot_sf_ll7")
        == []
    )


@pytest.mark.parametrize("field", ["scheduler", "interaction_audit"])
@pytest.mark.parametrize("change", ["planner", "docs", "rename"])
def test_non_receipt_changes_after_execution_are_rejected(monkeypatch, tmp_path, field, change):
    """Any non-receipt net change invalidates earlier execution, including rename endpoints."""
    git, header, scope = _run_receipt(monkeypatch, tmp_path)
    execution_source = header[field]["source_sha"]
    if change == "rename":
        (tmp_path / "docs").mkdir()
        git("mv", "receipts/behaviour/sweep.json", "docs/unbound.json")
        git("commit", "-m", "rename receipt outside namespace")
        header["rows_artifact"]["path"] = "docs/unbound.json"
        # Direct provenance validation needs no payload read to detect the endpoint.
        receipt = _receipt()
    else:
        path = "robot_sf/planner/new.py" if change == "planner" else "docs/notes.md"
        file = tmp_path / path
        file.parent.mkdir(parents=True, exist_ok=True)
        file.write_text("changed\n")
        git("add", path)
        git("commit", "-m", "non-receipt edit after execution")
        receipt = adapter.load_receipt(header, header["head_sha"])
    head = git("rev-parse", "HEAD")
    receipt["head_sha"] = receipt["refute_review"]["head_sha"] = head
    receipt["scheduler"]["source_sha"] = receipt["interaction_audit"]["source_sha"] = head
    receipt[field]["source_sha"] = execution_source
    label = "job" if field == "scheduler" else "real-row audit"
    with pytest.raises(ValueError, match=f"non-receipt changes after {label} source"):
        adapter.validate_receipt(receipt, scope, head, "0.0.7", "c" * 40)
    if change != "rename":
        _bind_final_head(monkeypatch, header, head)
        assert adapter.check_receipt(
            _body(header), ["robot_sf/planner/policy.py"], "ll7/robot_sf_ll7"
        )


@pytest.mark.parametrize("field", ["scheduler", "interaction_audit"])
def test_non_ancestor_execution_source_is_rejected(monkeypatch, tmp_path, field):
    """A disconnected source with identical tree bytes cannot certify this branch."""
    git, header, scope = _run_receipt(monkeypatch, tmp_path)
    head = header["head_sha"]
    orphan = git(
        "commit-tree", git("rev-parse", "HEAD^{tree}"), "-m", "unrelated execution history"
    )
    assert git("diff", "--name-only", orphan, head) == ""
    receipt = adapter.load_receipt(header, head)
    receipt["scheduler"]["source_sha"] = receipt["interaction_audit"]["source_sha"] = head
    receipt[field]["source_sha"] = orphan
    label = "job" if field == "scheduler" else "real-row audit"
    with pytest.raises(ValueError, match=f"{label} source is not an ancestor of PR head"):
        adapter.validate_receipt(receipt, scope, head, "0.0.7", "c" * 40)


def test_refute_review_still_requires_final_head(monkeypatch, tmp_path):
    """Execution ancestry does not excuse a review made before the rows commit."""
    _, header, scope = _run_receipt(monkeypatch, tmp_path)
    receipt = adapter.load_receipt(header, header["head_sha"])
    receipt["refute_review"]["head_sha"] = header["scheduler"]["source_sha"]
    with pytest.raises(ValueError, match="exact-head refute verdict is missing"):
        adapter.validate_receipt(receipt, scope, header["head_sha"], "0.0.7", "c" * 40)


@pytest.mark.parametrize("fault", [None, "missing_object", "wrong_digest"])
def test_committed_lfs_receipt_uses_verified_object(monkeypatch, tmp_path, fault):
    """Resolve the source pointer through real LFS, rejecting missing or unbound data."""
    git = _git_repo(tmp_path)
    git("lfs", "install", "--local")
    attributes = (ROOT / ".gitattributes").read_text()
    rule = "receipts/behaviour/*.json filter=lfs diff=lfs merge=lfs -text"
    if rule not in attributes:
        attributes += rule + "\n"
    (tmp_path / ".gitattributes").write_text(attributes)
    git("add", ".gitattributes")
    git("commit", "-m", "LFS policy before execution")
    _, header, _ = _run_receipt(monkeypatch, tmp_path)
    path = header["rows_artifact"]["path"]
    head = header["head_sha"]
    pointer = git("cat-file", "blob", f"{head}:{path}")
    digest = header["rows_artifact"]["sha256"]
    assert pointer.startswith("version https://git-lfs.github.com/spec/v1\n")
    assert f"oid sha256:{digest}" in pointer
    original = (tmp_path / path).read_bytes()
    assert hashlib.sha256(original).hexdigest() == digest
    # Neither checkout corruption nor skip-smudge/skip-download-error settings
    # may turn a pointer or mutable file into successful receipt evidence.
    (tmp_path / path).write_text("dirty bytes\n")
    monkeypatch.setenv("GIT_LFS_SKIP_SMUDGE", "1")
    git("config", "lfs.skipdownloaderrors", "true")
    if fault == "missing_object":
        object_file = tmp_path / git(
            "rev-parse", "--git-path", f"lfs/objects/{digest[:2]}/{digest[2:4]}/{digest}"
        )
        object_file.unlink()
    elif fault == "wrong_digest":
        header["rows_artifact"]["sha256"] = "0" * 64
    # This test isolates LFS transport; the run identity fixtures cover ancestry.
    header["scheduler"]["source_sha"] = header["interaction_audit"]["source_sha"] = head
    blockers = adapter.check_receipt(
        _body(header), ["robot_sf/planner/policy.py"], "ll7/robot_sf_ll7"
    )
    if fault is None:
        assert blockers == []
    else:
        assert len(blockers) == 1 and "behaviour receipt rejected" in blockers[0]


def test_repository_receipt_attributes_route_json_to_lfs():
    """The repository's actual attribute rules route new JSON receipts to LFS."""
    output = subprocess.check_output(
        ["git", "check-attr", "filter", "--", "receipts/behaviour/sweep.json"], cwd=ROOT, text=True
    ).strip()
    assert output == "receipts/behaviour/sweep.json: filter: lfs"
