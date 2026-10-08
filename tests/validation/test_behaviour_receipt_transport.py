"""Regression proof for compact transport, exact Git source and base-owned policy."""

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from scripts.ci import behaviour_receipt as adapter
from scripts.ci import pr_contract_check as checker
from scripts.validation import check_seed_holdout_diff as seed_checker
from tests.validation.test_behaviour_receipt_gate import (
    _commit_receipt,
    _header,
    _inventory_receipt,
)

ROOT = Path(__file__).resolve().parents[2]


def _git_repo(root):
    """Create an isolated real Git repository and return its bounded command helper."""
    root.mkdir(exist_ok=True)

    def git(*args):
        return subprocess.check_output(["git", *args], cwd=root, text=True).strip()

    git("init", "-b", "main")
    git("config", "user.name", "Test")
    git("config", "user.email", "test@example.invalid")
    return git


def _commit_file(root, git, path, text, message):
    target = root / path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(text)
    git("add", path)
    git("commit", "-m", message)
    return git("rev-parse", "HEAD")


@pytest.mark.parametrize("package", ["torch", "stable-baselines3", "numpy", "gymnasium", "ruff"])
@pytest.mark.parametrize(
    "file", ["uv.lock", "pyproject.toml", "fast-pysf/uv.lock", "fast-pysf/pyproject.toml"]
)
def test_sensitive_dependency_real_git_diff(monkeypatch, tmp_path, package, file):
    """Only changes to the four runtime-sensitive package rows require receipts."""
    git = _git_repo(tmp_path)

    def document(version):
        if file.endswith("uv.lock"):
            return f'version = 1\n[[package]]\nname = "{package}"\nversion = "{version}"\n'
        return f'[project]\nname = "test"\ndependencies = ["{package}=={version}"]\n'

    base = _commit_file(tmp_path, git, file, document("1.0"), "base")
    git("update-ref", "refs/remotes/origin/main", base)
    head = _commit_file(tmp_path, git, file, document("2.0"), "dependency update")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(adapter, "ROOT", tmp_path)
    monkeypatch.setenv("BEHAVIOUR_PR_HEAD_SHA", head)
    changed = checker.get_changed_files(None, base)
    assert changed == [file]
    blockers = adapter.check_receipt("", changed, "ll7/robot_sf_ll7")
    if package == "ruff":
        assert blockers == []
    else:
        assert blockers == [
            "BLOCKER: behaviour receipt missing or duplicated for a behaviour-changing PR"
        ]


def test_dependency_rule_is_separate_from_path_rule(monkeypatch, tmp_path):
    """An author-owned dependency switch never disables planner admission."""
    git = _git_repo(tmp_path)
    base = _commit_file(
        tmp_path, git, "uv.lock", '[[package]]\nname="torch"\nversion="1"\n', "base"
    )
    head = _commit_file(
        tmp_path, git, "uv.lock", '[[package]]\nname="torch"\nversion="2"\n', "bump"
    )
    monkeypatch.setattr(adapter, "ROOT", tmp_path)
    monkeypatch.setenv("BEHAVIOUR_PR_HEAD_SHA", head)
    monkeypatch.setattr(adapter, "DEPENDENCY_RECEIPTS_ENABLED", False)
    assert adapter.check_receipt("", ["uv.lock"], "ll7/robot_sf_ll7", base) == []
    assert adapter.check_receipt(
        "", ["uv.lock", "robot_sf/planner/goal.py"], "ll7/robot_sf_ll7", base
    )


def test_dependency_base_drift_and_comments_stay_exempt(monkeypatch, tmp_path):
    """A stale branch must not inherit a sensitive bump made only on main."""
    git = _git_repo(tmp_path)
    original = '[[package]]\nname="torch"\nversion="1"\n[[package]]\nname="ruff"\nversion="1"\n'
    ancestor = _commit_file(tmp_path, git, "uv.lock", original, "ancestor")
    _commit_file(
        tmp_path,
        git,
        "uv.lock",
        original.replace('name="torch"\nversion="1"', 'name="torch"\nversion="2"'),
        "main runtime bump",
    )
    git("checkout", "-b", "source", ancestor)
    head = _commit_file(
        tmp_path,
        git,
        "uv.lock",
        original.replace('name="ruff"\nversion="1"', 'name="ruff"\nversion="2"'),
        "tooling bump",
    )
    monkeypatch.setattr(adapter, "ROOT", tmp_path)
    monkeypatch.setenv("BEHAVIOUR_PR_HEAD_SHA", head)
    assert adapter.check_receipt("", ["uv.lock"], "ll7/robot_sf_ll7", "main") == []
    _commit_file(
        tmp_path,
        git,
        "pyproject.toml",
        '[project]\nname="test"\ndependencies=["torch==1"]\n',
        "project",
    )
    base = git("rev-parse", "HEAD")
    head = _commit_file(
        tmp_path,
        git,
        "pyproject.toml",
        '# torch is unchanged\n[project]\nname="test"\ndependencies=["torch==1"]\n',
        "comment",
    )
    monkeypatch.setenv("BEHAVIOUR_PR_HEAD_SHA", head)
    assert adapter.check_receipt("", ["pyproject.toml"], "ll7/robot_sf_ll7", base) == []


@pytest.mark.parametrize("change", ["add", "remove", "group", "malformed"])
def test_sensitive_dependency_add_remove_group_and_malformed(monkeypatch, tmp_path, change):
    """Addition, removal and group edits cannot bypass the semantic dependency rule."""
    git = _git_repo(tmp_path)
    before = '[project]\nname="test"\ndependencies=["torch==1"]\n'
    after = '[project]\nname="test"\ndependencies=[]\n'
    if change == "add":
        before, after = after, before
    elif change == "group":
        before = '[project]\nname="test"\n[dependency-groups]\ntraining=["torch==1"]\n'
        after = before.replace("torch==1", "torch==2")
    elif change == "malformed":
        after = "[invalid"
    base = _commit_file(tmp_path, git, "pyproject.toml", before, "base")
    head = _commit_file(tmp_path, git, "pyproject.toml", after, "change")
    monkeypatch.setattr(adapter, "ROOT", tmp_path)
    monkeypatch.setenv("BEHAVIOUR_PR_HEAD_SHA", head)
    blockers = adapter.check_receipt("", ["pyproject.toml"], "ll7/robot_sf_ll7", base)
    expected = "rejected" if change == "malformed" else "missing or duplicated"
    assert len(blockers) == 1 and expected in blockers[0]


def test_stale_branch_preserves_merge_checkout_for_existing_guards(monkeypatch, tmp_path):
    """The merge checkout excludes main-only removals from seed/new-file guards."""
    workflow = yaml.safe_load((ROOT / ".github/workflows/pr-contract-check.yml").read_text())
    steps = workflow["jobs"]["pr-contract-check"]["steps"]
    checkout = next(step for step in steps if step.get("name") == "Checkout")
    assert "ref" not in checkout["with"], "existing guards require the default merge checkout"
    gate = next(
        step for step in steps if step.get("name") == "Validate behaviour receipt with base policy"
    )
    assert gate["env"]["BEHAVIOUR_PR_HEAD_SHA"] == "${{ github.event.pull_request.head.sha }}"
    git = _git_repo(tmp_path)
    ancestor = _commit_file(tmp_path, git, "configs/removed.yaml", "seeds: [1001]\n", "ancestor")
    git("rm", "configs/removed.yaml")
    git("commit", "-m", "main removal")
    base = git("rev-parse", "HEAD")
    git("checkout", "-b", "source", ancestor)
    head = _commit_file(tmp_path, git, "docs/change.md", "Documentation\n", "source")
    monkeypatch.chdir(tmp_path)
    assert "configs/removed.yaml" in checker.get_new_files(base)
    assert "configs/removed.yaml" in git("diff", base, head)
    # Construct the combined tree and two-parent commit without running git merge.
    git("rm", "configs/removed.yaml")
    tree = git("write-tree")
    merge = git("commit-tree", tree, "-p", base, "-p", head, "-m", "synthetic merge")
    git("checkout", "--detach", merge)
    monkeypatch.setattr(adapter, "ROOT", tmp_path)
    monkeypatch.setenv("BEHAVIOUR_PR_HEAD_SHA", head)
    assert adapter.current_head() == head != git("rev-parse", "HEAD")
    assert checker.get_new_files(base) == {"docs/change.md"}
    diff = git("diff", base, "HEAD")
    assert "configs/removed.yaml" not in diff
    assert seed_checker.check_diff(diff, tmp_path) == []


def _trusted_fixture(root):
    """Copy real policy and its imports into an independent fixture base commit."""
    git = _git_repo(root)
    shutil.copytree(
        ROOT / "robot_sf", root / "robot_sf", ignore=shutil.ignore_patterns("__pycache__", "*.pyc")
    )
    files = [
        "scripts/ci/behaviour_receipt.py",
        "scripts/ci/behaviour_receipt.schema.json",
        "scripts/dev/check_dependabot_update_policy.py",
        "scripts/dev/check_dependency_coherence.py",
        "configs/benchmarks/releases/behaviour_gate_0_1_0.json",
        "model/registry.yaml",
    ]
    for path in files:
        target = root / path
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / path, target)
    (root / ".gitignore").write_text("__pycache__/\n")
    git("add", "robot_sf", ".gitignore", *files)
    git("commit", "-m", "trusted gate base")
    git("tag", "0.0.7")
    return git, git("rev-parse", "HEAD")


def test_full_roster_pr_body_file_and_base_policy_end_to_end(tmp_path):
    """The deployed shell accepts compact source data and rejects head-policy bypasses."""
    root = tmp_path / "repo"
    git, base = _trusted_fixture(root)
    scope = json.loads((ROOT / adapter.SCOPE_FILE).read_text())
    receipt = _inventory_receipt(scope)
    raw = json.dumps(
        {
            "schema_version": "behaviour-change-rows.v1",
            "rows": receipt["rows"],
            "classifications": receipt["classifications"],
        }
    ).encode()
    path = "receipts/behaviour/sweep.json"
    target = root / path
    target.parent.mkdir(parents=True)
    target.write_bytes(raw)
    # A behaviour PR tampers with all three head policy surfaces. Base execution
    # must still accept honest rows and reject absent/corrupted receipts.
    (root / "scripts/ci/behaviour_receipt.py").write_text("raise SystemExit(0)\n")
    (root / "scripts/ci/behaviour_receipt.schema.json").write_text("{}\n")
    (root / adapter.SCOPE_FILE).write_text('{"arms": []}\n')
    planner = root / "robot_sf/planner/new_policy.py"
    planner.write_text("VALUE = 1\n")
    git(
        "add",
        path,
        "scripts/ci/behaviour_receipt.py",
        "scripts/ci/behaviour_receipt.schema.json",
        adapter.SCOPE_FILE,
        "robot_sf/planner/new_policy.py",
    )
    git("commit", "-m", "PR source with untrusted policy edits")
    head = git("rev-parse", "HEAD")
    # Leave CI at a distinct synthetic merge identity.
    merge = git(
        "commit-tree",
        git("rev-parse", "HEAD^{tree}"),
        "-p",
        base,
        "-p",
        head,
        "-m",
        "synthetic merge",
    )
    git("checkout", "--detach", merge)
    for item in (
        receipt,
        receipt["scheduler"],
        receipt["interaction_audit"],
        receipt["refute_review"],
    ):
        key = "source_sha" if "source_sha" in item else "head_sha"
        item[key] = head
    receipt["baseline"]["source_sha"] = base
    body = _header(receipt, raw)
    assert len(body) < 65536 < len(raw)
    assert len(receipt["rows"]) == 21420
    # Deliberately dirty worktree bytes must not replace the committed source blob.
    target.write_text("forged checkout bytes\n")
    tools = tmp_path / "tools"
    tools.mkdir()
    (tools / "uv").write_text(
        f'#!{sys.executable}\nimport os,sys\nassert sys.argv[1:3] == ["run", "python"]\nos.execv(sys.executable, [sys.executable, *sys.argv[3:]])\n'
    )
    (tools / "gh").write_text(
        f'#!{sys.executable}\nimport json\nprint(json.dumps([[{{"draft":False,"prerelease":False,"tag_name":"0.0.7","published_at":"2026-01-01"}}]]))\n'
    )
    for executable in tools.iterdir():
        executable.chmod(0o755)
    workflow = yaml.safe_load((ROOT / ".github/workflows/pr-contract-check.yml").read_text())
    step = next(
        step
        for step in workflow["jobs"]["pr-contract-check"]["steps"]
        if step.get("name") == "Validate behaviour receipt with base policy"
    )
    event_path = tmp_path / "event.json"
    env = {
        **os.environ,
        "PATH": f"{tools}:{os.environ['PATH']}",
        "PR_BASE_SHA": base,
        "BEHAVIOUR_PR_HEAD_SHA": head,
        "GITHUB_WORKSPACE": str(root),
        "GITHUB_EVENT_PATH": str(event_path),
    }
    for index, variant in enumerate(("valid", "missing", "digest", "classification")):
        header = json.loads(body.split("\n", 1)[1].rsplit("\n", 1)[0])
        if variant == "digest":
            header["rows_artifact"]["sha256"] = "0" * 64
        elif variant == "classification":
            header["classifications"]["count"] = 1
        candidate = (
            ""
            if variant == "missing"
            else "<!-- behaviour-change-receipt:v2\n" + json.dumps(header) + "\n-->"
        )
        event_path.write_text(
            json.dumps(
                {
                    "repository": {"full_name": "ll7/robot_sf_ll7"},
                    "pull_request": {"head": {"sha": head}, "body": candidate},
                }
            )
        )
        run_dir = tmp_path / f"runner-{index}"
        run_dir.mkdir()
        result = subprocess.run(
            ["bash", "-c", step["run"]],
            cwd=root,
            env={**env, "RUNNER_TEMP": str(run_dir)},
            text=True,
            capture_output=True,
            check=False,
        )
        output = result.stdout + result.stderr
        if variant == "valid":
            assert result.returncode == 0, output
            assert "accepted or out of scope" in output
        else:
            assert result.returncode == 1, output
            assert "BLOCKER: behaviour receipt" in output
        assert git("rev-parse", "HEAD") == merge
    # An inline all-row receipt would exceed GitHub's body cap; the new body is small.
    assert len(json.dumps(receipt)) > 65536


def test_base_gate_bootstrap_never_executes_head_policy(tmp_path):
    """First installation reports inactive base policy rather than trusting head code."""
    git = _git_repo(tmp_path / "repo")
    root = tmp_path / "repo"
    base = _commit_file(root, git, "README.md", "Base\n", "base without gate")
    _commit_file(
        root,
        git,
        "scripts/ci/behaviour_receipt.py",
        'raise RuntimeError("must never execute head policy")\n',
        "untrusted head",
    )
    workflow = yaml.safe_load((ROOT / ".github/workflows/pr-contract-check.yml").read_text())
    step = next(
        step
        for step in workflow["jobs"]["pr-contract-check"]["steps"]
        if step.get("name") == "Validate behaviour receipt with base policy"
    )
    result = subprocess.run(
        ["bash", "-c", step["run"]],
        cwd=root,
        env={**os.environ, "PR_BASE_SHA": base},
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "not installed on the event base" in result.stdout


def test_full_roster_compact_body_and_committed_rows(monkeypatch, tmp_path):
    """All 21,420 rows pass admission via a GitHub-sized body and committed file."""
    scope = json.loads((ROOT / adapter.SCOPE_FILE).read_text())
    receipt = _inventory_receipt(scope)
    body = _commit_receipt(monkeypatch, tmp_path, receipt)
    monkeypatch.setattr(adapter, "latest_release", lambda repo: "0.0.7")
    monkeypatch.setattr(adapter, "release_source", lambda release: "c" * 40)
    assert len(body) < 65536
    assert len((tmp_path / "receipts/behaviour/sweep.json").read_bytes()) > 65536
    assert (
        adapter.check_receipt(body, ["robot_sf/planner/guarded_ppo.py"], "ll7/robot_sf_ll7") == []
    )


def test_event_head_overrides_merge_identity(monkeypatch, tmp_path):
    """The immutable event SHA, rather than the distinct merge commit, binds receipts."""
    git = _git_repo(tmp_path)
    source = _commit_file(tmp_path, git, "README.md", "source", "source")
    git("commit", "--allow-empty", "-m", "merge identity")
    assert git("rev-parse", "HEAD") != source
    monkeypatch.setattr(adapter, "ROOT", tmp_path)
    monkeypatch.setenv("BEHAVIOUR_PR_HEAD_SHA", source)
    assert adapter.current_head() == source
