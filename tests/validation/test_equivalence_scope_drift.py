"""Exact reviewed base blobs cannot waive candidate or protected scope changes."""

# evidence-writer-exempt: all writes create throwaway pytest tmp_path Git trees,
# including synthetic catalog/evidence paths used to test exact committed bytes;
# these fixtures never publish durable evidence, and shared writers alter the bytes.

import hashlib
import subprocess
from pathlib import Path

import pytest

from scripts.ci import behaviour_receipt as gate
from scripts.ci import default_path_equivalence as eq
from tests.validation.test_default_path_equivalence import AFTER, BEFORE, PATH

MAIN = "e251a2f94e9478ab7fd5252e88d0296666876ab1"
SOURCE_ROOT = Path(__file__).resolve().parents[2]
REVIEWED = {
    "SLURM/ftplan/released_ppo_pilot.slurm": "e0e9dbb2127a1e6b491bedb4f2668eacbeee39924de39cf76aede783d678ad8c",
    "configs/training/ppo/ftplan/pilot_protocol_20261012.yaml": "50e891bfe72189dcbc667f8301f4b6357a0846575bd4c5c08c1d32e6cd976548",
    "configs/training/ppo/ftplan/released_warm_weighted_3m_20261012.template": "39b28f6451a64ce99067809e81fd1c875e36fc2735f729fb709fdabe4810d4ee",
    "docs/context/catalog.yaml": "45328def9fbe8deb4e083bccbfb2d0abcf2cd51fbf40a1db23d78f6afab2b0c9",
    "docs/context/evidence/ppo_wall_probe_dev_20261012/probe.json": "d12bb0c7b4c23cd429f840c1d23a50cba597f1f79bad0a9dbbcb3321c7bea6b3",
    "scripts/analysis/ppo_wall_probe_dev.py": "2b41527c564ab48fa6a1694e5875f8c28be67f7246e6c381a7e7b343feaca457",
    "scripts/validation/evidence_registry_baseline.json": "a31b2bffa4585496600b3bd2dcf71e2319049e3934ac7698e2d682458ebd7fd7",
    "scripts/validation/analyze_examiner_robustness_20261012.py": "e372f62c538cc4fed6efd8350a58d4c316633e5f068beea6fcab750b4baab357",
    "scripts/validation/examiner_crowd_probes_20261012.py": "72b15e48bf73f2a6a9c974d0021dc5ce4d4bf285e6ce8ad2b257f288165d64b3",
    "scripts/validation/examiner_robustness_20261012.py": "9d14e552087f98b5a23e1f4bfe7a0656b6f8ef840b6322a1e5b0488f9b6ea02a",
    "scripts/validation/render_examiner_robustness_20261012.py": "e2fd98b977be44b69d2e195d93fc1c9727745c51c5a4814cf220a66993f7cb3e",
    "scripts/validation/resume_examiner_mppi_20261012.py": "e381fc2baf56ce1c64f8686a93796225bd35d22b732fc9df3b5f65eff0f35d3c",
}


@pytest.fixture
def scope_repo(tmp_path, monkeypatch):
    def git(*args):
        return subprocess.check_output(["git", *args], cwd=tmp_path, text=True).strip()

    git("init", "-b", "main")
    git("config", "user.email", "test@example.invalid")
    git("config", "user.name", "Test")
    module = tmp_path / PATH
    module.parent.mkdir(parents=True)
    module.write_text(BEFORE)
    for name in REVIEWED:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("old common ancestor bytes\n")
    git("add", ".")
    git("commit", "-m", "common ancestor")
    fork = git("rev-parse", "HEAD")
    git("checkout", "-b", "candidate")
    module.write_text(AFTER)
    git("add", ".")
    git("commit", "-m", "neutral validator")
    candidate = git("rev-parse", "HEAD")
    git("checkout", "main")
    for name, digest in REVIEWED.items():
        blob = subprocess.check_output(
            ["git", "cat-file", "blob", f"{MAIN}:{name}"], cwd=SOURCE_ROOT
        )
        assert hashlib.sha256(blob).hexdigest() == digest
        (tmp_path / name).write_bytes(blob)
    git("add", ".")
    git("commit", "-m", "reviewed preparation on main")
    main = git("rev-parse", "HEAD")
    monkeypatch.setattr(gate, "ROOT", tmp_path)
    return git, fork, main, candidate


def test_only_exact_reviewed_base_blobs_allow_neutral_transition(scope_repo):
    _, _, main, candidate = scope_repo
    assert eq.classify_scope([], main, candidate) == [PATH + ":_episode_identity"]


@pytest.mark.parametrize("path", REVIEWED)
@pytest.mark.parametrize("fault", ["edit", "delete", "type", "same_as_main"])
def test_reviewed_blob_never_waives_candidate_change(scope_repo, tmp_path, path, fault):
    git, _, main, _ = scope_repo
    git("checkout", "candidate")
    target = tmp_path / path
    if fault == "edit":
        target.write_text("candidate edit\n")
    elif fault == "delete":
        target.unlink()
    elif fault == "type":
        target.unlink()
        target.symlink_to("unreviewed-target")
    else:
        target.write_bytes(
            subprocess.check_output(["git", "cat-file", "blob", f"{main}:{path}"], cwd=tmp_path)
        )
    git("add", ".")
    git("commit", "-m", "candidate changed reviewed path")
    changed = git("rev-parse", "HEAD")
    with pytest.raises(ValueError, match="candidate changed reviewed base-only path"):
        eq.classify_scope([], main, changed)


@pytest.mark.parametrize("path", REVIEWED)
@pytest.mark.parametrize("fault", ["edit", "delete", "type"])
def test_reviewed_base_blob_requires_exact_bytes_and_regular_type(
    scope_repo, tmp_path, path, fault
):
    git, _, _, candidate = scope_repo
    target = tmp_path / path
    if fault == "edit":
        target.write_text("new unreviewed main bytes\n")
    elif fault == "delete":
        target.unlink()
    else:
        target.unlink()
        target.symlink_to("unreviewed-target")
    git("add", ".")
    git("commit", "-m", "main changed reviewed path")
    changed_main = git("rev-parse", "HEAD")
    with pytest.raises(
        ValueError, match="reviewed base-only (blob changed|path is not a regular blob)"
    ):
        eq.classify_scope([], changed_main, candidate)


def test_protected_config_refuses_even_if_added_to_reviewed_map(scope_repo, tmp_path, monkeypatch):
    git, _, main, candidate = scope_repo
    path = "configs/scenarios/unreviewed.yaml"
    target = tmp_path / path
    target.parent.mkdir(parents=True)
    target.write_text("protected current-main config\n")
    git("add", ".")
    git("commit", "-m", "protected main config")
    changed_main = git("rev-parse", "HEAD")
    monkeypatch.setattr(
        eq,
        "REVIEWED_BASE_ONLY_BLOBS",
        {**REVIEWED, path: hashlib.sha256(target.read_bytes()).hexdigest()},
        raising=False,
    )
    with pytest.raises(ValueError, match="protected default path"):
        eq.classify_scope([], changed_main, candidate)
    assert main != changed_main


def test_reviewed_base_blobs_require_a_separate_neutral_transition(scope_repo):
    _, fork, main, _ = scope_repo
    with pytest.raises(ValueError, match="no reviewed.*neutral transition"):
        eq.classify_scope([], main, fork)


@pytest.mark.parametrize(
    "policy_path",
    [
        "scripts/ci/receipt_outcomes.py",
        "scripts/ci/behaviour_receipt.py",
        "scripts/ci/receipt_provenance.py",
        "scripts/ci/default_path_equivalence.py",
        "scripts/ci/make_behaviour_receipt.py",
        "scripts/dev/merge_queue_gate.py",
        ".github/workflows/merge-queue-gate.yml",
        ".gitattributes",
    ],
)
def test_candidate_authority_edit_refuses_when_final_bytes_equal_main(
    tmp_path, monkeypatch, policy_path
):
    def git(*args):
        return subprocess.check_output(["git", *args], cwd=tmp_path, text=True).strip()

    git("init", "-b", "main")
    git("config", "user.email", "test@example.invalid")
    git("config", "user.name", "Test")
    module = tmp_path / PATH
    module.parent.mkdir(parents=True)
    module.write_text(BEFORE)
    policy = tmp_path / policy_path
    policy.parent.mkdir(parents=True, exist_ok=True)
    old = (
        "receipts/behaviour/*.json filter=lfs diff=lfs merge=lfs -text\n"
        if policy_path == ".gitattributes"
        else "old common ancestor policy\n"
    )
    promoted = (
        old + "receipts/behaviour/*-sources/**/*.tar.gz filter=lfs diff=lfs merge=lfs -text\n"
        if policy_path == ".gitattributes"
        else "same independently promoted main policy\n"
    )
    policy.write_text(old)
    git("add", ".")
    git("commit", "-m", "common ancestor")
    fork = git("rev-parse", "HEAD")
    git("checkout", "-b", "candidate")
    module.write_text(AFTER)
    git("add", ".")
    git("commit", "-m", "neutral change without authority edit")
    neutral = git("rev-parse", "HEAD")
    git("checkout", "main")
    policy.write_text(promoted)
    git("add", ".")
    git("commit", "-m", "independent main authority promotion")
    main = git("rev-parse", "HEAD")
    monkeypatch.setattr(gate, "ROOT", tmp_path)
    assert eq.classify_scope([], main, neutral) == [PATH + ":_episode_identity"]
    git("checkout", "candidate")
    policy.write_text(promoted)
    git("add", ".")
    git("commit", "-m", "candidate edits authority to same final main bytes")
    candidate = git("rev-parse", "HEAD")
    assert policy_path not in git("diff", "--name-only", main, candidate).splitlines()
    assert policy_path in git("diff", "--name-only", fork, candidate).splitlines()
    with pytest.raises(ValueError, match="unreviewed executable path " + policy_path):
        eq.classify_scope([], main, candidate)


@pytest.fixture
def closed_tree_repo(tmp_path, monkeypatch):
    root = "docs/context/robsweep"

    def git(*args):
        return subprocess.check_output(["git", *args], cwd=tmp_path, text=True).strip()

    git("init", "-b", "main")
    git("config", "user.email", "test@example.invalid")
    git("config", "user.name", "Test")
    module = tmp_path / PATH
    module.parent.mkdir(parents=True)
    module.write_text(BEFORE)
    folder = tmp_path / root
    folder.mkdir(parents=True)
    (folder / "approved.md").write_text("old ancestor bytes\n")
    (folder / "data.json").write_text("{}\n")
    git("add", ".")
    git("commit", "-m", "ancestor")
    git("checkout", "-b", "candidate")
    module.write_text(AFTER)
    git("add", ".")
    git("commit", "-m", "neutral validator")
    candidate = git("rev-parse", "HEAD")
    git("checkout", "main")
    (folder / "approved.md").write_text("approved main bytes\n")
    git("add", ".")
    git("commit", "-m", "approved closed tree")
    monkeypatch.setattr(gate, "ROOT", tmp_path)

    def pin():
        main = git("rev-parse", "HEAD")
        oid = git("rev-parse", f"{main}:{root}")
        raw = subprocess.check_output(["git", "cat-file", "tree", oid], cwd=tmp_path)
        entries = subprocess.check_output(
            ["git", "ls-tree", "-r", "-z", main, "--", root], cwd=tmp_path
        ).count(b"\0")
        monkeypatch.setattr(
            eq,
            "REVIEWED_CLOSED_TREE",
            {
                "root": root,
                "oid": oid,
                "sha256": hashlib.sha256(raw).hexdigest(),
                "entries": entries,
            },
            raising=False,
        )
        return main

    main = pin()
    assert eq.classify_scope([], main, candidate) == [PATH + ":_episode_identity"]
    return git, folder, main, candidate, pin


@pytest.mark.parametrize(
    "fault",
    [
        "edit_md",
        "edit_json",
        "add_md",
        "delete",
        "symlink",
        "mode",
        "root_file",
        "same_final",
        "newline_same_final",
        "tab_same_final",
    ],
)
def test_closed_tree_refuses_any_candidate_history_before_doc_skip(closed_tree_repo, fault):
    git, folder, main, _, pin = closed_tree_repo
    filename = "new\nname.md" if fault == "newline_same_final" else "new\tname.md"
    if fault in {"newline_same_final", "tab_same_final"}:
        (folder / filename).write_text("approved unusual filename\n")
        git("add", ".")
        git("commit", "-m", "approved unusual main path")
        main = pin()
    git("checkout", "candidate")
    target = folder / "approved.md"
    if fault in {"edit_md", "edit_json", "add_md"}:
        name = {"edit_md": "approved.md", "edit_json": "data.json", "add_md": "unexpected.md"}[
            fault
        ]
        (folder / name).write_text('{"candidate":true}\n')
    elif fault == "delete":
        target.unlink()
    elif fault == "symlink":
        target.unlink()
        target.symlink_to("data.json")
    elif fault == "mode":
        target.chmod(0o755)
    elif fault == "root_file":
        for child in folder.iterdir():
            child.unlink()
        folder.rmdir()
        folder.write_text("candidate root replacement\n")
    elif fault == "same_final":
        target.write_text("approved main bytes\n")
    else:
        (folder / filename).write_text("approved unusual filename\n")
    git("add", ".")
    git("commit", "-m", "candidate changed closed tree")
    candidate = git("rev-parse", "HEAD")
    with pytest.raises(ValueError, match="candidate changed reviewed closed tree"):
        eq.classify_scope([], main, candidate)


@pytest.mark.parametrize("fault", ["edit_md", "add_md", "symlink", "mode", "root_file", "raw_hash"])
def test_closed_tree_refuses_changed_current_main(closed_tree_repo, fault):
    git, folder, main, candidate, _ = closed_tree_repo
    target = folder / "approved.md"
    if fault == "raw_hash":
        eq.REVIEWED_CLOSED_TREE["sha256"] = "0" * 64
    else:
        if fault == "edit_md":
            target.write_text("unreviewed new main bytes\n")
        elif fault == "add_md":
            (folder / "unexpected.md").write_text("unreviewed addition\n")
        elif fault == "symlink":
            target.unlink()
            target.symlink_to("data.json")
        elif fault == "mode":
            target.chmod(0o755)
        else:
            for child in folder.iterdir():
                child.unlink()
            folder.rmdir()
            folder.write_text("unreviewed root replacement\n")
        git("add", ".")
        git("commit", "-m", "main changed closed tree")
        main = git("rev-parse", "HEAD")
    with pytest.raises(ValueError, match="reviewed closed tree"):
        eq.classify_scope([], main, candidate)
