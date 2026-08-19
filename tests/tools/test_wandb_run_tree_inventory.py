"""Tests for the symlink-safe W&B run-tree inventory and retirement planner."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest

from scripts.tools.wandb_run_tree_inventory import (
    DIRECTORY,
    REGULAR_FILE,
    SCHEMA_VERSION,
    SYMLINK_BROKEN,
    SYMLINK_CONTAINED,
    SYMLINK_EXTERNAL,
    build_inventory,
    main,
    validate_retirement_plan,
)

if TYPE_CHECKING:
    from pathlib import Path

RUN_NAME = "offline-run-20260818_120000-abcd1234"
RUN_DIR_REL = f"wandb/{RUN_NAME}"


def _write(path: Path, payload: str) -> None:
    """Write a small fixture file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(payload, encoding="utf-8")


def _make_run_tree(tmp_path: Path) -> Path:
    """Create a representative W&B-like run tree under ``tmp_path``."""
    root = tmp_path / "wandb_output"
    run_dir = root / RUN_DIR_REL
    _write(run_dir / "files" / "config.yaml", "seed: 7\n")
    _write(run_dir / "files" / "model" / "best.pt", "checkpoint-bytes")
    _write(run_dir / "logs" / "output.log", "training log line\n")
    _write(root / "README.txt", "synthetic run tree\n")
    return root


def _file_paths(report) -> list[str]:
    """Return every regular-file path in an inventory report."""
    return [entry.relative_path for entry in report.entries if entry.object_class == REGULAR_FILE]


def _leaf_paths(report) -> list[str]:
    """Return every non-directory path in an inventory report."""
    return [entry.relative_path for entry in report.entries if entry.object_class != DIRECTORY]


def test_clean_tree_inventory_reports_files_and_dirs(tmp_path: Path) -> None:
    """A clean tree keeps the plain inventory behavior: files and dirs, no blockers."""
    report = build_inventory(_make_run_tree(tmp_path))
    classes = {entry.object_class for entry in report.entries}
    assert classes == {REGULAR_FILE, DIRECTORY}
    assert report.blockers == ()
    assert report.summary[REGULAR_FILE] == 4
    assert report.summary[DIRECTORY] == 5
    assert report.summary["total"] == len(report.entries)
    paths = [entry.relative_path for entry in report.entries]
    assert paths == sorted(paths)
    sizes = {
        entry.relative_path: entry.size_bytes
        for entry in report.entries
        if entry.object_class == REGULAR_FILE
    }
    assert sizes["README.txt"] == len("synthetic run tree\n")
    for entry in report.entries:
        assert entry.link_target is None


def test_contained_relative_symlink_is_not_blocker(tmp_path: Path) -> None:
    """A relative link that stays inside the root is contained and not a blocker."""
    root = _make_run_tree(tmp_path)
    link = root / "wandb" / "latest-run"
    link.symlink_to(RUN_NAME)
    report = build_inventory(root)
    entry = next(e for e in report.entries if e.relative_path == "wandb/latest-run")
    assert entry.object_class == SYMLINK_CONTAINED
    assert entry.link_target == RUN_NAME
    assert entry.blocker is None
    assert report.blockers == ()
    # The link is recorded exactly once and never followed during the scan.
    assert not any(e.relative_path.startswith("wandb/latest-run/") for e in report.entries)


def test_broken_absolute_home_link_is_visible_blocker(tmp_path: Path) -> None:
    """A broken absolute /home/... link is a visible blocker, not a missing file."""
    root = _make_run_tree(tmp_path)
    target = "/home/issue-7444-nonexistent-user/checkpoints/best.pt"
    link = root / RUN_DIR_REL / "files" / "model" / "restored.pt"
    link.symlink_to(target)
    report = build_inventory(root)
    entry = next(e for e in report.entries if e.relative_path.endswith("restored.pt"))
    assert entry.object_class == SYMLINK_BROKEN
    assert entry.link_target == target
    assert entry.blocker is not None
    assert "host path" in entry.blocker
    assert "does not exist" in entry.blocker
    assert any("restored.pt" in blocker for blocker in report.blockers)


def test_symlinked_model_file_outside_root_is_external_blocker(tmp_path: Path) -> None:
    """A symlinked model file pointing outside the root is an external blocker."""
    root = _make_run_tree(tmp_path)
    outside_model = tmp_path / "outside" / "shared_model.pt"
    _write(outside_model, "outside-checkpoint")
    link = root / RUN_DIR_REL / "files" / "model" / "shared.pt"
    link.symlink_to(outside_model)
    report = build_inventory(root)
    entry = next(e for e in report.entries if e.relative_path.endswith("shared.pt"))
    assert entry.object_class == SYMLINK_EXTERNAL
    assert entry.link_target == str(outside_model)
    assert entry.blocker is not None
    assert any("shared.pt" in blocker for blocker in report.blockers)


def test_plan_drift_rejects_extra_tree_file(tmp_path: Path) -> None:
    """An extra tree file absent from the plan is fail-closed drift."""
    report = build_inventory(_make_run_tree(tmp_path))
    plan = _file_paths(report)
    assert validate_retirement_plan(report, plan).ok
    verdict = validate_retirement_plan(report, plan[:-1])
    assert not verdict.ok
    assert any("not covered by retirement plan" in reason for reason in verdict.reasons)


def test_path_escape_attempts_are_rejected(tmp_path: Path) -> None:
    """Escaping '..' link targets and escaping planned paths are both rejected."""
    root = _make_run_tree(tmp_path)
    outside = tmp_path / "escape_target.txt"
    _write(outside, "outside\n")
    link = root / "wandb" / "escape-link"
    link.symlink_to("../../escape_target.txt")
    report = build_inventory(root)
    entry = next(e for e in report.entries if e.relative_path == "wandb/escape-link")
    assert entry.object_class == SYMLINK_EXTERNAL
    assert entry.blocker is not None
    assert "'..'" in entry.blocker
    assert "escapes inventory root" in entry.blocker

    plan = _leaf_paths(report)
    verdict = validate_retirement_plan(report, [*plan, "../escape_target.txt"])
    assert not verdict.ok
    assert any("'..' component" in reason for reason in verdict.reasons)
    abs_verdict = validate_retirement_plan(report, [*plan, str(outside)])
    assert not abs_verdict.ok
    assert any("escapes inventory root" in reason for reason in abs_verdict.reasons)


def test_symlink_root_and_traversal_are_rejected(tmp_path: Path) -> None:
    """A planned path that is a symlink or traverses one is rejected."""
    root = _make_run_tree(tmp_path)
    link = root / "run-link"
    link.symlink_to(RUN_DIR_REL)
    report = build_inventory(root)
    plan = [*_file_paths(report), "run-link"]
    verdict = validate_retirement_plan(report, plan)
    assert not verdict.ok
    assert any("run-link" in reason and "symlink" in reason for reason in verdict.reasons)

    traversal_verdict = validate_retirement_plan(
        report, [*_file_paths(report), "run-link/files/config.yaml"]
    )
    assert not traversal_verdict.ok
    assert any("traverses symlink ancestor" in reason for reason in traversal_verdict.reasons)


def test_build_inventory_rejects_symlink_root(tmp_path: Path) -> None:
    """Scanning through a symlinked root fails closed."""
    real_root = _make_run_tree(tmp_path)
    link_root = tmp_path / "link_root"
    link_root.symlink_to(real_root)
    with pytest.raises(ValueError, match="must not be a symlink"):
        build_inventory(link_root)


def test_allowlisted_link_target_is_accepted(tmp_path: Path) -> None:
    """A plan covering a symlink fails closed until the verbatim target is allowlisted."""
    root = _make_run_tree(tmp_path)
    outside_model = tmp_path / "outside" / "shared_model.pt"
    _write(outside_model, "outside-checkpoint")
    link = root / RUN_DIR_REL / "files" / "model" / "shared.pt"
    link.symlink_to(outside_model)
    report = build_inventory(root)
    plan = _leaf_paths(report)
    verdict = validate_retirement_plan(report, plan)
    assert not verdict.ok
    assert any("not in allowlist" in reason for reason in verdict.reasons)
    allowed = validate_retirement_plan(
        report, plan, allowed_link_targets=frozenset({str(outside_model)})
    )
    assert allowed.ok
    assert allowed.reasons == ()


def test_scan_is_deterministic(tmp_path: Path) -> None:
    """Two consecutive scans of the same tree produce identical receipts."""
    root = _make_run_tree(tmp_path)
    (root / "wandb" / "latest-run").symlink_to(RUN_NAME)
    (root / "wandb" / "broken").symlink_to("/home/issue-7444-nonexistent-user/best.pt")
    first = build_inventory(root)
    second = build_inventory(root)
    assert first == second
    assert first.entries == second.entries
    assert first.summary == second.summary
    assert first.blockers == second.blockers


def test_cli_json_receipt(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """The CLI prints a deterministic JSON receipt for a clean tree."""
    root = _make_run_tree(tmp_path)
    exit_code = main([str(root), "--json"])
    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["schema_version"] == SCHEMA_VERSION
    assert payload["report"]["blockers"] == []
    assert payload["report"]["root"] == str(root)
