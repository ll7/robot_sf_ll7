"""Tests for the learned probabilistic graph predictor v1 capability inventory (#2844).

These tests run entirely on synthetic metadata and an injected importer; they do not
train, run a predictor, or execute any benchmark. They cover hook resolution, report
shape, the constant unblock boundary, and the real default hook set wiring in.
"""

from __future__ import annotations

from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

from robot_sf.benchmark.learned_predictor_capability_inventory import (
    ISSUE,
    LANE,
    LEARNED_PREDICTOR_V1_HOOKS,
    READINESS_EVIDENCE_GATE,
    CapabilityHook,
    build_inventory,
    iter_hook_targets,
    resolve_hook,
)


def _fake_importer(modules: dict[str, ModuleType]):
    """Return an importer that serves only the provided synthetic modules."""

    def _import(name: str) -> ModuleType:
        if name not in modules:
            raise ModuleNotFoundError(name)
        return modules[name]

    return _import


def test_importable_hook_present_when_symbol_resolves():
    """An importable hook is present when the symbol exists on the module."""
    module = SimpleNamespace(Thing=object())
    hook = CapabilityHook(
        name="thing",
        category="interface",
        requirement="importable",
        target="pkg.mod:Thing",
        description="synthetic",
    )
    status = resolve_hook(
        hook,
        repo_root=Path("/nonexistent"),
        importer=_fake_importer({"pkg.mod": module}),  # type: ignore[arg-type]
    )
    assert status.present is True
    assert "pkg.mod:Thing" in status.detail


def test_importable_hook_missing_when_symbol_absent():
    """A present module with a missing attribute resolves as not present."""
    module = SimpleNamespace(Other=object())
    hook = CapabilityHook(
        name="thing",
        category="interface",
        requirement="importable",
        target="pkg.mod:Thing",
        description="synthetic",
    )
    status = resolve_hook(
        hook,
        repo_root=Path("/nonexistent"),
        importer=_fake_importer({"pkg.mod": module}),  # type: ignore[arg-type]
    )
    assert status.present is False
    assert "no attribute" in status.detail


def test_importable_hook_missing_on_import_failure():
    """An import failure is reported as missing, not raised."""
    hook = CapabilityHook(
        name="thing",
        category="interface",
        requirement="importable",
        target="pkg.absent:Thing",
        description="synthetic",
    )
    status = resolve_hook(hook, repo_root=Path("/nonexistent"), importer=_fake_importer({}))
    assert status.present is False
    assert "import failed" in status.detail


def test_malformed_importable_target_raises():
    """A target without a 'module:symbol' shape is a programming error."""
    hook = CapabilityHook(
        name="bad",
        category="interface",
        requirement="importable",
        target="no_symbol_here",
        description="synthetic",
    )
    with pytest.raises(ValueError, match="module:symbol"):
        resolve_hook(hook, repo_root=Path("/nonexistent"), importer=_fake_importer({}))


def test_file_hook_presence_against_synthetic_repo(tmp_path: Path):
    """File hooks resolve against the provided repo root."""
    (tmp_path / "scripts").mkdir()
    (tmp_path / "scripts" / "present.py").write_text("# present\n", encoding="utf-8")
    present = CapabilityHook(
        name="present",
        category="dataset",
        requirement="file",
        target="scripts/present.py",
        description="synthetic",
    )
    absent = CapabilityHook(
        name="absent",
        category="dataset",
        requirement="file",
        target="scripts/absent.py",
        description="synthetic",
    )
    assert resolve_hook(present, repo_root=tmp_path, importer=_fake_importer({})).present is True
    assert resolve_hook(absent, repo_root=tmp_path, importer=_fake_importer({})).present is False


def test_build_inventory_report_shape_and_counts(tmp_path: Path):
    """The report summarizes counts, lists missing hooks, and stays honest on unblock."""
    (tmp_path / "exists.txt").write_text("x\n", encoding="utf-8")
    module = SimpleNamespace(Symbol=object())
    hooks = (
        CapabilityHook("a", "interface", "importable", "m:Symbol", "d"),
        CapabilityHook("b", "interface", "importable", "m:Missing", "d"),
        CapabilityHook("c", "dataset", "file", "exists.txt", "d"),
        CapabilityHook("d", "dataset", "file", "missing.txt", "d"),
    )
    report = build_inventory(
        hooks,
        repo_root=tmp_path,
        importer=_fake_importer({"m": module}),  # type: ignore[arg-type]
    )
    assert report["issue"] == ISSUE
    assert report["lane"] == LANE
    assert report["summary"] == {"total": 4, "present": 2, "missing": 2}
    assert report["capability_status"] == "incomplete"
    assert sorted(report["missing_hooks"]) == ["b", "d"]
    # The inventory never claims to unblock training, regardless of completeness.
    assert report["unblocks_training"] is False
    assert report["unblock_owner"] == READINESS_EVIDENCE_GATE


def test_build_inventory_complete_when_all_present(tmp_path: Path):
    """All-present hooks yield a 'complete' wiring status — still not an unblock claim."""
    module = SimpleNamespace(Symbol=object())
    hooks = (CapabilityHook("a", "interface", "importable", "m:Symbol", "d"),)
    report = build_inventory(
        hooks,
        repo_root=tmp_path,
        importer=_fake_importer({"m": module}),  # type: ignore[arg-type]
    )
    assert report["capability_status"] == "complete"
    assert report["missing_hooks"] == []
    assert report["unblocks_training"] is False


def test_default_hooks_cover_expected_categories():
    """The canonical hook set spans every prerequisite surface from the issue."""
    categories = {hook.category for hook in LEARNED_PREDICTOR_V1_HOOKS}
    assert categories == {"interface", "contract", "dataset", "registry", "readiness_gate"}
    # Every importable target is a well-formed module:symbol reference.
    for target in iter_hook_targets():
        importable = ":" in target and not target.endswith(".py")
        if importable:
            module_name, _, symbol = target.partition(":")
            assert module_name and symbol


def test_default_inventory_resolves_against_real_checkout():
    """Against the real repo the canonical hooks resolve and remain unblock-neutral.

    This is a wiring check, not a readiness claim: a complete inventory matches the
    2026-06-23 audit finding that the lane is blocked on evidence, not on missing hooks.
    """
    repo_root = Path(__file__).resolve().parents[2]
    report = build_inventory(LEARNED_PREDICTOR_V1_HOOKS, repo_root=repo_root)
    assert report["summary"]["total"] == len(LEARNED_PREDICTOR_V1_HOOKS)
    assert report["unblocks_training"] is False
    # Missing hooks (if any) must be reported by name so blockers stay visible.
    assert set(report["missing_hooks"]).issubset({h.name for h in LEARNED_PREDICTOR_V1_HOOKS})
