"""No-duplicate-helper contracts for the issue #9250 consolidation.

Family 1 delegates file-anchored ``_repo_root()`` copies to
``robot_sf.common.artifact_paths.get_repository_root`` (the git-anchored
``robot_sf.evidence.writers`` copy intentionally stays: worktree root differs
from the file-anchored root). Family 2 delegates file-hash copies to
``robot_sf.benchmark.identity.hash_utils.sha256_file``. Family 3 delegates
timestamp copies to ``scripts.dev.time_utils.utc_now_iso``.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]

_FAMILY_1_DELEGATED = (
    "robot_sf/benchmark/case_portfolio.py",
    "robot_sf/benchmark/scenario_denominator_manifest.py",
)

_FAMILY_2_DELEGATED = (
    "robot_sf/benchmark/issue_6971_safety_wrapper_preregistration.py",
    "robot_sf/benchmark/issue_6412_real_reexport.py",
    "robot_sf/benchmark/trace_dossier_package.py",
    "robot_sf/analysis_workbench/trace_dossier_renderer.py",
    "robot_sf/benchmark/case_dossier_figure.py",
)

_FAMILY_3_DELEGATED = (
    "scripts/benchmark/prepare_radius_sweep_admission_issue_7198.py",
    "scripts/dev/blocker_receipt.py",
    "scripts/validation/run_release_functional_badge_smoke.py",
    "scripts/validation/check_release_artifact_badging.py",
)


def _top_level_defs(path: Path) -> set[str]:
    """Return the top-level function names defined in a source file."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    return {
        node.name for node in tree.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }


@pytest.mark.parametrize("rel", _FAMILY_1_DELEGATED)
def test_repo_root_copies_stay_delegated(rel: str) -> None:
    """File-anchored ``_repo_root`` copies must not reappear (issue #9250)."""
    assert "_repo_root" not in _top_level_defs(REPO_ROOT / rel)


def test_writers_keeps_git_anchored_root_with_contract_note() -> None:
    """The worktree-root copy stays, with its documented reason (issue #9250)."""
    path = REPO_ROOT / "robot_sf/evidence/writers.py"
    assert "_repo_root" in _top_level_defs(path)
    assert "worktree root" in path.read_text(encoding="utf-8")


@pytest.mark.parametrize("rel", _FAMILY_2_DELEGATED)
def test_sha256_copies_stay_delegated(rel: str) -> None:
    """File-hash helper copies must not reappear (issue #9250)."""
    defs = _top_level_defs(REPO_ROOT / rel)
    assert not ({"_sha256", "_sha256_file", "_file_sha256", "sha256_file"} & defs)


@pytest.mark.parametrize("rel", _FAMILY_3_DELEGATED)
def test_timestamp_copies_stay_delegated(rel: str) -> None:
    """Timestamp helper copies must not reappear (issue #9250)."""
    defs = _top_level_defs(REPO_ROOT / rel)
    assert not ({"_utc_now", "_utc_now_iso", "_now_utc", "utc_now_iso"} & defs)


def test_timestamp_contract_format() -> None:
    """The shared timestamp contract emits ISO-8601 Z-suffixed strings."""
    import sys

    sys.path.insert(0, str(REPO_ROOT))
    from scripts.dev.time_utils import utc_now_iso

    value = utc_now_iso()
    assert value.endswith("Z")
    assert "T" in value
