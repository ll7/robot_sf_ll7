"""Focused tests for the compat import-profile probe (issue #9355)."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

from scripts.dev.check_compat_import_profile import check_profile


def _write_tree(root: Path, files: dict[str, str]) -> None:
    for relative, text in files.items():
        target = root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text, encoding="utf-8")


def test_clean_lane_passes(tmp_path: Path) -> None:
    """In-profile imports (numpy) pass with no errors."""
    _write_tree(tmp_path, {"tests/common/test_ok.py": "import numpy\n"})
    errors, report = check_profile(tmp_path)
    assert errors == []
    assert "numpy" in report


def test_heavy_collection_import_fails_with_module_name(tmp_path: Path) -> None:
    """A torch import in the lane fails naming the module and the remedy."""
    _write_tree(tmp_path, {"tests/sim/test_heavy.py": "import definitely_not_installed_xyz\n"})
    errors, _report = check_profile(tmp_path)
    assert len(errors) == 1
    assert "definitely_not_installed_xyz" in errors[0]
    assert "compat profile" in errors[0]


def test_guarded_import_is_tolerated(tmp_path: Path) -> None:
    """try/except ImportError imports never veto the lane."""
    _write_tree(
        tmp_path,
        {
            "tests/nav/test_guarded.py": (
                "try:\n    import definitely_not_installed_xyz\nexcept ImportError:\n    pass\n"
            )
        },
    )
    errors, _report = check_profile(tmp_path)
    assert errors == []


def test_unknown_importorskip_fails_closed(tmp_path: Path) -> None:
    """A silent skip on an out-of-profile module fails with its name."""
    _write_tree(
        tmp_path,
        {
            "tests/render/test_skippy.py": (
                "import pytest\npytest.importorskip('definitely_not_installed_xyz')\n"
            )
        },
    )
    errors, _report = check_profile(tmp_path)
    assert len(errors) == 1
    assert "definitely_not_installed_xyz" in errors[0]
    assert "silently skips" in errors[0]


def test_accepted_skips_do_not_fail(tmp_path: Path) -> None:
    """Playwright/sklearn skips are explicit policy, not violations."""
    _write_tree(
        tmp_path,
        {
            "tests/render/test_viewer.py": (
                "import pytest\npytest.importorskip('playwright.sync_api')\n"
            ),
            "tests/unit/test_annot.py": ("import pytest\npytest.importorskip('sklearn')\n"),
        },
    )
    errors, _report = check_profile(tmp_path)
    assert errors == []


def test_transitive_owner_import_is_checked(tmp_path: Path) -> None:
    """A heavy import reached through an in-repo owner fails with the path."""
    _write_tree(
        tmp_path,
        {
            "tests/sim/test_owner.py": "from robot_sf.nav.heavy_thing import x\n",
            "robot_sf/nav/heavy_thing.py": "import definitely_not_installed_xyz\n",
            "robot_sf/nav/__init__.py": "",
        },
    )
    errors, _report = check_profile(tmp_path)
    assert any(
        "heavy_thing" in error and "definitely_not_installed_xyz" in error for error in errors
    )
