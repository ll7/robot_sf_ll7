"""Focused tests for the compat import-profile probe (issue #9355)."""

from __future__ import annotations

from pathlib import Path

from scripts.dev.check_compat_import_profile import COMPAT_TEST_DIRS, check_profile


def _write_tree(root: Path, files: dict[str, str]) -> None:
    for dirname in COMPAT_TEST_DIRS:
        (root / dirname).mkdir(parents=True, exist_ok=True)
    (root / "tests" / "conftest.py").write_text("", encoding="utf-8")
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


def test_import_error_handler_and_finally_imports_are_checked(tmp_path: Path) -> None:
    """A tolerated import attempt cannot hide imports in its handler or cleanup."""
    _write_tree(
        tmp_path,
        {
            "tests/common/test_guarded_cleanup.py": (
                "try:\n"
                "    import definitely_not_installed_attempt\n"
                "except ImportError:\n"
                "    import definitely_not_installed_handler\n"
                "finally:\n"
                "    import definitely_not_installed_cleanup\n"
            )
        },
    )
    errors, _report = check_profile(tmp_path)
    assert any("definitely_not_installed_handler" in error for error in errors)
    assert any("definitely_not_installed_cleanup" in error for error in errors)
    assert not any("definitely_not_installed_attempt" in error for error in errors)


def test_deferred_test_import_fails_with_module_name(tmp_path: Path) -> None:
    """A function-body import is part of the lane and must be profile-covered."""
    _write_tree(
        tmp_path,
        {
            "tests/common/test_deferred.py": (
                "def test_deferred():\n    import definitely_not_installed_xyz\n"
            )
        },
    )
    errors, _report = check_profile(tmp_path)
    assert any(
        "definitely_not_installed_xyz" in error and "deferred test execution" in error
        for error in errors
    )


def test_package_initializer_is_scanned(tmp_path: Path) -> None:
    """A selected package initializer can contribute imports to collection."""
    _write_tree(
        tmp_path,
        {"tests/maps/__init__.py": "import definitely_not_installed_xyz\n"},
    )
    errors, _report = check_profile(tmp_path)
    assert any(
        "__init__.py" in error and "definitely_not_installed_xyz" in error for error in errors
    )


def test_missing_selected_directory_fails_closed(tmp_path: Path) -> None:
    """A missing lane root cannot become an empty-success scan."""
    _write_tree(tmp_path, {"tests/sim/test_ok.py": "import numpy\n"})
    (tmp_path / "tests" / "common").rmdir()
    errors, _report = check_profile(tmp_path)
    assert any("tests/common" in error and "directory is missing" in error for error in errors)


def test_syntax_error_fails_closed(tmp_path: Path) -> None:
    """An unreadable AST must veto the profile check with a bounded reason."""
    _write_tree(tmp_path, {"tests/common/test_broken.py": "def broken(:\n"})
    errors, _report = check_profile(tmp_path)
    assert any("test_broken.py" in error and "cannot parse source" in error for error in errors)


def test_read_error_fails_closed(tmp_path: Path, monkeypatch) -> None:
    """A source read failure must veto the profile check instead of disappearing."""
    _write_tree(tmp_path, {"tests/common/test_unreadable.py": "import numpy\n"})
    unreadable = tmp_path / "tests" / "common" / "test_unreadable.py"
    original_read_bytes = Path.read_bytes

    def fail_for_target(path: Path) -> bytes:
        if path == unreadable:
            raise OSError("permission denied")
        return original_read_bytes(path)

    monkeypatch.setattr(Path, "read_bytes", fail_for_target)
    errors, _report = check_profile(tmp_path)
    assert any("test_unreadable.py" in error and "cannot read source" in error for error in errors)


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
