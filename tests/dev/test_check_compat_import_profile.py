"""Focused tests for the compat import-profile probe (issue #9355)."""

from __future__ import annotations

from pathlib import Path

import scripts.dev.check_compat_import_profile as compat_profile
from scripts.dev.check_compat_import_profile import COMPAT_TEST_DIRS, check_profile


def _write_tree(root: Path, files: dict[str, str]) -> None:
    for dirname in COMPAT_TEST_DIRS:
        directory = root / dirname
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "test_compat_placeholder.py").write_text("", encoding="utf-8")
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


def test_import_error_guard_does_not_exempt_nested_function_or_try(
    tmp_path: Path,
) -> None:
    """An outer guard exempts only imports directly covered by its body."""
    _write_tree(
        tmp_path,
        {
            "tests/common/test_nested_guard.py": (
                "try:\n"
                "    import definitely_not_installed_optional\n"
                "    def deferred():\n"
                "        import definitely_not_installed_function\n"
                "    try:\n"
                "        import definitely_not_installed_nested\n"
                "    except ValueError:\n"
                "        pass\n"
                "except ImportError:\n"
                "    pass\n"
            )
        },
    )
    errors, _report = check_profile(tmp_path)
    assert any("definitely_not_installed_function" in error for error in errors)
    assert any("definitely_not_installed_nested" in error for error in errors)
    assert not any("definitely_not_installed_optional" in error for error in errors)


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


def test_deferred_local_import_joins_transitive_closure(tmp_path: Path) -> None:
    """A local import reached only when a test runs still exposes owner imports."""
    _write_tree(
        tmp_path,
        {
            "tests/common/test_deferred_owner.py": (
                "def test_deferred_owner():\n"
                "    from robot_sf.synthetic.owner import value\n"
                "    assert value\n"
            ),
            "robot_sf/synthetic/__init__.py": "",
            "robot_sf/synthetic/owner.py": "import definitely_not_installed_xyz\nvalue = 1\n",
        },
    )
    errors, _report = check_profile(tmp_path)
    assert any(
        "robot_sf/synthetic/owner.py" in error and "definitely_not_installed_xyz" in error
        for error in errors
    )


def test_deferred_owner_import_fails_with_module_name(tmp_path: Path) -> None:
    """An owner function import cannot hide an unavailable runtime dependency."""
    _write_tree(
        tmp_path,
        {
            "tests/common/test_deferred_owner_runtime.py": (
                "from robot_sf.synthetic.owner import value\nassert value\n"
            ),
            "robot_sf/synthetic/__init__.py": "",
            "robot_sf/synthetic/owner.py": (
                "def load():\n"
                "    import definitely_not_installed_xyz\n"
                "    return definitely_not_installed_xyz\n"
                "value = 1\n"
            ),
        },
    )
    errors, _report = check_profile(tmp_path)
    assert any(
        "robot_sf/synthetic/owner.py" in error
        and "definitely_not_installed_xyz" in error
        and "deferred owner execution" in error
        for error in errors
    )


def test_deferred_owner_importorskip_fails_with_module_name(tmp_path: Path) -> None:
    """An owner importorskip target must use an explicitly accepted exemption."""
    _write_tree(
        tmp_path,
        {
            "tests/common/test_deferred_owner_skip.py": (
                "from robot_sf.synthetic.owner import value\nassert value\n"
            ),
            "robot_sf/synthetic/__init__.py": "",
            "robot_sf/synthetic/owner.py": (
                "import pytest\n"
                "def load():\n"
                "    return pytest.importorskip('definitely_not_installed_xyz')\n"
                "value = 1\n"
            ),
        },
    )
    errors, _report = check_profile(tmp_path)
    assert any(
        "robot_sf/synthetic/owner.py" in error
        and "definitely_not_installed_xyz" in error
        and "first-party owner" in error
        for error in errors
    )


def test_dynamic_import_boundary_is_not_static_dependency(tmp_path: Path) -> None:
    """Runtime-selected importlib modules remain outside static AST closure."""
    _write_tree(
        tmp_path,
        {
            "tests/common/test_dynamic_owner.py": (
                "from robot_sf.synthetic.owner import load\nassert load\n"
            ),
            "robot_sf/synthetic/__init__.py": "",
            "robot_sf/synthetic/owner.py": (
                "import importlib\ndef load(name):\n    return importlib.import_module(name)\n"
            ),
        },
    )
    errors, _report = check_profile(tmp_path)
    assert errors == []


def test_relative_import_joins_transitive_closure(tmp_path: Path) -> None:
    """Relative package imports are resolved before scanning their owner files."""
    _write_tree(
        tmp_path,
        {
            "tests/common/test_relative_owner.py": (
                "from robot_sf.synthetic import value\n"
                "\n"
                "def test_relative_owner():\n"
                "    assert value\n"
            ),
            "robot_sf/synthetic/__init__.py": "from .owner import value\n",
            "robot_sf/synthetic/owner.py": "import definitely_not_installed_xyz\nvalue = 1\n",
        },
    )
    errors, _report = check_profile(tmp_path)
    assert any(
        "robot_sf/synthetic/owner.py" in error and "definitely_not_installed_xyz" in error
        for error in errors
    )


def test_guarded_sibling_import_is_not_hidden(tmp_path: Path) -> None:
    """A guarded optional import cannot hide a hard sibling in the same try body."""
    _write_tree(
        tmp_path,
        {
            "tests/common/test_guarded_sibling.py": (
                "try:\n"
                "    import definitely_not_installed_optional\n"
                "    import definitely_not_installed_sibling\n"
                "except ImportError:\n"
                "    pass\n"
            )
        },
    )
    errors, _report = check_profile(tmp_path)
    assert any("definitely_not_installed_optional" in error for error in errors)
    assert any("definitely_not_installed_sibling" in error for error in errors)


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
    for path in (tmp_path / "tests" / "common").glob("*.py"):
        path.unlink()
    (tmp_path / "tests" / "common").rmdir()
    errors, _report = check_profile(tmp_path)
    assert any("tests/common" in error and "directory is missing" in error for error in errors)


def test_empty_selected_directory_fails_closed(tmp_path: Path) -> None:
    """A present but empty lane root cannot become an empty-success scan."""
    _write_tree(tmp_path, {"tests/sim/test_ok.py": "import numpy\n"})
    for path in (tmp_path / "tests" / "common").glob("*.py"):
        path.unlink()
    errors, _report = check_profile(tmp_path)
    assert any("tests/common" in error and "contains no selected" in error for error in errors)


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


def test_unreadable_nested_directory_fails_closed(tmp_path: Path, monkeypatch) -> None:
    """A traversal error from a nested directory must be visible to the caller."""
    _write_tree(tmp_path, {"tests/common/test_ok.py": "import numpy\n"})
    target = tmp_path / "tests" / "common"
    original_walk = compat_profile.os.walk

    def fail_nested_directory(path, *args, **kwargs):
        if Path(path) == target:
            onerror = kwargs.get("onerror")
            if onerror is not None:
                onerror(PermissionError("nested directory is unreadable"))
            return iter(())
        return original_walk(path, *args, **kwargs)

    monkeypatch.setattr(compat_profile.os, "walk", fail_nested_directory)
    errors, _report = check_profile(tmp_path)
    assert any("tests/common" in error and "cannot enumerate" in error for error in errors)


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
