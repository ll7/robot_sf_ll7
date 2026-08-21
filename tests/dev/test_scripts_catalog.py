"""Tests for the scripts command catalog, checker, and README renderer."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
CHECKER = REPO_ROOT / "scripts" / "dev" / "check_scripts_catalog.py"
RENDERER = REPO_ROOT / "scripts" / "dev" / "render_scripts_readme.py"

sys.path.insert(0, str(REPO_ROOT / "scripts" / "dev"))
import scripts_catalog  # noqa: E402
from scripts_catalog import (  # noqa: E402
    STATUS_COMPATIBILITY,
    STATUS_DEBUG_ONLY,
    Catalog,
    CatalogCommand,
    inventory_root_commands,
    load_catalog,
    parse_catalog,
    render_readme,
    render_root_status_table,
    validate_catalog,
)

CATALOG = load_catalog(REPO_ROOT)


def _command(**overrides) -> CatalogCommand:
    values = {
        "name": "sample.py",
        "path": "scripts/sample.py",
        "domain": "tooling",
        "status": "canonical",
        "purpose": "Sample command.",
        "invocation": "uv run python scripts/sample.py",
    }
    values.update(overrides)
    return CatalogCommand(**values)


def _catalog_with(
    commands: list[CatalogCommand], repo_files: dict[str, str]
) -> tuple[Catalog, Path]:
    root = repo_root_stub(repo_files)
    catalog = Catalog(version=1, commands={cmd.name: cmd for cmd in commands})
    return catalog, root


def repo_root_stub(files: dict[str, str]) -> Path:
    """Return a temp repo stub containing the given relative files."""
    import tempfile

    root = Path(tempfile.mkdtemp(prefix="scripts-catalog-test-"))
    for rel, content in files.items():
        target = root / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
    return root


def test_complete_root_inventory_matches_filesystem():
    """Every direct-child executable under scripts/ is covered by the committed catalog."""
    errors = validate_catalog(CATALOG, REPO_ROOT)
    assert not errors, errors


def test_missing_catalog_entry_fails_closed(tmp_path: Path):
    """A new root-level script absent from the catalog produces an error."""
    root = tmp_path
    (root / "scripts").mkdir(parents=True)
    (root / "scripts" / "brand_new.py").write_text("", encoding="utf-8")
    (root / "scripts" / "known.py").write_text("", encoding="utf-8")
    catalog = parse_catalog(
        {
            "version": 1,
            "commands": {
                "known": {
                    "path": "scripts/known.py",
                    "domain": "tooling",
                    "status": "canonical",
                    "purpose": "Known.",
                    "invocation": "uv run python scripts/known.py",
                }
            },
        }
    )
    errors = validate_catalog(catalog, root)
    assert any("brand_new.py" in error for error in errors)


def test_duplicate_command_paths_rejected():
    """Two entries claiming one path fail schema validation."""
    raw = {
        "version": 1,
        "commands": {
            "a": _command().__dict__ | {"path": "scripts/a.py"},
            "b": _command().__dict__ | {"path": "scripts/a.py"},
        },
    }
    try:
        parse_catalog(raw)
    except scripts_catalog.CatalogError as exc:
        assert "duplicate path" in str(exc)
    else:
        raise AssertionError("duplicate path accepted")


def test_broken_replacement_path_rejected(tmp_path: Path):
    """Compatibility entries pointing at non-existent replacements fail."""
    root = tmp_path
    (root / "scripts").mkdir(parents=True)
    (root / "scripts" / "old.py").write_text("", encoding="utf-8")
    catalog = parse_catalog(
        {
            "version": 1,
            "commands": {
                "old": {
                    "path": "scripts/old.py",
                    "domain": "tooling",
                    "status": STATUS_COMPATIBILITY,
                    "purpose": "Old.",
                    "replacement": "scripts/does_not_exist.py",
                    "invocation": "uv run python scripts/old.py",
                }
            },
        }
    )
    errors = validate_catalog(catalog, root)
    assert any("does_not_exist.py" in error for error in errors)


def test_unknown_status_value_rejected():
    """Statuses outside the closed vocabulary fail schema validation."""
    import pytest

    with pytest.raises(scripts_catalog.CatalogError):
        parse_catalog(
            {
                "version": 1,
                "commands": {"x": _command().__dict__ | {"status": "supported"}},
            }
        )


def test_render_is_deterministic():
    """Rendering twice from the same catalog yields identical bytes."""
    first = render_root_status_table(CATALOG)
    second = render_root_status_table(CATALOG)
    assert first == second


def test_committed_readme_matches_catalog_render():
    """The committed README contains exactly the deterministic generated sections."""
    current = (REPO_ROOT / "scripts" / "README.md").read_text(encoding="utf-8")
    assert render_readme(current, CATALOG) == current


def test_canonical_command_help_path_runs():
    """One canonical command exposes a working --help path end to end."""
    proc = subprocess.run(
        [sys.executable, str(REPO_ROOT / "scripts" / "failure_extractor.py"), "--help"],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert proc.returncode == 0
    assert "usage" in (proc.stdout + proc.stderr).lower()


def test_compatibility_fail_closed_migration_output():
    """A retired compatibility entry point fails closed with migration guidance."""
    entry = next(
        cmd
        for cmd in CATALOG.commands.values()
        if cmd.smoke_mode == "expected_fail_closed" and cmd.status == STATUS_COMPATIBILITY
    )
    proc = subprocess.run(
        [sys.executable, str(REPO_ROOT / entry.path)],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    combined = (proc.stdout + proc.stderr).lower()
    assert proc.returncode != 0, f"{entry.path} unexpectedly succeeded"
    assert "training/train_ppo.py" in combined or "migration" in combined or "retired" in combined


def test_debug_only_not_presented_as_supported_workflow():
    """Debug-only rows keep their bounded-utility status in generated output."""
    table = render_root_status_table(CATALOG)
    debug_rows = [line for line in table.splitlines() if f"| {STATUS_DEBUG_ONLY} |" in line]
    assert debug_rows, "expected at least one debug-only row"
    for row in debug_rows:
        assert "canonical" not in row.split("|")[3]


def test_compatibility_rationales_and_readme_links_render_without_duplication():
    """Generated guidance uses rationale text and links directories to local READMEs."""
    table = render_root_status_table(CATALOG)
    assert "Prefer `Prefer " not in table
    assert "| `benchmark_repro_check.py` | compatibility |" in table
    assert "Prefer benchmark release/validation tools under scripts/tools/." in table

    overview = scripts_catalog.render_directory_overview(CATALOG)
    assert "[`scripts/dev/`](dev/README.md)" in overview
    assert "[`scripts/dev/`]( dev/ )" not in overview


def test_inventory_helper_ignores_nested_and_non_executable_files():
    """Only direct .py/.sh children of scripts/ count as root commands."""
    root = repo_root_stub(
        {
            "scripts/a.py": "",
            "scripts/b.sh": "",
            "scripts/c.md": "",
            "scripts/nested/d.py": "",
        }
    )
    assert inventory_root_commands(root) == {"scripts/a.py", "scripts/b.sh"}
