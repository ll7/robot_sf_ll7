"""Tests for the bootstrap-recipe lockfile digest refresh helper."""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from types import ModuleType


SCRIPT = Path(__file__).resolve().parents[2] / "scripts/dev/refresh_bootstrap_recipe_lock_hashes.py"


@pytest.fixture(scope="module")
def helper() -> ModuleType:
    """Load the script under test without requiring scripts/dev to be a package."""

    spec = importlib.util.spec_from_file_location("refresh_bootstrap_recipe_lock_hashes", SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load script: {SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _fixture_tree(tmp_path: Path, declared: str) -> tuple[Path, Path, Path]:
    """Create a tiny recipe tree that binds to a synthetic uv.lock."""

    repo_root = tmp_path / "repo"
    recipe_root = repo_root / "configs/bootstrap_recipes"
    recipe_root.mkdir(parents=True)
    lockfile = repo_root / "uv.lock"
    lockfile.write_bytes(b"version = 1\n")
    recipe = recipe_root / "cpu_batch.v1.json"
    recipe.write_text(
        '{\n  "schema": "bootstrap_recipe.v1",\n'
        f'  "source_identity": {{"lockfile": "uv.lock", "lockfile_sha256": "{declared}"}}\n'
        "}\n",
        encoding="utf-8",
    )
    return repo_root, recipe_root, recipe


def test_check_reports_stale_hash_without_writing(helper: ModuleType, tmp_path: Path) -> None:
    """Check mode reports expected digest and preserves the original bytes."""

    stale = "0" * 64
    repo_root, recipe_root, recipe = _fixture_tree(tmp_path, stale)
    before = recipe.read_bytes()

    changes = helper.refresh(repo_root, recipe_root, write=False)

    assert changes == [
        (recipe, stale, hashlib.sha256((repo_root / "uv.lock").read_bytes()).hexdigest())
    ]
    assert recipe.read_bytes() == before


def test_check_cli_accepts_flag_and_returns_nonzero_for_drift(
    helper: ModuleType,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The explicit check flag reports drift without changing recipe bytes."""

    repo_root, recipe_root, recipe = _fixture_tree(tmp_path, "0" * 64)
    before = recipe.read_bytes()
    monkeypatch.setattr(helper, "REPOSITORY_ROOT", repo_root)
    monkeypatch.setattr(helper, "RECIPE_ROOT", recipe_root)

    assert helper.main(["--check"]) == 1
    assert "Found 1 stale" in capsys.readouterr().out
    assert recipe.read_bytes() == before


def test_write_refreshes_only_the_declared_hash(helper: ModuleType, tmp_path: Path) -> None:
    """Write mode changes the digest while preserving all other recipe bytes."""

    stale = "0" * 64
    repo_root, recipe_root, recipe = _fixture_tree(tmp_path, stale)
    before = recipe.read_text(encoding="utf-8")
    expected = hashlib.sha256((repo_root / "uv.lock").read_bytes()).hexdigest()

    changes = helper.refresh(repo_root, recipe_root, write=True)

    assert changes == [(recipe, stale, expected)]
    assert recipe.read_text(encoding="utf-8") == before.replace(stale, expected)
    assert (
        json.loads(recipe.read_text(encoding="utf-8"))["source_identity"]["lockfile_sha256"]
        == expected
    )
    assert helper.refresh(repo_root, recipe_root, write=False) == []


def test_write_refuses_malformed_hash_without_partial_updates(
    helper: ModuleType, tmp_path: Path
) -> None:
    """Malformed declarations fail closed without rewriting any recipe."""

    repo_root, recipe_root, recipe = _fixture_tree(tmp_path, "not-a-hash")
    before = recipe.read_bytes()
    earlier_valid_recipe = recipe_root / "00_valid.v1.json"
    stale = "0" * 64
    earlier_valid_recipe.write_text(
        '{"source_identity":{"lockfile":"uv.lock","lockfile_sha256":"' + stale + '"}}\n',
        encoding="utf-8",
    )
    earlier_before = earlier_valid_recipe.read_bytes()

    with pytest.raises(ValueError, match="malformed"):
        helper.refresh(repo_root, recipe_root, write=True)

    assert recipe.read_bytes() == before
    assert earlier_valid_recipe.read_bytes() == earlier_before


def test_repository_recipes_are_current(helper: ModuleType) -> None:
    """The tracked recipe set is bound to the committed root uv.lock."""

    assert helper.refresh(helper.REPOSITORY_ROOT, helper.RECIPE_ROOT, write=False) == []
