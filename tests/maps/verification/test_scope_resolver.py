"""Direct unit coverage for :mod:`robot_sf.maps.verification.scope_resolver`.

These tests lock the public scope-resolution contracts of :class:`ScopeResolver`
using a synthetic :class:`MapInventory` backed entirely by ``tmp_path``. No
repository map assets are read or modified.

Locked contracts
----------------
- Whitespace and case normalization of the scope specifier before dispatch.
- ``all`` -> every map in the inventory.
- ``ci`` -> only ``ci_enabled`` maps.
- Exact map id (``"<id>"``) and ``"<id>.svg"`` filename routes.
- Glob routes using ``*`` and ``?``.
- Fail-closed errors: a missing specific id and an empty glob match both raise
  an actionable :class:`ValueError`.
- ``changed`` scope: the git subprocess is mocked; only changed SVG files that
  resolve under the configured maps root are included, and a git failure or a
  missing git executable both fall back to the ``all`` scope.

Evidence boundary
-----------------
This is test coverage only. No claim is made that any real map scope was
executed, and no benchmark, conversion, metric, or map asset is touched.
"""

from __future__ import annotations

from subprocess import CalledProcessError, CompletedProcess
from typing import TYPE_CHECKING

import pytest

from robot_sf.maps.verification.map_inventory import MapInventory
from robot_sf.maps.verification.scope_resolver import ScopeResolver

if TYPE_CHECKING:
    from pathlib import Path

# Module-qualified target for mocking ``subprocess.run`` exactly where the
# resolver calls it, so unrelated subprocess users are not disturbed.
SCOPE_MODULE = "robot_sf.maps.verification.scope_resolver"


def _write_svg(path: Path) -> Path:
    """Write a minimal valid SVG file at ``path`` (parent dirs created)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<svg xmlns="http://www.w3.org/2000/svg" width="10" height="10"></svg>\n',
        encoding="utf-8",
    )
    return path


def _make_inventory(tmp_path: Path, *map_ids: str) -> MapInventory:
    """Build a synthetic inventory under ``tmp_path`` with the named maps."""
    for map_id in map_ids:
        _write_svg(tmp_path / f"{map_id}.svg")
    return MapInventory(maps_root=tmp_path)


def _ids(maps: list) -> set[str]:
    """Return the set of map ids for a list of :class:`MapRecord` objects."""
    return {m.map_id for m in maps}


def _fake_git_run(stdout: str = ""):
    """Return a ``subprocess.run`` replacement that yields ``stdout``."""

    def _run(cmd, **_kwargs):
        return CompletedProcess(cmd, returncode=0, stdout=stdout, stderr="")

    return _run


def _raise_called_process_error(cmd, **_kwargs):
    """Simulate a failing git command (non-zero exit)."""
    raise CalledProcessError(returncode=128, cmd=cmd)


def _raise_file_not_found(cmd, **_kwargs):
    """Simulate the git executable being absent from the system."""
    raise FileNotFoundError(2, "No such file or directory", "git")


class TestScopeNormalization:
    """The scope string is stripped and lower-cased before route dispatch."""

    def test_all_keyword_normalizes_whitespace_and_case(self, tmp_path):
        inv = _make_inventory(tmp_path, "alpha", "beta")
        resolver = ScopeResolver(inv)
        assert _ids(resolver.resolve("  ALL  ")) == {"alpha", "beta"}

    def test_ci_keyword_normalizes_whitespace_and_case(self, tmp_path):
        inv = _make_inventory(tmp_path, "alpha", "beta")
        inv.get_map_by_id("alpha").ci_enabled = False
        resolver = ScopeResolver(inv)
        assert _ids(resolver.resolve("\tCi\n")) == {"beta"}

    def test_changed_keyword_normalizes_whitespace_and_case(self, tmp_path, monkeypatch):
        inv = _make_inventory(tmp_path, "alpha", "beta")
        resolver = ScopeResolver(inv)
        monkeypatch.setattr(
            f"{SCOPE_MODULE}.subprocess.run",
            _fake_git_run(str((tmp_path / "alpha.svg").resolve())),
        )
        assert _ids(resolver.resolve("  CHANGED ")) == {"alpha"}


class TestAllScope:
    """``all`` returns every map known to the inventory."""

    def test_all_returns_every_map(self, tmp_path):
        inv = _make_inventory(tmp_path, "alpha", "beta", "gamma")
        resolver = ScopeResolver(inv)
        assert _ids(resolver.resolve("all")) == {"alpha", "beta", "gamma"}

    def test_all_empty_inventory_returns_empty_list(self, tmp_path):
        inv = MapInventory(maps_root=tmp_path)
        resolver = ScopeResolver(inv)
        assert resolver.resolve("all") == []


class TestCiScope:
    """``ci`` returns only maps whose ``ci_enabled`` flag is set."""

    def test_ci_returns_only_enabled_maps(self, tmp_path):
        inv = _make_inventory(tmp_path, "alpha", "beta", "gamma")
        inv.get_map_by_id("beta").ci_enabled = False
        resolver = ScopeResolver(inv)
        assert _ids(resolver.resolve("ci")) == {"alpha", "gamma"}

    def test_ci_all_disabled_returns_empty_list(self, tmp_path):
        inv = _make_inventory(tmp_path, "alpha")
        inv.get_map_by_id("alpha").ci_enabled = False
        resolver = ScopeResolver(inv)
        assert resolver.resolve("ci") == []


class TestSpecificScope:
    """Exact id and ``.svg`` filename routes resolve a single map."""

    def test_exact_id_without_extension(self, tmp_path):
        inv = _make_inventory(tmp_path, "alpha", "beta")
        resolver = ScopeResolver(inv)
        assert [m.map_id for m in resolver.resolve("alpha")] == ["alpha"]

    def test_svg_filename_route(self, tmp_path):
        inv = _make_inventory(tmp_path, "alpha", "beta")
        resolver = ScopeResolver(inv)
        assert [m.map_id for m in resolver.resolve("beta.svg")] == ["beta"]

    def test_specific_id_is_case_normalized(self, tmp_path):
        inv = _make_inventory(tmp_path, "alpha")
        resolver = ScopeResolver(inv)
        assert [m.map_id for m in resolver.resolve("ALPHA.SVG")] == ["alpha"]
        assert [m.map_id for m in resolver.resolve("Alpha")] == ["alpha"]

    def test_missing_svg_filename_raises_actionable_value_error(self, tmp_path):
        inv = _make_inventory(tmp_path, "alpha")
        resolver = ScopeResolver(inv)
        with pytest.raises(ValueError, match=r"Map not found: ghost\.svg"):
            resolver.resolve("ghost.svg")

    def test_missing_id_without_extension_raises_actionable_value_error(self, tmp_path):
        inv = _make_inventory(tmp_path, "alpha")
        resolver = ScopeResolver(inv)
        with pytest.raises(ValueError, match=r"Map not found: ghost\.svg"):
            resolver.resolve("ghost")


class TestGlobScope:
    """Glob routes match map filenames via ``*`` and ``?``."""

    def test_star_glob_matches_subset(self, tmp_path):
        inv = _make_inventory(tmp_path, "classic_one", "classic_two", "other")
        resolver = ScopeResolver(inv)
        assert _ids(resolver.resolve("classic_*.svg")) == {"classic_one", "classic_two"}

    def test_question_glob_matches_single_char(self, tmp_path):
        inv = _make_inventory(tmp_path, "map_a", "map_b", "map_ab")
        resolver = ScopeResolver(inv)
        # "?" matches exactly one character, so map_ab is excluded.
        assert _ids(resolver.resolve("map_?.svg")) == {"map_a", "map_b"}

    def test_glob_pattern_is_case_normalized(self, tmp_path):
        inv = _make_inventory(tmp_path, "classic_one", "classic_two")
        resolver = ScopeResolver(inv)
        assert _ids(resolver.resolve("CLASSIC_*.SVG")) == {"classic_one", "classic_two"}

    def test_empty_glob_match_raises_actionable_value_error(self, tmp_path):
        inv = _make_inventory(tmp_path, "alpha")
        resolver = ScopeResolver(inv)
        with pytest.raises(ValueError, match=r"No maps match pattern: zzz_\*\.svg"):
            resolver.resolve("zzz_*.svg")


class TestChangedScope:
    """``changed`` consults git (mocked) and filters to maps-root SVG files."""

    def test_includes_only_changed_svgs_under_maps_root(
        self, tmp_path, tmp_path_factory, monkeypatch
    ):
        inv = _make_inventory(tmp_path, "alpha", "beta")
        # A non-SVG file under the maps root must be ignored.
        (tmp_path / "notes.txt").write_text("x", encoding="utf-8")
        # An SVG that resolves outside the configured maps root must be ignored.
        outside = tmp_path_factory.mktemp("outside")
        _write_svg(outside / "ghost.svg")
        # beta exists in the inventory but is not reported as changed.
        stdout = "\n".join(
            [
                str((tmp_path / "alpha.svg").resolve()),
                str((tmp_path / "notes.txt").resolve()),
                str((outside / "ghost.svg").resolve()),
            ]
        )
        monkeypatch.setattr(f"{SCOPE_MODULE}.subprocess.run", _fake_git_run(stdout))
        resolver = ScopeResolver(inv)
        assert _ids(resolver.resolve("changed")) == {"alpha"}

    def test_non_svg_changed_file_yields_empty_list(self, tmp_path, monkeypatch):
        inv = _make_inventory(tmp_path, "alpha", "beta")
        (tmp_path / "notes.txt").write_text("x", encoding="utf-8")
        monkeypatch.setattr(
            f"{SCOPE_MODULE}.subprocess.run",
            _fake_git_run(str((tmp_path / "notes.txt").resolve())),
        )
        resolver = ScopeResolver(inv)
        # No matches is not an error for the changed scope (unlike glob/specific).
        assert resolver.resolve("changed") == []

    def test_empty_git_output_yields_empty_list(self, tmp_path, monkeypatch):
        inv = _make_inventory(tmp_path, "alpha")
        monkeypatch.setattr(f"{SCOPE_MODULE}.subprocess.run", _fake_git_run(""))
        resolver = ScopeResolver(inv)
        assert resolver.resolve("changed") == []

    def test_git_calledprocess_error_falls_back_to_all(self, tmp_path, monkeypatch):
        inv = _make_inventory(tmp_path, "alpha", "beta")
        monkeypatch.setattr(f"{SCOPE_MODULE}.subprocess.run", _raise_called_process_error)
        resolver = ScopeResolver(inv)
        assert _ids(resolver.resolve("changed")) == {"alpha", "beta"}

    def test_git_executable_missing_falls_back_to_all(self, tmp_path, monkeypatch):
        inv = _make_inventory(tmp_path, "alpha", "beta")
        monkeypatch.setattr(f"{SCOPE_MODULE}.subprocess.run", _raise_file_not_found)
        resolver = ScopeResolver(inv)
        assert _ids(resolver.resolve("changed")) == {"alpha", "beta"}
