"""Tests for the SocNavBench traversible generation wrapper (issue #4291).

These cover the CI-safe surfaces only: path resolution, skip-if-absent preflight,
``--dry-run`` behavior, idempotent skip, and fail-closed build refusal when the mesh
is not staged. The heavy SocNavBench mesh build is environment-only and is never
imported by any of these paths.
"""

from __future__ import annotations

import hashlib
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from pathlib import Path

from scripts.tools.generate_socnavbench_traversible import (
    EXIT_BLOCKED,
    STATUS_ALREADY_PRESENT,
    STATUS_BLOCKED_MISSING_MESH,
    STATUS_READY,
    TraversibleGenerationError,
    build_traversible,
    main,
    output_tree_checksum,
    preflight,
    resolve_paths,
    sha256_file,
)

# ``--root`` / the ``root`` kwarg is the *external data root*; the registry places the
# SocNavBench tree under a ``socnavbench`` subdirectory, then the S3DIS dataset below that.
SOCNAV_SUBPATH = "socnavbench"
DATASET_SUBPATH = "sd3dis/stanford_building_parser_dataset"


def _mesh_dir(root: Path, map_name: str = "ETH") -> Path:
    """Return the mesh directory for a map under a synthetic external data root."""
    return root / SOCNAV_SUBPATH / DATASET_SUBPATH / "mesh" / map_name


def _stage_mesh(root: Path, map_name: str = "ETH") -> Path:
    """Create a non-empty mesh directory so preflight treats the mesh as staged."""
    mesh_dir = _mesh_dir(root, map_name)
    mesh_dir.mkdir(parents=True, exist_ok=True)
    (mesh_dir / "mesh.obj").write_text("v 0 0 0\n", encoding="utf-8")
    return mesh_dir


def _output_pkl(root: Path, map_name: str = "ETH") -> Path:
    """Return the traversible output path for a map under a synthetic external data root."""
    return root / SOCNAV_SUBPATH / DATASET_SUBPATH / "traversibles" / map_name / "data.pkl"


def _expected_single_file_tree_hash(path: Path, *, root: Path) -> str:
    """Return the registry-style tree hash for one file under root."""
    file_sha = sha256_file(path)
    digest = hashlib.sha256()
    digest.update(path.relative_to(root).as_posix().encode("utf-8"))
    digest.update(b"\0")
    digest.update(str(path.stat().st_size).encode("ascii"))
    digest.update(b"\0")
    digest.update(file_sha.encode("ascii"))
    digest.update(b"\0")
    return digest.hexdigest()


def test_resolve_paths_uses_expected_layout(tmp_path: Path) -> None:
    """Resolved mesh/output paths match the SocNavBench + manifest layout."""
    paths = resolve_paths("ETH", root=tmp_path)
    assert paths.map_name == "ETH"
    assert paths.mesh_dir == _mesh_dir(tmp_path)
    assert paths.output_pkl == _output_pkl(tmp_path)


@pytest.mark.parametrize("bad", ["", "  ", "../evil", "a/b", ".", ".."])
def test_resolve_paths_rejects_unsafe_map_names(tmp_path: Path, bad: str) -> None:
    """Unsafe or empty map names are rejected before any path is built."""
    with pytest.raises(ValueError):
        resolve_paths(bad, root=tmp_path)


def test_preflight_blocked_when_mesh_absent(tmp_path: Path) -> None:
    """A missing mesh yields a fail-closed blocked status with an actionable message."""
    report = preflight("ETH", root=tmp_path)
    assert report["status"] == STATUS_BLOCKED_MISSING_MESH
    assert report["blocked"] is True
    assert report["mesh_present"] is False
    assert str(_mesh_dir(tmp_path)) in report["next_action"]


def test_preflight_treats_empty_mesh_dir_as_absent(tmp_path: Path) -> None:
    """An empty mesh directory is a placeholder shell, not a staged mesh."""
    _mesh_dir(tmp_path).mkdir(parents=True)
    report = preflight("ETH", root=tmp_path)
    assert report["status"] == STATUS_BLOCKED_MISSING_MESH
    assert report["mesh_present"] is False


def test_preflight_ready_when_mesh_staged(tmp_path: Path) -> None:
    """A staged mesh with no output yields a ready status."""
    _stage_mesh(tmp_path)
    report = preflight("ETH", root=tmp_path)
    assert report["status"] == STATUS_READY
    assert report["blocked"] is False
    assert report["mesh_present"] is True
    assert report["output_exists"] is False


def test_preflight_already_present_reports_hash(tmp_path: Path) -> None:
    """An existing output reports file and registry-style tree hashes."""
    _stage_mesh(tmp_path)
    out = _output_pkl(tmp_path)
    out.parent.mkdir(parents=True)
    out.write_bytes(b"fake-traversible")
    report = preflight("ETH", root=tmp_path)
    assert report["status"] == STATUS_ALREADY_PRESENT
    assert report["output_exists"] is True
    assert report["output_sha256"] == sha256_file(out)
    assert report["output_tree_sha256"] == _expected_single_file_tree_hash(out, root=out.parent)
    assert report["output_tree_file_count"] == 1
    assert report["output_tree_total_size_bytes"] == len(b"fake-traversible")


def test_output_tree_checksum_missing_output_reports_empty_tree(tmp_path: Path) -> None:
    """Missing output yields explicit empty-tree metadata."""
    paths = resolve_paths("ETH", root=tmp_path)
    assert output_tree_checksum(paths) == {
        "output_tree_sha256": None,
        "output_tree_file_count": 0,
        "output_tree_total_size_bytes": 0,
    }


def test_dry_run_absent_mesh_exits_blocked(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    """--dry-run with no mesh exits with the blocked code and does not build."""
    code = main(["--map", "ETH", "--root", str(tmp_path), "--dry-run"])
    assert code == EXIT_BLOCKED
    assert not _output_pkl(tmp_path).exists()
    out = capsys.readouterr().out
    assert STATUS_BLOCKED_MISSING_MESH in out


def test_dry_run_staged_mesh_exits_ok_without_building(
    tmp_path: Path, capsys: pytest.CaptureFixture
) -> None:
    """--dry-run with a staged mesh validates inputs, exits 0, and writes nothing."""
    _stage_mesh(tmp_path)
    code = main(["--map", "ETH", "--root", str(tmp_path), "--dry-run"])
    assert code == 0
    assert not _output_pkl(tmp_path).exists()
    out = capsys.readouterr().out
    assert '"dry_run": true' in out.lower()


def test_dry_run_writes_report_json(tmp_path: Path) -> None:
    """--report-json persists the preflight report to disk."""
    report_path = tmp_path / "report.json"
    code = main(
        ["--map", "ETH", "--root", str(tmp_path), "--dry-run", "--report-json", str(report_path)]
    )
    assert code == EXIT_BLOCKED
    assert report_path.is_file()


def test_main_blocked_without_dry_run(tmp_path: Path) -> None:
    """A real run with no mesh fails closed (exit 2) instead of importing SocNavBench."""
    code = main(["--map", "ETH", "--root", str(tmp_path)])
    assert code == EXIT_BLOCKED
    assert not _output_pkl(tmp_path).exists()


def test_build_traversible_refuses_when_mesh_absent(tmp_path: Path) -> None:
    """build_traversible raises an actionable error before any SocNavBench import."""
    with pytest.raises(TraversibleGenerationError, match="mesh not staged"):
        build_traversible("ETH", root=tmp_path)


def test_build_traversible_idempotent_when_output_present(tmp_path: Path) -> None:
    """An existing output is returned as already-present without rebuilding."""
    _stage_mesh(tmp_path)
    out = _output_pkl(tmp_path)
    out.parent.mkdir(parents=True)
    out.write_bytes(b"fake-traversible")
    report = build_traversible("ETH", root=tmp_path)
    assert report["status"] == STATUS_ALREADY_PRESENT
    assert report["built"] is False
    assert report["output_sha256"] == sha256_file(out)
    assert report["output_tree_sha256"] == _expected_single_file_tree_hash(out, root=out.parent)
