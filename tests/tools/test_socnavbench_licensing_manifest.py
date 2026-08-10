"""Tests for the machine-readable SocNavBench license manifest."""

from __future__ import annotations

from pathlib import Path

import yaml

_EXCLUDED_TOP_LEVEL = {"LICENSE", "LICENSING.yaml", "UPSTREAM.md"}


def _vendored_source_files(root_files: Path) -> set[str]:
    """Inventory licenseable source files under the vendored tree.

    Bytecode caches are transient interpreter artifacts, not upstream or local source:
    CI shards that import the vendored package create ``__pycache__`` directories before
    this gate runs, so they must not pollute the inventory.
    """
    actual_files: set[str] = set()
    for path in root_files.rglob("*"):
        if not path.is_file():
            continue
        rel = path.relative_to(root_files)
        if "__pycache__" in rel.parts:
            continue
        if rel.parts[0] in _EXCLUDED_TOP_LEVEL:
            continue
        if rel.as_posix().startswith("LICENSES/"):
            continue
        actual_files.add(rel.as_posix())
    return actual_files


def test_manifest_scan_ignores_bytecode_artifacts(tmp_path: Path) -> None:
    """Bytecode caches from importing the vendored package must not pollute the scan."""
    (tmp_path / "mp_env").mkdir()
    (tmp_path / "mp_env" / "map_utils.py").write_text("", encoding="utf-8")
    pycache = tmp_path / "mp_env" / "__pycache__"
    pycache.mkdir()
    (pycache / "map_utils.cpython-312.pyc").write_bytes(b"")

    assert _vendored_source_files(tmp_path) == {"mp_env/map_utils.py"}


def test_socnavbench_manifest_makes_mixed_license_scope_explicit() -> None:
    """The vendored subset must distinguish MIT defaults from Apache overrides."""
    root = Path(__file__).resolve().parents[2]
    manifest = yaml.safe_load(
        (root / "third_party" / "socnavbench" / "LICENSING.yaml").read_text(encoding="utf-8")
    )

    assert manifest["schema_version"] == "robot_sf.third_party_licensing.v1"
    assert manifest["default_license_spdx"] == "MIT"
    root_files = root / "third_party" / "socnavbench"
    actual_files = _vendored_source_files(root_files)
    inventory = set(manifest["upstream_files"]) | {item["path"] for item in manifest["local_files"]}
    assert actual_files <= inventory
    overrides = manifest["license_overrides"]
    assert len(overrides) == 1
    assert overrides[0]["license_spdx"] == "Apache-2.0"
    assert set(overrides[0]["files"]) == {
        "mp_env/map_utils.py",
        "mp_env/render/rotation_utils.py",
        "mp_env/render/swiftshader_renderer.py",
        "sbpd/sbpd.py",
    }
    assert manifest["local_files"] == [
        {
            "path": "__init__.py",
            "license_spdx": "GPL-3.0-only",
            "copyright": "Robot SF contributors",
            "note": "Package marker added for the vendored planner subset.",
        },
        {
            "path": "dotmap.py",
            "license_spdx": "GPL-3.0-only",
            "copyright": "Robot SF contributors",
            "note": "Compatibility shim written for the vendored planner subset.",
        },
    ]
