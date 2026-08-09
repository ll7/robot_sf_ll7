"""Tests for the machine-readable SocNavBench license manifest."""

from __future__ import annotations

from pathlib import Path

import yaml


def test_socnavbench_manifest_makes_mixed_license_scope_explicit() -> None:
    """The vendored subset must distinguish MIT defaults from Apache overrides."""
    root = Path(__file__).resolve().parents[2]
    manifest = yaml.safe_load(
        (root / "third_party" / "socnavbench" / "LICENSING.yaml").read_text(encoding="utf-8")
    )

    assert manifest["schema_version"] == "robot_sf.third_party_licensing.v1"
    assert manifest["default_license_spdx"] == "MIT"
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
            "path": "dotmap.py",
            "license_spdx": "GPL-3.0-only",
            "copyright": "Robot SF contributors",
            "note": "Compatibility shim written for the vendored planner subset.",
        }
    ]
