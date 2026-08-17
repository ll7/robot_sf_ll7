"""Tests for the lock-to-environment dependency license inventory."""

from __future__ import annotations

from email.message import Message
from typing import TYPE_CHECKING

from scripts.tools.check_dependency_license_inventory import build_inventory

if TYPE_CHECKING:
    from pathlib import Path


class _Distribution:
    """Minimal importlib.metadata distribution double."""

    def __init__(self, name: str, version: str, **fields: str) -> None:
        self.name = name
        self.version = version
        self.metadata = Message()
        self.metadata["Name"] = name
        for key, value in fields.items():
            self.metadata[key.replace("_", "-")] = value

    def __repr__(self) -> str:
        return f"_Distribution({self.name!r}, {self.version!r})"


def _write_inputs(root: Path) -> None:
    (root / "pyproject.toml").write_text(
        '[project]\nname = "robot_sf"\nlicense = "GPL-3.0-only"\n',
        encoding="utf-8",
    )
    (root / "uv.lock").write_text(
        """
version = 1
revision = 3

[[package]]
name = "robot-sf"
source = { editable = "." }

[[package]]
name = "demo-package"
version = "1.0.0"

[[package]]
name = "missing-package"
version = "2.0.0"
""".lstrip(),
        encoding="utf-8",
    )


def test_inventory_keeps_missing_and_non_spdx_metadata_blocked(tmp_path: Path) -> None:
    """The report must preserve unresolved rights instead of guessing an SPDX license."""
    _write_inputs(tmp_path)
    distributions = [
        _Distribution("robot_sf", "0.0.0.dev0", License_Expression="GPL-3.0-only"),
        _Distribution("demo-package", "1.0.0", License="MIT License"),
    ]

    inventory = build_inventory(tmp_path, distributions=distributions)

    statuses = {item["name"]: item["license_status"] for item in inventory["packages"]}
    assert statuses == {
        "demo-package": "review_required",
        "missing-package": "not_installed",
        "robot-sf": "resolved",
    }
    assert inventory["summary"]["status"] == "blocked"
    assert inventory["summary"]["unresolved_count"] == 2
    assert any("missing-package" in failure for failure in inventory["failures"])


def test_inventory_flags_custom_or_proprietary_license_refs(tmp_path: Path) -> None:
    """Custom and proprietary identifiers must require human redistribution review."""
    _write_inputs(tmp_path)
    distributions = [
        _Distribution("robot_sf", "0.0.0.dev0", License_Expression="GPL-3.0-only"),
        _Distribution(
            "demo-package",
            "1.0.0",
            License_Expression="LicenseRef-NVIDIA-SOFTWARE-LICENSE",
        ),
    ]

    inventory = build_inventory(tmp_path, distributions=distributions)

    package = next(item for item in inventory["packages"] if item["name"] == "demo-package")
    assert package["license_status"] == "review_required"
    assert "custom, proprietary, or restricted-license marker" in package["review_reasons"][0]
