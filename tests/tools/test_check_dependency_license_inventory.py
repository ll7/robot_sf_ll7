"""Tests for the lock/profile/environment dependency license inventory."""

from __future__ import annotations

import json
from email.message import Message
from pathlib import Path
from typing import TYPE_CHECKING

from scripts.tools.check_dependency_license_inventory import (
    build_inventory,
    check_report_freshness,
    main,
)

if TYPE_CHECKING:
    from collections.abc import Iterable


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


def _write_inputs(root: Path, *, all_excluded: Iterable[str] = ()) -> None:
    """Write a compact root lock, profile matrix, and disposition policy."""
    (root / "scripts" / "validation").mkdir(parents=True)
    (root / "pyproject.toml").write_text(
        """
[project]
name = "robot_sf"
license = "GPL-3.0-only"
dependencies = ["demo-package>=1"]

[project.optional-dependencies]
foo = ["demo-package>=1"]
all = ["robot_sf[foo]"]
""".lstrip(),
        encoding="utf-8",
    )
    (root / "uv.lock").write_text(
        """
version = 1
revision = 3

[[package]]
name = "robot-sf"
source = { editable = "." }
dependencies = [{ name = "demo-package" }]
[package.dev-dependencies]
dev = [
    { name = "dev-tool" },
    { name = "inactive-tool", marker = "python_full_version < '3.12'" },
]

[[package]]
name = "demo-package"
version = "1.0.0"
source = { registry = "https://pypi.org/simple" }
sdist = { url = "https://files.example/demo-package-1.0.0.tar.gz", hash = "sha256:abc123", size = 12 }

[[package]]
name = "dev-tool"
version = "1.0.0"
source = { registry = "https://pypi.org/simple" }

[[package]]
name = "inactive-tool"
version = "1.0.0"
source = { registry = "https://pypi.org/simple" }
resolution-markers = ["python_full_version < '3.12'"]

[[package]]
name = "orphan-tool"
version = "1.0.0"
source = { registry = "https://pypi.org/simple" }
""".lstrip(),
        encoding="utf-8",
    )
    profiles = {
        "schema_version": "robot-sf.dependency-license-profiles.v1",
        "target": {
            "os": "linux",
            "architecture": "x86_64",
            "python": {"implementation": "CPython", "version": "3.13", "requires": ">=3.11"},
            "resolver": {"name": "uv", "version": "0.11.21", "lock_mode": "frozen"},
            "source_policy": {"indexes": ["https://pypi.org/simple"], "network": "offline"},
        },
        "unrepresented_policy": {
            "schema_version": "robot-sf.dependency-license-unrepresented.v1",
            "rules": [
                {
                    "id": "fixture-development-group",
                    "lockfile": "uv.lock",
                    "root_package": "robot-sf",
                    "field": "dev-dependencies",
                    "groups": ["dev"],
                    "reason_code": "development_group",
                    "reviewed": True,
                    "rationale": "fixture development inputs are outside the release profile",
                },
                {
                    "id": "fixture-inactive-marker",
                    "reason_code": "marker_inactive",
                    "reviewed": True,
                    "rationale": "fixture row is inactive for the target",
                },
            ],
            "unresolved": {
                "id": "fixture-unresolved",
                "reason_code": "unresolved_membership",
                "reviewed": False,
                "rationale": "fixture orphan row has no reviewed membership reason",
            },
        },
        "profiles": [
            {
                "id": "core",
                "kind": "root",
                "extras": [],
                "pyproject": "pyproject.toml",
                "lockfile": "uv.lock",
                "root_package": "robot-sf",
                "expected_resolution": "locked",
                "distribution_mode": "user_installed",
            },
            {
                "id": "foo",
                "kind": "root-extra",
                "extra": "foo",
                "extras": ["foo"],
                "pyproject": "pyproject.toml",
                "lockfile": "uv.lock",
                "root_package": "robot-sf",
                "expected_resolution": "locked",
                "distribution_mode": "user_installed",
            },
            {
                "id": "all",
                "kind": "root-extra-closure",
                "extra": "all",
                "extras": ["foo"],
                "excluded_extras": list(all_excluded),
                "pyproject": "pyproject.toml",
                "lockfile": "uv.lock",
                "root_package": "robot-sf",
                "expected_resolution": "locked",
                "distribution_mode": "user_installed",
            },
        ],
    }
    (root / "scripts" / "validation" / "dependency_license_profiles.v1.json").write_text(
        json.dumps(profiles, indent=2) + "\n", encoding="utf-8"
    )
    policy = {
        "schema_version": "robot-sf.dependency-license-policy.v1",
        "claim_boundary": "metadata is evidence, not legal permission",
        "rules": [
            {
                "id": "user-installed",
                "distribution_mode": "user_installed",
                "disposition": "review_required",
                "rationale": "requires explicit release-surface review",
            },
            {
                "id": "bundled-source",
                "distribution_mode": "bundled_source",
                "disposition": "review_required",
                "rationale": "requires explicit release-surface review",
            },
            {
                "id": "built-companion",
                "distribution_mode": "built_companion",
                "disposition": "review_required",
                "rationale": "requires explicit release-surface review",
            },
            {
                "id": "not-distributed",
                "distribution_mode": "not_distributed",
                "disposition": "excluded",
                "rationale": "not on the release surface",
            },
        ],
        "components": [],
    }
    (root / "scripts" / "validation" / "dependency_license_policy.v1.json").write_text(
        json.dumps(policy, indent=2) + "\n", encoding="utf-8"
    )


def _resolved_distributions(*fields: tuple[str, str, dict[str, str]]) -> list[_Distribution]:
    """Build test distributions from compact tuples."""
    return [_Distribution(name, version, **metadata) for name, version, metadata in fields]


def test_inventory_records_profiles_sources_and_deterministic_output(tmp_path: Path) -> None:
    """Every profile resolves the lock closure and preserves artifact identity."""
    _write_inputs(tmp_path)
    distributions = _resolved_distributions(
        ("robot_sf", "0.0.0.dev0", {"License_Expression": "GPL-3.0-only"}),
        ("demo-package", "1.0.0", {"License_Expression": "MIT OR Apache-2.0"}),
    )

    first = build_inventory(tmp_path, distributions=distributions)
    second = build_inventory(tmp_path, distributions=distributions)

    assert first == second
    assert first["summary"]["status"] == "blocked"
    assert first["structural_issues"] == []
    assert {profile["id"] for profile in first["profiles"]} == {"core", "foo", "all"}
    demo = next(item for item in first["packages"] if item["name"] == "demo-package")
    assert demo["license_status"] == "spdx_expression"
    assert demo["source"]["registry"] == "https://pypi.org/simple"
    assert demo["artifacts"][0]["sha256"] == "abc123"
    assert set(demo["profiles"]) == {"core", "foo", "all"}


def test_unrepresented_rows_require_reviewed_reason_or_remain_strictly_unresolved(
    tmp_path: Path,
) -> None:
    """Development, inactive-marker, and unexplained rows stay distinguishable."""
    _write_inputs(tmp_path)

    inventory = build_inventory(tmp_path, distributions=[])

    dispositions = {
        row["name"]: row for row in inventory["unrepresented_lock_package_dispositions"]
    }
    assert dispositions["dev-tool"]["status"] == "reviewed_exclusion"
    assert dispositions["dev-tool"]["reason_codes"] == ["development_group"]
    assert dispositions["inactive-tool"]["status"] == "reviewed_exclusion"
    assert dispositions["inactive-tool"]["reason_codes"] == ["marker_inactive"]
    assert dispositions["orphan-tool"]["status"] == "unresolved"
    assert dispositions["orphan-tool"]["reviewed"] is False
    assert inventory["summary"]["unrepresented_reviewed_exclusion_count"] == 2
    assert inventory["summary"]["unrepresented_unresolved_count"] == 1
    assert any("orphan-tool" in failure for failure in inventory["failures"])


def test_inventory_keeps_unknown_proprietary_and_conflicting_metadata_blocked(
    tmp_path: Path,
) -> None:
    """Raw metadata conflicts and restricted identifiers never become approval."""
    _write_inputs(tmp_path)
    distributions = _resolved_distributions(
        ("robot_sf", "0.0.0.dev0", {"License_Expression": "GPL-3.0-only"}),
        (
            "demo-package",
            "1.0.0",
            {
                "License_Expression": "LicenseRef-NVIDIA-SOFTWARE-LICENSE",
                "License": "MIT",
            },
        ),
    )

    inventory = build_inventory(tmp_path, distributions=distributions)
    demo = next(item for item in inventory["packages"] if item["name"] == "demo-package")
    assert demo["license_status"] == "metadata_conflict"
    assert demo["raw_license_metadata"]["License-Expression"].startswith("LicenseRef-")
    assert inventory["summary"]["unresolved_count"] > 0


def test_freshness_fails_when_a_locked_input_changes(tmp_path: Path) -> None:
    """The report binds its input digest set and detects lock drift."""
    _write_inputs(tmp_path)
    distributions = _resolved_distributions(
        ("robot_sf", "0.0.0.dev0", {"License_Expression": "GPL-3.0-only"}),
        ("demo-package", "1.0.0", {"License_Expression": "MIT"}),
    )
    report = build_inventory(tmp_path, distributions=distributions)
    report_path = tmp_path / "report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    assert check_report_freshness(tmp_path, report_path) == []
    (tmp_path / "uv.lock").write_text(
        (tmp_path / "uv.lock").read_text(encoding="utf-8") + "\n# drift\n",
        encoding="utf-8",
    )
    issues = check_report_freshness(tmp_path, report_path)
    assert any("uv.lock" in issue for issue in issues)


def test_current_profile_matrix_is_explicit_and_all_exclusion_is_visible() -> None:
    """The checked-in matrix covers every declared extra without hiding rllib."""
    root = Path(__file__).resolve().parents[2]
    inventory = build_inventory(root, distributions=[])
    profile_ids = {profile["id"] for profile in inventory["profiles"]}
    assert "rllib" in profile_ids
    all_profile = next(profile for profile in inventory["profiles"] if profile["id"] == "all")
    assert "rllib" not in all_profile["extras"]
    assert all_profile["excluded_extras"] == ["rllib"]
    assert inventory["structural_issues"] == []
    assert all(
        profile["status"] in {"complete", "not_applicable"} for profile in inventory["profiles"]
    )


def test_current_unrepresented_rows_have_explicit_dispositions() -> None:
    """The live lock closure cannot hide unrepresented rows behind a count only."""
    root = Path(__file__).resolve().parents[2]
    inventory = build_inventory(root, distributions=[])

    rows = inventory["unrepresented_lock_package_dispositions"]
    assert {row["package_id"] for row in rows} == set(inventory["unrepresented_lock_packages"])
    assert len(rows) == inventory["summary"]["unrepresented_lock_package_count"]
    assert all(row["reason_codes"] for row in rows)
    assert all(row["status"] in {"reviewed_exclusion", "unresolved"} for row in rows)
    assert inventory["summary"]["unrepresented_reviewed_exclusion_count"] + inventory["summary"][
        "unrepresented_unresolved_count"
    ] == len(rows)
    if inventory["summary"]["unrepresented_unresolved_count"]:
        assert any("unrepresented lock row" in failure for failure in inventory["failures"])


def test_freshness_rejects_a_report_built_from_a_substitute_policy(tmp_path: Path) -> None:
    """A relaxed policy cannot be laundered into a report that still validates as fresh."""
    _write_inputs(tmp_path)
    canonical_policy = tmp_path / "scripts" / "validation" / "dependency_license_policy.v1.json"
    relaxed_policy = tmp_path / "scripts" / "validation" / "relaxed_policy.json"
    relaxed_policy.write_text(canonical_policy.read_text(encoding="utf-8"), encoding="utf-8")
    report = build_inventory(tmp_path, distributions=[], policy_path=relaxed_policy)
    report_path = tmp_path / "report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    issues = check_report_freshness(tmp_path, report_path)

    assert any("not generated from the canonical input" in issue for issue in issues)
    assert any("dependency_license_policy.v1.json" in issue for issue in issues)


def test_profile_referencing_an_undeclared_extra_fails_closed(tmp_path: Path) -> None:
    """A profile extra that pyproject no longer declares must not resolve to base deps."""
    _write_inputs(tmp_path)
    manifest_path = tmp_path / "scripts" / "validation" / "dependency_license_profiles.v1.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for profile in manifest["profiles"]:
        if profile["id"] == "foo":
            profile["extra"] = "foo_typo"
            profile["extras"] = ["foo_typo"]
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    inventory = build_inventory(tmp_path, distributions=[])

    profile = next(item for item in inventory["profiles"] if item["id"] == "foo")
    assert profile["status"] == "blocked"
    assert any("references undeclared extra" in issue for issue in profile["missing_dependencies"])
    assert any("references undeclared extra" in failure for failure in inventory["failures"])


def test_freshness_with_strict_flag_reapplies_the_unresolved_exit_code(tmp_path: Path) -> None:
    """`--check-freshness --fail-on-unresolved` must not report success on blocked evidence."""
    _write_inputs(tmp_path)
    report = build_inventory(tmp_path, distributions=[])
    report_path = tmp_path / "report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    assert report["summary"]["unresolved_count"] > 0
    assert main(["--repo-root", str(tmp_path), "--check-freshness", str(report_path)]) == 0
    assert (
        main(
            [
                "--repo-root",
                str(tmp_path),
                "--check-freshness",
                str(report_path),
                "--fail-on-unresolved",
            ]
        )
        == 2
    )
