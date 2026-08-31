"""Tests for the lock/profile/environment dependency license inventory."""

# evidence-writer-exempt: Unit tests write isolated temporary fixture inputs and reports outside
# docs/context/evidence; using shared evidence writers would change the fixture contract.

from __future__ import annotations

import hashlib
import io
import json
import tarfile
import zipfile
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
        "package_dispositions": [],
    }
    (root / "scripts" / "validation" / "dependency_license_policy.v1.json").write_text(
        json.dumps(policy, indent=2) + "\n", encoding="utf-8"
    )


def _resolved_distributions(*fields: tuple[str, str, dict[str, str]]) -> list[_Distribution]:
    """Build test distributions from compact tuples."""
    return [_Distribution(name, version, **metadata) for name, version, metadata in fields]


def _write_candidate_bundle(root: Path) -> Path:
    """Write an exact candidate bundle for the fixture's core profile."""
    bundle = root / "candidate-bundle"
    bundle.mkdir()
    version = "0.0.1"
    repository = "ll7/robot_sf_ll7"
    source_sha = "0" * 40
    workflow = {"run_attempt": 1, "run_id": "1"}
    validation = {
        "checks": [
            {
                "command": "python scripts/dev/check_version_alignment.py",
                "id": "version-alignment",
                "status": "passed",
            },
            {
                "command": "twine check --strict $DIST_DIR/*.whl $DIST_DIR/*.tar.gz",
                "id": "metadata",
                "status": "passed",
            },
            {
                "command": (
                    "cd $BUILD_SOURCE && python scripts/tools/check_distribution_licenses.py "
                    "$DIST_DIR --strict-asset-rights --repo-root $BUILD_SOURCE "
                    "--inventory $BUILD_SOURCE/scripts/validation/software_candidate_asset_rights.v1.json "
                    "--source-tree-ref HEAD"
                ),
                "id": "archive-license",
                "status": "passed",
            },
            {
                "command": (
                    "cd $BUILD_SOURCE && bash scripts/validation/wheel_install_smoke.sh "
                    "$DIST_DIR/robot_sf-*.whl"
                ),
                "id": "wheel-install",
                "status": "passed",
            },
        ],
        "status": "passed",
    }
    metadata = (
        "Metadata-Version: 2.4\n"
        "Name: robot_sf\n"
        f"Version: {version}\n"
        "Requires-Dist: demo-package>=1\n"
        "\n"
    ).encode()
    wheel = bundle / f"robot_sf-{version}-py3-none-any.whl"
    with zipfile.ZipFile(wheel, "w") as archive:
        archive.writestr(f"robot_sf-{version}.dist-info/METADATA", metadata)
    sdist = bundle / f"robot_sf-{version}.tar.gz"
    with tarfile.open(sdist, "w:gz") as archive:
        info = tarfile.TarInfo(f"robot_sf-{version}/PKG-INFO")
        info.size = len(metadata)
        archive.addfile(info, io.BytesIO(metadata))
    sbom = bundle / f"robot_sf-{version}.cyclonedx.json"
    sbom.write_text(
        json.dumps(
            {
                "bomFormat": "CycloneDX",
                "specVersion": "1.5",
                "version": 1,
                "metadata": {
                    "component": {
                        "bom-ref": f"pkg:pypi/robot-sf@{version}",
                        "name": "robot-sf",
                        "purl": f"pkg:pypi/robot-sf@{version}",
                        "type": "library",
                        "version": version,
                    }
                },
                "components": [{"name": "demo-package", "version": "1.0.0"}],
                "dependencies": [],
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    provenance = bundle / "candidate-provenance.json"

    def member(path: Path, kind: str) -> dict[str, str | int]:
        return {
            "filename": path.name,
            "kind": kind,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "size": path.stat().st_size,
        }

    wheel_member = member(wheel, "wheel")
    sdist_member = member(sdist, "sdist")
    sbom_member = member(sbom, "sbom")
    materialization = {
        "candidate_commit_sha": "1" * 40,
        "candidate_tree_sha": "2" * 40,
        "policy_path": "scripts/validation/software_candidate_policy.v1.json",
        "policy_sha256": "3" * 64,
        "source_inventory_path": "scripts/validation/asset_rights_inventory.v1.yaml",
        "source_inventory_sha256": "4" * 64,
        "candidate_inventory_path": "scripts/validation/software_candidate_asset_rights.v1.json",
        "candidate_metadata_path": "SOFTWARE_CANDIDATE.json",
    }
    provenance_payload = {
        "build": {
            "command": "cd $BUILD_SOURCE && uv build --out-dir $DIST_DIR",
            "count": 1,
            "source_role": "disposable-exact-commit",
        },
        "package": {"name": "robot_sf", "version": version},
        "repository": repository,
        "sbom": sbom_member,
        "schema_version": "robot_sf.software_candidate.provenance.v1",
        "source_sha": source_sha,
        "subjects": [wheel_member, sdist_member],
        "validation": validation,
        "workflow": workflow,
        "materialization": materialization,
    }
    provenance.write_text(
        json.dumps(provenance_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    provenance_member = member(provenance, "provenance")
    members = [wheel_member, sdist_member, sbom_member, provenance_member]
    (bundle / "candidate-manifest.json").write_text(
        json.dumps(
            {
                "members": members,
                "package": {"name": "robot_sf", "version": version},
                "repository": repository,
                "schema_version": "robot_sf.software_candidate.v1",
                "source_sha": source_sha,
                "validation": validation,
                "workflow": workflow,
                "materialization": materialization,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return bundle


def test_inventory_records_profiles_sources_and_deterministic_output(tmp_path: Path) -> None:
    """Every profile resolves the lock closure and preserves artifact identity."""
    _write_inputs(tmp_path)
    distributions = _resolved_distributions(
        ("robot_sf", "0.0.0.dev0", {"License_Expression": "GPL-3.0-only"}),
        (
            "demo-package",
            "1.0.0",
            {"License_Expression": "BSD-2-Clause AND Apache-2.0 WITH LLVM-exception"},
        ),
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


def test_explicit_profile_selection_keeps_other_lock_context_visible(tmp_path: Path) -> None:
    """A narrow surface changes strict scope without silently dropping context."""
    _write_inputs(tmp_path)

    inventory = build_inventory(tmp_path, distributions=[], selected_profile_ids=["core"])

    assert inventory["surface"] == {
        "profile_ids": ["core"],
        "selection": "explicit_profile_selection",
        "selected_lockfiles": ["uv.lock"],
    }
    assert inventory["summary"]["selected_profile_count"] == 1
    assert inventory["summary"]["selected_package_count"] == 2
    assert all(
        row["surface_membership"] == "outside_selected_profiles"
        for row in inventory["packages"]
        if not row["selected_profiles"]
    )
    assert all(
        row["surface_membership"] == "outside_selected_profiles"
        for row in inventory["unrepresented_lock_package_dispositions"]
        if row["status"] == "reviewed_exclusion"
    )
    assert all(
        row["surface_membership"] == "outside_selected_profiles"
        for row in inventory["unrepresented_lock_package_dispositions"]
        if row["status"] == "unresolved"
    )
    assert not any("fast-pysf/uv.lock" in failure for failure in inventory["failures"])


def test_profile_follows_lock_dependency_extras_into_optional_dependencies(tmp_path: Path) -> None:
    """A locked dependency extra must include its optional dependency closure."""
    _write_inputs(tmp_path)
    pyproject = tmp_path / "pyproject.toml"
    pyproject.write_text(
        pyproject.read_text(encoding="utf-8").replace(
            'dependencies = ["demo-package>=1"]',
            'dependencies = ["demo-package[plug]>=1"]',
        ),
        encoding="utf-8",
    )
    lockfile = tmp_path / "uv.lock"
    lockfile.write_text(
        lockfile.read_text(encoding="utf-8")
        .replace(
            'dependencies = [{ name = "demo-package" }]',
            'dependencies = [{ name = "demo-package", extras = ["plug"] }]',
        )
        .replace(
            'sdist = { url = "https://files.example/demo-package-1.0.0.tar.gz", '
            'hash = "sha256:abc123", size = 12 }',
            'sdist = { url = "https://files.example/demo-package-1.0.0.tar.gz", '
            'hash = "sha256:abc123", size = 12 }\n'
            "[package.optional-dependencies]\n"
            'plug = [{ name = "nested-package" }]\n',
        )
        + "\n[[package]]\n"
        'name = "nested-package"\n'
        'version = "1.0.0"\n'
        'source = { registry = "https://pypi.org/simple" }\n',
        encoding="utf-8",
    )

    inventory = build_inventory(tmp_path, distributions=[], selected_profile_ids=["core"])

    assert inventory["summary"]["selected_package_count"] == 3
    core = next(profile for profile in inventory["profiles"] if profile["id"] == "core")
    assert {row.split("@", maxsplit=1)[0] for row in core["package_ids"]} == {
        "robot-sf",
        "demo-package",
        "nested-package",
    }


def test_unknown_profile_selection_fails_closed(tmp_path: Path) -> None:
    """A typo cannot silently produce an empty release surface."""
    _write_inputs(tmp_path)

    inventory = build_inventory(tmp_path, distributions=[], selected_profile_ids=["missing"])

    assert inventory["surface"]["profile_ids"] == []
    assert any(
        "selected profile does not exist: missing" in failure for failure in inventory["failures"]
    )


def test_candidate_bundle_binds_archives_and_sbom_to_selected_lock_closure(
    tmp_path: Path,
) -> None:
    """Candidate bytes and the SBOM must describe the selected frozen closure exactly."""
    _write_inputs(tmp_path)
    bundle = _write_candidate_bundle(tmp_path)

    inventory = build_inventory(
        tmp_path,
        distributions=_resolved_distributions(
            ("robot_sf", "0.0.1", {"License_Expression": "GPL-3.0-only"}),
            ("demo-package", "1.0.0", {"License_Expression": "MIT"}),
        ),
        selected_profile_ids=["core"],
        candidate_bundle_path=bundle,
    )

    assert inventory["summary"]["candidate_bound"] is True
    assert inventory["candidate_binding"]["status"] == "bound"
    assert inventory["candidate_binding"]["profile_ids"] == ["core"]
    assert inventory["candidate_binding"]["materialization"]["candidate_commit_sha"] == "1" * 40
    assert inventory["candidate_binding"]["sbom"]["component_count"] == 1
    root = next(row for row in inventory["packages"] if row["name"] == "robot-sf")
    assert root["observed_version"] == "0.0.1"
    demo = next(row for row in inventory["packages"] if row["name"] == "demo-package")
    assert demo["observation_status"] == "artifact_bound"
    assert demo["metadata_binding"] == "candidate_sbom_component_identity"
    assert not any(
        "candidate bundle binding failed" in failure for failure in inventory["failures"]
    )


def test_candidate_bundle_drift_fails_closed(tmp_path: Path) -> None:
    """Changing an admitted candidate member cannot produce a bound report."""
    _write_inputs(tmp_path)
    bundle = _write_candidate_bundle(tmp_path)
    sbom = next(bundle.glob("*.cyclonedx.json"))
    sbom.write_text(
        sbom.read_text(encoding="utf-8").replace("demo-package", "other-package"), encoding="utf-8"
    )

    inventory = build_inventory(
        tmp_path,
        distributions=[],
        selected_profile_ids=["core"],
        candidate_bundle_path=bundle,
    )

    assert inventory["summary"]["candidate_bound"] is False
    assert inventory["candidate_binding"]["status"] == "blocked"
    assert any("candidate bundle binding failed" in failure for failure in inventory["failures"])


def test_candidate_materialization_path_drift_fails_closed(tmp_path: Path) -> None:
    """A traversal path in the producer's materialization envelope cannot be bound."""
    _write_inputs(tmp_path)
    bundle = _write_candidate_bundle(tmp_path)
    manifest_path = bundle / "candidate-manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["materialization"]["candidate_metadata_path"] = "../SOFTWARE_CANDIDATE.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    inventory = build_inventory(
        tmp_path,
        distributions=[],
        selected_profile_ids=["core"],
        candidate_bundle_path=bundle,
    )

    assert inventory["summary"]["candidate_bound"] is False
    assert any(
        "candidate materialization candidate_metadata_path is invalid" in failure
        for failure in inventory["failures"]
    )


def test_candidate_bound_report_freshness_rechecks_candidate_bytes(tmp_path: Path) -> None:
    """Freshness must cover candidate bytes as well as repository inputs."""
    _write_inputs(tmp_path)
    bundle = _write_candidate_bundle(tmp_path)
    inventory = build_inventory(
        tmp_path,
        distributions=[],
        selected_profile_ids=["core"],
        candidate_bundle_path=bundle,
    )
    report_path = tmp_path / "candidate-report.json"
    report_path.write_text(json.dumps(inventory, indent=2) + "\n", encoding="utf-8")

    assert any(
        "candidate-bound report freshness requires --candidate-bundle" in issue
        for issue in check_report_freshness(tmp_path, report_path)
    )
    assert (
        check_report_freshness(
            tmp_path,
            report_path,
            candidate_bundle_path=bundle,
        )
        == []
    )

    sbom = next(bundle.glob("*.cyclonedx.json"))
    sbom.write_text(
        sbom.read_text(encoding="utf-8").replace("demo-package", "other-package"), encoding="utf-8"
    )
    assert any(
        "candidate bundle binding differs" in issue
        for issue in check_report_freshness(
            tmp_path,
            report_path,
            candidate_bundle_path=bundle,
        )
    )


def test_candidate_provenance_drift_fails_closed(tmp_path: Path) -> None:
    """A rehashed but inconsistent provenance member cannot become bound evidence."""
    _write_inputs(tmp_path)
    bundle = _write_candidate_bundle(tmp_path)
    provenance = bundle / "candidate-provenance.json"
    provenance.write_text(
        provenance.read_text(encoding="utf-8").replace('"count": 1', '"count": 2'), encoding="utf-8"
    )
    manifest_path = bundle / "candidate-manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["members"][3]["sha256"] = hashlib.sha256(provenance.read_bytes()).hexdigest()
    manifest["members"][3]["size"] = provenance.stat().st_size
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    inventory = build_inventory(
        tmp_path,
        distributions=[],
        selected_profile_ids=["core"],
        candidate_bundle_path=bundle,
    )

    assert inventory["summary"]["candidate_bound"] is False
    assert any(
        "candidate provenance does not exactly bind" in failure for failure in inventory["failures"]
    )


def test_cli_resolves_candidate_bundle_relative_to_repo_root(tmp_path: Path) -> None:
    """CLI candidate paths follow the repository-root path contract."""
    _write_inputs(tmp_path)
    _write_candidate_bundle(tmp_path)
    report_path = tmp_path / "candidate-report.json"

    assert (
        main(
            [
                "--repo-root",
                str(tmp_path),
                "--profile",
                "core",
                "--candidate-bundle",
                "candidate-bundle",
                "--fail-on-unresolved",
                "--output",
                str(report_path),
            ]
        )
        == 2
    )
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["summary"]["candidate_bound"] is True
    assert (
        main(
            [
                "--repo-root",
                str(tmp_path),
                "--check-freshness",
                str(report_path),
                "--candidate-bundle",
                "candidate-bundle",
            ]
        )
        == 0
    )


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
    assert dispositions["orphan-tool"]["surface_membership"] == "unresolved_membership"
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


def test_current_llvmlite_disposition_matches_both_frozen_lock_profiles() -> None:
    """The reviewed llvmlite row covers root and standalone profile closures."""
    root = Path(__file__).resolve().parents[2]
    inventory = build_inventory(
        root,
        distributions=[
            _Distribution(
                "llvmlite",
                "0.49.0",
                License_Expression="BSD-2-Clause AND Apache-2.0 WITH LLVM-exception",
            )
        ],
    )

    rows = [row for row in inventory["packages"] if row["normalized_name"] == "llvmlite"]
    assert {row["lockfile"] for row in rows} == {"uv.lock", "fast-pysf/uv.lock"}
    assert all(row["exact_policy_status"] == "accepted" for row in rows)
    assert all(row["policy_disposition"] == "external_dependency_not_redistributed" for row in rows)
    assert not any("llvmlite" in failure for failure in inventory["failures"])
    assert inventory["summary"]["policy_exact_match_count"] == 2


def test_exact_llvmlite_policy_rejects_another_with_expression(tmp_path: Path) -> None:
    """An arbitrary SPDX WITH expression cannot inherit the reviewed exception."""
    root = Path(__file__).resolve().parents[2]
    policy = json.loads(
        (root / "scripts/validation/dependency_license_policy.v1.json").read_text(encoding="utf-8")
    )
    policy["package_dispositions"][0]["license_expression"] = "BSD-2-Clause WITH Apache-2.0"
    policy_path = tmp_path / "policy.json"
    policy_path.write_text(json.dumps(policy, indent=2) + "\n", encoding="utf-8")

    inventory = build_inventory(
        root,
        policy_path=policy_path,
        distributions=[
            _Distribution(
                "llvmlite",
                "0.49.0",
                License_Expression="BSD-2-Clause AND Apache-2.0 WITH LLVM-exception",
            )
        ],
    )

    assert any(
        "llvmlite: exact policy llvmlite-0-49-0-external-install:"
        " observed license expression does not match exact policy" in failure
        for failure in inventory["failures"]
    )


def test_exact_policy_evidence_is_bound_into_report_freshness(tmp_path: Path) -> None:
    """Changing the exact disposition evidence invalidates an otherwise fresh report."""
    root = Path(__file__).resolve().parents[2]
    inventory = build_inventory(root, distributions=[])
    evidence_path = root / "docs/context/evidence/llvmlite_0.49.0_surface_disposition_2026-08-20.md"
    input_paths = {row["path"] for row in inventory["repository_inputs"]}
    assert evidence_path.relative_to(root).as_posix() in input_paths

    report_path = tmp_path / "report.json"
    report_path.write_text(json.dumps(inventory, indent=2) + "\n", encoding="utf-8")
    assert check_report_freshness(root, report_path) == []
