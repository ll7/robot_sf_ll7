"""Tests for the lock/profile/environment dependency license inventory."""

# evidence-writer-exempt: Unit tests write isolated temporary fixture inputs and reports outside
# docs/context/evidence; using shared evidence writers would change the fixture contract.

from __future__ import annotations

import copy
import hashlib
import io
import json
import os
import subprocess
import sys
import tarfile
import zipfile
from email.message import Message
from pathlib import Path
from typing import TYPE_CHECKING

from scripts.tools.check_dependency_license_inventory import (
    SUPPORTED_SOFTWARE_CANDIDATE_DISTRIBUTION_EXTRA_IDS,
    _archive_audit_semantic_issues,
    _archive_notice_paths,
    _candidate_receipt_semantic_issues,
    _effective_profile_coverage,
    _exact_policy_coverage_failures,
    _github_notice_reference,
    _match_package_disposition,
    _policy_archive_notice_kinds,
    _policy_archive_notice_mapping,
    _policy_records,
    _policy_source_matches,
    _report_content_digest,
    _upstream_tags_semantic_issues,
    build_inventory,
    check_report_freshness,
    main,
    validate_dependency_license_receipt,
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


def _write_candidate_bundle(root: Path, *, extras: Iterable[str] = ()) -> Path:
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
        + "".join(f"Provides-Extra: {extra}\n" for extra in extras)
        + "\n"
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


def test_candidate_bound_all_rejects_rllib_distribution_metadata(tmp_path: Path) -> None:
    """The supported all-surface binding rejects the development-only RLlib extra."""
    _write_inputs(tmp_path)
    bundle = _write_candidate_bundle(
        tmp_path,
        extras=[*SUPPORTED_SOFTWARE_CANDIDATE_DISTRIBUTION_EXTRA_IDS, "foo", "rllib"],
    )

    inventory = build_inventory(
        tmp_path,
        distributions=_resolved_distributions(
            ("robot_sf", "0.0.1", {"License_Expression": "GPL-3.0-only"}),
            ("demo-package", "1.0.0", {"License_Expression": "MIT"}),
        ),
        selected_profile_ids=["all"],
        candidate_bundle_path=bundle,
    )

    assert inventory["summary"]["candidate_bound"] is False
    assert any(
        "closed v0.0.6 supported extra roster" in failure for failure in inventory["failures"]
    )


def test_sanitized_project_can_retain_explicit_rllib_exclusion(tmp_path: Path) -> None:
    """A candidate project may omit rllib when its profile exclusion remains explicit."""
    _write_inputs(tmp_path, all_excluded=["rllib"])

    inventory = build_inventory(tmp_path, distributions=[], selected_profile_ids=["all"])

    assert inventory["structural_issues"] == []
    assert inventory["surface"]["profile_ids"] == ["all"]


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


def test_direct_checkout_invocation_writes_marked_report_without_package_import(
    tmp_path: Path,
) -> None:
    """The workflow's direct ``python scripts/tools/...`` invocation stays portable."""
    _write_inputs(tmp_path)
    report_path = tmp_path / "output" / "dependency-license-inventory.json"
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    repo_root = Path(__file__).resolve().parents[2]
    script = repo_root / "scripts" / "tools" / "check_dependency_license_inventory.py"

    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--repo-root",
            str(tmp_path),
            "--profile",
            "core",
            "--output",
            str(report_path),
        ],
        check=False,
        cwd=checkout,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["review_marker"] == "AI-GENERATED NEEDS-REVIEW"
    assert report["surface"]["profile_ids"] == ["core"]


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


def test_freshness_rejects_summary_failure_and_package_mutations(tmp_path: Path) -> None:
    """Retained input digests cannot hide edits to report content."""
    _write_inputs(tmp_path)
    report = build_inventory(tmp_path, distributions=[])
    report_path = tmp_path / "report.json"
    for field, mutate in (
        ("summary", lambda value: value.update({"status": "complete"})),
        ("failures", lambda value: value.append("injected failure")),
        ("packages", lambda value: value[0].update({"name": "shadowed-package"})),
    ):
        mutated = copy.deepcopy(report)
        mutate(mutated[field])
        report_path.write_text(json.dumps(mutated, indent=2) + "\n", encoding="utf-8")
        assert any(
            "content digest differs" in issue
            for issue in check_report_freshness(tmp_path, report_path)
        )


def test_duplicate_package_identity_cannot_be_masked_by_another_row(tmp_path: Path) -> None:
    """A duplicate name/version/source disposition fails closed regardless of row order."""
    root = Path(__file__).resolve().parents[2]
    policy = json.loads(
        (root / "scripts/validation/dependency_license_policy.v1.json").read_text(encoding="utf-8")
    )
    duplicate = copy.deepcopy(policy["package_dispositions"][0])
    duplicate["id"] = "shadow-duplicate-package-identity"
    duplicate["status"] = "pending_review"
    duplicate["reviewer"] = None
    duplicate["reviewed_at"] = None
    policy["package_dispositions"].append(duplicate)
    policy_path = tmp_path / "policy.json"
    policy_path.write_text(json.dumps(policy, indent=2) + "\n", encoding="utf-8")

    inventory = build_inventory(root, distributions=[], policy_path=policy_path)

    assert any(
        "duplicate package disposition identity" in issue
        for issue in inventory["structural_issues"]
    )
    _rules, _components, by_name, _records, _issues = _policy_records(policy, root)
    clean_inventory = build_inventory(root, distributions=[])
    package = next(
        row for row in clean_inventory["packages"] if row["normalized_name"] == "llvmlite"
    )
    matched, match_issues = _match_package_disposition(
        package,
        {"license_expression": package.get("license_expression")},
        "user_installed",
        {"core"},
        clean_inventory["target"],
        by_name,
    )
    assert matched is None
    assert any("exact policy identity is ambiguous" in issue for issue in match_issues)


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


def test_issue_8163_policy_records_retain_metadata_and_archive_evidence() -> None:
    """Exact batch provenance survives checker normalization without source drift."""
    root = Path(__file__).resolve().parents[2]
    policy = json.loads(
        (root / "scripts/validation/dependency_license_policy.v1.json").read_text(encoding="utf-8")
    )

    _rules, _components, by_name, _records, issues = _policy_records(policy, root)

    assert issues == []
    for name in (
        "absl-py",
        "alembic",
        "attrs",
        "click",
        "cma",
        "cyclopts",
        "fsspec",
        "geopandas",
        "idna",
        "imageio",
        "joblib",
        "jsonschema",
        "jsonschema-specifications",
        "lazy-loader",
        "markdown",
        "narwhals",
        "networkx",
        "opentelemetry-api",
        "opt-einsum",
        "osmnx",
        "platformdirs",
        "pooch",
        "proglog",
        "pydantic",
        "pyparsing",
        "python-dotenv",
        "pyvista",
        "referencing",
        "rich-rst",
        "scooby",
        "setuptools",
        "termcolor",
        "typing-inspection",
        "urllib3",
        "werkzeug",
        "wheel",
    ):
        record = by_name[name][0]
        assert record["source"]["metadata_url"] == (
            f"https://pypi.org/pypi/{name}/{record['version']}/json"
        )
        assert record["status"] == "pending_review"
        assert record["reviewer"] is None
        assert record["reviewed_at"] is None
        assert record["upstream"]["archive_notice_paths"]
        assert record["upstream"]["archive_notice_absences"] == []


def test_policy_metadata_url_does_not_change_exact_lock_source_matching() -> None:
    """The retained PyPI response URL is provenance, not a second lock source."""
    assert _policy_source_matches(
        {"registry": "https://pypi.org/simple"},
        {
            "registry": "https://pypi.org/simple",
            "metadata_url": "https://pypi.org/pypi/demo-package/1.0.0/json",
        },
    )
    assert not _policy_source_matches(
        {"registry": "https://example.invalid/simple"},
        {
            "registry": "https://pypi.org/simple",
            "metadata_url": "https://pypi.org/pypi/demo-package/1.0.0/json",
        },
    )


def test_all_profile_policy_ignores_transitive_membership_edges() -> None:
    """An aggregate ``all`` row does not fail on its closure's profile edges."""
    policy = {
        "id": "demo-1-external-install",
        "package": "demo",
        "version": "1.0.0",
        "source": {"registry": "https://pypi.org/simple"},
        "profiles": ["all"],
    }
    record = {
        "normalized_name": "demo",
        "version": "1.0.0",
        "source": {"registry": "https://pypi.org/simple"},
        "profiles": ["all", "core", "viz"],
        "selected_profiles": ["all", "core", "viz"],
    }

    assert _effective_profile_coverage({"all", "core", "viz"}, {"all"}) == {"all"}
    assert (
        _exact_policy_coverage_failures(
            [policy],
            [record],
            {"all", "core", "viz"},
            {"all", "core", "viz"},
        )
        == []
    )


def test_moving_notice_url_requires_a_pending_durable_blocker() -> None:
    """Moving notice pointers cannot be laundered into reviewed evidence."""
    root = Path(__file__).resolve().parents[2]
    policy = json.loads(
        (root / "scripts/validation/dependency_license_policy.v1.json").read_text(encoding="utf-8")
    )
    row = next(row for row in policy["package_dispositions"] if row["package"] == "python-dotenv")
    moving = copy.deepcopy(policy)
    moving_row = next(
        row for row in moving["package_dispositions"] if row["package"] == "python-dotenv"
    )
    moving_row["upstream"]["notice_paths"][0] = (
        "https://github.com/theskumar/python-dotenv/blob/v1.2.1/LICENSE"
    )
    moving_row.pop("evidence_blockers", None)
    _rules, _components, _by_name, _records, issues = _policy_records(moving, root)
    assert any("durable blocker for moving notice URLs" in issue for issue in issues)

    pending = copy.deepcopy(moving)
    pending_row = next(
        row for row in pending["package_dispositions"] if row["package"] == "python-dotenv"
    )
    pending_row["evidence_blockers"] = [
        "The tag URL is moving; immutable source pinning remains unresolved."
    ]
    _rules, _components, _by_name, _records, issues = _policy_records(pending, root)
    assert not any("durable blocker for moving notice URLs" in issue for issue in issues)

    reviewed = copy.deepcopy(pending)
    reviewed_row = next(
        row for row in reviewed["package_dispositions"] if row["package"] == "python-dotenv"
    )
    reviewed_row["status"] = "reviewed"
    reviewed_row["reviewer"] = "independent-maintainer"
    reviewed_row["reviewed_at"] = "2026-09-01"
    _rules, _components, _by_name, _records, issues = _policy_records(reviewed, root)
    assert any("reviewed evidence contains moving" in issue for issue in issues)
    assert row["upstream"]["commit_sha"] == "eaf2a9129ccec6febda0f741eb3bb852c3f947bd"


def test_issue_8163_receipt_binds_policy_license_and_strict_inputs(tmp_path: Path) -> None:
    """The checked-in receipt stays blocked when retained files are stale or unverifiable."""
    root = Path(__file__).resolve().parents[2]
    receipt_path = root / "docs/context/evidence/dependency_license_batch_2026-09-01.receipt.json"
    issues = validate_dependency_license_receipt(root, receipt_path)
    assert any("candidate manifest_path" in issue for issue in issues)
    assert any("strict report SHA-256" in issue for issue in issues)

    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["review_binding"]["normalized_records_sha256"] = "0" * 64
    tampered = tmp_path / "receipt-tampered.json"
    tampered.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    issues = validate_dependency_license_receipt(root, tampered)
    assert any("normalized_records_sha256 differs" in issue for issue in issues)
    assert any(
        "reviewed_head_sha differs" in issue
        for issue in validate_dependency_license_receipt(
            root,
            receipt_path,
            expected_reviewed_head="0" * 40,
        )
    )


def test_issue_8163_receipt_summaries_are_bound_fail_closed(tmp_path: Path) -> None:
    """Receipt status, scope, archive, strict, and candidate claims cannot drift."""
    root = Path(__file__).resolve().parents[2]
    receipt_path = root / "docs/context/evidence/dependency_license_batch_2026-09-01.receipt.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    mutations = (
        ("status", lambda value: value.__setitem__("status", "complete"), "receipt status"),
        (
            "scope",
            lambda value: value.__setitem__("package_count", 1),
            "scope package_count",
        ),
        (
            "archive",
            lambda value: value.__setitem__("artifact_count", 1),
            "archive_audit artifact_count",
        ),
        (
            "strict",
            lambda value: value["summary"].__setitem__("unresolved_count", 1),
            "strict_report unresolved count",
        ),
        (
            "candidate",
            lambda value: value.__setitem__("status", "blocked"),
            "candidate_binding",
        ),
    )
    for name, mutate, expected in mutations:
        tampered = copy.deepcopy(receipt)
        target = {
            "status": tampered,
            "scope": tampered["scope"],
            "archive": tampered["archive_audit"],
            "strict": tampered["strict_report"],
            "candidate": tampered["candidate_binding"],
        }[name]
        mutate(target)
        tampered_path = tmp_path / f"receipt-{name}.json"
        tampered_path.write_text(json.dumps(tampered, indent=2) + "\n", encoding="utf-8")
        assert any(
            expected in issue for issue in validate_dependency_license_receipt(root, tampered_path)
        )

    for name, mutate, expected in (
        ("claim", lambda value: value.__setitem__("claim_boundary", "approved"), "claim_boundary"),
        (
            "reviewer",
            lambda value: value["review"].__setitem__("reviewer", "self"),
            "review identity",
        ),
        (
            "legal",
            lambda value: value["review"].__setitem__("legal_or_redistribution_approval", True),
            "legal_or_redistribution_approval",
        ),
    ):
        tampered = copy.deepcopy(receipt)
        mutate(tampered)
        tampered_path = tmp_path / f"receipt-{name}.json"
        tampered_path.write_text(json.dumps(tampered, indent=2) + "\n", encoding="utf-8")
        assert any(
            expected in issue for issue in validate_dependency_license_receipt(root, tampered_path)
        )


def test_issue_8163_rehashed_audit_mutations_still_fail_semantically(  # noqa: PLR0915
    tmp_path: Path,
) -> None:
    """Receipt file hashes cannot launder forged archive or upstream evidence."""
    root = Path(__file__).resolve().parents[2]
    receipt_path = root / "docs/context/evidence/dependency_license_batch_2026-09-01.receipt.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    policy = json.loads(
        (root / "scripts/validation/dependency_license_policy.v1.json").read_text(encoding="utf-8")
    )
    rows = [
        row
        for row in policy["package_dispositions"]
        if "docs/context/evidence/dependency_license_batch_2026-09-01.md"
        in row.get("evidence_paths", [])
    ]
    archive = {
        "schema_version": "robot-sf.issue-8163-archive-audit.v1",
        "packages": [
            {
                "name": row["package"],
                "version": row["version"],
                "expected_expression": row["license_expression"],
                "source": {
                    key: value for key, value in row["source"].items() if key != "metadata_url"
                },
                "pypi_metadata_url": row["source"]["metadata_url"],
                "pypi_info": {
                    "name": row["package"],
                    "version": row["version"],
                    "requires_python": row["python_requires"],
                    "license": None,
                    "classifiers": [],
                    "home_page": None,
                    "project_urls": {},
                },
                "artifacts": [
                    {
                        **artifact,
                        "url": "https://files.example/" + artifact["filename"],
                        "archive_path": "",
                        "member_count": 1,
                        "notice_paths": [
                            path
                            for path, kind in _policy_archive_notice_kinds(row)
                            if kind == artifact["kind"]
                        ],
                        "metadata_path": "",
                        "metadata_license": None,
                        "metadata_project_urls": {},
                        "metadata_fields": {},
                    }
                    for artifact in row["artifacts"]
                ],
            }
            for row in rows
        ],
        "failures": [],
    }
    tags = []
    for row in rows:
        checks = []
        for archive_path, archive_kind, notice_path, notice_url in _policy_archive_notice_mapping(
            row
        ):
            checks.append(
                {
                    "archive_kind": archive_kind,
                    "archive_path": archive_path,
                    "review_url": notice_url,
                    "status": "present",
                    "upstream_path": notice_path,
                }
            )
        tags.append(
            {
                "name": row["package"],
                "version": row["version"],
                "repository": row["upstream"]["repository"],
                "tag": row["upstream"]["tag"],
                "tags": [row["upstream"]["tag"]],
                "matching_tags": [row["upstream"]["tag"]],
                "errors": [],
                "source_url_key": "Source",
                "notice_checks": checks,
            }
        )

    assert _archive_audit_semantic_issues(archive, rows) == []
    assert _upstream_tags_semantic_issues(tags, rows, archive) == []

    for package_name in ("absl-py", "python-dotenv", "rich-rst"):
        forged = copy.deepcopy(archive)
        package = next(item for item in forged["packages"] if item["name"] == package_name)
        package["artifacts"][0]["notice_paths"], package["artifacts"][1]["notice_paths"] = (
            package["artifacts"][1]["notice_paths"],
            package["artifacts"][0]["notice_paths"],
        )
        assert _archive_audit_semantic_issues(forged, rows)

    referencing_row = next(row for row in rows if row["package"] == "referencing")
    referencing_package = next(
        package for package in archive["packages"] if package["name"] == "referencing"
    )
    assert "referencing-0.37.0/suite/LICENSE" in _archive_notice_paths(referencing_package)
    referencing_tags = next(item for item in tags if item["name"] == "referencing")
    assert all(
        check["archive_path"] != "referencing-0.37.0/suite/LICENSE"
        for check in referencing_tags["notice_checks"]
    )
    fabricated = copy.deepcopy(tags)
    fabricated_row = next(item for item in fabricated if item["name"] == "referencing")
    fabricated_row["notice_checks"].append(
        {
            "archive_kind": "sdist",
            "archive_path": "referencing-0.37.0/suite/LICENSE",
            "upstream_path": "suite/LICENSE",
            "status": "present",
            "review_url": (
                referencing_row["upstream"]["repository"]
                + "/blob/"
                + referencing_row["upstream"]["commit_sha"]
                + "/suite/LICENSE"
            ),
        }
    )
    assert _upstream_tags_semantic_issues(fabricated, rows, archive)

    def write_json(name: str, value: object) -> Path:
        path = tmp_path / name
        path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
        return path

    archive_path = write_json("archive-audit.json", archive)
    tags_path = write_json("upstream-tags.json", tags)
    receipt["archive_audit"]["path"] = f"operator-local:{archive_path}"
    receipt["archive_audit"]["sha256"] = hashlib.sha256(archive_path.read_bytes()).hexdigest()
    receipt["archive_audit"]["upstream_tags_path"] = f"operator-local:{tags_path}"
    receipt["archive_audit"]["upstream_tags_sha256"] = hashlib.sha256(
        tags_path.read_bytes()
    ).hexdigest()

    archive_forged = copy.deepcopy(receipt)
    archive_forged_data = copy.deepcopy(archive)
    archive_forged_data["packages"][0]["expected_expression"] = "MIT"
    archive_forged_path = write_json("archive-audit-forged.json", archive_forged_data)
    archive_forged["archive_audit"]["path"] = f"operator-local:{archive_forged_path}"
    archive_forged["archive_audit"]["sha256"] = hashlib.sha256(
        archive_forged_path.read_bytes()
    ).hexdigest()
    archive_forged_receipt = write_json("receipt-forged-archive.json", archive_forged)
    archive_issues = validate_dependency_license_receipt(root, archive_forged_receipt)
    assert any("archive audit license expression differs" in issue for issue in archive_issues)
    assert not any("archive_audit SHA-256 differs" in issue for issue in archive_issues)

    tags_forged = copy.deepcopy(receipt)
    tags_forged_data = copy.deepcopy(tags)
    tags_forged_data[0]["repository"] = "https://github.com/attacker/forged"
    tags_forged_path = write_json("upstream-tags-forged.json", tags_forged_data)
    tags_forged["archive_audit"]["upstream_tags_path"] = f"operator-local:{tags_forged_path}"
    tags_forged["archive_audit"]["upstream_tags_sha256"] = hashlib.sha256(
        tags_forged_path.read_bytes()
    ).hexdigest()
    tags_forged_receipt = write_json("receipt-forged-tags.json", tags_forged)
    tags_issues = validate_dependency_license_receipt(root, tags_forged_receipt)
    assert any("upstream repository differs" in issue for issue in tags_issues), tags_issues
    assert not any("upstream_tags SHA-256 differs" in issue for issue in tags_issues)

    for name, mutate, expected in (
        (
            "archive-extra",
            lambda value: value["packages"][0].__setitem__("forged", True),
            "archive audit package",
        ),
        (
            "pypi-extra",
            lambda value: value["packages"][0]["pypi_info"].__setitem__("forged", True),
            "PyPI metadata schema",
        ),
        (
            "pypi-requires-python",
            lambda value: value["packages"][0]["pypi_info"].__setitem__("requires_python", ">=0"),
            "PyPI requires_python differs",
        ),
        (
            "artifact-platform",
            lambda value: value["packages"][0]["artifacts"][0].__setitem__(
                "platform_tags", ["forged"]
            ),
            "artifact 0 platform_tags differs",
        ),
        (
            "archive-malformed-version",
            lambda value: value["packages"][0].__setitem__("version", []),
            "unexpected package identity",
        ),
    ):
        forged = copy.deepcopy(receipt)
        forged_data = copy.deepcopy(archive)
        mutate(forged_data)
        forged_path = write_json(f"{name}.json", forged_data)
        forged["archive_audit"]["path"] = f"operator-local:{forged_path}"
        forged["archive_audit"]["sha256"] = hashlib.sha256(forged_path.read_bytes()).hexdigest()
        forged_receipt = write_json(f"receipt-{name}.json", forged)
        forged_issues = validate_dependency_license_receipt(root, forged_receipt)
        assert any(expected in issue for issue in forged_issues), forged_issues

    tag_extra = copy.deepcopy(receipt)
    tag_extra_data = copy.deepcopy(tags)
    tag_extra_data[0]["forged"] = True
    tag_extra_path = write_json("upstream-tags-extra.json", tag_extra_data)
    tag_extra["archive_audit"]["upstream_tags_path"] = f"operator-local:{tag_extra_path}"
    tag_extra["archive_audit"]["upstream_tags_sha256"] = hashlib.sha256(
        tag_extra_path.read_bytes()
    ).hexdigest()
    tag_extra_receipt = write_json("receipt-upstream-tags-extra.json", tag_extra)
    assert any(
        "upstream tags row 0 has unclassified fields" in issue
        for issue in validate_dependency_license_receipt(root, tag_extra_receipt)
    )

    tag_path_forged = copy.deepcopy(receipt)
    tag_path_data = copy.deepcopy(tags)
    tag_path_data[0]["notice_checks"][0]["review_url"] = (
        rows[0]["upstream"]["repository"] + "/blob/" + rows[0]["upstream"]["commit_sha"] + "/FORGED"
    )
    tag_path = write_json("upstream-tags-path-forged.json", tag_path_data)
    tag_path_forged["archive_audit"]["upstream_tags_path"] = f"operator-local:{tag_path}"
    tag_path_forged["archive_audit"]["upstream_tags_sha256"] = hashlib.sha256(
        tag_path.read_bytes()
    ).hexdigest()
    tag_path_receipt = write_json("receipt-upstream-tags-path.json", tag_path_forged)
    assert any(
        "notice URL path differs from policy" in issue
        for issue in validate_dependency_license_receipt(root, tag_path_receipt)
    )


def test_issue_8163_receipt_paths_reject_traversal_and_symlinks(tmp_path: Path) -> None:
    """Operator-local evidence paths cannot escape lexically or through symlinks."""
    root = Path(__file__).resolve().parents[2]
    receipt_path = root / "docs/context/evidence/dependency_license_batch_2026-09-01.receipt.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    traversal = copy.deepcopy(receipt)
    traversal["archive_audit"]["path"] = "operator-local:/tmp/../forged/archive.json"
    traversal_path = tmp_path / "receipt-traversal.json"
    traversal_path.write_text(json.dumps(traversal, indent=2) + "\n", encoding="utf-8")
    assert any(
        "lexical traversal" in issue
        for issue in validate_dependency_license_receipt(root, traversal_path)
    )

    target = tmp_path / "archive-target.json"
    target.write_text("{}\n", encoding="utf-8")
    symlink = tmp_path / "archive-link.json"
    os.symlink(target, symlink)
    linked = copy.deepcopy(receipt)
    linked["archive_audit"]["path"] = f"operator-local:{symlink}"
    linked_path = tmp_path / "receipt-symlink.json"
    linked_path.write_text(json.dumps(linked, indent=2) + "\n", encoding="utf-8")
    assert any(
        "archive audit is missing" in issue
        for issue in validate_dependency_license_receipt(root, linked_path)
    )


def test_issue_8163_cross_repository_notice_mapping_is_exact() -> None:
    """A vendored notice may use its own immutable repository, but never a swapped tuple."""
    root = Path(__file__).resolve().parents[2]
    policy = json.loads(
        (root / "scripts/validation/dependency_license_policy.v1.json").read_text(encoding="utf-8")
    )
    row = next(
        item
        for item in policy["package_dispositions"]
        if item["package"] == "alembic"
        and "docs/context/evidence/dependency_license_batch_2026-09-01.md"
        in item.get("evidence_paths", [])
    )
    mappings = sorted(_policy_archive_notice_mapping(row))
    archive = {
        "schema_version": "robot-sf.issue-8163-archive-audit.v1",
        "packages": [
            {
                "name": row["package"],
                "version": row["version"],
                "expected_expression": row["license_expression"],
                "source": {
                    key: value for key, value in row["source"].items() if key != "metadata_url"
                },
                "pypi_metadata_url": row["source"]["metadata_url"],
                "pypi_info": {
                    "name": row["package"],
                    "version": row["version"],
                    "requires_python": row["python_requires"],
                    "license": None,
                    "classifiers": [],
                    "home_page": None,
                    "project_urls": {},
                },
                "artifacts": [
                    {
                        **artifact,
                        "url": "https://files.example/" + artifact["filename"],
                        "archive_path": "",
                        "member_count": 1,
                        "notice_paths": [
                            path for path, kind, _path, _url in mappings if kind == artifact["kind"]
                        ],
                        "metadata_path": "",
                        "metadata_license": None,
                        "metadata_project_urls": {},
                        "metadata_fields": {},
                    }
                    for artifact in row["artifacts"]
                ],
            }
        ],
        "failures": [],
    }
    tags = [
        {
            "name": row["package"],
            "version": row["version"],
            "source_url_key": "Source",
            "repository": row["upstream"]["repository"],
            "tags": [row["upstream"]["tag"]],
            "matching_tags": [row["upstream"]["tag"]],
            "errors": [],
            "tag": row["upstream"]["tag"],
            "notice_checks": [
                {
                    "archive_kind": kind,
                    "archive_path": archive_path,
                    "upstream_path": upstream_path,
                    "status": "present",
                    "review_url": review_url,
                }
                for archive_path, kind, upstream_path, review_url in mappings
            ],
        }
    ]
    assert _archive_audit_semantic_issues(archive, [row]) == []
    assert _upstream_tags_semantic_issues(tags, [row], archive) == []

    forged = copy.deepcopy(tags)
    forged[0]["notice_checks"][1]["review_url"] = (
        "https://github.com/FortAwesome/Font-Awesome/blob/" + "0" * 40 + "/LICENSE.txt"
    )
    assert _upstream_tags_semantic_issues(forged, [row], archive)

    swapped = copy.deepcopy(tags)
    (
        swapped[0]["notice_checks"][0]["archive_path"],
        swapped[0]["notice_checks"][1]["archive_path"],
    ) = (
        swapped[0]["notice_checks"][1]["archive_path"],
        swapped[0]["notice_checks"][0]["archive_path"],
    )
    assert _upstream_tags_semantic_issues(swapped, [row], archive)


def test_issue_8163_rehashed_strict_report_rejects_extra_semantics(tmp_path: Path) -> None:
    """A strict report's content contract remains fail-closed after hash recomputation."""
    root = Path(__file__).resolve().parents[2]
    receipt_path = root / "docs/context/evidence/dependency_license_batch_2026-09-01.receipt.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    report = {
        "schema_version": "robot-sf.dependency-license-inventory.v1",
        "candidate_binding": {},
        "environment": {},
        "failures": [],
        "installed_not_locked": [],
        "packages": [],
        "policy": {},
        "profile_manifest": {},
        "profiles": [],
        "project": {},
        "repository_inputs": {},
        "structural_issues": [],
        "summary": {},
        "surface": {},
        "target": {},
        "unrepresented_lock_package_dispositions": [],
        "unrepresented_lock_packages": [],
        "unexpected_forged_field": "attacker-controlled",
    }
    report["report_content_sha256"] = _report_content_digest(report)
    report_path = tmp_path / "forged-strict-report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    receipt["strict_report"]["path"] = f"operator-local:{report_path}"
    receipt["strict_report"]["sha256"] = hashlib.sha256(report_path.read_bytes()).hexdigest()
    forged_receipt = tmp_path / "receipt-forged-strict.json"
    forged_receipt.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    issues = validate_dependency_license_receipt(root, forged_receipt)
    assert any("strict report has missing or unclassified fields" in issue for issue in issues)
    assert not any("strict report SHA-256 differs" in issue for issue in issues)

    canonical = build_inventory(root, distributions=[], selected_profile_ids=["all"])
    for name, mutate, expected in (
        ("failures", lambda value: value["failures"].append("forged"), "failures"),
        ("packages", lambda value: value["packages"].append({}), "packages"),
        (
            "environment",
            lambda value: value["environment"].__setitem__("machine", "forged"),
            "environment",
        ),
        (
            "repository-inputs",
            lambda value: value["repository_inputs"].append(
                {"path": "pyproject.toml", "sha256": "0" * 64}
            ),
            "repository_inputs",
        ),
    ):
        forged = copy.deepcopy(canonical)
        mutate(forged)
        forged["report_content_sha256"] = _report_content_digest(forged)
        forged_path = tmp_path / f"strict-{name}.json"
        forged_path.write_text(json.dumps(forged, indent=2) + "\n", encoding="utf-8")
        forged_receipt = copy.deepcopy(receipt)
        forged_receipt["strict_report"]["path"] = f"operator-local:{forged_path}"
        forged_receipt["strict_report"]["sha256"] = hashlib.sha256(
            forged_path.read_bytes()
        ).hexdigest()
        forged_receipt_path = tmp_path / f"receipt-strict-{name}.json"
        forged_receipt_path.write_text(
            json.dumps(forged_receipt, indent=2) + "\n", encoding="utf-8"
        )
        semantic_issues = validate_dependency_license_receipt(root, forged_receipt_path)
        assert any(
            f"strict report {expected} differs from canonical inventory" in issue
            for issue in semantic_issues
        ), semantic_issues


def test_issue_8163_candidate_manifest_rejects_rehashed_invalid_member(
    tmp_path: Path,
) -> None:
    """A candidate manifest cannot admit an extra member merely by rehashing it."""
    _write_inputs(tmp_path)
    bundle = _write_candidate_bundle(tmp_path)
    manifest_path = bundle / "candidate-manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["members"].append(
        {
            "filename": "forged.txt",
            "kind": "not-a-candidate-kind",
            "sha256": "0" * 64,
            "size": 0,
        }
    )
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    candidate = build_inventory(
        tmp_path,
        distributions=[],
        selected_profile_ids=["core"],
        candidate_bundle_path=bundle,
    )["candidate_binding"]
    candidate["manifest_path"] = str(manifest_path)
    candidate["manifest_sha256"] = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    issues, _bundle, _canonical = _candidate_receipt_semantic_issues(
        candidate,
        repo_root=tmp_path,
        expected_profiles=["core"],
    )
    assert any("candidate bundle is invalid" in issue for issue in issues)


def test_issue_8163_candidate_receipt_uses_manifest_member_order(tmp_path: Path) -> None:
    """A canonical candidate receipt validates when its members use manifest order."""
    _write_inputs(tmp_path)
    bundle = _write_candidate_bundle(tmp_path)
    manifest_path = bundle / "candidate-manifest.json"
    binding = build_inventory(
        tmp_path,
        distributions=[],
        selected_profile_ids=["core"],
        candidate_bundle_path=bundle,
    )["candidate_binding"]
    candidate = copy.deepcopy(binding)
    candidate["manifest_path"] = str(manifest_path)
    for member in candidate["members"]:
        member["path"] = str(bundle / member["filename"])
    issues, _bundle, canonical = _candidate_receipt_semantic_issues(
        candidate,
        repo_root=tmp_path,
        expected_profiles=["core"],
    )
    assert issues == []
    assert canonical is not None


def test_github_notice_reference_requires_an_immutable_commit() -> None:
    """The URL parser rejects tags, branches, queries, and missing blob paths."""
    assert _github_notice_reference(
        "https://github.com/theskumar/python-dotenv/blob/"
        "eaf2a9129ccec6febda0f741eb3bb852c3f947bd/LICENSE"
    ) == ("theskumar/python-dotenv", "eaf2a9129ccec6febda0f741eb3bb852c3f947bd", "blob")
    for url in (
        "https://github.com/theskumar/python-dotenv/blob/v1.2.1/LICENSE",
        "https://github.com/theskumar/python-dotenv/tree/main",
        "https://github.com/theskumar/python-dotenv/blob/"
        "eaf2a9129ccec6febda0f741eb3bb852c3f947bd/LICENSE?raw=1",
        "https://fontawesome.com/license/free",
    ):
        assert _github_notice_reference(url) is None


def test_notice_commit_mismatch_and_missing_commit_fail_closed() -> None:
    """A pinned-looking URL still must match the row's resolved source commit."""
    root = Path(__file__).resolve().parents[2]
    policy = json.loads(
        (root / "scripts/validation/dependency_license_policy.v1.json").read_text(encoding="utf-8")
    )
    mismatch = copy.deepcopy(policy)
    mismatch_row = next(
        row for row in mismatch["package_dispositions"] if row["package"] == "python-dotenv"
    )
    mismatch_row["upstream"]["notice_paths"][0] = (
        "https://github.com/theskumar/python-dotenv/blob/"
        "0000000000000000000000000000000000000000/LICENSE"
    )
    _rules, _components, _by_name, _records, issues = _policy_records(mismatch, root)
    assert any("does not match upstream commit_sha" in issue for issue in issues)

    missing = copy.deepcopy(policy)
    missing_row = next(
        row for row in missing["package_dispositions"] if row["package"] == "python-dotenv"
    )
    del missing_row["upstream"]["commit_sha"]
    _rules, _components, _by_name, _records, issues = _policy_records(missing, root)
    assert any("must bind upstream provenance" in issue for issue in issues)


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
