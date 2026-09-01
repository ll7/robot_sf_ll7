"""Tests for the fail-closed distribution-license archive gate."""

from __future__ import annotations

import io
import os
import subprocess
import sys
import tarfile
import zipfile
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from scripts.tools.check_distribution_licenses import (
    DistributionLicenseError,
    check_archive_member_contract,
    check_distribution,
    check_source_tree_member_contract,
)

if TYPE_CHECKING:
    from collections.abc import Mapping


REPO_ROOT = Path(__file__).resolve().parents[2]


PAYLOADS = {
    "LICENSE": (
        "GNU GENERAL PUBLIC LICENSE\nVersion 3\nCopyright (C) 2007 Free Software Foundation\n"
    ),
    "fast-pysf/LICENSE": (
        "MIT License\nCopyright (c) 2020 Yuxiang Gao\nPermission is hereby granted\n"
    ),
    "third_party/python-rvo2/LICENSE": (
        "Apache License\nVersion 2.0\n"
        "TERMS AND CONDITIONS FOR USE, REPRODUCTION, AND DISTRIBUTION\n"
    ),
    "third_party/socnavbench/LICENSE": (
        "MIT License\nCopyright (c) 2020 Transportation, Bots, and Disability (TBD) Lab\n"
        "Permission is hereby granted\n"
    ),
    "third_party/socnavbench/LICENSES/Apache-2.0.txt": (
        "Apache License\nVersion 2.0\n"
        "TERMS AND CONDITIONS FOR USE, REPRODUCTION, AND DISTRIBUTION\n"
    ),
    "third_party/socnavbench/LICENSING.yaml": (
        "schema_version: robot_sf.third_party_licensing.v1\n"
        "source_repository: https://github.com/CMU-TBD/SocNavBench\n"
        "source_revision: 0123456789abcdef0123456789abcdef01234567\n"
        "default_license_spdx: MIT\n"
        "upstream_files: []\n"
        "license_overrides: []\n"
        "local_files: []\n"
    ),
    "third_party/socnavbench/UPSTREAM.md": (
        "Origin: https://github.com/CMU-TBD/SocNavBench\n"
        "Commit: 0123456789abcdef0123456789abcdef01234567\n"
        "License: MIT\n"
    ),
    "THIRD_PARTY_NOTICES.md": (
        "# Third-party notices\nfast-pysf\nMIT License\nYuxiang Gao\n"
        "python-rvo2\nApache License, Version 2.0\nSocNavBench\nTBD) Lab\n"
        "does not include model weights\n"
    ),
    "third_party/python-rvo2/UPSTREAM.md": (
        "upstream_repository\nsource_archive_sha256\nLOCAL_CHANGES.patch\n"
    ),
    "third_party/python-rvo2/LOCAL_CHANGES.patch": ("diff -ruN\nthird_party/python-rvo2\n"),
}


def _archive_members(prefix: str, payloads: Mapping[str, str]) -> dict[str, str]:
    """Prefix fixture members as a build backend would prefix an archive."""
    return {f"{prefix}/{name}": content for name, content in payloads.items()}


def _write_wheel(
    path: Path,
    payloads: Mapping[str, str],
    *,
    package_payloads: Mapping[str, str] | None = None,
) -> None:
    """Write a minimal wheel-like ZIP fixture."""
    with zipfile.ZipFile(path, "w") as archive:
        for name, content in _archive_members(
            "robot_sf-0.0.0.dist-info/licenses", payloads
        ).items():
            archive.writestr(name, content)
        for name, content in (package_payloads or {}).items():
            archive.writestr(name, content)


def _write_sdist(path: Path, payloads: Mapping[str, str]) -> None:
    """Write a minimal source-distribution tarball fixture."""
    with tarfile.open(path, "w:gz") as archive:
        for name, content in _archive_members("robot_sf-0.0.0", payloads).items():
            data = content.encode()
            info = tarfile.TarInfo(name)
            info.size = len(data)
            archive.addfile(info, fileobj=io.BytesIO(data))


def _write_robot_sf_archives(
    dist_dir: Path,
    payloads: Mapping[str, str] = PAYLOADS,
    *,
    package_payloads: Mapping[str, str] | None = None,
) -> None:
    """Create the minimum Robot SF wheel and sdist fixture."""
    _write_wheel(
        dist_dir / "robot_sf-0.0.0-py3-none-any.whl",
        payloads,
        package_payloads=package_payloads,
    )
    _write_sdist(dist_dir / "robot_sf-0.0.0.tar.gz", payloads)


def _write_pyrvo2_wheel(
    path: Path,
    *,
    content: str,
    package_payloads: Mapping[str, str] | None = None,
) -> None:
    """Write a minimal companion wheel carrying legal and provenance payloads."""
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("pyrvo2-0.0.0.dist-info/licenses/LICENSE", content)
        archive.writestr(
            "pyrvo2-0.0.0.dist-info/licenses/UPSTREAM.md",
            PAYLOADS["third_party/python-rvo2/UPSTREAM.md"],
        )
        archive.writestr(
            "pyrvo2-0.0.0.dist-info/licenses/LOCAL_CHANGES.patch",
            PAYLOADS["third_party/python-rvo2/LOCAL_CHANGES.patch"],
        )
        for name, value in (package_payloads or {}).items():
            archive.writestr(name, value)


def test_valid_wheel_and_sdist_pass(tmp_path: Path) -> None:
    """A valid wheel and sdist carry every required text payload."""
    _write_robot_sf_archives(tmp_path)

    result = check_distribution(tmp_path)

    assert len(result.wheels) == 1
    assert len(result.sdists) == 1


def test_direct_script_invocation_is_importable(tmp_path: Path) -> None:
    """The CI's no-project direct-script invocation must import repository helpers."""
    env = {key: value for key, value in os.environ.items() if key != "PYTHONPATH"}
    result = subprocess.run(
        [sys.executable, str(REPO_ROOT / "scripts/tools/check_distribution_licenses.py"), "--help"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )

    assert result.returncode == 0
    assert "--strict-asset-rights" in result.stdout
    assert "ModuleNotFoundError" not in result.stderr


def test_socnavbench_file_partition_is_checked_semantically(tmp_path: Path) -> None:
    """A declared upstream source file is accepted only when it is in the archive."""
    payloads = dict(PAYLOADS)
    payloads["third_party/socnavbench/agents/agent.py"] = "MIT upstream source\n"
    payloads["third_party/socnavbench/LICENSING.yaml"] = payloads[
        "third_party/socnavbench/LICENSING.yaml"
    ].replace("upstream_files: []", "upstream_files:\n  - agents/agent.py")
    _write_robot_sf_archives(tmp_path, payloads)

    result = check_distribution(tmp_path)

    assert len(result.sdists) == 1


def test_socnavbench_apache_override_of_an_upstream_file_is_accepted(tmp_path: Path) -> None:
    """An Apache-2.0 override refines an upstream file and is not a separate partition."""
    payloads = dict(PAYLOADS)
    payloads["third_party/socnavbench/mp_env/map_utils.py"] = (
        "# Apache License, Version 2.0\nupstream mesh loader\n"
    )
    payloads["third_party/socnavbench/LICENSING.yaml"] = (
        payloads["third_party/socnavbench/LICENSING.yaml"]
        .replace("upstream_files: []", "upstream_files:\n  - mp_env/map_utils.py")
        .replace(
            "license_overrides: []",
            "license_overrides:\n"
            "  - license_spdx: Apache-2.0\n"
            "    files:\n"
            "      - mp_env/map_utils.py\n",
        )
    )
    _write_robot_sf_archives(tmp_path, payloads)

    result = check_distribution(tmp_path)

    assert len(result.sdists) == 1


def test_socnavbench_external_checkout_notices_without_source_are_accepted(
    tmp_path: Path,
) -> None:
    """A sanitized candidate may retain SocNavBench notices without its source checkout."""
    payloads = dict(PAYLOADS)
    payloads["third_party/socnavbench/LICENSING.yaml"] = (
        payloads["third_party/socnavbench/LICENSING.yaml"]
        .replace(
            "upstream_files: []",
            "upstream_files:\n  - mp_env/map_utils.py",
        )
        .replace(
            "license_overrides: []",
            "license_overrides:\n"
            "  - license_spdx: Apache-2.0\n"
            "    files:\n"
            "      - mp_env/map_utils.py\n",
        )
    )
    _write_robot_sf_archives(tmp_path, payloads)

    result = check_distribution(tmp_path)

    assert len(result.sdists) == 1


def test_socnavbench_override_outside_upstream_files_fails_closed(tmp_path: Path) -> None:
    """An override entry that is not declared upstream leaves the manifest ambiguous."""
    payloads = dict(PAYLOADS)
    payloads["third_party/socnavbench/mp_env/map_utils.py"] = (
        "# Apache License, Version 2.0\nupstream mesh loader\n"
    )
    payloads["third_party/socnavbench/LICENSING.yaml"] = payloads[
        "third_party/socnavbench/LICENSING.yaml"
    ].replace(
        "license_overrides: []",
        "license_overrides:\n"
        "  - license_spdx: Apache-2.0\n"
        "    files:\n"
        "      - mp_env/map_utils.py\n",
    )
    _write_robot_sf_archives(tmp_path, payloads)

    with pytest.raises(
        DistributionLicenseError,
        match="overrides must also be listed as upstream files",
    ):
        check_distribution(tmp_path)


def test_socnavbench_unclassified_source_file_fails_closed(tmp_path: Path) -> None:
    """A vendored SocNavBench source file cannot bypass the ownership manifest."""
    payloads = dict(PAYLOADS)
    payloads["third_party/socnavbench/agents/agent.py"] = "MIT upstream source\n"
    _write_robot_sf_archives(tmp_path, payloads)

    with pytest.raises(DistributionLicenseError, match="source files are unclassified"):
        check_distribution(tmp_path)


def test_missing_sdist_fails_closed(tmp_path: Path) -> None:
    """A wheel-only directory is not a releasable distribution."""
    _write_wheel(tmp_path / "robot_sf-0.0.0-py3-none-any.whl", PAYLOADS)

    with pytest.raises(DistributionLicenseError, match="missing at least one Robot SF sdist"):
        check_distribution(tmp_path)


def test_wrong_license_content_fails_closed(tmp_path: Path) -> None:
    """A present license filename with tampered text must fail the gate."""
    wrong_payloads = dict(PAYLOADS)
    wrong_payloads["third_party/socnavbench/LICENSE"] = "MIT License\nnot the required copyright\n"
    _write_robot_sf_archives(tmp_path, wrong_payloads)

    with pytest.raises(DistributionLicenseError, match="SocNavBench MIT license.*wrong"):
        check_distribution(tmp_path)


def test_required_pyrvo2_companion_is_enforced(tmp_path: Path) -> None:
    """The companion-wheel lane can opt into a separate required artifact check."""
    _write_robot_sf_archives(tmp_path)

    with pytest.raises(DistributionLicenseError, match="missing required pyrvo2 companion wheel"):
        check_distribution(tmp_path, require_pyrvo2=True)

    _write_pyrvo2_wheel(
        tmp_path / "pyrvo2-0.0.0-cp312-cp312-linux_x86_64.whl",
        content=PAYLOADS["third_party/python-rvo2/LICENSE"],
    )
    check_distribution(tmp_path, require_pyrvo2=True)


def test_pyrvo2_license_content_is_checked(tmp_path: Path) -> None:
    """A companion wheel with a missing or tampered Apache license is rejected."""
    _write_robot_sf_archives(tmp_path)
    wheel = tmp_path / "pyrvo2-0.0.0-py3-none-any.whl"
    _write_pyrvo2_wheel(wheel, content="Apache License\nVersion 2.0\n")

    with pytest.raises(DistributionLicenseError, match="pyrvo2 Apache license.*wrong"):
        check_distribution(tmp_path, require_pyrvo2=True)


def test_license_gate_rejects_noncanonical_payload_paths(tmp_path: Path) -> None:
    """A decoy nested filename must not satisfy the canonical archive contract."""
    decoy_payloads = {f"decoy/{name}": content for name, content in PAYLOADS.items()}
    _write_wheel(tmp_path / "robot_sf-0.0.0-py3-none-any.whl", decoy_payloads)
    _write_sdist(tmp_path / "robot_sf-0.0.0.tar.gz", decoy_payloads)

    with pytest.raises(DistributionLicenseError, match="missing root GPL license"):
        check_distribution(tmp_path)


def test_license_gate_rejects_model_artifacts_in_source_distribution(tmp_path: Path) -> None:
    """A source archive must not smuggle model artifacts past the license gate."""
    payloads = dict(PAYLOADS)
    payloads["model/run_043.zip"] = "checkpoint bytes"
    _write_robot_sf_archives(tmp_path, payloads)

    with pytest.raises(
        DistributionLicenseError,
        match="forbidden source-distribution model artifact members",
    ):
        check_distribution(tmp_path)


def test_strict_archive_contract_accepts_clean_fixture(tmp_path: Path) -> None:
    """Strict member validation accepts a fixture with no release asset members."""
    _write_robot_sf_archives(tmp_path)

    result = check_distribution(tmp_path, strict_asset_rights=True, repo_root=REPO_ROOT)

    assert len(result.wheels) == 1
    assert len(result.sdists) == 1


def test_strict_archive_contract_uses_repo_root_default_inventory(tmp_path: Path) -> None:
    """An alternate checkout's inventory is selected when no inventory path is supplied."""
    repo_root = tmp_path / "repo"
    (repo_root / "assets").mkdir(parents=True)
    (repo_root / "assets/example.svg").write_text("<svg />\n", encoding="utf-8")
    canonical_path = repo_root / "scripts" / "validation" / "asset_rights_inventory.v1.yaml"
    canonical_path.parent.mkdir(parents=True)
    (repo_root / "evidence.txt").write_text("fixture evidence\n", encoding="utf-8")
    canonical_path.write_text(
        """schema_version: robot_sf.asset_rights_inventory.v1
claim_boundary: synthetic test inventory only
tracked_scopes:
  - id: assets
    globs: [assets/**]
    release_relevant: true
rows:
  - id: asset
    scope: assets
    globs: [assets/*.svg]
    status: project-authored
    source: fixture
    source_revision_or_access_date: fixture
    license_or_rights: fixture
    attribution: fixture
    checksum_policy: fixture
    modification_status: fixture
    evidence: [evidence.txt]
""",
        encoding="utf-8",
    )
    subprocess.run(["git", "init", "--quiet", str(repo_root)], check=True)
    subprocess.run(["git", "-C", str(repo_root), "config", "user.name", "fixture"], check=True)
    subprocess.run(
        ["git", "-C", str(repo_root), "config", "user.email", "fixture@example.invalid"],
        check=True,
    )
    subprocess.run(["git", "-C", str(repo_root), "add", "."], check=True)
    subprocess.run(["git", "-C", str(repo_root), "commit", "--quiet", "-m", "fixture"], check=True)
    dist_dir = tmp_path / "dist"
    dist_dir.mkdir()
    _write_wheel(
        dist_dir / "robot_sf-0.0.0-py3-none-any.whl",
        PAYLOADS,
        package_payloads={"assets/example.svg": "<svg />\n"},
    )

    errors = check_archive_member_contract(
        dist_dir / "robot_sf-0.0.0-py3-none-any.whl",
        repo_root=repo_root,
    )

    assert errors == ()


def test_strict_archive_contract_rejects_known_blocked_asset(tmp_path: Path) -> None:
    """A known blocked inventory path cannot enter either software archive."""
    payloads = dict(PAYLOADS)
    payloads["examples/datasets/2024-12-06_15-39-44.json"] = "blocked example data"
    _write_robot_sf_archives(tmp_path, payloads)

    with pytest.raises(
        DistributionLicenseError,
        match="asset member has non-release inventory status 'blocked'",
    ):
        check_distribution(tmp_path, strict_asset_rights=True, repo_root=REPO_ROOT)


def test_strict_archive_contract_rejects_unclassified_asset(tmp_path: Path) -> None:
    """A newly added asset-like archive member needs an inventory row first."""
    payloads = dict(PAYLOADS)
    payloads["examples/datasets/new-recording.json"] = "unclassified example data"
    _write_robot_sf_archives(tmp_path, payloads)

    with pytest.raises(
        DistributionLicenseError,
        match="asset member is not covered by the tracked rights inventory",
    ):
        check_distribution(tmp_path, strict_asset_rights=True, repo_root=REPO_ROOT)


def test_strict_archive_contract_rejects_model_member_in_wheel(tmp_path: Path) -> None:
    """Model bytes are forbidden even when a wheel carries them outside the sdist root."""
    _write_robot_sf_archives(
        tmp_path,
        package_payloads={"model/checkpoint.zip": "checkpoint bytes"},
    )

    with pytest.raises(
        DistributionLicenseError,
        match="model artifact member is forbidden in a software distribution",
    ):
        check_distribution(tmp_path, strict_asset_rights=True, repo_root=REPO_ROOT)


def test_strict_archive_contract_rejects_nested_reserved_model_members(tmp_path: Path) -> None:
    """Noncanonical nested reserved directories cannot hide model payloads in a wheel."""
    _write_robot_sf_archives(
        tmp_path,
        package_payloads={
            "evil.data/model/hidden-checkpoint.zip": "checkpoint bytes",
            "evil.dist-info/model/other-checkpoint.zip": "checkpoint bytes",
        },
    )

    with pytest.raises(
        DistributionLicenseError,
        match="model artifact member is forbidden in a software distribution",
    ):
        check_distribution(tmp_path, strict_asset_rights=True, repo_root=REPO_ROOT)


def test_strict_archive_contract_checks_pyrvo2_companion_members(tmp_path: Path) -> None:
    """A required companion wheel cannot carry a model payload outside its metadata root."""
    _write_robot_sf_archives(tmp_path)
    _write_pyrvo2_wheel(
        tmp_path / "pyrvo2-0.0.0-cp312-cp312-linux_x86_64.whl",
        content=PAYLOADS["third_party/python-rvo2/LICENSE"],
        package_payloads={"pyrvo2-0.0.0/model/hidden-checkpoint.zip": "checkpoint bytes"},
    )

    with pytest.raises(
        DistributionLicenseError,
        match="pyrvo2-0.0.0-cp312-cp312-linux_x86_64.whl: model artifact member",
    ):
        check_distribution(
            tmp_path,
            require_pyrvo2=True,
            strict_asset_rights=True,
            repo_root=REPO_ROOT,
        )


def test_strict_archive_contract_binds_pyrvo2_members_to_vendored_root(
    tmp_path: Path,
) -> None:
    """Companion payloads are mapped to their vendored source root for inventory checks."""
    wheel = tmp_path / "pyrvo2-0.0.0-cp312-cp312-linux_x86_64.whl"
    _write_pyrvo2_wheel(
        wheel,
        content=PAYLOADS["third_party/python-rvo2/LICENSE"],
        package_payloads={"maps/hidden.svg": "<svg />\n"},
    )

    errors = check_archive_member_contract(wheel, repo_root=REPO_ROOT)

    assert any("source path third_party/python-rvo2/maps/hidden.svg" in error for error in errors)


def test_strict_archive_contract_rejects_unsafe_member_path(tmp_path: Path) -> None:
    """Archive traversal names fail before they can be mapped to a repository path."""
    archive_path = tmp_path / "robot_sf-0.0.0.tar.gz"
    with tarfile.open(archive_path, "w:gz") as archive:
        info = tarfile.TarInfo("robot_sf-0.0.0/../escape.txt")
        data = b"escape"
        info.size = len(data)
        archive.addfile(info, fileobj=io.BytesIO(data))

    errors = check_archive_member_contract(archive_path, repo_root=REPO_ROOT)

    assert errors == ("unsafe archive member path: 'robot_sf-0.0.0/../escape.txt'",)


def test_strict_archive_contract_rejects_windows_absolute_path(tmp_path: Path) -> None:
    """Windows drive-qualified names cannot bypass POSIX traversal checks."""
    archive_path = tmp_path / "robot_sf-0.0.0.tar.gz"
    with tarfile.open(archive_path, "w:gz") as archive:
        info = tarfile.TarInfo("C:/escape.txt")
        data = b"escape"
        info.size = len(data)
        archive.addfile(info, fileobj=io.BytesIO(data))

    errors = check_archive_member_contract(archive_path, repo_root=REPO_ROOT)

    assert errors == ("unsafe archive member path: 'C:/escape.txt'",)


@pytest.mark.parametrize(
    "member_name",
    [
        "C:relative.txt",
        "robot_sf-0.0.0/root/.. /escape.txt",
        "robot_sf-0.0.0/root/escape. /file.txt",
    ],
)
def test_strict_archive_contract_rejects_cross_platform_unsafe_names(
    tmp_path: Path, member_name: str
) -> None:
    """Drive-relative and Windows-normalized names cannot enter a release archive."""
    archive_path = tmp_path / "robot_sf-0.0.0.tar.gz"
    with tarfile.open(archive_path, "w:gz") as archive:
        info = tarfile.TarInfo(member_name)
        data = b"unsafe"
        info.size = len(data)
        archive.addfile(info, fileobj=io.BytesIO(data))

    errors = check_archive_member_contract(archive_path, repo_root=REPO_ROOT)

    assert errors == (f"unsafe archive member path: {member_name!r}",)


def test_strict_archive_contract_rejects_duplicate_members(tmp_path: Path) -> None:
    """Duplicate ZIP names cannot be silently collapsed during archive inspection."""
    archive_path = tmp_path / "robot_sf-0.0.0-py3-none-any.whl"
    member_name = "robot_sf-0.0.0.dist-info/licenses/LICENSE"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr(member_name, "first")
        archive.writestr(member_name, "second")

    errors = check_archive_member_contract(archive_path, repo_root=REPO_ROOT)

    assert errors == (
        "robot_sf-0.0.0-py3-none-any.whl: duplicate archive member names: "
        "robot_sf-0.0.0.dist-info/licenses/LICENSE",
    )


def test_strict_archive_contract_rejects_tar_symlink(tmp_path: Path) -> None:
    """Symlinks are not regular release payload members and must be rejected."""
    archive_path = tmp_path / "robot_sf-0.0.0.tar.gz"
    with tarfile.open(archive_path, "w:gz") as archive:
        info = tarfile.TarInfo("robot_sf-0.0.0/LICENSE")
        info.type = tarfile.SYMTYPE
        info.linkname = "../../outside"
        archive.addfile(info)

    errors = check_archive_member_contract(archive_path, repo_root=REPO_ROOT)

    assert errors == (
        "robot_sf-0.0.0.tar.gz: non-regular archive members are forbidden: robot_sf-0.0.0/LICENSE",
    )


def test_strict_source_tree_contract_rejects_tracked_symlink(tmp_path: Path) -> None:
    """The Git source-tree route rejects symlink modes instead of discarding file types."""
    repo_root = tmp_path / "repo"
    (repo_root / "assets").mkdir(parents=True)
    (repo_root / "outside.txt").write_text("outside\n", encoding="utf-8")
    (repo_root / "assets/escape.svg").symlink_to(repo_root / "outside.txt")
    subprocess.run(["git", "init", "--quiet", str(repo_root)], check=True)
    subprocess.run(["git", "-C", str(repo_root), "config", "user.name", "fixture"], check=True)
    subprocess.run(
        ["git", "-C", str(repo_root), "config", "user.email", "fixture@example.invalid"],
        check=True,
    )
    subprocess.run(["git", "-C", str(repo_root), "add", "."], check=True)
    subprocess.run(["git", "-C", str(repo_root), "commit", "--quiet", "-m", "fixture"], check=True)

    errors = check_source_tree_member_contract(repo_root, source_ref="HEAD")

    assert errors[0] == (
        "source tree 'HEAD' contains non-regular Git member 'assets/escape.svg' "
        "(mode 120000, type blob)"
    )


def test_strict_source_tree_contract_rejects_current_blockers() -> None:
    """The proposed current Git tree is not publishable while known blockers remain tracked."""
    errors = check_source_tree_member_contract(REPO_ROOT, source_ref="HEAD")

    assert any("examples/datasets/2024-12-06_15-39-44.json" in error for error in errors)
    assert any("model/registry.yaml" in error for error in errors)
