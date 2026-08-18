"""Tests for the fail-closed distribution-license archive gate."""

from __future__ import annotations

import io
import tarfile
import zipfile
from typing import TYPE_CHECKING

import pytest

from scripts.tools.check_distribution_licenses import (
    DistributionLicenseError,
    check_distribution,
)

if TYPE_CHECKING:
    from collections.abc import Mapping
    from pathlib import Path


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


def _write_wheel(path: Path, payloads: Mapping[str, str]) -> None:
    """Write a minimal wheel-like ZIP fixture."""
    with zipfile.ZipFile(path, "w") as archive:
        for name, content in _archive_members(
            "robot_sf-0.0.0.dist-info/licenses", payloads
        ).items():
            archive.writestr(name, content)


def _write_sdist(path: Path, payloads: Mapping[str, str]) -> None:
    """Write a minimal source-distribution tarball fixture."""
    with tarfile.open(path, "w:gz") as archive:
        for name, content in _archive_members("robot_sf-0.0.0", payloads).items():
            data = content.encode()
            info = tarfile.TarInfo(name)
            info.size = len(data)
            archive.addfile(info, fileobj=io.BytesIO(data))


def _write_robot_sf_archives(dist_dir: Path, payloads: Mapping[str, str] = PAYLOADS) -> None:
    """Create the minimum Robot SF wheel and sdist fixture."""
    _write_wheel(dist_dir / "robot_sf-0.0.0-py3-none-any.whl", payloads)
    _write_sdist(dist_dir / "robot_sf-0.0.0.tar.gz", payloads)


def _write_pyrvo2_wheel(path: Path, *, content: str) -> None:
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


def test_valid_wheel_and_sdist_pass(tmp_path: Path) -> None:
    """A valid wheel and sdist carry every required text payload."""
    _write_robot_sf_archives(tmp_path)

    result = check_distribution(tmp_path)

    assert len(result.wheels) == 1
    assert len(result.sdists) == 1


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
