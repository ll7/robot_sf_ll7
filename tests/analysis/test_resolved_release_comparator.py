"""Real release identities; runtime resolution is offline and executes no episodes."""

import hashlib
import importlib.machinery
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from scripts.analysis import compare_release_0_0_7_to_0_0_8 as comparator

SOURCE = "66f402ba176b13e45210d0da0b2cf20fcdc0cc02"
RECEIPT = "cd29d6c9a3213e4dbfabc1b8b8c23305066a6a39c5eb088f54d8f8d539827464"


@pytest.fixture(scope="module")
def resolved_source(tmp_path_factory):
    repository = Path(__file__).resolve().parents[2]
    root = tmp_path_factory.mktemp("resolved-source") / "source"
    subprocess.run(
        ["git", "-C", str(repository), "worktree", "add", "--detach", str(root), SOURCE],
        check=True,
        capture_output=True,
    )
    try:
        receipt = root / "output/release-008/calibration/determinism-receipt.json"
        receipt.parent.mkdir(parents=True)
        receipt.write_bytes(
            (
                repository
                / "docs/context/evidence/2026-10-04_freeze008_f2_calibration/determinism-receipt.json"
            ).read_bytes()
        )
        yield root
    finally:
        subprocess.run(
            ["git", "-C", str(repository), "worktree", "remove", "--force", str(root)],
            check=True,
            capture_output=True,
        )


@pytest.mark.parametrize(
    "track,template,digest,count",
    [
        (
            "main",
            "benchmark_data_release_s30_h600.template.yaml",
            "527f9dc5e9ee3004e93444789472eb25a29438e64741c4c4b7b95a10f0db3e71",
            20160,
        ),
        (
            "doorway",
            "three_width_doorway_release_0_0_8_v1.template.yaml",
            "6fce4eb41436213a2c4d9790b74811b3492126297ea654ef89a1f3d94e17e199",
            1260,
        ),
    ],
)
def test_real_resolved_identity_admitted_without_relabelling(
    resolved_source, track, template, digest, count
):
    root = resolved_source
    path = _generate_identity(root, track, template)
    assert hashlib.sha256(path.read_bytes()).hexdigest() == digest
    verified = comparator._verified_successor_manifest(path, digest, root, {})
    assert verified["source_commit"] == SOURCE
    assert len(verified["expected_slots"]) == count
    assert len(verified["planner_keys"]) == 14
    assert (
        json.loads(path.read_bytes())["schema_version"] == "benchmark-release-resolved-identity.v1"
    )
    tooling = comparator._tooling_identity()
    assert tooling["commit"] != SOURCE
    with pytest.raises(ValueError, match="must be clean"):
        comparator._tooling_identity(SOURCE)
    citation = root / "CITATION.cff"
    original_citation = citation.read_bytes()
    citation.write_bytes(original_citation + b"\n# synthetic dirty-source control\n")
    try:
        with pytest.raises(ValueError, match="must be clean"):
            comparator._verified_successor_manifest(path, digest, root, {})
    finally:
        citation.write_bytes(original_citation)
    # A freshly hashed forgery must still fail canonical source verification.
    original = path.read_bytes()
    forged = json.loads(original)
    forged["resolved_manifest"]["canonical_campaign_config_sha256"] = "0" * 64
    path.write_text(json.dumps(forged))
    try:
        with pytest.raises(ValueError, match="stale or non-canonical"):
            comparator._verified_successor_manifest(
                path, hashlib.sha256(path.read_bytes()).hexdigest(), root, {}
            )
    finally:
        path.write_bytes(original)


def _generate_identity(root, track, template):
    path = root / f"output/release-008/{track}/release_identity.resolved.json"
    subprocess.run(
        [
            sys.executable,
            str(root / "scripts/tools/resolve_benchmark_release_identity.py"),
            "generate",
            "--repository-root",
            str(root),
            "--template",
            "configs/benchmarks/releases/" + template,
            "--output",
            str(path),
            "--source-commit",
            SOURCE,
            "--release-tag",
            f"paper-matrix-v2-h600-s30-2026-10-{SOURCE}",
            "--concept-doi",
            "10.5281/zenodo.23150471",
            "--version-doi",
            "10.5281/zenodo.23150472",
            "--determinism-receipt-path",
            "output/release-008/calibration/determinism-receipt.json",
            "--determinism-receipt-sha256",
            RECEIPT,
        ],
        cwd=root,
        env={**os.environ, "PYTHONPATH": str(root) + os.pathsep + str(root / "fast-pysf")},
        check=True,
        capture_output=True,
    )
    return path


def test_ignored_native_extension_cannot_shadow_frozen_runtime(resolved_source):
    """An ignored import override in custody must never enter the frozen worker."""
    root = resolved_source
    path = _generate_identity(root, "main", "benchmark_data_release_s30_h600.template.yaml")
    shadow = root / (
        "robot_sf/benchmark/camera_ready/_util" + importlib.machinery.EXTENSION_SUFFIXES[0]
    )
    shadow.write_bytes(b"synthetic invalid native extension; no executable code")
    try:
        clean = subprocess.run(
            ["git", "-C", str(root), "status", "--porcelain", "--untracked-files=all"],
            check=True,
            capture_output=True,
            text=True,
        )
        assert not clean.stdout
        verified = comparator._verified_successor_manifest(
            path,
            hashlib.sha256(path.read_bytes()).hexdigest(),
            root,
            {},
        )
        assert len(verified["expected_slots"]) == 20160
        assert verified["source_commit"] == SOURCE
    finally:
        shadow.unlink()
