"""Tests for the release evidence reproduction gate (issue #3205).

These exercise the real ``benchmark_publication_bundle.py dissertation-bundle`` generator
through a small synthetic fixture so the gate is validated against the actual reproduction
contract, not a mock. They prove a faithful reproduction yields ``PASS`` and that tampered
rows or a corrupted archive fail closed.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import subprocess
import sys
import tarfile
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
_GATE_PATH = _REPO_ROOT / "scripts" / "dev" / "release_evidence_gate.py"
_BUNDLE_TOOL = _REPO_ROOT / "scripts" / "tools" / "benchmark_publication_bundle.py"

_spec = importlib.util.spec_from_file_location("release_evidence_gate", _GATE_PATH)
assert _spec and _spec.loader
gate = importlib.util.module_from_spec(_spec)
sys.modules["release_evidence_gate"] = gate  # required so dataclasses can resolve the module
_spec.loader.exec_module(gate)


def _make_source_root(
    root: Path, body: str = "| planner | success |\n| --- | --- |\n| ppo | 0.88 |\n"
) -> Path:
    """Create a synthetic ``payload/reports`` source root and return the reports dir."""
    reports = root / "payload" / "reports"
    reports.mkdir(parents=True)
    (reports / "campaign_table_core.md").write_text(body)
    return reports


def _make_spec(spec_path: Path) -> None:
    """Write a minimal one-artifact dissertation artifact spec."""
    spec_path.write_text(
        json.dumps(
            {
                "artifacts": [
                    {
                        "artifact_id": "tab_results_overview",
                        "source_path": "campaign_table_core.md",
                        "source_artifact": "Synthetic fixture core table",
                        "caption_draft": "Synthetic core planner overview for gate tests.",
                        "claim_boundary": (
                            "Synthetic fixture rows for reproduction-gate testing only; "
                            "not benchmark evidence and not a manuscript claim."
                        ),
                        "recommended_manuscript_use": "discussion",
                        "fallback_degraded_summary": "Synthetic fixture; no fallback rows.",
                        "metadata": {"table_label": "tab:results-overview"},
                    }
                ]
            }
        )
    )


def _build_reference_manifest(source_root: Path, spec_path: Path, out: Path, commit: str) -> Path:
    """Generate one bundle and return its manifest to act as the tracked reference."""
    subprocess.run(
        [
            sys.executable,
            str(_BUNDLE_TOOL),
            "dissertation-bundle",
            "--source-root",
            str(source_root),
            "--out-dir",
            str(out),
            "--bundle-name",
            "gate_fixture_bundle",
            "--artifact-spec",
            str(spec_path),
            "--command",
            "fixture",
            "--commit",
            commit,
            "--overwrite",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return out / "gate_fixture_bundle" / "artifact_manifest.json"


@pytest.fixture
def fixture(tmp_path: Path) -> dict[str, Path]:
    """Build a source root, spec, and a reference manifest from one faithful generation."""
    commit = "deadbeefcafe"
    src = _make_source_root(tmp_path / "src")
    spec = tmp_path / "spec.json"
    _make_spec(spec)
    ref = _build_reference_manifest(src, spec, tmp_path / "ref", commit)
    return {"src": src, "spec": spec, "ref": ref, "commit": Path(commit)}


def test_faithful_reproduction_passes(fixture: dict[str, Path]) -> None:
    """A faithful regeneration of the canonical rows reproduces every checksum (PASS)."""
    snapshot = gate.run_gate(
        artifact_spec=fixture["spec"],
        reference_manifest=fixture["ref"],
        source_commit=str(fixture["commit"]),
        acquisition=gate.AcquisitionSpec(source_root=fixture["src"]),
    )
    assert snapshot["status"] == "PASS"
    assert snapshot["reproduced_artifact_count"] == snapshot["expected_artifact_count"] == 1
    assert snapshot["failures"] == []
    assert snapshot["artifacts"][0]["match"] is True


def test_tampered_rows_fail_closed(fixture: dict[str, Path]) -> None:
    """Mutating a source row changes its checksum and fails the gate closed."""
    (fixture["src"] / "campaign_table_core.md").write_text("| planner | success |\n| x | 0.99 |\n")
    snapshot = gate.run_gate(
        artifact_spec=fixture["spec"],
        reference_manifest=fixture["ref"],
        source_commit=str(fixture["commit"]),
        acquisition=gate.AcquisitionSpec(source_root=fixture["src"]),
    )
    assert snapshot["status"] == "FAIL"
    assert any("checksum mismatch" in f for f in snapshot["failures"])


def test_missing_artifact_fails_closed(fixture: dict[str, Path]) -> None:
    """A reference artifact that cannot be regenerated fails the gate closed."""
    reference = json.loads(fixture["ref"].read_text())
    reference["artifacts"].append({"artifact_id": "tab_absent", "sha256": "0" * 64})
    fixture["ref"].write_text(json.dumps(reference))
    snapshot = gate.run_gate(
        artifact_spec=fixture["spec"],
        reference_manifest=fixture["ref"],
        source_commit=str(fixture["commit"]),
        acquisition=gate.AcquisitionSpec(source_root=fixture["src"]),
    )
    assert snapshot["status"] == "FAIL"
    assert any("not regenerated" in f for f in snapshot["failures"])


def test_archive_integrity_pass_and_fail(tmp_path: Path, fixture: dict[str, Path]) -> None:
    """Archive mode passes on a matching SHA-256 and fails closed on a corrupted one."""
    archive = tmp_path / "bundle.tar.gz"
    with tarfile.open(archive, "w:gz") as tar:
        tar.add(fixture["src"].parent.parent, arcname="bundle")
    good_sha = hashlib.sha256(archive.read_bytes()).hexdigest()

    ok = gate.run_gate(
        artifact_spec=fixture["spec"],
        reference_manifest=fixture["ref"],
        source_commit=str(fixture["commit"]),
        acquisition=gate.AcquisitionSpec(archive=archive, expected_archive_sha256=good_sha),
    )
    assert ok["status"] == "PASS"
    assert ok["archive"]["match"] is True

    bad = gate.run_gate(
        artifact_spec=fixture["spec"],
        reference_manifest=fixture["ref"],
        source_commit=str(fixture["commit"]),
        acquisition=gate.AcquisitionSpec(archive=archive, expected_archive_sha256="f" * 64),
    )
    assert bad["status"] == "FAIL"
    assert any("archive SHA-256 mismatch" in f for f in bad["failures"])


def test_requires_exactly_one_source(fixture: dict[str, Path]) -> None:
    """Exactly one of archive/source-root must be provided."""
    with pytest.raises(ValueError):
        gate.run_gate(
            artifact_spec=fixture["spec"],
            reference_manifest=fixture["ref"],
            source_commit=str(fixture["commit"]),
            acquisition=gate.AcquisitionSpec(),
        )
