"""Tests for the issue #4195 h600 F-C4(ii) interpretation-gate checker.

Covers the committed evidence bundle (happy path) plus fail-closed cases on a
synthetic fixture: unlisted evidence file, digest mismatch, dropped boundary
section, and a scenario-matrix-hash disagreement.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = ROOT / "scripts" / "validation" / "check_issue_4195_f_c4_ii_gate.py"

_SPEC = importlib.util.spec_from_file_location("check_issue_4195_f_c4_ii_gate", SCRIPT_PATH)
assert _SPEC is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
assert _SPEC.loader is not None
sys.modules[_SPEC.name] = _MODULE
_SPEC.loader.exec_module(_MODULE)

build_report = _MODULE.build_report
DEFAULT_EVIDENCE_DIR = _MODULE.DEFAULT_EVIDENCE_DIR

SHARED_HASH = "c10df617a87c"

GATE_NOTE = f"""# F-C4(ii) Interpretation Gate

F-C4(ii) reading over matrix hash {SHARED_HASH}.
See hybrid_roster_h600_transfer_packet.md.

### SUPPORTED
- hybrids separate from prediction arms.

### DIAGNOSTIC-ONLY
- hybrids vs ORCA CI-overlapping.

### NOT SUPPORTED
- no horizon-only causality.

## Integration report
- author_signoff: RECORDED
"""

HYBRID_PACKET = f"""# Hybrid roster packet
Scenario matrix hash {SHARED_HASH}; job 13282.
"""

SOURCE_MANIFEST = {
    "schema_version": "issue_4195_h600_aggregation.v1.source_manifest",
    "runs": [
        {"job_id": "13268", "campaign": {"scenario_matrix_hash": SHARED_HASH}},
        {"job_id": "13273", "campaign": {"scenario_matrix_hash": SHARED_HASH}},
    ],
}


def _write_sha256sums(evidence_dir: Path, prefix: str = "docs/context/evidence/x") -> None:
    """Write a SHA256SUMS ledger covering every .md/.json/.csv in ``evidence_dir``."""
    lines = []
    for path in sorted(evidence_dir.iterdir()):
        if path.is_file() and path.suffix in {".md", ".json", ".csv"}:
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            lines.append(f"{digest}  {prefix}/{path.name}")
    (evidence_dir / "SHA256SUMS").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _make_valid_fixture(evidence_dir: Path) -> None:
    """Populate ``evidence_dir`` with a minimal internally-consistent bundle."""
    (evidence_dir / "f_c4_ii_interpretation_gate.md").write_text(GATE_NOTE, encoding="utf-8")
    (evidence_dir / "hybrid_roster_h600_transfer_packet.md").write_text(
        HYBRID_PACKET, encoding="utf-8"
    )
    (evidence_dir / "source_manifest.json").write_text(
        json.dumps(SOURCE_MANIFEST), encoding="utf-8"
    )
    _write_sha256sums(evidence_dir)


def test_committed_bundle_passes() -> None:
    """The real committed evidence bundle satisfies the fail-closed gate."""
    report = build_report(DEFAULT_EVIDENCE_DIR)
    assert report["status"] == "pass", report["violations"]
    assert report["facts"]["shared_matrix_hash"] == SHARED_HASH


def test_synthetic_fixture_passes(tmp_path: Path) -> None:
    """A minimal internally-consistent fixture passes."""
    _make_valid_fixture(tmp_path)
    report = build_report(tmp_path)
    assert report["status"] == "pass", report["violations"]


def test_sha256sums_comment_marker_is_ignored() -> None:
    """A review marker comment does not invalidate a standard checksum ledger."""
    parsed = _MODULE._parse_sha256sums(
        "# AI-GENERATED NEEDS-REVIEW\n"
        "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef  "
        "docs/context/evidence/x/sample.json\n"
    )
    assert parsed == {
        "sample.json": "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"
    }


def test_unlisted_evidence_file_fails_closed(tmp_path: Path) -> None:
    """An evidence file absent from SHA256SUMS is a coverage violation."""
    _make_valid_fixture(tmp_path)
    (tmp_path / "sneaked_in.json").write_text("{}", encoding="utf-8")  # after ledger written
    report = build_report(tmp_path)
    assert report["status"] == "fail"
    assert any(v["field"] == "sha256sums_coverage" for v in report["violations"])


def test_digest_mismatch_fails_closed(tmp_path: Path) -> None:
    """Editing a file after the ledger is written trips the digest check."""
    _make_valid_fixture(tmp_path)
    (tmp_path / "hybrid_roster_h600_transfer_packet.md").write_text(
        HYBRID_PACKET + "\ntampered\n", encoding="utf-8"
    )
    report = build_report(tmp_path)
    assert report["status"] == "fail"
    assert any(v["field"] == "sha256sums_digest" for v in report["violations"])


def test_missing_boundary_section_fails_closed(tmp_path: Path) -> None:
    """Dropping a required boundary heading fails the gate."""
    _make_valid_fixture(tmp_path)
    broken = GATE_NOTE.replace("### NOT SUPPORTED", "### OTHER")
    (tmp_path / "f_c4_ii_interpretation_gate.md").write_text(broken, encoding="utf-8")
    _write_sha256sums(tmp_path)  # refresh so this is a marker failure, not a digest one
    report = build_report(tmp_path)
    assert report["status"] == "fail"
    assert any(v["field"] == "gate_note_marker" for v in report["violations"])


def test_matrix_hash_disagreement_fails_closed(tmp_path: Path) -> None:
    """A hybrid packet citing a different matrix hash blocks the gate."""
    _make_valid_fixture(tmp_path)
    (tmp_path / "hybrid_roster_h600_transfer_packet.md").write_text(
        HYBRID_PACKET.replace(SHARED_HASH, "deadbeef1234"), encoding="utf-8"
    )
    _write_sha256sums(tmp_path)
    report = build_report(tmp_path)
    assert report["status"] == "fail"
    assert any(v["field"] == "matrix_hash" for v in report["violations"])


def test_missing_evidence_dir_fails_closed(tmp_path: Path) -> None:
    """A non-existent evidence directory fails closed rather than erroring."""
    report = build_report(tmp_path / "does_not_exist")
    assert report["status"] == "fail"
    assert any(v["field"] == "evidence_dir" for v in report["violations"])


def test_unrecognized_signoff_value_fails_closed(tmp_path: Path) -> None:
    """A sign-off status outside the recognized set fails the gate."""
    _make_valid_fixture(tmp_path)
    bogus = GATE_NOTE.replace("author_signoff: RECORDED", "author_signoff: MAYBE")
    (tmp_path / "f_c4_ii_interpretation_gate.md").write_text(bogus, encoding="utf-8")
    _write_sha256sums(tmp_path)
    report = build_report(tmp_path)
    assert report["status"] == "fail"
    assert any(v["field"] == "gate_note_signoff" for v in report["violations"])
