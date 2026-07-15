"""Tests for the job-13274 rank-stability claim-packet validator."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

from scripts.validation.check_issue_5247_rank_stability_claim_packet import check_bundle

REPO_ROOT = Path(__file__).resolve().parents[2]
BUNDLE = REPO_ROOT / "docs/context/evidence/issue_5247_job_13274_rank_stability"


def test_preserved_job_13274_bundle_satisfies_claim_contract() -> None:
    """The committed evidence bundle passes its cross-file integrity checks."""
    assert check_bundle(BUNDLE) == []


def test_claim_packet_fails_when_decision_count_drifts(tmp_path: Path) -> None:
    """A decision packet count drift is not silently accepted."""
    bundle = tmp_path / "bundle"
    shutil.copytree(BUNDLE, bundle)
    result_path = bundle / "result.json"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    result["decision_packet"]["adjacent_overlap_count"] = 234
    result_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")

    problems = check_bundle(bundle)

    assert any("decision_packet.adjacent_overlap_count" in problem for problem in problems)


def test_claim_packet_fails_when_snqi_boundary_is_removed(tmp_path: Path) -> None:
    """The claim card must retain the explicit invalid-SNQI boundary."""
    bundle = tmp_path / "bundle"
    shutil.copytree(BUNDLE, bundle)
    claim_path = bundle / "claim_decision.md"
    claim_path.write_text(
        claim_path.read_text(encoding="utf-8").replace(
            "No SNQI rank or SNQI-based adjacent-order claim is promoted.\n", ""
        ),
        encoding="utf-8",
    )

    problems = check_bundle(bundle)

    assert any("No SNQI rank" in problem for problem in problems)
