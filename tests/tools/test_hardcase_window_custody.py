"""The retained positive-control packet must satisfy the existing registry."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.tools.lint_evidence_registry import _artifact_findings

ROOT = Path(__file__).resolve().parents[2]
EVIDENCE = Path("docs/context/evidence/hardcase_positive_control")
CASE_ID = "case-082cc08f6d75b6f7721974c7e52ef9cc8d4609e0754ca396e504dd4ebb14e524"


@pytest.mark.parametrize(
    "relative",
    [
        EVIDENCE / "admission/window_receipt.json",
        EVIDENCE / "admission/corpus/cases" / (CASE_ID + ".json"),
        EVIDENCE / "followup/packet/window_replay_1.jsonl.provenance.json",
        EVIDENCE / "followup/packet/window_replay_2.jsonl.provenance.json",
    ],
)
def test_real_window_receipts_have_registry_resolvable_custody(relative: Path) -> None:
    """Losing a config locator or raw-row location must fail on the actual packet."""
    payload = json.loads((ROOT / relative).read_text())
    assert _artifact_findings(ROOT, relative, payload) == []
