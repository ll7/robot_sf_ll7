"""The retained positive-control packet must satisfy the existing registry."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.tools.lint_evidence_registry import _artifact_findings

ROOT = Path(__file__).resolve().parents[2]
EVIDENCE = Path("docs/context/evidence/hardcase_positive_control")
CASE_ID = "case-ee2bab20aef65d75986bc4a1e08e9a04a2d65ac6764435aef03a5f71b0d63490"


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
