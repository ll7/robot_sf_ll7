"""Validate the terminal preparation dossier's machine-readable boundary."""

from __future__ import annotations

import json
import re
from pathlib import Path

from robot_sf.benchmark.continuation_experiment_plan import (
    BLOCKED_ADMISSION,
    ContinuationExperimentPlan,
    evaluate_admission,
)

_ROOT = Path(__file__).resolve().parents[2]
_DOSSIER = _ROOT / "docs/context/scenario_window_research_dossier_2026-09-08.json"
_CONTEXT_README = _ROOT / "docs/context/README.md"
_EXPECTED_REPOSITORY_HEADS = {
    "origin_main_at_start": "d6a41ba2395b19eb604978836f6c35735df45460",
    "rw03_rw04_pr_8620_head": "4ef0858d33c91291129098c08975525a54fb0121",
    "rw05_rw06_rw07_pr_8622_code_head": "40c33c26068cee38cbf59a5ff9a889736eff86e1",
}
_STALE_CODE_HEAD = "d4d5ed586244edb2e4bcf4cebc0751d119c3641d"


def test_terminal_dossier_has_only_allowed_hypothesis_statuses() -> None:
    """Every listed hypothesis has a terminal status and no campaign claim."""
    payload = json.loads(_DOSSIER.read_text(encoding="utf-8"))
    assert payload["schema_version"] == "scenario_window_research_dossier.v1"
    assert payload["status"] == "preparation_only"
    statuses = {entry["status"] for entry in payload["hypotheses"]}
    assert statuses <= {"supported", "rejected", "not_evaluable", "blocked"}
    assert payload["authority"]["execution_admitted"] is False
    assert payload["submission_boundary"]["manuscript_changed"] is False
    assert len(payload["timelines"]) == 3
    assert payload["unsafe_crop_counterexample"]["result"] == "rejected"


def test_blocked_plan_is_explicit_and_non_executing() -> None:
    """The RW-06/RW-07 scaffolding reports its exact missing authority."""
    plan = ContinuationExperimentPlan("rw06-rw07")
    receipt = evaluate_admission(plan)
    assert receipt.status == BLOCKED_ADMISSION
    assert receipt.blockers
    assert plan.execution_allowed is False


def test_dossier_sources_and_receipts_are_resolvable() -> None:
    """The packet must bind to current source paths and full immutable heads."""
    payload = json.loads(_DOSSIER.read_text(encoding="utf-8"))
    for source in payload["source_documents"]:
        if source.startswith("https://"):
            continue
        assert (_ROOT / source).is_file(), source
    for head in payload["repository_heads"].values():
        assert re.fullmatch(r"[0-9a-f]{40}", head), head
    assert payload["repository_heads"] == _EXPECTED_REPOSITORY_HEADS
    binding = payload["receipt_binding"]
    assert binding["base_head"] == _EXPECTED_REPOSITORY_HEADS["origin_main_at_start"]
    assert binding["base_head_role"] == (
        "historical origin/main at dossier capture; not the child PR base"
    )
    assert binding["code_head"] == _EXPECTED_REPOSITORY_HEADS["rw05_rw06_rw07_pr_8622_code_head"]
    assert binding["status"] == "exact_code_head_validated"
    focused_receipt = next(
        receipt
        for receipt in payload["command_receipts"]
        if "test_relevance_windows.py tests/benchmark/test_typed_snapshot.py" in receipt["command"]
    )
    assert focused_receipt["exit_code"] == 0
    assert "160 passed" in focused_receipt["result"]
    assert "historical origin/main at dossier capture" in focused_receipt["result"]
    assert (
        _EXPECTED_REPOSITORY_HEADS["rw05_rw06_rw07_pr_8622_code_head"] in focused_receipt["result"]
    )
    serialized_dossier = json.dumps(payload, sort_keys=True)
    assert _STALE_CODE_HEAD not in serialized_dossier
    serialized_receipts = json.dumps(payload["command_receipts"], sort_keys=True)
    assert "ce31aa2" not in serialized_receipts
    assert "144 passed" not in serialized_receipts


def test_dossier_is_linked_from_context_readme() -> None:
    """The full context discoverability surface must link the preparation dossier."""
    assert "scenario_window_research_dossier_2026-09-08.md" in _CONTEXT_README.read_text(
        encoding="utf-8"
    )
