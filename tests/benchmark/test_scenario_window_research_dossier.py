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
    binding = payload["receipt_binding"]
    assert binding["base_head"] == payload["repository_heads"]["origin_main_at_start"]
    assert binding["code_head"] == payload["repository_heads"]["rw05_rw06_rw07_pr_8622_code_head"]
    assert binding["status"] == "exact_code_head_validated"
