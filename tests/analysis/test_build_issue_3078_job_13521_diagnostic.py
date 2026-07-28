"""Tests for the deterministic issue #3078 job-13521 diagnostic builder."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "scripts/analysis/build_issue_3078_job_13521_diagnostic.py"
SPEC = importlib.util.spec_from_file_location("issue_3078_job_13521_diagnostic", SCRIPT)
assert SPEC and SPEC.loader
BUILDER = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = BUILDER
SPEC.loader.exec_module(BUILDER)


def test_builder_is_byte_deterministic(tmp_path: Path) -> None:
    """Two builds from tracked compact inputs produce identical bytes."""
    first = tmp_path / "first"
    second = tmp_path / "second"

    first_paths = BUILDER.build_outputs(BUILDER.DEFAULT_BUNDLE, first)
    second_paths = BUILDER.build_outputs(BUILDER.DEFAULT_BUNDLE, second)

    assert [path.name for path in first_paths] == list(BUILDER.OUTPUT_NAMES)
    assert [path.read_bytes() for path in first_paths] == [
        path.read_bytes() for path in second_paths
    ]


def test_builder_preserves_diagnostic_claim_boundary(tmp_path: Path) -> None:
    """Generated evidence stays fail-closed and preserves adapter labeling."""
    output_dir = tmp_path / "diagnostic"
    BUILDER.build_outputs(BUILDER.DEFAULT_BUNDLE, output_dir)

    payload = json.loads(
        (output_dir / "seed_rank_stability_diagnostic.json").read_text(encoding="utf-8")
    )
    assert payload["headline_rank_stability_contract"]["label"] == "not_identifiable"
    assert payload["headline_rank_stability_contract"]["promotion_allowed"] is False
    assert payload["heldout_transfer_delta_classification"]["label"] == "not_identifiable"
    assert payload["heldout_transfer_delta_classification"]["claim_eligible"] is False
    assert {row["planner"]: row["row_status"] for row in payload["planner_rank_stability"]} == {
        "goal": "native",
        "social_force": "adapter",
        "orca": "adapter",
    }


def test_checked_in_outputs_match_fresh_build() -> None:
    """The committed JSON and PNG bytes match the documented builder."""
    assert BUILDER.check_outputs(BUILDER.DEFAULT_BUNDLE) == []
