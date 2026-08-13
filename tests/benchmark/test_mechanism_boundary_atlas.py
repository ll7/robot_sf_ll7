"""Tests for the issue #7032 mechanism-boundary atlas contract."""

# evidence-writer-exempt: this module writes only tmp_path fixtures for malformed atlas inputs
# and symlink-escape probes; checked-in #7032 evidence artifacts are emitted separately.

from __future__ import annotations

import json
import os
import subprocess
import sys
from copy import deepcopy
from pathlib import Path

import pytest

from robot_sf.benchmark.mechanism_boundary_atlas import (
    CONTROLLED_STATES,
    MechanismBoundaryAtlasError,
    build_atlas,
    load_atlas,
    validate_atlas_payload,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
INPUT = REPO_ROOT / "docs/context/evidence/issue_7032_mechanism_boundary_atlas/atlas_input.v1.json"
BUILDER = REPO_ROOT / "scripts/analysis/build_negative_result_mechanism_atlas.py"


def _payload() -> dict:
    return json.loads(INPUT.read_text(encoding="utf-8"))


def test_builds_typed_schema_validated_six_card_atlas(tmp_path: Path) -> None:
    output = tmp_path / "atlas.json"

    atlas = build_atlas(INPUT, repo_root=REPO_ROOT, output_path=output)
    loaded = load_atlas(output, repo_root=REPO_ROOT)

    assert atlas.schema_version == "mechanism_boundary_atlas.v1"
    assert loaded.issue == 7032
    assert len(loaded.cards) == 6
    assert {card.result_state.controlled_state for card in loaded.cards} <= CONTROLLED_STATES
    assert {
        "supported_negative",
        "inconclusive",
        "invalid_evidence_contract",
        "unavailable",
    } <= {card.result_state.controlled_state for card in loaded.cards}


def test_cards_preserve_the_full_issue_interpretation_contract() -> None:
    atlas = build_atlas(INPUT, repo_root=REPO_ROOT)

    for card in atlas.cards:
        assert card.question_and_hypothesis
        assert card.code_or_config_identity
        assert card.mechanism_activation_evidence
        assert card.observed_result
        assert card.hypotheses_contradicted
        assert card.hypotheses_still_viable
        assert card.dissertation_admission_status in {
            "blocked",
            "not_admitted",
            "pending_independent_review",
        }


def test_source_linkage_digests_are_verified() -> None:
    atlas = build_atlas(INPUT, repo_root=REPO_ROOT)

    available_sources = [
        source
        for card in atlas.cards
        for source in card.source_refs
        if source.status == "available"
    ]

    assert available_sources
    assert all(source.digest_verified is True for source in available_sources)
    assert all(source.tracked is True for source in available_sources)


def test_digest_mismatch_fails_closed() -> None:
    payload = _payload()
    payload["cards"][0]["source_refs"][0]["sha256"] = "0" * 64

    issues = validate_atlas_payload(payload, repo_root=REPO_ROOT)

    assert any(
        issue.path.endswith("/sha256") and "checksum mismatch" in issue.message for issue in issues
    )


@pytest.mark.parametrize("bad_path", ["/tmp/source.json", "../source.json", "docs/../source.json"])
def test_source_paths_reject_absolute_and_traversal(bad_path: str) -> None:
    payload = _payload()
    payload["cards"][0]["source_refs"][0]["path"] = bad_path

    issues = validate_atlas_payload(payload, repo_root=REPO_ROOT)

    assert any("repository-root relative" in issue.message for issue in issues)


def test_missing_available_source_fails_closed() -> None:
    payload = _payload()
    payload["cards"][0]["source_refs"][0]["path"] = "docs/context/evidence/missing.json"

    issues = validate_atlas_payload(payload, repo_root=REPO_ROOT)

    assert any(
        issue.path.endswith("/path") and "available source is missing" in issue.message
        for issue in issues
    )


def test_symlink_escape_fails_closed(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    outside = tmp_path / "outside.json"
    outside.write_text("outside\n", encoding="utf-8")
    (repo_root / "source.json").symlink_to(outside)

    payload = _payload()
    payload["cards"][0]["source_refs"][0].update(
        path="source.json",
        sha256="0" * 64,
    )

    issues = validate_atlas_payload(payload, repo_root=repo_root)

    assert any(
        issue.path.endswith("/path") and "resolve inside the repository" in issue.message
        for issue in issues
    )


def test_result_state_and_mechanism_boundary_are_separate_dimensions() -> None:
    payload = _payload()
    payload["cards"][0]["mechanism_boundary"]["status"] = payload["cards"][0]["result_state"][
        "controlled_state"
    ]

    issues = validate_atlas_payload(payload, repo_root=REPO_ROOT, verify_sources=False)

    assert any("must not be reused as mechanism status" in issue.message for issue in issues)


def test_replacement_result_state_vocabulary_is_rejected() -> None:
    payload = _payload()
    payload["cards"][0]["result_state"]["controlled_state"] = "diagnostic_only"

    issues = validate_atlas_payload(payload, repo_root=REPO_ROOT, verify_sources=False)

    assert any("#7032 exact vocabulary" in issue.message for issue in issues)


def test_requires_six_case_minimum() -> None:
    payload = _payload()
    payload["cards"] = payload["cards"][:5]

    issues = validate_atlas_payload(payload, repo_root=REPO_ROOT, verify_sources=False)

    assert any(issue.path == "/cards" and "at least six" in issue.message for issue in issues)


def test_unavailable_or_blocked_sources_remain_explicit() -> None:
    atlas = build_atlas(INPUT, repo_root=REPO_ROOT)

    unavailable = [
        source
        for card in atlas.cards
        for source in card.source_refs
        if source.status in {"unavailable", "blocked"}
    ]

    assert unavailable
    assert all(source.unavailable_reason for source in unavailable)
    assert all(source.digest_verified is False for source in unavailable)


def test_deterministic_output(tmp_path: Path) -> None:
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"

    build_atlas(INPUT, repo_root=REPO_ROOT, output_path=first)
    build_atlas(INPUT, repo_root=REPO_ROOT, output_path=second)

    assert first.read_bytes() == second.read_bytes()


def test_no_claim_promotion_in_bounded_claims() -> None:
    payload = _payload()
    payload["cards"][0]["claim_boundary"]["bounded_claims"].append("This is benchmark evidence.")

    issues = validate_atlas_payload(payload, repo_root=REPO_ROOT, verify_sources=False)

    assert any("promotion pattern" in issue.message for issue in issues)


def test_no_claim_promotion_in_top_level_claim_boundary() -> None:
    payload = _payload()
    payload["claim_boundary"] = "This is benchmark evidence."

    issues = validate_atlas_payload(payload, repo_root=REPO_ROOT, verify_sources=False)

    assert any(
        issue.path == "/claim_boundary" and "promotion pattern" in issue.message for issue in issues
    )


def test_top_level_claim_boundary_allows_explicit_negative_wording() -> None:
    payload = _payload()
    payload["claim_boundary"] = "Diagnostic planning atlas only; not benchmark evidence."

    issues = validate_atlas_payload(payload, repo_root=REPO_ROOT, verify_sources=False)

    assert not any(issue.path == "/claim_boundary" for issue in issues)


def test_forbidden_wording_can_name_disallowed_phrases() -> None:
    payload = _payload()
    payload["cards"][0]["claim_boundary"]["forbidden_wording"].append("paper-grade evidence")

    issues = validate_atlas_payload(payload, repo_root=REPO_ROOT)

    assert not issues


def test_cli_error_path_does_not_write_output_for_invalid_digest(tmp_path: Path) -> None:
    payload = _payload()
    payload["cards"][0]["source_refs"][0]["sha256"] = "0" * 64
    bad_input = tmp_path / "bad.json"
    bad_input.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    output = tmp_path / "out.json"

    with pytest.raises(MechanismBoundaryAtlasError):
        build_atlas(bad_input, repo_root=REPO_ROOT, output_path=output)

    assert not output.exists()


def test_builder_help_forwards_arguments_without_writing(tmp_path: Path) -> None:
    result = subprocess.run(
        [sys.executable, str(BUILDER), "--help"],
        cwd=tmp_path,
        env={**os.environ, "PYTHONPATH": str(REPO_ROOT)},
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0
    assert "--input" in result.stdout
    assert not (
        tmp_path / "docs/context/evidence/issue_7032_mechanism_boundary_atlas/atlas.v1.json"
    ).exists()


def test_loader_returns_independent_nested_dataclasses(tmp_path: Path) -> None:
    output = tmp_path / "atlas.json"
    build_atlas(INPUT, repo_root=REPO_ROOT, output_path=output)

    atlas = load_atlas(output, repo_root=REPO_ROOT)
    card = deepcopy(atlas.cards[0])

    assert card.result_state.controlled_state != card.mechanism_boundary.status
    assert card.result_state.state_reason != card.mechanism_boundary.evidence_boundary
