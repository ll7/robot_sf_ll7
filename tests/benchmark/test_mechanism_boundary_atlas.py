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
    BOUNDARY_LABELS,
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
        "unavailable",
    } <= {card.result_state.controlled_state for card in loaded.cards}
    assert all(card.mechanism_boundary.boundary_labels for card in loaded.cards)
    assert all(
        set(card.mechanism_boundary.boundary_labels) <= BOUNDARY_LABELS for card in loaded.cards
    )


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        ("cards", [None]),
        ("cards", ["not a card"]),
        ("source_refs", {"bad": {}}),
    ],
)
def test_malformed_manifest_shapes_fail_closed(
    tmp_path: Path,
    field: str,
    replacement: object,
) -> None:
    payload = _payload()
    if field == "cards":
        payload[field] = replacement
    else:
        payload["cards"][0][field] = replacement
    malformed = tmp_path / f"malformed-{field}.json"
    malformed.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")

    with pytest.raises(MechanismBoundaryAtlasError) as error:
        build_atlas(malformed, repo_root=REPO_ROOT)

    assert error.value.issues
    assert "AttributeError" not in str(error.value)


@pytest.mark.parametrize("field", ["digest_verified", "tracked"])
def test_durable_source_provenance_flags_must_match_inspection(
    tmp_path: Path,
    field: str,
) -> None:
    payload = json.loads(
        (
            REPO_ROOT / "docs/context/evidence/issue_7032_mechanism_boundary_atlas/atlas.v1.json"
        ).read_text(encoding="utf-8")
    )
    payload["cards"][0]["source_refs"][0][field] = False
    mutated = tmp_path / f"mutated-{field}.json"
    mutated.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")

    with pytest.raises(MechanismBoundaryAtlasError) as error:
        load_atlas(mutated, repo_root=REPO_ROOT)

    assert any(
        issue.path.endswith(f"/{field}") and "flag mismatch" in issue.message
        for issue in error.value.issues
    )


def test_blocked_source_provenance_flags_must_remain_false(tmp_path: Path) -> None:
    payload = json.loads(
        (
            REPO_ROOT / "docs/context/evidence/issue_7032_mechanism_boundary_atlas/atlas.v1.json"
        ).read_text(encoding="utf-8")
    )
    payload["cards"][3]["source_refs"][0]["tracked"] = True
    mutated = tmp_path / "mutated-blocked-source.json"
    mutated.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")

    with pytest.raises(MechanismBoundaryAtlasError) as error:
        load_atlas(mutated, repo_root=REPO_ROOT)

    assert any(
        issue.path.endswith("/tracked") and "expected False" in issue.message
        for issue in error.value.issues
    )


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


def test_code_config_identities_are_structured_and_bound() -> None:
    atlas = build_atlas(INPUT, repo_root=REPO_ROOT)

    identities = [identity for card in atlas.cards for identity in card.code_or_config_identity]
    assert identities
    assert {identity.kind for identity in identities} == {"path", "commit", "digest"}
    assert all(identity.label for identity in identities)


def test_code_config_identity_rejects_unstructured_text() -> None:
    payload = _payload()
    payload["cards"][0]["code_or_config_identity"][0] = "free-form identity"

    issues = validate_atlas_payload(payload, repo_root=REPO_ROOT, verify_sources=False)

    assert any(
        issue.path.endswith("/code_or_config_identity/0") and "object" in issue.message
        for issue in issues
    )


def test_supported_result_requires_available_source() -> None:
    payload = _payload()
    card = payload["cards"][0]
    for source in card["source_refs"]:
        source.update(status="blocked", unavailable_reason="source withheld for test")

    issues = validate_atlas_payload(payload, repo_root=REPO_ROOT, verify_sources=False)

    assert any(
        issue.path.endswith("/source_refs")
        and "requires at least one available source" in issue.message
        for issue in issues
    )


def test_code_config_identity_rejects_missing_path() -> None:
    payload = _payload()
    identity = payload["cards"][0]["code_or_config_identity"][0]
    identity["path"] = "configs/does-not-exist.yaml"

    issues = validate_atlas_payload(payload, repo_root=REPO_ROOT)

    assert any(
        issue.path.endswith("/path") and "identity path is missing" in issue.message
        for issue in issues
    )


def test_code_config_identity_rejects_unknown_local_commit() -> None:
    payload = _payload()
    identity = payload["cards"][1]["code_or_config_identity"][2]
    identity["commit"] = "0" * 40

    issues = validate_atlas_payload(payload, repo_root=REPO_ROOT)

    assert any(
        issue.path.endswith("/commit") and "not present in the repository" in issue.message
        for issue in issues
    )


def test_code_config_identity_rejects_digest_drift() -> None:
    payload = _payload()
    identity = payload["cards"][2]["code_or_config_identity"][0]
    identity["sha256"] = "0" * 64

    issues = validate_atlas_payload(payload, repo_root=REPO_ROOT)

    assert any(
        issue.path.endswith("/sha256") and "identity checksum mismatch" in issue.message
        for issue in issues
    )


def test_held_out_missing_outcomes_are_inconclusive() -> None:
    atlas = build_atlas(INPUT, repo_root=REPO_ROOT)

    card = next(card for card in atlas.cards if card.case_id == "held_out_proposal_boundary")
    assert card.result_state.controlled_state == "inconclusive"
    assert card.mechanism_boundary.status == "blocked_by_missing_outcomes"


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


@pytest.mark.parametrize(
    "local_path", ["output/derived.json", "results/derived.json", ".venv/cache.json"]
)
def test_source_paths_reject_local_only_roots(local_path: str) -> None:
    payload = _payload()
    payload["cards"][0]["source_refs"][0]["path"] = local_path

    issues = validate_atlas_payload(payload, repo_root=REPO_ROOT)

    assert any("resolve inside the repository" in issue.message for issue in issues)


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


def test_in_repository_symlink_fails_closed(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    (repo_root / "target.json").write_text("inside\n", encoding="utf-8")
    (repo_root / "source.json").symlink_to(repo_root / "target.json")

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


def test_boundary_labels_use_controlled_vocabulary() -> None:
    payload = _payload()
    payload["cards"][0]["mechanism_boundary"]["boundary_labels"] = ["free_form_label"]

    issues = validate_atlas_payload(payload, repo_root=REPO_ROOT, verify_sources=False)

    assert any("controlled vocabulary" in issue.message for issue in issues)


def test_boundary_labels_reject_non_string_values_without_raising() -> None:
    payload = _payload()
    payload["cards"][0]["mechanism_boundary"]["boundary_labels"] = [{}]

    issues = validate_atlas_payload(payload, repo_root=REPO_ROOT, verify_sources=False)

    assert any("only strings" in issue.message for issue in issues)


def test_boundary_labels_preserve_multi_label_cases() -> None:
    atlas = build_atlas(INPUT, repo_root=REPO_ROOT)

    assert any(len(card.mechanism_boundary.boundary_labels) > 1 for card in atlas.cards)


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
