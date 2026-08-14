"""Tests for the issue #7086 representative-seed selector."""

from __future__ import annotations

import copy

import pytest

from robot_sf.benchmark.trace_dossier_selection import (
    TRACE_DOSSIER_SELECTOR_SCHEMA_VERSION,
    SelectionManifest,
    TraceDossierSelectionError,
    select_representative,
)


def _candidate(
    *,
    cell_id: str = "cell-a",
    verdict: str = "success",
    label_strength: float = 1.0,
    primary_order: float = 0.0,
    seed_id: str = "seed-1",
) -> dict[str, object]:
    """Build a compact mapping candidate for one test."""

    return {
        "cell_id": cell_id,
        "verdict": verdict,
        "label_strength": label_strength,
        "primary_order": primary_order,
        "seed_id": seed_id,
    }


def test_majority_verdict_pool_wins() -> None:
    """Candidates with the unique majority verdict are the only eligible pool."""

    result = select_representative(
        [
            _candidate(verdict="collision", seed_id="seed-c"),
            _candidate(seed_id="seed-a", primary_order=10.0),
            _candidate(seed_id="seed-b", primary_order=20.0),
        ]
    )

    assert result.selected_verdict == "success"
    assert result.majority_count == 2
    assert result.selection_reason == "majority_verdict"


def test_weakest_label_strength_breaks_majority_pool() -> None:
    """The smallest explicit label-strength value is selected next."""

    result = select_representative(
        [
            _candidate(label_strength=2.0, seed_id="seed-strong"),
            _candidate(label_strength=0.5, seed_id="seed-weak"),
            _candidate(label_strength=1.0, seed_id="seed-mid"),
        ]
    )

    assert result.selected_seed_id == "seed-weak"
    assert result.selection_reason == "weakest_label"


def test_median_primary_order_is_used_after_label_tie() -> None:
    """The candidate closest to the median order parameter is selected."""

    result = select_representative(
        [
            _candidate(primary_order=0.0, seed_id="seed-0"),
            _candidate(primary_order=50.0, seed_id="seed-50"),
            _candidate(primary_order=100.0, seed_id="seed-100"),
        ]
    )

    assert result.selected_seed_id == "seed-50"
    assert result.selection_reason == "median_primary_order"


def test_seed_identity_breaks_exact_median_tie() -> None:
    """Equal numeric candidates resolve by stable identity, never input order."""

    result = select_representative(
        [
            _candidate(primary_order=10.0, seed_id="seed-z"),
            _candidate(primary_order=10.0, seed_id="seed-a"),
        ]
    )

    assert result.selected_seed_id == "seed-a"
    assert result.selection_reason == "seed_identity"


def test_tied_verdicts_choose_the_unique_weaker_label() -> None:
    """Equal verdict counts use explicit weakest-label semantics."""

    result = select_representative(
        [
            _candidate(verdict="clearly_present", label_strength=2.0, seed_id="seed-strong"),
            _candidate(verdict="absent_or_negligible", label_strength=0.5, seed_id="seed-weak"),
        ]
    )

    assert result.selected_verdict == "absent_or_negligible"
    assert result.selected_seed_id == "seed-weak"
    assert result.selection_reason == "weaker_verdict"


def test_single_candidate_is_valid() -> None:
    """A one-row cell remains explicit and deterministic."""

    result = select_representative([_candidate()])

    assert result.selection_reason == "single_candidate"
    assert result.selected_seed_id == "seed-1"


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("cell_id", ""),
        ("verdict", " "),
        ("seed_id", ""),
        ("label_strength", float("nan")),
        ("primary_order", float("inf")),
        ("label_strength", True),
    ],
)
def test_invalid_required_fields_fail_closed(field: str, value: object) -> None:
    """Malformed or non-finite fields cannot enter a dossier selection."""

    candidate = _candidate()
    candidate[field] = value

    with pytest.raises(TraceDossierSelectionError, match=field):
        select_representative([candidate])


def test_missing_required_field_fails_closed() -> None:
    """A partial row is not silently completed from defaults."""

    candidate = _candidate()
    del candidate["primary_order"]

    with pytest.raises(TraceDossierSelectionError, match="primary_order"):
        select_representative([candidate])


def test_mixed_cells_fail_closed() -> None:
    """A selector cannot compare candidates from different campaign cells."""

    with pytest.raises(TraceDossierSelectionError, match="mixed cell_id"):
        select_representative([_candidate(), _candidate(cell_id="cell-b", seed_id="seed-2")])


def test_duplicate_seed_identity_fails_closed() -> None:
    """Duplicate identities would make provenance binding ambiguous."""

    with pytest.raises(TraceDossierSelectionError, match="seed_id values must be unique"):
        select_representative([_candidate(), _candidate(seed_id="seed-1", primary_order=2.0)])


def test_tied_verdicts_fail_closed() -> None:
    """Equal-strength tied labels cannot be resolved lexically."""

    with pytest.raises(TraceDossierSelectionError, match="unique weaker label"):
        select_representative(
            [
                _candidate(verdict="collision", label_strength=1.0, seed_id="seed-c"),
                _candidate(verdict="success", label_strength=1.0, seed_id="seed-s"),
            ]
        )


def test_empty_candidates_fail_closed() -> None:
    """Empty cells cannot yield a representative."""

    with pytest.raises(TraceDossierSelectionError, match="non-empty"):
        select_representative([])


def test_mapping_input_is_not_mutated() -> None:
    """Selection copies validated values and leaves caller data untouched."""

    candidates = [_candidate(), _candidate(seed_id="seed-2", primary_order=2.0)]
    original = copy.deepcopy(candidates)

    select_representative(candidates)

    assert candidates == original


def test_selection_manifest_is_serializable_and_frozen() -> None:
    """The manifest is stable JSON-shaped metadata, not a mutable result record."""

    result = select_representative([_candidate()])

    assert isinstance(result, SelectionManifest)
    assert result.to_dict()["schema_version"] == TRACE_DOSSIER_SELECTOR_SCHEMA_VERSION
    assert "generated_at" not in result.to_dict()
    with pytest.raises(AttributeError):
        result.cell_id = "other"  # type: ignore[misc]
