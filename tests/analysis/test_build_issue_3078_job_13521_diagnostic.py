"""Tests for the deterministic issue #3078 job-13521 diagnostic builder."""

from __future__ import annotations

import json
import shutil
from typing import TYPE_CHECKING

import matplotlib as mpl
import pytest

if TYPE_CHECKING:
    from pathlib import Path
    from types import ModuleType


def _copy_bundle(tmp_path: Path, diagnostic_builder: ModuleType) -> Path:
    bundle = tmp_path / "bundle"
    shutil.copytree(diagnostic_builder.DEFAULT_BUNDLE, bundle)
    return bundle


def test_builder_is_byte_deterministic(tmp_path: Path, diagnostic_builder: ModuleType) -> None:
    """Two builds from tracked compact inputs produce identical bytes."""
    first = tmp_path / "first"
    second = tmp_path / "second"

    first_paths = diagnostic_builder.build_outputs(diagnostic_builder.DEFAULT_BUNDLE, first)
    second_paths = diagnostic_builder.build_outputs(diagnostic_builder.DEFAULT_BUNDLE, second)

    assert [path.name for path in first_paths] == list(diagnostic_builder.OUTPUT_NAMES)
    assert [path.name for path in second_paths] == list(diagnostic_builder.OUTPUT_NAMES)
    assert [(path.name, path.read_bytes()) for path in first_paths] == [
        (path.name, path.read_bytes()) for path in second_paths
    ]


def test_builder_preserves_diagnostic_claim_boundary(
    tmp_path: Path, diagnostic_builder: ModuleType
) -> None:
    """Generated evidence stays fail-closed and preserves adapter labeling."""
    output_dir = tmp_path / "diagnostic"
    diagnostic_builder.build_outputs(diagnostic_builder.DEFAULT_BUNDLE, output_dir)

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


def test_checked_in_outputs_match_fresh_build(diagnostic_builder: ModuleType) -> None:
    """The committed JSON and PNG bytes match the documented builder."""
    assert diagnostic_builder.check_outputs(diagnostic_builder.DEFAULT_BUNDLE) == []


def test_builder_rejects_duplicate_transfer_planners(
    tmp_path: Path,
    diagnostic_builder: ModuleType,
) -> None:
    """Duplicate planner identities cannot be silently overwritten."""
    bundle = _copy_bundle(tmp_path, diagnostic_builder)
    transfer_path = bundle / "transfer_delta.csv"
    lines = transfer_path.read_text(encoding="utf-8").splitlines()
    transfer_path.write_text("\n".join([*lines, lines[1]]) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="planner identities must be unique"):
        diagnostic_builder._build_payload(bundle)


def test_builder_records_non_default_bundle_source(
    tmp_path: Path,
    diagnostic_builder: ModuleType,
) -> None:
    """A bundle override is recorded instead of the default source path."""
    bundle = _copy_bundle(tmp_path, diagnostic_builder)

    payload = diagnostic_builder._build_payload(bundle)

    assert payload["generated_for"] == str(bundle.resolve())


def test_builder_ignores_ambient_matplotlib_style(
    monkeypatch: pytest.MonkeyPatch, diagnostic_builder: ModuleType
) -> None:
    """Figure bytes do not depend on plotting state leaked by another caller."""
    monkeypatch.setitem(mpl.rcParams, "font.size", 17)
    monkeypatch.setitem(mpl.rcParams, "axes.titlesize", 19)

    assert diagnostic_builder.check_outputs(diagnostic_builder.DEFAULT_BUNDLE) == []


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("fallback_degraded_rows", 1, "unexpected fallback/degraded rows"),
        ("synthetic_fixture_used", True, "synthetic fixture usage must remain false"),
    ],
)
def test_builder_rejects_invalid_acceptance_provenance(
    tmp_path: Path,
    diagnostic_builder: ModuleType,
    field: str,
    value: object,
    message: str,
) -> None:
    """Fallback/degraded or synthetic evidence cannot enter the diagnostic."""
    bundle = _copy_bundle(tmp_path, diagnostic_builder)
    acceptance_path = bundle / "row_acceptance.json"
    acceptance = json.loads(acceptance_path.read_text(encoding="utf-8"))
    acceptance[field] = value
    acceptance_path.write_text(json.dumps(acceptance), encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        diagnostic_builder._build_payload(bundle)
