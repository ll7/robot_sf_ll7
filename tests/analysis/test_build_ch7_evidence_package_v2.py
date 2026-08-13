"""Contract tests for the Chapter 7 release-cell evidence package v2."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import pytest

from scripts.analysis import build_ch7_evidence_package_v2 as builder

SOURCE_PACKAGE = (
    Path(__file__).parents[2] / "docs/context/evidence/issue_6792_ch7_evidence_package_v1"
)
CONFIG_PATH = Path(__file__).parents[2] / "configs/analysis/ch7_evidence_package.v2.yaml"


def _read_atlas(output: Path) -> dict[str, object]:
    return json.loads((output / "publication/reduced_atlas.json").read_text(encoding="utf-8"))


def test_v2_projection_is_deterministic_and_contains_both_inversions(tmp_path: Path) -> None:
    output = tmp_path / "package"
    manifest = builder.build_ch7_evidence_package_v2(
        source_package=SOURCE_PACKAGE,
        output=output,
        config_path=CONFIG_PATH,
        check_determinism=True,
    )
    atlas = _read_atlas(output)
    cells = atlas["cells"]
    assert isinstance(cells, list)
    assert manifest["status"] == "blocked_pending_domain_approval"
    assert manifest["admission_status"] == "not_admitted"
    assert manifest["source"]["v1_package_sha256sums"] == builder.SOURCE_PACKAGE_SHA256SUMS
    assert len(cells) == 28
    assert sum(cell["panel"] == "cross_topology" for cell in cells) == 10
    assert sum(cell["panel"] == "cross_mechanism" for cell in cells) == 4
    assert sum(cell["panel"] == "narrow_doorway_terminal" for cell in cells) == 14
    assert atlas["projections"] == builder._projection_metadata()
    assert atlas["roles"] == [
        "cross_topology_inversion",
        "cross_mechanism_inversion",
        "feasibility_criticism",
    ]

    topology_planners = {cell["planner_key"] for cell in cells if cell["panel"] == "cross_topology"}
    assert topology_planners == set(builder.TOPOLOGY_PLANNERS)
    mechanism_scenarios = {
        cell["scenario_id"] for cell in cells if cell["panel"] == "cross_mechanism"
    }
    assert mechanism_scenarios == set(builder.MECHANISM_SCENARIOS)
    doorway_cells = [cell for cell in cells if cell["panel"] == "narrow_doorway_terminal"]
    assert all(cell["terminal_counts_status"] == "available" for cell in doorway_cells)
    assert next(cell for cell in doorway_cells if cell["planner_key"] == "orca")[
        "terminal_counts"
    ] == {"timeout": 30}
    for cell in cells:
        assert not set(cell).intersection(builder.EXCLUDED_METRICS)
        assert cell["source_provenance"] == {
            "package_sha256sums_sha256": builder.SOURCE_PACKAGE_SHA256SUMS,
            "member": builder.SOURCE_AUDIT_MEMBER,
            "member_sha256": builder.SOURCE_AUDIT_SHA256,
            "source_row_sha256": cell["source_provenance"]["source_row_sha256"],
        }

    csv_rows = list(
        csv.DictReader((output / "publication/reduced_atlas.csv").open(encoding="utf-8"))
    )
    assert len(csv_rows) == 28
    assert not set(csv_rows[0]).intersection(builder.EXCLUDED_METRICS)
    assert len((output / "SHA256SUMS").read_text(encoding="ascii").splitlines()) == 8


def test_v2_publishes_terminal_mapping_and_keeps_receipt_external(tmp_path: Path) -> None:
    output = tmp_path / "package"
    manifest = builder.build_ch7_evidence_package_v2(
        source_package=SOURCE_PACKAGE,
        output=output,
        config_path=CONFIG_PATH,
    )
    atlas = _read_atlas(output)
    mapping = manifest["terminal_label_normalization"]
    assert mapping == builder.v1.terminal_label_normalization()
    assert atlas["terminal_label_normalization"] == mapping
    assert "terminated" in mapping["normalized_timeout_reasons"]
    assert manifest["admission"] == {
        "status": "not_admitted",
        "receipt_required": True,
        "receipt_schema": "ch7-evidence-admission.v2",
        "reason": "v2 domain approval and the external admission receipt remain pending",
    }
    source_verification = json.loads(
        (output / "review/source_verification.json").read_text(encoding="utf-8")
    )
    assert source_verification["admission_receipt"]["schema"] == "ch7-evidence-admission.v2"
    assert not (output / "admission/receipt.json").exists()


def test_v2_selection_fails_closed_when_a_requested_cell_is_missing() -> None:
    with (SOURCE_PACKAGE / builder.SOURCE_AUDIT_MEMBER).open(
        newline="", encoding="utf-8"
    ) as stream:
        rows = list(csv.DictReader(stream))
    rows = [
        row
        for row in rows
        if (row["scenario_id"], row["planner_key"])
        != (builder.MECHANISM_SCENARIOS[1], builder.MECHANISM_PLANNERS[1])
    ]
    source = {
        "package_sha256sums_sha256": builder.SOURCE_PACKAGE_SHA256SUMS,
        "audit_member": builder.SOURCE_AUDIT_MEMBER,
        "audit_member_sha256": builder.SOURCE_AUDIT_SHA256,
        "reduced_atlas_member": builder.SOURCE_REDUCED_ATLAS_MEMBER,
        "reduced_atlas_member_sha256": builder.SOURCE_REDUCED_ATLAS_SHA256,
    }
    with pytest.raises(builder.Ch7EvidencePackageV2Error, match="missing the v2 cell"):
        builder.select_v2_cells(rows, source=source, terminal_counts={})


def test_v2_source_digest_mismatch_stops_before_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(builder, "SOURCE_PACKAGE_SHA256SUMS", "0" * 64)
    with pytest.raises(builder.Ch7EvidencePackageV2Error, match="source binding|SHA256SUMS digest"):
        builder.build_ch7_evidence_package_v2(
            source_package=SOURCE_PACKAGE,
            output=tmp_path / "package",
        )
    assert not (tmp_path / "package").exists()


def test_v2_source_binding_matches_immutable_member_bytes() -> None:
    assert (
        hashlib.sha256((SOURCE_PACKAGE / builder.SOURCE_AUDIT_MEMBER).read_bytes()).hexdigest()
        == builder.SOURCE_AUDIT_SHA256
    )
