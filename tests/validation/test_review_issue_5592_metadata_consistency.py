"""Review-gate tests for the issue #5592 metadata consistency checker.

These tests cover:
- Happy-path: all three topology rows agree between archetype YAML and docs table.
- Drift detection: map_id, primary_capability, and target_failure_mode mismatches
  each raise MetadataConsistencyError.
- Missing scenario: a scenario absent from the doc table or the archetype raises
  MetadataConsistencyError.
- Invalid fixture detection: a row flagged invalid_fixture=yes is rejected.
- CLI smoke: the default CLI invocation emits a compact human-readable pass line.
- CLI JSON mode: the default JSON CLI returns a parseable consistent payload.
"""

from __future__ import annotations

import json
import textwrap
from pathlib import Path

import pytest
import yaml

from scripts.validation import review_issue_5592_metadata_consistency as reviewer

REPO_ROOT = Path(__file__).resolve().parents[2]
DOC_TABLE = REPO_ROOT / "docs/context/issue_596_atomic_scenario_matrix.md"
ARCHETYPE = REPO_ROOT / "configs/scenarios/archetypes/issue_596_topology.yaml"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_doc_table(tmp_path: Path, extra_rows: str = "") -> Path:
    """Minimal docs table with the three topology rows plus any extras."""
    content = textwrap.dedent(
        f"""\
        # Issue 596 Atomic Scenario Matrix

        | Scenario | Map ID | Primary Capability | Target Failure Mode | Verified-Simple | Invalid Fixture |
        | --- | --- | --- | --- | --- | --- |
        | `corner_90_turn` | `atomic_corner_90_test` | `topology` | `oscillation` | no | no |
        | `u_trap_local_minimum` | `atomic_u_trap_test` | `topology` | `local_minima` | no | no |
        | `corridor_following` | `atomic_corridor_test` | `topology` | `oscillation` | no | no |
        {extra_rows}
        """
    )
    p = tmp_path / "issue_596_atomic_scenario_matrix.md"
    p.write_text(content, encoding="utf-8")
    return p


def _write_archetype(tmp_path: Path, scenarios: list[dict]) -> Path:
    """Minimal archetype YAML with the supplied scenario list."""
    content = yaml.dump({"scenarios": scenarios})
    p = tmp_path / "issue_596_topology.yaml"
    p.write_text(content, encoding="utf-8")
    return p


def _topology_scenario(
    name: str,
    map_id: str,
    primary_capability: str = "topology",
    target_failure_mode: str = "oscillation",
) -> dict:
    return {
        "name": name,
        "map_id": map_id,
        "metadata": {
            "primary_capability": primary_capability,
            "target_failure_mode": target_failure_mode,
        },
    }


_GOOD_SCENARIOS = [
    _topology_scenario(
        "corner_90_turn", "atomic_corner_90_test", target_failure_mode="oscillation"
    ),
    _topology_scenario(
        "u_trap_local_minimum", "atomic_u_trap_test", target_failure_mode="local_minima"
    ),
    _topology_scenario(
        "corridor_following", "atomic_corridor_test", target_failure_mode="oscillation"
    ),
]


# ---------------------------------------------------------------------------
# Happy-path: canonical repo files
# ---------------------------------------------------------------------------


def test_real_repo_files_are_consistent() -> None:
    """The committed archetype and docs table agree for all three topology rows."""
    result = reviewer.review_metadata_consistency(
        doc_path=DOC_TABLE,
        archetype_path=ARCHETYPE,
    )
    assert result["status"] == "consistent"
    assert result["issue"] == 5592
    assert result["selected_scenario_count"] == 3
    assert len(result["rows"]) == 3
    for row in result["rows"]:
        assert row["status"] == "consistent"
        assert row["primary_capability"] == "topology"


# ---------------------------------------------------------------------------
# Synthetic happy-path
# ---------------------------------------------------------------------------


def test_synthetic_consistent_rows_pass(tmp_path: Path) -> None:
    """All three rows match → status is consistent."""
    doc = _write_doc_table(tmp_path)
    arch = _write_archetype(tmp_path, _GOOD_SCENARIOS)
    result = reviewer.review_metadata_consistency(doc_path=doc, archetype_path=arch)
    assert result["status"] == "consistent"
    assert result["selected_scenario_count"] == 3


# ---------------------------------------------------------------------------
# Drift detection
# ---------------------------------------------------------------------------


def test_rejects_map_id_mismatch(tmp_path: Path) -> None:
    """A wrong map_id in the archetype raises MetadataConsistencyError."""
    doc = _write_doc_table(tmp_path)
    bad = [
        _topology_scenario("corner_90_turn", "WRONG_MAP_ID"),  # map_id drifted
        _topology_scenario(
            "u_trap_local_minimum", "atomic_u_trap_test", target_failure_mode="local_minima"
        ),
        _topology_scenario("corridor_following", "atomic_corridor_test"),
    ]
    arch = _write_archetype(tmp_path, bad)
    with pytest.raises(reviewer.MetadataConsistencyError, match="map_id mismatch"):
        reviewer.review_metadata_consistency(doc_path=doc, archetype_path=arch)


def test_rejects_primary_capability_mismatch(tmp_path: Path) -> None:
    """A wrong primary_capability raises MetadataConsistencyError."""
    doc = _write_doc_table(tmp_path)
    bad = [
        _topology_scenario(
            "corner_90_turn", "atomic_corner_90_test", primary_capability="static_avoidance"
        ),
        _topology_scenario(
            "u_trap_local_minimum", "atomic_u_trap_test", target_failure_mode="local_minima"
        ),
        _topology_scenario("corridor_following", "atomic_corridor_test"),
    ]
    arch = _write_archetype(tmp_path, bad)
    with pytest.raises(reviewer.MetadataConsistencyError, match="primary_capability mismatch"):
        reviewer.review_metadata_consistency(doc_path=doc, archetype_path=arch)


def test_rejects_target_failure_mode_mismatch(tmp_path: Path) -> None:
    """A wrong target_failure_mode raises MetadataConsistencyError."""
    doc = _write_doc_table(tmp_path)
    bad = [
        _topology_scenario(
            "corner_90_turn", "atomic_corner_90_test", target_failure_mode="local_minima"
        ),
        _topology_scenario(
            "u_trap_local_minimum", "atomic_u_trap_test", target_failure_mode="local_minima"
        ),
        _topology_scenario("corridor_following", "atomic_corridor_test"),
    ]
    arch = _write_archetype(tmp_path, bad)
    with pytest.raises(reviewer.MetadataConsistencyError, match="target_failure_mode mismatch"):
        reviewer.review_metadata_consistency(doc_path=doc, archetype_path=arch)


# ---------------------------------------------------------------------------
# Missing-scenario detection
# ---------------------------------------------------------------------------


def test_rejects_scenario_missing_from_doc_table(tmp_path: Path) -> None:
    """A scenario not in the docs table raises MetadataConsistencyError."""
    # Doc table only has two of the three rows.
    content = textwrap.dedent(
        """\
        # Stripped table

        | Scenario | Map ID | Primary Capability | Target Failure Mode | Verified-Simple | Invalid Fixture |
        | --- | --- | --- | --- | --- | --- |
        | `corner_90_turn` | `atomic_corner_90_test` | `topology` | `oscillation` | no | no |
        | `u_trap_local_minimum` | `atomic_u_trap_test` | `topology` | `local_minima` | no | no |
        """
    )
    doc = tmp_path / "partial_table.md"
    doc.write_text(content, encoding="utf-8")
    arch = _write_archetype(tmp_path, _GOOD_SCENARIOS)
    with pytest.raises(reviewer.MetadataConsistencyError, match="not found in documentation table"):
        reviewer.review_metadata_consistency(doc_path=doc, archetype_path=arch)


def test_rejects_scenario_missing_from_archetype(tmp_path: Path) -> None:
    """A scenario absent from the archetype YAML raises MetadataConsistencyError."""
    doc = _write_doc_table(tmp_path)
    partial = _GOOD_SCENARIOS[:2]  # only two scenarios
    arch = _write_archetype(tmp_path, partial)
    with pytest.raises(reviewer.MetadataConsistencyError, match="not found in archetype file"):
        reviewer.review_metadata_consistency(doc_path=doc, archetype_path=arch)


# ---------------------------------------------------------------------------
# Invalid-fixture gate
# ---------------------------------------------------------------------------


def test_rejects_invalid_fixture_row(tmp_path: Path) -> None:
    """A row marked invalid_fixture=yes in the docs table is rejected."""
    content = textwrap.dedent(
        """\
        # Table

        | Scenario | Map ID | Primary Capability | Target Failure Mode | Verified-Simple | Invalid Fixture |
        | --- | --- | --- | --- | --- | --- |
        | `corner_90_turn` | `atomic_corner_90_test` | `topology` | `oscillation` | no | yes |
        | `u_trap_local_minimum` | `atomic_u_trap_test` | `topology` | `local_minima` | no | no |
        | `corridor_following` | `atomic_corridor_test` | `topology` | `oscillation` | no | no |
        """
    )
    doc = tmp_path / "bad_fixture.md"
    doc.write_text(content, encoding="utf-8")
    arch = _write_archetype(tmp_path, _GOOD_SCENARIOS)
    with pytest.raises(reviewer.MetadataConsistencyError, match="invalid fixture"):
        reviewer.review_metadata_consistency(doc_path=doc, archetype_path=arch)


# ---------------------------------------------------------------------------
# File-not-found paths
# ---------------------------------------------------------------------------


def test_missing_doc_table_raises_file_not_found(tmp_path: Path) -> None:
    arch = _write_archetype(tmp_path, _GOOD_SCENARIOS)
    with pytest.raises(FileNotFoundError):
        reviewer.review_metadata_consistency(
            doc_path=tmp_path / "nonexistent.md",
            archetype_path=arch,
        )


def test_missing_archetype_raises_file_not_found(tmp_path: Path) -> None:
    doc = _write_doc_table(tmp_path)
    with pytest.raises(FileNotFoundError):
        reviewer.review_metadata_consistency(
            doc_path=doc,
            archetype_path=tmp_path / "nonexistent.yaml",
        )


# ---------------------------------------------------------------------------
# CLI smoke tests
# ---------------------------------------------------------------------------


def test_cli_text_mode_passes_on_real_files(capsys) -> None:
    """Default CLI (text mode) passes and emits a human-readable 'consistent:' line."""
    exit_code = reviewer.main([])
    assert exit_code == 0
    out = capsys.readouterr().out
    assert out.startswith("consistent:")
    assert "3 topology scenario rows" in out


def test_cli_json_mode_passes_on_real_files(capsys) -> None:
    """--json CLI returns a parseable consistent payload."""
    exit_code = reviewer.main(["--json"])
    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out.strip())
    assert payload["status"] == "consistent"
    assert payload["issue"] == 5592
    assert payload["selected_scenario_count"] == 3


def test_cli_json_mode_returns_nonzero_on_bad_archetype(tmp_path: Path, capsys) -> None:
    """Inconsistent archetype returns exit code 1 and a JSON error payload."""
    bad_arch = tmp_path / "bad.yaml"
    bad_arch.write_text(yaml.dump({"scenarios": []}), encoding="utf-8")
    exit_code = reviewer.main(["--archetype", str(bad_arch), "--json"])
    assert exit_code == 1
    payload = json.loads(capsys.readouterr().out.strip())
    assert payload["status"] == "inconsistent"
    assert "error" in payload
