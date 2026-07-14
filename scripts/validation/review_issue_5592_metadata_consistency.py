#!/usr/bin/env python3
"""Review-gate: verify that the issue #5592 atomic-topology rows are internally
consistent with the ``docs/context/issue_596_atomic_scenario_matrix.md``
documentation conventions.

This is the second-pass metadata consistency checker referenced in the issue
#5592 acceptance criteria::

    Review gate: a second pass confirms the new matrix's per-scenario metadata
    (failure mode / capability tags) is populated and internally consistent with
    the existing atomic_scenario_matrix documentation conventions.

The checker:

1. Parses the Markdown table in ``docs/context/issue_596_atomic_scenario_matrix.md``
   as the authoritative source-of-truth for ``(scenario, map_id,
   primary_capability, target_failure_mode, invalid_fixture)`` triples.
2. Loads the ``configs/scenarios/archetypes/issue_596_topology.yaml`` archetype
   file and extracts per-scenario metadata fields for the three selected rows
   (``corner_90_turn``, ``u_trap_local_minimum``, ``corridor_following``).
3. Cross-checks each loaded scenario row against the documentation table and
   fails closed on any drift — a mismatched field or a missing scenario row is
   a hard error.
4. Emits a compact JSON or text result so downstream tools can consume the gate
   status programmatically.

No planner episodes are run, no compute is submitted, and no structural ranking
is produced or modified.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
DOC_TABLE_PATH = REPO_ROOT / "docs/context/issue_596_atomic_scenario_matrix.md"
ARCHETYPE_PATH = REPO_ROOT / "configs/scenarios/archetypes/issue_596_topology.yaml"

# The three topology rows selected by issue #5592, in matrix order.
SELECTED_SCENARIO_IDS = ("corner_90_turn", "u_trap_local_minimum", "corridor_following")

# Fields that must be present and non-empty in each archetype scenario's metadata.
REQUIRED_METADATA_FIELDS = ("map_id", "primary_capability", "target_failure_mode")

# Canonical capability tag for the atomic topology distribution.
EXPECTED_PRIMARY_CAPABILITY = "topology"


class MetadataConsistencyError(ValueError):
    """Raised when archetype metadata drifts from documentation conventions."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise MetadataConsistencyError(message)


# ---------------------------------------------------------------------------
# Documentation table parser
# ---------------------------------------------------------------------------

_MD_TABLE_ROW = re.compile(
    r"^\|\s*`([^`]+)`\s*\|"  # Scenario (backtick-quoted)
    r"\s*`([^`]+)`\s*\|"  # Map ID
    r"\s*`([^`]+)`\s*\|"  # Primary Capability
    r"\s*`([^`]+)`\s*\|"  # Target Failure Mode
    r"\s*(yes|no)\s*\|"  # Verified-Simple
    r"\s*(yes|no)\s*\|",  # Invalid Fixture
    re.IGNORECASE,
)


def parse_doc_table(doc_path: Path) -> dict[str, dict[str, str]]:
    """Return a mapping from scenario_id to its documentation-table row fields.

    The returned dict has the form::

        {
            "corner_90_turn": {
                "map_id": "atomic_corner_90_test",
                "primary_capability": "topology",
                "target_failure_mode": "oscillation",
                "verified_simple": "no",
                "invalid_fixture": "no",
            },
            ...
        }

    Raises ``FileNotFoundError`` when the documentation file is absent.
    Raises ``MetadataConsistencyError`` when the table cannot be parsed.
    """
    if not doc_path.is_file():
        raise FileNotFoundError(f"Documentation table not found: {doc_path}")

    rows: dict[str, dict[str, str]] = {}
    for line in doc_path.read_text(encoding="utf-8").splitlines():
        m = _MD_TABLE_ROW.match(line.strip())
        if not m:
            continue
        scenario_id = m.group(1)
        rows[scenario_id] = {
            "map_id": m.group(2),
            "primary_capability": m.group(3),
            "target_failure_mode": m.group(4),
            "verified_simple": m.group(5).lower(),
            "invalid_fixture": m.group(6).lower(),
        }

    _require(rows, f"No table rows parsed from {doc_path}; check file format")
    return rows


# ---------------------------------------------------------------------------
# Archetype YAML loader
# ---------------------------------------------------------------------------


def load_archetype_scenarios(archetype_path: Path) -> dict[str, dict[str, Any]]:
    """Return a mapping from scenario name to its archetype definition dict.

    Raises ``FileNotFoundError`` when the archetype file is absent.
    Raises ``MetadataConsistencyError`` on structural problems.
    """
    if not archetype_path.is_file():
        raise FileNotFoundError(f"Archetype file not found: {archetype_path}")

    raw: Any = yaml.safe_load(archetype_path.read_text(encoding="utf-8"))
    _require(isinstance(raw, dict), f"Archetype root must be a mapping: {archetype_path}")
    scenarios_list = raw.get("scenarios", [])
    _require(isinstance(scenarios_list, list), "Archetype 'scenarios' must be a list")

    scenarios: dict[str, dict[str, Any]] = {}
    for index, entry in enumerate(scenarios_list):
        _require(isinstance(entry, dict), f"scenarios[{index}] must be a mapping")
        name = entry.get("name")
        _require(isinstance(name, str) and name.strip(), f"scenarios[{index}].name missing")
        scenarios[name] = entry

    return scenarios


# ---------------------------------------------------------------------------
# Cross-check engine
# ---------------------------------------------------------------------------


def _check_row(
    scenario_id: str,
    doc_row: dict[str, str],
    archetype_row: dict[str, Any],
) -> dict[str, Any]:
    """Cross-check one scenario row.  Returns a per-row result dict.

    Raises ``MetadataConsistencyError`` on any field mismatch or missing data.
    """
    metadata = archetype_row.get("metadata", {})
    _require(isinstance(metadata, dict), f"{scenario_id}: 'metadata' must be a mapping")

    # map_id is a top-level field in the archetype scenario, not under metadata.
    archetype_map_id = archetype_row.get("map_id")
    _require(
        isinstance(archetype_map_id, str) and archetype_map_id.strip(),
        f"{scenario_id}: archetype 'map_id' is missing or empty",
    )
    doc_map_id = doc_row["map_id"]
    _require(
        archetype_map_id == doc_map_id,
        (
            f"{scenario_id}: map_id mismatch — "
            f"archetype has '{archetype_map_id}', docs table has '{doc_map_id}'"
        ),
    )

    for field in ("primary_capability", "target_failure_mode"):
        archetype_val = metadata.get(field)
        _require(
            isinstance(archetype_val, str) and archetype_val.strip(),
            f"{scenario_id}: archetype metadata.{field} is missing or empty",
        )
        doc_val = doc_row[field]
        _require(
            archetype_val == doc_val,
            (
                f"{scenario_id}: {field} mismatch — "
                f"archetype has '{archetype_val}', docs table has '{doc_val}'"
            ),
        )

    # The three selected rows must not be flagged as invalid fixtures.
    _require(
        doc_row["invalid_fixture"] == "no",
        f"{scenario_id}: documentation table marks this row as an invalid fixture",
    )

    # All topology rows must carry the canonical primary_capability tag.
    archetype_cap = metadata.get("primary_capability", "")
    _require(
        archetype_cap == EXPECTED_PRIMARY_CAPABILITY,
        (
            f"{scenario_id}: primary_capability must be '{EXPECTED_PRIMARY_CAPABILITY}', "
            f"got '{archetype_cap}'"
        ),
    )

    return {
        "scenario_id": scenario_id,
        "map_id": archetype_map_id,
        "primary_capability": archetype_cap,
        "target_failure_mode": metadata.get("target_failure_mode"),
        "invalid_fixture": doc_row["invalid_fixture"],
        "status": "consistent",
    }


def review_metadata_consistency(
    *,
    doc_path: Path = DOC_TABLE_PATH,
    archetype_path: Path = ARCHETYPE_PATH,
    selected_ids: tuple[str, ...] = SELECTED_SCENARIO_IDS,
) -> dict[str, Any]:
    """Run the full metadata consistency review.

    Returns a result dict with ``status`` equal to ``"consistent"`` on success,
    or raises ``MetadataConsistencyError`` (or ``FileNotFoundError``) on failure.
    """
    doc_table = parse_doc_table(doc_path)
    archetype_scenarios = load_archetype_scenarios(archetype_path)

    row_results: list[dict[str, Any]] = []
    for scenario_id in selected_ids:
        # Check doc table contains the scenario.
        _require(
            scenario_id in doc_table,
            (
                f"'{scenario_id}' not found in documentation table "
                f"({doc_path.name}); the scenario may have been renamed or removed"
            ),
        )
        # Check archetype contains the scenario.
        _require(
            scenario_id in archetype_scenarios,
            (
                f"'{scenario_id}' not found in archetype file "
                f"({archetype_path.name}); the scenario may have been renamed or removed"
            ),
        )
        row_results.append(
            _check_row(scenario_id, doc_table[scenario_id], archetype_scenarios[scenario_id])
        )

    def _rel(p: Path) -> str:
        try:
            return str(p.relative_to(REPO_ROOT))
        except ValueError:
            return str(p)

    return {
        "status": "consistent",
        "issue": 5592,
        "review_gate": "metadata_consistency",
        "doc_source": _rel(doc_path),
        "archetype_source": _rel(archetype_path),
        "selected_scenario_count": len(selected_ids),
        "rows": row_results,
        "claim_boundary": (
            "CPU-only metadata review; no campaign run, no ranking produced, "
            "no paper or dissertation claim modified."
        ),
    }


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    """Run the metadata consistency review and emit a result."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--doc-table", type=Path, default=DOC_TABLE_PATH)
    parser.add_argument("--archetype", type=Path, default=ARCHETYPE_PATH)
    parser.add_argument("--json", action="store_true", dest="as_json")
    args = parser.parse_args(argv)

    try:
        result = review_metadata_consistency(
            doc_path=args.doc_table,
            archetype_path=args.archetype,
        )
    except (FileNotFoundError, MetadataConsistencyError, yaml.YAMLError, OSError) as exc:
        if args.as_json:
            print(json.dumps({"status": "inconsistent", "error": str(exc)}))
        else:
            print(f"inconsistent: {exc}", file=sys.stderr)
        return 1

    if args.as_json:
        print(json.dumps(result, indent=2, sort_keys=True))
    else:
        count = result["selected_scenario_count"]
        print(
            f"consistent: {count} topology scenario rows match "
            "docs/context/issue_596_atomic_scenario_matrix.md conventions"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
