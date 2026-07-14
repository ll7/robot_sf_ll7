#!/usr/bin/env python3
"""Export a versioned scenario-evidence crosswalk for coverage and exemplar selection.

Joins canonical scenario taxonomy, #5593 predicate-export availability, and an
optional evidence/eligibility catalog into one deterministic
``scenario_evidence_crosswalk.v1`` artifact plus a compact Markdown coverage
report and a CSV summary.

Consumes #5593's predicate export when available; until that lane lands,
predicate fields are explicit ``unavailable`` (never inferred from motivation
text). Geometry groups and validated mechanisms are emitted as separate fields
with separate provenance -- geometry never implies a causal mechanism.

Usage:
    python scripts/tools/export_scenario_evidence_crosswalk.py \
        configs/scenarios/nominal_v1.yaml \
        --output-json output/crosswalk.json \
        --output-markdown output/crosswalk.md \
        --output-csv output/crosswalk.csv
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping
from pathlib import Path

from robot_sf.benchmark.scenario_evidence_crosswalk import (
    SCHEMA_VERSION,
    build_scenario_evidence_crosswalk,
    validate_scenario_evidence_crosswalk,
    write_scenario_evidence_crosswalk,
)
from robot_sf.training.scenario_loader import load_scenarios
from robot_sf.training.task_bundles import is_task_bundle_reference


def load_scenario_matrix(path: str | Path) -> list[dict]:
    """Load scenario rows without importing benchmark runtime planners."""
    import yaml

    scenario_path = Path(path)
    if is_task_bundle_reference(scenario_path):
        return [dict(scenario) for scenario in load_scenarios(scenario_path)]

    with scenario_path.open("r", encoding="utf-8") as handle:
        raw_docs = list(yaml.safe_load_all(handle))
    docs = [doc for doc in raw_docs if doc is not None]
    if not docs:
        raise ValueError(f"Scenario matrix '{scenario_path}' is empty.")
    if len(raw_docs) > 1:
        invalid_docs = [
            index
            for index, scenario in enumerate(raw_docs)
            if scenario is not None and not isinstance(scenario, Mapping)
        ]
        if invalid_docs:
            raise ValueError(
                f"Scenario matrix '{scenario_path}' contains non-object YAML documents: "
                f"{invalid_docs}"
            )
        return [dict(scenario) for scenario in docs]
    return _load_single_document(scenario_path, docs[0])


def _is_abstract_scenario(scenario: Mapping[str, object]) -> bool:
    """Return whether an entry has no manifest/map-loader identity fields."""
    return not any(key in scenario for key in ("name", "scenario_id", "map_file", "map_id"))


def _load_single_document(scenario_path: Path, single_doc: object) -> list[dict]:
    """Load one parsed YAML document without reparsing direct abstract matrices."""
    if isinstance(single_doc, list):
        if not single_doc or any(not isinstance(scenario, Mapping) for scenario in single_doc):
            raise ValueError(
                f"Scenario matrix '{scenario_path}' single-document lists must contain mappings."
            )
        if all(_is_abstract_scenario(scenario) for scenario in single_doc):
            return [dict(scenario) for scenario in single_doc]
    elif isinstance(single_doc, Mapping) and "scenarios" in single_doc:
        manifest_keys = frozenset(
            {
                "includes",
                "include",
                "scenario_files",
                "select_scenarios",
                "scenario_overrides",
                "scenario_overrides_by_name",
                "map_search_paths",
            }
        )
        scenario_entries = single_doc["scenarios"]
        if (
            not manifest_keys & single_doc.keys()
            and isinstance(scenario_entries, list)
            and scenario_entries
            and all(isinstance(scenario, Mapping) for scenario in scenario_entries)
            and all(_is_abstract_scenario(scenario) for scenario in scenario_entries)
        ):
            return [dict(scenario) for scenario in scenario_entries]
    elif not isinstance(single_doc, Mapping):
        raise ValueError(
            f"Scenario matrix '{scenario_path}' single YAML document must be a mapping or list."
        )
    return [dict(scenario) for scenario in load_scenarios(scenario_path, base_dir=scenario_path)]


def _load_json(path: Path | None) -> dict[str, object] | None:
    """Load an optional JSON manifest; return ``None`` when not provided."""
    if path is None:
        return None
    return json.loads(Path(path).read_text(encoding="utf-8"))


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("matrix", type=Path, help="Canonical scenario matrix YAML file.")
    parser.add_argument("--output-json", type=Path, help="Output crosswalk JSON path.")
    parser.add_argument("--output-markdown", type=Path, help="Output Markdown report path.")
    parser.add_argument("--output-csv", type=Path, help="Output CSV summary path.")
    parser.add_argument(
        "--predicate-export",
        type=Path,
        help="Optional #5593 predicate-export manifest (JSON).",
    )
    parser.add_argument(
        "--evidence-catalog",
        type=Path,
        help="Optional evidence/eligibility catalog (JSON: scenario id -> record).",
    )
    parser.add_argument(
        "--artifact-root",
        type=Path,
        help="Optional root to fail-closed on missing referenced artifacts.",
    )
    parser.add_argument(
        "--no-validate",
        action="store_true",
        help="Skip JSON Schema validation (not recommended).",
    )
    return parser.parse_args()


def main() -> int:
    """Run the crosswalk export and write artifacts."""
    args = parse_args()

    scenarios = load_scenario_matrix(args.matrix)
    predicate_export = _load_json(args.predicate_export)
    evidence_catalog = _load_json(args.evidence_catalog)
    artifact_root = Path(args.artifact_root) if args.artifact_root else None

    crosswalk = build_scenario_evidence_crosswalk(
        scenarios,
        source=str(args.matrix),
        predicate_export=predicate_export,
        evidence_catalog=evidence_catalog,
        artifact_root=artifact_root,
    )

    if not args.no_validate:
        errors = validate_scenario_evidence_crosswalk(crosswalk)
        if errors:
            print("Crosswalk validation failed:", file=sys.stderr)
            for error in errors:
                print(f"  - {error}", file=sys.stderr)
            return 1

    write_scenario_evidence_crosswalk(
        crosswalk,
        json_path=args.output_json,
        markdown_path=args.output_markdown,
        csv_path=args.output_csv,
    )

    summary = crosswalk["summary"]
    print(
        json.dumps(
            {
                "schema_version": SCHEMA_VERSION,
                "source": crosswalk["source"],
                "scenario_count": summary["scenario_count"],
                "eligible_scenarios": summary["eligible_scenarios"],
                "predicate_export_available": summary["predicate_export_available"],
                "predicate_unavailable": summary["predicate_unavailable"],
                "content_sha256": crosswalk["content_sha256"],
                "output_json": str(args.output_json) if args.output_json else None,
                "output_markdown": str(args.output_markdown) if args.output_markdown else None,
                "output_csv": str(args.output_csv) if args.output_csv else None,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
