#!/usr/bin/env python3
"""Compare immutable receipt rows without importing or invoking an episode runner.

This produces a comparison artifact, never an admission receipt or an accepted classification.
Existing rows carry execution-time metrics; the scoring identity binds their scorer/semantics,
without claiming arbitrary trajectory rescoring is available.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from robot_sf.benchmark.metric_definitions import (
    require_uniform_metric_schema,
)
from robot_sf.benchmark.receipt_stages import atomic_json, file_digest, stage_key
from scripts.ci import make_behaviour_receipt as producer

ROOT = Path(__file__).resolve().parents[2]


def _inventory(directory: Path) -> dict[str, str]:
    paths = [*directory.rglob("*.jsonl"), directory / "contract.json"]
    return {
        str(path.relative_to(directory)): file_digest(path)
        for path in sorted(paths)
        if path.is_file()
    }


def _scoring_contract(directory: Path, rows: dict) -> dict:
    """Recover the actual execution-time scorer identity, never the current checkout's scorer.

    Returns:
        Source-bound scoring provenance from the immutable producing contract.
    """
    path = directory / "contract.json"
    if not path.is_file():
        raise ValueError("immutable producing contract required for scoring provenance")
    contract = json.loads(path.read_bytes())
    execution = contract["execution"]
    if contract["execution_key"] != stage_key("execution", execution):
        raise ValueError("producing execution contract digest mismatch")
    source = contract["source_sha"]
    definitions = execution["metric_definitions_sha256"]
    if execution["source_sha"] != source:
        raise ValueError("producing scoring source differs from execution contract")
    for row in rows.values():
        if row.get("git_hash") != source:
            raise ValueError("row scoring source differs from producing execution contract")
        recorded = row.get("metrics", {}).get("metric_definitions_sha256")
        if recorded is not None and recorded != definitions:
            raise ValueError("row metric definitions differ from producing contract")
    return {
        "source_sha": source,
        "execution_key": contract["execution_key"],
        "metric_schema_version": contract["metric_schema_version"],
        "metric_definitions_sha256": definitions,
        "scorer_bundle": execution.get("scorer_bundle", {"complete_producing_source_sha": source}),
        "mode": "execution-time-metrics",
    }


def compare(head_dir: Path, baseline_dir: Path, scope: dict, output: Path) -> dict:
    """Read immutable raw/summary inputs, keep full coverage, and publish a comparison.

    Returns:
        The validated stage result."""
    before = [_inventory(head_dir), _inventory(baseline_dir)]
    head = producer._load_sweep_rows(head_dir)
    baseline = producer._load_sweep_rows(baseline_dir)
    version = require_uniform_metric_schema([*head.values(), *baseline.values()])
    rows, classifications, totals = producer.build_rows_and_classifications(
        head,
        baseline,
        scope,
        classification_class="measurement_inconsistency",
        evidence_base_uri="https://example.invalid/requires-independent-classification",
    )
    scoring = {
        "head": _scoring_contract(head_dir, head),
        "baseline": _scoring_contract(baseline_dir, baseline),
    }
    if (
        scoring["head"]["metric_definitions_sha256"]
        != scoring["baseline"]["metric_definitions_sha256"]
    ):
        raise ValueError("incompatible metric definitions in producing scoring contracts")
    if any(contract["metric_schema_version"] != version for contract in scoring.values()):
        raise ValueError("row metric schema differs from producing scoring contract")
    scoring_keys = {
        label: stage_key(
            "scoring",
            {
                "sorted_row_digests": sorted(stage_key("row", row) for row in data.values()),
                **scoring[label],
            },
        )
        for label, data in (("head", head), ("baseline", baseline))
    }
    comparison_key = stage_key(
        "comparison",
        {
            "scoring_keys": scoring_keys,
            "baseline_digest": stage_key(
                "baseline", sorted(stage_key("row", row) for row in baseline.values())
            ),
            "comparator_bundle": {
                "producer": file_digest(Path(producer.__file__)),
                "comparison": file_digest(Path(__file__)),
                "identities": file_digest(ROOT / "robot_sf/benchmark/receipt_stages.py"),
            },
            "comparison_source_sha": producer._full_sha(ROOT, "HEAD"),
            "acceptance_policy": scope,
        },
    )
    if before != [_inventory(head_dir), _inventory(baseline_dir)]:
        raise ValueError("comparison input rows changed while reading")
    result = {
        "schema_version": "receipt-comparison.v1",
        "comparison_key": comparison_key,
        "scoring_keys": scoring_keys,
        "scoring_contracts": scoring,
        "comparison_source_sha": producer._full_sha(ROOT, "HEAD"),
        "input_file_digests": before,
        "totals": totals,
        "rows": rows,
        "unresolved_classification_events": [
            {key: value for key, value in event.items() if key not in {"class", "evidence"}}
            for event in classifications
        ],
        "admission": "not_evaluated",
    }
    atomic_json(output, result, immutable=True)
    return result


def main(argv: list[str] | None = None) -> int:
    """Run comparison only.

    Returns:
        Zero after immutable comparison publication.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--head-sweep-dir", type=Path, required=True)
    parser.add_argument("--baseline-sweep-dir", type=Path, required=True)
    parser.add_argument("--scope-path", type=Path, default=producer.behaviour_receipt.SCOPE_PATH)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    result = compare(
        args.head_sweep_dir,
        args.baseline_sweep_dir,
        json.loads(args.scope_path.read_bytes()),
        args.output,
    )
    print(
        json.dumps(
            {
                "comparison_key": result["comparison_key"],
                "totals": result["totals"],
                "admission": result["admission"],
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
