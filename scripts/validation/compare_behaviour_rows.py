#!/usr/bin/env python3
"""Compare immutable receipt rows without importing or invoking an episode runner.

This produces a comparison artifact, never an admission receipt or an accepted classification.
Existing rows carry execution-time metrics; the scoring identity binds their scorer/semantics,
without claiming arbitrary trajectory rescoring is available.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
from pathlib import Path

from robot_sf.benchmark.metric_definitions import (
    require_uniform_metric_schema,
)
from robot_sf.benchmark.receipt_stages import (
    DESCRIPTOR,
    METRIC_SOURCE_PATHS,
    ROW_SCHEMA_PATH,
    atomic_json,
    encode_row,
    file_digest,
    stage_key,
)
from scripts.ci import make_behaviour_receipt as producer

ROOT = Path(__file__).resolve().parents[2]

OUTCOME_PROJECTION = "receipt-navigation-outcomes.v1"
# Producer fingerprints covered by the R1 source comparison. New metric/runner
# bytes require a renewed equivalence check, even under the same schema version.
_OUTCOME_PRODUCERS = {
    # Published 0.0.8.
    "34a9b5eb6644b841ea2fe0b73446f8c8e11178ef1bb8b288c7920b5fe48ca6d0": "legacy-force-reduction",
    # Main before the behaviour integration train.
    "74bae03b49f8bda7ba51eb1a58ea8dc2b47642f092d0daf1af22c969d15f1e54": "presence-aware-force-reduction",
    # Batch 2 integration candidate, PR #10479.
    "535fce599852ab02d75fabda58676185f5ff3924f17e73b29c4e74aabad1b640": "presence-aware-force-reduction",
}


def _inventory(directory: Path) -> dict[str, str]:
    paths = [
        *directory.rglob("*.jsonl"),
        directory / "contract.json",
        *directory.glob(f"campaigns/*/{DESCRIPTOR}"),
        *directory.glob(f"campaigns/*/runs/*/{DESCRIPTOR}"),
        *directory.glob("campaigns/*/runs/*/completed/*.json"),
    ]
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
    producer._verify_completion_views(directory, contract, rows)
    return {
        "source_sha": source,
        "execution_key": contract["execution_key"],
        "metric_schema_version": contract["metric_schema_version"],
        "row_schema_sha256": contract.get("row_schema_sha256"),
        "metric_definitions_sha256": definitions,
        "metric_semantics_sha256": execution.get("metric_semantics_sha256", definitions),
        "scorer_bundle": execution.get("scorer_bundle", {"complete_producing_source_sha": source}),
        "mode": "execution-time-metrics",
    }


def _committed_digest(source: str, path: str) -> str:
    result = subprocess.run(
        ["git", "cat-file", "blob", f"{source}:{path}"],
        cwd=ROOT,
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        raise ValueError(f"cannot verify committed producing bytes for {path}")
    return hashlib.sha256(result.stdout).hexdigest()


def _verify_projection_producer(directory: Path, scoring: dict) -> str:
    """Verify the schema and producer bytes separately for each source."""
    source = scoring["source_sha"]
    if not isinstance(source, str) or not re.fullmatch(r"[0-9a-f]{40}", source):
        raise ValueError("outcome projection requires a full producing source SHA")
    contract = json.loads((directory / "contract.json").read_bytes())
    if contract.get("execution_key") != scoring["execution_key"]:
        raise ValueError("outcome projection requires the original producing contract")
    execution = contract["execution"]
    if stage_key("execution", execution) != scoring["execution_key"]:
        raise ValueError("outcome projection producing contract changed")
    digest = scoring["metric_semantics_sha256"]
    if digest not in _OUTCOME_PRODUCERS:
        raise ValueError("outcome projection has no equivalence evidence for the producer")
    fingerprints = {path: _committed_digest(source, path) for path in METRIC_SOURCE_PATHS}
    if (
        fingerprints != execution.get("metric_semantics_sources")
        or stage_key("metric-semantics", fingerprints) != digest
        or _committed_digest(source, ROW_SCHEMA_PATH) != scoring["row_schema_sha256"]
    ):
        raise ValueError("outcome projection producer fingerprint differs from committed source")
    return _OUTCOME_PRODUCERS[digest]


def _project_row(row: dict) -> tuple[bool, int]:
    """Read the common terminal metric fields, without fallback to trace status."""
    metrics = row.get("metrics")
    if not isinstance(metrics, dict):
        raise ValueError("outcome projection requires the producing metrics")
    success = metrics.get("success")
    collisions = metrics.get("total_collision_count")
    if type(success) is not bool or type(collisions) is not int or collisions < 0:
        raise ValueError(
            "outcome projection requires boolean success and an integer collision count"
        )
    if metrics.get("collisions") != collisions or (success and collisions):
        raise ValueError("outcome projection collision and success fields disagree")
    return success, collisions


def _project_pair(head: dict, baseline: dict) -> tuple[bool, int, bool, int]:
    return (*_project_row(head), *_project_row(baseline))


def compare(
    head_dir: Path,
    baseline_dir: Path,
    scope: dict,
    output: Path,
    *,
    outcome_projection: bool = False,
) -> dict:
    """Read immutable raw/summary inputs, keep full coverage, and publish a comparison.

    Returns:
        The validated stage result, with admission left to independent review.
    """
    before = [_inventory(head_dir), _inventory(baseline_dir)]
    head = producer._load_sweep_rows(head_dir)
    baseline = producer._load_sweep_rows(baseline_dir)
    scoring = {
        "head": _scoring_contract(head_dir, head),
        "baseline": _scoring_contract(baseline_dir, baseline),
    }
    projection = None
    if outcome_projection:
        projection = {
            "schema_version": OUTCOME_PROJECTION,
            "metrics": ["success", "total_collision_count"],
            "source_specific_force_definitions": {
                "head": _verify_projection_producer(head_dir, scoring["head"]),
                "baseline": _verify_projection_producer(baseline_dir, scoring["baseline"]),
            },
        }
        for label, data in (("head", head), ("baseline", baseline)):
            if (
                require_uniform_metric_schema(data.values())
                != scoring[label]["metric_schema_version"]
            ):
                raise ValueError("row metric schema differs from producing scoring contract")
    else:
        version = require_uniform_metric_schema([*head.values(), *baseline.values()])
        if (
            scoring["head"]["metric_semantics_sha256"]
            != scoring["baseline"]["metric_semantics_sha256"]
        ):
            raise ValueError("incompatible metric definitions in producing scoring contracts")
        if any(contract["metric_schema_version"] != version for contract in scoring.values()):
            raise ValueError("row metric schema differs from producing scoring contract")

    rows, classifications, totals = producer.build_rows_and_classifications(
        head,
        baseline,
        scope,
        classification_class="measurement_inconsistency",
        evidence_base_uri="https://example.invalid/requires-independent-classification",
        outcome_projection=_project_pair if outcome_projection else None,
    )
    scoring_keys = {
        label: stage_key(
            "scoring",
            {
                "sorted_row_digests": sorted(
                    stage_key("row", encode_row(row)) for row in data.values()
                ),
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
                "baseline", sorted(stage_key("row", encode_row(row)) for row in baseline.values())
            ),
            "comparator_bundle": {
                "producer": file_digest(Path(producer.__file__)),
                "comparison": file_digest(Path(__file__)),
                "identities": file_digest(ROOT / "robot_sf/benchmark/receipt_stages.py"),
            },
            "comparison_source_sha": producer._full_sha(ROOT, "HEAD"),
            "acceptance_policy": scope,
            "outcome_projection": projection,
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
        "outcome_projection": projection,
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
    parser.add_argument(
        "--outcome-projection",
        action="store_true",
        help="Compare source-verified terminal success and collision-category timesteps only",
    )
    args = parser.parse_args(argv)
    result = compare(
        args.head_sweep_dir,
        args.baseline_sweep_dir,
        json.loads(args.scope_path.read_bytes()),
        args.output,
        outcome_projection=args.outcome_projection,
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
