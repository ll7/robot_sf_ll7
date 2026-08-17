"""Run the fail-closed deterministic diagnosis fixture evaluation (#7197)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from robot_sf.benchmark.failure_diagnosis import (
    FAILURE_DIAGNOSIS_QUALITY_SCHEMA_VERSION,
    FailureDiagnosisError,
)
from robot_sf.benchmark.failure_diagnosis_fixture import (
    evaluate_deterministic_failure_diagnosis_fixture,
    load_failure_diagnosis_fixture_manifest,
)


def _load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise FailureDiagnosisError(f"unable to load JSON input: {path}") from exc


def _unavailable_report(reason: str) -> dict[str, Any]:
    return {
        "schema_version": FAILURE_DIAGNOSIS_QUALITY_SCHEMA_VERSION,
        "output_status": "unavailable",
        "output_reason": reason,
        "metrics": None,
        "claim_boundary": {
            "deterministic_diagnostic_metric_integrity_only": True,
            "no_general_diagnostic_accuracy_claim": True,
            "no_correction_usefulness_claim": True,
            "no_campaign_or_benchmark_ranking": True,
        },
    }


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    parser = argparse.ArgumentParser(
        description=(
            "Run deterministic failure-diagnosis metric-integrity evaluation "
            "after source and reference admission."
        )
    )
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--source-predicates", required=True, type=Path)
    parser.add_argument("--reference-fixture", required=True, type=Path)
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional path for the JSON report; stdout always receives the report.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Evaluate one manifest/source/reference bundle."""
    args = build_parser().parse_args(argv)
    try:
        manifest = load_failure_diagnosis_fixture_manifest(args.manifest)
        source_predicates = _load_json(args.source_predicates)
        reference_fixture = _load_json(args.reference_fixture)
        report = evaluate_deterministic_failure_diagnosis_fixture(
            manifest,
            source_predicates,
            reference_fixture,
        )
    except FailureDiagnosisError as exc:
        report = _unavailable_report(str(exc))
        exit_code = 2
    else:
        exit_code = 0

    serialized = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        try:
            args.output.write_text(serialized, encoding="utf-8")
        except OSError as exc:
            raise FailureDiagnosisError(f"unable to write report: {args.output}") from exc
    print(serialized, end="")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
