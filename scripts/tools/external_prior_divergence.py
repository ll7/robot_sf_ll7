"""External-prior divergence de-risk tool (issue #3192).

Roughly half of the open research backlog is gated on licensed external datasets that may be
months away. This tool quantifies how far the priors we already have (authored / repository
configured) diverge from *published summary statistics* of those datasets, to answer a
strategic question without downloading any licensed data: are our existing priors close enough
for the intended diagnostic claims, or would the raw data materially change scope?

Honesty contract (mirrors the repository hard rule): a divergence is only computed for a
statistic when BOTH a cited published reference value and an authored value are present. Any
statistic whose reference value is uncited (``source_citation: NEEDS_CITATION`` or null value)
is reported as ``not-comparable`` and never silently treated as agreement. Per-dataset verdicts
use only the three canonical labels.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import yaml

VERDICT_SUFFICIENT = "priors-sufficient-for-diagnostic"
VERDICT_MATERIAL = "raw-data-materially-changes-scope"
VERDICT_INCONCLUSIVE = "inconclusive-need-pilot"

_NEEDS_CITATION = "NEEDS_CITATION"


def _coerce_finite_float(value: Any, *, field: str, dataset_key: str, stat_key: str) -> float:
    """Return a finite float or raise an actionable fail-closed input error."""
    try:
        coerced = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"{field} value for dataset '{dataset_key}' statistic '{stat_key}' must be numeric; "
            f"got {value!r}."
        ) from exc
    if not math.isfinite(coerced):
        raise ValueError(
            f"{field} value for dataset '{dataset_key}' statistic '{stat_key}' must be finite; "
            f"got {value!r}. Use null to mark the statistic not-comparable until a finite value is "
            "available."
        )
    return coerced


def _validate_threshold(value: float, *, name: str) -> None:
    """Reject non-finite CLI thresholds before classification uses them."""
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite; got {value!r}.")


def _is_cited(stat: dict[str, Any]) -> bool:
    """Return True only when the reference statistic has a value and a real citation."""
    citation = stat.get("source_citation")
    value = stat.get("value")
    return (
        value is not None
        and isinstance(citation, str)
        and citation.strip() not in {"", _NEEDS_CITATION}
    )


def _relative_divergence(authored: float, reference: float) -> float:
    """Return |authored - reference| normalized by the reference magnitude.

    Falls back to absolute difference when the reference is zero so a zero baseline does not
    produce an infinite or undefined divergence.
    """
    denom = abs(reference)
    if denom == 0.0:
        return abs(authored - reference)
    return abs(authored - reference) / denom


def classify_dataset(
    comparisons: list[dict[str, Any]],
    *,
    sufficient_threshold: float,
    material_threshold: float,
) -> str:
    """Classify a dataset from its per-statistic comparisons using the canonical verdicts."""
    comparable = [c for c in comparisons if c["status"] == "comparable"]
    if not comparable:
        return VERDICT_INCONCLUSIVE
    divergences = [c["relative_divergence"] for c in comparable]
    if any(d > material_threshold for d in divergences):
        return VERDICT_MATERIAL
    if all(d <= sufficient_threshold for d in divergences):
        return VERDICT_SUFFICIENT
    return VERDICT_INCONCLUSIVE


def compute_report(
    reference: dict[str, Any],
    authored: dict[str, Any],
    *,
    sufficient_threshold: float = 0.25,
    material_threshold: float = 0.5,
) -> dict[str, Any]:
    """Compute the per-dataset divergence report from reference and authored statistics."""
    _validate_threshold(sufficient_threshold, name="sufficient_threshold")
    _validate_threshold(material_threshold, name="material_threshold")
    authored_datasets = authored.get("datasets", {})
    datasets_report: list[dict[str, Any]] = []

    for dataset in reference.get("datasets", []):
        key = dataset["key"]
        authored_stats = authored_datasets.get(key, {})
        comparisons: list[dict[str, Any]] = []
        for stat in dataset.get("statistics", []):
            stat_key = stat["key"]
            authored_value = authored_stats.get(stat_key)
            if not _is_cited(stat):
                status, divergence = "not-comparable", None
                reason = "reference value uncited (NEEDS_CITATION)"
            elif authored_value is None:
                status, divergence = "not-comparable", None
                reason = "no authored value for this statistic"
            else:
                authored_float = _coerce_finite_float(
                    authored_value,
                    field="authored",
                    dataset_key=key,
                    stat_key=stat_key,
                )
                reference_float = _coerce_finite_float(
                    stat["value"],
                    field="reference",
                    dataset_key=key,
                    stat_key=stat_key,
                )
                status = "comparable"
                divergence = _relative_divergence(authored_float, reference_float)
                reason = None
            comparisons.append(
                {
                    "key": stat_key,
                    "unit": stat.get("unit"),
                    "reference_value": stat.get("value"),
                    "authored_value": authored_value,
                    "source_citation": stat.get("source_citation"),
                    "status": status,
                    "relative_divergence": divergence,
                    "reason": reason,
                }
            )
        verdict = classify_dataset(
            comparisons,
            sufficient_threshold=sufficient_threshold,
            material_threshold=material_threshold,
        )
        datasets_report.append(
            {
                "key": key,
                "name": dataset.get("name"),
                "verdict": verdict,
                "comparable_count": sum(1 for c in comparisons if c["status"] == "comparable"),
                "not_comparable_count": sum(
                    1 for c in comparisons if c["status"] == "not-comparable"
                ),
                "comparisons": comparisons,
            }
        )

    return {
        "schema_version": "external-prior-divergence-report.v1",
        "evidence_tier": "analysis_only",
        "thresholds": {
            "sufficient_relative_divergence": sufficient_threshold,
            "material_relative_divergence": material_threshold,
        },
        "claim_boundary": (
            "Compares authored priors to published summary statistics only. Not a realism claim, "
            "not planner ranking, and not a substitute for a raw-data pilot where one is genuinely "
            "needed. Uncited reference statistics are reported not-comparable, never as agreement."
        ),
        "datasets": datasets_report,
    }


def _load_yaml(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text()) or {}


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--reference", type=Path, required=True, help="External prior reference stats YAML."
    )
    parser.add_argument(
        "--authored", type=Path, required=True, help="Authored prior summary stats YAML."
    )
    parser.add_argument(
        "--out", type=Path, default=None, help="Write the divergence report JSON here."
    )
    parser.add_argument("--sufficient-threshold", type=float, default=0.25)
    parser.add_argument("--material-threshold", type=float, default=0.5)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the divergence CLI and print the report."""
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        report = compute_report(
            _load_yaml(args.reference),
            _load_yaml(args.authored),
            sufficient_threshold=args.sufficient_threshold,
            material_threshold=args.material_threshold,
        )
    except ValueError as exc:
        parser.error(str(exc))
    rendered = json.dumps(report, indent=2)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(rendered + "\n")
    print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
