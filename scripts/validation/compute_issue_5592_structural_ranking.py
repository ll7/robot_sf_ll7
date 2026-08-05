#!/usr/bin/env python3
"""Issue #5592 structural-class ranking metric: ``constraints_first_structural_rank``.

This module implements the metric named but never defined in the pre-registration
packet (`configs/benchmarks/issue_5592_cross_matrix_preregistration.yaml`,
``comparison_contract.metric: constraints_first_structural_rank``). It converts
per-planner episode aggregates (the output of the frozen-contract paid campaign once
it exists) into an independent ranking of the four structural planner classes, one
ranking per matrix. Ranks use standard competition ("1224") semantics: classes with
exactly equal score tuples share the same rank and the following rank is skipped, so
the output is a unique ``1..4`` permutation only when no exact tie occurs. The
ranking CSV it writes carries the preregistered 12-planner roster signature and is
consumed directly by
``scripts/validation/build_issue_5592_cross_matrix_agreement.py``.

The metric is pure CPU aggregation: it never runs a campaign, Slurm job, or training
run. It is the artifact-first gap-filler between campaign episode rows and the
cross-matrix agreement table.

Invalid inputs remain fail-closed, but the CLI emits a prominent warning with the exact
reason and a required remediation checklist instead of an opaque one-line error.

Scoring semantics (constraints-first ordering): a structural class is ranked better
when its planners complete routes more often (higher success rate), collide less
(lower collision event rate), cause fewer near-miss events, time out less, and
achieve higher social-navigation quality (SNQI). The score tuple orders classes so
that rank 1 is the best-performing structural class for the matrix under test.
Exact tuple equality is a true tie: tied classes receive the same shared rank and
are serialized in stable structural-class identity order without implying that one
tied class outperformed the other.

Metric cells are parsed from their decimal representation and compared with exact rational
means, so mathematically equal decimal inputs cannot become rank differences through binary
floating-point summation. Unit-interval rates are validated, nested execution metadata is
accepted only for status ``ok`` and native execution, and SNQI is either complete for the
frozen roster or absent for the frozen roster.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections.abc import Mapping
from decimal import Decimal, InvalidOperation
from fractions import Fraction
from pathlib import Path
from typing import TYPE_CHECKING, Any

import yaml

from scripts.validation.issue_5592_diagnostics import format_fail_closed_warning

if TYPE_CHECKING:
    from collections.abc import Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PACKET = REPO_ROOT / "configs/benchmarks/issue_5592_cross_matrix_preregistration.yaml"
SCHEMA_VERSION = "issue_5592_cross_matrix_preregistration.v1"

STRUCTURAL_CLASS_ORDER = [
    "constraint_first_hybrid",
    "learned_policy",
    "predictive",
    "baseline_reactive",
]

RANKING_COLUMNS = ["structural_class", "rank", "roster_signature"]
ROSTER_SIGNATURE_COLUMN = "roster_signature"
# Core per-planner metric fields every episode-aggregate row must carry so the
# ranking cannot silently impute a best-case (0.0 collision/timeout) value for a
# missing safety metric. ``snqi_mean`` remains optional for the full roster and
# is handled as an all-present or all-absent tie-breaker.
REQUIRED_METRIC_FIELDS = (
    "success_rate",
    "collision_event_rate",
    "near_miss_event_rate",
    "timeout_rate",
)
RATE_FIELDS = frozenset(REQUIRED_METRIC_FIELDS)
INVALID_STATUS_VALUES = frozenset(
    {
        "error",
        "failed",
        "failure",
        "fallback",
        "degraded",
        "unavailable",
        "not_available",
        "partial-failure",
        "unknown",
        "malformed",
    }
)


class RankingMetricError(ValueError):
    """Raised when issue #5592 ranking inputs or the pre-registration are malformed."""


def _load_yaml(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RankingMetricError(f"{path} must contain a YAML mapping")
    return payload


def _roster_signature(packet: Mapping[str, Any]) -> str:
    """Return the deterministic SHA-256 signature of the preregistered planner roster."""
    roster = packet.get("planner_roster")
    if not isinstance(roster, dict):
        raise RankingMetricError("packet.planner_roster must be a mapping")
    structural_classes = roster.get("structural_classes")
    if not isinstance(structural_classes, dict):
        raise RankingMetricError("packet.planner_roster.structural_classes must be a mapping")
    if set(structural_classes) != set(STRUCTURAL_CLASS_ORDER):
        raise RankingMetricError("packet planner roster structural classes mismatch")

    canonical: dict[str, list[str]] = {}
    planners: list[str] = []
    for structural_class in STRUCTURAL_CLASS_ORDER:
        class_planners = structural_classes.get(structural_class)
        if not isinstance(class_planners, list) or not class_planners:
            raise RankingMetricError(
                f"planner roster for {structural_class!r} must be a non-empty list"
            )
        normalized = [str(planner).strip() for planner in class_planners]
        if any(not planner for planner in normalized):
            raise RankingMetricError(
                f"planner roster for {structural_class!r} contains an empty planner"
            )
        canonical[structural_class] = normalized
        planners.extend(normalized)
    if len(planners) != len(set(planners)):
        raise RankingMetricError("packet planner roster contains duplicate planner keys")
    serialized = json.dumps(canonical, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _load_packet(packet_path: Path) -> dict[str, Any]:
    packet = _load_yaml(packet_path)
    if packet.get("schema_version") != SCHEMA_VERSION:
        raise RankingMetricError("packet schema_version mismatch")
    if packet.get("issue") != 5592:
        raise RankingMetricError("packet.issue must be 5592")
    if packet.get("status") != "pre_registered":
        raise RankingMetricError("packet.status must be pre_registered")
    return packet


def _to_decimal(value: Any) -> Decimal | None:
    """Parse one finite decimal value without introducing binary float error."""
    if value is None or value == "" or isinstance(value, bool):
        return None
    try:
        parsed = Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError):
        return None
    return parsed if parsed.is_finite() else None


def _normalise_status(value: Any) -> str:
    """Return a normalized status token for eligibility checks."""
    return str(value or "").strip().lower()


def _metadata_ineligibility_reason(metadata: Any) -> str | None:
    """Return an ineligibility reason for nested algorithm metadata, if any."""
    if not isinstance(metadata, Mapping):
        return "algorithm_metadata is malformed"
    metadata_status = _normalise_status(metadata.get("status"))
    if metadata_status != "ok":
        return f"algorithm_metadata.status={metadata_status!r}"
    kinematics = metadata.get("planner_kinematics")
    if not isinstance(kinematics, Mapping):
        return "algorithm_metadata.planner_kinematics is missing or malformed"
    execution_mode = _normalise_status(kinematics.get("execution_mode"))
    if execution_mode != "native":
        return f"algorithm_metadata.planner_kinematics.execution_mode={execution_mode!r}"
    availability_status = _normalise_status(metadata.get("availability_status"))
    if availability_status and availability_status != "available":
        return f"algorithm_metadata.availability_status={availability_status!r}"
    for key in ("readiness_status", "preflight_status"):
        value = _normalise_status(metadata.get(key))
        if value in INVALID_STATUS_VALUES:
            return f"algorithm_metadata.{key}={value!r}"
    return None


def _ineligible_execution_reason(row: Mapping[str, Any]) -> str | None:
    """Return the reason a row cannot enter the native ranking, if any."""
    for key in (
        "status",
        "run_status",
        "planner_status",
        "availability_status",
        "execution_mode",
        "readiness_status",
    ):
        value = _normalise_status(row.get(key))
        if value in INVALID_STATUS_VALUES:
            return f"{key}={value!r}"
    return (
        _metadata_ineligibility_reason(row["algorithm_metadata"])
        if "algorithm_metadata" in row
        else None
    )


def _score(
    structural_class: str,
    class_aggregates: Sequence[Mapping[str, Any]],
) -> tuple[Fraction, Fraction, Fraction, Fraction, Fraction]:
    """Aggregate a structural class across its planners into a comparable score tuple.

    Lower collision/near-miss/timeout and higher success/SNQI rank better. Returns a
    5-tuple of exact rational values ordered for descending-quality sort:
        (-success_rate, collision_event_rate, near_miss_event_rate, timeout_rate, -snqi_mean).
    """
    success = [a["success_rate"] for a in class_aggregates]
    collision = [a["collision_event_rate"] for a in class_aggregates]
    near_miss = [a["near_miss_event_rate"] for a in class_aggregates]
    timeout = [a["timeout_rate"] for a in class_aggregates]
    snqi = [a.get("snqi_mean") for a in class_aggregates]
    present_snqi = [value for value in snqi if value is not None]
    if present_snqi and len(present_snqi) != len(snqi):
        raise RankingMetricError(
            f"incomplete snqi_mean coverage for structural class {structural_class!r}; "
            "SNQI must be present for every planner in a class or absent for the full roster"
        )

    def _mean(values: Sequence[Decimal]) -> Fraction:
        return sum((Fraction(value) for value in values), Fraction(0, 1)) / len(values)

    snqi_mean = _mean(present_snqi) if present_snqi else Fraction(0, 1)

    return (
        -_mean(success),
        _mean(collision),
        _mean(near_miss),
        _mean(timeout),
        -snqi_mean,
    )


def _parse_metric_aggregate(row: Mapping[str, Any], planner_key: str) -> dict[str, Decimal | None]:
    """Parse one planner's metrics and reject malformed, non-finite, or invalid rates."""
    parsed_metrics: dict[str, Decimal | None] = {}
    for field in REQUIRED_METRIC_FIELDS:
        parsed = _to_decimal(row[field])
        if parsed is None:
            raise RankingMetricError(
                f"invalid or non-finite metric field {field!r} for {planner_key!r}"
            )
        if field in RATE_FIELDS and not Decimal("0") <= parsed <= Decimal("1"):
            raise RankingMetricError(
                f"metric field {field!r} for {planner_key!r} must be in [0, 1], got {parsed}"
            )
        parsed_metrics[field] = parsed

    raw_snqi = row.get("snqi_mean")
    parsed_snqi = _to_decimal(raw_snqi)
    if raw_snqi not in (None, "") and parsed_snqi is None:
        raise RankingMetricError(
            f"invalid or non-finite metric field 'snqi_mean' for {planner_key!r}"
        )
    return {**parsed_metrics, "snqi_mean": parsed_snqi}


def compute_structural_ranking(
    episode_rows: Sequence[Mapping[str, Any]],
    *,
    planner_to_class: Mapping[str, str],
) -> dict[str, int]:
    """Compute a structural-class ranking for one matrix from episode rows.

    Args:
        episode_rows: Iterable of per-planner aggregate records, each carrying
            ``planner_key`` (or ``planner``), ``success_rate``,
            ``collision_event_rate``, ``near_miss_event_rate``, ``timeout_rate``,
            and optional ``snqi_mean``.
        planner_to_class: Mapping from planner key to one of the four structural
            class names.

    Returns:
        Mapping from structural class to an integer rank (1 = best) with standard
        competition ("1224") semantics: a class's rank is one plus the number of
        classes with a strictly better score tuple. Classes whose score tuples are
        exactly equal share the same rank and the next rank is skipped, so the
        ranks form a unique ``1..4`` permutation only when no exact tie occurs.

    Raises:
        RankingMetricError: If a row is ineligible, a planner key is unknown, missing,
            or duplicated, or a required metric field is missing or invalid.
    """
    expected_planner_keys = set(planner_to_class)
    expected_classes = set(STRUCTURAL_CLASS_ORDER)
    observed_classes = set(planner_to_class.values())
    if observed_classes != expected_classes:
        raise RankingMetricError(
            "planner_to_class must cover exactly the four structural classes; "
            f"missing={sorted(expected_classes - observed_classes)}, "
            f"extra={sorted(observed_classes - expected_classes)}"
        )
    observed_planner_keys, by_class = _collect_class_aggregates(
        episode_rows,
        planner_to_class=planner_to_class,
    )

    if observed_planner_keys != expected_planner_keys:
        missing_planner_keys = sorted(expected_planner_keys - observed_planner_keys)
        extra_planner_keys = sorted(observed_planner_keys - expected_planner_keys)
        raise RankingMetricError(
            "matrix aggregate does not cover the frozen planner roster exactly once; "
            f"missing planner key(s): {missing_planner_keys}; "
            f"extra planner key(s): {extra_planner_keys}"
        )

    snqi_values = [aggregate.get("snqi_mean") for rows in by_class.values() for aggregate in rows]
    present_snqi_count = sum(value is not None for value in snqi_values)
    if present_snqi_count not in (0, len(snqi_values)):
        raise RankingMetricError(
            "incomplete snqi_mean coverage across the frozen planner roster; "
            "SNQI must be present for every planner or absent for every planner"
        )

    class_scores: dict[str, tuple[Fraction, Fraction, Fraction, Fraction, Fraction]] = {}
    for klass, rows in by_class.items():
        class_scores[klass] = _score(klass, rows)

    # Standard competition ("1224") ranking: exact score-tuple equality is a true
    # tie, so tied classes share a rank instead of being strictly ordered by the
    # stable structural-class identity order.
    return {
        klass: 1
        + sum(1 for other in STRUCTURAL_CLASS_ORDER if class_scores[other] < class_scores[klass])
        for klass in STRUCTURAL_CLASS_ORDER
    }


def _collect_class_aggregates(
    episode_rows: Sequence[Mapping[str, Any]],
    *,
    planner_to_class: Mapping[str, str],
) -> tuple[set[str], dict[str, list[dict[str, Any]]]]:
    """Validate and group per-planner aggregates by structural class."""
    observed_planner_keys: set[str] = set()
    by_class: dict[str, list[dict[str, Any]]] = {klass: [] for klass in STRUCTURAL_CLASS_ORDER}
    for row in episode_rows:
        ineligible_reason = _ineligible_execution_reason(row)
        if ineligible_reason is not None:
            label = row.get("planner_key") or row.get("planner") or "<unknown>"
            raise RankingMetricError(
                f"ineligible execution row excluded from ranking: {label} ({ineligible_reason})"
            )
        planner_key = row.get("planner_key") or row.get("planner")
        if planner_key is None:
            raise RankingMetricError("episode row missing planner_key/planner")
        normalized_planner_key = str(planner_key).strip()
        klass = planner_to_class.get(normalized_planner_key)
        if klass is None:
            raise RankingMetricError(f"planner not in preregistered roster: {planner_key!r}")
        if normalized_planner_key in observed_planner_keys:
            raise RankingMetricError(
                f"duplicate planner key in matrix aggregate: {normalized_planner_key!r}"
            )
        observed_planner_keys.add(normalized_planner_key)

        missing_fields = [field for field in REQUIRED_METRIC_FIELDS if row.get(field) in (None, "")]
        if missing_fields:
            raise RankingMetricError(
                f"episode row for {planner_key!r} missing required metric field(s): "
                f"{missing_fields}"
            )
        by_class[klass].append(_parse_metric_aggregate(row, normalized_planner_key))
    return observed_planner_keys, by_class


def _planner_to_class(packet: Mapping[str, Any]) -> dict[str, str]:
    roster = packet.get("planner_roster", {})
    structural_classes = roster.get("structural_classes", {})
    mapping: dict[str, str] = {}
    for klass, planners in structural_classes.items():
        for planner in planners:
            mapping[str(planner).strip()] = str(klass)
    return mapping


def _read_episode_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return [dict(row) for row in reader]


def _write_ranking_csv(path: Path, ranking: Mapping[str, int], *, roster_signature: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=RANKING_COLUMNS, lineterminator="\n")
        writer.writeheader()
        for klass in STRUCTURAL_CLASS_ORDER:
            writer.writerow(
                {
                    "structural_class": klass,
                    "rank": ranking[klass],
                    ROSTER_SIGNATURE_COLUMN: roster_signature,
                }
            )


def build_ranking_for_matrix(
    *,
    packet_path: Path,
    episode_rows_path: Path,
    output_path: Path,
) -> dict[str, int]:
    """Compute and write the structural-class ranking CSV for one matrix.

    Reads the per-planner episode aggregates, derives the constraints-first
    structural ranking, and writes it to ``output_path`` with the frozen roster
    signature. Exact-score ties are represented explicitly as shared (competition)
    ranks in the ``rank`` column; rows are serialized in stable structural-class
    identity order. Returns the ranking mapping.
    """
    packet = _load_packet(packet_path)
    roster_signature = _roster_signature(packet)
    planner_to_class = _planner_to_class(packet)
    rows = _read_episode_rows(episode_rows_path)
    ranking = compute_structural_ranking(rows, planner_to_class=planner_to_class)
    _write_ranking_csv(output_path, ranking, roster_signature=roster_signature)
    return ranking


def _parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--packet", type=Path, default=DEFAULT_PACKET)
    parser.add_argument(
        "--episode-rows",
        type=Path,
        required=True,
        help="Per-planner episode aggregate CSV for one matrix",
    )
    parser.add_argument(
        "--output", type=Path, required=True, help="Output structural-class ranking CSV path"
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point for the issue #5592 structural ranking metric."""
    args = _parse_args(argv or sys.argv[1:])
    try:
        ranking = build_ranking_for_matrix(
            packet_path=args.packet,
            episode_rows_path=args.episode_rows,
            output_path=args.output,
        )
    except (OSError, RankingMetricError) as exc:
        print(
            format_fail_closed_warning(
                tool="compute_issue_5592_structural_ranking",
                reason=str(exc),
                input_paths=[args.packet, args.episode_rows],
                output_path=args.output,
            ),
            file=sys.stderr,
        )
        return 2
    print(f"matrix_ranking: {args.output}")
    for klass in STRUCTURAL_CLASS_ORDER:
        print(f"  {klass}: rank {ranking[klass]}")
    for rank in sorted(set(ranking.values())):
        tied = [klass for klass in STRUCTURAL_CLASS_ORDER if ranking[klass] == rank]
        if len(tied) > 1:
            print(f"  tie: {', '.join(tied)} share rank {rank} (exact-equal score tuples)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
