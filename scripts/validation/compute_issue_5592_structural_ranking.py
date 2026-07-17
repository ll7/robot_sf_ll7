#!/usr/bin/env python3
"""Issue #5592 structural-class ranking metric: ``constraints_first_structural_rank``.

This module implements the metric named but never defined in the pre-registration
packet (`configs/benchmarks/issue_5592_cross_matrix_preregistration.yaml`,
``comparison_contract.metric: constraints_first_structural_rank``). It converts
per-planner episode aggregates (the output of the frozen-contract paid campaign once
it exists) into an independent ``1..4`` ranking of the four structural planner
classes, one ranking per matrix. The ranking CSV it writes carries the preregistered
12-planner roster signature and is consumed directly by
``scripts/validation/build_issue_5592_cross_matrix_agreement.py``.

The metric is pure CPU aggregation: it never runs a campaign, Slurm job, or training
run. It is the artifact-first gap-filler between campaign episode rows and the
cross-matrix agreement table.

Scoring semantics (constraints-first ordering): a structural class is ranked better
when its planners complete routes more often (higher success rate), collide less
(lower collision event rate), cause fewer near-miss events, time out less, and
achieve higher social-navigation quality (SNQI). The score tuple orders classes so
that rank 1 is the best-performing structural class for the matrix under test.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

import yaml

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

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
# missing safety metric. ``snqi_mean`` remains optional (handled in ``_score``).
REQUIRED_METRIC_FIELDS = (
    "success_rate",
    "collision_event_rate",
    "near_miss_event_rate",
    "timeout_rate",
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


def _to_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _is_fallback_or_degraded(row: Mapping[str, Any]) -> bool:
    """Reject fallback/degraded/unavailable planner rows so they cannot enter the ranking."""
    fallback_statuses = {"fallback", "degraded", "unavailable", "not_available", "unavailable_mode"}
    for key in ("status", "run_status", "planner_status", "availability_status", "execution_mode"):
        value = row.get(key)
        if value is not None and str(value).strip().lower() in fallback_statuses:
            return True
    return False


def _score(
    class_aggregates: Sequence[Mapping[str, Any]],
) -> tuple[float, float, float, float, float]:
    """Aggregate a structural class across its planners into a comparable score tuple.

    Lower collision/near-miss/timeout and higher success/SNQI rank better. Returns a
    5-tuple ordered for descending-quality sort:
        (-success_rate, collision_event_rate, near_miss_event_rate, timeout_rate, -snqi_mean).
    """
    success = [float(a["success_rate"]) for a in class_aggregates]
    collision = [float(a["collision_event_rate"]) for a in class_aggregates]
    near_miss = [float(a["near_miss_event_rate"]) for a in class_aggregates]
    timeout = [float(a["timeout_rate"]) for a in class_aggregates]
    snqi = [_to_float(a.get("snqi_mean")) for a in class_aggregates]
    snqi_mean = sum(v for v in snqi if v is not None) / max(
        1, sum(1 for v in snqi if v is not None)
    )

    def _mean(values: Sequence[float]) -> float:
        return sum(values) / max(1, len(values))

    return (
        -_mean(success),
        _mean(collision),
        _mean(near_miss),
        _mean(timeout),
        -snqi_mean,
    )


def compute_structural_ranking(
    episode_rows: Sequence[Mapping[str, Any]],
    *,
    planner_to_class: Mapping[str, str],
) -> dict[str, int]:
    """Compute a ``1..4`` structural-class ranking for one matrix from episode rows.

    Args:
        episode_rows: Iterable of per-planner aggregate records, each carrying
            ``planner_key`` (or ``planner``), ``success_rate``,
            ``collision_event_rate``, ``near_miss_event_rate``, ``timeout_rate``,
            and optional ``snqi_mean``.
        planner_to_class: Mapping from planner key to one of the four structural
            class names.

    Returns:
        Mapping from structural class to a unique integer rank in ``1..4`` (1 = best).

    Raises:
        RankingMetricError: If a row is a fallback/degraded execution, a planner key
            is unknown, a required metric field is missing, or a structural class has
            no eligible rows.
    """
    by_class: dict[str, list[dict[str, Any]]] = {klass: [] for klass in STRUCTURAL_CLASS_ORDER}
    for row in episode_rows:
        if _is_fallback_or_degraded(row):
            label = row.get("planner_key") or row.get("planner") or "<unknown>"
            raise RankingMetricError(f"fallback/degraded row excluded from ranking: {label}")
        planner_key = row.get("planner_key") or row.get("planner")
        if planner_key is None:
            raise RankingMetricError("episode row missing planner_key/planner")
        missing_fields = [field for field in REQUIRED_METRIC_FIELDS if row.get(field) in (None, "")]
        if missing_fields:
            raise RankingMetricError(
                f"episode row for {planner_key!r} missing required metric field(s): "
                f"{missing_fields}"
            )
        klass = planner_to_class.get(str(planner_key).strip())
        if klass is None:
            raise RankingMetricError(f"planner not in preregistered roster: {planner_key!r}")
        try:
            aggregate = {
                "success_rate": float(row["success_rate"]),
                "collision_event_rate": float(row["collision_event_rate"]),
                "near_miss_event_rate": float(row["near_miss_event_rate"]),
                "timeout_rate": float(row["timeout_rate"]),
                "snqi_mean": row.get("snqi_mean"),
            }
        except (TypeError, ValueError) as exc:
            raise RankingMetricError(f"invalid metric row for {planner_key!r}: {exc}") from exc
        by_class[klass].append(aggregate)

    missing_classes = [klass for klass, rows in by_class.items() if not rows]
    if missing_classes:
        raise RankingMetricError(
            f"structural class(es) have no eligible rows: {sorted(missing_classes)}"
        )

    class_scores: dict[str, tuple[float, float, float, float, float]] = {}
    for klass, rows in by_class.items():
        class_scores[klass] = _score(rows)

    ranked = sorted(STRUCTURAL_CLASS_ORDER, key=lambda klass: class_scores[klass])
    return {klass: rank for rank, klass in enumerate(ranked, start=1)}


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
    signature. Returns the ranking mapping.
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
    except RankingMetricError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(f"matrix_ranking: {args.output}")
    for klass in STRUCTURAL_CLASS_ORDER:
        print(f"  {klass}: rank {ranking[klass]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
