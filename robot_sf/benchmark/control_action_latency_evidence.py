"""Promote control-action-latency sweep episode rows into durable evidence (issue #5034).

Issue #5034 asks to *execute* the control-action-latency fidelity sweep (wired by
PR #5026, scoped by parent #4977) and record whether the 0/100/300 ms-equivalent
delays change the configured safety metrics. PR #5061 merged the fail-closed
launch/readiness preflight plus a historical "blocked" packet. Once the native
campaign runs and emits raw episode rows under ``output/``, this module is the
**metric-evidence promotion** step that remains: it reads those rows, isolates the
``control_action_latency`` axis, reports the latency metadata plus success,
collision, and minimum-clearance metrics for each completed cell, and classifies
every non-native / fallback / degraded row as an **exclusion** rather than a
result (per the issue #691 benchmark fallback policy).

This module runs **no episodes** and makes **no benchmark / simulator-realism /
sim-to-real / paper-facing claim**. It is the deterministic promoter that turns a
raw campaign row file into a durable compact evidence bundle. It fails closed when
the raw rows do not cover the required action-latency step set (0, 1, 3) among
native result rows, so a partial or non-latency run cannot be silently promoted as
the latency sweep.

Episode row contract (the shape :func:`run_episode` in
``scripts/benchmark/run_fidelity_sensitivity_campaign.py`` emits)::

    {
        "axis": "control_action_latency",
        "variant": "control_action_latency__three_step_300ms",
        "variant_source_key": "three_step_300ms",
        "baseline_variant": false,
        "runtime_binding": "sim_config.action_latency_steps",
        "action_latency": {  # sim_config.action_latency_metadata()
            "configured_steps": 3,
            "configured_ms": null,
            "effective_steps": 3,
            "effective_ms": 300.0,
        },
        "planner": "baseline_social_force",
        "scenario_id": "...",
        "seed": 111,
        "success": true,
        "collision": false,
        "metrics": {"success_rate": 1.0, "collision_rate": 0.0, "min_clearance": 0.42},
    }
"""

from __future__ import annotations

import csv
import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from robot_sf.benchmark.control_action_latency_preflight import (
    AXIS_KEY,
    DECISION_READY,
    REQUIRED_LATENCY_STEPS,
    check_control_action_latency_axis,
)
from robot_sf.benchmark.identity.hash_utils import sha256_file

PROMOTION_SCHEMA_VERSION = "control-action-latency-sweep-evidence-promotion.v1"
ISSUE = 5034
PARENT_ISSUE = 4977
DEPENDENCY_PR = "#5026"

#: Metrics the issue #5034 evidence summary must report per latency cell.
REQUIRED_RESULT_METRICS: tuple[str, ...] = ("success_rate", "collision_rate", "min_clearance")

#: Execution modes that count as native benchmark-success rows (issue #691 policy).
NATIVE_EXECUTION_MODES: frozenset[str] = frozenset({"native", "adapter"})
AVAILABLE_AVAILABILITY_STATUSES: frozenset[str] = frozenset({"available"})

CLAIM_BOUNDARY = (
    "control-action-latency metric-evidence promotion only: reads raw fidelity-campaign episode "
    "rows, isolates the control_action_latency axis, and reports action-latency metadata plus "
    "success / collision / minimum-clearance metrics per native latency cell. It runs no episode "
    "and promotes no claim beyond the declared campaign evidence tier; it is not "
    "simulator-realism evidence, not sim-to-real evidence, and not paper-facing evidence."
)

EXCLUSION_POLICY = (
    "Per the issue #691 benchmark fallback policy, any row whose execution_mode is not "
    "native/adapter or whose availability_status is not available, plus any latency-axis row "
    "missing action_latency metadata, is recorded as an exclusion and never contributes to the "
    "result metrics. This keeps fallback/degraded execution out of the latency result set."
)


class LatencyEvidenceError(RuntimeError):
    """Raised when raw rows are not promotable as latency-sweep evidence (fail closed)."""


@dataclass(frozen=True)
class LatencyCell:
    """One classified control-action-latency episode row.

    ``classification`` is ``result`` for native, latency-metadata-bearing rows and
    ``exclusion`` for everything else (fallback / degraded / non-native / missing
    latency metadata). Only ``result`` cells contribute to the aggregate metrics.
    """

    planner: str
    latency_step: int | None
    latency_ms: float | None
    variant: str
    baseline_variant: bool
    seed: int
    scenario_id: str
    success_rate: float
    collision_rate: float
    min_clearance: float | None
    classification: str
    exclusion_reason: str | None
    execution_mode: str
    availability_status: str


def _coerce_int(value: Any) -> int | None:
    """Return an int for numeric values, rejecting bools and non-numbers."""
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and float(value).is_integer():
        return int(value)
    return None


def _row_execution_mode(row: Mapping[str, Any]) -> str:
    """Return a row's execution mode, defaulting to ``native`` when unset.

    The fail-closed campaign runner (``allow_fallback=False``) emits no fallback
    rows, so a completed row without an explicit marker is native. Rows that
    *do* carry a non-native marker are honored so the promoter degrades safely if
    a future runner path emits one.
    """
    mode = row.get("execution_mode")
    return str(mode) if isinstance(mode, str) and mode else "native"


def _row_availability_status(row: Mapping[str, Any]) -> str:
    """Return a row's availability status, defaulting to ``available`` when unset."""
    status = row.get("availability_status")
    return str(status) if isinstance(status, str) and status else "available"


def _latency_step_from_row(row: Mapping[str, Any]) -> int | None:
    """Extract the effective action-latency step for a row.

    Prefers the structured ``action_latency.effective_steps`` metadata; falls back
    to ``action_latency.configured_steps`` then to the row-level
    ``action_latency_steps`` marker so rows produced before the metadata dict
    landed still resolve.

    Returns:
        The effective action-latency step, or ``None`` when no marker resolves.
    """
    metadata = row.get("action_latency")
    if isinstance(metadata, Mapping):
        for key in ("effective_steps", "configured_steps"):
            step = _coerce_int(metadata.get(key))
            if step is not None:
                return step
    return _coerce_int(row.get("action_latency_steps"))


def _latency_ms_from_row(row: Mapping[str, Any]) -> float | None:
    """Return the effective action-latency milliseconds for a row, if recorded.

    Returns:
        The effective latency in ms, or ``None`` when not recorded.
    """
    metadata = row.get("action_latency")
    if isinstance(metadata, Mapping):
        for key in ("effective_ms", "configured_ms"):
            value = metadata.get(key)
            if isinstance(value, int | float) and not isinstance(value, bool):
                return float(value)
    return None


def classify_latency_row(row: Mapping[str, Any]) -> LatencyCell:
    """Classify one episode row as a latency ``result`` cell or an ``exclusion``.

    The row must belong to the ``control_action_latency`` axis. A row is a
    ``result`` only when it carries action-latency metadata AND its execution mode
    is native/adapter AND its availability status is available. Anything else
    (fallback / degraded / non-native / missing latency metadata) becomes an
    ``exclusion`` with a precise reason, so fallback execution can never enter the
    latency result set.

    Returns:
        The classified :class:`LatencyCell` for the row.
    """
    metrics = row.get("metrics")
    metric_map = metrics if isinstance(metrics, Mapping) else {}
    success_rate = float(metric_map.get("success_rate", 0.0) or 0.0)
    collision_rate = float(metric_map.get("collision_rate", 0.0) or 0.0)
    raw_min_clearance = metric_map.get("min_clearance")
    min_clearance = (
        float(raw_min_clearance)
        if isinstance(raw_min_clearance, int | float) and not isinstance(raw_min_clearance, bool)
        else None
    )

    execution_mode = _row_execution_mode(row)
    availability_status = _row_availability_status(row)
    latency_step = _latency_step_from_row(row)
    latency_ms = _latency_ms_from_row(row)

    reasons: list[str] = []
    if latency_step is None:
        reasons.append("missing_action_latency_metadata")
    if execution_mode not in NATIVE_EXECUTION_MODES:
        reasons.append(f"non_native_execution_mode:{execution_mode}")
    if availability_status not in AVAILABLE_AVAILABILITY_STATUSES:
        reasons.append(f"unavailable:{availability_status}")

    return LatencyCell(
        planner=str(row.get("planner") or "unknown"),
        latency_step=latency_step,
        latency_ms=latency_ms,
        variant=str(row.get("variant") or row.get("variant_source_key") or "unknown"),
        baseline_variant=bool(row.get("baseline_variant", False)),
        seed=int(row.get("seed") or 0),
        scenario_id=str(row.get("scenario_id") or "unknown"),
        success_rate=success_rate,
        collision_rate=collision_rate,
        min_clearance=min_clearance,
        classification="result" if not reasons else "exclusion",
        exclusion_reason="; ".join(reasons) if reasons else None,
        execution_mode=execution_mode,
        availability_status=availability_status,
    )


def extract_latency_cells(rows: Sequence[Mapping[str, Any]]) -> list[LatencyCell]:
    """Return latency cells for every ``control_action_latency`` axis row.

    Rows on other fidelity axes (clearance_radius, integration_timestep, ...) are
    ignored: they are not part of the latency sweep and must never contribute to
    its result metrics.
    """
    cells: list[LatencyCell] = []
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        if str(row.get("axis") or "") != AXIS_KEY:
            continue
        cells.append(classify_latency_row(row))
    return cells


def _mean(values: Sequence[float]) -> float | None:
    """Return the arithmetic mean of a non-empty sequence, else ``None``."""
    if not values:
        return None
    return float(sum(values) / len(values))


def aggregate_latency_metrics(cells: Sequence[LatencyCell]) -> list[dict[str, Any]]:
    """Aggregate result cells into per (planner, latency_step) metric rows.

    Exclusion cells never contribute. Each aggregate row reports the issue-#5034
    required metrics (success_rate, collision_rate, min_clearance) plus the cell
    count and the latency metadata (steps and ms). Rows are ordered by planner
    then latency step for deterministic output.

    Returns:
        One aggregate metric row per ``(planner, latency_step)`` result bucket.
    """
    buckets: dict[tuple[str, int], list[LatencyCell]] = {}
    for cell in cells:
        if cell.classification != "result" or cell.latency_step is None:
            continue
        buckets.setdefault((cell.planner, cell.latency_step), []).append(cell)

    aggregates: list[dict[str, Any]] = []
    for planner, latency_step in sorted(buckets, key=lambda item: (item[0][0], item[0][1])):
        bucket = buckets[(planner, latency_step)]
        min_clearances = [c.min_clearance for c in bucket if c.min_clearance is not None]
        aggregates.append(
            {
                "planner": planner,
                "action_latency_steps": latency_step,
                "action_latency_ms": _mean(
                    [c.latency_ms for c in bucket if c.latency_ms is not None]
                ),
                "cell_count": len(bucket),
                "seeds": sorted({c.seed for c in bucket}),
                "success_rate": _mean([c.success_rate for c in bucket]),
                "collision_rate": _mean([c.collision_rate for c in bucket]),
                "min_clearance": _mean(min_clearances) if min_clearances else None,
            }
        )
    return aggregates


def _required_step_coverage(
    aggregates: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Return which required latency steps are covered by native result aggregates.

    Returns:
        Coverage dict with ``required_latency_steps``, ``observed_latency_steps``,
        ``missing_latency_steps``, and ``coverage_complete``.
    """
    observed_steps = sorted({int(row["action_latency_steps"]) for row in aggregates})
    missing = [step for step in REQUIRED_LATENCY_STEPS if step not in observed_steps]
    return {
        "required_latency_steps": list(REQUIRED_LATENCY_STEPS),
        "observed_latency_steps": observed_steps,
        "missing_latency_steps": missing,
        "coverage_complete": not missing,
    }


def _exclusion_summary(cells: Sequence[LatencyCell]) -> dict[str, Any]:
    """Return a summary of excluded latency rows and their reasons (#691 discipline).

    Returns:
        Summary dict with ``excluded_row_count``, ``reason_counts``, and a small
        ``sample_exclusions`` list.
    """
    exclusions = [cell for cell in cells if cell.classification == "exclusion"]
    reason_counts: dict[str, int] = {}
    for exclusion in exclusions:
        for reason in (exclusion.exclusion_reason or "unknown").split("; "):
            reason_counts[reason] = reason_counts.get(reason, 0) + 1
    return {
        "excluded_row_count": len(exclusions),
        "reason_counts": dict(sorted(reason_counts.items())),
        "sample_exclusions": [asdict(cell) for cell in exclusions[:5]],
    }


def build_latency_evidence(
    rows: Sequence[Mapping[str, Any]],
    *,
    config: Mapping[str, Any],
    config_path: str,
    git_head: str,
    date: str | None,
    raw_rows_path: str,
) -> dict[str, Any]:
    """Build the durable compact control-action-latency evidence packet.

    Fails closed (:class:`LatencyEvidenceError`) when the latency preflight is not
    ``ready`` or when native result rows do not cover every required action-latency
    step (0, 1, 3), so a partial or non-latency run cannot be promoted as the
    latency sweep.

    Args:
        rows: Raw episode rows from the fidelity campaign runner.
        config: Raw fidelity-sensitivity study config mapping.
        config_path: Repo-relative config path recorded for provenance.
        git_head: Git head recorded for provenance.
        date: ISO date string recorded for provenance.
        raw_rows_path: Repo-relative path of the source raw row file.

    Returns:
        JSON-serializable evidence packet with per-cell and aggregate latency
        metrics plus the exclusion classification.
    """
    preflight = check_control_action_latency_axis(
        config, config_path=config_path, git_head=git_head, date=date
    )
    if preflight["decision"] != DECISION_READY:
        raise LatencyEvidenceError(
            "control-action-latency preflight is not ready; refusing to promote. Blockers: "
            + "; ".join(preflight.get("blockers") or ["unknown"])
        )

    cells = extract_latency_cells(rows)
    if not cells:
        raise LatencyEvidenceError(
            f"raw rows contain no '{AXIS_KEY}' axis rows; cannot promote as the latency sweep"
        )

    aggregates = aggregate_latency_metrics(cells)
    coverage = _required_step_coverage(aggregates)
    if not coverage["coverage_complete"]:
        raise LatencyEvidenceError(
            "native latency result rows do not cover every required action-latency step. "
            f"required={list(REQUIRED_LATENCY_STEPS)} "
            f"observed={coverage['observed_latency_steps']} "
            f"missing={coverage['missing_latency_steps']}"
        )

    result_cells = [cell for cell in cells if cell.classification == "result"]
    return {
        "schema_version": PROMOTION_SCHEMA_VERSION,
        "issue": ISSUE,
        "parent_issue": PARENT_ISSUE,
        "depends_on_pr": DEPENDENCY_PR,
        "date": date,
        "git_head": git_head,
        "config_path": config_path,
        "raw_rows_path": raw_rows_path,
        "claim_boundary": CLAIM_BOUNDARY,
        "exclusion_policy": EXCLUSION_POLICY,
        "preflight_decision": preflight["decision"],
        "preflight_axis_present": preflight["axis_present"],
        "preflight_observed_latency_steps": preflight["observed_latency_steps"],
        "required_result_metrics": list(REQUIRED_RESULT_METRICS),
        "latency_coverage": coverage,
        "scope": {
            "latency_row_count": len(cells),
            "result_row_count": len(result_cells),
            "excluded_row_count": len(cells) - len(result_cells),
            "planners": sorted({cell.planner for cell in result_cells}),
            "seeds": sorted({cell.seed for cell in result_cells}),
            "scenario_ids": sorted({cell.scenario_id for cell in result_cells}),
        },
        "aggregate_metrics": aggregates,
        "per_cell_metrics": [asdict(cell) for cell in cells],
        "exclusions": _exclusion_summary(cells),
        "artifact_policy": (
            "Compact promotion artifacts are tracked here. Raw episode JSONL remains ignored under "
            "output/ and needs a durable external storage pointer before paper-facing use."
        ),
    }


def _format_markdown(packet: Mapping[str, Any]) -> str:
    """Return a compact human-readable evidence Markdown string.

    Returns:
        Markdown rendering of the promotion packet.
    """
    scope = packet["scope"]
    coverage = packet["latency_coverage"]
    exclusions = packet["exclusions"]
    lines = [
        f"# Issue #{ISSUE} Control-action-latency sweep evidence {packet.get('date') or ''}",
        "",
        "Plain-language summary: this bundle promotes raw fidelity-campaign episode rows into a "
        "compact control-action-latency evidence summary. It reports the 0/100/300 ms-equivalent "
        "delay cells' success, collision, and minimum-clearance metrics and excludes any "
        "fallback/degraded/non-native rows. It is not paper-facing evidence.",
        "",
        f"- Schema: `{packet['schema_version']}`",
        f"- Git head: `{packet.get('git_head')}`",
        f"- Raw rows: `{packet.get('raw_rows_path')}`",
        f"- Preflight decision: `{packet['preflight_decision']}`",
        f"- Claim boundary: {packet['claim_boundary']}",
        "",
        "## Scope",
        "",
        f"- Latency rows: `{scope['latency_row_count']}` (results `{scope['result_row_count']}`, "
        f"excluded `{scope['excluded_row_count']}`)",
        f"- Planners: `{', '.join(scope['planners']) or 'none'}`",
        f"- Seeds: `{', '.join(str(seed) for seed in scope['seeds']) or 'none'}`",
        f"- Latency-step coverage: required `{coverage['required_latency_steps']}`, "
        f"observed `{coverage['observed_latency_steps']}`, "
        f"missing `{coverage['missing_latency_steps'] or 'none'}`",
        "",
        "## Aggregate metrics per latency cell",
        "",
        "| Planner | Latency steps | Latency ms | Cells | Success | Collision | Min clearance |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in packet["aggregate_metrics"]:
        ms = row.get("action_latency_ms")
        min_clearance = row.get("min_clearance")
        lines.append(
            f"| `{row['planner']}` | {row['action_latency_steps']} | "
            f"{ms if ms is not None else 'NA'} | {row['cell_count']} | "
            f"{row['success_rate']:.6g} | {row['collision_rate']:.6g} | "
            f"{min_clearance if min_clearance is not None else 'NA'} |"
        )
    lines.extend(
        [
            "",
            "## Exclusions (fallback / degraded / non-native)",
            "",
            f"- Excluded rows: `{exclusions['excluded_row_count']}`",
            f"- Reasons: `{exclusions['reason_counts'] or 'none'}`",
            "",
            "Per the issue #691 benchmark fallback policy, excluded rows never contribute to "
            "the result metrics above.",
            "",
            "## Files",
            "",
            "- `summary.json`: full promotion packet (aggregate + per-cell + exclusions).",
            "- `per_cell_metrics.csv`: compact per-cell latency metrics table.",
            "- `manifest.sha256`: checksums for promoted compact artifacts.",
            "- `README.md`: this human-readable summary.",
            "",
        ]
    )
    return "\n".join(lines)


def write_latency_evidence(packet: Mapping[str, Any], evidence_dir: str | Path) -> list[Path]:
    """Write the durable compact latency evidence bundle.

    Writes ``summary.json``, ``per_cell_metrics.csv``, ``README.md``, and
    ``manifest.sha256`` into ``evidence_dir``.

    Returns:
        The list of written artifact paths.
    """
    out = Path(evidence_dir)
    out.mkdir(parents=True, exist_ok=True)

    summary_path = out / "summary.json"
    summary_path.write_text(
        json.dumps(packet, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    csv_path = out / "per_cell_metrics.csv"
    fieldnames = [
        "planner",
        "latency_step",
        "latency_ms",
        "variant",
        "baseline_variant",
        "seed",
        "scenario_id",
        "success_rate",
        "collision_rate",
        "min_clearance",
        "classification",
        "exclusion_reason",
        "execution_mode",
        "availability_status",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for cell in packet["per_cell_metrics"]:
            writer.writerow(
                {
                    "planner": cell["planner"],
                    "latency_step": cell["latency_step"]
                    if cell["latency_step"] is not None
                    else "",
                    "latency_ms": cell["latency_ms"] if cell["latency_ms"] is not None else "",
                    "variant": cell["variant"],
                    "baseline_variant": cell["baseline_variant"],
                    "seed": cell["seed"],
                    "scenario_id": cell["scenario_id"],
                    "success_rate": cell["success_rate"],
                    "collision_rate": cell["collision_rate"],
                    "min_clearance": cell["min_clearance"]
                    if cell["min_clearance"] is not None
                    else "",
                    "classification": cell["classification"],
                    "exclusion_reason": cell["exclusion_reason"] or "",
                    "execution_mode": cell["execution_mode"],
                    "availability_status": cell["availability_status"],
                }
            )

    readme_path = out / "README.md"
    readme_path.write_text(_format_markdown(packet), encoding="utf-8")

    copied = [summary_path, csv_path, readme_path]
    manifest_path = out / "manifest.sha256"
    manifest_path.write_text(
        "\n".join(f"{sha256_file(path)}  {path.name}" for path in copied) + "\n",
        encoding="utf-8",
    )
    copied.append(manifest_path)
    return copied


def load_latency_rows(raw_rows_path: str | Path) -> list[dict[str, Any]]:
    """Return newline-delimited JSON episode rows emitted by the campaign runner.

    Returns:
        The list of parsed row mappings.
    """
    path = Path(raw_rows_path)
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                row = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise LatencyEvidenceError(
                    f"raw rows file {path} has invalid JSON on line {line_number}: {exc}"
                ) from exc
            if isinstance(row, dict):
                rows.append(row)
    return rows


def promote_latency_evidence(
    raw_rows_path: str | Path,
    evidence_dir: str | Path,
    *,
    config: Mapping[str, Any],
    config_path: str,
    git_head: str,
    date: str | None,
) -> dict[str, Any]:
    """Load raw rows, build the latency evidence packet, and write the bundle.

    Convenience entry point that combines :func:`load_latency_rows`,
    :func:`build_latency_evidence`, and :func:`write_latency_evidence`. Raises
    :class:`LatencyEvidenceError` (fail closed) when the rows cannot be promoted
    as the latency sweep.

    Returns:
        A promotion result dict with status, evidence dir, written files, and
        coverage summary.
    """
    rows = load_latency_rows(raw_rows_path)
    packet = build_latency_evidence(
        rows,
        config=config,
        config_path=config_path,
        git_head=git_head,
        date=date,
        raw_rows_path=str(raw_rows_path),
    )
    written = write_latency_evidence(packet, evidence_dir)
    return {
        "schema_version": PROMOTION_SCHEMA_VERSION,
        "status": "promoted",
        "issue": ISSUE,
        "evidence_dir": str(evidence_dir),
        "written_files": [str(path) for path in written],
        "result_row_count": packet["scope"]["result_row_count"],
        "excluded_row_count": packet["scope"]["excluded_row_count"],
        "latency_coverage": packet["latency_coverage"],
        "claim_boundary": CLAIM_BOUNDARY,
    }
