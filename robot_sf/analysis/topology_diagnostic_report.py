"""Topology diagnostic report snapshot.

Aggregates topology-hypothesis JSONL traces into a compact Markdown/JSON summary
covering selected hypotheses, near-parity gate reasons, reuse-penalty activations,
route-progress deltas, top regressions, and top unchanged cases.

This report is diagnostic/reviewability support, not planner-promotion evidence.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path  # noqa: TC003
from typing import Any

CLAIM_BOUNDARY = "diagnostic_only_not_benchmark_success"


def _load_traces(trace_paths: list[Path]) -> list[dict[str, Any]]:
    """Load topology diagnostic traces from JSON or JSONL files.

    Returns:
        List of parsed trace dictionaries.
    """
    traces: list[dict[str, Any]] = []
    for p in trace_paths:
        if not p.is_file():
            continue
        traces.extend(_load_single_trace(p))
    return traces


def _load_single_trace(p: Path) -> list[dict[str, Any]]:
    """Load trace dicts from a single JSON or JSONL file.

    Returns:
        List of parsed trace dictionaries from the file.
    """
    text = p.read_text(encoding="utf-8")
    suffix = p.suffix.lower()
    if suffix == ".json":
        return _parse_json_trace(text)
    if suffix == ".jsonl":
        return _parse_jsonl_trace(text, source_path=p)
    return []


def _parse_json_trace(text: str) -> list[dict[str, Any]]:
    """Parse a JSON trace file content.

    Returns:
        List of trace dicts (single dict or list of dicts).
    """
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return []
    if isinstance(data, dict):
        return [data]
    if isinstance(data, list):
        return [item for item in data if isinstance(item, dict)]
    return []


def _parse_jsonl_trace(text: str, *, source_path: Path | None = None) -> list[dict[str, Any]]:
    """Parse a JSONL trace file content.

    Returns:
        List of trace dicts parsed from non-empty lines.
    """
    traces: list[dict[str, Any]] = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        stripped = line.strip()
        if not stripped:
            continue
        try:
            row = json.loads(stripped)
        except json.JSONDecodeError as exc:
            source = str(source_path) if source_path is not None else "<jsonl>"
            raise ValueError(f"{source}: invalid JSONL at line {line_number}") from exc
        if isinstance(row, dict):
            traces.append(row)
    return traces


def _int_count(value: Any) -> int:
    """Return integer count values while treating null/missing counters as zero."""
    return int(value) if isinstance(value, int | float) else 0


def _extract_step_topology(trace: dict[str, Any]) -> list[dict[str, Any]]:
    """Return per-step topology rows from one trace payload."""
    steps = trace.get("steps")
    if isinstance(steps, list):
        return [s for s in steps if isinstance(s, dict)]
    return []


def _topology_payload(row: dict[str, Any]) -> dict[str, Any]:
    """Return the topology instrumentation payload from a step row."""
    for key in ("topology", "topology_instrumentation"):
        payload = row.get(key)
        if isinstance(payload, dict):
            return payload
    return {}


def _reuse_penalty_payload(row: dict[str, Any]) -> dict[str, Any] | None:
    """Return the reuse-penalty payload from a step row."""
    corridor = row.get("planner_route_corridor")
    if not isinstance(corridor, dict):
        return None
    payload = corridor.get("topology_reuse_penalty")
    return payload if isinstance(payload, dict) else None


def _accumulate_summary_reuse_penalty(
    summary: dict[str, Any],
    applied: int,
    eligible: int,
    reason_counts: Counter[str],
) -> tuple[int, int]:
    """Accumulate reuse-penalty counters from a trace summary dict.

    Returns:
        Tuple of (updated_applied, updated_eligible).
    """
    reuse_diag = summary.get("topology_reuse_penalty", {})
    if not isinstance(reuse_diag, dict):
        return applied, eligible
    applied += _int_count(reuse_diag.get("applied_steps", 0))
    eligible += _int_count(reuse_diag.get("eligible_near_parity_alternative_steps", 0))
    for k, v in reuse_diag.get("reason_counts", {}).items():
        reason_counts[str(k)] += _int_count(v)
    return applied, eligible


def _accumulate_step_reuse_penalty(
    steps: list[dict[str, Any]],
    applied: int,
    eligible: int,
    reason_counts: Counter[str],
) -> tuple[int, int]:
    """Accumulate reuse-penalty counters from per-step data.

    Returns:
        Tuple of (updated_applied, updated_eligible).
    """
    for step_row in steps:
        reuse = _reuse_penalty_payload(step_row)
        if reuse is None:
            continue
        if bool(reuse.get("reuse_penalty_applied", False)):
            applied += 1
            reason = str(reuse.get("reuse_penalty_reason") or "unknown")
            reason_counts[reason] += 1
        if bool(reuse.get("eligible_near_parity_alternative_exists", False)):
            eligible += 1
    return applied, eligible


def _accumulate_progress_deltas(
    summary: dict[str, Any],
    trace: dict[str, Any],
    progress_deltas: list[dict[str, Any]],
    regressions: list[dict[str, Any]],
    unchanged: list[dict[str, Any]],
) -> None:
    """Extract and classify route-progress deltas from a trace summary."""
    hypothesis_progress = summary.get("hypothesis_progress_by_rank", {})
    if not isinstance(hypothesis_progress, dict):
        return
    for rank, row in hypothesis_progress.items():
        if not isinstance(row, dict):
            continue
        delta = row.get("progress_delta_m")
        entry = {
            "trace_scenario": trace.get("scenario_id", "unknown"),
            "trace_seed": trace.get("seed"),
            "rank": rank,
            "samples": row.get("samples", 0),
            "first_corridor": row.get("first_corridor_name"),
            "last_corridor": row.get("last_corridor_name"),
            "first_remaining_m": row.get("first_remaining_distance_m"),
            "last_remaining_m": row.get("last_remaining_distance_m"),
            "progress_delta_m": delta,
            "min_static_clearance_m": row.get("min_static_clearance_m"),
            "min_dynamic_clearance_m": row.get("min_dynamic_clearance_m"),
        }
        if delta is None:
            continue
        progress_deltas.append(entry)
        if isinstance(delta, int | float):
            if delta < -1e-9:
                regressions.append(entry)
            elif abs(delta) < 1e-9:
                unchanged.append(entry)


def _accumulate_trace(  # noqa: C901, PLR0913
    trace: dict[str, Any],
    selected_hypothesis_counts: Counter[str],
    near_parity_reason_counts: Counter[str],
    reuse_penalty_applied: int,
    reuse_penalty_eligible: int,
    reuse_penalty_reason_counts: Counter[str],
    progress_deltas: list[dict[str, Any]],
    regressions: list[dict[str, Any]],
    unchanged: list[dict[str, Any]],
    terminal_outcomes: Counter[str],
    diagnostic_status_counts: Counter[str],
) -> tuple[int, int, int]:
    """Accumulate diagnostic counters from one trace into shared accumulators.

    Returns:
        Tuple of (total_steps, reuse_penalty_applied, reuse_penalty_eligible).
    """
    diag_status = trace.get("diagnostic_status", "unknown")
    diagnostic_status_counts[str(diag_status)] += 1

    summary = trace.get("summary") if isinstance(trace.get("summary"), dict) else {}
    corrective = summary.get("corrective_behavior", {})
    if isinstance(corrective, dict):
        outcome = corrective.get("terminal_outcome", {})
        if isinstance(outcome, dict):
            terminal_outcomes[str(outcome.get("outcome", "unknown"))] += 1

    hypothesis_counts = summary.get("route_selector_selected_hypothesis_counts", {})
    has_summary_hypothesis_counts = isinstance(hypothesis_counts, dict) and bool(hypothesis_counts)
    if has_summary_hypothesis_counts:
        for k, v in hypothesis_counts.items():
            selected_hypothesis_counts[str(k)] += _int_count(v)

    near_parity_reasons = summary.get("selected_row_near_parity_gate_reasons", {})
    has_summary_gate_reasons = isinstance(near_parity_reasons, dict) and bool(near_parity_reasons)
    if has_summary_gate_reasons:
        for k, v in near_parity_reasons.items():
            near_parity_reason_counts[str(k)] += _int_count(v)

    steps = _extract_step_topology(trace)
    total_steps = len(steps)

    if steps:
        reuse_penalty_applied, reuse_penalty_eligible = _accumulate_step_reuse_penalty(
            steps,
            reuse_penalty_applied,
            reuse_penalty_eligible,
            reuse_penalty_reason_counts,
        )
    else:
        reuse_penalty_applied, reuse_penalty_eligible = _accumulate_summary_reuse_penalty(
            summary,
            reuse_penalty_applied,
            reuse_penalty_eligible,
            reuse_penalty_reason_counts,
        )

    _accumulate_progress_deltas(summary, trace, progress_deltas, regressions, unchanged)

    for step_row in steps:
        topo = _topology_payload(step_row)
        selected_hyp = topo.get("selected_hypothesis")
        if selected_hyp is not None and not has_summary_hypothesis_counts:
            selected_hypothesis_counts[str(selected_hyp)] += 1
        gate_reason = topo.get("near_parity_gate_reason")
        if gate_reason is not None and not has_summary_gate_reasons:
            near_parity_reason_counts[str(gate_reason)] += 1

    return total_steps, reuse_penalty_applied, reuse_penalty_eligible


def aggregate_traces(traces: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate topology diagnostic data across multiple trace payloads.

    Returns:
        Aggregated summary dictionary with all diagnostic fields.
    """
    selected_hypothesis_counts: Counter[str] = Counter()
    near_parity_reason_counts: Counter[str] = Counter()
    reuse_penalty_applied = 0
    reuse_penalty_eligible = 0
    reuse_penalty_reason_counts: Counter[str] = Counter()
    total_steps = 0
    progress_deltas: list[dict[str, Any]] = []
    regressions: list[dict[str, Any]] = []
    unchanged: list[dict[str, Any]] = []
    terminal_outcomes: Counter[str] = Counter()
    diagnostic_status_counts: Counter[str] = Counter()

    for trace in traces:
        steps_count, reuse_penalty_applied, reuse_penalty_eligible = _accumulate_trace(
            trace,
            selected_hypothesis_counts,
            near_parity_reason_counts,
            reuse_penalty_applied,
            reuse_penalty_eligible,
            reuse_penalty_reason_counts,
            progress_deltas,
            regressions,
            unchanged,
            terminal_outcomes,
            diagnostic_status_counts,
        )
        total_steps += steps_count

    regressions.sort(key=lambda r: r.get("progress_delta_m") or 0)
    unchanged.sort(key=lambda r: abs(r.get("progress_delta_m") or 0))

    return {
        "claim_boundary": CLAIM_BOUNDARY,
        "trace_count": len(traces),
        "total_steps": total_steps,
        "diagnostic_status_counts": dict(sorted(diagnostic_status_counts.items())),
        "selected_hypothesis_counts": dict(sorted(selected_hypothesis_counts.items())),
        "near_parity_gate_reason_counts": dict(sorted(near_parity_reason_counts.items())),
        "reuse_penalty": {
            "applied_steps": reuse_penalty_applied,
            "eligible_near_parity_alternative_steps": reuse_penalty_eligible,
            "reason_counts": dict(sorted(reuse_penalty_reason_counts.items())),
        },
        "route_progress_deltas": progress_deltas,
        "top_regressions": regressions[:10],
        "top_unchanged": unchanged[:10],
        "terminal_outcome_counts": dict(sorted(terminal_outcomes.items())),
    }


def build_report_payload(
    trace_paths: list[Path],
    *,
    label: str = "topology-diagnostic-snapshot",
) -> dict[str, Any]:
    """Build the full report payload from one or more trace files.

    Returns:
        Complete report payload dictionary.
    """
    traces = _load_traces(trace_paths)
    agg = aggregate_traces(traces)
    return {
        "report_kind": "topology_diagnostic_report_snapshot",
        "label": label,
        "claim_boundary": CLAIM_BOUNDARY,
        "source_trace_count": len(traces),
        "source_trace_paths": [str(p) for p in trace_paths],
        **agg,
    }


def render_markdown(payload: dict[str, Any]) -> str:
    """Render the report payload as a Markdown string.

    Returns:
        Markdown-formatted report string.
    """
    lines = [
        f"# Topology Diagnostic Report Snapshot: {payload.get('label', '')}",
        "",
        "Diagnostic-only evidence; this is not planner-promotion or benchmark success evidence.",
        "",
        f"- Claim boundary: `{payload.get('claim_boundary', '')}`",
        f"- Trace count: {payload.get('trace_count', 0)}",
        f"- Total steps: {payload.get('total_steps', 0)}",
        f"- Diagnostic status counts: `{payload.get('diagnostic_status_counts', {})}`",
        f"- Terminal outcome counts: `{payload.get('terminal_outcome_counts', {})}`",
        "",
        "## Selected Hypotheses",
        "",
    ]
    lines.extend(_render_count_dict(payload.get("selected_hypothesis_counts", {})))
    lines.extend(["", "## Near-Parity Gate Reasons", ""])
    lines.extend(_render_count_dict(payload.get("near_parity_gate_reason_counts", {})))
    lines.extend(["", "## Reuse-Penalty Activations", ""])
    lines.extend(_render_reuse_penalty_section(payload.get("reuse_penalty", {})))
    lines.extend(["", "## Route-Progress Deltas", ""])
    lines.extend(
        _render_progress_table(payload.get("route_progress_deltas", []), "progress_delta_m")
    )
    lines.extend(["", "## Top Regressions", ""])
    lines.extend(_render_regression_table(payload.get("top_regressions", [])))
    lines.extend(["", "## Top Unchanged Cases", ""])
    lines.extend(_render_unchanged_table(payload.get("top_unchanged", [])))
    lines.append("")
    return "\n".join(lines)


def _render_count_dict(counts: dict[str, Any]) -> list[str]:
    """Render a count dictionary as Markdown bullet items.

    Returns:
        List of Markdown bullet item strings.
    """
    if not counts:
        return ["- (none)"]
    return [f"- `{k}`: {v}" for k, v in counts.items()]


def _render_reuse_penalty_section(rp: dict[str, Any]) -> list[str]:
    """Render the reuse-penalty section as Markdown lines.

    Returns:
        List of Markdown strings for the reuse-penalty section.
    """
    if not isinstance(rp, dict):
        return ["- (none)"]
    lines = [
        f"- Applied steps: {rp.get('applied_steps', 0)}",
        f"- Eligible near-parity alternative steps: "
        f"{rp.get('eligible_near_parity_alternative_steps', 0)}",
    ]
    rp_reasons = rp.get("reason_counts", {})
    if rp_reasons:
        for reason, count in rp_reasons.items():
            lines.append(f"  - `{reason}`: {count}")
    return lines


def _render_progress_table(
    deltas: list[dict[str, Any]],
    delta_key: str,
) -> list[str]:
    """Render a progress-delta table as Markdown lines.

    Returns:
        List of Markdown table row strings.
    """
    if not deltas:
        return ["- (none)"]
    lines = [
        "| Scenario | Seed | Rank | Samples | Progress delta (m) |",
        "|---|---:|---:|---:|---:|",
    ]
    for d in deltas[:20]:
        lines.append(
            f"| {d.get('trace_scenario', '')} "
            f"| {d.get('trace_seed', '')} "
            f"| {d.get('rank', '')} "
            f"| {d.get('samples', '')} "
            f"| {d.get(delta_key, '')} |"
        )
    return lines


def _render_regression_table(regs: list[dict[str, Any]]) -> list[str]:
    """Render the top-regressions table as Markdown lines.

    Returns:
        List of Markdown table row strings.
    """
    if not regs:
        return ["- (none)"]
    lines = [
        "| Scenario | Seed | Rank | Progress delta (m) | Last corridor |",
        "|---|---:|---:|---:|---|",
    ]
    for r in regs:
        lines.append(
            f"| {r.get('trace_scenario', '')} "
            f"| {r.get('trace_seed', '')} "
            f"| {r.get('rank', '')} "
            f"| {r.get('progress_delta_m', '')} "
            f"| {r.get('last_corridor', '')} |"
        )
    return lines


def _render_unchanged_table(rows: list[dict[str, Any]]) -> list[str]:
    """Render the top-unchanged table as Markdown lines.

    Returns:
        List of Markdown table row strings.
    """
    if not rows:
        return ["- (none)"]
    lines = [
        "| Scenario | Seed | Rank | Progress delta (m) | First corridor |",
        "|---|---:|---:|---:|---|",
    ]
    for u in rows:
        lines.append(
            f"| {u.get('trace_scenario', '')} "
            f"| {u.get('trace_seed', '')} "
            f"| {u.get('rank', '')} "
            f"| {u.get('progress_delta_m', '')} "
            f"| {u.get('first_corridor', '')} |"
        )
    return lines
