#!/usr/bin/env python3
"""Generate a why-first benchmark report from compact evidence."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Sequence

REQUIRED_SECTIONS = (
    "Outcome Summary",
    "Mechanism Activation",
    "Failure Mechanism Classification",
    "Paired Comparator",
    "Trace Evidence",
    "Alternative Explanations",
    "Continue / Revise / Stop Decision",
)
CLAIM_BOUNDARY = "Report strength is limited to the compact input evidence."
_NON_SUCCESS_STATUSES = {"fallback", "degraded", "failed", "not_available", "skipped"}
_REPORT_MODES = ("why-first", "dissertation")


class WhyFirstReportError(ValueError):
    """Raised when compact evidence cannot produce a faithful report."""


def _build_parser() -> argparse.ArgumentParser:
    """Build the why-first report parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Compact JSON evidence input.")
    parser.add_argument("--output", type=Path, required=True, help="Markdown report output path.")
    parser.add_argument(
        "--mode",
        choices=_REPORT_MODES,
        default="why-first",
        help="Report mode to render. The default preserves the original why-first report.",
    )
    return parser


def load_evidence(path: Path) -> dict[str, Any]:
    """Load compact JSON evidence.

    Returns:
        dict[str, Any]: Parsed evidence mapping.
    """
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise WhyFirstReportError("compact evidence must be a JSON object")
    return payload


def generate_report(evidence: Mapping[str, Any], *, mode: str = "why-first") -> str:
    """Generate the why-first report Markdown.

    Returns:
        str: Markdown report with the required why-first sections.
    """
    if mode not in _REPORT_MODES:
        raise WhyFirstReportError(f"unsupported report mode: {mode}")
    title = _text(evidence.get("title"), default="Why-First Benchmark Report")
    lines = [f"# {title}", ""]
    lines.extend(_section("Outcome Summary", _outcome_summary(evidence)))
    lines.extend(_section("Mechanism Activation", _mechanism_activation(evidence)))
    lines.extend(
        _section(
            "Failure Mechanism Classification",
            _classification(evidence),
        )
    )
    lines.extend(_section("Paired Comparator", _paired_comparator(evidence)))
    lines.extend(_section("Trace Evidence", _trace_evidence(evidence)))
    lines.extend(_section("Alternative Explanations", _alternative_explanations(evidence)))
    lines.extend(_section("Continue / Revise / Stop Decision", _decision(evidence)))
    if mode == "dissertation":
        lines.extend(_section("Dissertation-Facing Handoff", _dissertation_handoff(evidence)))
    lines.extend(["## Claim Boundary", "", CLAIM_BOUNDARY, ""])
    return "\n".join(lines)


def _section(title: str, body: list[str]) -> list[str]:
    """Return a Markdown section."""
    return [f"## {title}", "", *body, ""]


def _outcome_summary(evidence: Mapping[str, Any]) -> list[str]:
    """Render the outcome summary section."""
    outcome = _text(evidence.get("outcome"), default="not specified")
    planner = _text(evidence.get("planner"), default="unknown planner")
    scenario = _text(evidence.get("scenario_id"), default="unknown scenario")
    status = _status(evidence)
    rows = [
        f"- Planner/scenario: `{planner}` on `{scenario}`.",
        f"- Outcome: {outcome}.",
        f"- Execution status: `{status}`.",
    ]
    caveat = _fallback_caveat(status)
    if caveat:
        rows.append(caveat)
    metrics = evidence.get("metrics")
    if isinstance(metrics, Mapping) and metrics:
        rows.append("- Metrics: " + _inline_mapping(metrics) + ".")
    return rows


def _mechanism_activation(evidence: Mapping[str, Any]) -> list[str]:
    """Render the mechanism activation section."""
    activation = evidence.get("mechanism_activation")
    if not isinstance(activation, Mapping) or not activation:
        return ["- Mechanism activation: not specified in compact evidence."]
    name = _text(activation.get("name"), default="unknown")
    status = _text(activation.get("status"), default="unknown")
    evidence_text = _text(activation.get("evidence"), default="not specified")
    return [
        f"- Mechanism: `{name}`.",
        f"- Activation status: `{status}`.",
        f"- Evidence: {evidence_text}.",
    ]


def _classification(evidence: Mapping[str, Any]) -> list[str]:
    """Render the failure mechanism classification section."""
    classification = evidence.get("failure_mechanism")
    if isinstance(classification, Mapping):
        label = _text(classification.get("label"), default="unclassified")
        rationale = _text(classification.get("rationale"), default="not specified")
    else:
        label = _text(classification, default="unclassified")
        rationale = "not specified"
    return [f"- Classification: `{label}`.", f"- Rationale: {rationale}."]


def _paired_comparator(evidence: Mapping[str, Any]) -> list[str]:
    """Render the paired comparator section."""
    comparator = evidence.get("paired_comparator")
    if not isinstance(comparator, Mapping) or not comparator:
        return ["- No paired comparator was provided."]
    name = _text(comparator.get("name"), default="unnamed comparator")
    outcome = _text(comparator.get("outcome"), default="not specified")
    delta = comparator.get("delta")
    rows = [f"- Comparator: `{name}`.", f"- Comparator outcome: {outcome}."]
    if isinstance(delta, Mapping) and delta:
        rows.append("- Delta: " + _inline_mapping(delta) + ".")
    return rows


def _trace_evidence(evidence: Mapping[str, Any]) -> list[str]:
    """Render the trace evidence section."""
    trace = evidence.get("trace_evidence")
    if isinstance(trace, list) and trace:
        return [f"- {_text(item, default='not specified')}." for item in trace]
    if isinstance(trace, Mapping) and trace:
        return [
            f"- {_text(key, default='trace')}: {_text(value, default='not specified')}."
            for key, value in trace.items()
        ]
    return ["- No trace evidence reference was provided."]


def _alternative_explanations(evidence: Mapping[str, Any]) -> list[str]:
    """Render the alternative explanations section."""
    alternatives = evidence.get("alternative_explanations")
    if isinstance(alternatives, list) and alternatives:
        return [f"- {_text(item, default='not specified')}." for item in alternatives]
    return ["- No alternative explanations were provided."]


def _decision(evidence: Mapping[str, Any]) -> list[str]:
    """Render the continue/revise/stop decision section."""
    decision = evidence.get("decision")
    if isinstance(decision, Mapping):
        action = _text(decision.get("action"), default="revise")
        rationale = _text(decision.get("rationale"), default="not specified")
        next_step = _text(decision.get("next_step"), default="not specified")
    else:
        action = _text(decision, default="revise")
        rationale = "not specified"
        next_step = "not specified"
    return [
        f"- Decision: `{action}`.",
        f"- Rationale: {rationale}.",
        f"- Next step: {next_step}.",
    ]


def _dissertation_handoff(evidence: Mapping[str, Any]) -> list[str]:
    """Render conservative dissertation-facing handoff fields."""
    dissertation = evidence.get("dissertation")
    values = dissertation if isinstance(dissertation, Mapping) else evidence
    status = _status(evidence)
    caveat = (
        "fallback/degraded/failed/not-available evidence must not be counted as benchmark success"
        if _fallback_caveat(status)
        else "no fallback/degraded status was declared in the compact evidence"
    )
    return [
        "- `reader_takeaway`: "
        + _text(values.get("reader_takeaway"), default="not specified")
        + ".",
        "- `allowed_wording`: "
        + _list_or_text(values.get("allowed_wording"), default="not specified")
        + ".",
        "- `not_claimed`: "
        + _list_or_text(values.get("not_claimed"), default="not specified")
        + ".",
        "- `figure_table_candidates`: "
        + _list_or_text(
            values.get("figure_table_candidates")
            or values.get("artifact_ids")
            or values.get("artifact_id"),
            default="not specified",
        )
        + ".",
        f"- `fallback_degraded_caveat`: {caveat}.",
    ]


def _status(evidence: Mapping[str, Any]) -> str:
    """Return the normalized execution status."""
    return _text(
        evidence.get("execution_status") or evidence.get("readiness_status"),
        default="unknown",
    ).lower()


def _fallback_caveat(status: str) -> str | None:
    """Return the non-success caveat for fallback/degraded style rows."""
    if status not in _NON_SUCCESS_STATUSES:
        return None
    return (
        "- Limitation: this row is fallback/degraded/failed/not-available evidence and must not "
        "be counted as benchmark success."
    )


def _inline_mapping(values: Mapping[str, Any]) -> str:
    """Render a compact key/value mapping."""
    return ", ".join(f"`{key}`={_text(value, default='null')}" for key, value in values.items())


def _text(value: Any, *, default: str) -> str:
    """Convert scalar-ish values into report text."""
    if value is None:
        return default
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int | float):
        return str(value)
    text = str(value).strip()
    return text or default


def _list_or_text(value: Any, *, default: str) -> str:
    """Render a scalar or list value as compact report text."""
    if isinstance(value, list):
        rendered = [_text(item, default="").strip() for item in value]
        rendered = [item for item in rendered if item]
        return "; ".join(rendered) if rendered else default
    return _text(value, default=default)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the report generator."""
    args = _build_parser().parse_args(argv)
    evidence = load_evidence(args.input)
    report = generate_report(evidence, mode=args.mode)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(report, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
