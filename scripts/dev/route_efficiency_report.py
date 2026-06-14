#!/usr/bin/env python3
"""Local route-efficiency report for routed-worker manifests.

Reads one or more routed-worker manifest JSON files and optionally a simple
PR-loop/snapshot JSON file.  Emits a compact efficiency summary that measures
whether delegated agent work is saving review effort.

Route success is **not** task success.  The report surfaces this boundary
explicitly in the output.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

SCHEMA_VERSION = "route_efficiency_report.v1"
DASHBOARD_SCHEMA_VERSION = "route_efficiency_dashboard.v1"
ROUTE_EVIDENCE_WARNING = (
    "Route success and complete artifact presence are route evidence only; "
    "they are not task acceptance. The orchestrator must still inspect the "
    "diff and run the required local validation."
)
_EXPECTED_ARTIFACT_KEYS = frozenset(
    {"result_json", "result_md", "diffstat", "status", "validation"}
)
EXPECTED_ARTIFACT_KEYS = _EXPECTED_ARTIFACT_KEYS


def _is_complete_artifact_set(compact: dict[str, Any]) -> bool:
    """Return True when every expected artifact key is present."""
    if not isinstance(compact, dict):
        return False
    return all(
        isinstance(compact.get(k), dict) and compact[k].get("present") is True
        for k in _EXPECTED_ARTIFACT_KEYS
    )


def is_complete_artifact_set(compact: dict[str, Any]) -> bool:
    """Public wrapper for routed-worker artifact completeness checks."""
    return _is_complete_artifact_set(compact)


def _has_validation_success(compact: dict[str, Any]) -> bool | None:
    """Return True/False when the validation artifact signals outcome, else None."""
    val = compact.get("validation") if isinstance(compact, dict) else None
    if not isinstance(val, dict) or not val.get("present"):
        return None
    result_text = val.get("result") or ""
    if isinstance(result_text, str):
        lowered = result_text.lower()
        neutralized = re.sub(r"\b0\s+(?:failed|failures?|errors?)\b", "", lowered)
        if (
            "unsuccessful" in neutralized
            or re.search(r"\b(?:failed|failure|errors?)\b", neutralized)
            or re.search(r"\b[1-9]\d*\s+(?:failed|failures?|errors?)\b", neutralized)
        ):
            return False
        if re.search(r"\b0\s+passed\b", neutralized):
            return False
        if re.search(r"\b(?:pass|passed|success|successful|succeeded|ok)\b", neutralized):
            return True
    return None


def has_validation_success(compact: dict[str, Any]) -> bool | None:
    """Public wrapper for routed-worker validation-result parsing."""
    return _has_validation_success(compact)


def _provider_name(route: Any) -> str:
    """Extract a provider label from a route dict, or 'unknown'."""
    if isinstance(route, dict):
        return route.get("provider") or route.get("name") or "unknown"
    return "unknown"


def _validate_present(compact: dict[str, Any]) -> bool:
    """Return True if the validation artifact is present."""
    if not isinstance(compact, dict):
        return False
    val = compact.get("validation")
    return isinstance(val, dict) and val.get("present") is True


class _Accumulator:
    """Mutable counters for the manifest scan loop."""

    def __init__(self) -> None:
        self.total_attempts = 0
        self.complete_count = 0
        self.validation_present = 0
        self.validation_success = 0
        self.reroutes = 0
        self.provider_incomplete: Counter[str] = Counter()
        self.provider_total: Counter[str] = Counter()
        self.provider_complete: Counter[str] = Counter()
        self.failure_class_incomplete: Counter[str] = Counter()
        self.missing_artifacts: Counter[str] = Counter()

    def process_attempt(self, idx: int, attempt: dict[str, Any]) -> None:
        if not isinstance(attempt, dict):
            compact: dict[str, Any] = {}
            provider = "unknown"
            fc = "malformed"
        else:
            compact = attempt.get("compact_artifacts") or {}
            provider = _provider_name(attempt.get("route"))
            fc = attempt.get("failure_class") or "unknown"

        self.provider_total[provider] += 1

        if _is_complete_artifact_set(compact):
            self.complete_count += 1
            self.provider_complete[provider] += 1
        else:
            self.provider_incomplete[provider] += 1
            self.failure_class_incomplete[str(fc)] += 1
            if isinstance(compact, dict):
                for key in sorted(_EXPECTED_ARTIFACT_KEYS):
                    artifact = compact.get(key)
                    if not isinstance(artifact, dict) or artifact.get("present") is not True:
                        self.missing_artifacts[key] += 1

        if _validate_present(compact):
            self.validation_present += 1
            if _has_validation_success(compact) is True:
                self.validation_success += 1

        if idx > 0:
            self.reroutes += 1


def _extract_snapshot_outcome(
    snapshot: dict[str, Any] | None,
) -> tuple[bool | None, str | None]:
    """Pull accepted and outcome from optional snapshot metadata."""
    if not isinstance(snapshot, dict):
        return None, None
    accepted_raw = snapshot.get("accepted")
    accepted = None if accepted_raw is None else bool(accepted_raw)
    note_raw = snapshot.get("outcome")
    note = None if note_raw is None else str(note_raw)
    return accepted, note


def _routing_recommendation(
    recommendation_class: str,
    action: str,
    evidence: str,
) -> dict[str, str]:
    """Build one route-evidence-only recommendation entry."""
    return {
        "class": recommendation_class,
        "action": action,
        "evidence": evidence,
        "caveat": "Route evidence only; not task acceptance.",
    }


def _no_routing_recommendation(evidence: str) -> dict[str, str]:
    """Build the deterministic no-recommendation entry."""
    return _routing_recommendation(
        "no_recommendation",
        "no_recommendation: insufficient data for routing policy",
        evidence,
    )


def _provider_recommendations(report: dict[str, Any]) -> list[dict[str, str]]:
    """Return deterministic provider preference and avoidance hints."""
    incomplete_by_provider = report.get("incomplete_by_provider", {})
    by_provider: dict[str, dict[str, int]] = report.get("by_provider", {})
    recommendations: list[dict[str, str]] = []

    for provider, counts in sorted(by_provider.items()):
        p_total = counts.get("total", 0)
        p_complete = counts.get("complete", 0)
        if p_total == 0:
            continue
        if p_complete == p_total and p_total >= 2 and incomplete_by_provider:
            recommendations.append(
                _routing_recommendation(
                    "prefer_provider",
                    f"prefer_provider: {provider} ({p_complete}/{p_total} complete, 100%)",
                    f"{p_complete}/{p_total} attempts complete for {provider}",
                )
            )
        if p_complete == 0 and p_total >= 2:
            recommendations.append(
                _routing_recommendation(
                    "avoid_provider",
                    f"avoid_provider: {provider} (0/{p_total} complete, 0%)",
                    f"0/{p_total} attempts complete for {provider}",
                )
            )
    return recommendations


def _failure_class_recommendations(report: dict[str, Any]) -> list[dict[str, str]]:
    """Return deterministic failure-class investigation hints."""
    incomplete_by_fc = report.get("incomplete_by_failure_class", {})
    incomplete_total = sum(incomplete_by_fc.values())
    recommendations: list[dict[str, str]] = []

    for failure_class, count in sorted(incomplete_by_fc.items()):
        if incomplete_total and count > incomplete_total / 2:
            pct = round(count / incomplete_total * 100)
            recommendations.append(
                _routing_recommendation(
                    "investigate_failure_class",
                    (
                        f"investigate_failure_class: {failure_class} "
                        f"({count}/{incomplete_total} incomplete, {pct}%)"
                    ),
                    f"{count}/{incomplete_total} incomplete attempts have failure class {failure_class}",
                )
            )
    return recommendations


def _reroute_threshold_recommendation(report: dict[str, Any]) -> dict[str, str] | None:
    """Return an overall reroute recommendation when completion is low."""
    total = report.get("delegated_attempts", 0)
    complete_artifacts = report.get("complete_artifacts", {})
    rate = complete_artifacts.get("rate", 0.0)
    if rate >= 0.5 or total < 2:
        return None
    return _routing_recommendation(
        "reroute_threshold_met",
        (
            f"reroute_threshold_met: {complete_artifacts.get('count', 0)}/{total} "
            f"complete ({rate:.0%}), consider fallback strategy"
        ),
        f"overall completion rate {rate:.0%} across {total} attempts",
    )


def _derive_routing_recommendations(report: dict[str, Any]) -> list[dict[str, str]]:
    """Derive deterministic routing recommendations from report metrics."""
    total = report.get("delegated_attempts", 0)
    if total < 2:
        return [_no_routing_recommendation(f"{total} delegated attempt(s)")]

    recommendations = _provider_recommendations(report)
    recommendations.extend(_failure_class_recommendations(report))
    reroute_recommendation = _reroute_threshold_recommendation(report)
    if reroute_recommendation is not None:
        recommendations.append(reroute_recommendation)

    if recommendations:
        return recommendations

    rate = report.get("complete_artifacts", {}).get("rate", 0.0)
    return [
        _no_routing_recommendation(
            f"{total} attempts, {rate:.0%} completion rate, no dominant provider or failure pattern"
        )
    ]


def _analyze_single_manifest(
    manifest: dict[str, Any],
    source: str,
) -> dict[str, Any]:
    """Build a per-manifest sub-report with source label."""
    acc = _Accumulator()
    attempts = manifest.get("attempted_routes") if isinstance(manifest, dict) else None
    if isinstance(attempts, list):
        acc.total_attempts = len(attempts)
        for idx, attempt in enumerate(attempts):
            acc.process_attempt(idx, attempt)

    rate = round(acc.complete_count / acc.total_attempts, 4) if acc.total_attempts else 0.0
    return {
        "source": source,
        "delegated_attempts": acc.total_attempts,
        "complete_artifacts": {"count": acc.complete_count, "rate": rate},
        "validation_presence": {
            "present": acc.validation_present,
            "success_inferable": acc.validation_success,
        },
        "reroutes": acc.reroutes,
        "incomplete_by_provider": dict(acc.provider_incomplete),
        "incomplete_by_failure_class": dict(acc.failure_class_incomplete),
        "missing_artifacts": dict(acc.missing_artifacts),
        "by_provider": {
            p: {"total": t, "complete": acc.provider_complete[p]}
            for p, t in acc.provider_total.items()
        },
    }


def analyze_dashboard(
    manifests_with_sources: list[tuple[dict[str, Any], str]],
    *,
    snapshot: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a dashboard report from multiple manifests with source labels."""
    per_manifest: list[dict[str, Any]] = []
    for manifest, source in manifests_with_sources:
        per_manifest.append(_analyze_single_manifest(manifest, source))

    overall_acc = _Accumulator()
    for manifest, _source in manifests_with_sources:
        attempts = manifest.get("attempted_routes") if isinstance(manifest, dict) else None
        if isinstance(attempts, list):
            overall_acc.total_attempts += len(attempts)
            for idx, attempt in enumerate(attempts):
                overall_acc.process_attempt(idx, attempt)

    total = overall_acc.total_attempts
    rate = round(overall_acc.complete_count / total, 4) if total else 0.0

    provider_trend_map: dict[str, list[dict[str, Any]]] = {}
    for sub in per_manifest:
        for provider, counts in sub.get("by_provider", {}).items():
            p_total = counts.get("total", 0)
            p_complete = counts.get("complete", 0)
            p_rate = round(p_complete / p_total, 4) if p_total else 0.0
            entry = {
                "source": sub["source"],
                "total": p_total,
                "complete": p_complete,
                "rate": p_rate,
            }
            provider_trend_map.setdefault(provider, []).append(entry)

    provider_trend = [
        {"provider": provider, "per_manifest": entries}
        for provider, entries in sorted(provider_trend_map.items())
    ]

    accepted, acceptance_note = _extract_snapshot_outcome(snapshot)

    overall_report = {
        "schema": DASHBOARD_SCHEMA_VERSION,
        "generated_at": datetime.now(UTC).isoformat(),
        "manifests_analyzed": len(manifests_with_sources),
        "overall": {
            "delegated_attempts": overall_acc.total_attempts,
            "complete_artifacts": {"count": overall_acc.complete_count, "rate": rate},
            "validation_presence": {
                "present": overall_acc.validation_present,
                "success_inferable": overall_acc.validation_success,
            },
            "total_reroutes": overall_acc.reroutes,
            "estimated_inspections_avoided": overall_acc.complete_count,
        },
        "per_manifest": per_manifest,
        "provider_trend": provider_trend,
        "incomplete_by_provider": dict(overall_acc.provider_incomplete),
        "incomplete_by_failure_class": dict(overall_acc.failure_class_incomplete),
        "missing_artifacts": dict(overall_acc.missing_artifacts),
        "by_provider": {
            p: {"total": t, "complete": overall_acc.provider_complete[p]}
            for p, t in overall_acc.provider_total.items()
        },
        "final_outcome": {
            "accepted": accepted,
            "note": (
                acceptance_note
                if acceptance_note is not None
                else "route evidence only; not task acceptance"
            ),
        },
        "warning": ROUTE_EVIDENCE_WARNING,
    }

    recommendations_source = {
        "delegated_attempts": overall_acc.total_attempts,
        "complete_artifacts": {"count": overall_acc.complete_count, "rate": rate},
        "incomplete_by_provider": dict(overall_acc.provider_incomplete),
        "incomplete_by_failure_class": dict(overall_acc.failure_class_incomplete),
        "missing_artifacts": dict(overall_acc.missing_artifacts),
        "by_provider": {
            p: {"total": t, "complete": overall_acc.provider_complete[p]}
            for p, t in overall_acc.provider_total.items()
        },
    }
    overall_report["routing_recommendations"] = _derive_routing_recommendations(
        recommendations_source
    )
    return overall_report


def _append_dashboard_summary(lines: list[str], dashboard: dict[str, Any]) -> None:
    """Append dashboard headline metrics."""
    overall = dashboard["overall"]
    ca = overall["complete_artifacts"]
    vp = overall["validation_presence"]
    lines.append("# Route Efficiency Dashboard\n")
    lines.append(f"- **Manifests analyzed:** {dashboard['manifests_analyzed']}")
    lines.append(f"- **Generated at:** {dashboard['generated_at']}")
    lines.append(f"- **Delegated attempts:** {overall['delegated_attempts']}")
    lines.append(
        f"- **Complete artifacts:** {ca['count']}/{overall['delegated_attempts']} ({ca['rate']:.0%})"
    )
    lines.append(
        f"- **Validation present:** {vp['present']}"
        f" | **success inferable:** {vp['success_inferable']}"
    )
    lines.append(f"- **Total reroutes:** {overall['total_reroutes']}")
    lines.append(f"- **Estimated inspections avoided:** {overall['estimated_inspections_avoided']}")

    ibp = dashboard.get("incomplete_by_provider", {})
    if ibp:
        parts = [f"{k}: {v}" for k, v in sorted(ibp.items())]
        lines.append(f"- **Incomplete by provider:** {', '.join(parts)}")

    ibf = dashboard.get("incomplete_by_failure_class", {})
    if ibf:
        parts = [f"{k}: {v}" for k, v in sorted(ibf.items())]
        lines.append(f"- **Incomplete by failure class:** {', '.join(parts)}")

    missing_artifacts = dashboard.get("missing_artifacts", {})
    if missing_artifacts:
        parts = [f"{k}: {v}" for k, v in sorted(missing_artifacts.items())]
        lines.append(f"- **Missing artifacts:** {', '.join(parts)}")

    lines.append(f"\n> {dashboard['warning']}\n")


def _append_dashboard_tables(lines: list[str], dashboard: dict[str, Any]) -> None:
    """Append dashboard per-manifest and provider-trend tables."""
    per_manifest = dashboard.get("per_manifest", [])
    if per_manifest:
        lines.append("## Per-manifest breakdown\n")
        lines.append("| Source | Attempts | Complete | Rate | Reroutes |")
        lines.append("|--------|----------|----------|------|----------|")
        for sub in per_manifest:
            src = sub["source"]
            att = sub["delegated_attempts"]
            cnt = sub["complete_artifacts"]["count"]
            r = sub["complete_artifacts"]["rate"]
            rr = sub["reroutes"]
            lines.append(f"| {src} | {att} | {cnt} | {r:.0%} | {rr} |")
        lines.append("")

    provider_trend = dashboard.get("provider_trend", [])
    if provider_trend:
        lines.append("## Provider trend\n")
        lines.append("| Provider | Source | Total | Complete | Rate |")
        lines.append("|----------|--------|-------|----------|------|")
        for pt in provider_trend:
            provider = pt["provider"]
            for entry in pt["per_manifest"]:
                lines.append(
                    f"| {provider} | {entry['source']} | {entry['total']} |"
                    f" {entry['complete']} | {entry['rate']:.0%} |"
                )
        lines.append("")


def _append_routing_recommendations(lines: list[str], recs: list[dict[str, str]]) -> None:
    """Append compact routing recommendation bullets."""
    lines.append("## Routing recommendations\n")
    for rec in recs:
        lines.append(f"- **{rec['class']}**: {rec['action']}")
        lines.append(f"  - Evidence: {rec['evidence']}")
        lines.append(f"  - Caveat: {rec['caveat']}")
    lines.append("")


def _format_dashboard_markdown(dashboard: dict[str, Any]) -> str:
    """Render the dashboard as a compact Markdown summary."""
    lines: list[str] = []
    _append_dashboard_summary(lines, dashboard)
    _append_dashboard_tables(lines, dashboard)
    recs = dashboard.get("routing_recommendations")
    if recs:
        _append_routing_recommendations(lines, recs)
    return "\n".join(lines)


def analyze_manifests(
    manifests: list[dict[str, Any]],
    *,
    snapshot: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build an efficiency report from one or more routed-worker manifests."""
    acc = _Accumulator()

    for manifest in manifests:
        if not isinstance(manifest, dict):
            continue
        attempts = manifest.get("attempted_routes")
        if not isinstance(attempts, list):
            continue
        acc.total_attempts += len(attempts)
        for idx, attempt in enumerate(attempts):
            acc.process_attempt(idx, attempt)

    accepted, acceptance_note = _extract_snapshot_outcome(snapshot)

    rate = round(acc.complete_count / acc.total_attempts, 4) if acc.total_attempts else 0.0
    report = {
        "schema": SCHEMA_VERSION,
        "manifests_analyzed": len(manifests),
        "delegated_attempts": acc.total_attempts,
        "complete_artifacts": {"count": acc.complete_count, "rate": rate},
        "validation_presence": {
            "present": acc.validation_present,
            "success_inferable": acc.validation_success,
        },
        "reroutes": acc.reroutes,
        "incomplete_by_provider": dict(acc.provider_incomplete),
        "incomplete_by_failure_class": dict(acc.failure_class_incomplete),
        "missing_artifacts": dict(acc.missing_artifacts),
        "by_provider": {
            p: {"total": t, "complete": acc.provider_complete[p]}
            for p, t in acc.provider_total.items()
        },
        "final_outcome": {
            "accepted": accepted,
            "note": (
                acceptance_note
                if acceptance_note is not None
                else "route evidence only; not task acceptance"
            ),
        },
        "estimated_inspections_avoided": acc.complete_count,
        "warning": ROUTE_EVIDENCE_WARNING,
    }
    report["routing_recommendations"] = _derive_routing_recommendations(report)
    return report


def _format_markdown(report: dict[str, Any]) -> str:
    """Render the report as a compact Markdown summary."""
    lines: list[str] = []
    lines.append("# Route Efficiency Report\n")
    lines.append(f"- **Manifests analyzed:** {report['manifests_analyzed']}")
    lines.append(f"- **Delegated attempts:** {report['delegated_attempts']}")

    ca = report["complete_artifacts"]
    lines.append(
        f"- **Complete artifacts:** {ca['count']}/{report['delegated_attempts']} ({ca['rate']:.0%})"
    )

    vp = report["validation_presence"]
    lines.append(
        f"- **Validation present:** {vp['present']}"
        f" | **success inferable:** {vp['success_inferable']}"
    )
    lines.append(f"- **Reroutes:** {report['reroutes']}")

    ibp = report["incomplete_by_provider"]
    if ibp:
        parts = [f"{k}: {v}" for k, v in sorted(ibp.items())]
        lines.append(f"- **Incomplete by provider:** {', '.join(parts)}")

    ibf = report["incomplete_by_failure_class"]
    if ibf:
        parts = [f"{k}: {v}" for k, v in sorted(ibf.items())]
        lines.append(f"- **Incomplete by failure class:** {', '.join(parts)}")

    fo = report["final_outcome"]
    lines.append(f"- **Final accepted:** {fo['accepted']} - {fo['note']}")
    lines.append(f"- **Estimated inspections avoided:** {report['estimated_inspections_avoided']}")
    lines.append(f"\n> {report['warning']}\n")

    recs = report.get("routing_recommendations")
    if recs:
        lines.append("## Routing recommendations\n")
        for rec in recs:
            lines.append(f"- **{rec['class']}**: {rec['action']}")
            lines.append(f"  - Evidence: {rec['evidence']}")
            lines.append(f"  - Caveat: {rec['caveat']}")
        lines.append("")

    return "\n".join(lines)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "manifests",
        nargs="+",
        help="Routed-worker manifest JSON files.",
    )
    parser.add_argument(
        "--snapshot",
        help="Optional PR-loop or snapshot JSON with accepted/outcome metadata.",
    )
    parser.add_argument(
        "--format",
        choices=["json", "markdown"],
        default="json",
        dest="output_format",
        help="Output format (default: json).",
    )
    parser.add_argument(
        "--output",
        help="Write report to this file instead of stdout.",
    )
    parser.add_argument(
        "--dashboard",
        action="store_true",
        help="Enable dashboard mode over multiple manifests with per-manifest breakdown and provider trends.",
    )
    return parser.parse_args(argv)


def _load_manifest_file(path: Path) -> list[dict[str, Any]]:
    """Load one manifest file, accepting either one object or a list of objects."""
    raw = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(raw, dict):
        return [raw]
    if isinstance(raw, list):
        if not all(isinstance(item, dict) for item in raw):
            raise ValueError(f"{path}: manifest list entries must be JSON objects")
        return raw
    raise ValueError(f"{path}: manifest input must be a JSON object or list of objects")


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    args = _parse_args(argv)
    manifests: list[dict[str, Any]] = []
    for path_str in args.manifests:
        manifests.extend(_load_manifest_file(Path(path_str)))

    snapshot: dict[str, Any] | None = None
    if args.snapshot:
        snapshot = json.loads(Path(args.snapshot).read_text(encoding="utf-8"))

    if args.dashboard:
        manifests_with_sources: list[tuple[dict[str, Any], str]] = [
            (m, str(p)) for p in (Path(s) for s in args.manifests) for m in _load_manifest_file(p)
        ]
        report = analyze_dashboard(manifests_with_sources, snapshot=snapshot)
    else:
        report = analyze_manifests(manifests, snapshot=snapshot)

    if args.output_format == "markdown":
        if args.dashboard:
            text = _format_dashboard_markdown(report)
        else:
            text = _format_markdown(report)
    else:
        text = json.dumps(report, indent=2, sort_keys=True) + "\n"

    if args.output:
        Path(args.output).write_text(text, encoding="utf-8")
    else:
        print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
