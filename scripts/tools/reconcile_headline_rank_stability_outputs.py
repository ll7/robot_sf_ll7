#!/usr/bin/env python3
"""Build a read-only reconciliation packet for headline rank-stability outputs.

The checker consumes artifacts produced by the #3216 headline CI/rank-stability
report and the #3780 seed-sufficiency prep tooling. It does not run campaigns,
submit jobs, or promote paper/dissertation claims.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

SCHEMA_VERSION = "headline_rank_stability_reconciliation_packet.v1"
DEFAULT_PACKET_ID = "headline-rank-stability-post-seed-20260629"
CLAIM_BOUNDARY = (
    "Read-only reconciliation only: no full benchmark campaign run, no Slurm/GPU "
    "submission, and no paper/dissertation claim edits."
)


@dataclass(frozen=True)
class ArtifactSpec:
    """Expected headline rank-stability output artifact."""

    key: str
    relative_path: str
    required: bool
    json_kind: str | None = None


ARTIFACT_SPECS = (
    ArtifactSpec("headline_report_json", "result.json", True, "headline_report"),
    ArtifactSpec("headline_report_markdown", "report.md", True, None),
    ArtifactSpec("seed_sufficiency_json", "seed_sufficiency_analysis.json", True, "seed_analysis"),
    ArtifactSpec(
        "headline_contract_json",
        "headline_rank_stability_contract.json",
        True,
        "headline_contract",
    ),
    ArtifactSpec("headline_pairwise_csv", "headline_rank_stability_pairwise.csv", False, None),
)


def build_packet(
    roots: list[Path],
    *,
    packet_id: str = DEFAULT_PACKET_ID,
    fresh_after: datetime | None = None,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Build the read-only reconciliation packet from expected output artifacts."""

    generated_at = (now or datetime.now(UTC)).astimezone(UTC)
    root_summaries = [_inspect_root(root, fresh_after=fresh_after) for root in roots]
    artifacts = [artifact for root in root_summaries for artifact in root["artifacts"]]
    missing_required = [
        artifact
        for artifact in artifacts
        if artifact["required"] and artifact["status"] == "missing"
    ]
    stale_required = [
        artifact for artifact in artifacts if artifact["required"] and artifact["status"] == "stale"
    ]
    available_required = [
        artifact
        for artifact in artifacts
        if artifact["required"] and artifact["status"] == "available"
    ]
    claim_inputs = _collect_claim_inputs(artifacts)
    cannot_claim = _cannot_claim(
        missing_required=missing_required,
        stale_required=stale_required,
        claim_inputs=claim_inputs,
    )
    readiness = "ready_for_claim_review" if not cannot_claim else "not_claim_ready"
    if missing_required:
        readiness = "blocked_missing_outputs"
    elif stale_required:
        readiness = "blocked_stale_outputs"

    return {
        "schema_version": SCHEMA_VERSION,
        "packet_id": packet_id,
        "generated_at_utc": generated_at.isoformat(),
        "fresh_after_utc": fresh_after.astimezone(UTC).isoformat() if fresh_after else None,
        "claim_boundary": CLAIM_BOUNDARY,
        "roots": root_summaries,
        "summary": {
            "readiness": readiness,
            "root_count": len(root_summaries),
            "artifact_count": len(artifacts),
            "required_available": len(available_required),
            "required_missing": len(missing_required),
            "required_stale": len(stale_required),
            "optional_available": sum(
                1
                for artifact in artifacts
                if not artifact["required"] and artifact["status"] == "available"
            ),
        },
        "claim_inputs": claim_inputs,
        "cannot_claim": cannot_claim,
    }


def render_markdown(packet: dict[str, Any]) -> str:
    """Render a compact Markdown reconciliation packet."""

    summary = packet["summary"]
    lines = [
        f"# Headline Rank-Stability Reconciliation ({packet['packet_id']})",
        "",
        f"- **Readiness**: `{summary['readiness']}`",
        f"- **Generated at UTC**: `{packet['generated_at_utc']}`",
        f"- **Fresh after UTC**: `{packet['fresh_after_utc']}`",
        f"- **Claim boundary**: {packet['claim_boundary']}",
        "",
        "## Artifact Status",
        "",
        "| root | artifact | status | detail |",
        "| --- | --- | --- | --- |",
    ]
    for root in packet["roots"]:
        for artifact in root["artifacts"]:
            detail = artifact.get("detail") or artifact.get("path") or ""
            lines.append(
                f"| `{root['root']}` | `{artifact['key']}` | `{artifact['status']}` | {detail} |"
            )

    lines.extend(["", "## Claim Boundary", ""])
    if packet["cannot_claim"]:
        for blocked in packet["cannot_claim"]:
            lines.append(f"- `{blocked['claim']}`: {blocked['reason']}")
    else:
        lines.append("- No read-only reconciliation blocker detected; still requires claim review.")
    lines.append("")
    return "\n".join(lines)


def _inspect_root(root: Path, *, fresh_after: datetime | None) -> dict[str, Any]:
    artifacts = [_inspect_artifact(root, spec, fresh_after=fresh_after) for spec in ARTIFACT_SPECS]
    return {"root": str(root), "artifacts": artifacts}


def _inspect_artifact(
    root: Path,
    spec: ArtifactSpec,
    *,
    fresh_after: datetime | None,
) -> dict[str, Any]:
    path = _resolve_artifact_path(root, spec.relative_path)
    if path is None:
        return {
            "key": spec.key,
            "relative_path": spec.relative_path,
            "required": spec.required,
            "status": "missing",
            "detail": "expected artifact not found",
        }

    payload: dict[str, Any] | None = None
    parse_error: str | None = None
    if spec.json_kind:
        try:
            loaded = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            parse_error = f"invalid JSON: {exc}"
        else:
            if isinstance(loaded, dict):
                payload = loaded
            else:
                # Fail closed on a structurally wrong artifact (list/primitive) so a later
                # ``payload.get(...)`` cannot raise an uncaught AttributeError.
                parse_error = f"invalid JSON: expected object, got {type(loaded).__name__}"

    artifact_time = _artifact_time(path, payload)
    stale = bool(fresh_after and artifact_time and artifact_time < fresh_after)
    detail = parse_error or (
        f"timestamp {artifact_time.isoformat()} before freshness floor"
        if stale and artifact_time
        else "present"
    )
    status = "stale" if stale else "available"
    if parse_error:
        status = "missing" if spec.required else "stale"

    item: dict[str, Any] = {
        "key": spec.key,
        "relative_path": spec.relative_path,
        "path": str(path),
        "required": spec.required,
        "status": status,
        "detail": detail,
        "artifact_time_utc": artifact_time.isoformat() if artifact_time else None,
    }
    if payload is not None:
        item.update(_claim_fields(spec.json_kind, payload))
    return item


def _resolve_artifact_path(root: Path, relative_path: str) -> Path | None:
    candidates = (root / relative_path, root / "reports" / relative_path)
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None


def _artifact_time(path: Path, payload: dict[str, Any] | None) -> datetime | None:
    if payload:
        for key in ("generated_at_utc", "generated_at"):
            parsed = _parse_datetime(payload.get(key))
            if parsed:
                return parsed
    try:
        return datetime.fromtimestamp(path.stat().st_mtime, UTC)
    except OSError:
        return None


def _parse_datetime(value: Any) -> datetime | None:
    if not isinstance(value, str) or not value.strip():
        return None
    text = value.strip().replace("Z", "+00:00")
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def _claim_fields(json_kind: str | None, payload: dict[str, Any]) -> dict[str, Any]:
    if json_kind == "headline_report":
        decision_packet = payload.get("decision_packet")
        if not isinstance(decision_packet, dict):
            decision_packet = {}
        return {
            "schema_version": payload.get("schema_version"),
            "classification": payload.get("classification"),
            "classification_rationale": payload.get("classification_rationale"),
            "manuscript_table_status": decision_packet.get("manuscript_table_status"),
            "s30_decision_status": decision_packet.get("s30_decision_status"),
            "s30_reasons": decision_packet.get("s30_reasons"),
            "adjacent_overlap_count": decision_packet.get("adjacent_overlap_count"),
            "invalid_metric_claim_count": decision_packet.get("invalid_metric_claim_count"),
            "manuscript_blockers": decision_packet.get("manuscript_blockers"),
        }
    if json_kind == "headline_contract":
        return {
            "schema_version": payload.get("schema_version"),
            "claim_status": payload.get("claim_status"),
            "label": payload.get("label"),
            "promotion_allowed": payload.get("promotion_allowed"),
            "max_seed_count": payload.get("max_seed_count"),
            "missing_durable_roots": payload.get("missing_durable_roots"),
            "row_status_exclusions": payload.get("row_status_exclusions"),
        }
    if json_kind == "seed_analysis":
        summary = payload.get("summary") if isinstance(payload.get("summary"), dict) else {}
        return {
            "schema_version": payload.get("schema_version"),
            "rank_metric": payload.get("rank_metric"),
            "campaign_count": summary.get("campaign_count"),
            "underpowered_or_unstable": summary.get("underpowered_or_unstable"),
        }
    return {}


def _collect_claim_inputs(artifacts: list[dict[str, Any]]) -> dict[str, Any]:
    report = _first_json_artifact(artifacts, "headline_report_json")
    contract = _first_json_artifact(artifacts, "headline_contract_json")
    seed_analysis = _first_json_artifact(artifacts, "seed_sufficiency_json")
    return {
        "headline_report_classification": report.get("classification"),
        "manuscript_table_status": report.get("manuscript_table_status"),
        "s30_decision_status": report.get("s30_decision_status"),
        "s30_reasons": report.get("s30_reasons"),
        "adjacent_overlap_count": report.get("adjacent_overlap_count"),
        "invalid_metric_claim_count": report.get("invalid_metric_claim_count"),
        "manuscript_blockers": report.get("manuscript_blockers"),
        "headline_contract_claim_status": contract.get("claim_status"),
        "headline_contract_label": contract.get("label"),
        "headline_contract_promotion_allowed": contract.get("promotion_allowed"),
        "max_seed_count": contract.get("max_seed_count"),
        "seed_analysis_campaign_count": seed_analysis.get("campaign_count"),
        "seed_analysis_underpowered_or_unstable": seed_analysis.get("underpowered_or_unstable"),
    }


def _first_json_artifact(artifacts: list[dict[str, Any]], key: str) -> dict[str, Any]:
    for artifact in artifacts:
        if artifact["key"] == key and artifact["status"] == "available":
            return artifact
    return {}


def _cannot_claim(
    *,
    missing_required: list[dict[str, Any]],
    stale_required: list[dict[str, Any]],
    claim_inputs: dict[str, Any],
) -> list[dict[str, str]]:
    blocked: list[dict[str, str]] = []
    if missing_required:
        blocked.append(
            {
                "claim": "headline_rank_stability_complete",
                "reason": "required reconciliation artifacts are missing",
            }
        )
    if stale_required:
        blocked.append(
            {
                "claim": "headline_rank_stability_fresh",
                "reason": "required reconciliation artifacts predate the freshness floor",
            }
        )

    classification = claim_inputs.get("headline_report_classification")
    if classification and classification != "paper_grade":
        blocked.append(
            {
                "claim": "paper_grade_headline_planner_ranking",
                "reason": f"headline report classification is {classification!r}, not 'paper_grade'",
            }
        )

    blocked.extend(_decision_packet_blockers(claim_inputs))

    claim_status = claim_inputs.get("headline_contract_claim_status")
    promotion_allowed = claim_inputs.get("headline_contract_promotion_allowed")
    if claim_status and claim_status != "paper_grade":
        blocked.append(
            {
                "claim": "dissertation_ready_rank_stability_claim",
                "reason": f"headline contract claim_status is {claim_status!r}",
            }
        )
    if promotion_allowed is False:
        blocked.append(
            {
                "claim": "promote_headline_rank_stability_outputs",
                "reason": "headline contract promotion_allowed false",
            }
        )

    blocked.append(
        {
            "claim": "new_paper_or_dissertation_text",
            "reason": "this packet is read-only reconciliation and does not edit claims",
        }
    )
    return blocked


def _decision_packet_blockers(claim_inputs: dict[str, Any]) -> list[dict[str, str]]:
    """Return claim blockers derived from the #3216 local decision packet."""
    blocked: list[dict[str, str]] = []
    manuscript_status = claim_inputs.get("manuscript_table_status")
    if manuscript_status and manuscript_status != "ready_for_table_review_no_claim_promotion":
        blocked.append(
            {
                "claim": "manuscript_headline_table_ready",
                "reason": f"decision packet manuscript_table_status is {manuscript_status!r}",
            }
        )

    s30_status = claim_inputs.get("s30_decision_status")
    if s30_status in {"needs_review", "blocked"}:
        reasons = claim_inputs.get("s30_reasons") or []
        reason_text = (
            ", ".join(str(reason) for reason in reasons)
            or f"decision packet s30_decision_status is {s30_status!r}"
        )
        blocked.append(
            {
                "claim": "s30_not_required_by_local_preflight",
                "reason": reason_text,
            }
        )

    adjacent_overlap_count = claim_inputs.get("adjacent_overlap_count")
    if isinstance(adjacent_overlap_count, int) and adjacent_overlap_count > 0:
        blocked.append(
            {
                "claim": "strict_adjacent_planner_ordering",
                "reason": f"{adjacent_overlap_count} adjacent rank confidence interval(s) overlap",
            }
        )

    invalid_metric_claim_count = claim_inputs.get("invalid_metric_claim_count")
    if isinstance(invalid_metric_claim_count, int) and invalid_metric_claim_count > 0:
        blocked.append(
            {
                "claim": "rank_metric_contract_valid",
                "reason": f"{invalid_metric_claim_count} adjacent claim(s) use an invalid metric contract",
            }
        )

    manuscript_blockers = claim_inputs.get("manuscript_blockers")
    if isinstance(manuscript_blockers, list) and manuscript_blockers:
        blocked.append(
            {
                "claim": "manuscript_headline_claim_ready",
                "reason": "decision packet blockers: "
                + ", ".join(str(blocker) for blocker in manuscript_blockers),
            }
        )

    claim_status = claim_inputs.get("headline_contract_claim_status")
    promotion_allowed = claim_inputs.get("headline_contract_promotion_allowed")
    if claim_status and claim_status != "paper_grade":
        blocked.append(
            {
                "claim": "dissertation_ready_rank_stability_claim",
                "reason": f"headline contract claim_status is {claim_status!r}",
            }
        )
    if promotion_allowed is False:
        blocked.append(
            {
                "claim": "promote_headline_rank_stability_outputs",
                "reason": "headline contract promotion_allowed is false",
            }
        )

    blocked.append(
        {
            "claim": "new_paper_or_dissertation_text",
            "reason": "this packet is read-only reconciliation and does not edit claims",
        }
    )
    return blocked


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments."""

    parser = argparse.ArgumentParser(
        description="Reconcile existing headline rank-stability outputs without running campaigns."
    )
    parser.add_argument(
        "roots",
        nargs="+",
        type=Path,
        help="Output root(s) containing result.json/report.md and seed-analysis artifacts.",
    )
    parser.add_argument("--packet-id", default=DEFAULT_PACKET_ID)
    parser.add_argument(
        "--fresh-after",
        help="ISO-8601 freshness floor; older generated_at/file mtimes are marked stale.",
    )
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""

    args = parse_args(argv)
    fresh_after = _parse_datetime(args.fresh_after) if args.fresh_after else None
    if args.fresh_after and fresh_after is None:
        raise SystemExit(f"invalid --fresh-after timestamp: {args.fresh_after}")
    packet = build_packet(args.roots, packet_id=args.packet_id, fresh_after=fresh_after)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "headline_rank_stability_reconciliation.json").write_text(
        json.dumps(packet, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (args.out_dir / "headline_rank_stability_reconciliation.md").write_text(
        render_markdown(packet),
        encoding="utf-8",
    )
    print(
        f"readiness={packet['summary']['readiness']} "
        f"missing={packet['summary']['required_missing']} "
        f"stale={packet['summary']['required_stale']} -> {args.out_dir}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
