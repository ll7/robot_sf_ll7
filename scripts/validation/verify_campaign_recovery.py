#!/usr/bin/env python3
"""Verify campaign resume and retry behavior before scarce submissions.

Evaluates interrupted and resumed campaign executions to guarantee:
- Completed valid identities are preserved without silent rerun or overwriting.
- Retries are admitted only for documented infrastructure-class interruptions.
- Authority-bearing inputs (commit, config, model) have not drifted.
- Degraded or fallback executions cannot become clean success through resume.
- Lineage across attempts and scheduler jobs is preserved.
- Output reconciles exactly to the expected-row ledger.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml
from jsonschema import Draft202012Validator

SCHEMA_VERSION = "campaign_recovery_receipt.v1"
SCHEMA_PATH = Path("docs/contracts/campaign_recovery_receipt.v1.schema.json")
SUPPORTED_RUNNERS = frozenset({"benchmark_matrix", "slurm_array"})
INFRASTRUCTURE_REASONS = frozenset(
    {
        "cluster_filesystem_interruption",
        "network_interruption",
        "node_failure",
        "preemption",
        "scheduler_requeue",
        "timeout",
        "walltime_kill",
    }
)
AUTHORITY_KEYS = (
    "commit",
    "source_commit",
    "config_hash",
    "config_sha256",
    "model_hash",
    "model_digest",
)
VALID_COMPLETED_STATUSES = frozenset({"present", "success", "completed"})


def _read_fixture(path: Path) -> dict[str, Any]:
    """Read fixture bundle from JSON or YAML."""
    if not path.is_file():
        raise FileNotFoundError(f"Fixture file not found: {path}")
    raw = path.read_text(encoding="utf-8")
    data = yaml.safe_load(raw) if path.suffix.lower() in (".yaml", ".yml") else json.loads(raw)
    if not isinstance(data, dict):
        raise ValueError(f"Fixture at {path} must be a dictionary")
    return data


def _check_campaign_and_runner(packet: dict[str, Any]) -> tuple[str, str, list[str]]:
    """Validate campaign_id and runner_class."""
    discrepancies: list[str] = []
    campaign_id = str(packet.get("campaign_id", "")).strip()
    if not campaign_id:
        discrepancies.append("missing_campaign_id: campaign_id must be provided")

    runner = str(packet.get("runner_class", "")).strip()
    if not runner:
        discrepancies.append("missing_runner_class: runner_class must be provided")
    elif runner not in SUPPORTED_RUNNERS:
        discrepancies.append(
            f"unsupported_runner: runner '{runner}' not in {sorted(SUPPORTED_RUNNERS)}"
        )
    return campaign_id, runner, discrepancies


def _check_authority_drift(init_auth: dict[str, Any], res_auth: dict[str, Any]) -> list[str]:
    """Check for drift in authority-bearing inputs across resume attempts."""
    discrepancies: list[str] = []
    for key in AUTHORITY_KEYS:
        if key in init_auth and key in res_auth and init_auth[key] != res_auth[key]:
            discrepancies.append(
                f"authority_input_drift: {key} changed from {init_auth[key]} to {res_auth[key]}"
            )
    return discrepancies


def _check_interruption_admission(
    packet: dict[str, Any], initial_run: dict[str, Any]
) -> tuple[str | None, str | None, bool, list[str]]:
    """Evaluate whether interruption class and reason permit campaign retry."""
    interruption = packet.get("interruption", {})
    ireason_raw = interruption.get("reason")
    ireason = str(ireason_raw).strip().lower() if ireason_raw else None
    iclass_raw = interruption.get("class")
    iclass = str(iclass_raw).strip().lower() if iclass_raw else None

    is_infra = (iclass == "infrastructure") or (
        ireason is not None and ireason in INFRASTRUCTURE_REASONS
    )
    discrepancies: list[str] = []
    if not is_infra and (initial_run.get("status") in ("failed", "error") or ireason):
        discrepancies.append(
            f"outcome_driven_retry_rejected: interruption '{ireason}' (class '{iclass}') is not an admitted infrastructure failure"
        )
    norm_iclass = iclass if iclass in ("infrastructure", "outcome", "unknown") else None
    return ireason, norm_iclass, is_infra, discrepancies


def _check_row_preservation_and_status(
    initial_rows: list[dict[str, Any]],
    resumed_rows: list[dict[str, Any]],
    retry_admitted: bool,
) -> tuple[dict[str, dict[str, Any]], list[str]]:
    """Enforce preservation of completed rows, prevent degraded promotion, and reject unadmitted retries."""
    discrepancies: list[str] = []
    completed_initial: dict[str, dict[str, Any]] = {
        r["row_id"]: r
        for r in initial_rows
        if str(r.get("status", "")).lower() in VALID_COMPLETED_STATUSES
    }
    degraded_initial: dict[str, dict[str, Any]] = {
        r["row_id"]: r
        for r in initial_rows
        if str(r.get("status", "")).lower() in ("degraded", "fallback")
    }
    failed_initial: dict[str, dict[str, Any]] = {
        r["row_id"]: r
        for r in initial_rows
        if str(r.get("status", "")).lower() in ("failed", "error")
    }

    for r in resumed_rows:
        rid = r.get("row_id")
        if rid in completed_initial:
            discrepancies.append(
                f"completed_identity_overwritten: valid completed row {rid} was rerun or overwritten"
            )
        if rid in degraded_initial:
            init_st = str(degraded_initial[rid].get("status", "")).lower()
            res_st = str(r.get("status", "")).lower()
            if res_st in ("present", "success", "native"):
                discrepancies.append(
                    f"degraded_became_success: row {rid} was {init_st} but resumed as {res_st}"
                )
        if rid in failed_initial and not retry_admitted:
            discrepancies.append(
                f"unadmitted_failure_retry: row {rid} failed initially but was retried without infrastructure admission"
            )

    reconciled_rows: dict[str, dict[str, Any]] = dict(completed_initial)
    for r in resumed_rows:
        if str(r.get("status", "")).lower() in VALID_COMPLETED_STATUSES:
            reconciled_rows.setdefault(r["row_id"], r)

    return reconciled_rows, discrepancies


def _check_expected_reconciliation(
    expected_rows: list[str], reconciled_ids: set[str]
) -> tuple[bool, list[str]]:
    """Compare reconciled rows against expected-row ledger."""
    expected_ids = set(expected_rows)
    missing_ids = expected_ids - reconciled_ids
    unexpected_ids = reconciled_ids - expected_ids
    discrepancies: list[str] = []
    if missing_ids:
        discrepancies.append(
            f"missing_expected_rows: {len(missing_ids)} expected rows not recovered: {sorted(missing_ids)[:3]}"
        )
    if unexpected_ids:
        discrepancies.append(
            f"unexpected_rows: {len(unexpected_ids)} unexpected rows in reconciled set: {sorted(unexpected_ids)[:3]}"
        )
    return not bool(missing_ids or unexpected_ids), discrepancies


def _build_attempt_lineage(
    initial_run: dict[str, Any],
    resume_attempt: dict[str, Any],
    initial_rows: list[dict[str, Any]],
    resumed_rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[str]]:
    """Assemble attempt metadata and verify lineage continuity."""
    attempts: list[dict[str, Any]] = []
    if initial_run:
        attempts.append(
            {
                "attempt_index": int(initial_run.get("attempt_index", 1)),
                "job_id": str(initial_run.get("job_id", "unknown")),
                "row_count": len(initial_rows),
                "status": str(initial_run.get("status", "unknown")),
            }
        )
    if resume_attempt:
        attempts.append(
            {
                "attempt_index": int(resume_attempt.get("attempt_index", 2)),
                "job_id": str(resume_attempt.get("job_id", "unknown")),
                "row_count": len(resumed_rows),
                "status": str(resume_attempt.get("status", "unknown")),
            }
        )
    discrepancies: list[str] = []
    if len(attempts) < 2 or any(a["job_id"] in ("unknown", "") for a in attempts):
        discrepancies.append("attempt_lineage_incomplete: attempt indices or job IDs missing")
    return attempts, discrepancies


def verify_recovery(packet: dict[str, Any]) -> dict[str, Any]:
    """Verify recovery and resume behavior under fail-closed contracts."""
    campaign_id, runner, init_disc = _check_campaign_and_runner(packet)

    initial_run = packet.get("initial_run", {})
    resume_attempt = packet.get("resume_attempt", {})
    init_auth = packet.get("authority_inputs") or initial_run.get("authority_inputs", {})
    res_auth = resume_attempt.get("authority_inputs", {})
    auth_disc = _check_authority_drift(init_auth, res_auth)

    ireason, iclass, retry_admitted, retry_disc = _check_interruption_admission(packet, initial_run)

    initial_rows = initial_run.get("rows", [])
    resumed_rows = resume_attempt.get("rows", [])
    reconciled_rows, pres_disc = _check_row_preservation_and_status(
        initial_rows, resumed_rows, retry_admitted
    )

    reconciled_ids = set(reconciled_rows.keys())
    expected_rows = packet.get("expected_rows", [])
    matched_ledger, ledger_disc = _check_expected_reconciliation(expected_rows, reconciled_ids)

    attempts, lineage_disc = _build_attempt_lineage(
        initial_run, resume_attempt, initial_rows, resumed_rows
    )

    discrepancies = init_disc + auth_disc + retry_disc + pres_disc + ledger_disc + lineage_disc
    is_unsupported = any(d.startswith("unsupported_runner") for d in discrepancies)
    verdict = "pass" if not discrepancies else "fail"
    status = "unsupported" if is_unsupported else verdict

    return {
        "schema": SCHEMA_VERSION,
        "campaign_id": campaign_id or "unknown",
        "runner_class": runner or "unknown",
        "status": status,
        "verdict": verdict,
        "interruption_reason": ireason,
        "interruption_class": iclass,
        "retry_admitted": retry_admitted,
        "authority_inputs_matched": not bool(auth_disc),
        "completed_rows_preserved": not any(
            d.startswith("completed_identity_overwritten") for d in pres_disc
        ),
        "reconciled_to_expected_ledger": matched_ledger,
        "expected_row_count": len(expected_rows),
        "reconciled_row_count": len(reconciled_ids),
        "discrepancies": discrepancies,
        "attempts": attempts,
        "recomputed_at_utc": datetime.now(UTC).isoformat(),
    }


def validate_schema(data: dict[str, Any], schema_path: Path = SCHEMA_PATH) -> list[str]:
    """Validate recovery receipt against JSON schema draft 2020-12."""
    if not schema_path.is_file():
        return [f"Schema file not found: {schema_path}"]
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    validator = Draft202012Validator(schema)
    errors = sorted(validator.iter_errors(data), key=lambda e: e.path)
    return [f"{'/'.join(map(str, err.path))}: {err.message}" for err in errors]


def format_summary(receipt: dict[str, Any]) -> str:
    """Format human-readable compact summary."""
    lines = [
        f"Campaign Recovery: {receipt.get('campaign_id')} ({receipt.get('runner_class')})",
        f"Status: {receipt.get('status', '').upper()} | Verdict: {receipt.get('verdict', '').upper()}",
        f"Reconciled: {receipt.get('reconciled_row_count')}/{receipt.get('expected_row_count')} rows",
        f"Retry Admitted: {receipt.get('retry_admitted')} (reason: {receipt.get('interruption_reason')})",
    ]
    for a in receipt.get("attempts", []):
        lines.append(f"  Attempt {a['attempt_index']}: job={a['job_id']} status={a['status']}")
    for d in receipt.get("discrepancies", []):
        lines.append(f"  Discrepancy: {d}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    """CLI entry point for campaign recovery verifier."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", type=Path, required=True, help="Path to recovery fixture")
    parser.add_argument("--format", choices=["json", "summary"], default="json")
    parser.add_argument("--output", type=Path, default=None, help="Output path")
    parser.add_argument("--check", action="store_true", help="Fail closed on recovery discrepancy")
    args = parser.parse_args(argv)

    try:
        packet = _read_fixture(args.fixture)
        receipt = verify_recovery(packet)
    except (ValueError, OSError, json.JSONDecodeError, yaml.YAMLError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    if SCHEMA_PATH.is_file():
        errs = validate_schema(receipt)
        if errs:
            print(f"ERROR: Schema validation failed: {errs}", file=sys.stderr)
            return 2

    text = (
        json.dumps(receipt, indent=2, sort_keys=True) + "\n"
        if args.format == "json"
        else format_summary(receipt) + "\n"
    )

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    else:
        sys.stdout.write(text)

    return 2 if (args.check and receipt.get("verdict") != "pass") else 0


if __name__ == "__main__":
    sys.exit(main())
