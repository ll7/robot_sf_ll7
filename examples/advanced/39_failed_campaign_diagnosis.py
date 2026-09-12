"""Inspect a retained failed-campaign dossier without rerunning it.

This smoke/diagnostic example applies the repository's existing validators in causal
order to a synthetic dossier: identity, startup/environment, execution, row accounting,
producer/finalizer, artifact integrity, and preservation. It reports observed facts,
failed gates, the first decisive failure, retained-but-unpromotable rows, and a bounded
next lane. It never submits a job, chooses parameters, changes retry authority, or treats
the synthetic result as scientific evidence.

Run the default dossier with::

    uv run python examples/advanced/39_failed_campaign_diagnosis.py --json

Use ``--case`` with ``missing_receipt``, ``contradictory_exits``, ``stale_source``,
``corrupt_checksum``, ``duplicate_rows``, ``fallback_rows``, ``incomplete_logs``, or
``scientific_negative`` to exercise the retained-file negative fixtures. The five
``diagnostic_outcome`` values are local presentation lanes assembled from canonical
facts; they are not a new global failure taxonomy or retry policy.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping
from datetime import date
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from scripts.tools import (
    chunk_manifest,
    reconcile_slurm_evidence,
    record_post_campaign_stage_status,
    scheduler_allocation_receipt,
    slurm_job_finalize,
)
from scripts.validation import check_durable_artifact_locality, platform_receipt

REPORT_SCHEMA = "failed_campaign_diagnosis.v1"
DOSSIER_SCHEMA = "failed_campaign_dossier.v1"
DEFAULT_DOSSIER = Path(__file__).resolve().parents[1] / "fixtures" / "failed_campaign_dossier.json"
OWNERS = {
    "identity": "campaign manifest + scheduler_allocation_receipt",
    "startup_environment": "platform_receipt",
    "execution": "scheduler_allocation_receipt",
    "row_accounting": "reconcile_slurm_evidence + issue_3076 row contract",
    "producer_finalizer": "post_campaign_stage_status + slurm_job_finalize",
    "artifact_integrity": "chunk_manifest",
    "preservation": "check_durable_artifact_locality",
}
ROW_STATUSES = frozenset(
    {"native", "adapter", "diagnostic_only", "fallback", "degraded", "unavailable", "failed"}
)
USABLE_ROWS = frozenset({"native", "adapter", "diagnostic_only", "failed"})


class DiagnosisError(ValueError):
    """Refuse malformed or ambiguous dossier input with a stable code."""

    def __init__(self, code: str, message: str) -> None:  # noqa: D107
        super().__init__(message)
        self.code = code


def _load_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DiagnosisError("malformed_json", "dossier JSON is unreadable") from exc
    if not isinstance(value, dict):
        raise DiagnosisError("invalid_schema", "dossier JSON must be an object")
    return value


def _set_path(payload: Any, path: str, value: Any) -> None:
    parts = path.split(".")
    if not parts or any(not part or part.startswith("_") for part in parts):
        raise DiagnosisError("invalid_patch", "fixture patch path is invalid")
    try:
        cursor = payload
        for part in parts[:-1]:
            cursor = cursor[int(part)] if isinstance(cursor, list) else cursor[part]
        if isinstance(cursor, list):
            cursor[int(parts[-1])] = value
        else:
            cursor[parts[-1]] = value
    except (KeyError, IndexError, TypeError, ValueError) as exc:
        raise DiagnosisError("invalid_patch", "fixture patch target is absent") from exc


def _write(root: Path, relative: str, value: str | Mapping[str, Any]) -> None:
    """Write one dossier member below its temporary root."""

    target = (root / relative).resolve()
    if root.resolve() not in target.parents:
        raise DiagnosisError("path_escape", "dossier member escapes its temporary root")
    target.parent.mkdir(parents=True, exist_ok=True)
    text = value if isinstance(value, str) else json.dumps(value, indent=2, sort_keys=True) + "\n"
    target.write_text(text, encoding="utf-8")


def _materialize(dossier: Mapping[str, Any], case: str, root: Path) -> None:
    cases = dossier.get("cases", {})
    spec = cases.get(case) if isinstance(cases, Mapping) else None
    if not isinstance(spec, Mapping):
        raise DiagnosisError("unknown_case", f"unknown dossier case: {case}")
    replacements = spec.get("replace_files", {})
    if not isinstance(replacements, Mapping):
        raise DiagnosisError("invalid_case", "replace_files must be a mapping")
    removals = {str(item) for item in spec.get("remove_files", [])}
    for relative, value in dossier["files"].items():
        relative = str(relative)
        if relative not in removals:
            _write(root, relative, replacements.get(relative, value))
    artifact = dossier.get("artifact")
    if not isinstance(artifact, Mapping):
        raise DiagnosisError("missing_artifact_spec", "artifact specification is missing")
    policy = chunk_manifest.ChunkingPolicy(
        chunk_size_bytes=int(artifact.get("chunk_size_bytes", 4096)),
        full_digest_threshold_bytes=int(artifact.get("full_digest_threshold_bytes", 4096)),
    )
    identity = chunk_manifest.ArtifactIdentity(
        artifact_id=str(artifact.get("artifact_id", "failed-campaign-rows")),
        artifact_version=str(artifact.get("artifact_version", "1.0.0")),
        root_identity=str(artifact.get("root_identity", "artifact")),
        retention_role=str(artifact.get("retention_role", "short-lived")),
    )
    _write(
        root,
        "artifact_manifest.json",
        chunk_manifest.build_manifest(root / "artifact", artifact=identity, policy=policy),
    )
    patches = spec.get("patches", [])
    if not isinstance(patches, list):
        raise DiagnosisError("invalid_case", "patches must be a list")
    for patch in patches:
        if not isinstance(patch, Mapping) or not isinstance(patch.get("file"), str):
            raise DiagnosisError("invalid_patch", "fixture patch must name a file")
        target = (root / str(patch["file"])).resolve()
        if root.resolve() not in target.parents:
            raise DiagnosisError("path_escape", "fixture patch escapes its temporary root")
        payload = _load_object(target)
        _set_path(payload, str(patch.get("path", "")), patch.get("value"))
        _write(root, str(patch["file"]), payload)


def _gate(
    name: str,
    status: str,
    *,
    reason_codes: list[str] | None = None,
    findings: list[str] | None = None,
    observations: list[str] | None = None,
    details: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a stable gate record with canonical codes separate from local findings."""

    result = {
        "name": name,
        "status": status,
        "owner": OWNERS[name],
        "reason_codes": sorted(set(reason_codes or [])),
        "findings": sorted(set(findings or [])),
        "observations": sorted(set(observations or [])),
    }
    if details:
        result["details"] = dict(details)
    return result


def diagnose_dossier(  # noqa: C901, PLR0912, PLR0915 - one bounded causal pass
    dossier_path: str | Path, *, case: str = "base"
) -> dict[str, Any]:
    """Return a deterministic diagnosis for one synthetic dossier case."""

    dossier = _load_object(Path(dossier_path))
    if dossier.get("schema") != DOSSIER_SCHEMA:
        raise DiagnosisError("unsupported_dossier_schema", "dossier schema is unsupported")
    files = dossier.get("files")
    expected = dossier.get("expected_identity")
    as_of = str(dossier.get("as_of", ""))
    if not isinstance(files, Mapping) or not files:
        raise DiagnosisError("missing_dossier_files", "dossier files must be non-empty")
    if not isinstance(expected, Mapping):
        raise DiagnosisError("missing_identity", "dossier expected_identity is missing")
    for relative, value in files.items():
        candidate = Path(str(relative))
        if (
            candidate.is_absolute()
            or ".." in candidate.parts
            or not isinstance(value, (str, Mapping))
        ):
            raise DiagnosisError("invalid_dossier_file", "dossier file entry is invalid")
    try:
        date.fromisoformat(as_of)
    except ValueError as exc:
        raise DiagnosisError("invalid_as_of", "dossier as_of must be an ISO date") from exc
    with TemporaryDirectory(prefix="failed-campaign-diagnosis-") as temporary:
        root = Path(temporary)
        _materialize(dossier, case, root)

        def read(relative: str) -> tuple[dict[str, Any] | None, list[str]]:
            """Load an optional gate input while preserving the rest of the diagnosis."""

            path = root / relative
            if not path.is_file():
                return None, ["missing_receipt" if "scheduler" in relative else "missing_file"]
            try:
                value = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                return None, ["malformed_json"]
            return (value, []) if isinstance(value, dict) else (None, ["invalid_schema"])

        campaign, campaign_errors = read("campaign_manifest.json")
        scheduler, scheduler_errors = read("scheduler_receipt.json")
        identity_findings = campaign_errors + scheduler_errors
        if campaign:
            for field in ("campaign_id", "job_alias", "config_digest"):
                if campaign.get(field) != expected.get(field):
                    identity_findings.append(f"{field}_identity_mismatch")
            if campaign.get("source_commit") != expected.get("source_commit"):
                identity_findings.append("stale_source")
        if campaign and scheduler:
            if scheduler.get("campaign_id") != campaign.get("campaign_id"):
                identity_findings.append("campaign_identity_mismatch")
            if scheduler.get("job_alias") != campaign.get("job_alias"):
                identity_findings.append("job_identity_mismatch")
            submission = scheduler.get("submission", {})
            if isinstance(submission, Mapping) and submission.get(
                "source_config_command_digest"
            ) != campaign.get("config_digest"):
                identity_findings.append("config_identity_mismatch")
        gates = [
            _gate(
                "identity",
                "fail" if identity_findings else "pass",
                findings=identity_findings,
                observations=(
                    [f"scheduler_state={scheduler.get('scheduler_state', 'unknown')}"]
                    if scheduler
                    else []
                ),
            )
        ]

        environment, environment_errors = read("environment_receipt.json")
        startup_codes = list(environment_errors)
        smoke = environment.get("startup_smoke", {}) if environment else {}
        if environment:
            startup_codes += [
                issue.code for issue in platform_receipt.validate_receipt(environment)
            ]
        if not isinstance(smoke, Mapping) or smoke.get("status") != "passed":
            startup_codes.append("startup_smoke_not_passed")
        gates.append(
            _gate(
                "startup_environment",
                "fail" if startup_codes else "pass",
                reason_codes=startup_codes,
                observations=[f"startup_smoke={smoke.get('status', 'unknown')}"],
            )
        )

        execution_codes = []
        if scheduler:
            execution_codes = [
                issue.code for issue in scheduler_allocation_receipt.validate_receipt(scheduler)
            ]
        else:
            execution_codes = ["execution_not_observed"]
        state = scheduler.get("scheduler_state") if scheduler else None
        if state not in {"failed", "completed"}:
            execution_codes.append("non_terminal_scheduler_state")
        execution = scheduler.get("execution", {}) if scheduler else {}
        exit_code = (
            execution.get("exit_code", "unknown") if isinstance(execution, Mapping) else "unknown"
        )
        execution_failed = bool(execution_codes) or state == "failed"
        gates.append(
            _gate(
                "execution",
                "fail" if execution_failed else "pass",
                reason_codes=execution_codes,
                observations=[f"scheduler_state={state}", f"exit_code={exit_code}"],
            )
        )

        row_path = root / "artifact" / "rows.jsonl"
        rows: list[dict[str, Any]] = []
        row_findings: list[str] = []
        if not row_path.is_file():
            row_findings.append("missing_rows")
        else:
            for number, line in enumerate(row_path.read_text(encoding="utf-8").splitlines(), 1):
                try:
                    value = json.loads(line)
                except json.JSONDecodeError:
                    row_findings.append(f"malformed_row_{number}")
                    continue
                if not isinstance(value, dict):
                    row_findings.append(f"non_object_row_{number}")
                else:
                    rows.append(value)
        if len({str(row.get("episode_id")) for row in rows}) != len(rows):
            row_findings.append("duplicate_rows")
        for row in rows:
            if not {"episode_id", "seed", "row_status", "outcome"}.issubset(row):
                row_findings.append("incomplete_row")
            elif row["row_status"] not in ROW_STATUSES:
                row_findings.append("invalid_row_status")
            elif row["row_status"] in {"fallback", "degraded"}:
                row_findings.append("fallback_or_degraded_rows")
        expected_count = int(campaign.get("expected_row_count", 0)) if campaign else 0
        if len(rows) != expected_count:
            row_findings.append("partial_rows")
        accounting = {
            "expected_rows": expected_count,
            "observed_rows": len(rows),
            "retained_diagnostic_rows": sum(row.get("row_status") in USABLE_ROWS for row in rows),
            "row_statuses": sorted(str(row.get("row_status", "unknown")) for row in rows),
        }
        try:
            reconciliation = reconcile_slurm_evidence.reconcile(
                queue_path=root / "queue.yaml",
                submission_manifests=[root / "submission_manifest.yaml"],
                evidence_root=root / "evidence",
                generated_at="2026-09-10T00:00:00+00:00",
            )
        except (OSError, RuntimeError, ValueError):
            reconciliation = {
                "observations": [],
                "errors": ["reconciliation_error"],
                "warnings": [],
            }
        row_findings += ["reconciliation_error"] * bool(reconciliation.get("errors"))
        gates.append(
            _gate(
                "row_accounting",
                "fail" if row_findings else "pass",
                findings=row_findings,
                observations=[f"observed_rows={len(rows)}"],
                details={"reconciliation_errors": len(reconciliation.get("errors", []))},
            )
        )

        report, report_errors = read("report.json")
        producer_findings = list(report_errors)
        producer_observations: list[str] = []
        if not (root / "campaign.log").is_file():
            producer_findings.append("missing_logs")
        stage = None
        finalizer = None
        try:
            stage = record_post_campaign_stage_status.load_stage_status(root / "stage_status.json")
        except (FileNotFoundError, OSError, ValueError):
            producer_findings.append("invalid_stage_status")
        if report:
            if report.get("schema") != "synthetic_campaign_report.v1":
                producer_findings.append("report_schema_mismatch")
            if report.get("row_count") != len(rows):
                producer_findings.append("report_row_count_mismatch")
        if scheduler and stage:
            finalizer = slurm_job_finalize.build_finalization_report(
                issue_number=8902,
                job_id=str(scheduler.get("job_alias", "unknown")),
                job_state={"failed": "FAILED", "completed": "COMPLETED"}.get(str(state), "UNKNOWN"),
                expected_artifacts=["artifact/rows.jsonl"],
                repo_root=root,
                post_campaign_stage_status=root / "stage_status.json",
            )
            if finalizer["classification"] != "success":
                producer_observations.append("finalizer_not_success")
        producer_details = None
        if finalizer and stage:
            producer_details = {
                "classification": finalizer["classification"],
                "artifact_status": finalizer["artifact_status"],
                "claim_boundary": finalizer["claim_boundary"],
                "stage_status": stage["post_campaign_stage"]["status"],
            }
        gates.append(
            _gate(
                "producer_finalizer",
                "fail" if producer_findings or producer_observations else "pass",
                findings=producer_findings,
                observations=producer_observations,
                details=producer_details,
            )
        )

        artifact_findings: list[str] = []
        try:
            manifest = chunk_manifest.load_manifest_file(root / "artifact_manifest.json")
            verification = chunk_manifest.verify_manifest(root / "artifact", manifest=manifest)
            artifact_findings = [str(item["code"]) for item in verification["failures"]]
            artifact_details = {"manifest_id": manifest["manifest_id"]}
        except chunk_manifest.ChunkManifestError as exc:
            artifact_details = None
            artifact_findings = [exc.code]
        gates.append(
            _gate(
                "artifact_integrity",
                "fail" if artifact_findings else "pass",
                findings=artifact_findings,
                details=artifact_details,
            )
        )

        try:
            custody = check_durable_artifact_locality.audit_locality(
                root / "preservation.json", as_of=date.fromisoformat(as_of)
            )
            preservation_findings = [finding.code for finding in custody.findings]
            custody_status = custody.status
        except (OSError, ValueError):
            preservation_findings, custody_status = ["invalid_projection"], "unknown"
        gates.append(
            _gate(
                "preservation",
                "fail" if preservation_findings else "pass",
                findings=preservation_findings,
                observations=[f"custody_status={custody_status}"],
            )
        )

        by_name = {gate["name"]: gate for gate in gates}
        if "stale_source" in by_name["identity"]["findings"]:
            outcome = "retry_not_authorized"
        elif by_name["identity"]["status"] == "fail":
            outcome = "unknown"
        elif (
            set(by_name["row_accounting"]["findings"]) - {"partial_rows"}
            or by_name["producer_finalizer"]["findings"]
            or by_name["execution"]["reason_codes"]
        ):
            outcome = "unknown"
        elif execution_failed and state == "failed":
            outcome = "retry_requires_differential"
        elif execution_failed:
            outcome = "unknown"
        elif by_name["row_accounting"]["status"] == "pass" and any(
            by_name[name]["status"] == "fail" for name in ("artifact_integrity", "preservation")
        ):
            outcome = "artifact_recovery_only"
        elif (
            all(gate["status"] == "pass" for gate in gates)
            and campaign
            and campaign.get("scientific_outcome") == "negative"
        ):
            outcome = "scientific_negative_terminal"
        else:
            outcome = "unknown"
        first = next((gate for gate in gates if gate["status"] == "fail"), None)
        retained = "diagnostic_only" if accounting["retained_diagnostic_rows"] else "excluded"
        report_payload = {
            "schema": REPORT_SCHEMA,
            "evidence_tier": "smoke/diagnostic",
            "claim_boundary": "Synthetic retained-file diagnosis only; no job submission, retry authorization, benchmark result, or scientific claim is produced.",
            "campaign": {
                "campaign_id": campaign.get("campaign_id") if campaign else None,
                "job_alias": campaign.get("job_alias") if campaign else None,
                "source_commit": campaign.get("source_commit") if campaign else None,
            },
            "observed_facts": gates,
            "failed_gates": [gate["name"] for gate in gates if gate["status"] == "fail"],
            "first_decisive_failure": first,
            "secondary_findings": [
                {"gate": gate["name"], "finding": finding}
                for gate in gates
                if first is None or gate["name"] != first["name"]
                for finding in (*gate["reason_codes"], *gate["findings"])
            ],
            "row_accounting": accounting,
            "retained_useful_artifacts": [
                {
                    "path": "artifact/rows.jsonl",
                    "row_count": len(rows),
                    "status": retained,
                    "promotable": False,
                }
            ],
            "reconciliation": {
                "observations": [
                    {
                        key: row.get(key)
                        for key in ("queue_id", "seed", "status", "job_ids", "notes")
                    }
                    for row in reconciliation.get("observations", [])
                ],
                "warnings": sorted(str(item) for item in reconciliation.get("warnings", [])),
            },
            "report_manifest": {
                "status": report.get("status") if report else "unavailable",
                "scientific_outcome": report.get("scientific_outcome") if report else "unavailable",
            },
            "diagnostic_outcome": outcome,
            "rerun": {
                "authorized": False,
                "eligibility": outcome,
                "required_differential": {
                    "retry_requires_differential": "repair or replace the failed execution/producer path; do not repeat the same source, config, and command",
                    "retry_not_authorized": "re-establish source/config/job authority before any retry",
                    "artifact_recovery_only": "repair or preserve retained artifacts; do not rerun",
                    "scientific_negative_terminal": "none; the synthetic terminal result grants no retry authority",
                    "unknown": "repair missing or contradictory evidence before deciding",
                }[outcome],
            },
        }
    return report_payload


def _format_text(report: Mapping[str, Any]) -> str:
    """Render a compact operator-facing summary."""

    first = report["first_decisive_failure"]
    name = first["name"] if isinstance(first, Mapping) else "none"
    return "\n".join(
        (
            f"diagnostic_outcome: {report['diagnostic_outcome']}",
            f"first_decisive_failure: {name}",
            f"failed_gates: {', '.join(report['failed_gates']) or 'none'}",
            f"retained_rows: {report['row_accounting']['retained_diagnostic_rows']}",
            "rerun_authorized: false",
        )
    )


def main(argv: list[str] | None = None) -> int:
    """Run the local dossier diagnosis CLI."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dossier", type=Path, default=DEFAULT_DOSSIER)
    parser.add_argument("--case", default="base")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    try:
        report = diagnose_dossier(args.dossier, case=args.case)
    except DiagnosisError as exc:
        print(json.dumps({"schema": REPORT_SCHEMA, "error": exc.code}, sort_keys=True))
        return 2
    print(json.dumps(report, indent=2, sort_keys=True) if args.json else _format_text(report))
    return 0


if __name__ == "__main__":
    sys.exit(main())
