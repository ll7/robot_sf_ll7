"""Generate a compact research packet from a campaign manifest.

The runner is deliberately conservative: loading a manifest can create a dry-run
packet, but it cannot turn proposal text into benchmark evidence. Configured
validation commands run only when explicitly requested.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

import yaml

REQUIRED_SECTIONS = (
    "campaign",
    "scenario_suite",
    "planners",
    "seed_policy",
    "metrics",
    "row_status_policy",
    "outputs",
    "durable_evidence",
    "validation",
)
FAIL_CLOSED_ROW_STATUSES = {"not_available", "failed", "blocked"}


class ManifestError(ValueError):
    """Raised when the manifest cannot support a fail-closed packet."""


@dataclass(frozen=True)
class RunnerOptions:
    """Runtime switches that affect evidence strength."""

    execute_validation: bool
    require_configured_outputs: bool


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _json_default(value: object) -> str:
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _load_manifest(path: Path) -> dict[str, Any]:
    manifest = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise ManifestError("Manifest must load as a mapping")
    return manifest


def _posix_path(value: str, field_name: str) -> PurePosixPath:
    path = PurePosixPath(value)
    if path.is_absolute():
        raise ManifestError(f"{field_name} must be a repository-relative path: {value}")
    if ".." in path.parts:
        raise ManifestError(f"{field_name} must not traverse parent directories: {value}")
    return path


def _list_field(
    mapping: dict[str, Any],
    key: str,
    field_name: str,
    *,
    required: bool,
) -> list[Any]:
    value = mapping.get(key)
    if value is None:
        if required:
            raise ManifestError(f"{field_name} must be a list")
        return []
    if not isinstance(value, list):
        raise ManifestError(f"{field_name} must be a list")
    return value


def _validate_required_sections(manifest: dict[str, Any]) -> None:
    for section in REQUIRED_SECTIONS:
        if section not in manifest:
            raise ManifestError(f"Missing required section: {section}")
        if not isinstance(manifest[section], dict):
            raise ManifestError(f"{section} must be a mapping")


def _validate_campaign(manifest: dict[str, Any]) -> None:
    campaign = manifest["campaign"]
    if not isinstance(campaign, dict):
        raise ManifestError("campaign must be a mapping")
    if not str(campaign.get("id", "")).strip():
        raise ManifestError("campaign.id is required")
    if not str(campaign.get("claim_boundary", "")).strip():
        raise ManifestError("campaign.claim_boundary is required")
    if not str(campaign.get("evidence_tier", "")).strip():
        raise ManifestError("campaign.evidence_tier is required")


def _validate_row_status_policy(manifest: dict[str, Any]) -> None:
    row_status_policy = manifest["row_status_policy"]
    allowed_values = set(
        _list_field(
            row_status_policy,
            "allowed_values",
            "row_status_policy.allowed_values",
            required=True,
        )
    )
    success_values = set(
        _list_field(
            row_status_policy,
            "success_values",
            "row_status_policy.success_values",
            required=True,
        )
    )
    fail_closed_values = set(
        _list_field(
            row_status_policy,
            "fail_closed_values",
            "row_status_policy.fail_closed_values",
            required=True,
        )
    )
    if not allowed_values:
        raise ManifestError("row_status_policy.allowed_values is required")
    if not success_values <= allowed_values:
        raise ManifestError("row_status_policy.success_values must be allowed values")
    if not fail_closed_values <= allowed_values:
        raise ManifestError("row_status_policy.fail_closed_values must be allowed values")
    if success_values & (FAIL_CLOSED_ROW_STATUSES | fail_closed_values):
        raise ManifestError("Fail-closed row statuses cannot be success values")


def _validate_outputs(manifest: dict[str, Any], *, require_configured_outputs: bool) -> None:
    outputs = manifest["outputs"]
    if not isinstance(outputs, dict):
        raise ManifestError("outputs must be a mapping")
    local_root = _posix_path(str(outputs.get("local_root", "")), "outputs.local_root")
    if local_root.parts[:1] != ("output",):
        raise ManifestError("outputs.local_root must stay under output/")
    if outputs.get("disposable") is not True:
        raise ManifestError("outputs.disposable must be true for local output roots")

    if require_configured_outputs:
        repo_root = _repo_root()
        for required_path in _list_field(
            outputs,
            "required_paths",
            "outputs.required_paths",
            required=True,
        ):
            required_posix_path = _posix_path(str(required_path), "outputs.required_paths[]")
            candidate = repo_root / local_root / required_posix_path
            if not candidate.exists():
                raise ManifestError(f"Configured output path is missing: {candidate}")


def _validate_durable_evidence(manifest: dict[str, Any]) -> None:
    durable_evidence = manifest["durable_evidence"]
    if not isinstance(durable_evidence, dict) or not isinstance(durable_evidence.get("plan"), dict):
        raise ManifestError("durable_evidence.plan is required")
    durable_plan = durable_evidence["plan"]
    durable_path_text = str(durable_plan.get("path", "")).strip()
    if not durable_path_text:
        raise ManifestError("durable_evidence.plan.path is required")
    durable_path = _posix_path(durable_path_text, "durable_evidence.plan.path")
    if durable_path.parts[:1] == ("output",):
        raise ManifestError("durable_evidence.plan.path must not point into output/")
    if durable_plan.get("required_before_claim") is not True:
        raise ManifestError("durable_evidence.plan.required_before_claim must be true")


def _validate_manifest(manifest: dict[str, Any], *, options: RunnerOptions) -> None:
    _validate_required_sections(manifest)
    _validate_campaign(manifest)
    _validate_row_status_policy(manifest)
    _validate_outputs(
        manifest,
        require_configured_outputs=options.require_configured_outputs,
    )
    _validate_durable_evidence(manifest)


def _scenario_ids(manifest: dict[str, Any]) -> list[str]:
    scenario_suite = manifest["scenario_suite"]
    ids = _list_field(scenario_suite, "scenario_ids", "scenario_suite.scenario_ids", required=False)
    if not ids:
        ids = _list_field(
            scenario_suite,
            "scenario_families",
            "scenario_suite.scenario_families",
            required=False,
        )
    if not ids:
        ids = ["unspecified"]
    return [str(item) for item in ids]


def _seeds(manifest: dict[str, Any]) -> list[int | str]:
    seed_policy = manifest["seed_policy"]
    seeds = _list_field(seed_policy, "seeds", "seed_policy.seeds", required=False)
    if not seeds:
        seeds = ["unspecified"]
    return list(seeds)


def _planner_rows(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    planners = manifest["planners"]
    rows = planners.get("rows")
    if not isinstance(rows, list) or not rows:
        raise ManifestError("planners.rows must contain at least one planner row")
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ManifestError(f"planners.rows[{index}] must be a mapping")
    return rows


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=_repo_root(),
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _build_rows(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    metric_ids = [str(metric_id) for metric_id in manifest["metrics"].get("ids", [])]
    for scenario_id in _scenario_ids(manifest):
        for planner in _planner_rows(manifest):
            for seed in _seeds(manifest):
                row = {
                    "campaign_id": manifest["campaign"]["id"],
                    "scenario_id": scenario_id,
                    "planner_id": planner.get("planner_id", "unknown"),
                    "adapter_mode": planner.get("adapter_mode", "unknown"),
                    "seed": seed,
                    "row_status": "diagnostic_only",
                    "status_reason": "dry-run manifest packet; no benchmark row was executed",
                }
                for metric_id in metric_ids:
                    row[metric_id] = None
                rows.append(row)
    return rows


def _write_json(path: Path, payload: dict[str, Any] | list[dict[str, Any]]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=_json_default) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, sort_keys=True, default=_json_default) + "\n")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0])
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _run_validation_commands(manifest: dict[str, Any], *, execute: bool) -> list[dict[str, Any]]:
    commands = _list_field(
        manifest["validation"],
        "commands",
        "validation.commands",
        required=False,
    )
    results: list[dict[str, Any]] = []
    for command in commands:
        command_text = str(command)
        result: dict[str, Any] = {
            "command": command_text,
            "executed": execute,
        }
        if execute:
            completed = subprocess.run(
                command_text,
                cwd=_repo_root(),
                shell=True,
                text=True,
                capture_output=True,
                check=False,
            )
            result.update(
                {
                    "returncode": completed.returncode,
                    "stdout_excerpt": completed.stdout[-2000:],
                    "stderr_excerpt": completed.stderr[-2000:],
                }
            )
        results.append(result)
    return results


def _summary_payload(
    manifest: dict[str, Any],
    manifest_path: Path,
    rows: list[dict[str, Any]],
    validation_results: list[dict[str, Any]],
) -> dict[str, Any]:
    status_counts = Counter(str(row["row_status"]) for row in rows)
    outputs = manifest["outputs"]
    return {
        "campaign_id": manifest["campaign"]["id"],
        "source_manifest": str(manifest_path),
        "git_commit": _git_commit(),
        "evidence_tier": manifest["campaign"]["evidence_tier"],
        "claim_boundary": manifest["campaign"]["claim_boundary"],
        "final_decision": "diagnostic",
        "planner_rows": _planner_rows(manifest),
        "row_status_summary": dict(sorted(status_counts.items())),
        "artifact_paths": {
            "manifest_resolved": "manifest_resolved.json",
            "rows": "rows.jsonl",
            "report": "report.md",
            "context_note": "context_note.md",
            "configured_local_root": outputs["local_root"],
            "durable_evidence_plan": manifest["durable_evidence"]["plan"],
        },
        "caveats": [
            "Generated packet rows are diagnostic dry-run rows unless a campaign-specific runner "
            "replaces them with executed measurements.",
            "Fallback, degraded, failed, blocked, and not_available rows are not success evidence.",
            "Local output paths under output/ are disposable until represented by durable evidence.",
        ],
        "validation_results": validation_results,
    }


def _write_report(path: Path, summary: dict[str, Any], rows: list[dict[str, Any]]) -> None:
    lines = [
        f"# Research Campaign Packet: {summary['campaign_id']}",
        "",
        f"- Evidence tier: `{summary['evidence_tier']}`",
        f"- Final decision: `{summary['final_decision']}`",
        f"- Source manifest: `{summary['source_manifest']}`",
        f"- Git commit: `{summary['git_commit']}`",
        "",
        "## Claim Boundary",
        "",
        str(summary["claim_boundary"]).strip(),
        "",
        "## Row Status Summary",
        "",
    ]
    for status, count in summary["row_status_summary"].items():
        lines.append(f"- `{status}`: {count}")
    lines.extend(
        [
            "",
            "## Rows",
            "",
            "| scenario_id | planner_id | adapter_mode | seed | row_status |",
            "| --- | --- | --- | --- | --- |",
        ]
    )
    for row in rows:
        lines.append(
            "| {scenario_id} | {planner_id} | {adapter_mode} | {seed} | {row_status} |".format(
                **row
            )
        )
    lines.extend(
        [
            "",
            "## Caveats",
            "",
        ]
    )
    for caveat in summary["caveats"]:
        lines.append(f"- {caveat}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_context_note(path: Path, summary: dict[str, Any]) -> None:
    body = [
        f"# {summary['campaign_id']} Research Campaign Packet",
        "",
        "Status: generated draft from `scripts/validation/run_research_campaign_manifest.py`.",
        "",
        "## Claim Boundary",
        "",
        str(summary["claim_boundary"]).strip(),
        "",
        "## Evidence",
        "",
        f"- Evidence tier: `{summary['evidence_tier']}`",
        f"- Final decision: `{summary['final_decision']}`",
        f"- Source manifest: `{summary['source_manifest']}`",
        "",
        "## Durable Evidence Plan",
        "",
        f"- `{summary['artifact_paths']['durable_evidence_plan']}`",
        "",
    ]
    path.write_text("\n".join(body), encoding="utf-8")


def run(manifest_path: Path, output_dir: Path, *, options: RunnerOptions) -> dict[str, Any]:
    """Validate a manifest, emit packet files, and return the summary payload."""
    manifest = _load_manifest(manifest_path)
    _validate_manifest(manifest, options=options)

    output_dir.mkdir(parents=True, exist_ok=True)
    rows = _build_rows(manifest)
    validation_results = _run_validation_commands(
        manifest,
        execute=options.execute_validation,
    )
    summary = _summary_payload(manifest, manifest_path, rows, validation_results)

    resolved_manifest = {
        "schema": "research_campaign_manifest_packet.v1",
        "source_manifest": str(manifest_path),
        "manifest": manifest,
    }
    _write_json(output_dir / "manifest_resolved.json", resolved_manifest)
    _write_jsonl(output_dir / "rows.jsonl", rows)
    _write_csv(output_dir / "campaign_table.csv", rows)
    _write_json(output_dir / "summary.json", summary)
    _write_report(output_dir / "report.md", summary, rows)
    _write_context_note(output_dir / "context_note.md", summary)
    return summary


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate a research campaign manifest and emit a compact packet.",
    )
    parser.add_argument("manifest", type=Path, help="Path to the research campaign manifest YAML.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Packet output directory. Defaults to <manifest outputs.local_root>/packet.",
    )
    parser.add_argument(
        "--execute-validation",
        action="store_true",
        help="Execute manifest validation commands. By default they are recorded only.",
    )
    parser.add_argument(
        "--require-configured-outputs",
        action="store_true",
        help="Fail if outputs.required_paths are not already present under outputs.local_root.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Run the command-line interface."""
    args = _parse_args(sys.argv[1:] if argv is None else argv)
    try:
        manifest = _load_manifest(args.manifest)
        output_dir = args.output_dir
        if output_dir is None:
            output_dir = _repo_root() / str(manifest["outputs"]["local_root"]) / "packet"
        summary = run(
            args.manifest,
            output_dir,
            options=RunnerOptions(
                execute_validation=args.execute_validation,
                require_configured_outputs=args.require_configured_outputs,
            ),
        )
    except (KeyError, ManifestError, OSError, yaml.YAMLError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps({"ok": True, "output_dir": str(output_dir), "summary": summary}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
