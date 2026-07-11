#!/usr/bin/env python3
"""Analyze verified #3216 campaign artifacts without starting a new campaign.

The runner accepts explicit local paths so harvest location and host routing stay
out of version control.  It verifies the harvest-completion marker, captures the
soft SNQI (Social Navigation Quality Index) contract failure that limits claims,
and delegates all CI and rank-stability statistics to the canonical #3216 CLI.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import subprocess
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[2]
_BUILDER = _REPO_ROOT / "scripts/benchmark/build_headline_ci_rank_stability_report_issue_3216.py"
_DEFAULT_PLANNER_CONFIG = (
    _REPO_ROOT / "configs/benchmarks/paper_experiment_matrix_v1_scenario_horizons_h500_s20.yaml"
)
_HARVEST_MARKER = re.compile(r"(?m)^VERIFIED_COMPLETE(?:\s|$)")
_ENFORCEMENT = re.compile(r"snqi_contract\.enforcement=(\w+)")


def _sha256(path: Path) -> str:
    """Return the SHA-256 digest of one regular file."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_mapping(path: Path, *, label: str) -> dict[str, Any]:
    """Load a JSON mapping or fail with a path-specific error."""

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"{label} is not valid JSON: {path}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must be a JSON object: {path}")
    return payload


def _required_file(path: Path, *, label: str) -> Path:
    """Return an existing regular file, otherwise raise a clear error."""

    if not path.is_file():
        raise FileNotFoundError(f"{label} not found: {path}")
    return path


def verify_harvest(log_path: Path) -> None:
    """Fail closed unless the external harvest explicitly reports completion."""

    text = _required_file(log_path, label="harvest log").read_text(encoding="utf-8")
    if not _HARVEST_MARKER.search(text):
        raise ValueError(f"harvest log lacks standalone VERIFIED_COMPLETE marker: {log_path}")


def _failed_contract_checks(diagnostics: Mapping[str, Any]) -> list[dict[str, float | str]]:
    """Return threshold failures recorded by the campaign SNQI diagnostics."""

    thresholds = diagnostics.get("thresholds")
    if not isinstance(thresholds, Mapping):
        raise ValueError("SNQI diagnostics lacks thresholds needed to identify the failed check")

    checks = (
        ("rank_alignment_spearman", "rank_alignment_fail", "below"),
        ("outcome_separation", "outcome_separation_fail", "below"),
        ("dominant_component_mean_abs", "max_component_dominance_fail", "above"),
    )
    failed: list[dict[str, float | str]] = []
    for value_key, threshold_key, direction in checks:
        value = diagnostics.get(value_key)
        threshold = thresholds.get(threshold_key)
        if (
            isinstance(value, bool)
            or isinstance(threshold, bool)
            or not isinstance(value, (int, float))
            or not isinstance(threshold, (int, float))
            or not math.isfinite(float(value))
            or not math.isfinite(float(threshold))
        ):
            raise ValueError(
                f"SNQI diagnostics {value_key}/{threshold_key} must be finite numeric values"
            )
        violates = value < threshold if direction == "below" else value > threshold
        if violates:
            failed.append(
                {
                    "check": value_key,
                    "value": float(value),
                    "fail_threshold": float(threshold),
                    "direction": direction,
                }
            )
    if not failed:
        raise ValueError("SNQI status is fail but diagnostics identify no violated fail threshold")
    return failed


def load_soft_snqi_failure(campaign_root: Path) -> tuple[Path, Path, dict[str, Any]]:
    """Load and validate the warning-level SNQI failure for a recovered campaign."""

    reports = campaign_root / "reports"
    summary_path = _required_file(reports / "campaign_summary.json", label="campaign summary")
    diagnostics_path = _required_file(reports / "snqi_diagnostics.json", label="SNQI diagnostics")
    summary = _load_mapping(summary_path, label="campaign summary")
    diagnostics = _load_mapping(diagnostics_path, label="SNQI diagnostics")
    campaign = summary.get("campaign")
    if not isinstance(campaign, Mapping):
        raise ValueError("campaign summary lacks campaign metadata")
    finished_at_utc = campaign.get("finished_at_utc")
    if not isinstance(finished_at_utc, str) or not finished_at_utc:
        raise ValueError(
            "campaign summary lacks finished_at_utc for reproducible report timestamps"
        )
    if campaign.get("snqi_contract_status") != "fail":
        raise ValueError("campaign summary does not record snqi_contract_status=fail")

    warnings = summary.get("warnings")
    if not isinstance(warnings, list):
        raise ValueError("campaign summary lacks warnings needed to verify warn enforcement")
    warning_text = " ".join(str(item) for item in warnings)
    enforcement_match = _ENFORCEMENT.search(warning_text)
    if enforcement_match is None or enforcement_match.group(1) != "warn":
        raise ValueError("campaign summary does not record snqi_contract.enforcement=warn")
    if (
        diagnostics.get("contract_status") != "fail"
        or diagnostics.get("contract_enforcement") != "warn"
    ):
        raise ValueError("SNQI diagnostics do not confirm status=fail under enforcement=warn")

    failure = {
        "contract_status": "fail",
        "enforcement": "warn",
        "campaign_finished_at_utc": finished_at_utc,
        "failed_checks": _failed_contract_checks(diagnostics),
        "warning": warning_text,
    }
    return summary_path, diagnostics_path, failure


def _failure_reason(failure: Mapping[str, Any]) -> str:
    """Format the precise contract limitation passed to the canonical CLI."""

    checks = failure["failed_checks"]
    rendered = ", ".join(
        f"{check['check']}={check['value']:.6g} ({check['direction']} fail threshold "
        f"{check['fail_threshold']:.6g})"
        for check in checks
    )
    return (
        "SNQI contract status=fail with snqi_contract.enforcement=warn; "
        f"failed check(s): {rendered}"
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse explicit harvested-artifact and output paths."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-root", type=Path, required=True)
    parser.add_argument("--harvest-log", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--planner-config", type=Path, default=_DEFAULT_PLANNER_CONFIG)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--rank-resamples", type=int, default=500)
    return parser.parse_args(argv)


def run(args: argparse.Namespace) -> int:
    """Verify inputs, invoke the canonical report builder, and write provenance."""

    verify_harvest(args.harvest_log)
    if not args.campaign_root.is_dir():
        raise FileNotFoundError(f"campaign root not found: {args.campaign_root}")
    _required_file(args.planner_config, label="planner config")
    summary_path, diagnostics_path, failure = load_soft_snqi_failure(args.campaign_root)
    seed_rows = _required_file(
        args.campaign_root / "reports/seed_episode_rows.csv", label="seed episode rows"
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        str(_BUILDER),
        "--campaign",
        str(args.campaign_root),
        "--rank-metric",
        "snqi",
        "--invalid-rank-metric-reason",
        _failure_reason(failure),
        "--expected-planners-from-config",
        str(args.planner_config),
        "--bootstrap-samples",
        str(args.bootstrap_samples),
        "--rank-resamples",
        str(args.rank_resamples),
        "--generated-at-utc",
        str(failure["campaign_finished_at_utc"]),
        "--output-dir",
        str(args.output_dir),
    ]
    completed = subprocess.run(command, check=False)
    if completed.returncode:
        return completed.returncode

    provenance = {
        "schema_version": "issue_5247_verified_harvest_rank_stability.v1",
        "evidence_status": "diagnostic-only",
        "claim_boundary": (
            "The complete harvested episode set is analyzed without rerunning a campaign. "
            "SNQI rank claims remain blocked because the source campaign records a warn-level "
            "SNQI contract failure."
        ),
        "input_sha256": {
            "campaign_summary.json": _sha256(summary_path),
            "snqi_diagnostics.json": _sha256(diagnostics_path),
            "seed_episode_rows.csv": _sha256(seed_rows),
        },
        "snqi_contract_failure": failure,
        "output_sha256": {
            "result.json": _sha256(args.output_dir / "result.json"),
            "report.md": _sha256(args.output_dir / "report.md"),
        },
    }
    (args.output_dir / "analysis_provenance.json").write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"verified_harvest=true output_dir={args.output_dir}")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """Run the verified-harvest analysis and render clear fail-closed errors."""

    try:
        return run(parse_args(argv))
    except (FileNotFoundError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
