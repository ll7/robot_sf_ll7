#!/usr/bin/env python3
"""Issue #6481 social-compliance preflight smoke receipt.

Run the frozen nine-row native smoke config and produce a compact preflight
receipt recording config/source SHA, expected/observed row identities,
execution modes, and metric-block schema version.

Evidence tier: smoke / preflight only.  No planner ranking, fairness,
ethics, safety, or real-world validity claim.

Usage:
    DISPLAY= MPLBACKEND=Agg SDL_VIDEODRIVER=dummy \
      uv run python scripts/validation/preflight_social_compliance_smoke_issue_6481.py \
        --output-root output/benchmarks/issue_6481_preflight

Exit codes:
    0  preflight passed (all nine rows native with schema-valid social block)
    1  preflight failed (row-count, schema, or execution-mode mismatch)
    2  campaign execution error
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

from robot_sf.benchmark.fallback_policy import resolve_execution_mode
from robot_sf.evidence.writers import write_json

ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "configs/benchmarks/issue_6481_social_compliance_preflight_smoke.yaml"
SCENARIO_MATRIX_PATH = ROOT / "configs/scenarios/issue_6481_social_compliance_preflight.yaml"
EXPECTED_PLANNERS = ("goal", "social_force", "orca")
EXPECTED_SEEDS = (111, 112, 113)
EXPECTED_SCENARIO = "single_ped_crossing_orthogonal"
EXPECTED_ROW_COUNT = len(EXPECTED_PLANNERS) * len(EXPECTED_SEEDS)  # 9
SOCIAL_COMPLIANCE_SCHEMA_VERSION = "social-compliance-metric-contract.v1"
EXPECTED_METRIC_FAMILIES = {
    "pedestrian_deviation_mean_m": "pedestrian_deviation",
    "flow_disruption_delay_s": "flow_disruption",
    "comfort_exposure_person_s": "comfort_exposure",
    "legibility_progress_deficit_m": "legibility_progress",
    "distributional_inconvenience_p90_p50_gap": "distributional_inconvenience",
}
REQUIRED_FAMILIES = set(EXPECTED_METRIC_FAMILIES.values())
VALID_STATUSES = {"available", "unavailable", "not_applicable"}
VALID_EXECUTION_MODES = {
    "native",
    "adapter",
    "mixed",
    "fallback",
    "degraded",
    "unavailable",
    "unknown",
}
VALID_READINESS_STATUSES = {"native", "adapter", "fallback", "degraded", "unknown"}
VALID_AVAILABILITY_STATUSES = {
    "available",
    "not_available",
    "partial-failure",
    "failed",
    "unknown",
}


def _file_sha256(path: Path) -> str:
    """Return the hex SHA-256 of a file."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git_head_sha() -> str:
    """Return the current HEAD commit SHA."""
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=True,
        cwd=ROOT,
    )
    return result.stdout.strip()


def _find_last_json_object(stdout: str) -> Any | None:
    """Return the last complete JSON object embedded in noisy process output."""
    depth = 0
    end = len(stdout)
    for i in range(end - 1, -1, -1):
        if stdout[i] == "}":
            if depth == 0:
                end = i + 1
            depth += 1
        elif stdout[i] == "{":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(stdout[i:end])
                except json.JSONDecodeError:
                    return None
    return None


def _parse_campaign_stdout(stdout: str, returncode: int) -> dict[str, Any]:
    """Parse the camera-ready runner's final JSON object and retain its process status."""
    if not stdout:
        return {"_runner_returncode": returncode}
    try:
        payload = json.loads(stdout)
    except json.JSONDecodeError:
        payload = _find_last_json_object(stdout)
    if isinstance(payload, dict):
        return {**payload, "_runner_returncode": returncode}
    return {"_runner_returncode": returncode, "raw_stdout": stdout[-2000:]}


def _is_zero_exit_code(value: Any) -> bool:
    """Return whether a JSON exit-code field is the canonical integer zero."""
    return isinstance(value, int) and not isinstance(value, bool) and value == 0


def _normalized_text(value: Any, *, default: str = "unknown") -> str:
    """Return a lower-case non-empty status label or a fail-closed default."""
    if not isinstance(value, str):
        return default
    normalized = value.strip().lower()
    return normalized or default


def _resolve_row_execution_mode(record: dict[str, Any]) -> str:
    """Resolve execution mode from canonical episode metadata without native defaults."""
    for key in ("algorithm_metadata", "algorithm_metadata_contract"):
        payload = record.get(key)
        mode = _normalized_text(resolve_execution_mode(payload))
        if mode != "unknown":
            return mode if mode in VALID_EXECUTION_MODES else "unknown"

    mode = _normalized_text(resolve_execution_mode(record))
    return mode if mode in VALID_EXECUTION_MODES else "unknown"


def _as_bool(value: Any) -> bool:
    """Parse the campaign summary's boolean fields without truthy-string coercion."""
    if isinstance(value, bool):
        return value
    return isinstance(value, str) and value.strip().lower() == "true"


def _load_campaign_planner_statuses(
    campaign_result: dict[str, Any],
) -> tuple[dict[str, dict[str, Any]], bool]:
    """Load canonical planner readiness fields from the campaign summary."""
    raw_rows = campaign_result.get("planner_rows")
    if not isinstance(raw_rows, list):
        summary_path = campaign_result.get("summary_json")
        if isinstance(summary_path, str) and summary_path:
            try:
                summary = json.loads(Path(summary_path).read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                return {}, False
            raw_rows = summary.get("planner_rows")
    if not isinstance(raw_rows, list):
        return {}, False

    statuses: dict[str, dict[str, Any]] = {}
    for raw_row in raw_rows:
        if not isinstance(raw_row, dict):
            return {}, False
        planner = _normalized_text(raw_row.get("planner_key") or raw_row.get("algo"))
        if planner in statuses or planner == "unknown":
            return {}, False
        statuses[planner] = {
            "execution_mode": _normalized_text(raw_row.get("execution_mode")),
            "readiness_status": _normalized_text(raw_row.get("readiness_status")),
            "availability_status": _normalized_text(raw_row.get("availability_status")),
            "benchmark_success": _as_bool(raw_row.get("benchmark_success")),
        }
    return statuses, True


def _run_campaign(output_root: Path) -> dict[str, Any]:
    """Execute the camera-ready campaign and return the JSON result."""
    cmd = [
        sys.executable,
        str(ROOT / "scripts/tools/run_camera_ready_benchmark.py"),
        "--config",
        str(CONFIG_PATH),
        "--output-root",
        str(output_root),
        "--skip-publication-bundle",
    ]
    env = {
        k: v for k, v in os.environ.items() if k not in ("DISPLAY", "MPLBACKEND", "SDL_VIDEODRIVER")
    }
    env["DISPLAY"] = ""
    env["MPLBACKEND"] = "Agg"
    env["SDL_VIDEODRIVER"] = "dummy"
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=ROOT,
            env=env,
            timeout=600,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return {"_runner_returncode": 124, "error": "campaign timed out after 600 seconds"}
    return _parse_campaign_stdout(result.stdout.strip(), result.returncode)


def _read_episodes(campaign_root: Path) -> list[dict[str, Any]]:
    """Read all episode JSONL files from the campaign runs directory."""
    episodes: list[dict[str, Any]] = []
    runs_dir = campaign_root / "runs"
    if not runs_dir.is_dir():
        return episodes
    for jsonl_path in sorted(runs_dir.glob("*/episodes.jsonl")):
        for line in jsonl_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                record = json.loads(line)
                if not isinstance(record, dict):
                    raise ValueError(f"episode row is not a JSON object: {jsonl_path}")
                episodes.append(record)
    return episodes


def _classify_row(record: dict[str, Any]) -> dict[str, Any]:
    """Classify one episode row for the preflight receipt."""
    scenario_params = record.get("scenario_params", {})
    if not isinstance(scenario_params, dict):
        scenario_params = {}
    algo = scenario_params.get("algo", record.get("algo", "unknown"))
    scenario_id = record.get("scenario_id", "unknown")
    seed = record.get("seed")
    metrics = record.get("metrics", {})
    social = metrics.get("social_compliance", {}) if isinstance(metrics, dict) else {}
    raw_social_metrics = social.get("metrics", {}) if isinstance(social, dict) else {}
    social_metrics = raw_social_metrics if isinstance(raw_social_metrics, dict) else {}

    schema_version = social.get("schema_version") if isinstance(social, dict) else None
    families_present = set()
    statuses: dict[str, str] = {}
    support_counts: dict[str, int] = {}
    denominators: dict[str, Any] = {}
    unavailable_reasons: dict[str, str | None] = {}
    schema_valid = (
        isinstance(social, dict)
        and social.get("claim_class") == "diagnostic_proxy"
        and isinstance(social_metrics, dict)
        and set(social_metrics) == set(EXPECTED_METRIC_FAMILIES)
        and schema_version == SOCIAL_COMPLIANCE_SCHEMA_VERSION
    )
    for metric_id, row in social_metrics.items():
        if isinstance(row, dict):
            family = row.get("family", metric_id)
            if isinstance(family, str):
                families_present.add(family)
            status = row.get("status", "unknown")
            support_count = row.get("support_count", 0)
            denominator = row.get("denominator")
            reason = row.get("unavailable_reason")
            statuses[metric_id] = status
            support_counts[metric_id] = support_count
            denominators[metric_id] = denominator
            unavailable_reasons[metric_id] = reason

            row_valid = (
                metric_id in EXPECTED_METRIC_FAMILIES
                and row.get("id") == metric_id
                and family == EXPECTED_METRIC_FAMILIES.get(metric_id)
                and row.get("claim_class") == "diagnostic_proxy"
                and isinstance(row.get("units"), str)
                and bool(row["units"].strip())
                and isinstance(denominator, str)
                and bool(denominator.strip())
                and isinstance(status, str)
                and status in VALID_STATUSES
                and isinstance(support_count, int)
                and not isinstance(support_count, bool)
                and support_count >= 0
            )
            if status == "available":
                row_valid = (
                    row_valid
                    and support_count > 0
                    and isinstance(row.get("value"), (int, float))
                    and not isinstance(row.get("value"), bool)
                    and math.isfinite(float(row["value"]))
                )
            else:
                row_valid = (
                    row_valid
                    and support_count == 0
                    and isinstance(reason, str)
                    and bool(reason.strip())
                )
            schema_valid = schema_valid and row_valid
        else:
            schema_valid = False

    schema_valid = schema_valid and set(statuses) == set(EXPECTED_METRIC_FAMILIES)

    execution_mode = _resolve_row_execution_mode(record)

    return {
        "planner": algo,
        "scenario_id": scenario_id,
        "seed": seed,
        "execution_mode": execution_mode,
        "social_compliance_schema_version": schema_version,
        "families_present": sorted(families_present),
        "statuses": statuses,
        "support_counts": support_counts,
        "denominators": denominators,
        "unavailable_reasons": unavailable_reasons,
        "schema_valid": schema_valid,
        "all_families_present": REQUIRED_FAMILIES <= families_present,
    }


def _attach_campaign_status(
    row: dict[str, Any],
    planner_statuses: dict[str, dict[str, Any]],
) -> None:
    """Attach canonical planner status and detect episode/summary mode drift."""
    status = planner_statuses.get(row["planner"])
    if status is None:
        row.update(
            {
                "readiness_status": "unknown",
                "availability_status": "unknown",
                "benchmark_success": False,
                "execution_mode_consistent": False,
            }
        )
        return

    summary_mode = status["execution_mode"]
    row.update(
        {
            "readiness_status": (
                status["readiness_status"]
                if status["readiness_status"] in VALID_READINESS_STATUSES
                else "unknown"
            ),
            "availability_status": (
                status["availability_status"]
                if status["availability_status"] in VALID_AVAILABILITY_STATUSES
                else "unknown"
            ),
            "benchmark_success": status["benchmark_success"],
            "execution_mode_consistent": (
                row["execution_mode"] != "unknown"
                and summary_mode in VALID_EXECUTION_MODES
                and row["execution_mode"] == summary_mode
            ),
        }
    )


def _campaign_result_is_ok(campaign_result: dict[str, Any]) -> bool:
    """Return whether the child process and canonical campaign both succeeded."""
    return (
        bool(campaign_result.get("campaign_root"))
        and _is_zero_exit_code(campaign_result.get("_runner_returncode"))
        and campaign_result.get("campaign_execution_status") == "completed"
        and _is_zero_exit_code(campaign_result.get("exit_code"))
    )


def _receipt_failure_messages(receipt: dict[str, Any]) -> list[str]:
    """Return concise fail-closed reasons for the CLI result."""
    failures: list[str] = []
    if not receipt["row_count_ok"]:
        failures.append(f"row count {receipt['observed_row_count']} != {EXPECTED_ROW_COUNT}")
    if not receipt["identities_ok"]:
        failures.append("row identity mismatch")
    if not receipt["all_native"]:
        failures.append(
            "native-only execution contract not met: "
            + ", ".join(
                f"{planner}={modes}" for planner, modes in receipt["execution_modes"].items()
            )
        )
    if not receipt["planner_summary_ok"]:
        failures.append("canonical planner status summary missing or malformed")
    if not receipt["all_execution_modes_recorded"]:
        failures.append("execution mode missing or unknown")
    if not receipt["execution_modes_consistent"]:
        failures.append("episode and campaign-summary execution modes differ")
    if not receipt["no_fallback_or_degraded"]:
        failures.append("fallback/degraded/unavailable planner status detected")
    if not receipt["all_schema_valid"]:
        failures.append("schema-invalid social_compliance block")
    if not receipt["all_families_present"]:
        failures.append("missing metric families")
    return failures


def build_receipt(
    campaign_result: dict[str, Any],
    episodes: list[dict[str, Any]],
    output_root: Path,
) -> dict[str, Any]:
    """Build the compact preflight receipt."""
    row_classifications = [_classify_row(ep) for ep in episodes]
    planner_statuses, planner_summary_ok = _load_campaign_planner_statuses(campaign_result)
    planner_summary_ok = planner_summary_ok and set(planner_statuses) == set(EXPECTED_PLANNERS)
    for row in row_classifications:
        _attach_campaign_status(row, planner_statuses)
    observed_identities = sorted(
        {(r["planner"], r["scenario_id"], r["seed"]) for r in row_classifications}
    )
    expected_identities = sorted(
        (p, EXPECTED_SCENARIO, s) for p in EXPECTED_PLANNERS for s in EXPECTED_SEEDS
    )

    all_native = all(r["execution_mode"] == "native" for r in row_classifications)
    all_execution_modes_recorded = bool(row_classifications) and all(
        r["execution_mode"] != "unknown" for r in row_classifications
    )
    execution_modes_consistent = bool(row_classifications) and all(
        r["execution_mode_consistent"] for r in row_classifications
    )
    no_fallback_or_degraded = bool(row_classifications) and all(
        r["readiness_status"] in {"native", "adapter"}
        and r["availability_status"] == "available"
        and r["benchmark_success"]
        for r in row_classifications
    )
    all_schema_valid = all(r["schema_valid"] for r in row_classifications)
    all_families = all(r["all_families_present"] for r in row_classifications)
    row_count_ok = len(episodes) == EXPECTED_ROW_COUNT
    identities_ok = observed_identities == expected_identities
    campaign_returncode = campaign_result.get("_runner_returncode")
    campaign_exit_code = campaign_result.get("exit_code")
    campaign_ok = _campaign_result_is_ok(campaign_result)

    passed = (
        campaign_ok
        and row_count_ok
        and identities_ok
        and planner_summary_ok
        and all_execution_modes_recorded
        and execution_modes_consistent
        and all_native
        and no_fallback_or_degraded
        and all_schema_valid
        and all_families
    )

    return {
        "receipt_schema": "issue_6481_preflight_receipt.v1",
        "evidence_tier": "smoke_preflight",
        "claim_class": "diagnostic_proxy",
        "passed": passed,
        "config_sha256": _file_sha256(CONFIG_PATH),
        "scenario_matrix_sha256": _file_sha256(SCENARIO_MATRIX_PATH),
        "source_sha": _git_head_sha(),
        "expected_row_count": EXPECTED_ROW_COUNT,
        "observed_row_count": len(episodes),
        "expected_identities": [list(t) for t in expected_identities],
        "observed_identities": [list(t) for t in observed_identities],
        "row_count_ok": row_count_ok,
        "identities_ok": identities_ok,
        "all_native": all_native,
        "all_execution_modes_recorded": all_execution_modes_recorded,
        "execution_modes_consistent": execution_modes_consistent,
        "no_fallback_or_degraded": no_fallback_or_degraded,
        "planner_summary_ok": planner_summary_ok,
        "all_schema_valid": all_schema_valid,
        "all_families_present": all_families,
        "campaign_ok": campaign_ok,
        "campaign_returncode": campaign_returncode,
        "campaign_exit_code": campaign_exit_code,
        "execution_modes": {
            planner: sorted(
                {r["execution_mode"] for r in row_classifications if r["planner"] == planner}
            )
            for planner in EXPECTED_PLANNERS
        },
        "social_compliance_schema_version": SOCIAL_COMPLIANCE_SCHEMA_VERSION,
        "rows": row_classifications,
        "campaign_result_status": campaign_result.get("campaign_execution_status", "unknown"),
        "local_output_root": {
            "status": "ignored_runtime_artifact",
            "note": (
                "Campaign rows were written under the --output-root location during "
                "generation. That path is intentionally omitted from the durable "
                "receipt because repository evidence must not point at ignored "
                "runtime output artifacts."
            ),
        },
    }


def main(argv: list[str] | None = None) -> int:
    """Run the preflight smoke and emit the receipt."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=ROOT / "output/benchmarks/issue_6481_preflight",
        help="Campaign output root directory.",
    )
    parser.add_argument(
        "--receipt-output",
        type=Path,
        default=None,
        help="Optional path to write the receipt JSON (defaults to <output-root>/preflight_receipt.json).",
    )
    args = parser.parse_args(argv)

    output_root = args.output_root
    receipt_path = args.receipt_output or (output_root / "preflight_receipt.json")

    print(f"[issue_6481] Running social-compliance preflight smoke ({EXPECTED_ROW_COUNT} rows)...")
    campaign_result = _run_campaign(output_root)

    campaign_root_str = campaign_result.get("campaign_root", "")
    episodes: list[dict[str, Any]] = []
    episode_read_error: str | None = None
    if campaign_root_str:
        campaign_root = Path(campaign_root_str)
        try:
            episodes = _read_episodes(campaign_root)
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            episode_read_error = str(exc)

    campaign_ok = _campaign_result_is_ok(campaign_result)
    if not campaign_ok:
        print("[issue_6481] ERROR: campaign did not complete successfully", file=sys.stderr)
        print(json.dumps(campaign_result, indent=2), file=sys.stderr)
    elif episode_read_error is not None:
        print(
            f"[issue_6481] ERROR: could not read campaign episodes: {episode_read_error}",
            file=sys.stderr,
        )

    if not campaign_ok or episode_read_error is not None:
        receipt = build_receipt(campaign_result, episodes, output_root)
        receipt_path.parent.mkdir(parents=True, exist_ok=True)
        write_json(receipt_path, receipt)
        return 2

    receipt = build_receipt(campaign_result, episodes, output_root)

    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    write_json(receipt_path, receipt)
    print(json.dumps(receipt, indent=2))

    if receipt["passed"]:
        print(
            f"\n[issue_6481] PREFLIGHT PASSED: {EXPECTED_ROW_COUNT} native rows, "
            "schema-valid social blocks, and no fallback/degraded rows."
        )
        return 0

    failures = _receipt_failure_messages(receipt)
    print(f"\n[issue_6481] PREFLIGHT FAILED: {'; '.join(failures)}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
