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
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "configs/benchmarks/issue_6481_social_compliance_preflight_smoke.yaml"
SCENARIO_MATRIX_PATH = ROOT / "configs/scenarios/issue_6481_social_compliance_preflight.yaml"
EXPECTED_PLANNERS = ("goal", "social_force", "orca")
EXPECTED_SEEDS = (111, 112, 113)
EXPECTED_SCENARIO = "single_ped_crossing_orthogonal"
EXPECTED_ROW_COUNT = len(EXPECTED_PLANNERS) * len(EXPECTED_SEEDS)  # 9
SOCIAL_COMPLIANCE_SCHEMA_VERSION = "social-compliance-metric-contract.v1"
REQUIRED_FAMILIES = {
    "pedestrian_deviation",
    "flow_disruption",
    "comfort_exposure",
    "legibility_progress",
    "distributional_inconvenience",
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
        k: v
        for k, v in __import__("os").environ.items()
        if k not in ("DISPLAY", "MPLBACKEND", "SDL_VIDEODRIVER")
    }
    env["DISPLAY"] = ""
    env["MPLBACKEND"] = "Agg"
    env["SDL_VIDEODRIVER"] = "dummy"
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        cwd=ROOT,
        env=env,
        timeout=600,
        check=False,
    )
    stdout = result.stdout.strip()
    # The campaign runner prints a multi-line JSON object as its final output.
    # Find the last complete JSON object in stdout.
    if not stdout:
        return {"returncode": result.returncode}
    # Try parsing the entire stdout as JSON first (common case).
    try:
        return json.loads(stdout)
    except json.JSONDecodeError:
        pass
    # Fall back to finding the last JSON object by scanning for the last '{'.
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
                    break
    return {"raw_stdout": stdout[-2000:], "returncode": result.returncode}


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
                episodes.append(json.loads(line))
    return episodes


def _classify_row(record: dict[str, Any]) -> dict[str, Any]:
    """Classify one episode row for the preflight receipt."""
    scenario_params = record.get("scenario_params", {})
    algo = scenario_params.get("algo", record.get("algo", "unknown"))
    scenario_id = record.get("scenario_id", "unknown")
    seed = record.get("seed")
    metrics = record.get("metrics", {})
    social = metrics.get("social_compliance", {})
    social_metrics = social.get("metrics", {}) if isinstance(social, dict) else {}

    schema_version = social.get("schema_version") if isinstance(social, dict) else None
    families_present = set()
    statuses: dict[str, str] = {}
    support_counts: dict[str, int] = {}
    for metric_id, row in social_metrics.items():
        if isinstance(row, dict):
            family = row.get("family", metric_id)
            families_present.add(family)
            statuses[metric_id] = row.get("status", "unknown")
            support_counts[metric_id] = row.get("support_count", 0)

    execution_mode = record.get("execution_mode", "native")
    if execution_mode not in ("native", "adapter"):
        execution_mode = "fallback_or_degraded"

    return {
        "planner": algo,
        "scenario_id": scenario_id,
        "seed": seed,
        "execution_mode": execution_mode,
        "social_compliance_schema_version": schema_version,
        "families_present": sorted(families_present),
        "statuses": statuses,
        "support_counts": support_counts,
        "schema_valid": schema_version == SOCIAL_COMPLIANCE_SCHEMA_VERSION,
        "all_families_present": REQUIRED_FAMILIES <= families_present,
    }


def build_receipt(
    campaign_result: dict[str, Any],
    episodes: list[dict[str, Any]],
    output_root: Path,
) -> dict[str, Any]:
    """Build the compact preflight receipt."""
    row_classifications = [_classify_row(ep) for ep in episodes]
    observed_identities = sorted(
        {(r["planner"], r["scenario_id"], r["seed"]) for r in row_classifications}
    )
    expected_identities = sorted(
        (p, EXPECTED_SCENARIO, s) for p in EXPECTED_PLANNERS for s in EXPECTED_SEEDS
    )

    all_native = all(r["execution_mode"] in ("native", "adapter") for r in row_classifications)
    all_schema_valid = all(r["schema_valid"] for r in row_classifications)
    all_families = all(r["all_families_present"] for r in row_classifications)
    row_count_ok = len(episodes) == EXPECTED_ROW_COUNT
    identities_ok = observed_identities == expected_identities

    passed = row_count_ok and identities_ok and all_native and all_schema_valid and all_families

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
        "all_schema_valid": all_schema_valid,
        "all_families_present": all_families,
        "execution_modes": {r["planner"]: r["execution_mode"] for r in row_classifications},
        "social_compliance_schema_version": SOCIAL_COMPLIANCE_SCHEMA_VERSION,
        "rows": row_classifications,
        "campaign_result_status": campaign_result.get("campaign_execution_status", "unknown"),
        "output_root": str(output_root),
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
    if not campaign_root_str:
        print("[issue_6481] ERROR: campaign did not produce a campaign_root", file=sys.stderr)
        print(json.dumps(campaign_result, indent=2), file=sys.stderr)
        return 2

    campaign_root = Path(campaign_root_str)
    episodes = _read_episodes(campaign_root)

    receipt = build_receipt(campaign_result, episodes, output_root)

    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=False) + "\n", encoding="utf-8")
    print(json.dumps(receipt, indent=2))

    if receipt["passed"]:
        print(
            f"\n[issue_6481] PREFLIGHT PASSED: {EXPECTED_ROW_COUNT} native rows, schema-valid social blocks."
        )
        return 0

    failures = []
    if not receipt["row_count_ok"]:
        failures.append(f"row count {receipt['observed_row_count']} != {EXPECTED_ROW_COUNT}")
    if not receipt["identities_ok"]:
        failures.append("row identity mismatch")
    if not receipt["all_native"]:
        failures.append("non-native execution mode detected")
    if not receipt["all_schema_valid"]:
        failures.append("schema-invalid social_compliance block")
    if not receipt["all_families_present"]:
        failures.append("missing metric families")
    print(f"\n[issue_6481] PREFLIGHT FAILED: {'; '.join(failures)}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
