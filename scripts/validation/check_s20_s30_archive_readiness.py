#!/usr/bin/env python3
"""Fail-closed readiness check for issue #1554 S20/S30 archive bundles.

This checker validates archive inputs only. It never runs benchmarks, submits
SLURM jobs, promotes artifacts, or upgrades S20/S30 evidence into a paper-facing
claim.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from scripts.tools.campaign_result_store import read_parquet_frame, validate_result_store

SCHEMA_VERSION = "s20-s30-archive-readiness.v1"
DEFAULT_PACKET_PATH = Path("configs/benchmarks/s20_s30_seed_budget_issue_1554_launch_packet.yaml")
READY = "ready"
BLOCKED = "blocked"
MALFORMED = "malformed"
EXIT_CODES = {READY: 0, BLOCKED: 1, MALFORMED: 2}
BENCHMARK_VALID_ROW_STATUSES = frozenset({"native", "adapter"})
METRIC_ALIASES = {
    "collisions": ("collisions", "collision"),
    "near_misses": ("near_misses", "near_miss"),
    "time_to_goal_norm": ("time_to_goal_norm", "time_to_goal"),
}
CLAIM_BOUNDARY = (
    "archive-readiness only; no full benchmark campaign run, no SLURM/GPU submission, "
    "no evidence promotion, and no paper/dissertation claim established"
)


@dataclass(frozen=True, slots=True)
class PacketContract:
    """Canonical issue #1554 archive-readiness contract from the launch packet."""

    campaign_id: str
    target_claim: str
    claim_status: str
    required_metrics: tuple[str, ...]
    planner_rows: tuple[str, ...]
    primary_seed_set: str
    escalation_seed_set: str
    seed_sets_path: Path
    result_store: Path
    required_result_store_files: tuple[str, ...]
    bundle_outputs: tuple[Path, ...]
    full_campaign_in_this_issue: bool
    submit_slurm_from_this_issue: bool
    bundle_status_until_run: str


def get_repository_root() -> Path:
    """Return repository root from this validation script location."""

    return Path(__file__).resolve().parents[2]


def load_packet(packet_path: Path) -> dict[str, Any]:
    """Load launch-packet YAML and reject malformed top-level payloads."""

    with packet_path.open(encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"{packet_path} must contain a YAML mapping")
    if payload.get("schema_version") != "s20-s30-seed-budget-launch-packet.v1":
        raise ValueError(
            f"{packet_path} schema_version {payload.get('schema_version')!r} is not "
            "'s20-s30-seed-budget-launch-packet.v1'"
        )
    if payload.get("no_benchmark_result_claim") is not True:
        raise ValueError(f"{packet_path} must keep no_benchmark_result_claim: true")
    return payload


def parse_contract(
    payload: dict[str, Any], repo_root: Path, result_store: Path | None
) -> PacketContract:
    """Extract and validate the issue #1554 launch-packet contract."""

    campaign_id = _required_str(payload, "campaign_id")
    claim_gate = _required_mapping(payload, "claim_map_gate")
    seed_policy = _required_mapping(payload, "seed_policy")
    expected_artifacts = _required_mapping(payload, "expected_artifacts")
    execution_boundary = _required_mapping(payload, "execution_boundary")

    required_metrics = _required_str_list(claim_gate, "required_metric_surface")
    planner_rows = _required_str_list(claim_gate, "planner_rows_to_confirm")
    if not _required_str(claim_gate, "target_claim").strip():
        raise ValueError("claim_map_gate.target_claim must be non-empty")
    if not _required_str(claim_gate, "why_s10_insufficient").strip():
        raise ValueError("why_s10_insufficient must be non-empty")

    raw_store = result_store or Path(_required_str(expected_artifacts, "result_store"))
    raw_files = _required_str_list(expected_artifacts, "result_store_required_files")
    raw_bundle = _required_str_list(expected_artifacts, "bundle")

    return PacketContract(
        campaign_id=campaign_id,
        target_claim=str(claim_gate["target_claim"]),
        claim_status=_required_str(claim_gate, "status"),
        required_metrics=tuple(required_metrics),
        planner_rows=tuple(planner_rows),
        primary_seed_set=_required_str(seed_policy, "primary_seed_set"),
        escalation_seed_set=_required_str(seed_policy, "escalation_seed_set"),
        seed_sets_path=_resolve_repo_path(repo_root, _required_str(seed_policy, "seed_sets_path")),
        result_store=_resolve_repo_path(repo_root, raw_store),
        required_result_store_files=tuple(raw_files),
        bundle_outputs=tuple(_resolve_repo_path(repo_root, p) for p in raw_bundle),
        full_campaign_in_this_issue=_required_bool(
            execution_boundary, "full_campaign_in_this_issue"
        ),
        submit_slurm_from_this_issue=_required_bool(
            execution_boundary, "submit_slurm_from_this_issue"
        ),
        bundle_status_until_run=_required_str(execution_boundary, "bundle_status_until_run"),
    )


def build_report(
    packet_path: Path, repo_root: Path, result_store: Path | None = None
) -> dict[str, Any]:
    """Build deterministic fail-closed readiness report for issue #1554 artifacts."""

    payload = load_packet(packet_path)
    contract = parse_contract(payload, repo_root, result_store)
    primary_seeds, escalation_seeds = _load_seed_tiers(contract)

    diagnostics: list[str] = []
    diagnostics.extend(_execution_boundary_diagnostics(contract))
    result_store_files = _result_store_file_status(contract)
    missing_files = [path for path, present in result_store_files.items() if not present]
    if missing_files:
        diagnostics.append(f"missing result-store files: {missing_files}")

    store_validation = validate_result_store(contract.result_store)
    if not store_validation.ok:
        diagnostics.extend(store_validation.errors)

    planner_seed_coverage: dict[str, Any] = {}
    metric_coverage: dict[str, bool] = dict.fromkeys(contract.required_metrics, False)
    row_status_counts: dict[str, int] = {}
    if not missing_files and store_validation.ok:
        episodes = read_parquet_frame(contract.result_store / "episodes.parquet")
        planner_seed_coverage = _planner_seed_coverage(
            episodes, contract.planner_rows, primary_seeds
        )
        metric_coverage = _metric_coverage(episodes, contract.required_metrics)
        row_status_counts = {
            str(status): int(count)
            for status, count in episodes["row_status"]
            .astype(str)
            .value_counts()
            .sort_index()
            .items()
        }
        diagnostics.extend(
            _coverage_diagnostics(planner_seed_coverage, metric_coverage, row_status_counts)
        )

    status = READY if not diagnostics else BLOCKED
    return {
        "schema_version": SCHEMA_VERSION,
        "status": status,
        "claim_boundary": CLAIM_BOUNDARY,
        "packet": str(packet_path),
        "campaign_id": contract.campaign_id,
        "target_claim_metadata": {
            "target_claim": contract.target_claim,
            "status": contract.claim_status,
            "why_s10_insufficient_present": True,
        },
        "planner_rows": list(contract.planner_rows),
        "seed_tier": {
            "primary_seed_set": contract.primary_seed_set,
            "primary_seed_count": len(primary_seeds),
            "escalation_seed_set": contract.escalation_seed_set,
            "escalation_seed_count": len(escalation_seeds),
            "primary_seed_set_ready_required": True,
            "s30_configured_as_escalation": True,
        },
        "required_metrics": list(contract.required_metrics),
        "metric_coverage": metric_coverage,
        "row_status_counts": row_status_counts,
        "planner_seed_coverage": planner_seed_coverage,
        "output_locations": {
            "result_store": str(contract.result_store),
            "bundle_outputs": [str(path) for path in contract.bundle_outputs],
        },
        "expected_result_store_files": result_store_files,
        "missing_artifact_diagnostics": diagnostics,
        "execution_boundary": {
            "full_campaign_in_this_issue": contract.full_campaign_in_this_issue,
            "submit_slurm_from_this_issue": contract.submit_slurm_from_this_issue,
            "bundle_status_until_run": contract.bundle_status_until_run,
        },
    }


def _load_seed_tiers(contract: PacketContract) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Load primary and escalation seed tiers from the configured seed-set YAML."""

    with contract.seed_sets_path.open(encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"{contract.seed_sets_path} must contain a YAML mapping")
    primary = _int_seed_list(payload, contract.primary_seed_set, expected_count=20)
    escalation = _int_seed_list(payload, contract.escalation_seed_set, expected_count=30)
    if tuple(primary) != tuple(escalation[: len(primary)]):
        raise ValueError(
            f"{contract.escalation_seed_set} must extend {contract.primary_seed_set} "
            "without reordering primary seeds"
        )
    return tuple(primary), tuple(escalation)


def _int_seed_list(payload: dict[str, Any], key: str, *, expected_count: int) -> list[int]:
    seeds = payload.get(key)
    if not isinstance(seeds, list) or len(seeds) != expected_count:
        raise ValueError(f"{key} must define exactly {expected_count} seeds")
    if any(type(seed) is not int for seed in seeds):
        raise ValueError(f"{key} must contain integer seeds")
    if len(set(seeds)) != expected_count:
        raise ValueError(f"{key} must contain {expected_count} unique seeds")
    return list(seeds)


def _result_store_file_status(contract: PacketContract) -> dict[str, bool]:
    return {
        rel_path: (contract.result_store / _validate_relative_path(rel_path)).is_file()
        for rel_path in contract.required_result_store_files
    }


def _planner_seed_coverage(
    episodes: pd.DataFrame, planners: tuple[str, ...], required_seeds: tuple[int, ...]
) -> dict[str, Any]:
    coverage: dict[str, Any] = {}
    for planner in planners:
        planner_rows = episodes[episodes["planner"].astype(str) == planner]
        present_seed_values: list[int] = []
        skipped_seed_count = 0
        for raw_seed in planner_rows["seed"].tolist():
            seed = _finite_int_seed(raw_seed)
            if seed is None:
                skipped_seed_count += 1
                continue
            present_seed_values.append(seed)
        present_seeds = sorted(set(present_seed_values))
        missing = [seed for seed in required_seeds if seed not in present_seeds]
        coverage[planner] = {
            "present_seed_count": len([seed for seed in required_seeds if seed in present_seeds]),
            "required_seed_count": len(required_seeds),
            "missing_primary_seeds": missing,
            "skipped_non_finite_seed_count": skipped_seed_count,
        }
    return coverage


def _metric_coverage(episodes: pd.DataFrame, required_metrics: tuple[str, ...]) -> dict[str, bool]:
    coverage: dict[str, bool] = {}
    for metric in required_metrics:
        candidates = METRIC_ALIASES.get(metric, (metric,))
        coverage[metric] = any(
            column in episodes.columns and episodes[column].notna().any() for column in candidates
        )
    return coverage


def _coverage_diagnostics(
    planner_seed_coverage: dict[str, Any],
    metric_coverage: dict[str, bool],
    row_status_counts: dict[str, int],
) -> list[str]:
    diagnostics: list[str] = []
    for planner, coverage in planner_seed_coverage.items():
        skipped_seed_count = int(coverage.get("skipped_non_finite_seed_count", 0))
        if skipped_seed_count:
            diagnostics.append(
                f"planner {planner!r} skipped non-finite/unparseable seed values: "
                f"{skipped_seed_count}"
            )
        missing = coverage["missing_primary_seeds"]
        if missing:
            diagnostics.append(f"planner {planner!r} missing primary S20 seeds: {missing}")
    missing_metrics = [metric for metric, present in metric_coverage.items() if not present]
    if missing_metrics:
        diagnostics.append(f"missing required metric columns: {missing_metrics}")
    invalid_statuses = {
        status: count
        for status, count in row_status_counts.items()
        if status not in BENCHMARK_VALID_ROW_STATUSES
    }
    if invalid_statuses:
        diagnostics.append(f"fail-closed/non-promotable row statuses present: {invalid_statuses}")
    return diagnostics


def _execution_boundary_diagnostics(contract: PacketContract) -> list[str]:
    """Return fail-closed diagnostics for packet boundaries outside this issue slice."""

    diagnostics: list[str] = []
    if contract.full_campaign_in_this_issue:
        diagnostics.append(
            "execution_boundary.full_campaign_in_this_issue must be false for archive-readiness"
        )
    if contract.submit_slurm_from_this_issue:
        diagnostics.append(
            "execution_boundary.submit_slurm_from_this_issue must be false for archive-readiness"
        )
    return diagnostics


def _required_mapping(payload: dict[str, Any], key: str) -> dict[str, Any]:
    value = payload.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"{key} must be a mapping")
    return value


def _finite_int_seed(value: Any) -> int | None:
    """Return integer seed when value is finite and parseable, else ``None``."""

    if pd.isna(value):
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(numeric):
        return None
    return int(numeric)


def _required_str(payload: dict[str, Any], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{key} must be a non-empty string")
    return value


def _required_str_list(payload: dict[str, Any], key: str) -> list[str]:
    value = payload.get(key)
    if not isinstance(value, list) or not value or not all(isinstance(item, str) for item in value):
        raise ValueError(f"{key} must be a non-empty string list")
    return value


def _required_bool(payload: dict[str, Any], key: str) -> bool:
    value = payload.get(key)
    if not isinstance(value, bool):
        raise ValueError(f"{key} must be a boolean")
    return value


def _resolve_repo_path(repo_root: Path, raw_path: str | Path) -> Path:
    """Resolve a launch-packet path and require it to stay inside repo root."""

    path = Path(raw_path)
    if ".." in path.parts:
        raise ValueError(f"path traversal is not allowed: {raw_path}")
    resolved_root = repo_root.resolve()
    resolved = path.resolve() if path.is_absolute() else (resolved_root / path).resolve()
    try:
        resolved.relative_to(resolved_root)
    except ValueError as exc:
        raise ValueError(f"path must stay within repository root: {raw_path}") from exc
    return resolved


def _validate_relative_path(raw_path: str) -> Path:
    """Validate result-store member paths before joining them under the store root."""

    path = Path(raw_path)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"result-store file path must be relative and contained: {raw_path}")
    return path


def _render_text(report: dict[str, Any]) -> str:
    lines = [
        f"status: {report['status']}",
        f"campaign_id: {report['campaign_id']}",
        f"claim_boundary: {report['claim_boundary']}",
        f"result_store: {report['output_locations']['result_store']}",
    ]
    diagnostics = report["missing_artifact_diagnostics"]
    if diagnostics:
        lines.append("diagnostics:")
        lines.extend(f"- {item}" for item in diagnostics)
    else:
        lines.append("diagnostics: none")
    return "\n".join(lines)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--packet", type=Path, default=None, help="Issue #1554 launch packet YAML.")
    parser.add_argument(
        "--repo-root", type=Path, default=None, help="Repository root for relative paths."
    )
    parser.add_argument(
        "--result-store", type=Path, default=None, help="Override result-store path."
    )
    parser.add_argument("--json", action="store_true", help="Emit JSON instead of text.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Run archive-readiness check and return fail-closed exit code."""

    args = _parse_args(argv)
    repo_root = args.repo_root.resolve() if args.repo_root else get_repository_root()
    packet_path = args.packet if args.packet else repo_root / DEFAULT_PACKET_PATH
    if not packet_path.is_absolute():
        packet_path = repo_root / packet_path
    result_store = args.result_store
    if result_store is not None and not result_store.is_absolute():
        result_store = repo_root / result_store
    try:
        report = build_report(packet_path, repo_root, result_store)
    except (OSError, ValueError, yaml.YAMLError) as exc:
        report = {
            "schema_version": SCHEMA_VERSION,
            "status": MALFORMED,
            "claim_boundary": CLAIM_BOUNDARY,
            "packet": str(packet_path),
            "error": str(exc),
        }
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    elif report["status"] == MALFORMED:
        print(f"status: {MALFORMED}\nclaim_boundary: {CLAIM_BOUNDARY}\nerror: {report['error']}")
    else:
        print(_render_text(report))
    return EXIT_CODES[report["status"]]


if __name__ == "__main__":
    raise SystemExit(main())
