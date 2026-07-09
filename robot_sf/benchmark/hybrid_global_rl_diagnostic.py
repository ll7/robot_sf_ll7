"""Diagnostic report builder for issue #4183 hybrid global/RL comparison.

The report is intentionally diagnostic-only: it pairs route-conditioned and
unconditioned learned-local rows, copies adapter diagnostics, and excludes
fallback or degraded rows from any route-conditioned effect evidence.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

from robot_sf.benchmark.identity.hash_utils import sha256_file as _sha256_file
from robot_sf.models import get_registry_entry

SCHEMA_VERSION = "issue-4183-hybrid-global-rl-diagnostic.v1"
ISSUE = 4183
ROUTE_ARM = "route_conditioned_hybrid_global_rl"
BASELINE_ARM = "learned_local_no_route_conditioning"
CLAIM_BOUNDARY = (
    "diagnostic-only paired route/occupancy comparison; not benchmark-improvement "
    "or paper-facing evidence"
)

CSV_FIELDS = [
    "scenario_id",
    "seed",
    "checkpoint",
    "pair_status",
    "row_inclusion",
    "route_progress_route_conditioned",
    "route_progress_unconditioned",
    "success_route_conditioned",
    "success_unconditioned",
    "safety_event_route_conditioned",
    "safety_event_unconditioned",
    "route_conditioning_status",
    "waypoint_status",
    "waypoint_reason",
    "fallback_status",
    "exclusion_reason",
]


class HybridGlobalRLDiagnosticError(ValueError):
    """Raised when issue #4183 diagnostic inputs violate the pairing contract."""


@dataclass(frozen=True)
class DiagnosticConfig:
    """Normalized issue #4183 diagnostic configuration."""

    path: Path
    arm: str
    scenario_matrix: str
    seeds: tuple[int, ...]
    horizon: int
    dt: float
    learned_policy_checkpoint: str
    learned_policy_model_id: str | None
    route_conditioning_enabled: bool
    row_inclusion_rule: str


def load_diagnostic_config(path: str | Path) -> DiagnosticConfig:
    """Load one issue #4183 benchmark config.

    Returns:
        DiagnosticConfig: Normalized diagnostic config contract.
    """

    config_path = Path(path)
    with config_path.open(encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    issue_block = payload.get("issue_4183_diagnostic")
    if not isinstance(issue_block, dict):
        raise HybridGlobalRLDiagnosticError(
            f"{config_path} missing issue_4183_diagnostic contract block"
        )
    seed_policy = payload.get("seed_policy") or {}
    seeds = tuple(int(seed) for seed in seed_policy.get("seeds", ()))
    if not seeds:
        raise HybridGlobalRLDiagnosticError(f"{config_path} must declare a non-empty seed list")
    return DiagnosticConfig(
        path=config_path,
        arm=str(issue_block["arm"]),
        scenario_matrix=str(payload["scenario_matrix"]),
        seeds=seeds,
        horizon=int(payload["horizon"]),
        dt=float(payload["dt"]),
        learned_policy_checkpoint=str(issue_block["learned_policy_checkpoint"]),
        learned_policy_model_id=str(issue_block.get("learned_policy_model_id") or "") or None,
        route_conditioning_enabled=bool(issue_block["route_conditioning_enabled"]),
        row_inclusion_rule=str(issue_block["row_inclusion_rule"]),
    )


def preflight_configs(
    route_config_path: str | Path,
    baseline_config_path: str | Path,
    *,
    repo_root: str | Path = ".",
) -> dict[str, Any]:
    """Validate the paired issue configs and learned checkpoint availability.

    Returns:
        dict[str, Any]: Versioned preflight payload with status and blocker details.
    """

    route_config = load_diagnostic_config(route_config_path)
    baseline_config = load_diagnostic_config(baseline_config_path)
    errors: list[str] = []
    if route_config.arm != ROUTE_ARM:
        errors.append(f"route config arm must be {ROUTE_ARM!r}")
    if baseline_config.arm != BASELINE_ARM:
        errors.append(f"baseline config arm must be {BASELINE_ARM!r}")
    shared_fields = {
        "scenario_matrix": (route_config.scenario_matrix, baseline_config.scenario_matrix),
        "seeds": (route_config.seeds, baseline_config.seeds),
        "horizon": (route_config.horizon, baseline_config.horizon),
        "dt": (route_config.dt, baseline_config.dt),
        "learned_policy_checkpoint": (
            route_config.learned_policy_checkpoint,
            baseline_config.learned_policy_checkpoint,
        ),
        "learned_policy_model_id": (
            route_config.learned_policy_model_id,
            baseline_config.learned_policy_model_id,
        ),
    }
    for field, (left, right) in shared_fields.items():
        if left != right:
            errors.append(f"paired configs disagree on {field}: {left!r} != {right!r}")
    if not route_config.route_conditioning_enabled:
        errors.append("route config must enable route conditioning")
    if baseline_config.route_conditioning_enabled:
        errors.append("baseline config must disable route conditioning")

    checkpoint_path, checkpoint_reference, checkpoint_errors = _resolve_checkpoint_reference(
        route_config, repo_root=Path(repo_root)
    )
    errors.extend(checkpoint_errors)
    if not checkpoint_path.exists():
        errors.append(f"blocked_missing_learned_checkpoint: {checkpoint_path}")
    scenario_path = Path(repo_root) / route_config.scenario_matrix
    if not scenario_path.exists():
        errors.append(f"missing_scenario_matrix: {scenario_path}")

    status = (
        "valid"
        if not errors
        else "blocked_missing_learned_checkpoint"
        if any(error.startswith("blocked_missing_learned_checkpoint") for error in errors)
        else "invalid"
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "issue": ISSUE,
        "status": status,
        "errors": errors,
        "route_config": str(route_config.path),
        "baseline_config": str(baseline_config.path),
        "scenario_matrix": route_config.scenario_matrix,
        "seeds": list(route_config.seeds),
        "horizon": route_config.horizon,
        "dt": route_config.dt,
        "learned_policy_checkpoint": route_config.learned_policy_checkpoint,
        "learned_policy_model_id": route_config.learned_policy_model_id,
        "checkpoint_reference": checkpoint_reference,
        "checkpoint_sha256": _sha256_file(checkpoint_path) if checkpoint_path.exists() else None,
        "claim_boundary": CLAIM_BOUNDARY,
    }


def _resolve_checkpoint_reference(
    config: DiagnosticConfig, *, repo_root: Path
) -> tuple[Path, dict[str, Any], list[str]]:
    """Resolve the learned checkpoint reference without downloading remote artifacts.

    Returns:
        tuple[Path, dict[str, Any], list[str]]: Local checkpoint path, metadata, and errors.
    """
    if config.learned_policy_model_id:
        try:
            entry = get_registry_entry(config.learned_policy_model_id)
        except KeyError:
            fallback_path = Path(config.learned_policy_checkpoint)
            return (
                fallback_path if fallback_path.is_absolute() else repo_root / fallback_path,
                {
                    "type": "model_id",
                    "model_id": config.learned_policy_model_id,
                    "status": "missing_registry_entry",
                },
                [f"missing_model_registry_entry: {config.learned_policy_model_id}"],
            )
        local_path = Path(str(entry.get("local_path", "")))
        promotion = entry.get("benchmark_promotion")
        checkpoint_path = local_path if local_path.is_absolute() else repo_root / local_path
        # A benchmark-promoted checkpoint is hydrated from its public GitHub release into
        # ``output/model_cache/<model_id>/<asset_name>``. That cached file name does not match
        # the registry ``local_path`` (``model.zip``), so a bare ``local_path`` existence check
        # would wrongly report ``blocked_missing_learned_checkpoint`` even when the checkpoint is
        # present and loadable via ``resolve_model_path``. Recognize the hydrated release asset so
        # preflight stays download-free but consistent with how the policy actually loads.
        if not checkpoint_path.exists():
            hydrated_path = _github_release_cache_path(entry, repo_root=repo_root)
            if hydrated_path is not None and hydrated_path.exists():
                checkpoint_path = hydrated_path
        return (
            checkpoint_path,
            {
                "type": "model_id",
                "model_id": config.learned_policy_model_id,
                "local_path": str(local_path),
                "resolved_path": str(checkpoint_path),
                "claim_boundary": (
                    promotion.get("claim_boundary") if isinstance(promotion, dict) else None
                ),
                "observation_mode": (
                    promotion.get("observation_mode") if isinstance(promotion, dict) else None
                ),
            },
            [],
        )
    checkpoint_path = Path(config.learned_policy_checkpoint)
    return (
        checkpoint_path if checkpoint_path.is_absolute() else repo_root / checkpoint_path,
        {"type": "local_path", "path": config.learned_policy_checkpoint},
        [],
    )


def _github_release_cache_path(entry: dict[str, Any], *, repo_root: Path) -> Path | None:
    """Return the canonical model-cache path for a github-release artifact, if declared.

    This mirrors ``robot_sf.models.registry._download_from_github_release`` which caches the
    asset under ``output/model_cache/<model_id>/<asset_name>``.

    Returns:
        Path | None: Expected hydrated artifact path, or ``None`` when no github release is set.
    """

    release = entry.get("github_release")
    if not isinstance(release, dict):
        return None
    asset_name = str(release.get("asset_name") or "").strip()
    model_id = str(entry.get("model_id") or "").strip()
    if not asset_name or not model_id:
        return None
    return repo_root / "output" / "model_cache" / model_id / asset_name


def load_jsonl_records(path: str | Path) -> list[dict[str, Any]]:
    """Load benchmark episode JSONL records.

    Returns:
        list[dict[str, Any]]: Decoded episode records in file order.
    """

    records: list[dict[str, Any]] = []
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                record = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise HybridGlobalRLDiagnosticError(
                    f"{path}:{line_number} is not valid JSON"
                ) from exc
            if not isinstance(record, dict):
                raise HybridGlobalRLDiagnosticError(f"{path}:{line_number} is not an object")
            records.append(record)
    return records


def build_diagnostic_report(
    *,
    route_records: list[dict[str, Any]],
    baseline_records: list[dict[str, Any]],
    route_config_path: str | Path,
    baseline_config_path: str | Path,
    output_dir: str | Path,
    repo_root: str | Path = ".",
    generated_at: str | None = None,
    run_failures: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Build and write the issue #4183 diagnostic packet.

    Returns:
        dict[str, Any]: Summary payload also written to ``summary.json``.
    """

    preflight = preflight_configs(route_config_path, baseline_config_path, repo_root=repo_root)
    if preflight["status"] != "valid":
        raise HybridGlobalRLDiagnosticError("; ".join(preflight["errors"]))

    route_config = load_diagnostic_config(route_config_path)
    baseline_config = load_diagnostic_config(baseline_config_path)
    paired_rows = _pair_rows(route_records, baseline_records, route_config, baseline_config)
    included = [row for row in paired_rows if row["row_inclusion"] == "included_diagnostic"]
    fallback_excluded = [
        row for row in paired_rows if row["row_inclusion"] == "excluded_fallback_or_degraded"
    ]
    invalid_pairs = [row for row in paired_rows if row["pair_status"] != "paired"]
    summary = {
        "schema_version": SCHEMA_VERSION,
        "issue": ISSUE,
        "generated_at": generated_at or datetime.now(UTC).isoformat(),
        "evidence_status": "diagnostic-only",
        "run_status": _run_status(paired_rows, run_failures or []),
        "claim_boundary": CLAIM_BOUNDARY,
        "linked_work": {"adapter_pr": 4161, "parent_issue": 4015},
        "preflight": preflight,
        "arms": [ROUTE_ARM, BASELINE_ARM],
        "run_failures": run_failures or [],
        "row_count": len(paired_rows),
        "included_diagnostic_rows": len(included),
        "fallback_or_degraded_excluded_rows": len(fallback_excluded),
        "invalid_pair_rows": len(invalid_pairs),
        "route_conditioned_effect_claim_rows": len(included),
        "row_inclusion_rule": route_config.row_inclusion_rule,
        "paired_rows_csv": "paired_rows.csv",
    }
    summary["integration_report"] = _integration_report_payload(summary)

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(out_dir / "paired_rows.csv", paired_rows)
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_readme(out_dir / "README.md", summary)
    return summary


def _run_status(
    paired_rows: list[dict[str, Any]],
    run_failures: list[dict[str, Any]],
) -> str:
    if run_failures and not paired_rows:
        return "blocked_no_valid_episode_rows"
    if run_failures:
        return "completed_with_fail_closed_exclusions"
    return "completed"


def _integration_report_payload(summary: dict[str, Any]) -> dict[str, Any]:
    """Classify issue #4183 packet state for successor handoff.

    Returns:
        dict[str, Any]: Integration-report fields embedded in ``summary.json``.
    """

    remaining_blockers: list[dict[str, Any]] = []
    intentional_exclusions = [
        {
            "blocker": "fallback_or_degraded_rows_excluded",
            "status": "intentional",
            "evidence": (
                f"{summary['fallback_or_degraded_excluded_rows']} rows excluded from "
                "route-conditioned effect evidence"
            ),
        },
        {
            "blocker": "pairing_errors_excluded",
            "status": "intentional",
            "evidence": (
                f"{summary['invalid_pair_rows']} rows excluded because the scenario, seed, "
                "or checkpoint pairing contract was not satisfied"
            ),
        },
    ]

    if summary["run_status"] == "blocked_no_valid_episode_rows":
        remaining_blockers.append(
            {
                "blocker": "no_valid_episode_rows",
                "status": "remaining",
                "evidence": "Fail-closed runner failures produced no paired episode rows.",
            }
        )
    elif summary["included_diagnostic_rows"] == 0:
        remaining_blockers.append(
            {
                "blocker": "no_included_route_conditioned_effect_rows",
                "status": "remaining",
                "evidence": "The packet has no native, paired route-conditioned effect rows.",
            }
        )

    preflight = summary.get("preflight", {})
    if preflight.get("status") != "valid":
        remaining_blockers.append(
            {
                "blocker": "preflight_not_valid",
                "status": "remaining",
                "evidence": "; ".join(preflight.get("errors", [])) or "preflight status invalid",
            }
        )

    for failure in summary["run_failures"]:
        blocker = {
            "blocker": f"fail_closed_{failure.get('arm', 'unknown_arm')}",
            "status": "remaining",
            "evidence": str(failure.get("reason", "missing failure reason")),
        }
        if failure.get("row_classification"):
            blocker["row_classification"] = str(failure["row_classification"])
        remaining_blockers.append(blocker)

    if remaining_blockers:
        next_action = (
            "Resolve the fail-closed runner blockers, then rerun the same paired route/occupancy "
            "diagnostic builder so route-conditioned and unconditioned arms emit matched native "
            "episode rows for the predeclared seeds."
        )
    else:
        next_action = (
            "Promote the diagnostic packet only after a broader predeclared benchmark campaign "
            "with fallback/degraded exclusions and provenance matching the claim boundary."
        )

    return {
        "blockers_remaining": remaining_blockers,
        "blockers_new": [],
        "intentional_exclusions": intentional_exclusions,
        "next_empirical_action": next_action,
    }


def _pair_rows(
    route_records: list[dict[str, Any]],
    baseline_records: list[dict[str, Any]],
    route_config: DiagnosticConfig,
    baseline_config: DiagnosticConfig,
) -> list[dict[str, Any]]:
    route_by_key = {_row_key(record, route_config): record for record in route_records}
    baseline_by_key = {_row_key(record, baseline_config): record for record in baseline_records}
    keys = sorted(set(route_by_key) | set(baseline_by_key))
    paired_rows = []
    for key in keys:
        route_record = route_by_key.get(key)
        baseline_record = baseline_by_key.get(key)
        if route_record is None or baseline_record is None:
            paired_rows.append(_missing_pair_row(key, route_record, baseline_record))
            continue
        paired_rows.append(_paired_row(route_record, baseline_record, route_config))
    return paired_rows


def _row_key(record: dict[str, Any], config: DiagnosticConfig) -> tuple[str, int, str]:
    scenario_id = str(record.get("scenario_id") or record.get("scenario") or "")
    seed = int(record.get("seed"))
    checkpoint = _record_checkpoint(record) or config.learned_policy_checkpoint
    return (scenario_id, seed, checkpoint)


def _paired_row(
    route_record: dict[str, Any],
    baseline_record: dict[str, Any],
    route_config: DiagnosticConfig,
) -> dict[str, Any]:
    route_diag = _route_diagnostics(route_record)
    route_fallback = _fallback_status(route_record)
    baseline_fallback = _fallback_status(baseline_record)
    route_conditioning_status = str(
        route_diag.get("route_conditioning_status")
        or route_diag.get("status")
        or route_record.get("route_conditioning_status")
        or "unknown"
    )
    waypoint_status = str(
        route_diag.get("waypoint_status")
        or route_diag.get("waypoint_decision", {}).get("status", "")
        or route_record.get("waypoint_status")
        or ""
    )
    waypoint_reason = str(
        route_diag.get("waypoint_reason")
        or route_diag.get("waypoint_decision", {}).get("reason", "")
        or route_record.get("waypoint_reason")
        or ""
    )
    exclusion_reason = ""
    if route_fallback != "native" or baseline_fallback != "native":
        row_inclusion = "excluded_fallback_or_degraded"
        exclusion_reason = f"fallback_or_degraded:{route_fallback}/{baseline_fallback}"
    elif route_conditioning_status not in {"conditioned", "waypoint_conditioned", "enabled"}:
        row_inclusion = "excluded_fallback_or_degraded"
        exclusion_reason = f"route_conditioning_status:{route_conditioning_status}"
    elif waypoint_status and waypoint_status not in {"ok", "selected", "conditioned"}:
        row_inclusion = "excluded_fallback_or_degraded"
        exclusion_reason = f"waypoint_status:{waypoint_status}"
    else:
        row_inclusion = "included_diagnostic"

    return {
        "scenario_id": str(route_record["scenario_id"]),
        "seed": int(route_record["seed"]),
        "checkpoint": _record_checkpoint(route_record) or route_config.learned_policy_checkpoint,
        "pair_status": "paired",
        "row_inclusion": row_inclusion,
        "route_progress_route_conditioned": _route_progress(route_record),
        "route_progress_unconditioned": _route_progress(baseline_record),
        "success_route_conditioned": _success(route_record),
        "success_unconditioned": _success(baseline_record),
        "safety_event_route_conditioned": _safety_event(route_record),
        "safety_event_unconditioned": _safety_event(baseline_record),
        "route_conditioning_status": route_conditioning_status,
        "waypoint_status": waypoint_status,
        "waypoint_reason": waypoint_reason,
        "fallback_status": f"{route_fallback}/{baseline_fallback}",
        "exclusion_reason": exclusion_reason,
    }


def _missing_pair_row(
    key: tuple[str, int, str],
    route_record: dict[str, Any] | None,
    baseline_record: dict[str, Any] | None,
) -> dict[str, Any]:
    scenario_id, seed, checkpoint = key
    missing = ROUTE_ARM if route_record is None else BASELINE_ARM
    return {
        "scenario_id": scenario_id,
        "seed": seed,
        "checkpoint": checkpoint,
        "pair_status": "missing_pair",
        "row_inclusion": "excluded_pairing_error",
        "route_progress_route_conditioned": "",
        "route_progress_unconditioned": "",
        "success_route_conditioned": "",
        "success_unconditioned": "",
        "safety_event_route_conditioned": "",
        "safety_event_unconditioned": "",
        "route_conditioning_status": "",
        "waypoint_status": "",
        "waypoint_reason": "",
        "fallback_status": "",
        "exclusion_reason": f"missing_{missing}",
    }


def _route_diagnostics(record: dict[str, Any]) -> dict[str, Any]:
    metadata = record.get("algorithm_metadata") or {}
    if not isinstance(metadata, dict):
        return {}
    for key in ("hybrid_global_rl_diagnostics", "adapter_diagnostics", "hybrid_global_rl"):
        value = metadata.get(key)
        if isinstance(value, dict):
            return value
    return {}


def _fallback_status(record: dict[str, Any]) -> str:
    metadata = record.get("algorithm_metadata") or {}
    execution_mode = (
        str(metadata.get("execution_mode", "")).lower() if isinstance(metadata, dict) else ""
    )
    local_policy = _route_diagnostics(record).get("local_policy_metadata") or {}
    local_status = (
        str(local_policy.get("status", "")).lower() if isinstance(local_policy, dict) else ""
    )
    status = str(record.get("status") or record.get("availability_status") or "").lower()
    values = {execution_mode, local_status, status}
    if values & {"fallback", "degraded", "failed", "not_available", "unavailable"}:
        return sorted(values & {"fallback", "degraded", "failed", "not_available", "unavailable"})[
            0
        ]
    return "native"


def _route_progress(record: dict[str, Any]) -> float | str:
    metrics = record.get("metrics") or {}
    for key in ("route_progress", "progress", "goal_progress"):
        if key in metrics:
            return float(metrics[key])
    if "success" in metrics:
        return 1.0 if bool(metrics["success"]) else 0.0
    return ""


def _success(record: dict[str, Any]) -> bool:
    metrics = record.get("metrics") or {}
    if "success" in metrics:
        return bool(metrics["success"])
    outcome = record.get("outcome") or {}
    if isinstance(outcome, dict) and "success" in outcome:
        return bool(outcome["success"])
    return str(record.get("termination_reason", "")).lower() in {"goal_reached", "success"}


def _safety_event(record: dict[str, Any]) -> bool:
    metrics = record.get("metrics") or {}
    outcome = record.get("outcome") or {}
    collision_count = metrics.get("total_collision_count", metrics.get("collisions", 0))
    near_misses = metrics.get("near_misses", 0)
    collision_event = (
        outcome.get("collision") if isinstance(outcome, dict) else record.get("collision_event")
    )
    return bool(collision_event) or float(collision_count or 0) > 0 or float(near_misses or 0) > 0


def _record_checkpoint(record: dict[str, Any]) -> str | None:
    metadata = record.get("algorithm_metadata") or {}
    candidates: list[Any] = [record.get("learned_policy_checkpoint")]
    if isinstance(metadata, dict):
        config = metadata.get("config")
        if isinstance(config, dict):
            candidates.extend([config.get("model_path"), config.get("checkpoint")])
        local_policy = _route_diagnostics(record).get("local_policy_metadata")
        if isinstance(local_policy, dict):
            local_config = local_policy.get("config")
            if isinstance(local_config, dict):
                candidates.extend([local_config.get("model_path"), local_config.get("checkpoint")])
    for candidate in candidates:
        if candidate:
            return str(candidate)
    return None


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in CSV_FIELDS})


def _write_readme(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# Issue #4183 Hybrid Global/RL Diagnostic",
        "",
        "This packet records a diagnostic-only paired route/occupancy comparison for "
        "`hybrid_global_rl` against the same learned local policy without route conditioning.",
        "",
        f"- Evidence status: `{summary['evidence_status']}`",
        f"- Run status: `{summary['run_status']}`",
        f"- Claim boundary: {summary['claim_boundary']}",
        f"- Included diagnostic rows: {summary['included_diagnostic_rows']}",
        f"- Fallback/degraded rows excluded: {summary['fallback_or_degraded_excluded_rows']}",
        f"- Invalid pair rows: {summary['invalid_pair_rows']}",
        "- Linked work: #4161 and #4015",
        "",
        "Rows marked fallback, degraded, unavailable, or missing-pair are not evidence for a "
        "route-conditioned effect. They remain diagnostic rows only.",
    ]
    _append_integration_report_section(lines, summary)
    if summary["run_failures"]:
        lines.extend(["", "## Fail-Closed Runner Failures", ""])
        for failure in summary["run_failures"]:
            lines.append(
                "- "
                + ", ".join(
                    f"{key}={value}"
                    for key, value in failure.items()
                    if value is not None and value != ""
                )
            )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _append_integration_report_section(lines: list[str], summary: dict[str, Any]) -> None:
    report = summary["integration_report"]
    lines.extend(
        [
            "",
            "## Integration Report",
            "",
            "This section classifies the diagnostic packet state so the next empirical action is "
            "clear without promoting diagnostic-only evidence.",
            "",
            f"- New blockers: {len(report['blockers_new'])}",
            f"- Next empirical action: {report['next_empirical_action']}",
            "",
            "### Blockers Remaining",
            "",
        ]
    )
    if report["blockers_remaining"]:
        lines.extend(_blocker_lines(report["blockers_remaining"]))
    else:
        lines.append("- none")
    lines.extend(["", "### New Blockers", ""])
    if report["blockers_new"]:
        lines.extend(_blocker_lines(report["blockers_new"]))
    else:
        lines.append("- none")
    lines.extend(["", "### Intentional Exclusions", ""])
    lines.extend(_blocker_lines(report["intentional_exclusions"]))


def _blocker_lines(blockers: list[dict[str, Any]]) -> list[str]:
    lines: list[str] = []
    for blocker in blockers:
        details = [
            f"status={blocker['status']}",
            f"evidence={blocker['evidence']}",
        ]
        if blocker.get("row_classification"):
            details.append(f"row_classification={blocker['row_classification']}")
        lines.append(f"- {blocker['blocker']}: " + ", ".join(details))
    return lines
