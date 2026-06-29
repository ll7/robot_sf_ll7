"""Fail-closed readiness/preflight checker for the predictive planner v2 same-seed comparison.

This module is the canonical owner for *preflighting* the same-seed predictive planner v2
four-way comparison described by issue #1490 (umbrella) and its ego-conditioning child #1504.
It is a read-only, coordination-only check: it validates that the comparison's prerequisite
metadata is present and self-consistent, and it surfaces the *blocked* Slurm/maintainer-gate
state. It never trains, evaluates, submits Slurm, or tunes planners.

The comparison contract validated here is the committed feature contract
``configs/training/predictive/predictive_ego_features_contract_v1.yaml``, which already declares:

* the four comparison variants (``baseline``, ``obstacle_only``, ``ego_only``,
  ``ego_obstacle_combined``) under ``row_identifiers``;
* the ego-obstacle conditioning metadata (``ego_motion_channel_producers`` and per-row
  ``ego_motion_channel_producer`` references);
* the same-seed comparability surface (``same_seed_comparability`` manifests/scenarios/grid).

Stage gates (each reported independently so missing metadata is actionable):

1. ``variant_completeness`` — all four variants present with schema name, input dim, and config.
2. ``provenance`` — every referenced config/manifest path exists on disk.
3. ``ego_obstacle_conditioning`` — ego variants declare a producer defined in the contract and
   share a single comparability producer key.
4. ``same_seed_schedule`` — seed manifests, scenario matrix, planner grid, and the
   forecast/navigation metric separation are declared.
5. ``blocked_slurm_gate`` — FAIL-CLOSED. The four-way expansion stays blocked behind the
   maintainer-selected revised hypothesis and a passing closed-loop coupling gate (#2916).
   This stage clears only when an explicit coupling-gate clearance artifact (recommendation
   ``continue``) is supplied together with the maintainer-hypothesis acknowledgement.

The default outcome (no gate clearance supplied) is ``blocked`` even when all metadata is
complete, mirroring the recorded #1490 maintainer decision. This is intentional: per
``docs/context/issue_691_benchmark_fallback_policy.md`` and the issue history (#1543 negative,
#1897 failed the coupling gate), the lane must fail closed, not silently report ready.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import yaml

# Stage status vocabulary shared with sibling readiness validators
# (see scripts/validation/validate_learned_prediction_readiness.py).
STATUS_PASSED = "passed"
STATUS_FAILED = "failed"
STATUS_BLOCKED = "blocked"

DEFAULT_CONTRACT_PATH = Path("configs/training/predictive/predictive_ego_features_contract_v1.yaml")

# The four same-seed comparison variants and their contract-declared schema/width expectations.
# input_dim values mirror the committed feature contract so a silent width drift is caught here
# rather than only at training time.
REQUIRED_VARIANTS: dict[str, dict[str, Any]] = {
    "baseline": {"schema_name": "predictive_legacy_v1", "input_dim": 4, "ego_conditioned": False},
    "obstacle_only": {
        "schema_name": "predictive_obstacle_features_v1",
        "input_dim": 10,
        "ego_conditioned": False,
    },
    "ego_only": {"schema_name": "predictive_ego_v1", "input_dim": 9, "ego_conditioned": True},
    "ego_obstacle_combined": {
        "schema_name": "predictive_obstacle_features_v1",
        "input_dim": 15,
        "ego_conditioned": True,
    },
}

# Same-seed comparability path fields that must be declared and exist on disk.
SAME_SEED_PATH_FIELDS = (
    "seed_manifest",
    "hard_seed_manifest",
    "scenario_matrix",
    "planner_grid",
)

_GATE_CONTINUE = "continue"

_READINESS_COMMAND_TEMPLATE = (
    "uv run python scripts/validation/validate_predictive_v2_comparison_readiness.py "
    "--json --contract configs/training/predictive/predictive_ego_features_contract_v1.yaml "
    "--coupling-gate <closed-loop-coupling-gate.json> --revised-hypothesis-recorded"
)

_PUBLIC_SURFACE_SNAPSHOT: tuple[dict[str, str], ...] = (
    {
        "surface": "issue_1543_negative_obstacle_audit",
        "issue": "#1543",
        "status": "completed",
        "source": "docs/context/issue_1543_predictive_v2_negative_audit.md",
        "signal": (
            "Obstacle-feature prerequisite was negative: predictive success 0.1014 "
            "vs baseline 0.1304; hard-seed success 0.0000 for both variants."
        ),
    },
    {
        "surface": "issue_1897_planner_coupling_gate",
        "issue": "#1897",
        "status": "completed_failed_gate",
        "source": "docs/context/issue_1897_predictive_coupling_gate_preflight.md",
        "signal": (
            "Revised planner-side coupling gate did not improve closed-loop success; "
            "baseline_like and phase_coupled_sequence_gate both recorded global and "
            "hard success 0.0000."
        ),
    },
    {
        "surface": "issue_2275_predictive_v2_fate",
        "issue": "#2275",
        "status": "completed_stop_decision",
        "source": "docs/context/issue_2275_predictive_v2_fate.md",
        "signal": "Selected stop_old_predictive_v2_expansion until a new coupling/objective gate passes.",
    },
    {
        "surface": "issue_2916_same_seed_coupling_gate",
        "issue": "#2916",
        "status": "completed_public_issue_closed",
        "source": "https://github.com/ll7/robot_sf_ll7/issues/2916",
        "signal": (
            "Public issue is closed, but no committed gate artifact is supplied to this "
            "readiness command by default; the packet therefore remains fail-closed."
        ),
    },
    {
        "surface": "issue_2902_live_same_seed_forecast_replay_gate",
        "issue": "#2902",
        "status": "completed_public_issue_closed",
        "source": "https://github.com/ll7/robot_sf_ll7/issues/2902",
        "signal": (
            "Public issue is closed with launch-packet tier evidence; it is not a direct "
            "go artifact for the #1490 four-way predictive-v2 comparison."
        ),
    },
    {
        "surface": "issue_1505_old_data_row_preflight_child",
        "issue": "#1505",
        "status": "completed_public_issue_closed",
        "source": "https://github.com/ll7/robot_sf_ll7/issues/1505",
        "signal": "Old data-row preflight child is closed and should not be used as a fresh queue target.",
    },
)


class PredictiveV2ComparisonReadinessError(Exception):
    """Raised when the comparison contract cannot be loaded or parsed."""


def _load_yaml(path: Path) -> Any:
    """Load a YAML document, raising a readiness error on missing/invalid input.

    Returns:
        The parsed YAML document (typically a mapping).
    """
    if not path.exists():
        raise PredictiveV2ComparisonReadinessError(f"contract file not found: {path}")
    try:
        with open(path, encoding="utf-8") as handle:
            return yaml.safe_load(handle)
    except (OSError, yaml.YAMLError) as exc:  # pragma: no cover - defensive
        raise PredictiveV2ComparisonReadinessError(f"could not parse contract {path}: {exc}")


def _resolve(repo_root: Path, raw: str) -> Path:
    """Resolve a contract-declared repository-relative path against the repo root.

    Returns:
        The absolute path to the referenced file.
    """
    candidate = Path(raw)
    return candidate if candidate.is_absolute() else repo_root / candidate


def _check_variant_completeness(contract: dict) -> tuple[str, list[str]]:
    """Verify all four comparison variants exist with schema name, width, and config.

    Returns:
        A ``(status, messages)`` tuple; status is ``passed`` or ``failed``.
    """
    rows = contract.get("row_identifiers")
    if not isinstance(rows, dict):
        return STATUS_FAILED, ["contract missing row_identifiers mapping"]

    errors: list[str] = []
    for variant, expected in REQUIRED_VARIANTS.items():
        entry = rows.get(variant)
        if not isinstance(entry, dict):
            errors.append(f"row_identifiers missing variant: {variant}")
            continue
        schema_name = entry.get("schema_name")
        if schema_name != expected["schema_name"]:
            errors.append(
                f"{variant} schema_name is {schema_name!r}, expected {expected['schema_name']!r}"
            )
        input_dim = entry.get("input_dim")
        if input_dim != expected["input_dim"]:
            errors.append(f"{variant} input_dim is {input_dim!r}, expected {expected['input_dim']}")
        if not entry.get("config"):
            errors.append(f"{variant} missing config provenance path")
    return (STATUS_FAILED, errors) if errors else (STATUS_PASSED, [])


def _check_provenance(contract: dict, repo_root: Path) -> tuple[str, list[str]]:
    """Verify every per-variant config and same-seed manifest path exists on disk.

    Returns:
        A ``(status, messages)`` tuple; status is ``passed`` or ``failed``.
    """
    errors: list[str] = []
    rows = contract.get("row_identifiers")
    if isinstance(rows, dict):
        for variant in REQUIRED_VARIANTS:
            entry = rows.get(variant)
            if not isinstance(entry, dict):
                continue
            config = entry.get("config")
            if config and not _resolve(repo_root, str(config)).exists():
                errors.append(f"{variant} config path does not exist: {config}")

    comparability = contract.get("same_seed_comparability")
    if isinstance(comparability, dict):
        for field in SAME_SEED_PATH_FIELDS:
            raw = comparability.get(field)
            if raw and not _resolve(repo_root, str(raw)).exists():
                errors.append(f"same_seed_comparability.{field} path does not exist: {raw}")

    return (STATUS_FAILED, errors) if errors else (STATUS_PASSED, [])


def _check_ego_obstacle_conditioning(contract: dict) -> tuple[str, list[str]]:
    """Verify ego variants declare a defined producer and share one comparability key.

    Returns:
        A ``(status, messages)`` tuple; status is ``passed`` or ``failed``.
    """
    errors: list[str] = []
    producers = contract.get("ego_motion_channel_producers")
    if not isinstance(producers, dict) or not producers:
        return STATUS_FAILED, ["contract missing ego_motion_channel_producers definitions"]

    rows = contract.get("row_identifiers")
    if not isinstance(rows, dict):
        return STATUS_FAILED, ["contract missing row_identifiers mapping"]

    declared_keys: set[str] = set()
    for variant, expected in REQUIRED_VARIANTS.items():
        entry = rows.get(variant)
        if not isinstance(entry, dict):
            continue
        producer_key = entry.get("ego_motion_channel_producer")
        if expected["ego_conditioned"]:
            if not producer_key:
                errors.append(f"{variant} missing ego_motion_channel_producer metadata")
                continue
            if producer_key not in producers:
                errors.append(f"{variant} references undefined ego producer: {producer_key}")
            else:
                declared_keys.add(producer_key)
        elif producer_key:
            errors.append(f"{variant} is not ego-conditioned but declares producer {producer_key}")

    # Mixed producer keys across ego variants are "not_comparable_without_caveat" per the
    # contract's comparability block; for a same-seed comparison they must match.
    if len(declared_keys) > 1:
        errors.append(
            "ego variants use mixed ego_motion_channel_producer keys "
            f"({sorted(declared_keys)}); same-seed comparison requires a single producer"
        )

    return (STATUS_FAILED, errors) if errors else (STATUS_PASSED, [])


def _check_same_seed_schedule(contract: dict) -> tuple[str, list[str]]:
    """Verify same-seed manifests, seed value, and forecast/navigation separation are declared.

    Returns:
        A ``(status, messages)`` tuple; status is ``passed`` or ``failed``.
    """
    errors: list[str] = []
    comparability = contract.get("same_seed_comparability")
    if not isinstance(comparability, dict):
        return STATUS_FAILED, ["contract missing same_seed_comparability block"]

    for field in SAME_SEED_PATH_FIELDS:
        if not comparability.get(field):
            errors.append(f"same_seed_comparability missing {field}")
    if comparability.get("seed") is None:
        errors.append("same_seed_comparability missing fixed seed")

    separation = contract.get("metric_separation")
    if not isinstance(separation, dict):
        errors.append("contract missing metric_separation block")
    else:
        if not separation.get("forecast_metrics"):
            errors.append("metric_separation missing forecast_metrics")
        if not separation.get("navigation_metrics"):
            errors.append("metric_separation missing navigation_metrics")

    return (STATUS_FAILED, errors) if errors else (STATUS_PASSED, [])


def _coupling_gate_recommendation(path: Path) -> tuple[str | None, str | None]:
    """Extract a coupling-gate recommendation from a JSON or Markdown artifact.

    Mirrors the JSON/Markdown handling in validate_learned_prediction_readiness.

    Returns:
        A ``(recommendation, error)`` tuple. Exactly one element is non-None.
    """
    if not path.exists():
        return None, f"coupling-gate clearance artifact not found: {path}"

    suffix = path.suffix.lower()
    if suffix == ".json":
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            return None, f"coupling-gate artifact is malformed JSON: {exc}"
        if not isinstance(payload, dict):
            return None, "coupling-gate artifact must be a JSON object"
        recommendation = payload.get("recommendation")
        if isinstance(recommendation, dict):
            recommendation = recommendation.get("decision")
        if not isinstance(recommendation, str) or not recommendation.strip():
            return None, "coupling-gate artifact missing recommendation/decision"
        return recommendation.strip().lower(), None

    if suffix in {".md", ".markdown"}:
        try:
            content = path.read_text(encoding="utf-8")
        except OSError as exc:
            return None, f"coupling-gate Markdown could not be read: {exc}"
        match = re.search(
            r"(?im)^[-*]?\s*(?:recommendation|decision)\s*:\s*`?([A-Za-z_]+)`?", content
        )
        if match is None:
            return None, "coupling-gate Markdown missing recommendation/decision line"
        return match.group(1).strip().lower(), None

    return None, "coupling-gate artifact must be JSON or Markdown"


def _check_blocked_slurm_gate(
    coupling_gate_path: Path | None,
    revised_hypothesis_recorded: bool,
) -> tuple[str, list[str]]:
    """Fail-closed gate: blocked unless a passing coupling gate AND maintainer ack are supplied.

    Returns:
        A ``(status, messages)`` tuple; status is ``passed`` or ``blocked``.
    """
    blocked_reasons = [
        "predictive-v2 four-way expansion (#1505/#1506/#1507) is blocked behind the "
        "maintainer-selected revised hypothesis and the same-seed coupling gate #2916",
        "Slurm/GPU submission is out of scope for this preflight (coordination-only)",
    ]

    if coupling_gate_path is None and not revised_hypothesis_recorded:
        return STATUS_BLOCKED, blocked_reasons

    errors: list[str] = []
    if not revised_hypothesis_recorded:
        errors.append("maintainer revised-hypothesis acknowledgement not provided")
    if coupling_gate_path is None:
        errors.append("coupling-gate clearance artifact not provided")
    else:
        recommendation, error = _coupling_gate_recommendation(coupling_gate_path)
        if error is not None:
            errors.append(error)
        elif recommendation != _GATE_CONTINUE:
            errors.append(
                f"coupling-gate recommendation is {recommendation!r}, expected {_GATE_CONTINUE!r}"
            )

    if errors:
        return STATUS_BLOCKED, errors
    return STATUS_PASSED, []


def _surface_status(repo_root: Path) -> list[dict[str, Any]]:
    """Return public issue/doc surfaces used by the decision packet."""
    surfaces: list[dict[str, Any]] = []
    for entry in _PUBLIC_SURFACE_SNAPSHOT:
        item: dict[str, Any] = dict(entry)
        source = entry["source"]
        if source.startswith("docs/"):
            item["source_exists"] = (repo_root / source).exists()
        else:
            item["source_exists"] = None
        surfaces.append(item)
    return surfaces


def _decision_packet(
    *,
    overall: str,
    stage_report: dict[str, dict[str, Any]],
    contract_path: Path,
    repo_root: Path,
    coupling_gate_path: Path | None,
    revised_hypothesis_recorded: bool,
) -> dict[str, Any]:
    """Build fail-closed #1490 decision packet from readiness status.

    Returns:
        Structured decision packet with public status surfaces and go/no-go fields.
    """
    gate_passed = stage_report["blocked_slurm_gate"]["status"] == STATUS_PASSED
    metadata_ready = all(
        payload["status"] == STATUS_PASSED
        for name, payload in stage_report.items()
        if name != "blocked_slurm_gate"
    )
    no_go_reasons = [
        message
        for message in stage_report["blocked_slurm_gate"]["messages"]
        if stage_report["blocked_slurm_gate"]["status"] != STATUS_PASSED
    ]
    if not metadata_ready:
        no_go_reasons.append("comparison metadata is incomplete; see failed readiness stages")

    recommendation = "go" if overall == "ready" else "no_go"
    next_action = (
        "run_bounded_same_seed_comparison_from_recorded_gate"
        if recommendation == "go"
        else "do_not_submit_compute; keep #1490 deferred until a committed closed-loop gate artifact is supplied"
    )
    return {
        "schema": "predictive_v2_same_seed_decision_packet.v1",
        "issue": "#1490",
        "public_surface_snapshot_date": "2026-06-29",
        "decision": recommendation,
        "go_no_go_recommendation": recommendation,
        "next_action": next_action,
        "completed_or_running_jobs": _surface_status(repo_root),
        "evidence_gap": {
            "status": "cleared" if gate_passed and metadata_ready else "open",
            "missing_or_blocking_inputs": no_go_reasons,
            "required_before_compute": [
                "committed closed-loop same-seed coupling gate artifact with recommendation 'continue'",
                "explicit maintainer-recorded revised predictive-v2 coupling/objective hypothesis",
                "durable artifact plan for any downstream Slurm or GPU run",
            ],
        },
        "candidate_queue_entry": {
            "lane": "predictive-v2 same-seed ego/obstacle conditioning comparison",
            "resource": "slurm_or_gpu_after_gate",
            "current_state": "ready_for_bounded_launch" if recommendation == "go" else "deferred_blocked",
            "submission_authorized_by_this_packet": False,
            "candidate_issue": "#1505" if recommendation == "go" else "#1490",
        },
        "cost_risk": {
            "local_readiness_cost": "low",
            "compute_cost_if_gate_clears": "medium_high_training_and_same_seed_evaluation",
            "risk": [
                "reopening the old four-way matrix without new coupling evidence repeats known negative path",
                "open-loop ADE/FDE improvements are insufficient without closed-loop success/progress/safety movement",
                "missing durable artifacts would make the comparison non-reproducible",
            ],
        },
        "exact_command_if_go": _READINESS_COMMAND_TEMPLATE if recommendation == "go" else None,
        "exact_readiness_command_to_clear_gate": _READINESS_COMMAND_TEMPLATE,
        "inputs": {
            "contract": str(contract_path),
            "repo_root": str(repo_root),
            "coupling_gate": str(coupling_gate_path) if coupling_gate_path else None,
            "revised_hypothesis_recorded": revised_hypothesis_recorded,
        },
    }


def validate_predictive_v2_comparison_readiness(
    contract_path: Path,
    repo_root: Path,
    coupling_gate_path: Path | None = None,
    revised_hypothesis_recorded: bool = False,
) -> dict:
    """Run all preflight stage gates and return a structured readiness report.

    Args:
        contract_path: Path to the predictive ego-features comparison contract YAML.
        repo_root: Repository root for resolving contract-relative provenance paths.
        coupling_gate_path: Optional coupling-gate clearance artifact (JSON/Markdown).
        revised_hypothesis_recorded: Explicit maintainer acknowledgement that a revised
            predictive-v2 hypothesis has been recorded (required to clear the blocked gate).

    Returns:
        A report dict with an overall ``status`` (``ready``, ``blocked``, or ``incomplete``),
        the per-stage results, and the inputs that were checked.

    Raises:
        PredictiveV2ComparisonReadinessError: If the contract cannot be loaded or parsed.
    """
    contract = _load_yaml(contract_path)
    if not isinstance(contract, dict):
        raise PredictiveV2ComparisonReadinessError(
            f"contract {contract_path} must be a mapping at the top level"
        )

    stages: dict[str, tuple[str, list[str]]] = {
        "variant_completeness": _check_variant_completeness(contract),
        "provenance": _check_provenance(contract, repo_root),
        "ego_obstacle_conditioning": _check_ego_obstacle_conditioning(contract),
        "same_seed_schedule": _check_same_seed_schedule(contract),
        "blocked_slurm_gate": _check_blocked_slurm_gate(
            coupling_gate_path, revised_hypothesis_recorded
        ),
    }

    stage_report: dict[str, dict[str, Any]] = {}
    metadata_failed = False
    gate_blocked = False
    for name, (status, messages) in stages.items():
        stage_report[name] = {"status": status, "messages": messages}
        if name == "blocked_slurm_gate":
            gate_blocked = status != STATUS_PASSED
        elif status != STATUS_PASSED:
            metadata_failed = True

    if metadata_failed:
        overall = "incomplete"
    elif gate_blocked:
        overall = "blocked"
    else:
        overall = "ready"

    report = {
        "status": overall,
        "issue": "#1490",
        "child_issue": "#1504",
        "stages": stage_report,
        "checked": {
            "contract": str(contract_path),
            "repo_root": str(repo_root),
            "coupling_gate": str(coupling_gate_path) if coupling_gate_path else None,
            "revised_hypothesis_recorded": revised_hypothesis_recorded,
            "required_variants": sorted(REQUIRED_VARIANTS),
        },
    }
    report["decision_packet"] = _decision_packet(
        overall=overall,
        stage_report=stage_report,
        contract_path=contract_path,
        repo_root=repo_root,
        coupling_gate_path=coupling_gate_path,
        revised_hypothesis_recorded=revised_hypothesis_recorded,
    )
    return report
