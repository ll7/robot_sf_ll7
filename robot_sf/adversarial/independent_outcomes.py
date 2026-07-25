"""Independent planner-outcome packet contract for issue #3275 (v2, row-level).

This module owns the frozen ``adversarial_independent_outcomes.v2`` contract: one
row per candidate x execution seed, binding every outcome to its candidate
manifest, selection arm, target planner/config, scenario family/seed, execution
commit/command/config lineage, native/fallback/degraded status, termination
reason and independent failure outcome, scenario and candidate certification
status, replay/confirmation lineage and record hash, and an exclusion reason
when inadmissible.

The v2 contract replaces the deprecated flat-array v1 contract. Aggregate
arrays (proposal/random/ranked outcomes) are DERIVED from admitted rows only and
are never supplied independently of them. Rows that are missing, malformed,
mismatched, fallback, degraded, or lineage-incomplete fail closed: they can never
open the held-out gate or drive a verdict. The candidate manifest hash must also
match an external, frozen ID-to-hash binding; the packet cannot self-attest that
lineage. A candidate manifest ID may appear in one arm only.

When valid v2 rows are available, the top-level proposal/random metrics, the
comparison, and the issue #2921 stop rule are computed EXCLUSIVELY from those
independent native execution outcomes. Archive-nearness is intentionally absent
from this module: it lives in a diagnostic-only namespace in the runner and can
never drive a verdict.

This module does not execute planners and produces no new empirical outcome.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from robot_sf.adversarial.disjoint_evaluation import (
    binary_yield_min_detectable_difference,
    classify_issue_3275_decision,
    fisher_exact_two_sided_table,
    shuffled_outcome_null_test,
)

#: Frozen row-level outcome schema for the #3275 contract.
OUTCOME_SCHEMA_VERSION = "adversarial_independent_outcomes.v2"

#: Selection arms admitted by the contract.
_ARMS = ("proposal", "random")

#: Every field that an ADMITTED row must carry, well-typed and consistent. Rows
#: whose ``admission_status`` is ``"excluded"`` only need ``row_id``,
#: ``candidate_manifest_id``, ``selection_arm``, ``admission_status`` and a
#: non-empty ``exclusion_reason`` (they were never executed).
REQUIRED_ADMITTED_ROW_FIELDS = (
    "row_id",
    "candidate_manifest_id",
    "candidate_manifest_sha256",
    "selection_arm",
    "selection_rank",
    "candidate_pool_seed",
    "candidate_pool_index",
    "target_planner_id",
    "target_planner_config_sha256",
    "scenario_family",
    "scenario_seed",
    "execution_commit",
    "execution_command",
    "execution_config_lineage",
    "execution_mode",
    "termination_reason",
    "independent_failure_outcome",
    "scenario_certification_status",
    "candidate_certification_status",
    "replay_lineage",
    "confirmation_lineage",
    "record_sha256",
)

#: Confirmation thresholds accepted by the contract and their numeric rule.
_CONFIRMATION_THRESHOLDS = {
    "3_of_5": {"min_confirmed": 3, "attempt_count": 5},
    "4_of_5": {"min_confirmed": 4, "attempt_count": 5},
}


@dataclass(frozen=True)
class AdmissionSpec:
    """Frozen admission parameters for the v2 row contract."""

    expected_target_planner_id: str
    expected_eval_family: str
    confirmation_threshold: str = "3_of_5"
    expected_target_planner_config_sha256: str | None = None
    expected_candidate_manifest_sha256_by_id: dict[str, str] | None = None
    expected_record_sha256_by_manifest: dict[str, str] | None = None


def payload_sha256(payload: dict[str, Any]) -> str:
    """Return a deterministic SHA-256 digest for a JSON-like payload."""
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def load_independent_outcomes(path: Any) -> tuple[str, str, dict[str, Any] | None]:
    """Load an independent planner-outcome payload.

    Returns:
        ``(state, reason, payload)``. Missing payloads are not available; supplied
        but unreadable payloads are blocked because the run explicitly requested
        an independent outcome surface.
    """
    if path is None:
        return (
            "not_available",
            "No independent outcome path provided; held-out evidence remains fail-closed.",
            None,
        )
    p = Path(path)
    if not p.exists():
        return "blocked", f"Independent outcome path {path} does not exist.", None
    if p.stat().st_size == 0:
        return "blocked", f"Independent outcome file {path} is empty.", None
    try:
        with open(p, encoding="utf-8") as f:
            payload = json.load(f)
    except (ValueError, TypeError, json.JSONDecodeError, OSError) as exc:
        return "blocked", f"Failed to load independent outcomes: {exc}.", None
    if not isinstance(payload, dict):
        return "blocked", "Independent outcome payload must be a JSON object.", None
    return "active", "Independent planner-execution outcomes loaded successfully.", payload


def _blocked(reason: str, **extra: Any) -> dict[str, Any]:
    """Build a fail-closed blocked evaluation result."""
    result: dict[str, Any] = {
        "schema_version": OUTCOME_SCHEMA_VERSION,
        "status": "blocked",
        "independent_outcomes_available": False,
        "certification_available": False,
        "admitted_row_count": 0,
        "excluded_row_count": 0,
        "null_tests_reject_null": False,
        "reason": reason,
    }
    result.update(extra)
    return result


def _validate_packet_metadata(payload: dict[str, Any], spec: AdmissionSpec) -> str | None:
    """Reject circular, deprecated, mismatched, or non-execution packet metadata."""
    if payload.get("schema_version") != OUTCOME_SCHEMA_VERSION:
        return (
            f"deprecated or unknown outcome schema: {payload.get('schema_version')!r}; "
            f"only {OUTCOME_SCHEMA_VERSION!r} (row-level) is admitted"
        )
    if payload.get("outcome_source") != "planner_execution":
        return "outcome_source must be planner_execution"
    if payload.get("objective") in {"archive_nearness", "objective_distance", "nearest_archive"}:
        return "archive-nearness outcomes are circular for issue #3275"
    if payload.get("target_planner_id") != spec.expected_target_planner_id:
        return (
            f"target_planner_id mismatch: packet={payload.get('target_planner_id')!r} "
            f"expected={spec.expected_target_planner_id!r}"
        )
    return None


def _confirmation_ok(confirmation_lineage: Any, *, threshold: str) -> tuple[bool, str]:
    """Validate independent-seed confirmation lineage against the frozen threshold."""
    if not isinstance(confirmation_lineage, dict):
        return False, "confirmation_lineage must be an object"
    rule = _CONFIRMATION_THRESHOLDS.get(threshold)
    if rule is None:
        return False, f"unsupported confirmation threshold: {threshold!r}"
    confirmed = confirmation_lineage.get("confirmed_count")
    attempts = confirmation_lineage.get("attempt_count")
    stable = confirmation_lineage.get("stable_attribution")
    if not isinstance(confirmed, int) or not isinstance(attempts, int):
        return False, "confirmation_lineage.confirmed_count/attempt_count must be integers"
    if attempts != rule["attempt_count"]:
        return False, f"confirmation attempt_count={attempts} != frozen {rule['attempt_count']}"
    if confirmed < rule["min_confirmed"]:
        return False, (
            f"confirmation confirmed_count={confirmed} below frozen threshold "
            f"{threshold} (min {rule['min_confirmed']})"
        )
    if stable is not True:
        return False, "confirmation_lineage.stable_attribution must be true"
    return True, "ok"


def _replay_ok(replay_lineage: Any) -> tuple[bool, str]:
    """Validate deterministic-replay lineage for one admitted row."""
    if not isinstance(replay_lineage, dict):
        return False, "replay_lineage must be an object"
    if replay_lineage.get("exact_signature_match") is not True:
        return False, "replay_lineage.exact_signature_match must be true"
    replay_sig = replay_lineage.get("replay_signature_sha256")
    original_sig = replay_lineage.get("original_signature_sha256")
    if not isinstance(replay_sig, str) or not replay_sig:
        return False, "replay_lineage.replay_signature_sha256 missing"
    if not isinstance(original_sig, str) or not original_sig:
        return False, "replay_lineage.original_signature_sha256 missing"
    if replay_sig != original_sig:
        return False, "replay signature SHA-256 does not match original signature SHA-256"
    return True, "ok"


def _row_missing_fields(row: dict[str, Any], _row_id: Any, _spec: AdmissionSpec) -> str | None:
    """Check required-field presence and well-typedness for an admitted row."""
    for field_name in REQUIRED_ADMITTED_ROW_FIELDS:
        if field_name not in row:
            return f"missing required field {field_name!r}"
    if row.get("selection_arm") not in _ARMS:
        return f"invalid selection_arm {row.get('selection_arm')!r}"
    if not isinstance(row.get("independent_failure_outcome"), bool):
        return "independent_failure_outcome must be bool"
    if not isinstance(row.get("candidate_manifest_sha256"), str) or not row.get(
        "candidate_manifest_sha256"
    ):
        return "candidate_manifest_sha256 must be a non-empty string"
    if not isinstance(row.get("execution_command"), list) or not row.get("execution_command"):
        return "execution_command must be a non-empty list"
    if not isinstance(row.get("execution_config_lineage"), dict):
        return "execution_config_lineage must be an object"
    if not isinstance(row.get("record_sha256"), str) or not row.get("record_sha256"):
        return "record_sha256 missing"
    return None


def _row_planner_family_drift(row: dict[str, Any], _row_id: Any, spec: AdmissionSpec) -> str | None:
    """Check the frozen target planner, config SHA, and evaluation family."""
    if row.get("target_planner_id") != spec.expected_target_planner_id:
        return f"target_planner_id mismatch {row.get('target_planner_id')!r}"
    if (
        spec.expected_target_planner_config_sha256 is not None
        and row.get("target_planner_config_sha256") != spec.expected_target_planner_config_sha256
    ):
        return "target_planner_config_sha256 mismatch"
    if row.get("scenario_family") != spec.expected_eval_family:
        return (
            f"scenario_family {row.get('scenario_family')!r} != held-out eval family "
            f"{spec.expected_eval_family!r}"
        )
    return None


def _row_execution_drift(row: dict[str, Any], _row_id: Any, _spec: AdmissionSpec) -> str | None:
    """Reject non-native (fallback/degraded) execution rows."""
    if row.get("execution_mode") != "native":
        return (
            f"execution_mode {row.get('execution_mode')!r} is not native "
            "(fallback/degraded fail closed)"
        )
    return None


def _row_certification_drift(row: dict[str, Any], _row_id: Any, _spec: AdmissionSpec) -> str | None:
    """Require passed scenario and candidate certification."""
    if row.get("scenario_certification_status") != "passed":
        return (
            f"scenario_certification_status {row.get('scenario_certification_status')!r} != passed"
        )
    if row.get("candidate_certification_status") != "passed":
        return f"candidate_certification_status {row.get('candidate_certification_status')!r} != passed"
    return None


def _row_lineage_drift(row: dict[str, Any], _row_id: Any, spec: AdmissionSpec) -> str | None:
    """Validate deterministic-replay and confirmation lineage."""
    replay_ok, replay_reason = _replay_ok(row.get("replay_lineage"))
    if not replay_ok:
        return replay_reason
    confirm_ok, confirm_reason = _confirmation_ok(
        row.get("confirmation_lineage"), threshold=spec.confirmation_threshold
    )
    if not confirm_ok:
        return confirm_reason
    return None


def _row_candidate_manifest_hash_drift(
    row: dict[str, Any], _row_id: Any, spec: AdmissionSpec
) -> str | None:
    """Require each row's manifest hash to match a frozen external binding."""
    expected_hashes = spec.expected_candidate_manifest_sha256_by_id
    if not expected_hashes:
        return "expected candidate_manifest_sha256 binding is unavailable"
    manifest_id = str(row["candidate_manifest_id"])
    expected = expected_hashes.get(manifest_id)
    if expected is None:
        return "candidate_manifest_id is absent from the expected manifest-hash binding"
    if row["candidate_manifest_sha256"] != expected:
        return "candidate_manifest_sha256 mismatch for manifest"
    return None


def _row_record_hash_drift(row: dict[str, Any], _row_id: Any, spec: AdmissionSpec) -> str | None:
    """Validate per-manifest record-hash binding when expected hashes are supplied."""
    if not spec.expected_record_sha256_by_manifest:
        return None
    manifest_id = str(row["candidate_manifest_id"])
    expected = spec.expected_record_sha256_by_manifest.get(manifest_id)
    if expected is not None and row["record_sha256"] != expected:
        return "record_sha256 mismatch for manifest"
    return None


_ROW_CHECKERS = (
    _row_missing_fields,
    _row_planner_family_drift,
    _row_execution_drift,
    _row_certification_drift,
    _row_lineage_drift,
    _row_candidate_manifest_hash_drift,
    _row_record_hash_drift,
)


def _admit_row(
    row: Any,
    *,
    row_index: int,
    spec: AdmissionSpec,
) -> tuple[dict[str, Any] | None, str | None]:
    """Admit one row fail-closed, or return ``(None, reason)`` to block."""
    if not isinstance(row, dict):
        return None, f"row[{row_index}]: not an object"
    row_id = row.get("row_id")
    prefix = f"row[{row_index}] ({row_id}): "
    admission_status = row.get("admission_status", "admitted")
    if admission_status == "excluded":
        exclusion_reason = row.get("exclusion_reason")
        if not isinstance(exclusion_reason, str) or not exclusion_reason.strip():
            return None, prefix + "excluded row missing non-empty exclusion_reason"
        if row.get("selection_arm") not in _ARMS:
            return None, prefix + "excluded row has invalid selection_arm"
        return {"_excluded": True, "row_id": row_id}, None
    if admission_status != "admitted":
        return None, prefix + f"unsupported admission_status {admission_status!r}"
    for checker in _ROW_CHECKERS:
        reason = checker(row, row_id, spec)
        if reason is not None:
            return None, prefix + reason
    return row, None


def _candidate_level_outcomes(
    admitted_rows: list[dict[str, Any]],
) -> tuple[dict[str, dict[str, Any]] | None, str | None]:
    """Cluster admitted rows by (arm, candidate) and return candidate outcomes.

    A candidate's failure outcome must be stable across its execution seeds. Any
    disagreement fails closed (unstable attribution). Returns a dict keyed by arm
    with ``{"count": n, "failures": k, "outcomes": [...], "ids": [...]}`` or
    ``(None, reason)`` to block.
    """
    by_arm_candidate: dict[str, dict[str, list[bool]]] = {"proposal": {}, "random": {}}
    for row in admitted_rows:
        arm = row["selection_arm"]
        manifest_id = str(row["candidate_manifest_id"])
        by_arm_candidate[arm].setdefault(manifest_id, []).append(
            bool(row["independent_failure_outcome"])
        )

    overlap_ids = sorted(set(by_arm_candidate["proposal"]) & set(by_arm_candidate["random"]))
    if overlap_ids:
        return None, (
            "candidate_manifest_id appears in both proposal and random arms: "
            f"{', '.join(overlap_ids)}"
        )

    result: dict[str, dict[str, Any]] = {}
    for arm in _ARMS:
        candidate_map = by_arm_candidate[arm]
        outcomes: list[bool] = []
        ids: list[str] = []
        for manifest_id, seed_outcomes in sorted(candidate_map.items()):
            if any(outcome != seed_outcomes[0] for outcome in seed_outcomes):
                return None, (
                    f"unstable attribution: candidate {manifest_id} in arm {arm} has "
                    "disagreement across execution seeds"
                )
            outcomes.append(seed_outcomes[0])
            ids.append(manifest_id)
        result[arm] = {
            "count": len(outcomes),
            "failures": int(sum(1 for o in outcomes if o)),
            "outcomes": outcomes,
            "ids": ids,
        }
    return result, None


def _summarize_admitted_rows(
    admitted_rows: list[dict[str, Any]],
    excluded_count: int,
    payload: dict[str, Any],
    *,
    minimally_important: float,
    alpha: float,
    n_permutations: int,
    seed: int,
) -> dict[str, Any]:
    """Compute candidate-level yields, comparison, null tests, and the decision."""
    candidate_outcomes, cluster_reason = _candidate_level_outcomes(admitted_rows)
    if candidate_outcomes is None:
        return _blocked(
            cluster_reason or "clustering_failed", payload_sha256=payload_sha256(payload)
        )

    proposal = candidate_outcomes["proposal"]
    random_arm = candidate_outcomes["random"]
    if proposal["count"] == 0 or random_arm["count"] == 0:
        return {
            "schema_version": OUTCOME_SCHEMA_VERSION,
            "status": "not_available_empty_outcomes",
            "independent_outcomes_available": False,
            "certification_available": True,
            "admitted_row_count": len(admitted_rows),
            "excluded_row_count": excluded_count,
            "null_tests_reject_null": False,
            "reason": "proposal and random arms must both have at least one admitted candidate",
            "proposal_candidate_count": proposal["count"],
            "random_candidate_count": random_arm["count"],
            "payload_sha256": payload_sha256(payload),
        }

    proposal_yield = proposal["failures"] / proposal["count"]
    random_yield = random_arm["failures"] / random_arm["count"]
    fisher_p = fisher_exact_two_sided_table(
        proposal["failures"],
        proposal["count"] - proposal["failures"],
        random_arm["failures"],
        random_arm["count"] - random_arm["failures"],
    )
    null_rejected = bool(fisher_p <= alpha)
    binding_k = min(proposal["count"], random_arm["count"])
    min_detectable = binary_yield_min_detectable_difference(binding_k, alpha=alpha)
    powered = bool(min_detectable <= float(minimally_important))
    shuffled_null = shuffled_outcome_null_test(
        [float(o) for o in proposal["outcomes"]],
        [float(o) for o in random_arm["outcomes"]],
        n_permutations=n_permutations,
        seed=seed,
    )
    decision = classify_issue_3275_decision(
        proposal_yield=proposal_yield,
        random_yield=random_yield,
        minimally_important=minimally_important,
        null_rejected=null_rejected,
        powered=powered,
        independent_available=True,
    )
    return {
        "schema_version": OUTCOME_SCHEMA_VERSION,
        "status": "complete",
        "source": payload.get("source", "unspecified"),
        "artifact": payload.get("artifact"),
        "eval_archive_sha256": payload.get("eval_archive_sha256"),
        "outcome_source": payload.get("outcome_source"),
        "objective": payload.get("objective"),
        "target_planner_id": payload.get("target_planner_id"),
        "independent_outcomes_available": True,
        "certification_available": True,
        "admitted_row_count": len(admitted_rows),
        "excluded_row_count": excluded_count,
        "proposal": proposal,
        "random": random_arm,
        "proposal_failure_yield": round(proposal_yield, 6),
        "random_failure_yield": round(random_yield, 6),
        "comparison": {
            "estimand": "proposal_minus_random_candidate_level_certified_failure_yield",
            "yield_improvement": round(proposal_yield - random_yield, 6),
            "fisher_exact_two_sided_p_value": round(fisher_p, 6),
            "alpha_two_sided": alpha,
            "null_rejected": null_rejected,
            "powered": powered,
            "min_detectable_yield_difference_at_binding_k": min_detectable,
            "binding_k": binding_k,
            "minimally_important_absolute_yield_improvement": float(minimally_important),
        },
        "null_tests": {
            "fisher_exact_two_sided": {
                "p_value": round(fisher_p, 6),
                "alpha": alpha,
                "status": "complete",
                "required_for_held_out_claim": True,
                "primary": True,
            },
            "shuffled_outcome_label_permutation": {
                **shuffled_null,
                "required_for_held_out_claim": True,
                "primary": False,
            },
        },
        "null_tests_reject_null": null_rejected,
        "decision": decision,
        "payload_sha256": payload_sha256(payload),
    }


def build_independent_outcome_evaluation(
    payload: dict[str, Any] | None,
    *,
    budget_per_arm: int,
    minimally_important: float,
    admission_spec: AdmissionSpec,
    alpha: float = 0.05,
    expected_eval_archive_sha256: str | None = None,
    n_permutations: int = 1000,
    seed: int = 0,
) -> dict[str, Any]:
    """Evaluate a v2 row-level independent-outcome packet and the #3275 decision.

    Admits rows fail-closed, derives candidate-level certified failure yields per
    arm from admitted rows only, computes the Fisher-exact two-sided null and the
    frozen power/sensitivity verdict, and returns the
    ``continue | stop | inconclusive`` decision. Archive-nearness never reaches
    this function.

    Args:
        payload: Parsed v2 outcome packet (or ``None``).
        budget_per_arm: Frozen candidate budget per arm (used for provenance).
        minimally_important: Frozen minimally important absolute yield improvement.
        admission_spec: Frozen admission parameters (planner, family, threshold...).
        alpha: Two-sided alpha for the Fisher-exact null.
        expected_eval_archive_sha256: Optional eval-split hash binding.
        n_permutations: Permutations for the diagnostic shuffled-outcome null.
        seed: Deterministic seed for the diagnostic permutation null.

    Returns:
        A JSON-safe evaluation dict. ``status`` is ``complete`` only when valid
        admitted rows from both arms produce candidate-level yields; otherwise it
        is ``not_available_*`` or ``blocked_*``.
    """
    del (
        budget_per_arm
    )  # reserved for future per-arm budget provenance; min-detectable uses realized arm sizes
    if payload is None:
        return {
            "schema_version": OUTCOME_SCHEMA_VERSION,
            "status": "not_available",
            "independent_outcomes_available": False,
            "certification_available": False,
            "admitted_row_count": 0,
            "excluded_row_count": 0,
            "null_tests_reject_null": False,
            "reason": "no_independent_outcome_payload",
            "decision": classify_issue_3275_decision(
                proposal_yield=0.0,
                random_yield=0.0,
                minimally_important=minimally_important,
                null_rejected=False,
                powered=False,
                independent_available=False,
            ),
            "payload_sha256": None,
        }

    metadata_reason = _validate_packet_metadata(payload, admission_spec)
    if metadata_reason is not None:
        return _blocked(metadata_reason, payload_sha256=payload_sha256(payload))
    if expected_eval_archive_sha256 is not None:
        observed = payload.get("eval_archive_sha256")
        if observed != expected_eval_archive_sha256:
            return _blocked(
                "independent outcome packet does not match the eval split hash",
                expected_eval_archive_sha256=expected_eval_archive_sha256,
                observed_eval_archive_sha256=observed,
                payload_sha256=payload_sha256(payload),
            )

    rows = payload.get("rows")
    if not isinstance(rows, list) or not rows:
        return _blocked("packet has no rows list", payload_sha256=payload_sha256(payload))

    admitted_rows: list[dict[str, Any]] = []
    excluded_count = 0
    for index, row in enumerate(rows):
        admitted, reason = _admit_row(row, row_index=index, spec=admission_spec)
        if reason is not None:
            return _blocked(reason, payload_sha256=payload_sha256(payload))
        if admitted is None:
            continue
        if admitted.get("_excluded"):
            excluded_count += 1
        else:
            admitted_rows.append(admitted)

    return _summarize_admitted_rows(
        admitted_rows,
        excluded_count,
        payload,
        minimally_important=minimally_important,
        alpha=alpha,
        n_permutations=n_permutations,
        seed=seed,
    )
