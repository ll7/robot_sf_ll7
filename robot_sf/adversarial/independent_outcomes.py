"""Independent planner-outcome packet contract for issue #3275.

This module validates the small artifact shape that a future planner-execution
runner must produce before the proposal-vs-random report can move beyond
circular archive-nearness plumbing. It deliberately does not execute planners.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from robot_sf.adversarial.disjoint_evaluation import (
    ranking_permutation_test,
    shuffled_outcome_null_test,
)

_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$", re.IGNORECASE)


@dataclass(frozen=True)
class _FrozenOutcomeContract:
    """Contract fields that every admitted v2 execution row must match."""

    manifest_sha256: str
    candidates: dict[str, dict[str, Any]]
    target_planner: str
    planner_config_sha256: str
    scenario_family: str


def payload_sha256(payload: dict[str, Any]) -> str:
    """Return a deterministic SHA-256 digest for a JSON-like payload."""
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def load_independent_outcomes(path: Path | None) -> tuple[str, str, dict[str, Any] | None]:
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
    if not path.exists():
        return "blocked", f"Independent outcome path {path} does not exist.", None
    if path.stat().st_size == 0:
        return "blocked", f"Independent outcome file {path} is empty.", None

    try:
        with open(path, encoding="utf-8") as f:
            payload = json.load(f)
    except (ValueError, TypeError, json.JSONDecodeError, OSError) as exc:
        return "blocked", f"Failed to load independent outcomes: {exc}.", None
    if not isinstance(payload, dict):
        return "blocked", "Independent outcome payload must be a JSON object.", None
    return "active", "Independent planner-execution outcomes loaded successfully.", payload


def _is_sha256(value: Any) -> bool:
    """Return whether ``value`` is a non-placeholder SHA-256 digest."""
    return isinstance(value, str) and _SHA256_PATTERN.fullmatch(value) is not None


def _validate_manifest_candidate(
    candidate: Any, idx: int
) -> tuple[str | None, dict[str, Any] | None]:
    """Validate one frozen candidate-manifest record."""
    if not isinstance(candidate, dict):
        return f"study contract candidate manifest row {idx} is not an object", None
    candidate_id = candidate.get("candidate_id")
    if not isinstance(candidate_id, str) or not candidate_id.strip():
        return f"study contract candidate manifest row {idx} has invalid candidate_id", None
    if candidate.get("selection_arm") not in {"proposal", "random"}:
        return f"study contract candidate {candidate_id} has invalid selection_arm", None
    if not isinstance(candidate.get("rank"), int) or candidate["rank"] < 0:
        return f"study contract candidate {candidate_id} has invalid rank", None
    if not isinstance(candidate.get("candidate_pool_seed"), int):
        return f"study contract candidate {candidate_id} has invalid candidate_pool_seed", None
    if not isinstance(candidate.get("scenario_seed"), int):
        return f"study contract candidate {candidate_id} has invalid scenario_seed", None
    execution_seeds = candidate.get("execution_seeds")
    if not isinstance(execution_seeds, list) or not execution_seeds:
        return f"study contract candidate {candidate_id} has invalid execution_seeds", None
    if not all(isinstance(seed, int) for seed in execution_seeds):
        return f"study contract candidate {candidate_id} has invalid execution_seeds", None
    return None, candidate


def _frozen_manifest_candidates(
    manifest: Any,
) -> tuple[str | None, str | None, dict[str, dict[str, Any]]]:
    """Return a frozen manifest hash and indexed candidates, or a reason."""
    if not isinstance(manifest, dict) or manifest.get("status") != "frozen":
        return "study contract candidate manifest is not frozen", None, {}
    manifest_hash = manifest.get("sha256")
    if not _is_sha256(manifest_hash):
        return "study contract candidate manifest sha256 is invalid", None, {}
    candidates = manifest.get("candidates")
    if not isinstance(candidates, list) or not candidates:
        return "study contract candidate manifest has no candidates", None, {}

    expected: dict[str, dict[str, Any]] = {}
    for idx, candidate in enumerate(candidates):
        reason, parsed_candidate = _validate_manifest_candidate(candidate, idx)
        if reason is not None or parsed_candidate is None:
            return reason, None, {}
        candidate_id = parsed_candidate["candidate_id"]
        if candidate_id in expected:
            return (
                f"study contract candidate manifest repeats candidate_id:{candidate_id}",
                None,
                {},
            )
        expected[candidate_id] = parsed_candidate
    return None, manifest_hash, expected


def _outcome_contract_requirements(
    outcome_contract: dict[str, Any] | None,
) -> tuple[bool, str, _FrozenOutcomeContract | None]:
    """Return frozen v2 requirements, or a fail-closed reason."""
    if not isinstance(outcome_contract, dict):
        return False, "v2 outcome rows require a frozen study contract", None

    admission = outcome_contract.get("outcome_admission")
    if not isinstance(admission, dict):
        return False, "study contract is missing outcome_admission", None
    if admission.get("schema_version") != "adversarial_independent_outcomes.v2":
        return False, "study contract does not require adversarial_independent_outcomes.v2", None
    if admission.get("execution_status") != "native":
        return False, "study contract does not require native execution", None

    reason, manifest_hash, candidates = _frozen_manifest_candidates(
        admission.get("candidate_manifest")
    )
    if reason is not None or manifest_hash is None:
        return False, reason or "study contract candidate manifest is invalid", None
    planner = outcome_contract.get("target_planner")
    planner_hash = outcome_contract.get("target_planner_config_sha256")
    family = outcome_contract.get("eval_scenario_family")
    if not isinstance(planner, str) or not planner.strip():
        return False, "study contract target_planner is invalid", None
    if not _is_sha256(planner_hash):
        return False, "study contract target_planner_config_sha256 is invalid", None
    if not isinstance(family, str) or not family.strip():
        return False, "study contract eval_scenario_family is invalid", None
    return (
        True,
        "ok",
        _FrozenOutcomeContract(manifest_hash, candidates, planner, planner_hash, family),
    )


def _missing_row_fields(
    row: dict[str, Any], required_keys: set[str], lineage_keys: set[str]
) -> str | None:
    """Return a missing-field reason for a v2 row, if any."""
    missing = [key for key in required_keys if key not in row]
    if missing:
        return f"missing required keys: {missing}"
    missing_lineage = [key for key in lineage_keys if key not in row]
    if missing_lineage:
        return f"missing lineage fields: {missing_lineage}"
    return None


def _row_matches_study_contract(
    row: dict[str, Any], contract: _FrozenOutcomeContract
) -> str | None:
    """Return a reason when a row differs from planner/family/manifest contract fields."""
    if row.get("manifest_sha256") != contract.manifest_sha256:
        return "manifest_sha256 does not match frozen candidate manifest"
    if row.get("target_planner_id") != contract.target_planner:
        return "target_planner_id does not match frozen study contract"
    if row.get("planner_config_sha256") != contract.planner_config_sha256:
        return "planner_config_sha256 does not match frozen study contract"
    if row.get("scenario_family") != contract.scenario_family:
        return "scenario_family does not match frozen study contract"
    return None


def _row_matches_manifest_candidate(
    row: dict[str, Any], candidates: dict[str, dict[str, Any]]
) -> str | None:
    """Return a reason when candidate, arm, rank, or seed differs from the manifest."""
    candidate_id = row.get("candidate_id")
    if not isinstance(candidate_id, str) or candidate_id not in candidates:
        return "candidate_id is absent from frozen candidate manifest"
    expected_candidate = candidates[candidate_id]
    for key in ("selection_arm", "rank", "candidate_pool_seed", "scenario_seed"):
        if row.get(key) != expected_candidate.get(key):
            return f"{key} does not match frozen candidate manifest"
    if row.get("execution_seed") not in expected_candidate.get("execution_seeds", []):
        return "execution_seed does not match frozen candidate manifest"
    return None


def _row_has_valid_execution_lineage(row: dict[str, Any]) -> str | None:
    """Return a reason when a row lacks structural native execution provenance."""
    if not _is_sha256(row.get("record_hash")):
        return "record_hash is not a SHA-256 digest"
    execution_commit = row.get("execution_commit")
    if not isinstance(execution_commit, str) or not re.fullmatch(
        r"[0-9a-f]{7,64}", execution_commit
    ):
        return "execution_commit is invalid"
    if not isinstance(row.get("command_lineage"), str) or not row["command_lineage"].strip():
        return "command_lineage is invalid"
    if not isinstance(row.get("replay_lineage"), str) or not row["replay_lineage"].strip():
        return "replay_lineage is invalid"
    if row.get("execution_status") != "native":
        return f"has invalid execution_status: {row.get('execution_status')}"
    return None


def _validate_single_row_lineage(
    idx: int,
    row: dict[str, Any],
    required_keys: set[str],
    lineage_keys: set[str],
    contract: _FrozenOutcomeContract,
) -> tuple[bool, str, tuple[str, float, int] | None]:
    """Validate a single row for lineage, status, and certification."""
    if not isinstance(row, dict):
        return False, f"row {idx} is not an object", None
    for reason in (
        _missing_row_fields(row, required_keys, lineage_keys),
        _row_matches_study_contract(row, contract),
        _row_matches_manifest_candidate(row, contract.candidates),
        _row_has_valid_execution_lineage(row),
    ):
        if reason is not None:
            return False, f"row {idx} {reason}", None

    cert_scen = str(row.get("scenario_certification_status", "")).lower()
    cert_cand = str(row.get("candidate_certification_status", "")).lower()
    if cert_scen != "passed" or cert_cand != "passed":
        return (
            False,
            f"row {idx} certification failed: scenario={cert_scen}, candidate={cert_cand}",
            None,
        )

    try:
        outcome = float(row["independent_failure_outcome"])
    except (TypeError, ValueError):
        return False, f"row {idx} invalid independent_failure_outcome", None

    return True, "ok", (row["selection_arm"], outcome, row["rank"])


def validate_row_lineage(
    rows: list[dict[str, Any]],
    *,
    outcome_contract: dict[str, Any] | None,
) -> tuple[bool, str, list[float], list[float], list[float]]:
    """Validate row-level outcome lineage for adversarial_independent_outcomes.v2.

    Returns:
        (ok, reason, proposal_outcomes, random_outcomes, ranked_outcomes)
    """
    if not isinstance(rows, list) or not rows:
        return False, "rows must be a non-empty list", [], [], []

    requirements_ok, requirements_reason, contract = _outcome_contract_requirements(
        outcome_contract
    )
    if not requirements_ok or contract is None:
        return False, requirements_reason, [], [], []

    required_keys = {
        "selection_arm",
        "independent_failure_outcome",
        "execution_status",
    }
    lineage_keys = {
        "candidate_id",
        "manifest_sha256",
        "target_planner_id",
        "planner_config_sha256",
        "scenario_family",
        "scenario_seed",
        "execution_seed",
        "execution_commit",
        "command_lineage",
        "termination_reason",
        "scenario_certification_status",
        "candidate_certification_status",
        "replay_lineage",
        "record_hash",
    }
    proposal_outcomes: list[float] = []
    random_outcomes: list[float] = []
    all_rows_with_rank = []

    for idx, row in enumerate(rows):
        ok, reason, item = _validate_single_row_lineage(
            idx,
            row,
            required_keys,
            lineage_keys,
            contract,
        )
        if not ok or item is None:
            return False, reason, [], [], []

        arm, outcome, rank = item
        if arm == "proposal":
            proposal_outcomes.append(outcome)
        else:
            random_outcomes.append(outcome)
        all_rows_with_rank.append((rank, outcome))

    all_rows_with_rank.sort(key=lambda item: item[0])
    ranked_outcomes = [item[1] for item in all_rows_with_rank]

    return True, "ok", proposal_outcomes, random_outcomes, ranked_outcomes


def _metadata_allows_independent_outcomes(payload: dict[str, Any]) -> tuple[bool, str]:
    """Reject known circular, fallback, degraded, or unavailable packet metadata."""
    if payload.get("outcome_source") != "planner_execution":
        return False, "outcome_source must be planner_execution"
    if payload.get("objective") in {"archive_nearness", "objective_distance", "nearest_archive"}:
        return False, "archive-nearness outcomes are circular for issue #3275"
    row_statuses = payload.get("row_statuses", [])
    if not isinstance(row_statuses, list):
        return False, "row_statuses must be a list when provided"
    bad_statuses = {"fallback", "degraded", "not_available", "failed", "blocked"}
    observed_bad = sorted({status for status in row_statuses if status in bad_statuses})
    if observed_bad:
        return False, f"non-success planner rows present: {observed_bad}"
    return True, "ok"


def compute_metrics(values: list[float]) -> dict[str, Any]:
    """Compute the small outcome summary used by the comparison report."""
    return {
        "mean_objective": round(sum(values) / len(values), 4) if values else 0.0,
        "max_objective": round(max(values), 4) if values else 0.0,
        "failure_count": sum(1 for value in values if value >= 8.0),
    }


def build_independent_outcome_evaluation(
    payload: dict[str, Any] | None,
    *,
    budget: int,
    n_permutations: int,
    seed: int,
    expected_eval_archive_sha256: str | None = None,
    outcome_contract: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Summarize an independent-outcome packet and its null-test readiness."""
    if payload is None:
        return {
            "status": "not_available",
            "independent_outcomes_available": False,
            "certification_available": False,
            "null_tests_reject_null": False,
            "reason": "no_independent_outcome_payload",
        }

    metadata_ok, metadata_reason = _metadata_allows_independent_outcomes(payload)
    if not metadata_ok:
        return {
            "status": "blocked_invalid_independent_outcomes",
            "independent_outcomes_available": False,
            "certification_available": False,
            "null_tests_reject_null": False,
            "reason": metadata_reason,
            "payload_sha256": payload_sha256(payload),
        }
    if expected_eval_archive_sha256 is not None:
        observed_eval_hash = payload.get("eval_archive_sha256")
        if observed_eval_hash != expected_eval_archive_sha256:
            return {
                "status": "blocked_eval_archive_hash_mismatch",
                "independent_outcomes_available": False,
                "certification_available": False,
                "null_tests_reject_null": False,
                "reason": "independent outcome packet does not match the eval split hash",
                "expected_eval_archive_sha256": expected_eval_archive_sha256,
                "observed_eval_archive_sha256": observed_eval_hash,
                "payload_sha256": payload_sha256(payload),
            }

    if payload.get("schema_version") != "adversarial_independent_outcomes.v2":
        return {
            "status": "blocked_legacy_independent_outcomes",
            "independent_outcomes_available": False,
            "certification_available": False,
            "null_tests_reject_null": False,
            "reason": "legacy flat outcome arrays are not admissible; require v2 rows bound to a frozen contract",
            "payload_sha256": payload_sha256(payload),
        }

    rows = payload.get("rows")
    if isinstance(rows, list):
        rows_ok, rows_reason, proposal_outcomes, random_outcomes, ranked_outcomes = (
            validate_row_lineage(rows, outcome_contract=outcome_contract)
        )
        if not rows_ok:
            return {
                "status": "blocked_invalid_independent_outcomes",
                "independent_outcomes_available": False,
                "certification_available": False,
                "null_tests_reject_null": False,
                "reason": f"invalid row lineage: {rows_reason}",
                "payload_sha256": payload_sha256(payload),
            }
    else:
        return {
            "status": "blocked_invalid_independent_outcomes",
            "independent_outcomes_available": False,
            "certification_available": False,
            "null_tests_reject_null": False,
            "reason": "v2 independent outcome payload must contain a non-empty rows list",
            "payload_sha256": payload_sha256(payload),
        }
    certification_available = True
    independent_available = bool(proposal_outcomes and random_outcomes and ranked_outcomes)
    if not independent_available:
        return {
            "status": "not_available_empty_outcomes",
            "independent_outcomes_available": False,
            "certification_available": certification_available,
            "null_tests_reject_null": False,
            "reason": "proposal, random, and ranked outcomes must all be non-empty",
            "payload_sha256": payload_sha256(payload),
        }

    shuffled = shuffled_outcome_null_test(
        proposal_outcomes,
        random_outcomes,
        n_permutations=n_permutations,
        seed=seed,
    )
    ranking = ranking_permutation_test(
        ranked_outcomes,
        selection_size=min(budget, len(ranked_outcomes)),
        n_permutations=n_permutations,
        seed=seed + 1,
    )
    nulls_reject = (
        shuffled.get("status") == "complete"
        and ranking.get("status") == "complete"
        and float(shuffled.get("p_value", 1.0)) <= 0.05
        and float(ranking.get("p_value", 1.0)) <= 0.05
    )
    return {
        "schema_version": payload.get("schema_version"),
        "status": "complete",
        "source": payload.get("source", "unspecified"),
        "objective": payload.get("objective", "unspecified"),
        "artifact": payload.get("artifact"),
        "eval_archive_sha256": payload.get("eval_archive_sha256"),
        "outcome_source": payload.get("outcome_source"),
        "independent_outcomes_available": True,
        "certification_available": certification_available,
        "null_tests_reject_null": nulls_reject,
        "proposal_metrics": compute_metrics(proposal_outcomes),
        "random_metrics": compute_metrics(random_outcomes),
        "null_tests": {
            "shuffled_archive_outcomes": shuffled,
            "proposal_ranking_permutation": ranking,
            "required_for_held_out_claim": True,
        },
        "payload_sha256": payload_sha256(payload),
    }
