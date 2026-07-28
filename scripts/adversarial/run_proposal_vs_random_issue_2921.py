#!/usr/bin/env python3
"""Run proposal model vs random candidate sampler under identical budget.

Side-effect-free contract check (issue #6103 / parent #3275):

    uv run python scripts/adversarial/run_proposal_vs_random_issue_2921.py \
        --check-contract configs/adversarial/issue_3275_same_planner_contract.json

The ``--check-contract`` command validates the frozen same-planner contract: it
derives the fit-only payload from the corrected recertification artifact,
asserts the frozen fit count/hash, planner, family, and exclusions, constructs
the fit-only FailureArchiveProposalModel, and verifies that excluded and
held-out-family records cannot influence scores or ranks. It executes no planner
and produces no new empirical outcome; its exact invocation is a required input
to the next sub-issue.

Supplying ``--contract`` to the normal comparison path applies that same frozen
contract: it reads the canonical source archive and candidate-pool search space,
constructs the fit-only, family-invariant model with the map-backed evaluation
geometry, uses the explicit held-out-family split, and rejects a non-frozen
budget, archive, or search-space override.

The comparison run keeps archive-nearness under an explicitly diagnostic
namespace. When valid independent native planner-execution outcomes (the frozen
``adversarial_independent_outcomes.v2`` row contract) are supplied, the
top-level comparison and the issue #2921 stop rule follow those outcomes
exclusively, and the decision vocabulary is exactly ``continue | stop |
inconclusive``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path
from typing import Any

CLAIM_BOUNDARY = (
    "plumbing_validation_only: proposal-vs-random deltas in this report exercise ranking and "
    "report plumbing only. The current objective is archive-nearness, so it is circular with "
    "archive-nearness ranking and is not held-out yield, benchmark evidence, planner-performance "
    "evidence, or evidence that learned proposals improve failure discovery."
)

HELD_OUT_DIAGNOSTIC_BOUNDARY = (
    "held_out_diagnostic_only: proposal-vs-random deltas use externally supplied independent "
    "planner-execution outcomes plus candidate certification and null-test checks. This is "
    "diagnostic evidence for issue #3275 only; it is not benchmark evidence, paper evidence, or "
    "a planner-performance claim without the durable execution artifacts named by the outcome "
    "payload."
)

#: Frozen decision vocabulary for the #3275 contract (no revise, no generic blocked).
ISSUE_3275_DECISION_VOCABULARY = ("continue", "stop", "inconclusive")


def classify_issue_2921_stop_rule(
    *,
    independent_evaluation: dict[str, Any] | None,
) -> dict[str, Any]:
    """Return the issue #2921 ``continue | stop | inconclusive`` decision.

    The decision follows independent native planner-execution outcomes only. When
    no valid independent outcome evaluation is available, the decision is
    ``inconclusive``; it is never ``continue`` or ``stop`` on archive-nearness.
    The vocabulary is exactly :data:`ISSUE_3275_DECISION_VOCABULARY` (no
    ``revise``, no generic ``blocked``).
    """
    if not independent_evaluation or not independent_evaluation.get(
        "independent_outcomes_available"
    ):
        reason = independent_evaluation.get("reason") if independent_evaluation else "not_available"
        return {
            "status": "inconclusive",
            "reason": f"independent_outcomes_unavailable_or_fail_closed:{reason}",
            "vocabulary": list(ISSUE_3275_DECISION_VOCABULARY),
            "evidence_tier": "analysis_only",
            "claim_boundary": (
                "no continue/stop decision without independent planner-execution outcomes"
            ),
        }
    decision = independent_evaluation["decision"]
    return {
        "status": decision["status"],
        "reason": decision["reason"],
        "vocabulary": list(ISSUE_3275_DECISION_VOCABULARY),
        "evidence_tier": "diagnostic_only",
        "claim_boundary": decision["claim_boundary"],
    }


def create_synthetic_search_space() -> Any:
    """Create a default synthetic search space config for diagnostics."""
    from robot_sf.adversarial.config import RangeConfig, SearchSpaceConfig

    return SearchSpaceConfig(
        start_x=RangeConfig(min=0.0, max=10.0),
        start_y=RangeConfig(min=0.0, max=10.0),
        goal_x=RangeConfig(min=0.0, max=10.0),
        goal_y=RangeConfig(min=0.0, max=10.0),
        spawn_time_s=RangeConfig(min=0.0, max=5.0),
        pedestrian_speed_mps=RangeConfig(min=0.5, max=2.0),
        pedestrian_delay_s=RangeConfig(min=0.0, max=3.0),
        scenario_seed=RangeConfig(min=1.0, max=100.0),
    )


def create_synthetic_archive() -> dict[str, Any]:
    """Create a small synthetic archive of failures for testing/diagnostics."""
    return {
        "schema_version": "adversarial_failure_archive.v1",
        "entries": [
            {
                "archive_id": "failure_0000",
                "candidate": {
                    "start": {"x": 2.0, "y": 2.0},
                    "goal": {"x": 8.0, "y": 8.0},
                    "spawn_time_s": 1.0,
                    "pedestrian_speed_mps": 1.2,
                    "pedestrian_delay_s": 0.5,
                    "scenario_seed": 42,
                },
                "failure_attribution": {
                    "primary_failure": "collision",
                    "details": {"termination_reason": "collision"},
                },
                "objective_value": 8.5,
                "normalized_perturbation": 0.1,
            },
            {
                "archive_id": "failure_0001",
                "candidate": {
                    "start": {"x": 3.0, "y": 3.0},
                    "goal": {"x": 7.0, "y": 7.0},
                    "spawn_time_s": 2.0,
                    "pedestrian_speed_mps": 1.5,
                    "pedestrian_delay_s": 1.0,
                    "scenario_seed": 43,
                },
                "failure_attribution": {
                    "primary_failure": "timeout",
                    "details": {"termination_reason": "timeout"},
                },
                "objective_value": 6.0,
                "normalized_perturbation": 0.25,
            },
        ],
    }


def _payload_sha256(payload: dict[str, Any]) -> str:
    """Return a deterministic digest for JSON-like report provenance."""
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _is_sha256_hex(value: Any) -> bool:
    """Return whether ``value`` is a complete SHA-256 hex digest."""
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdefABCDEF" for character in value)
    )


def load_search_space(path: Path | None) -> tuple[str, str, Any, bool]:
    """Load SearchSpaceConfig, returning state and synthetic fallback provenance."""
    from robot_sf.adversarial.config import SearchSpaceConfig

    if path is None:
        return (
            "diagnostic_only",
            "No search space path provided; using synthetic search-space fixture.",
            create_synthetic_search_space(),
            True,
        )
    if path.exists():
        try:
            return (
                "active",
                "Search space loaded successfully.",
                SearchSpaceConfig.from_file(path),
                False,
            )
        except (ValueError, TypeError, OSError) as exc:
            return (
                "blocked",
                f"Failed to load search space: {exc}; using synthetic fixture for plumbing only.",
                create_synthetic_search_space(),
                True,
            )
    return (
        "blocked",
        f"Search space path {path} does not exist; using synthetic fixture for plumbing only.",
        create_synthetic_search_space(),
        True,
    )


def load_archive(path: Path | None) -> tuple[str, str, dict[str, Any], bool]:
    """Load archive data or fallback to synthetic, returning (state, reason, archive_data, is_synthetic)."""
    if path is None:
        return (
            "diagnostic_only",
            "No archive path provided; using synthetic archive fixture.",
            create_synthetic_archive(),
            True,
        )
    if not path.exists():
        return (
            "diagnostic_only",
            f"Archive path {path} does not exist; using synthetic archive fixture.",
            create_synthetic_archive(),
            True,
        )
    if path.stat().st_size == 0:
        return (
            "blocked",
            f"Archive file {path} is empty; using synthetic archive fixture.",
            create_synthetic_archive(),
            True,
        )

    try:
        with open(path, encoding="utf-8") as f:
            archive_data = json.load(f)
        if not isinstance(archive_data, dict) or not archive_data.get("entries"):
            return (
                "blocked",
                "Loaded archive contains no entries or is malformed; using synthetic.",
                create_synthetic_archive(),
                True,
            )
        return "active", "Real archive loaded successfully.", archive_data, False
    except (ValueError, TypeError, json.JSONDecodeError, OSError) as exc:
        return (
            "blocked",
            f"Failed to load archive: {exc}; using synthetic.",
            create_synthetic_archive(),
            True,
        )


def compute_metrics(selection: list[Any], evaluate_fn: Any) -> dict[str, Any]:
    """Compute summary metrics for a candidate selection (archive-nearness diagnostic)."""
    objs = [evaluate_fn(c) for c in selection]
    return {
        "mean_objective": round(sum(objs) / len(objs), 4) if objs else 0.0,
        "max_objective": round(max(objs), 4) if objs else 0.0,
        "failure_count": sum(1 for o in objs if o >= 8.0),
    }


def build_archive_evaluation_provenance(
    archive_data: dict[str, Any],
    *,
    state: str,
    synthetic_archive: bool,
    split_seed: int,
    independent_evaluation: dict[str, Any] | None = None,
    frozen_contract: dict[str, Any] | None = None,
    model_provenance: dict[str, Any] | None = None,
    search_space_provenance: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build archive-evaluation provenance for the comparison report.

    Archive-nearness is recorded here as a diagnostic-only namespace. It can
    never drive the held-out verdict: that requires independent planner-execution
    outcomes, disjointness, certification, and a rejected null.
    """
    independent_evaluation = independent_evaluation or {}
    provenance: dict[str, Any] = {
        "archive_nearness_namespace": "diagnostic_only_cannot_drive_verdict",
        "archive_sha256": _payload_sha256(archive_data),
        "evaluation_outcome_sha256": independent_evaluation.get("payload_sha256"),
        "split_policy": "none_plumbing_fixture",
        "scenario_family_overlap": "not_checked",
        "seed_overlap": "not_checked",
        "archive_id_overlap": "not_checked",
        "disjointness_checks_passed": False,
        "required_for_held_out_claim": True,
    }
    if state != "active" or synthetic_archive:
        provenance["held_out_evidence_status"] = "not_available_plumbing_fixture"
        return provenance

    from robot_sf.adversarial.disjoint_evaluation import (
        archive_sha256,
        classify_held_out_evidence,
        compute_overlap_provenance,
        disjoint_family_split,
        frozen_held_out_family_split,
    )

    entries = archive_data.get("entries", [])
    if frozen_contract is None:
        split = disjoint_family_split(entries, eval_fraction=0.5, seed=split_seed)
    else:
        fit_cfg = frozen_contract["fit"]
        evaluation_cfg = frozen_contract["evaluation"]
        split = frozen_held_out_family_split(
            entries,
            fit_family=fit_cfg["scenario_family"],
            eval_family=evaluation_cfg["scenario_family"],
            fit_entry_ids=fit_cfg["entry_ids"],
        )
    overlap = compute_overlap_provenance(split.fit_entries, split.eval_entries)
    provenance.update(overlap)
    if frozen_contract is not None:
        provenance["split_policy"] = "frozen_same_planner_held_out_family"
        provenance["frozen_contract"] = {
            "schema_version": frozen_contract["schema_version"],
            "fit_family": frozen_contract["fit"]["scenario_family"],
            "eval_family": frozen_contract["evaluation"]["scenario_family"],
            "fit_entry_count": frozen_contract["fit"]["count"],
            "fit_entry_ids_sha256": frozen_contract["fit"]["entry_ids_sha256"],
            "candidate_budget_per_arm": frozen_contract["budget"]["candidate_budget_per_arm"],
            "candidate_pool_size": frozen_contract["budget"]["candidate_pool_size"],
            "candidate_pool_seed": frozen_contract["budget"]["candidate_pool_seed"],
            "null_tests": frozen_contract["null_tests"],
            "model": model_provenance or {},
            "search_space": search_space_provenance or {},
        }
    if search_space_provenance is not None:
        provenance["search_space"] = search_space_provenance
    provenance["fit_archive_sha256"] = archive_sha256(split.fit_entries)
    if frozen_contract is None:
        provenance["eval_archive_sha256"] = archive_sha256(split.eval_entries)
    else:
        # These five rows are held-out-family *goal-planner* failures explicitly
        # excluded from candidate selection and primary-result interpretation.
        # Keep their hash as exclusion provenance, never as lineage for the new
        # social_force candidate cohort materialized by the next slice.
        provenance["eval_archive_sha256"] = None
        provenance["excluded_held_out_archive_sha256"] = archive_sha256(split.eval_entries)
        provenance["excluded_held_out_archive_role"] = (
            "wrong_planner_records_excluded_from_primary_result_lineage"
        )
    provenance["independent_outcome_evaluation"] = independent_evaluation.get(
        "status", "not_available_requires_planner_execution"
    )
    provenance["held_out_evidence_status"] = classify_held_out_evidence(
        disjointness_checks_passed=overlap["disjointness_checks_passed"],
        independent_outcomes_available=bool(
            independent_evaluation.get("independent_outcomes_available")
        ),
        certification_available=bool(independent_evaluation.get("certification_available")),
        null_tests_reject_null=bool(independent_evaluation.get("null_tests_reject_null")),
    )
    return provenance


def _negative_regression_checks(
    payload: Any,
    archive: dict[str, Any],
    *,
    fit_cfg: dict[str, Any],
    excl_cfg: dict[str, Any],
) -> tuple[list[str], dict[str, Any]]:
    """Verify the full archive cannot reintroduce non-nominal or held-out records."""
    from robot_sf.adversarial.proposal_model import FailureArchiveProposalModel

    failures: list[str] = []
    full_model = FailureArchiveProposalModel(
        archive, fit_entry_ids=payload.entry_ids, feature_view="absolute"
    )
    full_entry_ids = [entry.get("archive_id") for entry in full_model.entries]
    same = sorted(full_entry_ids) == sorted(payload.entry_ids)
    dropped_ids = set(full_model.excluded_entry_ids)
    expected_held_out_ids = set(payload.excluded_entry_ids)
    expected_non_eligible_ids = set(payload.non_eligible_fit_entry_ids)
    expected_dropped_ids = expected_held_out_ids | expected_non_eligible_ids
    checks = {
        "negative_regression_full_archive_same_fit_entries": same,
        "negative_regression_non_fit_dropped_count": len(dropped_ids),
        "negative_regression_held_out_dropped_count": len(dropped_ids & expected_held_out_ids),
        "negative_regression_non_eligible_fit_dropped_count": len(
            dropped_ids & expected_non_eligible_ids
        ),
        "negative_regression_dropped_ids_match_contract": dropped_ids == expected_dropped_ids,
    }
    if not same:
        failures.append("negative regression failed: full archive changed fit entries")
    if payload.entry_ids != tuple(fit_cfg["entry_ids"]):
        failures.append("negative regression failed: fit IDs do not match contract IDs")
    if payload.excluded_entry_ids != tuple(excl_cfg["entry_ids"]):
        failures.append("negative regression failed: held-out exclusion IDs do not match contract")
    if len(dropped_ids & expected_held_out_ids) != excl_cfg["count"]:
        failures.append(
            "negative regression failed: held-out drop count "
            f"{len(dropped_ids & expected_held_out_ids)} != {excl_cfg['count']}"
        )
    if len(dropped_ids & expected_non_eligible_ids) != fit_cfg["excluded_from_nominal_fit_count"]:
        failures.append(
            "negative regression failed: non-eligible fit drop count "
            f"{len(dropped_ids & expected_non_eligible_ids)} "
            f"!= {fit_cfg['excluded_from_nominal_fit_count']}"
        )
    if dropped_ids != expected_dropped_ids:
        failures.append(
            "negative regression failed: full archive dropped IDs drifted from contract"
        )
    return failures, checks


def _check_fit_only_model(  # noqa: C901
    payload: Any,
    archive: dict[str, Any],
    *,
    search_space: Any,
    fit_cfg: dict[str, Any],
    excl_cfg: dict[str, Any],
    planner_cfg: dict[str, Any],
    drift: dict[str, Any],
) -> tuple[list[str], dict[str, Any]]:
    """Construct the fit-only model, run the negative regression, and collect checks."""
    from robot_sf.adversarial.proposal_model import FailureArchiveProposalModel

    failures: list[str] = []
    checks: dict[str, Any] = {
        "fit_count": payload.count,
        "fit_entry_ids_sha256": payload.entry_ids_sha256,
        "fit_entry_ids_sha256_matches_contract": (
            payload.entry_ids_sha256 == fit_cfg["entry_ids_sha256"]
        ),
        "fit_entry_ids_match_contract": tuple(fit_cfg["entry_ids"]) == payload.entry_ids,
        "excluded_from_nominal_fit_count": len(payload.non_eligible_fit_entry_ids),
        "excluded_from_nominal_fit_ids_sha256": _payload_sha256(
            list(payload.non_eligible_fit_entry_ids)
        ),
        "excluded_from_nominal_fit_ids_sha256_matches_contract": (
            _payload_sha256(list(payload.non_eligible_fit_entry_ids))
            == fit_cfg["excluded_from_nominal_fit_entry_ids_sha256"]
        ),
        "excluded_from_nominal_fit_ids_match_contract": (
            tuple(fit_cfg["excluded_from_nominal_fit_entry_ids"])
            == payload.non_eligible_fit_entry_ids
        ),
        "excluded_count": len(payload.excluded_entry_ids),
        "planner_family_drift": drift,
    }
    if payload.count != fit_cfg["count"]:
        failures.append(f"fit_count drift: {payload.count} != {fit_cfg['count']}")
    if payload.entry_ids_sha256 != fit_cfg["entry_ids_sha256"]:
        failures.append("fit_entry_ids_sha256 does not match contract")
    if payload.entry_ids != tuple(fit_cfg["entry_ids"]):
        failures.append("fit entry IDs do not match contract")
    if len(payload.non_eligible_fit_entry_ids) != fit_cfg["excluded_from_nominal_fit_count"]:
        failures.append("excluded-from-nominal-fit count does not match contract")
    if (
        _payload_sha256(list(payload.non_eligible_fit_entry_ids))
        != fit_cfg["excluded_from_nominal_fit_entry_ids_sha256"]
    ):
        failures.append("excluded-from-nominal-fit IDs SHA-256 does not match contract")
    if payload.non_eligible_fit_entry_ids != tuple(fit_cfg["excluded_from_nominal_fit_entry_ids"]):
        failures.append("excluded-from-nominal-fit IDs do not match contract")
    if len(payload.excluded_entry_ids) != excl_cfg["count"]:
        failures.append(
            f"excluded count drift: {len(payload.excluded_entry_ids)} != {excl_cfg['count']}"
        )
    if drift:
        failures.append(f"planner/family drift: {drift}")

    model = FailureArchiveProposalModel(
        payload.archive_payload,
        search_space,
        fit_entry_ids=payload.entry_ids,
        feature_view="family_invariant",
    )
    fit_ids = set(payload.entry_ids)
    model_entry_ids = {entry.get("archive_id") for entry in model.entries}
    excluded_ids = set(payload.excluded_entry_ids)
    checks.update(
        model_state=model.state,
        model_entry_count=len(model.entries),
        model_entry_ids_match_fit=model_entry_ids == fit_ids,
        no_excluded_record_in_model=model_entry_ids.isdisjoint(excluded_ids),
        no_held_out_family_in_model=all(
            "classic_cross_trap_medium" not in str(aid) for aid in model_entry_ids
        ),
        all_fit_entries_are_group_crossing=all(
            "classic_group_crossing_medium" in str(aid) for aid in model_entry_ids
        ),
    )
    from robot_sf.adversarial.disjoint_evaluation import family_invariant_features

    spatial_names = (
        "robot_start_x_space_fraction",
        "robot_start_y_space_fraction",
        "robot_goal_x_space_fraction",
        "robot_goal_y_space_fraction",
    )
    spatial_vectors = {
        tuple(
            family_invariant_features(entry["candidate"], search_space)[name]
            for name in spatial_names
        )
        for entry in model.entries
    }
    checks["distinct_fit_anchor_spatial_vector_count"] = len(spatial_vectors)
    checks["fit_anchor_spatial_variation_preserved"] = len(spatial_vectors) == payload.count
    if model.state != "active":
        failures.append(f"model not active: state={model.state} reason={model.state_reason}")
    if model_entry_ids != fit_ids:
        failures.append("model entries do not equal the frozen fit IDs")
    if not model_entry_ids.isdisjoint(excluded_ids):
        failures.append("an excluded record entered the fit-only model")
    if not checks["no_held_out_family_in_model"]:
        failures.append("a held-out family record entered the fit-only model")
    if not checks["fit_anchor_spatial_variation_preserved"]:
        failures.append("family-invariant feature view collapsed distinct fit-anchor robot routes")

    neg_failures, neg_checks = _negative_regression_checks(
        payload, archive, fit_cfg=fit_cfg, excl_cfg=excl_cfg
    )
    checks.update(neg_checks)
    failures.extend(neg_failures)
    return failures, checks


def run_check_contract(contract_path: Path, *, repo_root: Path | None = None) -> tuple[int, dict]:
    """Side-effect-free validation of the frozen #3275 contract.

    Derives the fit-only payload from the corrected recertification artifact,
    asserts the frozen fit count/hash/planner/family/exclusions, constructs the
    fit-only model, and verifies that excluded and held-out-family records cannot
    influence scores or ranks. Executes no planner and writes nothing.
    """
    from robot_sf.adversarial.proposal_model import (
        derive_fit_payload_from_recertification,
        load_issue_3275_contract,
        validate_fit_payload_integrity,
    )

    contract = load_issue_3275_contract(contract_path)
    try:
        null_test_params = _contract_null_test_params(contract)
    except ValueError as exc:
        return 1, {
            "ok": False,
            "checks": {},
            "failures": [str(exc)],
            "claim_boundary": contract["claim_boundary"]["label"],
        }
    root = repo_root if repo_root is not None else Path.cwd()
    source = contract["source_lineage"]
    recertification_path = root / source["corrected_recertification_path"]
    recertification_bytes = recertification_path.read_bytes()
    recert = json.loads(recertification_bytes)
    archive_path = root / source["pre_correction_archive_path"]
    archive_bytes = archive_path.read_bytes()
    archive = json.loads(archive_bytes)
    checks: dict[str, Any] = {
        "contract_schema_version": contract["schema_version"],
        "contract_path": str(contract_path),
        "null_tests": null_test_params,
        "recertification_sha256_expected": source["corrected_recertification_sha256"],
        "recertification_sha256_observed": recert.get("recertification_sha256"),
        "recertification_artifact_sha256_expected": source.get(
            "corrected_recertification_artifact_sha256"
        ),
        "recertification_artifact_sha256_observed": hashlib.sha256(
            recertification_bytes
        ).hexdigest(),
        "recertification_all_unchanged": (
            recert.get("counts", {}).get("before_after_status", {}).get("unchanged")
            == recert.get("counts", {}).get("record_count")
        ),
        "pre_correction_archive_sha256_expected": source["pre_correction_archive_sha256"],
        "pre_correction_archive_sha256_observed": hashlib.sha256(archive_bytes).hexdigest(),
        "human_review_gate_open": contract["claim_boundary"].get("human_review_gate_open"),
    }
    failures: list[str] = []
    try:
        search_space, search_space_provenance = _load_frozen_contract_search_space(
            contract, repo_root=root, requested_search_space=None
        )
        checks.update(
            {
                "search_space_path": search_space_provenance["path"],
                "search_space_raw_sha256_expected": contract["evaluation"]["search_space_sha256"],
                "search_space_raw_sha256_observed": search_space_provenance["raw_sha256"],
                "search_space_raw_sha256_matches_contract": True,
            }
        )
    except ValueError as exc:
        checks["search_space_error"] = str(exc)
        return 1, {
            "ok": False,
            "checks": checks,
            "failures": [str(exc)],
            "claim_boundary": contract["claim_boundary"]["label"],
        }
    if recert.get("recertification_sha256") != source["corrected_recertification_sha256"]:
        failures.append(
            "recertification_sha256_mismatch: "
            f"observed={recert.get('recertification_sha256')} "
            f"expected={source['corrected_recertification_sha256']}"
        )
    if (
        checks["recertification_artifact_sha256_observed"]
        != checks["recertification_artifact_sha256_expected"]
    ):
        failures.append("corrected recertification artifact SHA-256 does not match contract")
    if (
        checks["pre_correction_archive_sha256_observed"]
        != checks["pre_correction_archive_sha256_expected"]
    ):
        failures.append("pre-correction archive SHA-256 does not match contract")
    if checks["human_review_gate_open"] is not True:
        failures.append("human research-contract review gate must remain explicitly open")

    fit_cfg = contract["fit"]
    excl_cfg = contract["exclusions"]
    planner_cfg = contract["target_planner"]
    try:
        payload = derive_fit_payload_from_recertification(
            recert,
            archive,
            fit_family=fit_cfg["scenario_family"],
            fit_planner=fit_cfg["target_planner"],
            excluded_family=excl_cfg["scenario_family"],
            required_benchmark_eligibility=fit_cfg["required_benchmark_eligibility"],
            expected_count=fit_cfg["count"],
            expected_ids_sha256=fit_cfg["entry_ids_sha256"],
            expected_non_eligible_count=fit_cfg["excluded_from_nominal_fit_count"],
            expected_non_eligible_ids_sha256=fit_cfg["excluded_from_nominal_fit_entry_ids_sha256"],
        )
        drift = validate_fit_payload_integrity(
            payload,
            expected_planner=planner_cfg["id"],
            expected_planner_config_sha256=planner_cfg["config_sha256"],
        )
    except ValueError as exc:
        checks["error"] = str(exc)
        return 1, {
            "ok": False,
            "checks": checks,
            "failures": [str(exc)],
            "claim_boundary": contract["claim_boundary"]["label"],
        }

    model_failures, model_checks = _check_fit_only_model(
        payload,
        archive,
        search_space=search_space,
        fit_cfg=fit_cfg,
        excl_cfg=excl_cfg,
        planner_cfg=planner_cfg,
        drift=drift,
    )
    checks.update(model_checks)
    failures.extend(model_failures)
    return (0 if not failures else 1), {
        "ok": not failures,
        "checks": checks,
        "failures": failures,
        "claim_boundary": contract["claim_boundary"]["label"],
        "next_subissue_required_input": (
            "uv run python scripts/adversarial/run_proposal_vs_random_issue_2921.py "
            f"--check-contract {contract_path}"
        ),
    }


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Compare proposal model vs random sampler under identical budget."
    )
    parser.add_argument(
        "--check-contract",
        type=Path,
        default=None,
        help="Side-effect-free validation of the frozen #3275 contract config.",
    )
    parser.add_argument(
        "--contract",
        type=Path,
        default=None,
        help="Optional frozen #3275 contract config supplying planner/family/minimally-important.",
    )
    parser.add_argument(
        "--archive", type=Path, default=None, help="Path to failure archive JSON file."
    )
    parser.add_argument(
        "--search-space", type=Path, default=None, help="Path to search space config YAML file."
    )
    parser.add_argument(
        "--budget", type=int, default=12, help="Candidate budget per arm (identical)."
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for candidate pool.")
    parser.add_argument("--output", type=Path, default=None, help="Path to write the JSON report.")
    parser.add_argument(
        "--evaluation-outcomes",
        type=Path,
        default=None,
        help="Optional v2 row-level independent planner-execution outcome packet.",
    )
    parser.add_argument(
        "--expected-candidate-manifest-hashes",
        type=Path,
        default=None,
        help=(
            "Frozen external v2 arm-manifest binding (exact arm IDs, SHA-256 values, "
            "candidate-pool indexes, scenario seeds, record SHA-256 values, execution-seed "
            "lineage, and pool seed); required before supplied independent outcomes can drive a "
            "decision."
        ),
    )
    parser.add_argument(
        "--null-test-permutations", type=int, default=1000, help="Permutations for diagnostic null."
    )
    parser.add_argument(
        "--minimally-important",
        type=float,
        default=0.20,
        help="Frozen minimally important absolute yield improvement.",
    )
    args = parser.parse_args()
    if args.budget < 0:
        parser.error("--budget must be >= 0")
    if args.null_test_permutations < 1:
        parser.error("--null-test-permutations must be >= 1")
    return args


def load_expected_candidate_manifest_binding(  # noqa: C901, PLR0912
    path: Path | None,
) -> tuple[dict[str, Any] | None, str]:
    """Load an external frozen arm-manifest binding fail-closed.

    The outcome packet cannot establish its own manifest lineage. This separate
    input must use ``adversarial_candidate_manifest_bindings.v2`` and bind
    exact proposal/random membership, every candidate's manifest SHA-256,
    candidate-pool index, scenario seed, expected record SHA-256,
    execution-seed lineage, and the shared candidate-pool seed. The v1 hash-only
    format cannot establish a frozen denominator or complete record lineage and
    is intentionally rejected.
    """
    if path is None:
        return None, "expected candidate-manifest hash binding was not provided"
    if not path.exists():
        return None, f"expected candidate-manifest hash binding does not exist: {path}"
    if path.stat().st_size == 0:
        return None, f"expected candidate-manifest hash binding is empty: {path}"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
        return None, f"failed to load expected candidate-manifest hash binding: {exc}"
    if not isinstance(payload, dict):
        return None, "expected candidate-manifest hash binding must be a JSON object"
    if payload.get("schema_version") != "adversarial_candidate_manifest_bindings.v2":
        return None, "unexpected candidate-manifest binding schema_version; v2 is required"
    bindings = payload.get("candidate_manifest_sha256_by_id")
    if not isinstance(bindings, dict) or not bindings:
        return None, "candidate_manifest_sha256_by_id must be a non-empty object"
    if any(
        not isinstance(manifest_id, str) or not manifest_id or not _is_sha256_hex(digest)
        for manifest_id, digest in bindings.items()
    ):
        return (
            None,
            "candidate-manifest hash binding keys and values must be non-empty strings and SHA-256 hex",
        )
    ids_by_arm = payload.get("candidate_manifest_ids_by_arm")
    if not isinstance(ids_by_arm, dict) or set(ids_by_arm) != {"proposal", "random"}:
        return None, "candidate_manifest_ids_by_arm must define exactly proposal and random arms"
    normalized_ids_by_arm: dict[str, list[str]] = {}
    for arm in ("proposal", "random"):
        raw_ids = ids_by_arm[arm]
        if not isinstance(raw_ids, list) or not raw_ids:
            return None, f"candidate_manifest_ids_by_arm.{arm} must be a non-empty list"
        if any(not isinstance(manifest_id, str) or not manifest_id for manifest_id in raw_ids):
            return None, f"candidate_manifest_ids_by_arm.{arm} must contain non-empty strings"
        if len(set(raw_ids)) != len(raw_ids):
            return None, f"candidate_manifest_ids_by_arm.{arm} must be unique"
        normalized_ids_by_arm[arm] = list(raw_ids)
    expected_ids = set(normalized_ids_by_arm["proposal"]) | set(normalized_ids_by_arm["random"])
    if set(normalized_ids_by_arm["proposal"]) & set(normalized_ids_by_arm["random"]):
        return None, "candidate_manifest_ids_by_arm proposal/random sets must be disjoint"
    if set(bindings) != expected_ids:
        return None, "candidate_manifest_sha256_by_id must cover exactly the predeclared arm IDs"
    candidate_pool_indices = payload.get("candidate_pool_index_by_manifest_id")
    if not isinstance(candidate_pool_indices, dict) or set(candidate_pool_indices) != expected_ids:
        return (
            None,
            "candidate_pool_index_by_manifest_id must cover exactly the predeclared arm IDs",
        )
    if any(
        not isinstance(manifest_id, str)
        or not manifest_id
        or not isinstance(pool_index, int)
        or isinstance(pool_index, bool)
        or pool_index < 0
        for manifest_id, pool_index in candidate_pool_indices.items()
    ):
        return (
            None,
            "candidate_pool_index_by_manifest_id values must be non-negative integers",
        )
    if len(set(candidate_pool_indices.values())) != len(expected_ids):
        return None, "candidate_pool_index_by_manifest_id must assign unique indices"
    scenario_seeds = payload.get("scenario_seed_by_manifest_id")
    if not isinstance(scenario_seeds, dict) or set(scenario_seeds) != expected_ids:
        return None, "scenario_seed_by_manifest_id must cover exactly the predeclared arm IDs"
    if any(
        not isinstance(manifest_id, str)
        or not manifest_id
        or not isinstance(scenario_seed, int)
        or isinstance(scenario_seed, bool)
        for manifest_id, scenario_seed in scenario_seeds.items()
    ):
        return None, "scenario_seed_by_manifest_id values must be integers"
    record_hashes = payload.get("record_sha256_by_manifest_id")
    if not isinstance(record_hashes, dict) or set(record_hashes) != expected_ids:
        return None, "record_sha256_by_manifest_id must cover exactly the predeclared arm IDs"
    if any(
        not isinstance(manifest_id, str) or not manifest_id or not _is_sha256_hex(digest)
        for manifest_id, digest in record_hashes.items()
    ):
        return (
            None,
            "record_sha256_by_manifest_id keys and values must be non-empty strings and SHA-256 hex",
        )
    execution_seeds = payload.get("execution_seeds_by_manifest_id")
    if not isinstance(execution_seeds, dict) or set(execution_seeds) != expected_ids:
        return None, "execution_seeds_by_manifest_id must cover exactly the predeclared arm IDs"
    normalized_execution_seeds: dict[str, list[int]] = {}
    for manifest_id, raw_seeds in execution_seeds.items():
        if not isinstance(raw_seeds, list) or not raw_seeds:
            return None, f"execution seeds for {manifest_id} must be a non-empty list"
        if any(not isinstance(seed, int) or isinstance(seed, bool) for seed in raw_seeds):
            return None, f"execution seeds for {manifest_id} must be integers"
        if len(set(raw_seeds)) != len(raw_seeds):
            return None, f"execution seeds for {manifest_id} must be unique"
        if len(raw_seeds) != 5:
            return None, (
                f"execution seeds for {manifest_id} must contain exactly 5 seeds "
                "for the frozen 3_of_5 confirmation threshold"
            )
        normalized_execution_seeds[manifest_id] = list(raw_seeds)
    candidate_pool_seed = payload.get("candidate_pool_seed")
    if not isinstance(candidate_pool_seed, int) or isinstance(candidate_pool_seed, bool):
        return None, "candidate_pool_seed must be an integer"
    return {
        "schema_version": payload["schema_version"],
        "candidate_manifest_sha256_by_id": {
            str(manifest_id): str(digest) for manifest_id, digest in bindings.items()
        },
        "candidate_pool_index_by_manifest_id": {
            str(manifest_id): int(pool_index)
            for manifest_id, pool_index in candidate_pool_indices.items()
        },
        "scenario_seed_by_manifest_id": {
            str(manifest_id): int(scenario_seed)
            for manifest_id, scenario_seed in scenario_seeds.items()
        },
        "record_sha256_by_manifest_id": {
            str(manifest_id): str(digest) for manifest_id, digest in record_hashes.items()
        },
        "candidate_manifest_ids_by_arm": normalized_ids_by_arm,
        "execution_seeds_by_manifest_id": normalized_execution_seeds,
        "candidate_pool_seed": candidate_pool_seed,
    }, "ok"


def _contract_null_test_params(contract: dict[str, Any]) -> dict[str, Any]:
    """Load and validate the frozen #3275 null-test procedures."""
    null_tests = contract.get("null_tests")
    if not isinstance(null_tests, dict):
        raise ValueError("frozen contract null_tests must be an object")
    primary = null_tests.get("primary")
    if not isinstance(primary, dict) or primary.get("name") != "fisher_exact_two_sided":
        raise ValueError("frozen contract primary null test must be fisher_exact_two_sided")
    alpha = primary.get("alpha")
    if not isinstance(alpha, (int, float)) or isinstance(alpha, bool) or not 0.0 < alpha < 1.0:
        raise ValueError("frozen contract primary null-test alpha must be in (0, 1)")
    if alpha != contract["power_sensitivity"]["alpha_two_sided"]:
        raise ValueError("frozen null-test alpha does not match power-sensitivity alpha")

    diagnostic = null_tests.get("diagnostic_permutation_procedures")
    if not isinstance(diagnostic, dict):
        raise ValueError("frozen contract diagnostic permutation procedures must be an object")
    n_permutations = diagnostic.get("n_permutations")
    seed = diagnostic.get("seed")
    if (
        not isinstance(n_permutations, int)
        or isinstance(n_permutations, bool)
        or n_permutations < 1
    ):
        raise ValueError("frozen diagnostic n_permutations must be a positive integer")
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise ValueError("frozen diagnostic permutation seed must be an integer")

    shuffled = diagnostic.get("shuffled_outcome_label_permutation")
    if not isinstance(shuffled, dict) or shuffled != {
        "alternative": "two_sided",
        "statistic": "proposal_minus_random_candidate_level_failure_yield",
    }:
        raise ValueError("frozen shuffled-outcome permutation procedure is unsupported")
    ranking = diagnostic.get("ranking_permutation")
    if not isinstance(ranking, dict) or ranking != {
        "alternative": "greater",
        "selection_size": "candidate_budget_per_arm",
        "statistic": "proposal_arm_mean_candidate_level_failure_yield",
    }:
        raise ValueError("frozen ranking-permutation procedure is unsupported")
    return {
        "alpha_two_sided": float(alpha),
        "null_test_permutations": n_permutations,
        "null_test_seed": seed,
    }


def _contract_frozen_params(args: argparse.Namespace) -> dict[str, Any]:
    """Read optional frozen planner/family/minimally-important from a contract."""
    if args.contract is None:
        return {
            "expected_target_planner_id": "social_force",
            "expected_target_planner_config_sha256": None,
            "expected_eval_family": "classic_cross_trap_medium",
            "minimally_important": args.minimally_important,
            "confirmation_threshold": "3_of_5",
            "contract": None,
            "candidate_budget_per_arm": args.budget,
            "candidate_pool_size": max(args.budget * 5, 50),
            "candidate_pool_seed": args.seed,
            "expected_execution_commit": None,
            "alpha_two_sided": 0.05,
            "null_test_permutations": args.null_test_permutations,
            "null_test_seed": args.seed,
        }
    from robot_sf.adversarial.proposal_model import load_issue_3275_contract

    contract = load_issue_3275_contract(args.contract)
    null_test_params = _contract_null_test_params(contract)
    return {
        "expected_target_planner_id": contract["target_planner"]["id"],
        "expected_target_planner_config_sha256": contract["target_planner"]["config_sha256"],
        "expected_eval_family": contract["evaluation"]["scenario_family"],
        "minimally_important": contract["power_sensitivity"][
            "minimally_important_absolute_yield_improvement"
        ],
        "confirmation_threshold": (
            "3_of_5" if not contract["failure_admission"]["four_of_five_required"] else "4_of_5"
        ),
        "contract": contract,
        "candidate_budget_per_arm": contract["budget"]["candidate_budget_per_arm"],
        "candidate_pool_size": contract["budget"]["candidate_pool_size"],
        "candidate_pool_seed": contract["budget"]["candidate_pool_seed"],
        "expected_execution_commit": contract["target_planner"]["execution_commit"],
        **null_test_params,
    }


def _load_frozen_contract_archive(
    contract: dict[str, Any],
    *,
    repo_root: Path,
    requested_archive: Path | None,
) -> tuple[dict[str, Any], str]:
    """Load the contract's canonical archive and reject divergent overrides.

    ``--contract`` is a frozen experiment path, not a convenience setting for a
    generic archive. A caller may name an archive only when its raw SHA-256
    exactly matches the contract source; ranking still reads the canonical
    repository copy so the report has one auditable input location.
    """
    source = contract["source_lineage"]
    expected_sha256 = source["pre_correction_archive_sha256"]
    canonical_path = repo_root / source["pre_correction_archive_path"]
    if not canonical_path.is_file():
        raise ValueError(f"frozen contract archive is missing: {canonical_path}")
    observed_sha256 = hashlib.sha256(canonical_path.read_bytes()).hexdigest()
    if observed_sha256 != expected_sha256:
        raise ValueError(
            "frozen contract archive SHA-256 mismatch: "
            f"observed={observed_sha256} expected={expected_sha256}"
        )
    if requested_archive is not None:
        if not requested_archive.is_file():
            raise ValueError(f"--archive override is missing: {requested_archive}")
        requested_sha256 = hashlib.sha256(requested_archive.read_bytes()).hexdigest()
        if requested_sha256 != expected_sha256:
            raise ValueError(
                "--archive override does not match the frozen contract archive SHA-256: "
                f"observed={requested_sha256} expected={expected_sha256}"
            )
    try:
        archive_data = json.loads(canonical_path.read_text(encoding="utf-8"))
    except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError(f"failed to load frozen contract archive: {exc}") from exc
    if not isinstance(archive_data, dict) or not isinstance(archive_data.get("entries"), list):
        raise ValueError("frozen contract archive must be a JSON object with an entries list")
    return archive_data, canonical_path.as_posix()


def _frozen_search_space_contract_fields(contract: dict[str, Any]) -> tuple[str, str]:
    """Return the required canonical search-space path and raw digest from a contract."""
    evaluation = contract.get("evaluation")
    if not isinstance(evaluation, dict):
        raise ValueError("frozen contract evaluation must be an object")
    configured_path = evaluation.get("search_space_path")
    expected_sha256 = evaluation.get("search_space_sha256")
    if not isinstance(configured_path, str) or not configured_path:
        raise ValueError("frozen contract evaluation.search_space_path must be a non-empty string")
    if not isinstance(expected_sha256, str) or not expected_sha256:
        raise ValueError(
            "frozen contract evaluation.search_space_sha256 must be a non-empty string"
        )
    return configured_path, expected_sha256


def _raw_file_sha256(path: Path) -> str:
    """Return the SHA-256 digest of a file's raw bytes."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_frozen_contract_search_space(
    contract: dict[str, Any],
    *,
    repo_root: Path,
    requested_search_space: Path | None,
) -> tuple[Any, dict[str, Any]]:
    """Load the canonical search space and reject raw-byte drift or mismatched overrides."""
    configured_path, expected_sha256 = _frozen_search_space_contract_fields(contract)

    canonical_path = repo_root / configured_path
    resolved_root = repo_root.resolve()
    resolved_path = canonical_path.resolve()
    if not resolved_path.is_relative_to(resolved_root):
        raise ValueError("frozen contract search-space path must stay within the repository")
    if not canonical_path.is_file():
        raise ValueError(f"frozen contract search space is missing: {configured_path}")
    observed_sha256 = _raw_file_sha256(canonical_path)
    if observed_sha256 != expected_sha256:
        raise ValueError(
            "frozen contract search-space SHA-256 mismatch: "
            f"observed={observed_sha256} expected={expected_sha256}"
        )

    override_sha256 = None
    if requested_search_space is not None:
        if not requested_search_space.is_file():
            raise ValueError(f"--search-space override is missing: {requested_search_space}")
        override_sha256 = _raw_file_sha256(requested_search_space)
        if override_sha256 != expected_sha256:
            raise ValueError(
                "--search-space override does not match the frozen contract search-space "
                f"SHA-256: observed={override_sha256} expected={expected_sha256}"
            )

    search_space_state, search_space_reason, search_space, synthetic_search_space = (
        load_search_space(canonical_path)
    )
    if search_space_state != "active" or synthetic_search_space:
        raise ValueError(f"failed to load frozen contract search space: {search_space_reason}")
    return search_space, {
        "path": configured_path,
        "raw_sha256": observed_sha256,
        "override_path": requested_search_space.as_posix() if requested_search_space else None,
        "override_raw_sha256": override_sha256,
        "override_matches_frozen": (
            override_sha256 == expected_sha256 if requested_search_space is not None else None
        ),
    }


def _load_frozen_contract_run_inputs(
    contract: dict[str, Any],
    *,
    repo_root: Path,
    requested_archive: Path | None,
    requested_search_space: Path | None,
) -> tuple[Any, dict[str, Any], dict[str, Any], str, Any, dict[str, Any]]:
    """Load every normal-run input that is pinned by the frozen contract."""
    from robot_sf.adversarial.proposal_model import FailureArchiveProposalModel

    search_space, search_space_provenance = _load_frozen_contract_search_space(
        contract, repo_root=repo_root, requested_search_space=requested_search_space
    )
    archive_data, canonical_archive_path = _load_frozen_contract_archive(
        contract, repo_root=repo_root, requested_archive=requested_archive
    )
    model, model_provenance = FailureArchiveProposalModel.from_frozen_contract(
        contract, repo_root=repo_root
    )
    return (
        search_space,
        search_space_provenance,
        archive_data,
        canonical_archive_path,
        model,
        model_provenance,
    )


def _frozen_binding_matches_generated_arms(
    binding: dict[str, Any] | None,
    *,
    proposal_ids: list[str],
    random_ids: list[str],
    candidate_pool_indices_by_id: dict[str, int],
    candidate_scenario_seeds_by_id: dict[str, int],
    candidate_manifest_sha256_by_id: dict[str, str],
    candidate_pool_seed: int,
    budget_per_arm: int,
) -> str | None:
    """Return a reason when a supplied binding drifts from this frozen draw."""
    if binding is None:
        return "external frozen arm-manifest binding is unavailable"
    expected_ids_by_arm = binding["candidate_manifest_ids_by_arm"]
    for arm, generated_ids in (("proposal", proposal_ids), ("random", random_ids)):
        bound_ids = expected_ids_by_arm[arm]
        if len(bound_ids) != budget_per_arm:
            return (
                f"external {arm} manifest count {len(bound_ids)} != frozen candidate budget "
                f"{budget_per_arm}"
            )
        if bound_ids != generated_ids:
            return f"external {arm} manifest IDs do not match this frozen candidate draw"
        expected_pool_indices = binding["candidate_pool_index_by_manifest_id"]
        expected_scenario_seeds = binding["scenario_seed_by_manifest_id"]
        for manifest_id in generated_ids:
            if (
                binding["candidate_manifest_sha256_by_id"][manifest_id]
                != candidate_manifest_sha256_by_id[manifest_id]
            ):
                return "external candidate_manifest_sha256 does not match this frozen candidate"
            if expected_pool_indices[manifest_id] != candidate_pool_indices_by_id[manifest_id]:
                return "external candidate_pool_index does not match this frozen candidate draw"
            if expected_scenario_seeds[manifest_id] != candidate_scenario_seeds_by_id[manifest_id]:
                return "external scenario_seed does not match this frozen candidate draw"
    if binding["candidate_pool_seed"] != candidate_pool_seed:
        return "external candidate_pool_seed does not match this frozen candidate draw"
    return None


def _admission_spec_from_binding(frozen: dict[str, Any], binding: dict[str, Any] | None) -> Any:
    """Convert a parser-validated external binding into fail-closed admission inputs."""
    from robot_sf.adversarial.independent_outcomes import AdmissionSpec

    return AdmissionSpec(
        expected_target_planner_id=frozen["expected_target_planner_id"],
        expected_eval_family=frozen["expected_eval_family"],
        confirmation_threshold=frozen["confirmation_threshold"],
        expected_target_planner_config_sha256=frozen["expected_target_planner_config_sha256"],
        expected_candidate_manifest_sha256_by_id=(
            dict(binding["candidate_manifest_sha256_by_id"]) if binding is not None else None
        ),
        expected_candidate_pool_index_by_manifest_id=(
            dict(binding["candidate_pool_index_by_manifest_id"]) if binding is not None else None
        ),
        expected_scenario_seed_by_manifest_id=(
            dict(binding["scenario_seed_by_manifest_id"]) if binding is not None else None
        ),
        expected_record_sha256_by_manifest=(
            dict(binding["record_sha256_by_manifest_id"]) if binding is not None else None
        ),
        expected_candidate_manifest_ids_by_arm=(
            {
                arm: tuple(binding["candidate_manifest_ids_by_arm"][arm])
                for arm in ("proposal", "random")
            }
            if binding is not None
            else None
        ),
        expected_execution_seeds_by_manifest_id=(
            {
                manifest_id: tuple(execution_seeds)
                for manifest_id, execution_seeds in binding[
                    "execution_seeds_by_manifest_id"
                ].items()
            }
            if binding is not None
            else None
        ),
        expected_candidate_pool_seed=binding["candidate_pool_seed"]
        if binding is not None
        else None,
        expected_execution_commit=frozen["expected_execution_commit"],
    )


def _contract_configuration_error(reason: str) -> int:
    """Emit a compact fail-closed result for an invalid ``--contract`` invocation."""
    print(
        json.dumps(
            {
                "schema_version": "adversarial_proposal_comparison.v1",
                "state": "blocked",
                "result_classification": "contract_configuration_blocked",
                "reason": reason,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 2


def _contract_cli_override_error(args: argparse.Namespace, frozen: dict[str, Any]) -> str | None:
    """Return why a CLI override violates the frozen contract."""
    contract = frozen["contract"]
    run_budget = frozen["candidate_budget_per_arm"]
    candidate_pool_size = frozen["candidate_pool_size"]
    if contract is not None and args.budget != run_budget:
        return f"--budget {args.budget} does not match frozen candidate_budget_per_arm {run_budget}"
    if contract is not None and args.seed != frozen["candidate_pool_seed"]:
        return (
            f"--seed {args.seed} does not match frozen candidate_pool_seed "
            f"{frozen['candidate_pool_seed']}"
        )
    if contract is not None and args.null_test_permutations != frozen["null_test_permutations"]:
        return (
            f"--null-test-permutations {args.null_test_permutations} does not match "
            f"frozen diagnostic permutation count {frozen['null_test_permutations']}"
        )
    if candidate_pool_size < 2 * run_budget:
        return (
            "candidate pool too small for two disjoint frozen arms: "
            f"pool={candidate_pool_size} budget_per_arm={run_budget}"
        )
    return None


def _resolve_run_state(
    *,
    archive_state: str,
    archive_reason: str,
    search_space_state: str,
    search_space_reason: str,
    outcome_state: str,
    outcome_reason: str,
    synthetic_archive: bool,
    synthetic_search_space: bool,
) -> tuple[str, str]:
    """Compute the run-level state and human-readable reason."""
    state = archive_state
    reason_parts = [archive_reason, search_space_reason, outcome_reason]
    if not synthetic_archive and synthetic_search_space:
        state = "blocked"
        reason_parts.append("Real-archive runs require a real search-space config.")
    if search_space_state == "blocked" and state == "active":
        state = "blocked"
    if outcome_state == "blocked":
        state = "blocked"
    return state, " ".join(reason_parts)


def _diagnostic_archive_nearness(
    model: Any,
    proposal_selection: list[Any],
    random_selection: list[Any],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Compute diagnostic-only archive-nearness metrics (cannot drive the verdict)."""

    def evaluate_objective(candidate: Any) -> float:
        if not model.entries:
            return 0.0
        distances = [model._entry_distance(candidate, entry) for entry in model.entries]
        min_dist = min(distances) if distances else 999.0
        return max(0.0, 10.0 - min_dist)

    random_metrics = compute_metrics(random_selection, evaluate_objective)
    proposal_metrics = compute_metrics(proposal_selection, evaluate_objective)
    comparison = {
        "namespace": "archive_nearness_diagnostic_only_cannot_drive_verdict",
        "mean_objective_improvement": round(
            proposal_metrics["mean_objective"] - random_metrics["mean_objective"], 4
        ),
        "failure_count_improvement": (
            proposal_metrics["failure_count"] - random_metrics["failure_count"]
        ),
    }
    return random_metrics, proposal_metrics, comparison


def _rank_pool_ids_by_candidate_identity(
    model: Any,
    pool: list[Any],
    pool_ids: list[str],
) -> list[str]:
    """Return model rank order expressed in the shared pool's stable IDs.

    ``FailureArchiveProposalModel.rank_candidates`` returns candidate objects,
    not candidate-pool identifiers. The frozen disjoint-by-candidate policy is
    defined over stable pool/manifest IDs, so converting objects to formatted
    strings would disconnect proposal picks from the random arm's exclusion set.
    Object identity is safe here because the ranker returns the exact objects it
    received; duplicate or foreign objects fail closed.
    """
    if len(pool) != len(pool_ids) or len(set(pool_ids)) != len(pool_ids):
        raise ValueError("candidate pool IDs must be unique and match the candidate pool")
    id_by_candidate_identity = {
        id(candidate): pool_id for pool_id, candidate in zip(pool_ids, pool, strict=True)
    }
    if len(id_by_candidate_identity) != len(pool):
        raise ValueError("candidate pool contains duplicate object identities")
    ranked_pool = model.rank_candidates(pool, strategy="nearest_neighbor")
    ranked_ids: list[str] = []
    for candidate, _score in ranked_pool:
        pool_id = id_by_candidate_identity.get(id(candidate))
        if pool_id is None:
            raise ValueError("ranker returned a candidate not present in the shared pool")
        ranked_ids.append(pool_id)
    return ranked_ids


def _candidate_manifest_sha256_by_id(
    model: Any,
    pool_by_id: dict[str, Any],
    *,
    candidate_pool_indices_by_id: dict[str, int],
    candidate_pool_seed: int,
) -> dict[str, str]:
    """Return canonical candidate-manifest hashes for the deterministic pool.

    A v2 binding must identify the manifests regenerated by this frozen draw,
    rather than accepting arbitrary syntactically valid digests that a packet
    and its sidecar could self-attest.
    """
    return {
        candidate_id: _payload_sha256(
            model.emit_manifest(
                candidate,
                generator_seed=candidate_pool_seed,
                candidate_index=candidate_pool_indices_by_id[candidate_id],
            ).to_dict()
        )
        for candidate_id, candidate in pool_by_id.items()
    }


def _assemble_report(  # noqa: PLR0913
    *,
    state: str,
    reason: str,
    synthetic_archive: bool,
    synthetic_search_space: bool,
    search_space_state: str,
    args: argparse.Namespace,
    budget_per_arm: int,
    arms: Any,
    candidate_manifest_sha256_by_id: dict[str, str],
    diagnostic_random_metrics: dict[str, Any],
    diagnostic_proposal_metrics: dict[str, Any],
    diagnostic_comparison: dict[str, Any],
    independent_evaluation: dict[str, Any],
    provenance: dict[str, Any],
) -> dict[str, Any]:
    """Assemble the comparison report from its computed pieces."""
    independent_available = bool(independent_evaluation.get("independent_outcomes_available"))
    if independent_available:
        comparison = independent_evaluation["comparison"]
        comparison_interpretation = "independent_planner_execution_outcomes"
    else:
        # Keep archive-nearness exclusively in ``diagnostic_archive_nearness``.
        # Mirroring it into the top-level comparison would leave a circular
        # metric available to consumers that correctly treat ``comparison`` as
        # the authoritative proposal-vs-random result.
        comparison = {
            "status": "not_available",
            "reason": "independent_planner_execution_outcomes_required",
        }
        if independent_evaluation.get("status", "").startswith("blocked"):
            comparison_interpretation = "independent_outcomes_rejected_by_held_out_gate"
        else:
            comparison_interpretation = "independent_outcomes_not_available"
    held_out_evidence = provenance.get("held_out_evidence_status") == "eligible_held_out_diagnostic"
    return {
        "schema_version": "adversarial_proposal_comparison.v1",
        "state": state,
        "reason": reason,
        "claim_boundary": (
            HELD_OUT_DIAGNOSTIC_BOUNDARY if independent_available else CLAIM_BOUNDARY
        ),
        "result_classification": (
            "held_out_diagnostic_only" if independent_available else "plumbing_validation_only"
        ),
        "held_out_evidence": held_out_evidence,
        "benchmark_evidence": False,
        "planner_performance_claim": False,
        "decision_vocabulary": list(ISSUE_3275_DECISION_VOCABULARY),
        "synthetic_archive": synthetic_archive,
        "synthetic_search_space": synthetic_search_space,
        "search_space_state": search_space_state,
        "budget_per_arm": budget_per_arm,
        "arm_overlap_policy": arms.policy,
        "arm_overlap_ids": arms.overlap_ids,
        "arm_manifest_ids_by_arm": {
            "proposal": list(arms.proposal_ids),
            "random": list(arms.random_ids),
        },
        "arm_manifest_sha256_by_id": {
            candidate_id: candidate_manifest_sha256_by_id[candidate_id]
            for candidate_id in [*arms.proposal_ids, *arms.random_ids]
        },
        "seed": args.seed,
        "diagnostic_archive_nearness": {
            "random_metrics": diagnostic_random_metrics,
            "proposal_metrics": diagnostic_proposal_metrics,
            "comparison": diagnostic_comparison,
        },
        "comparison": comparison,
        "comparison_interpretation": comparison_interpretation,
        "issue_2921_stop_rule": classify_issue_2921_stop_rule(
            independent_evaluation=independent_evaluation
        ),
        "archive_evaluation_provenance": provenance,
        "independent_outcome_evaluation": independent_evaluation,
    }


def _apply_supplied_outcome_binding_gate(
    *,
    state: str,
    reason: str,
    contract: dict[str, Any] | None,
    outcome_data: dict[str, Any] | None,
    binding_for_admission: dict[str, Any] | None,
    binding_failure_reason: str,
) -> tuple[str, str]:
    """Block a frozen run when supplied outcomes lack a matching external binding."""
    if contract is None or outcome_data is None or binding_for_admission is not None:
        return state, reason
    # A supplied execution packet without a matching external binding has no
    # independently frozen denominator or manifest/seed lineage. Keep no-outcome
    # contract smoke runs active, but surface attempted unbound admission as
    # blocked at the top-level as well as in the outcome evaluation.
    return "blocked", f"{reason} External manifest binding blocked: {binding_failure_reason}"


def _resolve_frozen_binding_for_run(  # noqa: PLR0913
    *,
    contract: dict[str, Any] | None,
    manifest_binding: dict[str, Any] | None,
    manifest_binding_reason: str,
    outcome_data: dict[str, Any] | None,
    arms: Any,
    candidate_pool_indices_by_id: dict[str, int],
    candidate_scenario_seeds_by_id: dict[str, int],
    candidate_manifest_sha256_by_id: dict[str, str],
    candidate_pool_seed: int,
    budget_per_arm: int,
    state: str,
    reason: str,
) -> tuple[dict[str, Any] | None, str | None, str, str]:
    """Validate a supplied frozen binding and propagate any admission block."""
    frozen_binding_reason = None
    if contract is not None and manifest_binding is not None:
        frozen_binding_reason = _frozen_binding_matches_generated_arms(
            manifest_binding,
            proposal_ids=arms.proposal_ids,
            random_ids=arms.random_ids,
            candidate_pool_indices_by_id=candidate_pool_indices_by_id,
            candidate_scenario_seeds_by_id=candidate_scenario_seeds_by_id,
            candidate_manifest_sha256_by_id=candidate_manifest_sha256_by_id,
            candidate_pool_seed=candidate_pool_seed,
            budget_per_arm=budget_per_arm,
        )
    binding_for_admission = manifest_binding if frozen_binding_reason is None else None
    state, reason = _apply_supplied_outcome_binding_gate(
        state=state,
        reason=reason,
        contract=contract,
        outcome_data=outcome_data,
        binding_for_admission=binding_for_admission,
        binding_failure_reason=frozen_binding_reason or manifest_binding_reason,
    )
    return binding_for_admission, frozen_binding_reason, state, reason


def main() -> int:
    """Main execution function."""
    args = parse_args()

    if args.check_contract is not None:
        exit_code, verdict = run_check_contract(args.check_contract)
        print(json.dumps(verdict, indent=2, sort_keys=True))
        return exit_code

    try:
        frozen = _contract_frozen_params(args)
    except (KeyError, TypeError, ValueError) as exc:
        return _contract_configuration_error(str(exc))
    contract = frozen["contract"]
    run_budget = frozen["candidate_budget_per_arm"]
    candidate_pool_size = frozen["candidate_pool_size"]
    override_error = _contract_cli_override_error(args, frozen)
    if override_error is not None:
        return _contract_configuration_error(override_error)
    from robot_sf.adversarial.independent_outcomes import (
        build_independent_outcome_evaluation,
        load_independent_outcomes,
    )
    from robot_sf.adversarial.proposal_model import FailureArchiveProposalModel

    model_provenance: dict[str, Any] | None = None
    search_space_provenance: dict[str, Any] | None = None
    if contract is None:
        search_space_state, search_space_reason, search_space, synthetic_search_space = (
            load_search_space(args.search_space)
        )
        archive_state, archive_reason, archive_data, synthetic_archive = load_archive(args.archive)
        model = FailureArchiveProposalModel(archive_data, search_space)
    else:
        repo_root = Path(__file__).resolve().parents[2]
        try:
            (
                search_space,
                search_space_provenance,
                archive_data,
                canonical_archive_path,
                model,
                model_provenance,
            ) = _load_frozen_contract_run_inputs(
                contract,
                repo_root=repo_root,
                requested_archive=args.archive,
                requested_search_space=args.search_space,
            )
        except ValueError as exc:
            return _contract_configuration_error(str(exc))
        search_space_state, synthetic_search_space = "active", False
        search_space_reason = (
            "Frozen contract search space loaded from "
            f"{search_space_provenance['path']}; raw SHA-256 verified."
        )
        archive_state, synthetic_archive = "active", False
        archive_reason = (
            "Frozen contract archive loaded from "
            f"{canonical_archive_path}; fit-only model factory initialized."
        )

    outcome_state, outcome_reason, outcome_data = load_independent_outcomes(
        args.evaluation_outcomes
    )
    manifest_binding, manifest_binding_reason = load_expected_candidate_manifest_binding(
        args.expected_candidate_manifest_hashes
    )
    state, reason = _resolve_run_state(
        archive_state=archive_state,
        archive_reason=archive_reason,
        search_space_state=search_space_state,
        search_space_reason=search_space_reason,
        outcome_state=outcome_state,
        outcome_reason=outcome_reason,
        synthetic_archive=synthetic_archive,
        synthetic_search_space=synthetic_search_space,
    )

    rng = random.Random(args.seed)
    pool = [search_space.sample_candidate(rng) for _ in range(candidate_pool_size)]
    pool_ids = [f"pool_{i}" for i in range(len(pool))]
    candidate_pool_indices_by_id = {
        pool_id: pool_index for pool_index, pool_id in enumerate(pool_ids)
    }
    candidate_scenario_seeds_by_id = {
        pool_id: int(candidate.scenario_seed)
        for pool_id, candidate in zip(pool_ids, pool, strict=True)
    }
    ranked_ids = _rank_pool_ids_by_candidate_identity(model, pool, pool_ids)

    from robot_sf.adversarial.disjoint_evaluation import assign_arms_disjoint_by_candidate

    arms = assign_arms_disjoint_by_candidate(
        ranked_ids, pool_ids, budget_per_arm=run_budget, rng_seed=args.seed
    )
    pool_by_id = dict(zip(pool_ids, pool, strict=True))
    candidate_manifest_sha256_by_id = _candidate_manifest_sha256_by_id(
        model,
        pool_by_id,
        candidate_pool_indices_by_id=candidate_pool_indices_by_id,
        candidate_pool_seed=args.seed,
    )
    proposal_selection = [pool_by_id[candidate_id] for candidate_id in arms.proposal_ids]
    random_selection = [pool_by_id[candidate_id] for candidate_id in arms.random_ids]

    diagnostic_random_metrics, diagnostic_proposal_metrics, diagnostic_comparison = (
        _diagnostic_archive_nearness(model, proposal_selection, random_selection)
    )
    provenance = build_archive_evaluation_provenance(
        archive_data,
        state=state,
        synthetic_archive=synthetic_archive,
        split_seed=args.seed,
        frozen_contract=contract,
        model_provenance=model_provenance,
        search_space_provenance=search_space_provenance,
    )
    binding_for_admission, frozen_binding_reason, state, reason = _resolve_frozen_binding_for_run(
        contract=contract,
        manifest_binding=manifest_binding,
        manifest_binding_reason=manifest_binding_reason,
        outcome_data=outcome_data,
        arms=arms,
        candidate_pool_indices_by_id=candidate_pool_indices_by_id,
        candidate_scenario_seeds_by_id=candidate_scenario_seeds_by_id,
        candidate_manifest_sha256_by_id=candidate_manifest_sha256_by_id,
        candidate_pool_seed=args.seed,
        budget_per_arm=run_budget,
        state=state,
        reason=reason,
    )
    independent_evaluation = build_independent_outcome_evaluation(
        outcome_data,
        budget_per_arm=run_budget,
        minimally_important=frozen["minimally_important"],
        admission_spec=_admission_spec_from_binding(frozen, binding_for_admission),
        expected_eval_archive_sha256=provenance.get("eval_archive_sha256"),
        alpha=frozen["alpha_two_sided"],
        n_permutations=frozen["null_test_permutations"],
        seed=frozen["null_test_seed"],
    )
    if frozen_binding_reason is not None and outcome_data is not None:
        independent_evaluation["reason"] = frozen_binding_reason
    independent_evaluation["candidate_manifest_binding"] = {
        "required": True,
        "available": binding_for_admission is not None,
        "provided": manifest_binding is not None,
        "schema_version": (
            manifest_binding["schema_version"] if manifest_binding is not None else None
        ),
        "exact_arm_membership_required": True,
        "candidate_pool_index_lineage_required": True,
        "scenario_seed_lineage_required": True,
        "record_sha256_lineage_required": True,
        "execution_seed_lineage_required": True,
        "reason": frozen_binding_reason or manifest_binding_reason,
    }
    provenance = build_archive_evaluation_provenance(
        archive_data,
        state=state,
        synthetic_archive=synthetic_archive,
        split_seed=args.seed,
        independent_evaluation=independent_evaluation,
        frozen_contract=contract,
        model_provenance=model_provenance,
        search_space_provenance=search_space_provenance,
    )
    report = _assemble_report(
        state=state,
        reason=reason,
        synthetic_archive=synthetic_archive,
        synthetic_search_space=synthetic_search_space,
        search_space_state=search_space_state,
        args=args,
        budget_per_arm=run_budget,
        arms=arms,
        candidate_manifest_sha256_by_id=candidate_manifest_sha256_by_id,
        diagnostic_random_metrics=diagnostic_random_metrics,
        diagnostic_proposal_metrics=diagnostic_proposal_metrics,
        diagnostic_comparison=diagnostic_comparison,
        independent_evaluation=independent_evaluation,
        provenance=provenance,
    )
    report_str = json.dumps(report, indent=2, sort_keys=True)
    print(report_str)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(report_str + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
