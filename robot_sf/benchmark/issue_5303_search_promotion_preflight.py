"""Side-effect-free preflight for the issue #5303 search-promotion frozen contract.

This checker freezes and reproduces the promotion contract for the existing
adversarial v1 search (Optuna/TPE versus random) on the held-out
``classic_group_crossing_medium`` family for ``scenario_adaptive_hybrid_orca_v2_collision_guard``.

What this module deliberately does NOT do
-----------------------------------------
It never executes planners, never runs a search/replay/confirmation campaign, never
submits Slurm jobs, and never reads evaluation outcomes. Concretely it does not import
or call any of ``robot_sf.adversarial`` execution surfaces (``samplers``, ``search``,
``runtime``, ``qd``, ``warm_start``, ``transfer_matrix``, or any campaign/replay/
benchmark-runner module). It only reads the frozen contract config, the issue #6139
recertification receipt, and the preregistration manifest, recomputes SHA-256 hashes,
and asserts the frozen fields and the power analysis.

The companion test ``tests/adversarial/test_issue_5303_search_promotion_preflight.py``
AST-scans this module's source to prove the side-effect-free contract holds.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

SCHEMA_VERSION = "issue_5303_search_promotion_preflight.v1"
CONTRACT_SCHEMA_VERSION = "issue_5303_search_promotion_contract.v1"
MANIFEST_SCHEMA_VERSION = "issue_5303_search_promotion_manifest.v1"

EXPECTED_ISSUE = 5303
EXPECTED_STEP = 2
EXPECTED_PARENT = 5303

DEFAULT_CONTRACT_PATH = Path("configs/adversarial/issue_5303_search_promotion_contract.yaml")
DEFAULT_RECEIPT_PATH = Path(
    "docs/context/evidence/issue_5305_certified_archive/recertification_issue_6139.json"
)
DEFAULT_MANIFEST_PATH = Path(
    "docs/context/evidence/issue_5303_search_promotion_preregistration/contract_frozen.json"
)

EXPECTED_ELIGIBLE_COUNT = 8
EXPECTED_RECORD_COUNT = 17
EXPECTED_ELIGIBLE_FLOOR = 2
EXPECTED_FIT_FAMILY = "classic_cross_trap_medium"
EXPECTED_FRESH_FAMILY = "classic_group_crossing_medium"
EXPECTED_METHODS: tuple[str, ...] = ("optuna", "random")
EXPECTED_CANDIDATE_BUDGET = 64
EXPECTED_SEEDS_PER_METHOD = 3
EXPECTED_HORIZON_STEPS = 100
EXPECTED_DT_S = 0.1
EXPECTED_TIME_CAP_S = 10.0
EXPECTED_DOORWAY_SEEDS: tuple[int, ...] = (128, 130)
EXPECTED_NEGATIVE_CONTROL_FAMILY = "francis2023_blind_corner"
EXPECTED_NULL_TEST_COUNT = 2
EXPECTED_COUNTED_GATE_COUNT = 7
NULL_THRESHOLD = 0.05
DIAGNOSTIC_DECLARATION = "diagnostic_inconclusive"


@dataclass(frozen=True)
class Issue5303PreflightResult:
    """Structured promotion-contract preflight result."""

    contract_path: str
    ready: bool
    blocked: bool
    checks: dict[str, bool]
    blockers: tuple[str, ...]
    warnings: tuple[str, ...]
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_payload(self) -> dict[str, Any]:
        """Return a stable JSON payload for CLI and tests."""
        return {
            "schema_version": SCHEMA_VERSION,
            "contract_path": self.contract_path,
            "ready": self.ready,
            "blocked": self.blocked,
            "checks": self.checks,
            "blockers": list(self.blockers),
            "warnings": list(self.warnings),
            "metadata": self.metadata,
        }


def sha256_file(path: Path) -> str:
    """Return the SHA-256 of a file's raw bytes."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _repo_relative(path: Path, repo_root: Path) -> str:
    try:
        return path.resolve().relative_to(repo_root.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def _resolve(repo_root: Path, value: Any) -> Path | None:
    if not isinstance(value, str) or not value.strip():
        return None
    candidate = Path(value)
    return candidate if candidate.is_absolute() else repo_root / candidate


def _eligible_records_by_family(receipt: dict[str, Any]) -> dict[str, list[str]]:
    """Group eligible archive IDs by family using the receipt's corrected verdicts.

    Returns:
        Mapping of scenario family to the sorted eligible archive IDs recorded by
        the corrected #6139 recertification.
    """
    grouped: dict[str, list[str]] = {}
    for record in receipt.get("records", []):
        after = record.get("after") if isinstance(record.get("after"), dict) else {}
        eligibility = after.get("benchmark_eligibility")
        if eligibility != "eligible":
            continue
        family = record.get("scenario_family")
        archive_id = record.get("archive_id")
        if isinstance(family, str) and isinstance(archive_id, str):
            grouped.setdefault(family, []).append(archive_id)
    for family in grouped:
        grouped[family] = sorted(grouped[family])
    return grouped


def _as_int_tuple(value: Any) -> tuple[int, ...]:
    if not isinstance(value, list):
        return ()
    out: list[int] = []
    for item in value:
        if isinstance(item, bool) or not isinstance(item, int):
            return ()
        out.append(item)
    return tuple(out)


def _approx_equal(left: Any, right: float, tol: float = 1e-12) -> bool:
    try:
        return abs(float(left) - float(right)) <= tol
    except (TypeError, ValueError):
        return False


def _min_permutation_p_values(seeds_per_method: int) -> tuple[int, float, float]:
    """Return (arrangements, min one-sided p, min two-sided p) for a seed-label permutation.

    With ``seeds_per_method`` seeds per method, the permutation null relabels
    ``2 * seeds_per_method`` seed observations. The number of distinct label
    assignments is ``C(2n, n)``; the most extreme observed statistic is one
    arrangement, so the minimum one-sided p is ``1 / C(2n, n)`` and the minimum
    two-sided p is ``2 / C(2n, n)``.
    """
    total = 2 * seeds_per_method
    arrangements = math.comb(total, seeds_per_method)
    min_one_sided = 1.0 / arrangements
    min_two_sided = 2.0 / arrangements
    return arrangements, min_one_sided, min_two_sided


def preflight_issue_5303_contract(  # noqa: C901, PLR0912, PLR0915
    contract_path: Path = DEFAULT_CONTRACT_PATH,
    *,
    receipt_path: Path = DEFAULT_RECEIPT_PATH,
    manifest_path: Path = DEFAULT_MANIFEST_PATH,
    repo_root: Path | None = None,
) -> Issue5303PreflightResult:
    """Validate the frozen promotion contract without executing anything.

    Returns:
        Fail-closed result with per-check booleans and blockers.
    """
    root = (repo_root or Path.cwd()).resolve()
    contract_path = contract_path if contract_path.is_absolute() else root / contract_path
    receipt_path = receipt_path if receipt_path.is_absolute() else root / receipt_path
    manifest_path = manifest_path if manifest_path.is_absolute() else root / manifest_path

    checks: dict[str, bool] = {}
    blockers: list[str] = []
    warnings: list[str] = []
    metadata: dict[str, Any] = {}

    # ---- Load the frozen contract -------------------------------------------------
    checks["contract_exists"] = contract_path.is_file()
    if not checks["contract_exists"]:
        blockers.append(f"contract not found: {_repo_relative(contract_path, root)}")
        return Issue5303PreflightResult(
            contract_path=_repo_relative(contract_path, root),
            ready=False,
            blocked=True,
            checks=checks,
            blockers=tuple(blockers),
            warnings=tuple(warnings),
        )

    try:
        contract = yaml.safe_load(contract_path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        blockers.append(f"contract YAML could not be parsed: {exc}")
        return Issue5303PreflightResult(
            contract_path=_repo_relative(contract_path, root),
            ready=False,
            blocked=True,
            checks=checks,
            blockers=tuple(blockers),
            warnings=tuple(warnings),
        )
    if not isinstance(contract, dict):
        blockers.append("contract payload must be a mapping")
        contract = {}

    checks["contract_schema_version"] = contract.get("schema_version") == CONTRACT_SCHEMA_VERSION
    if not checks["contract_schema_version"]:
        blockers.append(
            f"schema_version must be {CONTRACT_SCHEMA_VERSION!r}, "
            f"got {contract.get('schema_version')!r}"
        )

    checks["contract_issue"] = contract.get("issue") == EXPECTED_ISSUE
    if not checks["contract_issue"]:
        blockers.append(f"issue must be {EXPECTED_ISSUE}")
    checks["contract_step"] = contract.get("step") == EXPECTED_STEP
    if not checks["contract_step"]:
        blockers.append(f"step must be {EXPECTED_STEP}")
    checks["contract_parent"] = contract.get("parent_issue") == EXPECTED_PARENT
    if not checks["contract_parent"]:
        blockers.append(f"parent_issue must be {EXPECTED_PARENT}")

    checks["evidence_boundary_proposal_only"] = (
        contract.get("evidence_boundary") == "proposal_preflight_only"
    )
    if not checks["evidence_boundary_proposal_only"]:
        blockers.append("evidence_boundary must stay proposal_preflight_only")

    # ---- Manifest + contract hash (reproduce the frozen contract hash) ------------
    checks["manifest_exists"] = manifest_path.is_file()
    contract_hash = sha256_file(contract_path)
    metadata["contract_file_sha256"] = contract_hash
    manifest_hash: str | None = None
    if checks["manifest_exists"]:
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            manifest = {}
            warnings.append(f"manifest JSON could not be parsed: {exc}")
        manifest_hash = manifest.get("contract_sha256") if isinstance(manifest, dict) else None
        checks["manifest_schema_version"] = (
            isinstance(manifest, dict) and manifest.get("schema_version") == MANIFEST_SCHEMA_VERSION
        )
        if not checks["manifest_schema_version"]:
            blockers.append(f"manifest schema_version must be {MANIFEST_SCHEMA_VERSION!r}")
        checks["contract_hash_matches_manifest"] = manifest_hash == contract_hash
        if not checks["contract_hash_matches_manifest"]:
            blockers.append(
                "contract SHA-256 does not match the frozen manifest hash "
                f"(manifest={manifest_hash!r}, recomputed={contract_hash!r}); "
                "the contract was changed after freezing and must be re-preregistered"
            )
    else:
        checks["manifest_schema_version"] = False
        checks["contract_hash_matches_manifest"] = False
        blockers.append(f"manifest not found: {_repo_relative(manifest_path, root)}")

    # ---- Receipt hashes (contract <-> receipt tamper-evidence) --------------------
    entry_gate = contract.get("entry_gate") if isinstance(contract.get("entry_gate"), dict) else {}
    receipt_resolved = (
        _resolve(root, entry_gate.get("recertification_receipt_path")) or receipt_path
    )
    checks["receipt_exists"] = receipt_resolved.is_file()
    receipt: dict[str, Any] = {}
    if checks["receipt_exists"]:
        try:
            receipt = json.loads(receipt_resolved.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            blockers.append(f"receipt JSON could not be parsed: {exc}")
            receipt = {}
        receipt_file_hash = sha256_file(receipt_resolved)
        metadata["receipt_file_sha256"] = receipt_file_hash
        checks["receipt_file_hash_matches_contract"] = receipt_file_hash == entry_gate.get(
            "recertification_receipt_file_sha256"
        )
        if not checks["receipt_file_hash_matches_contract"]:
            blockers.append(
                "recertification receipt file SHA-256 does not match the contract's "
                f"frozen value (contract={entry_gate.get('recertification_receipt_file_sha256')!r}, "
                f"recomputed={receipt_file_hash!r})"
            )
        receipt_self_declared = receipt.get("recertification_sha256")
        checks["receipt_self_declared_hash_matches_contract"] = (
            receipt_self_declared == entry_gate.get("recertification_self_declared_sha256")
        )
        if not checks["receipt_self_declared_hash_matches_contract"]:
            blockers.append(
                "receipt self-declared recertification_sha256 does not match the contract"
            )
        checks["archive_hash_consistent"] = receipt.get("archive_sha256") == entry_gate.get(
            "certified_archive_sha256"
        )
        if not checks["archive_hash_consistent"]:
            blockers.append(
                "receipt archive_sha256 does not match the contract's certified_archive_sha256"
            )
    else:
        checks["receipt_file_hash_matches_contract"] = False
        checks["receipt_self_declared_hash_matches_contract"] = False
        checks["archive_hash_consistent"] = False
        blockers.append(
            f"recertification receipt not found: {_repo_relative(receipt_resolved, root)}"
        )

    # ---- Entry-gate eligible counts ----------------------------------------------
    eligible_by_family = _eligible_records_by_family(receipt) if receipt else {}
    receipt_eligible_total = sum(len(ids) for ids in eligible_by_family.values())
    receipt_record_count = len(receipt.get("records", [])) if receipt else 0
    checks["entry_gate_record_count"] = receipt_record_count == EXPECTED_RECORD_COUNT
    if not checks["entry_gate_record_count"]:
        blockers.append(
            f"receipt must re-certify {EXPECTED_RECORD_COUNT} records, got {receipt_record_count}"
        )
    checks["entry_gate_eligible_count"] = receipt_eligible_total == EXPECTED_ELIGIBLE_COUNT
    if not checks["entry_gate_eligible_count"]:
        blockers.append(
            f"corrected recertification must leave {EXPECTED_ELIGIBLE_COUNT} eligible "
            f"records, got {receipt_eligible_total}"
        )
    checks["entry_gate_meets_floor"] = receipt_eligible_total >= EXPECTED_ELIGIBLE_FLOOR
    if not checks["entry_gate_meets_floor"]:
        blockers.append(
            "entry gate requires at least two eligible candidates; stop the promotion sequence"
        )
    checks["entry_gate_satisfied"] = bool(entry_gate.get("entry_gate_satisfied")) and (
        receipt_eligible_total >= EXPECTED_ELIGIBLE_FLOOR
    )
    if not checks["entry_gate_satisfied"]:
        blockers.append("entry_gate.entry_gate_satisfied must be true with >= 2 eligible records")

    # ---- Target and neutral reference planner configs exist ----------------------
    target = (
        contract.get("target_planner") if isinstance(contract.get("target_planner"), dict) else {}
    )
    target_cfg = _resolve(root, target.get("config_path"))
    checks["target_planner_name"] = (
        target.get("name") == "scenario_adaptive_hybrid_orca_v2_collision_guard"
    )
    if not checks["target_planner_name"]:
        blockers.append(
            "target_planner.name must be scenario_adaptive_hybrid_orca_v2_collision_guard"
        )
    checks["target_planner_config_exists"] = bool(target_cfg and target_cfg.is_file())
    if not checks["target_planner_config_exists"]:
        blockers.append("target_planner.config_path must point at an existing planner config")

    neutral = (
        contract.get("neutral_reference_planner")
        if isinstance(contract.get("neutral_reference_planner"), dict)
        else {}
    )
    neutral_cfg = _resolve(root, neutral.get("config_path"))
    checks["neutral_reference_config_exists"] = bool(neutral_cfg and neutral_cfg.is_file())
    if not checks["neutral_reference_config_exists"]:
        blockers.append("neutral_reference_planner.config_path must point at an existing config")
    checks["neutral_reference_not_target"] = neutral.get("name") != target.get("name")
    if not checks["neutral_reference_not_target"]:
        blockers.append("neutral reference planner must differ from the target planner")

    # ---- Family split matches the receipt exactly --------------------------------
    family_split = (
        contract.get("family_split") if isinstance(contract.get("family_split"), dict) else {}
    )
    fit_family = family_split.get("fit_tuning_warm_start_family")
    fresh_family = family_split.get("fresh_outcome_family")
    checks["family_split_disjoint"] = (
        fit_family == EXPECTED_FIT_FAMILY
        and fresh_family == EXPECTED_FRESH_FAMILY
        and fit_family != fresh_family
    )
    if not checks["family_split_disjoint"]:
        blockers.append(
            "family_split must be classic_cross_trap_medium (fit) and "
            "classic_group_crossing_medium (fresh outcome), family-disjoint"
        )

    contract_fit_ids = sorted(family_split.get("fit_family_eligible_records", []))
    contract_fresh_ids = sorted(family_split.get("fresh_outcome_family_eligible_records", []))
    receipt_fit_ids = eligible_by_family.get(EXPECTED_FIT_FAMILY, [])
    receipt_fresh_ids = eligible_by_family.get(EXPECTED_FRESH_FAMILY, [])
    checks["fit_family_eligible_ids_match_receipt"] = contract_fit_ids == receipt_fit_ids
    if not checks["fit_family_eligible_ids_match_receipt"]:
        blockers.append(
            "fit_family_eligible_records must match the receipt's eligible "
            f"classic_cross_trap_medium IDs (contract={contract_fit_ids}, receipt={receipt_fit_ids})"
        )
    checks["fresh_family_eligible_ids_match_receipt"] = contract_fresh_ids == receipt_fresh_ids
    if not checks["fresh_family_eligible_ids_match_receipt"]:
        blockers.append(
            "fresh_outcome_family_eligible_records must match the receipt's eligible "
            f"classic_group_crossing_medium IDs (contract={contract_fresh_ids}, receipt={receipt_fresh_ids})"
        )

    all_eligible_ids = set(contract_fit_ids) | set(contract_fresh_ids)
    receipt_excluded_ids = {
        record.get("archive_id")
        for record in receipt.get("records", [])
        if isinstance(record, dict)
        and isinstance(record.get("after"), dict)
        and record.get("after", {}).get("benchmark_eligibility") != "eligible"
    }
    checks["no_excluded_ids_in_eligible_sets"] = all_eligible_ids.isdisjoint(receipt_excluded_ids)
    if not checks["no_excluded_ids_in_eligible_sets"]:
        blockers.append(
            "an excluded (stress_only/knife_edge) record appears in an eligible set; "
            "excluded rows may never be discoveries or denominator rows"
        )
    checks["excluded_records_count"] = family_split.get("excluded_records_count") == (
        EXPECTED_RECORD_COUNT - EXPECTED_ELIGIBLE_COUNT
    )
    if not checks["excluded_records_count"]:
        blockers.append(
            f"excluded_records_count must be {EXPECTED_RECORD_COUNT - EXPECTED_ELIGIBLE_COUNT}"
        )

    # ---- Controls ----------------------------------------------------------------
    controls = contract.get("controls") if isinstance(contract.get("controls"), dict) else {}
    rejection_controls = controls.get("rejection_controls", [])
    doorway_seed_sets = [
        tuple(_as_int_tuple(item.get("seeds")))
        for item in rejection_controls
        if isinstance(item, dict) and item.get("family") == "doorway"
    ]
    checks["doorway_rejection_seeds"] = EXPECTED_DOORWAY_SEEDS in doorway_seed_sets
    if not checks["doorway_rejection_seeds"]:
        blockers.append(f"doorway rejection-control seeds must be {list(EXPECTED_DOORWAY_SEEDS)}")
    negative_control = controls.get("certifier_negative_control")
    checks["certifier_negative_control"] = (
        isinstance(negative_control, dict)
        and negative_control.get("family") == EXPECTED_NEGATIVE_CONTROL_FAMILY
    )
    if not checks["certifier_negative_control"]:
        blockers.append(
            "certifier_negative_control must be francis2023_blind_corner "
            "(certifier negative control only, never a candidate/denominator)"
        )

    # ---- Methods, budget, simulator-time cap -------------------------------------
    methods = contract.get("methods") if isinstance(contract.get("methods"), dict) else {}
    method_names = tuple(
        entry.get("name")
        for entry in methods.get("entries", [])
        if isinstance(entry, dict) and isinstance(entry.get("name"), str)
    )
    checks["methods_exactly_optuna_and_random"] = method_names == EXPECTED_METHODS
    if not checks["methods_exactly_optuna_and_random"]:
        blockers.append(
            f"methods must be exactly {list(EXPECTED_METHODS)}, got {list(method_names)}"
        )
    checks["warm_start_fit_family_only"] = methods.get("warm_start_source") in (
        "fit_family_eligible_records_only",
    )
    if not checks["warm_start_fit_family_only"]:
        blockers.append("warm_start must come from fit-family eligible records only")

    budget = contract.get("budget") if isinstance(contract.get("budget"), dict) else {}
    checks["candidate_budget_64_per_seed"] = (
        budget.get("candidate_budget_per_search_seed_per_method") == EXPECTED_CANDIDATE_BUDGET
    )
    if not checks["candidate_budget_64_per_seed"]:
        blockers.append("candidate budget must be exactly 64 per search seed per method")
    checks["search_seeds_exactly_three"] = (
        budget.get("search_seeds_per_method") == EXPECTED_SEEDS_PER_METHOD
        and len(_as_int_tuple(budget.get("search_seeds"))) == EXPECTED_SEEDS_PER_METHOD
    )
    if not checks["search_seeds_exactly_three"]:
        blockers.append("search seeds must be exactly three per method")

    time_cap = (
        contract.get("simulator_time_cap")
        if isinstance(contract.get("simulator_time_cap"), dict)
        else {}
    )
    checks["simulator_time_cap_frozen"] = (
        time_cap.get("horizon_steps") == EXPECTED_HORIZON_STEPS
        and _approx_equal(time_cap.get("dt_s"), EXPECTED_DT_S)
        and _approx_equal(time_cap.get("simulator_time_cap_s"), EXPECTED_TIME_CAP_S)
    )
    if not checks["simulator_time_cap_frozen"]:
        blockers.append("simulator_time_cap must be horizon=100, dt=0.1, simulator_time_cap_s=10.0")

    # ---- Objective ordering -------------------------------------------------------
    objective = contract.get("objective") if isinstance(contract.get("objective"), dict) else {}
    tier_names = [
        tier.get("name")
        for tier in objective.get("tiers", [])
        if isinstance(tier, dict) and isinstance(tier.get("name"), str)
    ]
    checks["objective_constraints_first"] = tier_names == [
        "collision_or_severe_intrusion",
        "liveness_or_goal_completion",
        "comfort_and_efficiency",
    ]
    if not checks["objective_constraints_first"]:
        blockers.append(
            "objective tiers must be constraints-first: "
            "collision_or_severe_intrusion, liveness_or_goal_completion, comfort_and_efficiency"
        )
    checks["objective_hard_constraint_veto"] = objective.get("ordering") in (
        "constraints_first_lexicographic",
    ) and bool(tier_names)
    if not checks["objective_hard_constraint_veto"]:
        blockers.append("objective.ordering must be constraints_first_lexicographic")

    # ---- Counted weak-point gates -------------------------------------------------
    gates_block = (
        contract.get("counted_weak_point_gates")
        if isinstance(contract.get("counted_weak_point_gates"), dict)
        else {}
    )
    gate_ids = sorted(
        gate.get("id")
        for gate in gates_block.get("gates", [])
        if isinstance(gate, dict) and isinstance(gate.get("id"), int)
    )
    checks["counted_weak_point_gates_all_seven"] = gate_ids == list(
        range(1, EXPECTED_COUNTED_GATE_COUNT + 1)
    )
    if not checks["counted_weak_point_gates_all_seven"]:
        blockers.append(
            f"counted_weak_point_gates must include all {EXPECTED_COUNTED_GATE_COUNT} gates, "
            f"got ids {gate_ids}"
        )
    checks["gates_fail_closed"] = bool(gates_block.get("fail_closed"))
    if not checks["gates_fail_closed"]:
        blockers.append("counted_weak_point_gates.fail_closed must be true")
    confirmation = (
        gates_block.get("confirmation") if isinstance(gates_block.get("confirmation"), dict) else {}
    )
    checks["confirmation_no_retries"] = confirmation.get("no_retries") is True
    if not checks["confirmation_no_retries"]:
        blockers.append("confirmation.no_retries must be true")

    # ---- Estimand / uncertainty / null tests -------------------------------------
    estimand = contract.get("estimand") if isinstance(contract.get("estimand"), dict) else {}
    checks["estimand_frozen"] = (
        estimand.get("clustering") == "candidate_level_clustering_across_search_seeds"
        and estimand.get("independent_unit") == "search_seed"
    )
    if not checks["estimand_frozen"]:
        blockers.append(
            "estimand must cluster candidates across seeds (independent unit = search seed)"
        )

    uncertainty = (
        contract.get("uncertainty") if isinstance(contract.get("uncertainty"), dict) else {}
    )
    checks["uncertainty_seed_clustered"] = (
        uncertainty.get("cluster_unit") == "search_seed"
        and uncertainty.get("clusters_per_method") == EXPECTED_SEEDS_PER_METHOD
    )
    if not checks["uncertainty_seed_clustered"]:
        blockers.append("uncertainty must bootstrap over search-seed clusters (3 per method)")

    null_tests = contract.get("null_tests") if isinstance(contract.get("null_tests"), dict) else {}
    null_test_names = tuple(
        test.get("name")
        for test in null_tests.get("tests", [])
        if isinstance(test, dict) and isinstance(test.get("name"), str)
    )
    checks["null_tests_two_seed_permutations"] = len(
        null_test_names
    ) == EXPECTED_NULL_TEST_COUNT and all(
        test.get("unit") == "search_seed" for test in null_tests.get("tests", [])
    )
    if not checks["null_tests_two_seed_permutations"]:
        blockers.append(
            "null_tests must be exactly the shuffled-outcome and ranking seed permutations "
            "(unit = search_seed)"
        )
    checks["null_tests_two_sided_threshold"] = null_tests.get(
        "sidedness"
    ) == "two_sided" and _approx_equal(null_tests.get("threshold_p"), NULL_THRESHOLD)
    if not checks["null_tests_two_sided_threshold"]:
        blockers.append("null_tests must be two-sided at p <= 0.05")

    mii = (
        contract.get("minimally_important_improvement")
        if isinstance(contract.get("minimally_important_improvement"), dict)
        else {}
    )
    checks["minimally_important_improvement_frozen"] = (
        mii.get("unit") == "additional_unique_fully_admitted_weak_point" and mii.get("value") == 1
    )
    if not checks["minimally_important_improvement_frozen"]:
        blockers.append("minimally_important_improvement must be frozen at one unique weak point")

    decision = (
        contract.get("decision_rule") if isinstance(contract.get("decision_rule"), dict) else {}
    )
    checks["decision_rule_three_outcomes"] = sorted(decision.get("outcomes", [])) == [
        "inconclusive",
        "promote",
        "stop",
    ]
    if not checks["decision_rule_three_outcomes"]:
        blockers.append("decision_rule.outcomes must be promote | stop | inconclusive")

    # ---- Positive gate kept frozen (NOT weakened) --------------------------------
    positive_gate = (
        contract.get("positive_gate") if isinstance(contract.get("positive_gate"), dict) else {}
    )
    checks["positive_gate_thresholds_kept"] = (
        positive_gate.get("admitted_weak_points_floor") == 2
        and positive_gate.get("tpe_minus_random_difference") == "positive"
        and positive_gate.get("ci_95_excludes_zero") is True
        and positive_gate.get("both_null_tests_p_le_0_05") is True
    )
    if not checks["positive_gate_thresholds_kept"]:
        blockers.append("positive_gate thresholds must be kept as proposed (not weakened)")
    checks["positive_gate_not_weakened"] = positive_gate.get("thresholds_weakened") is False
    if not checks["positive_gate_not_weakened"]:
        blockers.append("positive_gate.thresholds_weakened must be false")

    # ---- Power analysis recomputed and cross-checked -----------------------------
    power = (
        contract.get("power_analysis") if isinstance(contract.get("power_analysis"), dict) else {}
    )
    seeds_per_method = EXPECTED_SEEDS_PER_METHOD
    arrangements, min_one_sided, min_two_sided = _min_permutation_p_values(seeds_per_method)
    metadata["power_analysis"] = {
        "permutation_arrangements": arrangements,
        "min_one_sided_permutation_p": min_one_sided,
        "min_two_sided_permutation_p": min_two_sided,
        "null_test_threshold": NULL_THRESHOLD,
    }
    checks["power_arrangements_C_6_3"] = (
        power.get("permutation_arrangements_C_6_3") == arrangements == 20
    )
    if not checks["power_arrangements_C_6_3"]:
        blockers.append(
            f"power_analysis.permutation_arrangements_C_6_3 must equal C(6,3)={arrangements}"
        )
    checks["power_min_two_sided_p"] = _approx_equal(
        power.get("min_two_sided_permutation_p"), min_two_sided
    ) and _approx_equal(min_two_sided, 0.10)
    if not checks["power_min_two_sided_p"]:
        blockers.append(
            f"power_analysis.min_two_sided_permutation_p must equal 2/C(6,3)={min_two_sided:.2f}"
        )
    checks["power_min_one_sided_p"] = _approx_equal(
        power.get("min_one_sided_permutation_p"), min_one_sided
    )
    if not checks["power_min_one_sided_p"]:
        blockers.append(
            f"power_analysis.min_one_sided_permutation_p must equal 1/C(6,3)={min_one_sided:.2f}"
        )
    checks["power_two_sided_cannot_reject"] = (
        power.get("two_sided_can_reject_at_threshold") is False and min_two_sided > NULL_THRESHOLD
    )
    if not checks["power_two_sided_cannot_reject"]:
        blockers.append(
            "power_analysis must record that the two-sided null cannot reject at p<=0.05 "
            f"(min two-sided p = {min_two_sided:.2f} > {NULL_THRESHOLD})"
        )

    # ---- The honest diagnostic declaration (the crux) ----------------------------
    checks["positive_gate_not_robustly_testable"] = (
        power.get("positive_gate_robustly_testable") is False and min_two_sided > NULL_THRESHOLD
    )
    if not checks["positive_gate_not_robustly_testable"]:
        blockers.append(
            "power_analysis.positive_gate_robustly_testable must be false given the "
            "recomputed minimum two-sided permutation p exceeds 0.05"
        )

    future_run = (
        contract.get("future_run_declaration")
        if isinstance(contract.get("future_run_declaration"), dict)
        else {}
    )
    checks["future_run_diagnostic_inconclusive"] = (
        future_run.get("status") == DIAGNOSTIC_DECLARATION
    )
    if not checks["future_run_diagnostic_inconclusive"]:
        blockers.append(
            "future_run_declaration.status must be diagnostic_inconclusive because the "
            "positive gate is not robustly testable under three search seeds"
        )
    checks["diagnostic_thresholds_not_weakened"] = (
        future_run.get("thresholds_not_weakened") is True
        and positive_gate.get("thresholds_weakened") is False
    )
    if not checks["diagnostic_thresholds_not_weakened"]:
        blockers.append(
            "the diagnostic declaration must keep thresholds frozen (thresholds_not_weakened=true)"
        )
    checks["diagnostic_declared_before_outcomes"] = (
        future_run.get("declare_before_outcomes") is True
    )
    if not checks["diagnostic_declared_before_outcomes"]:
        blockers.append("the diagnostic declaration must be made before any outcomes")
    checks["re_preregistration_required_for_promote"] = (
        future_run.get("re_preregistration_required_to_claim_promote") is True
    )
    if not checks["re_preregistration_required_for_promote"]:
        blockers.append("claiming promote later requires re-preregistration with more seeds")

    # ---- Forbidden actions declared ----------------------------------------------
    forbidden = contract.get("forbidden_in_this_step", [])
    checks["forbidden_actions_declared"] = (
        isinstance(forbidden, list)
        and "planner_execution" in forbidden
        and "evaluation_outcome_import_or_read" in forbidden
        and "slurm_or_sbatch_or_srun_submission" in forbidden
    )
    if not checks["forbidden_actions_declared"]:
        blockers.append("forbidden_in_this_step must declare planner/Slurm/outcome-read bans")

    metadata["eligible_by_family"] = eligible_by_family
    metadata["receipt_eligible_total"] = receipt_eligible_total
    ready = not blockers
    return Issue5303PreflightResult(
        contract_path=_repo_relative(contract_path, root),
        ready=ready,
        blocked=not ready,
        checks=checks,
        blockers=tuple(blockers),
        warnings=tuple(warnings),
        metadata=metadata,
    )


def dump_preflight_payload(result: Issue5303PreflightResult, output: Path | None) -> None:
    """Write preflight payload to disk when requested."""
    if output is None:
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result.to_payload(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
