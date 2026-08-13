"""Deterministic content-addressed held-out preflight packet for issue #3275.

This module materializes the step-2 (issue #6104) preflight packet for the frozen
same-planner held-out experiment defined by ``configs/adversarial/issue_3275_same_planner_contract.json``
(issue #6103). It:

- generates the full ``classic_cross_trap_medium`` candidate pool from the frozen
  search space and the frozen ``candidate_pool_seed``;
- certifies structural eligibility of every candidate under the frozen contract
  (search-space bounds, normalized-control uniqueness, target planner, family);
- ranks the pool with the fit-only ``FailureArchiveProposalModel`` constructed on
  exactly the six frozen group-crossing/social-force fit anchors;
- assigns the proposal and random arms under the frozen ``disjoint_by_candidate``
  policy with identical frozen budgets;
- predeclares step-3 execution seeds in a domain demonstrably disjoint from the
  certified-archive certification seeds, the candidate-pool seed, and the candidate
  scenario-seed domain;
- emits content-addressed candidate-pool, proposal-arm, random-arm, external v2
  binding, and step-3 run-plan manifests whose SHA-256 values reproduce from the
  recorded command and code revision.

The module never executes a planner and never collects, imports, looks up, infers,
or inspects any outcome value. Outcome fields are only *declared* as lineage to be
produced by step 3.
"""

from __future__ import annotations

import hashlib
import json
import random
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from robot_sf.adversarial.config import CandidateSpec, SearchSpaceConfig
from robot_sf.adversarial.disjoint_evaluation import assign_arms_disjoint_by_candidate
from robot_sf.adversarial.proposal_model import (
    FailureArchiveProposalModel,
    load_issue_3275_contract,
    validate_frozen_contract_study_design,
)
from robot_sf.adversarial.scenario_manifest import (
    compute_control_hash,
    validate_candidate_manifest,
)

PREFLIGHT_SCHEMA_VERSION = "issue_3275_held_out_preflight.v1"
CANDIDATE_POOL_MANIFEST_SCHEMA = "issue_3275_candidate_pool_manifest.v1"
ARM_MANIFEST_SCHEMA = "issue_3275_arm_manifest.v1"
RUN_PLAN_SCHEMA_VERSION = "issue_3275_step3_run_plan.v1"
BINDINGS_SCHEMA_VERSION = "adversarial_candidate_manifest_bindings.v2"
OUTCOME_SCHEMA_VERSION = "adversarial_independent_outcomes.v2"

#: Frozen step-3 execution-seed domain base (declared, never executed here). It
#: lies above every certified-archive certification seed (maximum observed
#: 2000364) and far from the frozen candidate-pool seed (42) and the candidate
#: scenario-seed domain (100..999), so the declared execution seeds are
#: demonstrably disjoint from every other frozen seed domain.
EXECUTION_SEED_BASE = 8_100_000
CONFIRMATION_SEEDS_PER_CANDIDATE = 5

CLAIM_BOUNDARY = (
    "preflight_evidence_only: this packet proves deterministic manifest "
    "construction, equal frozen arm budgets, seed/lineage separation, duplicate "
    "handling, reproducible hashes, and readiness to run. It produces no planner "
    "execution, no outcome read, and no proposal-yield, benchmark, or "
    "generalization claim."
)

#: Filenames composing the content-addressed packet, excluding the aggregate
#: ``preflight_packet.json`` and ``SHA256SUMS`` themselves.
_PACKET_FILES = (
    "candidate_pool_manifest.json",
    "proposal_arm_manifest.json",
    "random_arm_manifest.json",
    "candidate_manifest_bindings.v2.json",
    "step3_run_plan.json",
    "README.md",
)

_SCENARIO_FAMILY = "classic_cross_trap_medium"
_TARGET_PLANNER = "social_force"


@dataclass(frozen=True)
class _RecordContext:
    """Frozen static lineage shared by every candidate-pool record."""

    candidate: CandidateSpec
    candidate_id: str
    pool_index: int
    pool_seed: int
    score: float
    model_rank: int
    arm: str | None
    selection_rank: int | None
    eligibility: dict[str, Any]
    manifest_sha256: str
    execution_commit: str
    execution_identity: dict[str, str]
    target_planner_id: str
    target_planner_config_sha256: str
    scenario_family: str


def payload_sha256(payload: Any) -> str:
    """Return the deterministic canonical-JSON SHA-256 of a payload."""
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def raw_sha256(path: Path) -> str:
    """Return the SHA-256 of a file's raw bytes."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _repo_relative_path(repo_root: Path, path: Path) -> str:
    """Return a repository-root-relative POSIX path, failing closed on escapes."""
    try:
        return path.resolve().relative_to(repo_root.resolve()).as_posix()
    except ValueError as exc:
        raise ValueError(f"path must live under the repository: {path}") from exc


def candidate_pool_id(pool_index: int) -> str:
    """Return the stable pool/manifest ID for a candidate (matches the main runner)."""
    return f"pool_{pool_index}"


def execution_seeds_for_candidate(pool_index: int) -> list[int]:
    """Return the five declared step-3 confirmation seeds for one candidate.

    The formula ``EXECUTION_SEED_BASE + pool_index * 5 + k`` guarantees unique
    seeds within and across candidates while remaining in a domain disjoint from
    the candidate-pool seed, candidate scenario seeds, and archive-certification
    seeds.
    """
    return [
        EXECUTION_SEED_BASE + pool_index * CONFIRMATION_SEEDS_PER_CANDIDATE + offset
        for offset in range(CONFIRMATION_SEEDS_PER_CANDIDATE)
    ]


def generate_candidate_pool(
    search_space: SearchSpaceConfig,
    *,
    pool_size: int,
    pool_seed: int,
) -> list[CandidateSpec]:
    """Generate the deterministic candidate pool with one shared seeded RNG."""
    if pool_size < 1:
        raise ValueError("candidate pool size must be >= 1")
    rng = random.Random(pool_seed)
    return [search_space.sample_candidate(rng) for _ in range(pool_size)]


def certify_structural_eligibility(
    candidate: CandidateSpec,
    search_space: SearchSpaceConfig,
    *,
    scenario_family: str,
    target_planner: str,
) -> dict[str, Any]:
    """Certify a candidate's structural eligibility under the frozen contract.

    Structural eligibility is outcome-free: the candidate must lie inside the
    frozen search-space bounds, carry a unique normalized control hash, and be
    bound to the frozen evaluation family and target planner.
    """
    errors, warnings = validate_candidate_manifest(
        candidate,
        search_space=search_space,
        existing_hashes=None,
    )
    return {
        "eligible": not errors,
        "errors": list(errors),
        "warnings": list(warnings),
        "family_matches": scenario_family == _SCENARIO_FAMILY,
        "planner_matches": target_planner == _TARGET_PLANNER,
        "normalized_control_hash": compute_control_hash(candidate),
    }


def _collect_archive_seed_domains(archive: dict[str, Any]) -> dict[str, Any]:
    """Collect the certified-archive construction/certification seed domains.

    Read-only metadata extraction: scenario seeds, search seeds, and independent
    confirmation seeds from the certified archive. No outcome value is read.
    """
    scenario_seeds: set[int] = set()
    search_seeds: set[int] = set()
    confirmation_seeds: set[int] = set()
    for entry in archive.get("entries", []):
        if not isinstance(entry, dict):
            continue
        provenance = entry.get("provenance")
        if not isinstance(provenance, dict):
            continue
        scenario_seed = provenance.get("scenario_seed")
        search_seed = provenance.get("search_seed")
        if isinstance(scenario_seed, int):
            scenario_seeds.add(scenario_seed)
        if isinstance(search_seed, int):
            search_seeds.add(search_seed)
        confirmation = provenance.get("independent_confirmation_seeds")
        if isinstance(confirmation, list):
            for seed in confirmation:
                if isinstance(seed, int):
                    confirmation_seeds.add(seed)
    union = scenario_seeds | search_seeds | confirmation_seeds
    return {
        "scenario_seeds": sorted(scenario_seeds),
        "search_seeds": sorted(search_seeds),
        "independent_confirmation_seeds": sorted(confirmation_seeds),
        "union": sorted(union),
        "union_min": min(union) if union else None,
        "union_max": max(union) if union else None,
        "union_size": len(union),
    }


def _check_seed_disjointness(*, candidate: set[int], archive: set[int]) -> dict[str, Any]:
    """Return a machine-checked disjointness verdict for one seed domain."""
    overlap = sorted(candidate & archive)
    return {
        "overlap": overlap,
        "overlap_count": len(overlap),
        "disjoint": not overlap,
    }


def _build_seed_provenance(
    *,
    records_by_id: dict[str, dict[str, Any]],
    archive: dict[str, Any],
    pool_seed: int,
    diagnostic_null_test_seed: int,
) -> dict[str, Any]:
    """Build the seed-provenance and disjointness section of the packet."""
    archive_seed_domains = _collect_archive_seed_domains(archive)
    archive_union = set(archive_seed_domains["union"])
    candidate_scenario_seeds = {int(record["scenario_seed"]) for record in records_by_id.values()}
    execution_seed_union = {
        seed for record in records_by_id.values() for seed in record["execution_seeds"]
    }
    return {
        "candidate_pool_seed": pool_seed,
        "arm_selection_seed": pool_seed,
        "arm_selection_seed_note": (
            "coincides with candidate_pool_seed by the frozen single-shared-pool "
            "candidate_pool_seed_policy; both are disjoint from archive-certification seeds."
        ),
        "fit_algorithm_seed": None,
        "fit_algorithm_seed_note": (
            "the nearest-neighbor family-invariant ranker is seed-free; no fit seed domain exists."
        ),
        "execution_seed_domain": {
            "base": EXECUTION_SEED_BASE,
            "per_candidate": CONFIRMATION_SEEDS_PER_CANDIDATE,
            "min": min(execution_seed_union) if execution_seed_union else None,
            "max": max(execution_seed_union) if execution_seed_union else None,
        },
        "diagnostic_null_test_seed": diagnostic_null_test_seed,
        "archive_certification_seed_domains": archive_seed_domains,
        "disjointness_checks": {
            "candidate_pool_seed_vs_archive": _check_seed_disjointness(
                candidate={pool_seed}, archive=archive_union
            ),
            "candidate_scenario_seeds_vs_archive": _check_seed_disjointness(
                candidate=candidate_scenario_seeds, archive=archive_union
            ),
            "execution_seeds_vs_archive": _check_seed_disjointness(
                candidate=execution_seed_union, archive=archive_union
            ),
            "execution_seeds_vs_candidate_scenario_seeds": _check_seed_disjointness(
                candidate=execution_seed_union, archive=candidate_scenario_seeds
            ),
        },
    }


def _expected_record_sha256(lineage: dict[str, Any]) -> str:
    """Return the predeclared record SHA-256 for one candidate.

    The digest covers the deterministic, outcome-independent candidate lineage
    that step-3 rows must reproduce before admission. Outcome-dependent fields
    are intentionally absent because no outcome exists at preflight time.
    """
    return payload_sha256(lineage)


def _rank_candidates(
    model: FailureArchiveProposalModel, pool: Sequence[CandidateSpec]
) -> list[tuple[str, float]]:
    """Rank the pool by the fit-only model, returning (candidate_id, score).

    Pool IDs are bound by candidate identity (position in the input pool), never
    by rank position, so the top-ranked candidate keeps its stable ``pool_<i>``
    ID and proposal picks stay connected to the random arm's exclusion set.
    """
    if model.state != "active":
        raise ValueError(f"fit-only model is not active for ranking: {model.state_reason}")
    id_to_pool_id = {
        id(candidate): candidate_pool_id(index) for index, candidate in enumerate(pool)
    }
    ranked = model.rank_candidates(list(pool), strategy="nearest_neighbor")
    result: list[tuple[str, float]] = []
    for candidate, score in ranked:
        pool_id = id_to_pool_id.get(id(candidate))
        if pool_id is None:
            raise ValueError("ranker returned a candidate not present in the shared pool")
        result.append((pool_id, float(score)))
    return result


def _outcome_row_id(
    candidate_id: str, arm: str | None, selection_rank: int | None, seed_offset: int
) -> str:
    """Return the predeclared independent-outcome row ID for one execution seed."""
    if arm is None or selection_rank is None:
        return f"{candidate_id}_unselected_seed_{seed_offset}"
    return f"{arm}_rank_{selection_rank}_seed_{seed_offset}"


def _build_candidate_record(ctx: _RecordContext) -> dict[str, Any]:
    """Build one JSON-safe candidate-pool manifest record with full lineage."""
    scenario_seed = int(ctx.candidate.scenario_seed)
    execution_seeds = execution_seeds_for_candidate(ctx.pool_index)
    row_lineage = {
        "candidate_manifest_id": ctx.candidate_id,
        "candidate_manifest_sha256": ctx.manifest_sha256,
        "selection_arm": ctx.arm,
        "selection_rank": ctx.selection_rank,
        "candidate_pool_seed": ctx.pool_seed,
        "candidate_pool_index": ctx.pool_index,
        "target_planner_id": ctx.target_planner_id,
        "target_planner_config_sha256": ctx.target_planner_config_sha256,
        "scenario_family": ctx.scenario_family,
        "scenario_seed": scenario_seed,
        "execution_commit": ctx.execution_commit,
        "execution_identity": dict(ctx.execution_identity),
    }
    return {
        "candidate_manifest_id": ctx.candidate_id,
        "candidate_pool_index": ctx.pool_index,
        "scenario_family": ctx.scenario_family,
        "target_planner_id": ctx.target_planner_id,
        "target_planner_config_sha256": ctx.target_planner_config_sha256,
        "execution_identity": dict(ctx.execution_identity),
        "candidate_controls": ctx.candidate.to_json(),
        "scenario_seed": scenario_seed,
        "normalized_control_hash": ctx.eligibility["normalized_control_hash"],
        "structural_eligibility": {
            "eligible": ctx.eligibility["eligible"],
            "errors": ctx.eligibility["errors"],
            "warnings": ctx.eligibility["warnings"],
        },
        "score": float(ctx.score),
        "model_rank": int(ctx.model_rank),
        "arm": ctx.arm,
        "selection_rank": ctx.selection_rank,
        "candidate_manifest_sha256": ctx.manifest_sha256,
        "record_sha256": _expected_record_sha256(row_lineage),
        "execution_seeds": execution_seeds,
        "expected_outcome_rows": [
            {
                "row_id": _outcome_row_id(
                    ctx.candidate_id, ctx.arm, ctx.selection_rank, seed_offset
                ),
                "execution_seed": execution_seeds[seed_offset],
            }
            for seed_offset in range(CONFIRMATION_SEEDS_PER_CANDIDATE)
        ],
    }


def _step3_run_plan(
    *,
    contract_path: str,
    contract_raw_sha: str,
    target_planner_id: str,
    execution_commit: str,
    execution_identity: dict[str, str],
    budget_per_arm: int,
) -> dict[str, Any]:
    """Return the frozen step-3 execution plan (declaration only, no execution)."""
    return {
        "schema_version": RUN_PLAN_SCHEMA_VERSION,
        "objective": (
            "certified_failure_outcome proposal-vs-random comparison over the predeclared "
            "disjoint candidate pool under identical frozen per-arm budget."
        ),
        "run_command": (
            "uv run python scripts/adversarial/run_proposal_vs_random_issue_2921.py "
            "--contract configs/adversarial/issue_3275_same_planner_contract.json "
            "--expected-candidate-manifest-hashes "
            "docs/context/evidence/issue_3275_same_planner_held_out/"
            "candidate_manifest_bindings.v2.json "
            "--evaluation-outcomes <step3 independent-outcomes packet path>"
        ),
        "resource_class": (
            "local single-CPU worker; canonical SocialForcePlannerAdapter execution only "
            "(no fallback/degraded); no GPU/SLURM requirement declared."
        ),
        "estimated_run_count": 2 * budget_per_arm * 6,
        "estimated_run_count_note": (
            f"{2 * budget_per_arm} selected candidates x (1 deterministic replay + 5 "
            "independent confirmation seeds) = 6 planner executions each; the exact "
            "admission loop is fixed by the step-3 runner."
        ),
        "output_locations": {
            "independent_outcomes_packet": "<step3 output>/independent_outcomes.json",
            "comparison_report": "<step3 output>/report.json",
        },
        "resumability_rules": (
            "step 3 is resumable per candidate and per row under the frozen "
            "adversarial_independent_outcomes.v2 row contract; every admitted row must "
            "match the external v2 binding (exact arm membership, candidate manifest "
            "SHA-256, pool index, scenario seed, execution seeds, pool seed, record "
            "SHA-256), exact canonical adapter identity, and the producer commit captured "
            "separately from the historical planner reference commit."
        ),
        "decision_rule": {
            "vocabulary": ["continue", "stop", "inconclusive"],
            "primary_null_test": {"name": "fisher_exact_two_sided", "alpha": 0.05},
            "underpowered_label": "diagnostic_or_inconclusive",
            "underpowered_note": (
                "at the frozen budget of 12 per arm the minimum detectable yield "
                "difference is about 0.417, above the 0.20 minimally important effect; "
                "results at this budget are diagnostic/inconclusive and can never "
                "continue/stop on an underpowered signal."
            ),
        },
        "frozen_contract_path": str(contract_path),
        "frozen_contract_sha256": contract_raw_sha,
        "target_planner_id": target_planner_id,
        "execution_commit": execution_commit,
        "execution_commit_role": "historical_planner_reference_lineage",
        "execution_identity": dict(execution_identity),
        "producer_commit": "recorded_by_producer_after_merge",
        "producer_command": (
            "uv run python scripts/adversarial/materialize_issue_6105_outcomes.py "
            "--contract configs/adversarial/issue_3275_same_planner_contract.json "
            "--bindings docs/context/evidence/issue_3275_same_planner_held_out/"
            "candidate_manifest_bindings.v2.json --execution-records <raw execution JSONL> "
            "--output <step3 output>/independent_outcomes.json"
        ),
        "budget_per_arm": budget_per_arm,
        "authorization": "compute is outside the scope of issue #6104; this plan is declared only.",
    }


def _build_readme(packet: dict[str, Any], pool_manifest: dict[str, Any]) -> str:
    """Build the deterministic plain-language README for the preflight packet."""
    seed_prov = packet["seed_provenance"]
    return (
        "# Issue #3275 same-planner held-out preflight packet (issue #6104)\n"
        "\n"
        "Plain-language summary: this packet predeclares the deterministic candidate pool, "
        "the fit-only model ranking, the disjoint proposal and random arms with identical "
        f"frozen budgets ({packet['candidate_pool']['budget_per_arm']} each), and the "
        "content-addressed step-3 lineage for the held-out "
        f"{packet['target_planner']['id']} experiment on "
        f"{pool_manifest['scenario_family']}. It executes no planner and reads no outcome.\n"
        "\n"
        "## Evidence boundary\n"
        "\n"
        f"{packet['claim_boundary']}\n"
        "\n"
        "## Provenance\n"
        "\n"
        f"- Contract: `{packet['frozen_contract']['path']}` "
        f"(SHA-256 `{packet['frozen_contract']['raw_sha256']}`).\n"
        f"- Archive: `{packet['certified_archive']['path']}` "
        f"(pre-correction SHA-256 `{packet['certified_archive']['pre_correction_archive_sha256']}`).\n"
        f"- Target planner: `{packet['target_planner']['id']}` "
        f"(config SHA-256 `{packet['target_planner']['config_sha256']}`).\n"
        f"- Candidate pool seed `{packet['candidate_pool']['seed']}`, pool size "
        f"{packet['candidate_pool']['size']}, budget {packet['candidate_pool']['budget_per_arm']} per arm.\n"
        f"- Execution-seed domain base {seed_prov['execution_seed_domain']['base']}, "
        "disjoint from every archive-certification seed (max "
        f"{seed_prov['archive_certification_seed_domains']['union_max']}).\n"
        f"- Code revision: `{packet['code_revision']}`.\n"
        "\n"
        "## Duplicate and overlap accounting\n"
        "\n"
        f"- Arm-overlap policy: `{packet['arm_overlap_policy']['name']}` "
        f"(overlap count {packet['arm_overlap_policy']['overlap_count']}).\n"
        f"- Unique normalized control hashes: "
        f"{packet['duplicate_accounting']['unique_normalized_control_hashes']} "
        f"(duplicates {packet['duplicate_accounting']['duplicate_count']}).\n"
        "\n"
        "## Files\n"
        "\n"
        "- `candidate_pool_manifest.json`: full candidate pool with structural eligibility, "
        "rank, score, arm membership, seeds, and hashes.\n"
        "- `proposal_arm_manifest.json` / `random_arm_manifest.json`: the two disjoint arms.\n"
        "- `candidate_manifest_bindings.v2.json`: external v2 binding consumed by step 3.\n"
        "- `preflight_packet.json`: aggregate packet with seed provenance and verification.\n"
        "- `step3_run_plan.json`: frozen step-3 run command, resource class, run count, "
        "output locations, and resumability rules.\n"
        "- `SHA256SUMS`: content-addressed digests for every generated file.\n"
    )


def _arm_manifests(
    pool_manifest: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Derive the proposal and random arm manifests from the candidate pool manifest.

    Arm IDs are ordered by each candidate's 1-based ``selection_rank`` so they
    exactly reproduce the frozen rank-order arms produced by the main runner.
    """
    by_id = {record["candidate_manifest_id"]: record for record in pool_manifest["candidates"]}

    def _arm_ids(arm_name: str) -> list[str]:
        records = [record for record in pool_manifest["candidates"] if record["arm"] == arm_name]
        records.sort(key=lambda record: record["selection_rank"])
        return [record["candidate_manifest_id"] for record in records]

    proposal_ids = _arm_ids("proposal")
    random_ids = _arm_ids("random")

    def _one(arm_name: str, arm_ids: list[str]) -> dict[str, Any]:
        return {
            "schema_version": ARM_MANIFEST_SCHEMA,
            "arm": arm_name,
            "budget": pool_manifest["budget_per_arm"],
            "target_planner": pool_manifest["target_planner"],
            "scenario_family": pool_manifest["scenario_family"],
            "arm_overlap_policy": pool_manifest["arm_overlap_policy"],
            "candidate_manifest_ids": arm_ids,
            "candidate_records": [by_id[candidate_id] for candidate_id in arm_ids],
        }

    return _one("proposal", proposal_ids), _one("random", random_ids)


def _bindings_payload(
    pool_manifest: dict[str, Any],
    proposal_ids: list[str],
    random_ids: list[str],
) -> dict[str, Any]:
    """Build the external ``adversarial_candidate_manifest_bindings.v2`` payload.

    The v2 format binds exactly the selected arm candidates (proposal + random),
    matching the runner's requirement that every id-side mapping covers exactly
    the predeclared arm IDs. Unselected pool candidates live in the candidate-pool
    manifest, not in the admission binding.
    """
    by_id = {record["candidate_manifest_id"]: record for record in pool_manifest["candidates"]}
    selected_ids = [*proposal_ids, *random_ids]

    def _map(key: str) -> dict[str, Any]:
        return {candidate_id: by_id[candidate_id][key] for candidate_id in selected_ids}

    return {
        "schema_version": BINDINGS_SCHEMA_VERSION,
        "candidate_manifest_sha256_by_id": _map("candidate_manifest_sha256"),
        "candidate_pool_index_by_manifest_id": _map("candidate_pool_index"),
        "scenario_seed_by_manifest_id": _map("scenario_seed"),
        "record_sha256_by_manifest_id": _map("record_sha256"),
        "execution_seeds_by_manifest_id": {
            candidate_id: list(by_id[candidate_id]["execution_seeds"])
            for candidate_id in selected_ids
        },
        "candidate_manifest_ids_by_arm": {
            "proposal": list(proposal_ids),
            "random": list(random_ids),
        },
        "candidate_pool_seed": pool_manifest["candidate_pool_seed"],
    }


def build_held_out_preflight(
    contract_path: Path,
    *,
    repo_root: Path,
    code_revision: str,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, str]]:
    """Build the deterministic preflight packet (no planner, no outcome read).

    Returns:
        ``(packet, pool_manifest, candidate_manifest_sha256_by_id)``. The packet
        is the aggregate preflight document; the pool manifest carries every
        candidate record; the digest map is the content-address for each candidate
        manifest payload.
    """
    contract = load_issue_3275_contract(contract_path)
    validate_frozen_contract_study_design(contract)

    model, model_provenance = FailureArchiveProposalModel.from_frozen_contract(
        contract, repo_root=repo_root
    )
    if model.state != "active":
        raise ValueError(f"fit-only model is not active: {model.state_reason}")
    search_space = model.search_space
    if search_space is None:
        raise ValueError("fit-only model must carry the frozen search space")

    source = contract["source_lineage"]
    archive_path = repo_root / source["pre_correction_archive_path"]
    archive = json.loads(archive_path.read_text(encoding="utf-8"))
    observed_archive_sha = raw_sha256(archive_path)
    if observed_archive_sha != source["pre_correction_archive_sha256"]:
        raise ValueError(
            "pre-correction archive SHA-256 mismatch during preflight: "
            f"observed={observed_archive_sha} expected={source['pre_correction_archive_sha256']}"
        )

    fit_cfg = contract["fit"]
    eval_cfg = contract["evaluation"]
    planner_cfg = contract["target_planner"]
    budget_cfg = contract["budget"]
    pool_seed = budget_cfg["candidate_pool_seed"]
    pool_size = budget_cfg["candidate_pool_size"]
    budget_per_arm = budget_cfg["candidate_budget_per_arm"]
    target_planner_id = planner_cfg["id"]
    target_planner_config_sha256 = planner_cfg["config_sha256"]
    scenario_family = eval_cfg["scenario_family"]
    execution_commit = planner_cfg["execution_commit"]
    execution_identity = dict(planner_cfg["execution_identity"])

    pool = generate_candidate_pool(search_space, pool_size=pool_size, pool_seed=pool_seed)
    pool_ids = [candidate_pool_id(index) for index in range(len(pool))]
    pool_by_id = dict(zip(pool_ids, pool, strict=True))

    eligibility_by_id, control_hashes = _certify_pool(
        pool_by_id,
        search_space,
        scenario_family=scenario_family,
        target_planner=target_planner_id,
    )
    duplicate_control_hashes = sorted(
        {control_hash for control_hash in control_hashes if control_hashes.count(control_hash) > 1}
    )

    ranked = _rank_candidates(model, pool)
    ranked_ids = [candidate_id for candidate_id, _score in ranked]
    score_by_id = dict(ranked)

    arms = assign_arms_disjoint_by_candidate(
        ranked_ids, pool_ids, budget_per_arm=budget_per_arm, rng_seed=pool_seed
    )
    arm_by_id, selection_rank_by_id = _arm_membership(arms, pool_ids)

    candidate_manifest_sha256_by_id = _candidate_manifest_digests(
        model, pool_by_id, pool_seed=pool_seed
    )

    records_by_id = _build_records(
        pool_by_id,
        eligibility_by_id,
        score_by_id,
        ranked_ids,
        arm_by_id,
        selection_rank_by_id,
        candidate_manifest_sha256_by_id,
        pool_seed=pool_seed,
        execution_commit=execution_commit,
        execution_identity=execution_identity,
        target_planner_id=target_planner_id,
        target_planner_config_sha256=target_planner_config_sha256,
        scenario_family=scenario_family,
    )

    seed_provenance = _build_seed_provenance(
        records_by_id=records_by_id,
        archive=archive,
        pool_seed=pool_seed,
        diagnostic_null_test_seed=contract["null_tests"]["diagnostic_permutation_procedures"][
            "seed"
        ],
    )

    pool_manifest = {
        "schema_version": CANDIDATE_POOL_MANIFEST_SCHEMA,
        "parent_issue": 3275,
        "contract_issue": 6103,
        "this_issue": 6104,
        "scenario_family": scenario_family,
        "target_planner": {
            "id": target_planner_id,
            "config_sha256": target_planner_config_sha256,
            "execution_commit": execution_commit,
            "execution_identity": execution_identity,
        },
        "candidate_pool_seed": pool_seed,
        "candidate_pool_size": pool_size,
        "budget_per_arm": budget_per_arm,
        "arm_overlap_policy": arms.policy,
        "duplicate_accounting": {
            "unique_normalized_control_hashes": len(set(control_hashes)),
            "duplicate_normalized_control_hashes": duplicate_control_hashes,
            "duplicate_count": len(duplicate_control_hashes),
        },
        "candidates": [records_by_id[candidate_id] for candidate_id in pool_ids],
    }

    packet = {
        "schema_version": PREFLIGHT_SCHEMA_VERSION,
        "parent_issue": 3275,
        "contract_issue": 6103,
        "this_issue": 6104,
        "proposal_model_issue": 2921,
        "certified_archive_issue": 5305,
        "claim_boundary": CLAIM_BOUNDARY,
        "evidence_status": "tracked-compact-evidence",
        "executed_planners": 0,
        "outcome_reads": 0,
        "planner_execution_proof": (
            "no planner was executed and no outcome value was collected, imported, "
            "looked up, inferred, or inspected during preflight; only fit-only archive "
            "metadata, the frozen search space, and the frozen contract were read."
        ),
        "frozen_contract": {
            "path": _repo_relative_path(repo_root, contract_path),
            "raw_sha256": raw_sha256(contract_path),
            "fit": {
                "count": fit_cfg["count"],
                "entry_ids_sha256": fit_cfg["entry_ids_sha256"],
                "excluded_from_nominal_fit_entry_ids_sha256": fit_cfg[
                    "excluded_from_nominal_fit_entry_ids_sha256"
                ],
                "entry_ids": list(fit_cfg["entry_ids"]),
            },
            "exclusions": {
                "count": contract["exclusions"]["count"],
                "entry_ids": list(contract["exclusions"]["entry_ids"]),
            },
        },
        "certified_archive": {
            "path": source["pre_correction_archive_path"],
            "pre_correction_archive_sha256": observed_archive_sha,
            "recertification_sha256": source["corrected_recertification_sha256"],
            "recertification_artifact_sha256": source["corrected_recertification_artifact_sha256"],
        },
        "search_space": {
            "path": eval_cfg["search_space_path"],
            "search_space_file_sha256": model_provenance.get("search_space_sha256"),
        },
        "evaluation_map": {
            "path": eval_cfg["map_file"],
            "map_file_sha256": model_provenance.get("evaluation_map_sha256"),
        },
        "target_planner": {
            "id": target_planner_id,
            "config_sha256": target_planner_config_sha256,
            "execution_commit": execution_commit,
            "execution_identity": execution_identity,
        },
        "candidate_pool": {
            "size": pool_size,
            "seed": pool_seed,
            "budget_per_arm": budget_per_arm,
            "identical_budget_both_arms": (
                budget_per_arm == len(arms.proposal_ids) == len(arms.random_ids)
            ),
        },
        "arm_overlap_policy": {
            "name": arms.policy,
            "overlap_ids": list(arms.overlap_ids),
            "overlap_count": len(arms.overlap_ids),
        },
        "duplicate_accounting": {
            "unique_normalized_control_hashes": len(set(control_hashes)),
            "duplicate_normalized_control_hashes": duplicate_control_hashes,
            "duplicate_count": len(duplicate_control_hashes),
        },
        "seed_provenance": seed_provenance,
        "arm_budget_equality": {
            "proposal": len(arms.proposal_ids),
            "random": len(arms.random_ids),
            "frozen_budget_per_arm": budget_per_arm,
            "equal_and_frozen": (len(arms.proposal_ids) == budget_per_arm == len(arms.random_ids)),
        },
        "code_revision": code_revision,
        "step3_run_plan": "step3_run_plan.json",
    }
    return packet, pool_manifest, candidate_manifest_sha256_by_id


def _certify_pool(
    pool_by_id: dict[str, CandidateSpec],
    search_space: SearchSpaceConfig,
    *,
    scenario_family: str,
    target_planner: str,
) -> tuple[dict[str, dict[str, Any]], list[str]]:
    """Certify structural eligibility for every pool candidate, fail-closed."""
    eligibility_by_id: dict[str, dict[str, Any]] = {}
    control_hashes: list[str] = []
    for candidate_id, candidate in pool_by_id.items():
        eligibility = certify_structural_eligibility(
            candidate,
            search_space,
            scenario_family=scenario_family,
            target_planner=target_planner,
        )
        eligibility_by_id[candidate_id] = eligibility
        control_hashes.append(eligibility["normalized_control_hash"])
        if not eligibility["eligible"]:
            raise ValueError(
                f"candidate {candidate_id} failed structural eligibility: {eligibility['errors']}"
            )
        if not eligibility["family_matches"] or not eligibility["planner_matches"]:
            raise ValueError(f"candidate {candidate_id} drifted from the frozen study design")
    return eligibility_by_id, control_hashes


def _arm_membership(
    arms: Any, pool_ids: Sequence[str]
) -> tuple[dict[str, str | None], dict[str, int | None]]:
    """Map every pool candidate to its arm and 1-based selection rank."""
    arm_by_id: dict[str, str | None] = {}
    selection_rank_by_id: dict[str, int | None] = {}
    for arm_name in ("proposal", "random"):
        for rank, candidate_id in enumerate(getattr(arms, f"{arm_name}_ids"), start=1):
            arm_by_id[candidate_id] = arm_name
            selection_rank_by_id[candidate_id] = rank
    for candidate_id in pool_ids:
        arm_by_id.setdefault(candidate_id, None)
        selection_rank_by_id.setdefault(candidate_id, None)
    return arm_by_id, selection_rank_by_id


def _candidate_manifest_digests(
    model: FailureArchiveProposalModel,
    pool_by_id: dict[str, CandidateSpec],
    *,
    pool_seed: int,
) -> dict[str, str]:
    """Compute the content-addressed candidate-manifest SHA-256 for every candidate."""
    digests: dict[str, str] = {}
    for pool_index, (candidate_id, candidate) in enumerate(pool_by_id.items()):
        manifest_payload = model.emit_manifest(
            candidate,
            generator_seed=pool_seed,
            candidate_index=pool_index,
        ).to_dict()
        digests[candidate_id] = payload_sha256(manifest_payload)
    return digests


def _build_records(  # noqa: PLR0913
    pool_by_id: dict[str, CandidateSpec],
    eligibility_by_id: dict[str, dict[str, Any]],
    score_by_id: dict[str, float],
    ranked_ids: Sequence[str],
    arm_by_id: dict[str, str | None],
    selection_rank_by_id: dict[str, int | None],
    candidate_manifest_sha256_by_id: dict[str, str],
    *,
    pool_seed: int,
    execution_commit: str,
    execution_identity: dict[str, str],
    target_planner_id: str,
    target_planner_config_sha256: str,
    scenario_family: str,
) -> dict[str, dict[str, Any]]:
    """Build the JSON-safe candidate records for the pool manifest."""
    records_by_id: dict[str, dict[str, Any]] = {}
    for pool_index, (candidate_id, candidate) in enumerate(pool_by_id.items()):
        records_by_id[candidate_id] = _build_candidate_record(
            _RecordContext(
                candidate=candidate,
                candidate_id=candidate_id,
                pool_index=pool_index,
                pool_seed=pool_seed,
                score=score_by_id[candidate_id],
                model_rank=ranked_ids.index(candidate_id),
                arm=arm_by_id[candidate_id],
                selection_rank=selection_rank_by_id[candidate_id],
                eligibility=eligibility_by_id[candidate_id],
                manifest_sha256=candidate_manifest_sha256_by_id[candidate_id],
                execution_commit=execution_commit,
                execution_identity=execution_identity,
                target_planner_id=target_planner_id,
                target_planner_config_sha256=target_planner_config_sha256,
                scenario_family=scenario_family,
            )
        )
    _assert_unique_expected_outcome_row_ids(records_by_id)
    return records_by_id


def _assert_unique_expected_outcome_row_ids(records_by_id: dict[str, dict[str, Any]]) -> None:
    """Fail closed when the candidate pool declares an ambiguous outcome row ID."""
    row_ids = [
        row["row_id"]
        for record in records_by_id.values()
        for row in record["expected_outcome_rows"]
    ]
    if len(row_ids) == len(set(row_ids)):
        return
    duplicates = sorted({row_id for row_id in row_ids if row_ids.count(row_id) > 1})
    raise ValueError(
        f"expected outcome row IDs must be globally unique; duplicate IDs: {duplicates}"
    )


def compose_preflight_packet_files(
    contract_path: Path,
    *,
    repo_root: Path,
    code_revision: str,
) -> dict[str, bytes]:
    """Compose the full content-addressed packet as ``{filename: bytes}``.

    ``preflight_packet.json`` lists the SHA-256 of every other generated file and
    is itself listed in ``SHA256SUMS``; neither document references itself, so the
    composition is acyclic and fully reproducible.
    """
    packet, pool_manifest, _manifest_digests = build_held_out_preflight(
        contract_path, repo_root=repo_root, code_revision=code_revision
    )
    proposal_manifest, random_manifest = _arm_manifests(pool_manifest)
    run_plan = _step3_run_plan(
        contract_path=_repo_relative_path(repo_root, contract_path),
        contract_raw_sha=raw_sha256(contract_path),
        target_planner_id=pool_manifest["target_planner"]["id"],
        execution_commit=pool_manifest["target_planner"]["execution_commit"],
        execution_identity=pool_manifest["target_planner"]["execution_identity"],
        budget_per_arm=pool_manifest["budget_per_arm"],
    )
    bindings = _bindings_payload(
        pool_manifest,
        proposal_manifest["candidate_manifest_ids"],
        random_manifest["candidate_manifest_ids"],
    )
    readme = _build_readme(packet, pool_manifest)

    files: dict[str, bytes] = {
        "candidate_pool_manifest.json": _serialize(pool_manifest),
        "proposal_arm_manifest.json": _serialize(proposal_manifest),
        "random_arm_manifest.json": _serialize(random_manifest),
        "candidate_manifest_bindings.v2.json": _serialize(bindings),
        "step3_run_plan.json": _serialize(run_plan),
        "README.md": readme.encode("utf-8"),
    }

    generated = [
        {"path": name, "file_sha256": hashlib.sha256(files[name]).hexdigest()}
        for name in _PACKET_FILES
    ]
    packet["generated_files"] = list(generated)
    packet["verification"] = {
        "planner_runs": 0,
        "outcome_reads": 0,
        "arm_budget_equality": packet["arm_budget_equality"],
        "hash_reproducibility": "all digests recompute from the recorded command and code revision",
    }
    files["preflight_packet.json"] = _serialize(packet)

    sums_lines = [
        f"{hashlib.sha256(files[name]).hexdigest()}  {name}"
        for name in (*_PACKET_FILES, "preflight_packet.json")
    ]
    files["SHA256SUMS"] = ("\n".join(sums_lines) + "\n").encode("utf-8")
    return files


def materialize_preflight_packet(
    out_dir: Path,
    *,
    contract_path: Path,
    repo_root: Path,
    code_revision: str,
) -> dict[str, Any]:
    """Write the content-addressed preflight packet and return a verification report.

    Reproducible: running this function again at the same ``code_revision`` and
    module revision yields byte-identical files and identical digests.
    """
    files = compose_preflight_packet_files(
        contract_path, repo_root=repo_root, code_revision=code_revision
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, content in files.items():
        (out_dir / name).write_bytes(content)
    return verify_preflight_packet(out_dir, contract_path=contract_path, repo_root=repo_root)


def verify_preflight_packet(
    out_dir: Path,
    *,
    contract_path: Path,
    repo_root: Path,
) -> dict[str, Any]:
    """Re-derive the packet in memory and compare every on-disk file byte-for-byte.

    Side-effect-free, repeatable null/check-only execution: it proves zero planner
    runs and zero outcome reads while confirming that the committed packet
    reproduces exactly from the frozen contract, the recorded code revision, and
    the current module revision.
    """
    packet_path = out_dir / "preflight_packet.json"
    if not packet_path.is_file():
        raise FileNotFoundError(f"preflight packet is missing: {packet_path}")
    recorded_revision = json.loads(packet_path.read_text(encoding="utf-8")).get("code_revision")
    if not isinstance(recorded_revision, str) or not recorded_revision:
        raise ValueError("on-disk preflight_packet.json must record a code_revision")
    recomputed = compose_preflight_packet_files(
        contract_path, repo_root=repo_root, code_revision=recorded_revision
    )

    failures: list[str] = []
    checked: dict[str, dict[str, Any]] = {}
    for name, expected_bytes in recomputed.items():
        path = out_dir / name
        on_disk = path.read_bytes() if path.is_file() else b""
        digest = hashlib.sha256(on_disk).hexdigest()
        expected_digest = hashlib.sha256(expected_bytes).hexdigest()
        checked[name] = {
            "path": name,
            "present": path.is_file(),
            "byte_identical": on_disk == expected_bytes,
            "sha256": digest,
            "recomputed_sha256": expected_digest,
        }
        if not path.is_file():
            failures.append(f"missing generated file: {name}")
        elif on_disk != expected_bytes:
            failures.append(
                f"byte drift in {name}: on-disk SHA-256 {digest} != recomputed {expected_digest}"
            )
    on_disk_sums = (
        (out_dir / "SHA256SUMS").read_bytes() if (out_dir / "SHA256SUMS").is_file() else b""
    )
    recomputed_sums = recomputed.get("SHA256SUMS", b"")
    if on_disk_sums != recomputed_sums:
        failures.append("SHA256SUMS does not match the recomputed packet")

    recomputed_packet: dict[str, Any] = {}
    seed_prov: dict[str, Any] = {}
    try:
        recomputed_packet = json.loads(recomputed["preflight_packet.json"])
        seed_prov = recomputed_packet["seed_provenance"]["disjointness_checks"]
    except (KeyError, TypeError, ValueError, json.JSONDecodeError):
        failures.append("recomputed packet is missing seed-provenance disjointness checks")

    return {
        "schema_version": PREFLIGHT_SCHEMA_VERSION,
        "status": "pass" if not failures else "fail",
        "failures": failures,
        "planner_runs": 0,
        "outcome_reads": 0,
        "files_checked": list(checked),
        "checks": checked,
        "arm_budget_equality": recomputed_packet.get("arm_budget_equality", {}),
        "disjointness_checks": seed_prov,
    }


def _serialize(payload: Any) -> bytes:
    """Serialize a payload with the same deterministic settings used on disk."""
    return (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")
