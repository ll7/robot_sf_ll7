# Issue #3275 same-planner held-out preflight packet (issue #6104)

Plain-language summary: this packet predeclares the deterministic candidate pool, the fit-only model ranking, the disjoint proposal and random arms with identical frozen budgets (12 each), and the content-addressed step-3 lineage for the held-out social_force experiment on classic_cross_trap_medium. It executes no planner and reads no outcome.

## Evidence boundary

preflight_evidence_only: this packet proves deterministic manifest construction, equal frozen arm budgets, seed/lineage separation, duplicate handling, reproducible hashes, and readiness to run. It produces no planner execution, no outcome read, and no proposal-yield, benchmark, or generalization claim.

## Provenance

- Contract: `configs/adversarial/issue_3275_same_planner_contract.json` (SHA-256 `46fd57a985debdf918954ca993ca7d7ea2f0b9e5a5b5cde1d8320cd02dddf974`).
- Archive: `docs/context/evidence/issue_5305_certified_archive/archive.json` (pre-correction SHA-256 `79e022587b35c1c42bc07cfefaf882af473e96841a99ef57f98a4cee26636445`).
- Target planner: `social_force` (config SHA-256 `dfdebd497e19a046e41cb2b1e7d7a7f54cd592ac0a465e4149efff19efa16735`).
- Execution identity: canonical `SocialForcePlannerAdapter` with `social_force_adapter` policy semantics, projecting `velocity_vector_xy` to `unicycle_vw` using `heading_safe_velocity_to_unicycle_vw`.
- Candidate pool seed `42`, pool size 64, budget 12 per arm.
- Execution-seed domain base 8100000, disjoint from every archive-certification seed (max 2000364).
- Code revision: `49bc2bc619c33ac8372d0167068e5273d4cc88fe`.

## Duplicate and overlap accounting

- Arm-overlap policy: `disjoint_by_candidate` (overlap count 0).
- Unique normalized control hashes: 64 (duplicates 0).

## Files

- `candidate_pool_manifest.json`: full candidate pool with structural eligibility, rank, score, arm membership, seeds, and hashes.
- `proposal_arm_manifest.json` / `random_arm_manifest.json`: the two disjoint arms.
- `candidate_manifest_bindings.v2.json`: external v2 binding consumed by step 3.
- `preflight_packet.json`: aggregate packet with seed provenance and verification.
- `step3_run_plan.json`: frozen step-3 run command, resource class, run count, output locations, and resumability rules.
- `SHA256SUMS`: content-addressed digests for every generated file.
