# Issue #5303 step 2b — frozen powered six-seed search-promotion preregistration (contract v2)

## Plain-language summary

This is a **preregistration**: before any new planner run happens, it freezes the exact
rules of the **powered successor** to the historical three-seed #5303 diagnostic. It asks
one narrow question — *does the existing Optuna/Tree-structured Parzen Estimator (TPE)
adversarial search find more fully certified, independently confirmed robot-failure
scenarios than the existing random search, for one specific planner on one held-out
scenario family, under a matched compute budget?* — with **six independent search seeds
per method** and **64 scheduled candidates per seed per method** (384 per method,
**768 total**), so that the approved two-sided promotion rule can actually be evaluated.

With six seed clusters per method, the exact permutation null enumerates
`C(12,6) = 924` arm-label assignments; the minimum attainable two-sided p-value is
`2/924 ≈ 0.00216`, so the frozen two-sided `p <= 0.05` decision boundary **is
representable**. No threshold from the historical design is weakened. This packet
declares the study **promotion-capable**, and it is equally explicit that **this packet
does not authorize the #6145 campaign execution**: execution remains separately reviewed,
and no transfer work may start before a hash-bound `promote` result exists.

Nothing here runs a planner, launches a search, replays a scenario, submits a cluster
job, or reads any evaluation outcome. It only freezes the design, reproduces its hashes,
and emits the deterministic 768 scheduled search identities.

## Evidence boundary

`proposal_preflight_only`. Completing this step proves the powered promotion study is
**preregistered, executable, and fail-closed**: its contract, inputs, complete row schema,
768-identity manifest, exact inference, sensitivity boundary, and machine-readable result
schema are checked before any execution. It is **not** evidence that TPE outperforms
random, that any weak point is promoted, that failures transfer across planners, or that
a minimax comparison is justified. No paper, dissertation, benchmark-wide, cross-planner,
minimax, or portfolio claim is made or implied.

- Issue: <https://github.com/ll7/robot_sf_ll7/issues/6861> (step 2b of 6), parent #5303.
- Historical predecessor: #6144 / PR #6291 (immutable diagnostic history; cannot authorize `promote`).
- Runtime-materialization prerequisite: #6475, terminally reconciled to merged PR #6586
  (merge commit `cfb15fb33009aeb68ab5e336a2c2b0824bcc062a`, merged 2026-08-01).
- Downstream campaign: #6145 (not authorized here). Downstream activation issue: #6146.
- Entry gate: merged #6139 corrected recertification (receipt/archive hashes below).
- Historical blocker #6858 closed 2026-08-10; current `main` green at the base commit below.

## Frozen base commit (hash-freeze point)

Every pre-existing input hash was reproduced byte-for-byte on the exact green
`origin/main` commit:

- **Base commit:** `2b3e3c199f1f0d283ffeed0e0bac55710d8efccc`
- **CI evidence:** all 20 GitHub check runs for that commit completed
  `success` (19) or skipped-by-design (`reproducibility-check`, 1) on 2026-08-10/11.
- The new powered surfaces (this packet's contract v2, space v2, template v2, module,
  CLI) did not exist at the base commit; their hashes are frozen as committed with this
  packet, and the checker recomputes every hash at each run.

## Outcome-free provenance refresh (2026-08-12)

Issue #6464 adds diagnostic-only BRNE entries to the shared algorithm metadata
registry. Because this registry is an explicitly hashed input to the powered
contract, its raw SHA-256 and the derived contract/manifest hashes were refreshed
before any #6145 outcome was generated. The target planner, family split, candidate
space, budget, estimator, gates, denominator, uncertainty, null tests, decision rule,
and evidence boundary are unchanged; this refresh does not authorize execution or
re-preregister the study's design.

## The claim under test (frozen, falsifiable)

> Under a frozen family-disjoint design and matched candidate (64 per search seed per
> method) and simulator-time budgets, the existing Optuna/TPE search yields more fully
> certified and independently confirmed weak points than the existing random search for
> `scenario_adaptive_hybrid_orca_v2_collision_guard` on the held-out
> `classic_group_crossing_medium` family, using exactly six independent search seeds per
> method (384 scheduled attempts per method, 768 total) and a constraints-first objective
> ordering.

## Seed and budget contract (outcome-free)

- **Seed roster:** exactly `530301, 530302, 530303, 530304, 530305, 530306` for **both**
  methods (each seed is a paired cluster). Derivation is an explicit, listed,
  outcome-independent set (`seed i = 530300 + i`, i = 1..6), frozen before any outcome.
- **No post-outcome seed addition, replacement, retry, or stopping.**
- **Budget:** exactly 64 scheduled candidates per seed per method; 384 per method;
  768 total, with complete intention-to-search accounting.
- **Scheduled identities:** the deterministic 768-identity manifest
  ([scheduled_search_identities.json](scheduled_search_identities.json)) is a pure
  function of the frozen constants; its SHA-256 is pinned in
  [contract_frozen.json](contract_frozen.json) and re-derived by the checker.

## Frozen design (unchanged boundary from #5303 unless stated)

- **Target planner:** `scenario_adaptive_hybrid_orca_v2_collision_guard`
  (`configs/policy_search/candidates/scenario_adaptive_hybrid_orca_v2_collision_guard.yaml`).
- **Neutral reference planner:** `scenario_adaptive_orca_v1`
  (`configs/policy_search/candidates/scenario_adaptive_orca_v1.yaml`).
- **Family split (family-disjoint):** fit/tuning/warm-start `classic_cross_trap_medium`
  only (2 eligible records); fresh outcomes `classic_group_crossing_medium` only
  (6 eligible records); no seed or archive-ID overlap; #3275 outcomes never reused.
- **Controls:** doorway rejection-control seeds 128/130; `francis2023_blind_corner`
  certifier negative control (never a candidate or denominator row).
- **Methods:** existing `OptunaCandidateSampler` (TPE) versus existing
  `RandomCandidateSampler`, reused by reference. No new optimizer.
- **Candidate space:** identical bounds to the approved design
  (`configs/adversarial/issue_5303_search_promotion_space_v2.yaml`), including the wide
  scenario-seed interval that contains the frozen warm starts (14821, 14890).
- **Runtime-effective space (new, merged PR #6586):** the space declares the pedestrian
  identity `issue_5303_powered_promotion_candidate` and pairs with the v2 scenario
  template (`configs/adversarial/issue_5303_classic_group_crossing_medium_v2.yaml`) that
  exposes the matching `single_pedestrians` entry. `spawn_time_s` and
  `pedestrian_delay_s` bind to that pedestrian's start delay and waypoint wait rule, so
  they change the effective runtime scenario and its canonical hash (provenance-only
  metadata excluded). The side-effect-free preflight classifies the powered pair
  `promotion_timing_ready` and keeps the historical no-pedestrian pair rejected as
  `blocked_no_pedestrian`; missing, metadata-only, unbound, or inert dimensions fail
  closed.
- **Simulator-time cap:** horizon 100 steps, dt 0.1 s (10.0 s), identical for both arms.
- **Objective:** `constraints_first_lexicographic_v1` — collision/severe intrusion (hard
  veto), then liveness/goal completion, then comfort/efficiency; no weighted comfort or
  SNQI (Social Navigation Quality Index) improvement compensates for a collision or zero
  goal completion.
- **Seven counted-weak-point gates (fail-closed):** certification; deterministic replay;
  target failure in ≥ 4 of 5 fresh confirmation seeds with no retries; same primary
  mechanism in ≥ 4 of 5; neutral reference success in ≥ 4 of 5 of the same seeds;
  shortlist passes the threshold in a second recorded execution context; no excluded row
  class (`fallback`, `degraded`, `unavailable`, `geometry_artifact`, `knife_edge`,
  `stress_only`, `duplicate`). Fixed confirmation shortlist/cap: the 4/5 gates plus the
  second-context gate.

## Estimand, exact inference, and decision rule

- **Primary estimand:** TPE-minus-random difference in unique fully admitted weak points;
  candidate-level clustering across search seeds; independent unit = search seed.
- **Primary denominator:** intention-to-search — all **384 scheduled attempts per
  method**. Missing, invalid, fallback, degraded, unavailable, and duplicate attempts
  remain in that denominator with a recorded reason; no optional seeds, retries, or
  outcome-dependent replacement/exclusion. Complete-case analysis is secondary
  sensitivity only.
- **Uncertainty:** exact cluster-level interval over **all 924 arm-label assignments**
  (12 labeled seed units, choose 6); the 10,000-resample seed-cluster bootstrap remains a
  secondary diagnostic only.
- **Null tests:** shuffled-outcome and ranking seed permutations (unit = search seed),
  two-sided, threshold `p <= 0.05`, both required, each evaluated by the full exact
  enumeration.
- **Minimally important improvement:** one additional unique fully admitted weak point.
- **Decision rule:** exactly one `promote | stop | inconclusive` function. Positive gate
  (NOT weakened): ≥ 2 admitted weak points, a positive TPE-minus-random difference whose
  95% cluster-level interval excludes zero, and both null tests at `p <= 0.05`.

## Outcome-free sensitivity analysis (attainable significance vs power)

- Exact enumeration: `C(12,6) = 924` arm-label assignments.
- Minimum attainable one-sided p = `1/924 ≈ 0.00108`; minimum attainable two-sided p =
  `2/924 ≈ 0.00216 <= 0.05` — the approved two-sided boundary **is representable**
  (attainable significance).
- Rejection-region boundary: an exact two-sided region of at most
  `floor(0.05 × 924) = 46` assignments has `p <= 46/924 ≈ 0.0498`; 47 assignments would
  exceed the threshold.
- **Power/sensitivity:** no outcome-free power claim is made against an unspecified
  alternative. Sensitivity is characterized by the rejection-region boundary above and
  the minimally important effect (one unique fully admitted weak point); the effect sizes
  the fixed design can identify are exactly those whose observed cluster-level statistic
  leaves at most 46 of the 924 assignments at least as extreme.

## Machine-readable result handoff (#6145 terminal schema)

Frozen schema `issue_5303_search_promotion_result.v2`; required fields: `schema_version`,
`decision`, `contract_sha256`, `execution_commit`, `admitted_candidate_count`,
`candidate_manifest_sha256`, `evidence_packet_sha256`. Downstream #6146 activation is
valid **only** when `decision == "promote"`, `admitted_candidate_count >= 5`, every
referenced hash verifies, and every admitted candidate passes the frozen eligibility and
lineage gates. Issue closure alone never activates downstream work.

## Historical boundary (v1 rejection)

The historical v1 contract
(`configs/adversarial/issue_5303_search_promotion_contract.yaml`) and the PR #6291
evidence packet remain immutable three-seed diagnostic artifacts. The powered checker
rejects the v1 contract for promotion-capable execution and proves it still declares
`diagnostic_inconclusive` with thresholds not weakened.

## Stop conditions (frozen)

Stop and do not treat the study as ready if: current `main` is red or input hashes cannot
be reproduced (non-triggering: green base commit recorded above); the timing dimensions
are missing, metadata-only, unbound, or inert (non-triggering: `promotion_timing_ready`
proven); exact six-seed inference cannot support the boundary (non-triggering: 924
assignments, min two-sided p ≈ 0.00216); any outcome was generated or inspected before
freezing (non-triggering: outcome-free); any approved target/family/budget/space/time-cap/
ordering/gate/threshold/exclusion would have to change (non-triggering: unchanged); or any
substantive field remains discretionary after execution starts (non-triggering: the
execution-stage requirements freeze the remaining wiring before the first attempt).

## Reproduce the frozen contract hash (side-effect-free)

```bash
uv run python scripts/tools/check_issue_5303_search_promotion_contract_v2.py
uv run python scripts/tools/check_issue_5303_search_promotion_contract_v2.py --identities
```

The first command recomputes the contract, receipt, archive, and input hashes, asserts the
frozen design, exact inference, sensitivity boundary, result schema, execution binding,
the v1 rejection, and the runtime-effective timing gate; it exits 0 only when the contract
verifies. The second emits exactly 768 scheduled search identities and performs no planner
execution and no outcome read. Neither command imports adversarial execution surfaces
(samplers/search/runtime/qd/warm_start/transfer_matrix/campaign/replay), subprocess, or
network modules; the focused test
`tests/adversarial/test_issue_5303_search_promotion_contract_v2.py` AST-scans the module
and CLI sources to prove this.

## Files

- Frozen powered contract: [issue_5303_search_promotion_contract_v2.yaml](../../../../configs/adversarial/issue_5303_search_promotion_contract_v2.yaml)
- Manifest (frozen hash): [contract_frozen.json](contract_frozen.json)
- Scheduled identities (768): [scheduled_search_identities.json](scheduled_search_identities.json)
- Powered preflight module: `robot_sf/benchmark/issue_5303_search_promotion_preregistration_v2.py` (raw SHA-256 pinned in the contract)
- Powered check CLI: `scripts/tools/check_issue_5303_search_promotion_contract_v2.py` (raw SHA-256 pinned in the contract)
- Powered search space: `configs/adversarial/issue_5303_search_promotion_space_v2.yaml`
- Powered scenario template: `configs/adversarial/issue_5303_classic_group_crossing_medium_v2.yaml`
- Timing gate (merged PR #6586): `robot_sf/benchmark/issue_5303_search_promotion_preflight.py`
- Focused tests: `tests/adversarial/test_issue_5303_search_promotion_contract_v2.py`
- Historical packet (immutable): [../issue_5303_search_promotion_preregistration/README.md](../issue_5303_search_promotion_preregistration/README.md)
