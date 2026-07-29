# Issue #5303 step 2 — frozen TPE-vs-random search-promotion preregistration

## Plain-language summary

This is a **preregistration**: before any new planner run happens, it freezes the exact
rules of one experiment that asks a single, narrow question — *does the existing
Optuna/Tree-structured Parzen Estimator (TPE) adversarial search find more fully
certified, independently confirmed robot-failure scenarios than the existing random
search, for one specific planner on one held-out scenario family, under a matched
compute budget?* It also records, honestly and before any outcome, that the frozen
three-seed budget **cannot robustly test** the proposed "promote TPE" threshold, so the
future run is pre-declared **diagnostic/inconclusive** rather than having its thresholds
quietly relaxed.

The evidence-grade promotion campaign is therefore **stopped before Step 3**. A separate,
fully specified adapter-mode search-stage diagnostic is allowed only to prove the runner,
provenance, and attempt-accounting handoff. It has a fixed `inconclusive` decision and
cannot become a TPE-promotion result.

Nothing here runs a planner, launches a search, replays a scenario, submits a cluster
job, or reads any evaluation outcome. It only freezes the design and reproduces its hash.

## Evidence boundary

`diagnostic` / `proposal_preflight_only`. Completing this step proves the frozen
**diagnostic handoff** is executable and falsifiable: its command, inputs, complete row
schema, and fixed inconclusive stop rule are checked before execution. The evidence-grade
promotion study remains stopped because its positive gate is untestable with three seeds.
This is not evidence that TPE outperforms random or that failures transfer across
planners. No paper, dissertation, benchmark-wide, cross-planner, minimax, or portfolio
claim is made or implied.

Parent: https://github.com/ll7/robot_sf_ll7/issues/5303 (step 2 of 6).
The entry gate is satisfied by the merged recertification in
https://github.com/ll7/robot_sf_ll7/issues/6139; this historical packet is not a campaign
authorization.

## Supersession boundary

The parent issue's 2026-07-28 domain ruling approved a new outcome-free design with
**six independent search seeds per method** and explicitly superseded this three-seed
design for any promotion-capable study. This frozen packet remains useful only as a
historical, executable diagnostic/preflight handoff. It is **not** the approved six-seed
preregistration and cannot authorize the #6145 campaign, a `promote` decision, or
downstream transfer work.

## The claim under test (frozen, falsifiable)

> Under a frozen, family-disjoint design and matched candidate (64 per search seed per
> method) and simulator-time budgets, the existing Optuna/TPE search yields more fully
> certified and independently confirmed weak points than the existing random search for
> `scenario_adaptive_hybrid_orca_v2_collision_guard` on the held-out
> `classic_group_crossing_medium` family, using exactly three search seeds per method
> and a constraints-first objective ordering.

## Entry gate (satisfied by merged #6139)

The corrected continuous swept-envelope / runtime simulator-collision recertification
(<https://github.com/ll7/robot_sf_ll7/issues/6139>) re-certified all 17 registered
records and left **8 eligible** (≥ the required floor of 2). At every side-effect-free
preflight, the receipt and the actual certified-archive file are each hashed and must
match the frozen entry-gate digest:

| Artifact | Path | SHA-256 |
| --- | --- | --- |
| Recertification receipt | [recertification_issue_6139.json](../issue_5305_certified_archive/recertification_issue_6139.json) | `0d643f2c36d0f1f11e2be2351359567215d47ed216d156018fc6909a79a42cfe` (raw file) |
| Receipt self-declared hash | (inside the receipt) | `7bade1d5008d66eba9dfe6c9c72e72e5e6720e7acb96f17a90a61a4f0798f3d0` |
| Certified archive | `docs/context/evidence/issue_5305_certified_archive/archive.json` | `79e022587b35c1c42bc07cfefaf882af473e96841a99ef57f98a4cee26636445` |

## Frozen design

- **Target planner:** `scenario_adaptive_hybrid_orca_v2_collision_guard`
  (`configs/policy_search/candidates/scenario_adaptive_hybrid_orca_v2_collision_guard.yaml`).
- **Neutral reference planner:** `scenario_adaptive_orca_v1`
  (`configs/policy_search/candidates/scenario_adaptive_orca_v1.yaml`) — a pure ORCA
  planner that must succeed where the target fails, proving a discovered weak point is
  target-specific rather than a scenario-wide impossibility.
- **Family split (family-disjoint):**
  - fit/tuning/warm-start family: `classic_cross_trap_medium` only (2 eligible records);
  - fresh outcome family: `classic_group_crossing_medium` only (6 eligible records);
  - train/evaluation family, scenario-seed, and archive-ID overlap are all zero.
- **Controls (never discoveries or denominator rows):**
  - doorway rejection-control seeds 128/130 — must not yield counted weak points;
  - `francis2023_blind_corner` — certifier negative control only.
- **Methods:** existing `OptunaCandidateSampler` (TPE) versus existing
  `RandomCandidateSampler`, reused by reference only. No new optimizer.
- **Budget:** exactly **64** candidates per search seed per method, exactly **3** search
  seeds per method (`530301, 530302, 530303`), simulator-time cap `horizon=100, dt=0.1`
  (10.0 s), identical candidate space, feasibility rules, and execution mode for both.
- **Candidate space:** the issue-specific
  `configs/adversarial/issue_5303_search_promotion_space.yaml` keeps the canonical
  crossing/TTC geometry and includes the two frozen fit-family warm-start seeds (`14821`,
  `14890`) in its declared scenario-seed interval.
- **Objective:** constraints-first lexicographic ordering — (1) collision or severe
  intrusion [hard veto], (2) liveness/goal completion, (3) comfort/efficiency. No
  weighted comfort or SNQI (Social Navigation Quality Index) improvement may compensate
  for a collision or zero goal completion. Scalar `worst_case_snqi` is retained as a
  named diagnostic only.

### The seven counted-weak-point gates (fail-closed)

A unique candidate counts only when **all** pass:

1. corrected swept-envelope + runtime simulator-collision certification;
2. deterministic replay (exact signature agreement);
3. target failure in ≥ 4 of 5 fresh confirmation seeds with **no retries**;
4. the same primary failure mechanism in ≥ 4 of 5 seeds;
5. the neutral reference planner succeeds in ≥ 4 of 5 of the same seeds;
6. the shortlist passes the same threshold in a **second recorded execution context**;
7. no `fallback`/`degraded`/`unavailable`/`geometry_artifact`/`knife_edge`/`stress_only`/`duplicate` row.

Candidate/config normalization uses a canonical sorted-key JSON hash. Duplicate handling
is **global within each method arm across all three search seeds** for the secondary
unique-candidate endpoint. A duplicate attempt is never silently dropped: every scheduled
attempt stays in the primary denominator, while duplicates cannot inflate a future unique
fully-admitted weak-point numerator.

## Estimand, uncertainty, and decision rule

- **Primary estimand:** TPE-minus-random difference in the number/rate of unique, fully
  admitted weak points, with **candidate-level clustering across search seeds** (the
  independent unit is the **search seed**, not the candidate).
- **Uncertainty:** non-parametric bootstrap (10 000 resamples) over the three seed
  clusters per method; 95% percentile interval.
- **Null tests:** shuffled-outcome and ranking seed permutations (unit = search seed),
  two-sided, threshold `p ≤ 0.05`, both required.
- **Primary denominator:** intention-to-search — all **192 scheduled attempts per method**.
  Missing, invalid, fallback, degraded, unavailable, and duplicate attempts remain in that
  denominator with a recorded reason; no optional seeds, retries, or outcome-dependent
  replacement/exclusion are allowed. A complete-case calculation is secondary sensitivity
  analysis only, never the primary estimand.
- **Minimally important improvement:** one additional unique fully admitted weak point.
- **Decision rule:** `promote | stop | inconclusive`. The proposed positive gate is
  ≥ 2 admitted weak points, a positive TPE-minus-random difference whose 95% interval
  excludes zero, **and** both null tests at `p ≤ 0.05`. These thresholds are kept frozen.

## Power analysis — why the run is pre-declared diagnostic/inconclusive

Because candidates cluster within a seed, each null test permutes **six labeled seeds**
(three per method). There are `C(6,3) = 20` label arrangements, so:

- minimum **two-sided** permutation p-value = `2/20 = 0.10 > 0.05` — the two-sided null
  **cannot** reject at `p ≤ 0.05`, no matter the outcome;
- minimum **one-sided** p-value = `1/20 = 0.05`, only at the single most-extreme
  arrangement (knife-edge, and inconsistent with the two-sided interval);
- the 95% bootstrap interval over three seed clusters is not robust.

The positive gate therefore **cannot be robustly tested** under the frozen three-seed
budget. The evidence-grade Step 3 promotion campaign is therefore **stopped**. No threshold
is weakened: the gate stays at ≥ 2 weak points, a 95% interval excluding zero, and both
null tests at `p ≤ 0.05`. Claiming `promote` later requires **re-preregistration** with
more search seeds.

The separately justified command below is not that stopped promotion campaign. It is an
adapter-mode, search-stage accounting diagnostic with a predeclared `inconclusive` decision.
It records every scheduled attempt but intentionally does not collect deterministic replay,
five-seed target/reference confirmation, or a second execution context; it cannot admit a
weak point or authorize a promotion/transfer claim.

## Stop conditions (frozen)

Stop the evidence-grade promotion sequence before Step 3 if: fewer than two eligible
candidates (already verified non-triggering: 8 eligible); any evaluation outcome was
inspected for this contract before freezing; fit/evaluation lineage cannot be made
disjoint; the target/reference planner or primary mechanism cannot be pinned; the budget
cannot test the positive gate (it cannot); a row can enter without complete
candidate/seed/execution/certification/mechanism lineage; or any threshold or metric
remains discretionary after execution starts.

The separate diagnostic command may run only with the frozen command and hashes below.
Its stop rule is coherent and fail-closed: every missing, invalid, fallback, degraded,
unavailable, or provenance-mismatched attempt remains recorded in the 192-attempt arm
denominator and yields `inconclusive`; it is never retried, replaced, or deleted.

## Reproduce the frozen contract hash (side-effect-free)

```bash
uv run python scripts/tools/check_issue_5303_search_promotion_contract.py
```

This recomputes the contract and #6139 receipt hashes, checks raw hashes for the target,
reference, family, search-space, runner, analysis, preflight, and checker inputs, and checks the
certified archive. It statically verifies the registered objective/runner options/complete outcome
schema, recomputes the permutation power math, and asserts the diagnostic-only stop rule. It imports no adversarial execution
surface (`samplers`, `search`, `runtime`, `qd`, `warm_start`, `transfer_matrix`, or any
campaign/replay/benchmark-runner module), no `subprocess`, and no network module; the focused test
`tests/adversarial/test_issue_5303_search_promotion_preflight.py` AST-scans the preflight
source to prove this.

## Separately justified diagnostic execution (not the stopped promotion campaign)

The authoritative complete adapter-mode command is
`step3_execution.diagnostic_search_command` in the
[frozen contract](../../../../configs/adversarial/issue_5303_search_promotion_contract.yaml).
It binds the target algorithm config, records the neutral reference config for later
confirmation, pins the held-out `classic_group_crossing_medium` template/search space,
requires the corrected certifier, reruns the side-effect-free preflight before any search
attempt, and writes all declared files. The paired
`step3_execution.analysis_command` validates the complete row matrix and writes the
fixed analysis result. Do not run either command as a substitute for the stopped
evidence-grade campaign.

The contract's `expected_artifacts` mapping names the worktree-local post-run files.
They are not durable evidence pointers. The analysis retains the matched 192-attempt
denominator per method and returns **`inconclusive`**; it does not run or replace replay,
target/reference confirmation, or second-context confirmation. Those absent gates are
explicitly recorded as not admitted.

Before accepting a diagnostic result, the analyzer reruns the frozen preflight and requires every
self-hashed row to match the frozen scenario family and path/hash, search-space path/hash,
target/reference-config path/hash, objective, and adapter execution mode. A row's own hash alone
cannot make a different input packet appear complete.

## Files

- Frozen contract: [issue_5303_search_promotion_contract.yaml](../../../../configs/adversarial/issue_5303_search_promotion_contract.yaml)
- Manifest (frozen hash): [contract_frozen.json](contract_frozen.json)
- Preflight module: `robot_sf/benchmark/issue_5303_search_promotion_preflight.py` (raw SHA-256 pinned in the contract)
- Check command: `scripts/tools/check_issue_5303_search_promotion_contract.py` (raw SHA-256 pinned in the contract)
- Diagnostic analysis: `robot_sf/benchmark/issue_5303_search_promotion_analysis.py`
- Diagnostic analysis CLI: `scripts/tools/analyze_issue_5303_search_promotion.py`
- Focused tests: `tests/adversarial/test_issue_5303_search_promotion_preflight.py`
