# Issue #3275 same-planner held-out contract (sub-issue #6103, step 1 of 3)

Status: **Frozen contract, posted for human research-contract review. Not self-closed.**
Evidence boundary: this note freezes a **preregistered execution path** and a
**failure-neighborhood ranker**. It is not a calibrated failure-probability model,
and producing it generated **no new empirical outcome** (no planner execution, no
adversarial campaign, no Slurm, no training, no replay/confirmation of new
candidates). It is not proposal-yield, benchmark, paper, or planner-performance
evidence.

Parent: #3275. Thesis: #2921. Prerequisite: #6139 (closed-completed; corrected
recertification merged on `origin/main`). Sibling draft: #6135.

## Plain-language summary

The earlier #3275 plumbing ranked candidates by distance to a failure archive and
then "evaluated" them with the same distance to the same archive — circular. This
contract repairs the result-integrity gaps and freezes one executable analysis
contract before any bounded campaign:

- the ranker is fit on the `social_force` failures of `classic_group_crossing_medium`;
- it is evaluated on the **held-out** `classic_cross_trap_medium` family with the
  **same `social_force` planner in both arms**;
- the five `classic_cross_trap_medium` / `goal` entries in the registered archive
  are excluded from every fit-side decision (wrong planner + held-out family);
- the comparison and the #2921 stop rule follow **independent native
  planner-execution outcomes only**; archive-nearness is diagnostic-only;
- the decision vocabulary is exactly `continue | stop | inconclusive`.

## Frozen inputs (derive from the corrected recertification, not the #6139 comment)

- Pre-correction archive: `docs/context/evidence/issue_5305_certified_archive/archive.json`
  (SHA-256 `79e022587b35c1c42bc07cfefaf882af473e96841a99ef57f98a4cee26636445`,
  pre-correction lineage only).
- Corrected recertification:
  `docs/context/evidence/issue_5305_certified_archive/recertification_issue_6139.json`
  (SHA-256 `7bade1d5008d66eba9dfe6c9c72e72e5e6720e7acb96f17a90a61a4f0798f3d0`).
  Note: the #6139 closing comment quotes a different SHA-256
  (`1406ea54...`); the file is the authority here, not the comment.
- Recertification left all 17 records **unchanged**, but its corrected
  eligibility decisions are authoritative. The six group-crossing `stress_only`
  / `knife_edge` records are removed from the nominal fit set and are not
  replaced.

Fit set: the six corrected `eligible` `classic_group_crossing_medium` /
`social_force` records (IDs and
`entry_ids_sha256 = 41fa8863e2345a9cdc665ede7ac8e0110d93da042b4dd03ee2cc579c72f12e25`
live in `configs/adversarial/issue_3275_same_planner_contract.json`). The six
removed `stress_only` IDs and their hash are frozen in the same config, so the
eligibility exclusion cannot silently drift.

Excluded sets: the six group-crossing `stress_only` records (not nominally
eligible) and the five `classic_cross_trap_medium` / `goal` records (wrong target
planner and held-out family).

Target planner: `social_force` (config SHA-256
`dfdebd497e19a046e41cb2b1e7d7a7f54cd592ac0a465e4149efff19efa16735`), in **both**
arms. Held-out evaluation family: `classic_cross_trap_medium`.

## Train-only ranking (issue #6103 gap 1)

`FailureArchiveProposalModel` is constructed from a fit-only payload of exactly
the six nominally eligible frozen IDs (`fit_entry_ids`). The constructor drops
every entry whose `archive_id` is not in the frozen set, so `stress_only`,
excluded, and held-out-family records cannot influence scores or rank order even
if the full archive is supplied. The negative-regression guarantee (tested):
feeding the full 17-record archive yields the same fit entries and identical
candidate scores/ranks as the six-record fit-only payload, and all eleven
non-fit records (six `stress_only` plus five held-out) are reported as dropped.

## Independent outcomes are authoritative (issue #6103 gap 2)

The `adversarial_independent_outcomes.v2` row-level contract is authoritative.
When valid v2 rows are available, the top-level proposal/random metrics, the
comparison, and the #2921 stop rule are computed exclusively from those rows.
Archive-nearness is recorded under an explicitly diagnostic namespace
(`archive_nearness_diagnostic_only_cannot_drive_verdict`) and can never drive a
verdict. Both opposite-sign regressions are required tests:

1. archive-nearness favors proposal while execution favors random -> decision
   **stop** (follows execution);
2. archive-nearness favors random while execution favors proposal -> decision
   follows execution.

## Row-level outcome lineage (issue #6103 gap 3)

One row per candidate x execution seed binds: candidate/manifest ID + manifest
SHA, selection arm + rank, candidate-pool seed/index, target planner ID + config
SHA, scenario family + seed, execution commit + command/config lineage +
native/fallback/degraded status, termination reason + independent failure
outcome, scenario and candidate certification status, replay/confirmation
lineage + record hash, and exclusion reason when inadmissible. Each admitted
manifest SHA must match a separate, frozen ID-to-hash binding from the arm
manifests; an outcome packet cannot self-attest it. A candidate manifest ID may
appear in one arm only. Aggregate arrays derive from admitted rows only.
Missing, malformed, mismatched, fallback, degraded, cross-arm-overlapping, or
lineage-incomplete rows fail closed (block the evaluation).

## Estimand, power/sensitivity, and decision rule (issue #6103 gaps 4–5)

Estimand: proposal-minus-random difference in **candidate-level certified failure
yield** under identical candidate budget. A candidate counts as a failure only
after predeclared deterministic replay (exact signature match), independent-seed
confirmation (at least **3 of 5** predeclared seeds; **4 of 5 is not required**),
and stable attribution.

Power/sensitivity (Fisher exact two-sided, alpha = 0.05): the boundary minimum
detectable yield difference is ~0.50 at k = 10/arm, ~0.417 at k = 12/arm, and
~0.25 at k = 20/arm. After excluding the six `stress_only` fit anchors, the
candidate-level calculation was rechecked at the unchanged k = 12/arm budget:
the ~0.417 boundary remains above the minimally important absolute yield
improvement of 0.20. The study is **underpowered**. Every future underpowered
result is **diagnostic/inconclusive**, whether it favors proposal or random;
only a maintainer-authorized, powered budget can permit a continue/stop decision.

Arm-overlap policy: **disjoint-by-candidate** (one deterministic predeclared
policy). The model rank is converted to the shared pool's stable candidate IDs
before assignment; the rank list must be a unique full permutation of that pool.
The proposal arm takes the top-k IDs; the random arm takes k from the remaining
pool with those exact IDs removed; a candidate is never in both arms.

Decision rule vocabulary is exactly `continue | stop | inconclusive`:

- `continue`: independent outcomes valid AND
  (proposal_yield - random_yield) >= minimally important AND null rejected AND
  powered;
- `stop`: independent outcomes valid AND powered AND
  (proposal_yield - random_yield) <= 0;
- `inconclusive`: outcomes unavailable/fail-closed, underpowered, or null not
  rejected. Underpowered comes first, so it is never `continue` or `stop`.

There is no `revise` and no generic `blocked` in this contract.

## Cross-family feature semantics (issue #6103 gap 6)

The frozen feature view is **family-invariant robot-path-relative**
(`robot_sf.adversarial.disjoint_evaluation.family_invariant_features`): each
pedestrian candidate is projected onto the robot's start-to-goal path, giving
lateral/longitudinal spawn and goal features (normalized by path length) plus the
three inherently family-invariant scalars (pedestrian speed, delay, spawn time).
The per-feature semantic argument is recorded in the contract config: lateral and
longitudinal features have identical operational meaning in
`classic_group_crossing_medium` and `classic_cross_trap_medium` ("how far off the
robot corridor the pedestrian appears, and where along the route"). The transform
is deterministic geometry, frozen against outcomes, and does not use the excluded
cross-trap/goal failures for tuning.

## Side-effect-free contract check (required input to the next sub-issue)

```bash
uv run python scripts/adversarial/run_proposal_vs_random_issue_2921.py \
    --check-contract configs/adversarial/issue_3275_same_planner_contract.json
```

The check derives the fit-only payload from the corrected recertification
artifact, asserts the frozen fit count/hash/planner/family/exclusions, constructs
the fit-only model, runs the negative regression, and exits 0 only when all
checks pass. It executes no planner and writes nothing.

## Human research-contract review gate (open)

Issue #6103's acceptance criterion "Review explicitly covers experimental
validity and result interpretation" is a **human gate** the autonomous worker
cannot satisfy. This contract is frozen and posted for human research-contract
review; the implementing PR is left in draft. The worker does not self-close
#6103 and does not mark the PR ready-for-review/merge.

## Risks and residual conditions

- The final nominal fit set contains six anchors. Its IDs, hash, and the six
  rejected `stress_only` IDs/hash derive directly from the corrected
  recertification artifact; no replacement occurred.
- The study is underpowered at the frozen budget; no future `continue` **or
  `stop`** claim is valid without a maintainer amendment authorizing a larger,
  powered budget.
- Changing the primary planner, held-out family, estimand, effect margin, or
  decision rule requires an amendment to parent #3275 before any campaign.
