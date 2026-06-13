# Dissertation Gap Report

**Purpose**: synthesis/planning aid; not new benchmark, paper, dissertation, or safety evidence.

Generated: 2026-06-13  | Schema: dissertation_gap_report.v1

Sources: ledger #2760 (6 rows), register #2762 (3 entries)

## Supported (release-backed, current)

### observation_robustness [ledger]

- **Tier/Classification**: release-backed
- **Promotion step or reason**: None (limitation remains)
- **Allowed wording / boundary**: The benchmark observation-level vocabulary defines observation-contract boundaries and compatibility gates so results can state which observation assumption a planner used.
- **Caveat**: The levels are benchmark evidence labels and compatibility gates, not sim-to-real validity claims. No real camera perception, detector training, calibrated tracking, or new environment observation implementation exists yet.
- **Claim gap / reason**: Requires actual perception pipeline, calibrated tracking, or simulator-level observation fidelity before any robustness claim. Cross-track comparison is not valid without matched observation contracts.

## Blocked (promotion path exists but not yet completed)

### topology_guidance [ledger]

- **Tier/Classification**: diagnostic
- **Promotion step or reason**: live replay: a fresh topology-guided planner campaign with non-same-family selector variants and an independent seed set is required before any benchmark promotion.
- **Allowed wording / boundary**: Near-parity topology selection can diversify route labels, but current evidence still routes the mechanism to revise without improving non-primary command influence, route progress, or terminal outcome.
- **Caveat**: Do not claim topology guidance improves success, transfer, or leaderboard performance. The lane is stop for same-family selector reruns on the canonical slice.
- **Claim gap / reason**: A new hypothesis with a different mechanism and metric is needed before further topology work. No current row supports Results wording.

### signalized_behavior [ledger]

- **Tier/Classification**: diagnostic
- **Promotion step or reason**: live replay: explicit runtime signal phase state, planner-observation policy, and planner_observable promotion are required before any benchmark row.
- **Allowed wording / boundary**: The repository has a proxy signal-state diagnostic surface that records phase labels in trace metadata, but the current evidence is not planner-consumed, not benchmark-evidence, and does not prove traffic-signal realism or crossing-legality compliance.
- **Caveat**: Signal state is proxy_diagnostic only. Do not claim planner observability, forced-waiting reasoning, legality compliance, or benchmark ranking improvement.
- **Claim gap / reason**: Requires explicit runtime signal phase state, planner-observation policy, zone/legality trace fields, and planner_observable promotion before any benchmark row.

### prediction [ledger]

- **Tier/Classification**: diagnostic
- **Promotion step or reason**: denominator repair: an executed planner campaign with matched metrics and fail-closed row handling is required before any benchmark claim.
- **Allowed wording / boundary**: The repository has a merged probabilistic prediction interface and a contract-smoke runner that materializes native and fail-closed row shapes for multimodal prediction configurations.
- **Caveat**: This is contract evidence only. The runner uses deterministic fixtures, does not execute a planner campaign, and does not measure prediction quality or compare planning performance.
- **Claim gap / reason**: Requires executed planner campaign with reactive, single-trajectory, and multimodal rows; matched metrics (success, collision, min-ped-distance, time-to-goal); and fail-closed row handling before any benchmark claim.

## Negative / Revise-Only

### pedestrian_density_stress [ledger]

- **Tier/Classification**: diagnostic
- **Promotion step or reason**: None (limitation remains)
- **Allowed wording / boundary**: A config-only scenario coverage entropy report can identify redundant and novel scenario candidates based on authored metadata features for scenario-set curation.
- **Caveat**: Coverage entropy is a diagnostic planning aid. It does not prove benchmark value, runtime stress effectiveness, or planner ranking. Dense stress rows should not be promoted without runtime and failure-semantics proof.
- **Claim gap / reason**: Requires runtime execution evidence, failure-semantics classification, and planner-comparison rows before any pedestrian-density stress benchmark claim.

### issue-2716-topology-reselection-cross-slice [register]

- **Tier/Classification**: revise
- **Promotion step or reason**: Design a successor targeting actual clearance or terminal-outcome movement. Stop same-family selector reruns on the canonical slice.
- **Allowed wording / boundary**: Diagnostic-only, not benchmark or paper evidence. Classification is revise, not promote.
- **Caveat**: Diagnostic-only, not benchmark or paper evidence. Classification is revise, not promote.
- **Claim gap / reason**: All 9 hard progress-gated rows ended horizon_exhausted (159 deadlock steps, 0 collision). The mechanism activates and generalizes diagnostically but does not clear any hard slice at h160. Negative-control rows succeeded cleanly (3/3 success, 0 topology switches).

### issue-2749-observation-noise-distant-pedestrian [register]

- **Tier/Classification**: diagnostic_only
- **Promotion step or reason**: Retest with a closer pedestrian, higher pedestrian count, or a scenario where the planner must actively avoid pedestrians.
- **Allowed wording / boundary**: Diagnostic-only, not benchmark or paper evidence. Confirms perturbation infrastructure works; does not measure behavioral degradation.
- **Caveat**: Diagnostic-only, not benchmark or paper evidence. Confirms perturbation infrastructure works; does not measure behavioral degradation.
- **Claim gap / reason**: All progress, risk, and planner metrics are identical between baseline and perception-limited. The single pedestrian is too far from the robot to influence planner decisions; the planner is dominated by static obstacle clearance. Perturbation plumbing is confirmed working (occlusion masking active, 1 missed detection at step 15) but no behavioral degradation is measurable.

### issue-2760-dissertation-evidence-ledger-diagnostic-rows [register]

- **Tier/Classification**: diagnostic_only
- **Promotion step or reason**: Use the ledger as a planning aid to identify which thesis areas need new experiments before Results wording is possible. Do not cite any current row as benchmark-strength evidence.
- **Allowed wording / boundary**: Synthesis/planning aid only. Does not produce new benchmark evidence, paper-facing results, or safety claims.
- **Caveat**: Synthesis/planning aid only. Does not produce new benchmark evidence, paper-facing results, or safety claims.
- **Claim gap / reason**: Four of six areas are classified as diagnostic-tier evidence (topology, signalized behavior, prediction, pedestrian density). One area (observation robustness) is release-backed but is a contract/provenance layer only. One area (exported tables) is stale/non-claimable. No row supports Results-chapter wording without qualification.

## Remove / Weaken

### exported_tables [ledger]

- **Tier/Classification**: non-claimable
- **Promotion step or reason**: stale artifact refresh: payload file recovery or re-export from a fresh campaign is required before any reuse.
- **Allowed wording / boundary**: do-not-use
- **Caveat**: The claim matrix reports missing payload files for both artifacts, and the stale-artifact detector classifies the bundle manifest as stale. The tables remain historical tracked evidence and do not establish new benchmark claims.
- **Claim gap / reason**: Requires payload file recovery or re-export before any reuse.

## Claim Boundaries

- This gap report is a synthesis/planning aid. It does not produce new benchmark evidence, paper-facing results, or safety claims.
- All allowed_wording and caveat fields are copied verbatim from source rows; no new wording is introduced.
- A non-null promotion_step_or_reason does not upgrade a row to stronger evidence. Promotion requires completing the path and reclassifying the evidence tier.
- Fallback behavior is not acceptable as a successful benchmark outcome unless the task explicitly measures fallback mode.
- Every gap classification preserves the source evidence tier and classification without upgrade.
