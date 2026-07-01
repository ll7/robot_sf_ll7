# Dissertation Gap Report

**Purpose**: synthesis/planning aid; not new benchmark, paper, dissertation, or safety evidence.

Generated: 2026-06-21  | Schema: dissertation_gap_report.v1

Sources: ledger #2760 (7 rows), register #2762 (5 entries)

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

### prediction_supported [ledger]

- **Tier/Classification**: diagnostic
- **Promotion step or reason**: benchmark promotion requires executed transfer-aware closed-loop comparison rows with calibrated ADE/FDE/miss-rate and collision/progress metrics.
- **Allowed wording / boundary**: The repository now supports forecast-lane infrastructure for typed artifact contracts, observation tiers, probabilistic metric pipelines, deterministic baselines, dataset-level provenance, transfer diagnostics, uncertainty pilots, and a closed-loop coupling gate.
- **Caveat**: Current evidence is infrastructure and diagnostic in nature: it marks what is implemented and what the next coupling step reports, but it is not yet benchmark- or paper-grade claim support.
- **Claim gap / reason**: Requires a transfer-aware, same-seed closed-loop campaign that compares forecast-enabled planners to deterministic and baseline modes using calibrated, fail-closed, and same-denominator metrics.

### prediction_unsupported [ledger]

- **Tier/Classification**: diagnostic
- **Promotion step or reason**: run transfer-focused closed-loop planner campaigns with matched deterministic baselines and explicit safety/progress deltas before promoting this claim.
- **Allowed wording / boundary**: Forecast artifacts and tooling are in place, but forecast-driven gains in safety, progress, or transfer performance are not established.
- **Caveat**: Forecast coupling and transfer diagnostics currently show mixed or insufficient evidence (#2843 revise recommendation; mixed baseline-point results in #2781; transferability matrix is not a safety proof).
- **Claim gap / reason**: Requires transfer-aware transferability campaigns with false-positive accounting, non-regression on success/progress, and statistical uncertainty before any safety/progress claim.

### exported_tables [ledger]
- **Tier/Classification**: diagnostic
- **Promotion step or reason**: SNQI contract repair or scoped claim boundary: fix SNQI rank-alignment failure, or predeclare a narrower boundary that does not use SNQI for Results promotion, then rerun a fresh bounded campaign before any benchmark or Results wording.
- **Allowed wording / boundary**: Payload-complete scenario-horizon dissertation table exports exist for discussion/provenance only; the 2026-07-01 rerun repaired the PPO row but failed the SNQI contract, so it does not establish benchmark-success ranking or Results-chapter evidence.
- **Caveat**: The 2026-07-01 Issue #3203 rerun exited 0 with benchmark_success and evidence_status valid for campaign row execution: 9 successful rows, 0 unexpected failed rows, 0 fallback/degraded rows counted as success, and PPO native with learned-policy contract pass. The readiness checker still returned diagnostic_only because SNQI contract status was fail from rank-alignment Spearman -0.19999999999999998 below the 0.3 fail threshold.
- **Claim gap / reason**: Payload absence and the stale PPO observation-contract failure are resolved for the 2026-07-01 rerun, but the tables remain diagnostic/provenance artifacts until the SNQI contract failure is repaired or explicitly scoped out by a new issue contract.

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

### issue-3201-observation-noise-live-dense-stress [register]

- **Tier/Classification**: diagnostic_only
- **Promotion step or reason**: Completed by follow-up #3233: the deterministic near-field fixture and same-seed clean-vs-perturbed comparison are recorded in docs/context/evidence/issue_3233_near_field_observation_noise/. Do not repeat the weak dense_stress live candidate for this claim boundary.
- **Allowed wording / boundary**: Diagnostic-only, not benchmark or paper evidence. Confirms live perturbation can change planner-input observations; does not show a planner behavior delta.
- **Caveat**: Diagnostic-only, not benchmark or paper evidence. Confirms live perturbation can change planner-input observations; does not show a planner behavior delta.
- **Claim gap / reason**: Perturbation plumbing activated (179 missed actor observations; observed actor count dropped from 17 to 5-12), but selected commands, progress/risk summary, collision flags, and closest-distance metrics were identical. The live scenario candidate did not satisfy the intended near-field pedestrian-dominated condition.

### issue-2760-dissertation-evidence-ledger-diagnostic-rows [register]

- **Tier/Classification**: diagnostic_only
- **Promotion step or reason**: Use the ledger as a planning aid to identify which thesis areas need new experiments before Results wording is possible. Do not cite any current row as benchmark-strength evidence.
- **Allowed wording / boundary**: Synthesis/planning aid only. Does not produce new benchmark evidence, paper-facing results, or safety claims.
- **Caveat**: Synthesis/planning aid only. Does not produce new benchmark evidence, paper-facing results, or safety claims.
- **Claim gap / reason**: Five of seven ledger rows are diagnostic-tier or blocked evidence (topology, signalized behavior, prediction, pedestrian density, exported tables). One area (observation robustness) is release-backed but is a contract/provenance layer only. The exported-tables payload gap is resolved, but the Issue #3203 rerun remains invalid as benchmark-success evidence because PPO partial-failed and SNQI contract status failed. No row supports Results-chapter wording without qualification.

### issue-3213-hardcase-authority-speedcap [register]

- **Tier/Classification**: diagnostic_only
- **Promotion step or reason**: Keep predictive_near_field_speed_cap as a minor safety-progress tuning knob, not a success driver. Stop further planner-authority tuning as a plateau fix; prioritize model-side bets (#3214 retraining, richer hard-case data) and proxy-vs-ADE selection (#3204).
- **Allowed wording / boundary**: Diagnostic-only, not benchmark or paper evidence. Small attributable speed-cap effect; classification diagnostic_only, not promote.
- **Caveat**: Diagnostic-only, not benchmark or paper evidence. Small attributable speed-cap effect; classification diagnostic_only, not promote.
- **Claim gap / reason**: Only the near-field speed cap moved hard-success (~0.07 to ~0.10, about +0.03 absolute / +33% relative, consistent across checkpoints, ~120 episodes/cell). nf_headings_only, nf_horizonboost_only, high_angular, dense_lattice, deep_sequence, and combined_max_authority were inert (about baseline). The lift does not close the plateau (hard-success ~0.10) and checkpoint choice barely moved baseline, indicating the binding constraint is model/data-side rather than planner authority.

## Remove / Weaken

No gaps in this bucket.

## Claim Boundaries

- This gap report is a synthesis/planning aid. It does not produce new benchmark evidence, paper-facing results, or safety claims.
- All allowed_wording and caveat fields are copied verbatim from source rows; no new wording is introduced.
- A non-null promotion_step_or_reason does not upgrade a row to stronger evidence. Promotion requires completing the path and reclassifying the evidence tier.
- Fallback behavior is not acceptable as a successful benchmark outcome unless the task explicitly measures fallback mode.
- Every gap classification preserves the source evidence tier and classification without upgrade.
