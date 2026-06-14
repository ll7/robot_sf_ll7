# Issue #2766 Mixed-Scenario Coverage Matrix

Generated at: 2026-06-14T12:01:06.607884+00:00
Status: synthesis draft only; not benchmark, paper, or safety evidence

This matrix cross-references evidence modules against canonical scenario
slices. Every blocked / unavailable / missing-denominator / stale /
diagnostic-only / proxy / fallback cell carries an explicit reason.

## Matrix

### corridor_interaction

| Module | Status | Reason |
|---|---|---|
| Topology Reselection | diagnostic_only | No topology reselection evidence on corridor slice; topology guidance diversifies labels but does not improve route progress or terminal outcome on canonical slices. |
| Signal Compliance | proxy_only | Signal state is trace_metadata_only / planner_observable=false outside signalized_crossing fixture. Proxy diagnostic only. |
| Prediction Baseline | diagnostic_only | CV forecast evaluated on bounded trace fixture. Diagnostic-only; no planner-campaign comparison. |
| Observation Perturbation | unavailable | No observation-perturbation evidence for this scenario slice. |
| Denominator Health | not_applicable | Denominator metric not defined for this scenario slice. |
| Stale/Current | stale | Exported tables (issue #1023) are stale/non-claimable: missing payload files. Historical tracked evidence only. |
| Claim Eligibility | not_eligible | Blockers: signal: proxy_only; observation: unavailable; stale_artifacts_present. |

### bottleneck

| Module | Status | Reason |
|---|---|---|
| Topology Reselection | diagnostic_only | Topology hard slice for bottleneck: horizon_exhausted on all progress-gated rows; no clearance achieved. Stop lane for same-family selector reruns. |
| Signal Compliance | proxy_only | Signal state is trace_metadata_only / planner_observable=false outside signalized_crossing fixture. Proxy diagnostic only. |
| Prediction Baseline | limited_no_motion | All pedestrian velocities zero; constant-velocity forecast produces degenerate predictions. |
| Observation Perturbation | unavailable | No observation-perturbation evidence for this scenario slice. |
| Denominator Health | not_applicable | Denominator metric not defined for this scenario slice. |
| Stale/Current | current | Source artifacts are current for this slice. |
| Claim Eligibility | not_eligible | Blockers: signal: proxy_only; observation: unavailable. |

### crossing

| Module | Status | Reason |
|---|---|---|
| Topology Reselection | diagnostic_only | Ledger area topology_guidance: artifact_status=current, evidence_tier=diagnostic. Near-parity topology selection can diversify route labels but does not improve outcome. |
| Signal Compliance | proxy_only | Signal state is trace_metadata_only / planner_observable=false outside signalized_crossing fixture. Proxy diagnostic only. |
| Prediction Baseline | diagnostic_only | Merged prediction interface exists; contract-smoke rows materialized but no executed planner campaign. Denominator repair required. |
| Observation Perturbation | unavailable | No observation-perturbation evidence for this scenario slice. |
| Denominator Health | not_applicable | Denominator metric not defined for this scenario slice. |
| Stale/Current | stale | Exported tables (issue #1023) are stale/non-claimable: missing payload files. Historical tracked evidence only. |
| Claim Eligibility | not_eligible | Blockers: signal: proxy_only; observation: unavailable; stale_artifacts_present. |

### signalized_crossing

| Module | Status | Reason |
|---|---|---|
| Topology Reselection | diagnostic_only | Ledger area topology_guidance: artifact_status=current, evidence_tier=diagnostic. Near-parity topology selection can diversify route labels but does not improve outcome. |
| Signal Compliance | available | Runtime denominator plumbing active: 2 observable row(s) with denominator=2. Does not prove traffic-signal realism or legality compliance. |
| Prediction Baseline | diagnostic_only | Merged prediction interface exists; contract-smoke rows materialized but no executed planner campaign. Denominator repair required. |
| Observation Perturbation | unavailable | No observation-perturbation evidence for this scenario slice. |
| Denominator Health | partial | 2 eligible row(s) denominator>0, 2 excluded (denominator=0). Planner-observable compliance evidence limited. |
| Stale/Current | current | Source artifacts are current for this slice. |
| Claim Eligibility | not_eligible | Blockers: observation: unavailable. |

### occluded_emergence

| Module | Status | Reason |
|---|---|---|
| Topology Reselection | diagnostic_only | Ledger area topology_guidance: artifact_status=current, evidence_tier=diagnostic. Near-parity topology selection can diversify route labels but does not improve outcome. |
| Signal Compliance | proxy_only | Signal state is trace_metadata_only / planner_observable=false outside signalized_crossing fixture. Proxy diagnostic only. |
| Prediction Baseline | diagnostic_only | Merged prediction interface exists; contract-smoke rows materialized but no executed planner campaign. Denominator repair required. |
| Observation Perturbation | partial_robustness | 1 condition(s) robustness_evidence (delay_only), 3 diagnostic_only, 3 scenario_too_weak across 7 conditions. Single-fixture, not paper-facing. |
| Denominator Health | partial | 4/7 observation conditions produce observable steps; remaining zeroed by occlusion/missed detection. |
| Stale/Current | current | Source artifacts are current for this slice. |
| Claim Eligibility | not_eligible | Blockers: signal: proxy_only. |

### dense_pedestrian

| Module | Status | Reason |
|---|---|---|
| Topology Reselection | diagnostic_only | Ledger area topology_guidance: artifact_status=current, evidence_tier=diagnostic. Near-parity topology selection can diversify route labels but does not improve outcome. |
| Signal Compliance | proxy_only | Signal state is trace_metadata_only / planner_observable=false outside signalized_crossing fixture. Proxy diagnostic only. |
| Prediction Baseline | diagnostic_only | Merged prediction interface exists; contract-smoke rows materialized but no executed planner campaign. Denominator repair required. |
| Observation Perturbation | diagnostic_only | Issue #2765: trace-derived dense-pedestrian stress fixture; 8/10 conditions expose forecast ambiguity. Stored trace action proxies, not live replay. |
| Denominator Health | not_applicable | Denominator metric not defined for this scenario slice. |
| Stale/Current | current | Source artifacts are current for this slice. |
| Claim Eligibility | not_eligible | Blockers: signal: proxy_only. |

### t_intersection

| Module | Status | Reason |
|---|---|---|
| Topology Reselection | diagnostic_only | Topology hard slice for t_intersection: horizon_exhausted on all progress-gated rows; no clearance achieved. Stop lane for same-family selector reruns. |
| Signal Compliance | proxy_only | Signal state is trace_metadata_only / planner_observable=false outside signalized_crossing fixture. Proxy diagnostic only. |
| Prediction Baseline | diagnostic_only | Merged prediction interface exists; contract-smoke rows materialized but no executed planner campaign. Denominator repair required. |
| Observation Perturbation | unavailable | No observation-perturbation evidence for this scenario slice. |
| Denominator Health | not_applicable | Denominator metric not defined for this scenario slice. |
| Stale/Current | current_negative | Negative-result candidates (issue #2788, NR-001) exist as not_promoted scenario candidates. Not benchmark evidence. |
| Claim Eligibility | not_eligible | Blockers: signal: proxy_only; observation: unavailable. |

### doorway

| Module | Status | Reason |
|---|---|---|
| Topology Reselection | diagnostic_only | Topology hard slice for doorway: horizon_exhausted on all progress-gated rows; no clearance achieved. Stop lane for same-family selector reruns. |
| Signal Compliance | proxy_only | Signal state is trace_metadata_only / planner_observable=false outside signalized_crossing fixture. Proxy diagnostic only. |
| Prediction Baseline | diagnostic_only | Merged prediction interface exists; contract-smoke rows materialized but no executed planner campaign. Denominator repair required. |
| Observation Perturbation | unavailable | No observation-perturbation evidence for this scenario slice. |
| Denominator Health | not_applicable | Denominator metric not defined for this scenario slice. |
| Stale/Current | current_negative | Negative-result candidates (issue #2788, NR-001) exist as not_promoted scenario candidates. Not benchmark evidence. |
| Claim Eligibility | not_eligible | Blockers: signal: proxy_only; observation: unavailable. |

## Summary Counts

| Status | Count |
|---|---|
| available | 1 |
| current | 4 |
| current_negative | 2 |
| diagnostic_only | 16 |
| limited_no_motion | 1 |
| not_applicable | 6 |
| not_eligible | 8 |
| partial | 2 |
| partial_robustness | 1 |
| proxy_only | 7 |
| stale | 2 |
| unavailable | 6 |

## Conservative Rules Applied

- Synthesis draft only; not benchmark, paper, or safety evidence.
- Diagnostic / stale / non-claimable / unavailable / fallback /
  degraded / proxy-only / missing-denominator rows weaken or block claims.
- Fallback behavior is not acceptable as a successful benchmark outcome.
- No invented values; missing tracked sources produce unavailable rows.
