# Issue #2230 AMV Actuation Evidence Ladder

Date: 2026-06-04

Related issue: <https://github.com/ll7/robot_sf_ll7/issues/2230>

Status: current claim-boundary guidance for AMV actuation evidence.

## Purpose

This note defines the AMV actuation evidence ladder used by benchmark reports, leaderboards,
research notes, and agent-authored issues. It prevents synthetic actuation diagnostics,
platform-class proxy sources, and hardware-calibrated AMV evidence from being reported as the same
claim.

## Ladder

| Level | Evidence source | May state | Must not state |
| --- | --- | --- | --- |
| Synthetic diagnostic | Software-defined actuation stress profile, such as `amv-actuation-stress-v0` from issue #1556/#1569. | The repository can run a diagnostic stress slice; emitted metrics such as command clipping, yaw-rate saturation, braking peak, and projection policy are useful for software sensitivity analysis. | Do not claim AMV hardware truth, deployment validity, calibrated envelopes, safety certification, or paper-facing AMV performance. |
| Platform-class proxy | Public non-AMV or adjacent-platform source with field-level provenance, such as the TRL e-scooter longitudinal proxy accepted in issue #2001. | The source can support the named fields only, currently longitudinal acceleration and braking/deceleration when cited with access date and caveats. | Do not claim full AMV truth; do not infer unsupported yaw rate, angular acceleration, latency, update rate, or command-response dynamics. |
| Hardware-calibrated | Real AMV-like command-response trace, official platform/controller specification, or accepted calibration source with manifest/checksums. | After #1585/#2000 accept a source, backed fields may support calibrated AMV actuation profiles and paper-facing actuation claims within the recorded source limits. | Do not use synthetic or proxy values as calibrated evidence; do not derive missing fields without source-backed measurements, units, and method. |

## Required Claim Language

Use the strongest true level and name excluded levels.

- Synthetic diagnostic wording: "This is a synthetic software stress diagnostic. It does not
  support calibrated or paper-facing AMV actuation claims."
- Platform-class proxy wording: "This field is backed by a platform-class proxy source and applies
  only to the listed fields. It is not hardware-calibrated AMV evidence."
- Hardware-calibrated wording: "This field is backed by the accepted AMV calibration source named
  in the provenance manifest. Unsupported fields remain unavailable."

## Current Repository Placement

- Synthetic diagnostic evidence is implemented by
  [issue_1556_amv_actuation_stress_slice.md](issue_1556_amv_actuation_stress_slice.md) and the
  compact issue #1569 smoke bundle. That smoke had valid executable rows, but every planner had
  `success_mean=0.0`; it is not an AMV performance claim.
- Platform-class proxy evidence is documented in
  [issue_2001_amv_actuation_proxy_source_analysis.md](issue_2001_amv_actuation_proxy_source_analysis.md).
  The accepted proxy currently supports longitudinal acceleration and braking/deceleration only.
- Hardware-calibrated evidence remains blocked on issue
  [#1585](https://github.com/ll7/robot_sf_ll7/issues/1585) and the real command-response trace
  acquisition issue [#2000](https://github.com/ll7/robot_sf_ll7/issues/2000).
- The issue #2011 sensitivity sweep remains diagnostic plumbing evidence because its executed rows
  were `accepted_unavailable_only`, not benchmark evidence.

## Reporting Rules

1. If an evidence row is fallback, degraded, failed, not available, or
   `accepted_unavailable_only`, do not report it as benchmark success under
   [issue_691_benchmark_fallback_policy.md](issue_691_benchmark_fallback_policy.md).
2. If `amv_coverage_status` is `warn`, keep the missing-coverage caveat attached to any AMV
   actuation summary.
3. If a profile includes synthetic yaw-rate, angular-acceleration, latency, or update-rate values,
   keep those fields at the synthetic diagnostic level even when longitudinal values use a proxy
   source.
4. If a paper, dissertation figure, leaderboard, or benchmark report cites AMV actuation evidence,
   it must name the ladder level and link the provenance surface.

## Minimal Promotion Checklist

Before moving a claim upward on the ladder, verify:

- the source class and `claim_boundary` are explicit;
- supported and unsupported fields are listed separately;
- values have units, sign conventions, and derivation method;
- source URI, owner, access constraints, version/date, and checksum or manifest are recoverable;
- synthetic and proxy fields are not mixed under a calibrated label;
- row statuses exclude fallback/degraded/unavailable evidence from benchmark success.

## Validation

This note is documentation guidance only. It creates no new benchmark evidence and does not modify
metrics or configs. Validate updates with the docs proof consistency checker and path/link checks.
