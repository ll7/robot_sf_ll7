# Goal

Make the published-release row gate reliable and give its BA-03 signals a typed, testable handoff to the Benchmark Auditor.

# Scope

- Fix reviewed defects in contact-speed fallback, detector identity, pedestrian-aware comparison scope, execution admission, and BA-03 handoff identity/idempotency.
- Discover pedestrian-free scenarios from baseline or configured comparison-planner
  observations by default, block unavailable or conflicting pedestrian counts, and report same-step
  terminal signatures without inferring cause.
- Add a bounded Benchmark Auditor store handoff for release-row reports; do not build the deferred BA-06 browser workflow.
- Block configured release-roster omissions even without a pedestrian-free
  cohort, reject aliased arm directories, and bind scenario annotations and
  handoff members to the verified source manifest.
- Preserve the 0.0.7 retro-detection criteria and avoid simulation, spawn, or planner changes.

# Evidence sources

- Issue #9734 and parent epic #9730, parts 4 and 6.
- Benchmark Auditor contracts and `docs/scenario_review/benchmark_auditor_v1_acceptance.md`.
- Pinned 0.0.7 publication bundle SHA-256 `684da7c557c426756f22ddbf5cb3270141ee8ae385669a39d36f324852a6fb2f`.

# Steps

1. Add focused regression tests for each reviewed defect and for the typed BA store handoff.
2. Implement the smallest contract-compatible fixes and document configuration and handoff behavior.
3. Replay the pinned 0.0.7 release rows and verify each requested incident pattern remains visible.
4. Run focused tests, lint/format, diff review, and exact-head readiness before publishing.
5. Supersede the stale draft PR only after a current-base replacement PR is open.

# Decisions and risks

- Observed: API and CLI used different registry owner names; the configured event list suppressed fallback metrics; comparison covered every non-baseline arm.
- Observed: BA-03-compatible signal records were emitted but had no supported store handoff.
- Prior reviews called out a hard-coded pedestrian-free scenario and a same-step detector with no shared outcome/event summary; the replacement reports observed signatures and labels causal attribution unavailable.
- Independent exact-head reviews found execution-status bypass, order-sensitive Auditor commits,
  signal/finding identity drift, and fail-open pedestrian cohort/baseline missingness; the gate
  now fails closed on each case, including conflicting pedestrian counts across paired rows.
- Follow-up exact-head review found that whole-arm omissions could pass without a pedestrian-free
  cohort, differential-drive suffix normalization could alias distinct directories, scenario
  annotations could cross manifests, and BA-03 findings did not validate their source members.
  Regressions now cover those boundaries.
- The BA-06 browser/reconnect/provider workflow remains outside this issue; no simulation or planner source is in scope.

# Validation route

- Focused release-row/anomaly and bundle tests, plus existing Benchmark Auditor detector/store tests touched by public admission reuse.
- Ruff check/format and `git diff --check`.
- Pinned 0.0.7 archive replay with the expected digest; exit 1 is expected because its unannotated findings block release.

# Recovery / handoff

- Changes live on `pr9791-roster-gate-fix-20260927`, branched from the PR head
  and refreshed with current `origin/main`; publish only after exact-head
  review.
- If a detector claim cannot be reproduced, report it as unavailable and do not weaken the gate to obtain a pass.
