# Issue #1937 Pedestrian Route-Offset Evidence

This bundle contains compact diagnostic evidence for issue #1937, the pedestrian-route-offset
follow-up to the #1610 scenario perturbation criticality pilot.

- `summary.json`: reviewable seed-limit-4 paired no-op-versus-route-offset summary for `goal`,
  `orca`, and `scenario_adaptive_hybrid_orca_v2_collision_guard`, grouped by planner, source
  scenario, and perturbation family.
- `SHA256SUMS`: checksum for the tracked summary.

Raw episode JSONL, generated scenario matrices, route override files, and local runner summaries
remain under ignored `output/` paths and are not mirrored here.

Claim boundary: diagnostic local pilot only; not benchmark-strength or paper-facing evidence.
