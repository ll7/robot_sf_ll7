# Issue #1935 Stronger Perturbation-Planner Evidence

This bundle contains compact diagnostic evidence for issue #1935, the stronger-planner follow-up
to the #1610 scenario perturbation criticality pilot.

- `summary.json`: reviewable seed-limit-4 paired no-op-versus-route-offset summary for `goal`,
  `orca`, and `scenario_adaptive_hybrid_orca_v2_collision_guard`, with planner execution metadata
  showing the policy-search candidate resolved to `hybrid_rule_local_planner` plus its candidate
  config path.
- `SHA256SUMS`: checksum for the tracked summary.

Raw episode JSONL, generated scenario matrices, route override files, and local runner summaries
remain under ignored `output/` paths and are not mirrored here.

Claim boundary: diagnostic local pilot only; not benchmark-strength or paper-facing evidence.
