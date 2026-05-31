# Learned-Policy Cards

Related issue: <https://github.com/ll7/robot_sf_ll7/issues/1865>

Policy cards are human-readable summaries for learned local-navigation policies and learned-policy
candidates. They sit above the canonical registries and reports: a card should make a policy's
status, contracts, artifacts, evidence, and limitations easy to inspect without turning planning
metadata into benchmark evidence.

## Evidence Boundary

These cards are documentation surfaces. They do not promote a checkpoint, create benchmark
eligibility, or override the fail-closed fallback policy. Treat each factual field as sourced from
the linked registry, config, artifact manifest, or report. If a field is absent from those sources,
write `unknown`, `pending`, or `blocked`; do not infer it from the policy name.

Canonical sources:

- [Learned local-navigation policy registry](../context/policy_search/learned_policy_registry.md)
- [Policy-search candidate registry](../context/policy_search/candidate_registry.yaml)
- [Model registry](../../model/registry.yaml)
- [Learned local-policy eligibility checklist](../context/policy_search/contracts/learned_local_policy_eligibility.md)
- [External learned-policy intake contract](../context/policy_search/contracts/external_policy_intake.md)
- [Artifact evidence vocabulary](../context/artifact_evidence_vocabulary.md)
- [Benchmark fallback policy](../context/issue_691_benchmark_fallback_policy.md)

## Initial Cards

| Policy card | Current status | Evidence boundary |
| --- | --- | --- |
| [`ppo_issue791_best_v1`](ppo_issue791_best_v1.md) | Implemented learned baseline with comparison evidence | Best current learned-only baseline for success-oriented comparison; not a safety promotion or OOD claim. |

## Template

Use this template for new cards. Keep list values short and link every durable claim.

```yaml
policy_id: stable identifier from learned_policy_registry.md
policy_family: learned_baseline | guarded_policy | residual_policy | auxiliary_risk |
  predictive_model | lidar_policy | external_graph_policy | external_learned_policy |
  external_visual_policy | external_world_model
card_status: draft | current | blocked | superseded
registry_status:
  integration_status: implemented | staged | adapter_needed | monitor_only | rejected
  reproducibility_status: smoke_proven | launch_packet | source_harness_required |
    source_smoke_proven | comparison_available | proposal | prototype_only | monitor_only |
    blocked | rejected
  benchmark_status: smoke_only | comparison_available | not_benchmark_evidence |
    blocked | rejected | rejected_for_current_adapter
benchmark_track: named benchmark or promotion track, smoke-only track, not_benchmark_evidence,
  blocked, or unknown
evidence_boundary: one sentence describing what the current evidence supports
not_for:
  - claims this card must not support
source_links:
  learned_policy_registry: path
  candidate_registry: path or not_applicable
  model_registry: path or not_applicable
  reports:
    - path
contracts:
  observation_contract: exact observation mode/level/keys or unknown
  action_contract: output family, frame, units, bounds, and projection policy
  fallback_policy: fail-closed behavior or fallback/degraded caveat
artifacts:
  checkpoint_uri: durable URI or pending
  checkpoint_checksum: checksum or unknown
  normalizer_uri: durable URI, not_required, unknown, or pending
  training_config: path or unknown
  training_data_or_split: training/eval source, split contract, caveat, unknown, or pending
  license_or_access: known license/access note or unknown
evidence:
  smoke: command/report/status or not_run
  benchmark: report/status or not_benchmark_evidence
known_failures:
  - observed limitation or blocked prerequisite
review_notes:
  - maintenance note for future updates
```

## Maintenance Rules

1. Add a card only when registry or model metadata is complete enough to avoid speculation.
2. Prefer one card per stable `policy_id`.
3. Separate existence, runnable smoke proof, benchmark comparison, and promotion.
4. Mark local `output/` paths as cache or reproduced-output references unless they are backed by a
   durable artifact entry.
5. Keep fallback, degraded, guard-dominated, or missing-artifact execution as caveats unless the
   issue explicitly measures that mode.
6. Update the card when the learned-policy registry, candidate registry, model registry, or linked
   report changes the policy's status.
7. Run `uv run python scripts/validation/validate_platform_docs.py` before publishing card changes.
