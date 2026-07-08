# Context-Pack Manifests

Issue: [#1871](https://github.com/ll7/robot_sf_ll7/issues/1871)

This directory contains small, source-controlled manifests for recurring agent context packs. They
describe what to include when building a focused Repomix or equivalent bundle; they are not generated
bundles themselves.

Generated pack outputs must stay under ignored paths such as `output/context_packs/`. Do not commit
packed repository dumps, raw benchmark output, videos, checkpoints, or local machine context.

## Manifests

| Pack | Entry point | Use for |
|---|---|---|
| [learned_policy_integration.yaml](learned_policy_integration.yaml) | `docs/context/policy_search/learned_policy_registry.md` | Learned-policy eligibility, adapter contracts, policy-card boundaries, and durable model metadata. |
| [policy_search.yaml](policy_search.yaml) | `docs/context/policy_search/INDEX.md` | Candidate lifecycle routing, stage-gated execution, promotion gates, and policy-search tooling. |
| [benchmark_evidence.yaml](benchmark_evidence.yaml) | `docs/context/issue_691_benchmark_fallback_policy.md` | Benchmark fallback policy, review/evidence vocabulary, release surfaces, and compact proof boundaries. |
| [visualization_workbench.yaml](visualization_workbench.yaml) | `docs/debug_visualization.md` | Diagnostic trace export, analysis-workbench schemas, and benchmark visualization boundaries. |

Each manifest uses the same required fields:

- `name`: stable pack slug.
- `purpose`: short human-readable scope.
- `entrypoint`: first file to read before expanding the pack.
- `include`: files or globs that form the pack.
- `exclude`: files or globs that must stay out of generated packs.
- `do_not_read_by_default`: high-volume, generated, historical, or optional surfaces that require a
  specific reason before reading.

