# Issue #3062 Campaign Manifest Flow

Issue: <https://github.com/ll7/robot_sf_ll7/issues/3062>

## Summary

Issue #3062 standardizes the first reusable contract for research campaign
manifests and artifact handoff. The contract is documented in
`docs/benchmark_campaign_manifest.md`, with a load-tested example at
`configs/benchmarks/research_campaign_manifest.example.yaml`.

The contract is intentionally proposal-level. It records the shape future
campaign manifests should expose before runner-specific automation treats them
as benchmark or paper-facing evidence.

## Covered Fields

The example manifest covers scenario suite selection, planner rows, seed policy,
metrics, row-status policy, local output layout, summary JSON/table
expectations, durable evidence pointers, and validation commands.

`output/` remains a disposable local cache. Durable evidence must be promoted to
`docs/context/evidence/`, a release/publication bundle, or an external artifact
pointer with checksums and fail-closed hydration behavior.

## Validation

Focused validation:

```bash
uv run pytest tests/benchmark/test_research_campaign_manifest_contract.py
```

This proves the example manifest loads and contains the required contract
sections. It does not run a campaign and does not claim benchmark results.
