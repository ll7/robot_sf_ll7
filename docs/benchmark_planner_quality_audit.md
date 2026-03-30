# Planner Quality Audit Workflow

This workflow produces a conservative planner decision table for the current benchmark suite.
It is intended to separate:

- credible local benchmark baselines,
- weak but honest local implementations,
- paper-reference mismatches,
- and planners that should not appear in the headline comparison set.

The audit combines two campaign surfaces:

- the corrected hard matrix campaign, and
- the planner sanity matrix.

It also overlays a checked-in policy file that records conservative paper-faithfulness judgments and
external reproduction priorities.

## Canonical command

```bash
uv run python scripts/tools/build_planner_quality_audit.py \
  --hard-matrix-root output/benchmarks/camera_ready/<hard_campaign_id> \
  --sanity-matrix-root output/benchmarks/camera_ready/<sanity_campaign_id> \
  --parity-config configs/benchmarks/planner_quality_audit_v1.yaml \
  --output-json output/benchmarks/camera_ready/<hard_campaign_id>/reports/planner_quality_audit.json \
  --output-md output/benchmarks/camera_ready/<hard_campaign_id>/reports/planner_quality_audit.md
```

## Output

The generated audit contains:

- one planner decision table with success/collision/max-steps/SNQI/runtime,
- sanity-matrix capability confirmation,
- primary failure mode from raw `episodes.jsonl`,
- a conservative headline-suite recommendation,
- per-planner paper-faithfulness notes,
- explicit parity-gap notes for the strongest external literature families,
- and an ordered external reproduction priority list.

## Interpretation rules

- `credible benchmark baseline`: safe to keep in the headline local benchmark suite.
- `weak but honest local implementation`: runnable and informative, but not strong enough for headline
  paper-facing comparison.
- `paper-reference mismatch`: the current result should not be interpreted as a literature-family result.
- `not suitable for headline comparison`: keep as a sanity/control baseline only.

Use the audit conservatively. Weak local performance is still useful evidence, but it should not be
reported as family-level underperformance unless the source implementation and benchmark contract are
close enough to support that claim.
