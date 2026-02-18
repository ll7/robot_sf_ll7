---
name: analyze-camera-ready-benchmark
description: "Analyze a camera-ready benchmark campaign for consistency, runtime hotspots, fallback/degraded planners, and reproducibility metadata."
---

# Analyze Camera-Ready Benchmark

## Overview

Use this skill when reviewing benchmark outputs under:

- `output/benchmarks/camera_ready/<campaign_id>/`

The skill runs the campaign analyzer, summarizes findings, and proposes next actions.

## Workflow

1. Resolve target campaign
   - Use the campaign root provided by the user.
   - If not provided, ask for a specific campaign id/path.

2. Run analyzer
   - Command:
     - `uv run python scripts/tools/analyze_camera_ready_campaign.py --campaign-root <campaign_root>`
   - This writes:
     - `reports/campaign_analysis.json`
     - `reports/campaign_analysis.md`

3. Validate core consistency
   - Check for findings in `campaign_analysis.json`:
     - episode count mismatches
     - summary-vs-episodes metric mismatches
     - adapter impact metadata mismatches
     - fallback preflight planners

4. Summarize runtime + quality
   - Highlight slowest planners by `runtime_sec`.
   - Highlight weakest planners by success/collision/SNQI from derived episode means.
   - Note experimental planners running with fallback mode.

5. Recommend actions
   - If inconsistencies are found, provide concrete file-level follow-ups.
   - If behavior is expected, document rationale and residual risk.

## Typical Commands

```bash
uv run python scripts/tools/analyze_camera_ready_campaign.py \
  --campaign-root output/benchmarks/camera_ready/<campaign_id>
```

Optional custom outputs:

```bash
uv run python scripts/tools/analyze_camera_ready_campaign.py \
  --campaign-root output/benchmarks/camera_ready/<campaign_id> \
  --output-json output/benchmarks/camera_ready/<campaign_id>/reports/custom_analysis.json \
  --output-md output/benchmarks/camera_ready/<campaign_id>/reports/custom_analysis.md
```
