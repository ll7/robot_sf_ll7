---
name: analyze-camera-ready-benchmark
description: Analyze a camera-ready benchmark campaign for consistency, runtime hotspots, fallback/degraded
  planners, and reproducibility metadata.
category: benchmark-evidence
kind: atomic
phase: context
requires_write: false
requires_slurm: false
requires_benchmark_artifacts: true
delegates_to: []
output_schema: skill_run_summary.v1
aliases:
- analyze-latest-policy-sweep
---

# Analyze Camera-Ready Benchmark

## When to use

Use this skill when reviewing `*_policy_analysis*` or camera-ready campaign outputs and you need
consistency, reproducibility, and fallback-mode visibility before claiming benchmark conclusions.

## Modes

Use the default `camera-ready` mode for a complete campaign. Select the `policy-sweep` mode when
comparing recent `*_policy_analysis_*` runs. The legacy `analyze-latest-policy-sweep` name is a
compatibility alias for this canonical mode.

### Policy-sweep mode

1. Confirm each run directory contains `episodes.jsonl` and `summary.json`.
2. Compare per-episode metrics with summary aggregates; any mismatch blocks interpretation.
3. Write a compact report under `specs/409-planner-comparison/` and include evidence-location links.

The policy-sweep report preserves this contract:

- aggregate success, collision, time-to-goal, speed, near-miss, comfort, energy, and raw/filtered
  jerk/curvature metrics;
- the five worst collision scenarios, five lowest-success scenarios, and top problem episodes
  (`collisions*10 + comfort*2 + failure`) with video paths;
- path-efficiency saturation, `shortest_path_len` presence, and the explicit low-speed filter
  (`epsilon=0.1`) diagnostics;
- explicit `native`, `adapter`, `fallback`, or `degraded` mode labels for every comparison.

Optional frame extraction uses ffmpeg at 25/50/75% of each top-problem episode and writes to
`output/analysis/<report_id>/frames/`. Use
`specs/409-planner-comparison/analyze_latest_policy_sweep_prompt.md` as the report template when
needed. Videos follow
`output/recordings/<run_dir>/<scenario>_seed<seed>_<policy>.mp4`.

| Legacy contract item | Canonical policy-sweep disposition |
| --- | --- |
| `*_policy_analysis_*` trigger and run inputs | Named `policy-sweep` mode; require `episodes.jsonl` and `summary.json`. |
| Aggregate, worst/lowest, and top-problem tables | Required report fields above; unchanged scoring and links. |
| Path-efficiency, shortest-path, and low-speed diagnostics | Required diagnostics above; filtering remains explicit. |
| Optional frames, template, and video names | Preserved as optional/reporting conventions above. |
| Mismatch/fallback/superiority guardrails | Canonical proof and guardrails below; no ranking from partial or fallback evidence. |

## Read First

- `docs/code_review.md`
- `docs/benchmark_spec.md`
- `docs/context/issue_691_benchmark_fallback_policy.md`
- `docs/benchmark_camera_ready.md`

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
   - Check `campaign_analysis.json` findings for:
     - episode count mismatches
     - summary-vs-episodes metric mismatches
     - adapter impact metadata mismatches
     - fallback preflight planners

4. Summarize runtime + quality
   - Start the report with claim boundary, evidence status, fallback/degraded exclusions, and major
     caveats before planner rankings or success interpretation when evidence is mixed or limited.
   - Highlight slowest planners by `runtime_sec`.
   - Highlight weakest planners by success/collision/SNQI from derived episode means.
   - Note experimental planners running with fallback mode and make uncertainty visible before any
     ranking language.

5. Recommend actions
   - If inconsistencies are found, propose concrete follow-ups in issue-facing wording.
   - If behavior is expected, explicitly document rationale and residual risk.

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

## Proof and Guardrails

- Classify each run mode as `native`, `adapter`, `fallback`, or `degraded`.
- Fail-closed rule: do not treat fallback/degraded as successful benchmark evidence.
- Fallback/degraded rows must appear as caveats or exclusions before comparative rankings,
  aggregate success language, or recommendations.
- Mixed/limited evidence outputs must lead with claim boundary, evidence status, and caveats; only
  then provide comparative interpretation.
- Require the produced JSON + markdown reports as proof artifacts before summarizing results.
- Only report planner ranking where metrics and episode counts are internally consistent.

## Output

Provide a concise judgment with:
- campaign id/path,
- evidence list (`.json` + `.md`),
- consistency status,
- planner ranking by observed-mode,
- concrete remediation steps or explicit pass conditions.
## Guardrails

- Stay within the skill scope declared in `.agents/skills/skills.yaml`.
- Prefer repository scripts and canonical docs before ad-hoc commands.
- Record blockers and validation gaps instead of overstating completion.
