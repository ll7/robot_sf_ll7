---
name: analyze-latest-policy-sweep
description: Analyze latest policy analysis sweep runs (*_policy_analysis_*) by comparing episodes/summary metrics, diagnostics, and video artifacts; generate a concise markdown report and optional frame snapshots.
---

# Analyze Latest Policy Sweep

## When to use

Use this skill when comparing recent `*_policy_analysis_*` runs and producing a compact, reproducible
diagnostic summary with caveats.

## Read First

- `docs/code_review.md`
- `docs/benchmark_spec.md`
- `docs/context/issue_691_benchmark_fallback_policy.md`

## Quick start

1. Confirm run directories contain `episodes.jsonl` and `summary.json`.
2. Load per-episode metrics and summary aggregates.
3. Write a report under `specs/409-planner-comparison/`.

## Required report contents
- Aggregate metrics table (success rate, collision rate, time_to_goal_norm, avg_speed, near_misses, comfort_exposure, energy, jerk/curvature raw + filtered if present).
- Worst collision scenarios (top 5) per run.
- Lowest success scenarios (bottom 5) per run.
- Top problem episodes (score = collisions*10 + comfort*2 + failure) with video paths.
- Diagnostics: path_efficiency saturation, `shortest_path_len` presence, low-speed filter behavior (ε=0.1).

## Workflow

1. Load each run and validate required files.
2. Compare `episodes.jsonl` and `summary.json` consistency.
3. Compute and rank diagnostics.
4. Attach video evidence for top problem episodes.
5. Add explicit mode labeling (`native`/`adapter`/`fallback`/`degraded`) to any planner comparison.

## Optional
- Extract frames (25/50/75% of duration) for top problem episodes using ffmpeg into `output/analysis/<report_id>/frames/` and list PNG paths in the report.

## Template
Use `specs/409-planner-comparison/analyze_latest_policy_sweep_prompt.md` as a template if you need a consistent structure.

## Video naming convention
`output/recordings/<run_dir>/<scenario>_seed<seed>_<policy>.mp4`

## Diagnostics details
- **Path-efficiency saturation**: fraction with `path_efficiency >= 0.999` (clipped at 1.0).
- **Shortest path presence**: ensure `shortest_path_len` is present in metrics when available.
- **Low-speed filter**: report `low_speed_frac` and filtered metrics (`jerk_mean_eps0p1`, `curvature_mean_eps0p1` when present). Reason: curvature divides by |v|^3; near-zero speed inflates noise.

## Proof and Guardrails

- Treat mismatches between `episodes.jsonl` and `summary.json` as blocking.
- Keep low-speed filtering explicit and report the method used.
- Fail-closed: do not claim planner superiority when any planner is only partially valid.
- Always include evidence location links in the final report.

## Output

Report:
- run set,
- consistency checks and pass/fail status,
- ranking summary,
- top failures/scenarios with links,
- reproducibility note and open risks.
