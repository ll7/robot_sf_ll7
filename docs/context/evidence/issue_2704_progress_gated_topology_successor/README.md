# Issue #2704 Progress-Gated Topology Successor Evidence

This directory preserves the compact reviewable summary for the diagnostic-only Issue #2704 paired
smoke. Raw diagnostic traces and reports remain disposable local artifacts.

## Payload

- [summary.json](summary.json): paired candidate/comparator diagnostic summary for
  `topology_guided_hybrid_rule_v0_progress_gated_reselection` versus
  `topology_guided_hybrid_rule_v0` on
  `classic_realworld_double_bottleneck_high`, seed `111`, horizon `160`.

## Boundary

The summary supports a `revise` classification only. It proves the progress-gated successor can run
and emit the intended fields, but it does not establish benchmark-strength evidence or planner
improvement.

## Reproduction Commands

```bash
LOGURU_LEVEL=WARNING TF_CPP_MIN_LOG_LEVEL=2 PYGAME_HIDE_SUPPORT_PROMPT=1 DISPLAY= \
  MPLBACKEND=Agg SDL_VIDEODRIVER=dummy uv run python \
  scripts/validation/run_topology_hypothesis_diagnostics.py \
  --candidate topology_guided_hybrid_rule_v0_progress_gated_reselection \
  --stage full_matrix \
  --scenario-name classic_realworld_double_bottleneck_high \
  --seed 111 \
  --horizon 160 \
  --max-hypotheses 3 \
  --min-hypotheses 2 \
  --output-dir /tmp/robot_sf_issue2704_progress_gated_successor

LOGURU_LEVEL=WARNING TF_CPP_MIN_LOG_LEVEL=2 PYGAME_HIDE_SUPPORT_PROMPT=1 DISPLAY= \
  MPLBACKEND=Agg SDL_VIDEODRIVER=dummy uv run python \
  scripts/validation/run_topology_hypothesis_diagnostics.py \
  --candidate topology_guided_hybrid_rule_v0 \
  --stage full_matrix \
  --scenario-name classic_realworld_double_bottleneck_high \
  --seed 111 \
  --horizon 160 \
  --max-hypotheses 3 \
  --min-hypotheses 2 \
  --output-dir /tmp/robot_sf_issue2704_baseline_comparator
```
