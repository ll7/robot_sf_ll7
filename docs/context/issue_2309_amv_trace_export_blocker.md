# Issue #2309 AMV Trace Export Blocker

Issue: [#2309](https://github.com/ll7/robot_sf_ll7/issues/2309)  
Parent issue: [#2159](https://github.com/ll7/robot_sf_ll7/issues/2159)  
Related issues: [#2168](https://github.com/ll7/robot_sf_ll7/issues/2168),
[#2269](https://github.com/ll7/robot_sf_ll7/issues/2269),
[#2281](https://github.com/ll7/robot_sf_ll7/issues/2281)  
Date: 2026-06-05  
Status: historical failed-closed trace-export blocker; partially superseded by Issue #2405.

Update 2026-06-06: Issue #2405 added the missing benchmark-worker pass-through for
`--record-simulation-step-trace` and proved one selected default/AMMV row per side can export as
loader-valid `simulation_trace_export.v1`. This note remains the historical failed-closed probe for
the pre-#2396 path; it should not be read as saying the single-row diagnostic export path is still
blocked.

## Decision

The `ammv_head_on_corridor_mechanism_activation` case remains blocked for renderable trace-review
use.

The Issue #2168 benchmark commands are regenerable, but they regenerate aggregate benchmark episode rows,
not step-event JSONL with `state.robot_pose` frames. `scripts/tools/build_simulation_trace_export.py`
therefore cannot convert them into `simulation_trace_export.v1` artifacts. The generated local
probe outputs are classified as `discard` / `non-evidence-local-only` and must not be cited as
durable trace evidence.

## Evidence

Compact summary:
`docs/context/evidence/issue_2309_amv_trace_export_probe_2026-06-05/summary.json`.

Observed local probe:

| Input | Result | Implication |
| --- | --- | --- |
| Issue #2168 default Social Force rerun | 3 aggregate episode rows; no step frames | Regenerable episode metrics, not a trace-review input. |
| Issue #2168 AMMV Social Force rerun | 3 aggregate episode rows; no step frames | Regenerable episode metrics, not a trace-review input. |
| `build_simulation_trace_export.py` on default rerun JSONL | failed with `has no step frames for conversion` | Existing exporter cannot materialize `simulation_trace_export.v1` from this output. |

The regenerated episode rows reproduce the prior structural blocker from #2168: the benchmark path
uses adapter-mode Social Force rows and does not surface AMMV force/intrusion metadata in the
episode JSONL. The direct mechanism probe remains useful diagnostic evidence, but it is not a
renderable simulation trace.

## Current Blockers

- Matched default-vs-AMMV `simulation_trace_export.v1` traces do not exist for
  `classic_head_on_corridor_low`.
- The current `robot_sf_bench run` episode JSONL output does not preserve per-step state frames for
  the trace exporter.
- The adapter benchmark rows do not expose AMMV force/intrusion metadata, so a trace export from
  the current benchmark path would still miss the key mechanism signal.

## Next Smallest Proof Step

Choose one implementation path before reopening the AMV-specific trace export child:

1. Add a benchmark or recorder path that emits step-event JSONL compatible with
   `scripts/tools/build_simulation_trace_export.py` while preserving AMMV force/intrusion metadata.
2. Add a narrow direct-probe trace exporter for the controlled `SocialForcePlanner` mechanism probe,
   with explicit limitations that it is robot-planner-only and does not measure pedestrian lateral
   deviation or speed adaptation.

Until one of those paths exists, use one of the three already durable compact trace-slice cases from
Issue #2281 for rendered review or annotation work.

## Claim Boundary

This note is blocker/provenance evidence only. It does not create benchmark-strength evidence,
paper-facing AMV evidence, or renderable AMV trace evidence.

## Validation

Commands run:

```bash
scripts/dev/run_worktree_shared_venv.sh -- uv run robot_sf_bench validate-config --matrix configs/scenarios/issue_2168_ammv_social_force_pair.yaml
scripts/dev/run_worktree_shared_venv.sh -- uv run robot_sf_bench preview-scenarios --matrix configs/scenarios/issue_2168_ammv_social_force_pair.yaml
scripts/dev/run_worktree_shared_venv.sh -- uv run robot_sf_bench run --matrix configs/scenarios/issue_2168_ammv_social_force_pair.yaml --out <worktree-local-default-jsonl> --base-seed 111 --repeats 1 --horizon 100 --dt 0.1 --record-forces --no-video --video-renderer none --algo social_force --workers 1 --no-resume --structured-output json
scripts/dev/run_worktree_shared_venv.sh -- uv run robot_sf_bench run --matrix configs/scenarios/issue_2168_ammv_social_force_pair.yaml --out <worktree-local-ammv-jsonl> --base-seed 111 --repeats 1 --horizon 100 --dt 0.1 --record-forces --no-video --video-renderer none --algo social_force --algo-config configs/baselines/social_force_ammv_aware.yaml --workers 1 --no-resume --structured-output json
scripts/dev/run_worktree_shared_venv.sh -- uv run python scripts/tools/build_simulation_trace_export.py --source <worktree-local-default-jsonl> --output <worktree-local-trace-export-json> --planner-id default_social_force --scenario-id classic_head_on_corridor_low
```

The final command failed for the expected blocker reason:
`<worktree-local-default-jsonl> has no step frames for conversion`.
