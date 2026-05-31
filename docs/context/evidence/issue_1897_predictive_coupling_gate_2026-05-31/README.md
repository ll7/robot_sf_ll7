# Issue #1897 Predictive Coupling Gate Evidence

Related issue: <https://github.com/ll7/robot_sf_ll7/issues/1897>

## Boundary

This is local preflight evidence only. The checkpoint came from a local machine cache in the main
checkout's ignored model-cache tree, with SHA-256:

```text
a28aed6d6ad7e1ebf597277ade1cf908efa6da038d0a9fcfdf80c7c31d8d1be1
```

The raw campaign JSONL files remain in ignored worktree-local storage and are not durable repository
artifacts. Do not use this result as paper-facing or promotion evidence.

## Command

```bash
LOGURU_LEVEL=WARNING TF_CPP_MIN_LOG_LEVEL=2 \
  scripts/dev/run_worktree_shared_venv.sh -- \
  python scripts/validation/run_predictive_success_campaign.py \
    --checkpoints <local-selected-v2-predictive-model.pt> \
    --planner-grid configs/benchmarks/predictive_sweep_planner_grid_v2_coupling_gate.yaml \
    --horizon 80 \
    --workers 1 \
    --bootstrap-samples 200 \
    --closed-loop-gate-baseline-variant baseline_like \
    --closed-loop-gate-min-global-success-delta 0.02 \
    --closed-loop-gate-min-hard-success-delta 0.0 \
    --closed-loop-gate-max-min-distance-regression 0.10 \
    --output-dir <ignored-local-campaign-dir>
```

The runner returned exit code `2` because the closed-loop gate failed. It still wrote
`campaign_summary.json` and `campaign_report.md` under the ignored local campaign directory.

## Result

| Variant | Hard episodes | Hard success | Hard mean min-distance | Global episodes | Global success | Global mean min-distance |
|---|---:|---:|---:|---:|---:|---:|
| `phase_coupled_sequence_gate` | 7 | 0.0000 | 2.2748 | 69 | 0.0000 | 3.5568 |
| `baseline_like` | 7 | 0.0000 | 2.2018 | 69 | 0.0000 | 3.5460 |

Closed-loop gate:

- status: `failed`
- reason: `global_success_delta_below_gate`
- global success delta: `0.0000`
- hard success delta: `0.0000`
- global mean-min-distance delta: `0.0108`

Interpretation: the phase-coupled row ranked higher only because clearance improved slightly. It
did not improve closed-loop success, so the predictive-v2 expansion should stay blocked and the
coupling/objective should be revised or stopped rather than routed to the old four-way matrix.

## Setup Repair

The first attempt failed before evaluation because
`configs/benchmarks/predictive_hard_seeds_v1.yaml` still listed deprecated
`classic_crossing_*` aliases that are not present in `configs/scenarios/classic_interactions.yaml`.
This branch removes those alias rows and adds a regression test proving the default hard manifest
loads against the default scenario matrix.
