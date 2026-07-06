# Behavior-token motion-prior diagnostics (experimental)

> **Status: very low priority, experimental, diagnostic-only.** This tooling is an
> offline prototype for inspecting saved interaction traces. It is **not** part of the
> validated metric stack, the benchmark pipeline, any release gate, or any
> paper/dissertation claim. **No safety decision may depend on these tokens.**

Tracking issue: [#4627](https://github.com/ll7/robot_sf_ll7/issues/4627).

## What this is

A small, offline experiment that asks: *can pedestrian/robot interaction traces in
`robot_sf_ll7` be represented as short discrete **behavior tokens**, and are those
tokens useful for diagnostics such as scenario grouping, failure analysis, or
regression-test stratification?*

The idea is loosely inspired by work on reusable, pretrained motion/control priors
(arXiv:2606.29148). **It is not a foundation-model claim.** There is no controller
training, no transformer dependency, and no new simulation campaign — the scripts read
existing saved traces and produce static diagnostic artifacts under `output/`.

## What this is not

- Not a new controller and not controller training.
- Not a replacement for existing safety metrics.
- Not a benchmark, leaderboard, or release input.
- Not ground-truth interaction labels — token ids and motif labels are heuristic
  *candidates* for manual inspection only.
- Not evidence for any paper/dissertation claim.

## Pipeline

Three offline stages, each a standalone CLI (`--help` on each for full options):

1. **`extract_windows.py`** — slide fixed-size windows over saved
   `algorithm_metadata.simulation_step_trace.steps` and convert each window into a
   compact, documented feature vector.
2. **`quantize_trace_windows.py`** — standardize the finite feature columns and assign
   each valid window a deterministic discrete **token id** (k-means; scikit-learn when
   available, else a NumPy-only fallback).
3. **`inspect_token_motifs.py`** — summarize token distributions by
   scenario/planner/outcome and export representative example windows plus heuristic
   *candidate* motif labels for manual inspection.

`schemas.py` holds the shared feature vocabulary and schema-version constants embedded
in every generated artifact for provenance.

### Input contract

The scripts read existing benchmark episode JSONL rows. A usable row carries:

- `algorithm_metadata.simulation_step_trace.steps` (**required** — rows without it are
  skipped with an explicit reason, never run through a new campaign);
- `scenario_id`, `seed`, and a planner identifier
  (`planner_key` / `scenario_params.algo` / `algorithm_metadata.algorithm`);
- `status` / `outcome` / `termination_reason` (used for stratification when present).

Each step is expected to look like the trace produced by the benchmark map runner:
`{step, time_s, robot: {position, heading, velocity}, pedestrians: [{id, position,
velocity?}], planner: {selected_action: {...}}}`.

### Missing-value discipline

If a feature cannot be derived for a window (for example, relative speed to the nearest
pedestrian when pedestrian velocities are absent), it is recorded as JSON `null` and
listed in the window's `missing_features`. **Genuine zero measurements** (for example,
an all-straight command sequence giving an oscillation rate of `0.0`) are recorded as
`0.0` and are *not* treated as missing. The quantizer clusters only on feature columns
that are finite across enough windows and excludes windows that lack any selected
feature (reported in its metadata), rather than back-filling fabricated values.

## Usage

```bash
# 1. Extract windows from saved traces (offline; reads only).
uv run python experiments/behavior_tokens/extract_windows.py \
  --episode-jsonl 'output/**/episodes.jsonl' \
  --window-steps 10 --stride-steps 5 \
  --output-jsonl output/experiments/behavior_tokens/windows.jsonl \
  --output-csv output/experiments/behavior_tokens/windows.csv

# 2. Quantize windows into deterministic discrete tokens.
uv run python experiments/behavior_tokens/quantize_trace_windows.py \
  --windows-jsonl output/experiments/behavior_tokens/windows.jsonl \
  --num-tokens 12 \
  --output-json output/experiments/behavior_tokens/token_assignments.json \
  --output-csv output/experiments/behavior_tokens/token_assignments.csv

# 3. Inspect token motifs and export representative examples.
uv run python experiments/behavior_tokens/inspect_token_motifs.py \
  --windows-jsonl output/experiments/behavior_tokens/windows.jsonl \
  --assignments-csv output/experiments/behavior_tokens/token_assignments.csv \
  --output-dir output/experiments/behavior_tokens/inspection
```

All outputs default to `output/experiments/behavior_tokens/...`, which is git-ignored
and worktree-local. Nothing here is promoted to durable evidence automatically.

## Feature vocabulary

The ordered feature list lives in `schemas.py` (`FEATURE_NAMES`) with one-line
descriptions in `FEATURE_DESCRIPTIONS`. It covers clearance (min/mean/slope), relative
speed to the nearest pedestrian, a time-to-contact proxy, robot speed statistics, an
acceleration/command-change proxy, a nearest-pedestrian speed-change proxy, a
stop/yield proxy, an oscillation/negotiation proxy, and a near-conflict recovery proxy.
Route-progress delta is left `null` because it is not present in the step trace.

## Tests

```bash
uv run pytest tests/experiments/test_behavior_tokens.py -q
```

The tests use synthetic in-memory traces (no saved campaign required) and cover window
counting, missing-feature handling, no-trace skipping, quantizer determinism, one token
per valid window, bounded example selection, and the experimental claim boundary.
