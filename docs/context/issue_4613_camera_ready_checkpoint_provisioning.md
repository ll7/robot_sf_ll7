# Issue #4613 — Camera-ready checkpoint provisioning runbook

## Plain-language summary

Camera-ready benchmark campaigns bind each learned-policy arm to a model
checkpoint through the arm's `algo_config` (a `model_id` resolved through the
model registry, or a direct `model_path`). The S30 campaign (horizon-600 hybrid
vs ORCA) failed twice ~14h into compute (jobs 13296 and 13301) because the
submit worktree's `output/model_cache` did not durably contain a requested PPO
checkpoint. The strict all-rows campaign policy converted one un-loadable arm
into a whole-campaign failure *after* the compute had already burned.

This runbook is the ops contract for preventing that failure class at
submit time. It is provisioning-only: it runs no benchmark, submits no Slurm
job, and is not benchmark evidence.

## Pre-submit contract

Run the **enforced staged** checkpoint preflight on the submit node **before**
sbatch:

```bash
scripts/benchmark/submit_camera_ready_checkpoint_gate.sh \
  --config configs/benchmarks/paper_experiment_matrix_v1_h600_hybrid_vs_orca_s30.yaml \
  --report-path output/benchmarks/camera_ready/<campaign_id>/preflight/checkpoint_staging.json
```

The gate:

- resolves every enabled arm's `model_id` / `model_path` (recursively, so a
  nested prior-policy checkpoint is covered);
- downloads and checksum-verifies each registry artifact into the durable
  cache (`stage=True`);
- writes a per-arm staging report (`submit_safe=true` on success);
- exits `3` (fail-closed; do not submit) on any unresolvable or corrupt
  checkpoint, naming the planner key, model id, config path, and remedy.

Equivalent direct CLI:

```bash
uv run python scripts/benchmark/preflight_campaign_checkpoints.py \
  --config <campaign-config> --stage --json \
  --report-path <campaign-root>/preflight/checkpoint_staging.json
```

## Modes (do not confuse)

- `metadata_only` (default, cheap, network-free): accepts `present_local` OR
  `stageable_remote`. The always-on guard inside
  `prepare_campaign_preflight()` runs in this mode so it never breaks offline
  preflight-only workflows. **`stageable_remote` is not submit-safe.** The
  JSON `submit_safe` field reports `false` when any arm is only
  `stageable_remote`.
- `enforced_staged` (`--stage`): downloads and checksum-verifies every registry
  artifact. The submit/sbatch wrapper must use this mode. After a successful
  run, `submit_safe` is `true`.

`prepare_campaign_preflight(checkpoint_preflight_mode="enforced_staged")`
exposes the same branch for callers that want the staging step inside the
preflight-only workflow; `run_camera_ready_benchmark.py --mode preflight
--checkpoint-preflight-mode enforced_staged` is the public CLI surface.

## S30 requeue rules

- S30 requeue must **resume-append** into the retained **15,469-episode**
  campaign root. Do **not** create a fresh root.
- Run the checkpoint provisioning gate above first; record the
  `checkpoint_staging.json` report with the requeue packet.
- `stageable_remote` without a staged cache is **not** sufficient — even if
  the metadata-only guard passes, the compute node would re-discover the
  same missing-cache failure. Always requeue with `--stage`.
- No benchmark interpretation changes from this provisioning fix. The
  requeue produces no new claim and does not by itself constitute benchmark
  evidence; it only removes a recurring missing-checkpoint whole-campaign
  failure mode.

## What the artifacts look like

When `prepare_campaign_preflight()` runs, the per-arm summary is persisted in
two places:

1. `preflight/checkpoint_staging.json` (enforced_staged mode) or
   `preflight/checkpoint_resolvability.json` (metadata_only mode) — the
   report next to the other preflight artifacts, also referenced from the
   campaign manifest under `artifacts.preflight_checkpoint_provisioning`.
2. `preflight/validate_config.json` `checkpoint_preflight` block — embedded
   summary for the validate-config artifact itself.

Both record `mode`, `stage`, `checked`, `resolved`, `submit_safe`, and the
per-arm list (`planner_key`, `algo`, `kind`, `value`, `status`,
`resolved_path`).

## Public surface added by #4613

- `robot_sf/benchmark/campaign_checkpoint_preflight.py` — module + `submit_safe`.
- `robot_sf/benchmark/camera_ready/_preflight.py::prepare_campaign_preflight`
  — `checkpoint_preflight_mode`, `checkpoint_cache_dir`,
  `checkpoint_registry_path` parameters + persisted report JSON.
- `scripts/benchmark/preflight_campaign_checkpoints.py` — `--stage`,
  `--report-path`, JSON `submit_safe`.
- `scripts/benchmark/submit_camera_ready_checkpoint_gate.sh` — public
  pre-sbatch gate (ops sbatch wrapper must call this before requeueing).
- `scripts/tools/run_camera_ready_benchmark.py` —
  `--checkpoint-preflight-mode`, `--checkpoint-cache-dir`,
  `--checkpoint-registry-path` (preflight-only mode).
- Tests: `tests/benchmark/test_campaign_checkpoint_preflight.py`,
  `tests/benchmark/test_camera_ready_checkpoint_submit_preflight.py`.

## Out of scope

This runbook does not授权 re-running the S30 campaign or interpreting its
results; the requeue itself is ops-owned compute. No benchmark metric, no
paper-facing claim, and no Slurm submission is performed by the provisioning
fix or by this runbook.