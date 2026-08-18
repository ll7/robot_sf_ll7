# Issue #5409 horizon ablation launch packet

The roster-matched, seed-matched h500-vs-h600 comparison of issue #5409 has had valid
campaign configs and a fail-closed pair validator since PR #5422, and a guarded-PPO
availability preflight since PR #5699. What it did not have was a single, checkable
**submission coordinate**: where each horizon writes its results, which pre-`sbatch`
receipt must exist, and which environment identity a completed run is reconstructable
from. This packet is that coordinate.

- Machine-readable packet:
  `configs/benchmarks/issue_5409_horizon_ablation_launch_packet.yaml`
- Fail-closed checker:
  `scripts/validation/check_issue_5409_horizon_ablation_launch_packet.py`

The checked-in packet is launch-packet v2. Its `campaign_identity` block is a versioned,
explicit h500/h600 pair consumed by both the launch checker and the paired-report builder. The
canonical pair remains `issue5409_horizon_ablation_h500` and
`issue5409_horizon_ablation_h600`. A historical v1 packet remains readable only with that fixed
pair; a reviewed rerun must declare a v2 pair instead of relying on a suffix convention.

```bash
uv run python scripts/validation/check_issue_5409_horizon_ablation_launch_packet.py --json
```

Exit `0` = ready, `1` = blocked (a contract requirement is unmet), `2` = malformed.

## What this packet is not

It runs no episodes, submits no SLURM job, stages no checkpoint, and establishes no
horizon finding. The paired result does not exist until both campaigns complete on the
cluster and the paired matched-key comparison is computed.

## Frozen matrix

Measured from `--mode preflight` on both configs at commit `d7b5ddfca43aaa39664abf36223324054b6032be`:

| Property | Value |
| --- | --- |
| Planners | 12 |
| Scenarios | 48 |
| Seed set | `eval` -> `111, 112, 113` |
| Rows per horizon | 1728 |
| Rows total | 3456 |
| Scenario matrix hash | `c10df617a87c` |
| Comparability mapping hash | `2321f90648b9` |
| Observation noise hash | `0a5609f0b2b1` |
| Manifest schema | `benchmark-camera-ready-campaign.v1` |

The scenario matrix, comparability mapping, and observation-noise hashes are identical
across both arms; only `horizon` and the derived per-config hash differ. The pair
validator confirms this independently:

```bash
uv run python scripts/benchmark/validate_horizon_ablation_pair.py \
  configs/benchmarks/issue_5409_horizon_ablation_h500.yaml \
  configs/benchmarks/issue_5409_horizon_ablation_h600.yaml --json
# -> "is_valid": true, "mismatch_count": 0
```

## Environment identity

The 12-arm roster needs `rvo2` (ORCA arms) and `stable_baselines3` (`ppo`,
`guarded_ppo`). A partial environment aborts fail-closed in preflight before any episode
runs, so the submit environment must be the full one:

```bash
uv sync --all-extras
```

Reproduce the environment fingerprint for either arm (CPU-only, no episodes):

```bash
uv run python scripts/tools/run_camera_ready_benchmark.py \
  --config configs/benchmarks/issue_5409_horizon_ablation_h500.yaml \
  --mode preflight \
  --campaign-id issue5409_horizon_ablation_h500 \
  --output-root output/benchmarks/camera_ready \
  --checkpoint-preflight-mode enforced_staged \
  --skip-publication-bundle
```

Arm isolation is `subprocess` so GPU memory is fully released between arms.

## Results directory per horizon

Each arm gets its own root; the two must never share one. `{submit_worktree}` is the
operator's dedicated public worktree on the submit host and is the only free variable.

| Arm | `CAMERA_READY_BENCHMARK_CAMPAIGN_ID` | `CAMERA_READY_RESULTS_DIR` |
| --- | --- | --- |
| h500 | `issue5409_horizon_ablation_h500` | `{submit_worktree}/output/benchmarks/camera_ready/issue5409_horizon_ablation_h500` |
| h600 | `issue5409_horizon_ablation_h600` | `{submit_worktree}/output/benchmarks/camera_ready/issue5409_horizon_ablation_h600` |

## Pre-`sbatch` checkpoint gate (still outstanding)

The gate is a **submit-node** command, not a compute job: it materializes and
checksum-verifies every arm checkpoint in seconds and fails closed before `sbatch`. It
must run on the submit host because the receipt binds that host's durable model cache,
so it cannot be produced from a developer workstation.

A resolvability-only probe at the packet commit passes for all 4 checkpoint references
but is explicitly **not** submit-safe: registry-backed arms report
`status=stageable_remote`, and only a staged, checksum-verified receipt counts.

Run once per horizon on the submit host, before either `sbatch`:

```bash
scripts/benchmark/submit_camera_ready_checkpoint_gate.sh \
  --config configs/benchmarks/issue_5409_horizon_ablation_h500.yaml \
  --report-path "$CAMERA_READY_RESULTS_DIR/preflight/checkpoint_staging.json"

scripts/benchmark/submit_camera_ready_checkpoint_gate.sh \
  --config configs/benchmarks/issue_5409_horizon_ablation_h600.yaml \
  --report-path "$CAMERA_READY_RESULTS_DIR/preflight/checkpoint_staging.json"
```

Each must report `submit-safe=true` (exit 0). Exit 3 means an arm checkpoint is
unresolvable: stage or promote it, do not submit.

## Artifact manifest

Per arm, all of these must exist for the horizon to count:

- `campaign_manifest.json`
- `preflight/validate_config.json`
- `preflight/checkpoint_staging.json`
- `reports/matrix_summary.json`
- `reports/comparability_matrix.json`
- `reports/amv_coverage_summary.json`
- `reports/campaign_table.csv`

Each must carry `generating_commit`, `config_sha256`, `scenario_matrix_hash`,
`resolved_seeds`, and `invoked_command`.

Paired, on matched `(planner_key, scenario_id, seed)` keys:

- `matched_key_completeness.json`
- `paired_horizon_deltas.json`
- `paired_uncertainty_summary.json`

Raw episode JSONL, checkpoints, and videos stay out of git and live in the durable
external destination; only compact reviewed summaries are promoted to
`docs/context/evidence/issue_5409_horizon_ablation_<run-date>/` after review.

## Fail-closed row policy

Valid row statuses are `native` and `adapter`. Any `fallback`, `degraded`, `unavailable`,
`failed`, `partial`, `not_available`, or `diagnostic_only` row, any hash drift, and any
missing paired row blocks the affected comparison and can never be counted as success
evidence.

## Remaining launch preconditions

The packet's `status_until_run` is
`ready_pending_gate_receipts_and_compute_authorization`. Two things remain, and neither
can be granted by the packet itself:

1. Both `checkpoint_staging.json` receipts, produced on the submit host, reporting
   `submit-safe=true`.
2. Explicit compute authorization from the repository owner. `compute_submit_authorized`
   is `false` and the checker rejects any packet that flips it.

## Evidence tier

A complete, preregistered matrix is nominal benchmark evidence for this fixed ablation
only. It is not paper-grade, and it does not retroactively validate the earlier
confounded h500/h600 tables, which differ on roster, seed budget, and tier composition
at once.
