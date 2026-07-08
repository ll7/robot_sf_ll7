# Issue #4826 Camera-Ready GPU Lifecycle

## Source Thread

- Issue: <https://github.com/ll7/robot_sf_ll7/issues/4826>
- PR #4836: <https://github.com/ll7/robot_sf_ll7/pull/4836>
- PR #4846: <https://github.com/ll7/robot_sf_ll7/pull/4846>
- Implementation plan: <https://github.com/ll7/robot_sf_ll7/issues/4826#issuecomment-4912327173>

## Failure Class

Slurm job 13333 (campaign `issue4365_h600_hybrid_vs_orca_s30_run_20260704`, S30 attempt 4, L40S 44.4 GiB) died after 14:52h with:

```
torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 2.00 MiB.
GPU 0 has a total capacity of 44.40 GiB of which 3.31 MiB is free.
Of the allocated memory 43.90 GiB is allocated by PyTorch
```

This is a distinct failure class from checkpoint preflight failures (#4620, #4663). Checkpoint preflight validates that required checkpoints exist before the campaign starts. Issue #4826 describes a runtime leak: VRAM accumulates monotonically as `run_camera_ready_benchmark.py` iterates arms because models/policies from completed arms are never released.

A 2 MiB allocation failing on a 44 GiB card after 14 hours means the leak is severe and cross-arm cleanup is mandatory for long campaigns.

## Distinction from Related Work

- **Issue #4520 / PR #4528**: Fixed GPU VRAM leaks in the serial map-runner path (`_serial_execute_map_jobs` in `robot_sf/benchmark/map_runner_batch_runner.py`). Issue #4826 is the camera-ready campaign equivalent, affecting `run_camera_ready_benchmark.py` and `robot_sf/benchmark/camera_ready/campaign.py`.
- **Checkpoint preflight (#4620, #4663)**: Validates that required checkpoints exist before the campaign starts. Issue #4826 addresses runtime leaks that occur during campaign execution, after all checkpoints are verified present.

## Implementation Status

### Phase 0: CPU-safe telemetry abstraction
Status: Complete (PR #4836)

A `cleanup_gpu_memory_between_arms()` helper in `scripts/tools/run_camera_ready_benchmark.py` provides:
- RSS measurement via `psutil.Process().memory_info().rss`
- CUDA allocated/reserved/max allocated tracking when PyTorch is available
- `torch.cuda.empty_cache()` and `torch.cuda.synchronize()` calls
- Per-arm VRAM high-water mark logging

### Phase 1: Per-arm cleanup in in-process runner
Status: Complete (PR #4836)

`_run_campaign_planner_matrix()` in `robot_sf/benchmark/camera_ready/campaign.py` now calls `cleanup_gpu_memory_between_arms()` after each planner/kinematics variant completes.

### Phase 2: Subprocess isolation per arm
Status: Complete (PR #4846)

Added `--arm-isolation subprocess` CLI flag to `run_camera_ready_benchmark.py`. When enabled, each planner/kinematics arm runs in a dedicated subprocess, guaranteeing that the OS reclaims all GPU memory when the subprocess exits.

Implementation:
- Worker entrypoint runs one arm and writes `summary.json` + `episodes.jsonl` to the planned `planner_dir`
- Parent launches one subprocess per arm, waits, reads summary, and continues
- Parent still owns campaign-level artifacts, aggregation, report generation, and stop-on-failure policy

### Phase 3: PYTORCH_ALLOC_CONF hardening
Status: Complete (PR #4836)

Set `PYTORCH_ALLOC_CONF=expandable_segments:True` in:
- `scripts/tools/run_camera_ready_benchmark.py`
- `robot_sf/benchmark/map_runner_batch_runner.py` (already present from #4528)

This is defense-in-depth against fragmentation, not the primary fix.

### Phase 4: Regression tests
Status: Complete (PR #4836)

Added `tests/benchmark/test_camera_ready_gpu_cleanup.py` with 6 tests:
- `test_cleanup_gpu_memory_between_arms_without_torch`: Cleanup without torch installed
- `test_cleanup_gpu_memory_between_arms_with_torch_cpu`: Cleanup with torch but no CUDA
- `test_cleanup_gpu_memory_between_arms_with_cuda`: Cleanup with full CUDA stack (monkeypatched)
- `test_cleanup_function_is_callable`: Cleanup function existence and structure
- `test_cleanup_metrics_structure`: Cleanup metrics structure
- `test_pytorch_alloc_conf_set_in_slurm_env`: PYTORCH_ALLOC_CONF environment variable

All tests are CPU-safe and use monkeypatching for CUDA paths.

### Phase 5: GPU smoke on S30 attempt 5
Status: Pending (requires GPU access, outside cheap-lane scope)

Required smoke command before S30 attempt 5:

```bash
uv run python scripts/tools/run_camera_ready_benchmark.py \
  --config configs/benchmarks/paper_experiment_matrix_v1_h600_hybrid_vs_orca_s30.yaml \
  --arm-isolation subprocess \
  --max-planners 3 \
  --max-scenarios 1 \
  --output-root output/issue_4826_gpu_lifecycle_smoke
```

Acceptance for smoke:
- At least one learned-policy arm loads
- At least one non-learned arm runs after it
- Telemetry shows CUDA allocated memory returning near baseline or subprocess exit proving isolation
- No Slurm submission from this PR

## Telemetry Fields

Per-arm `resource_lifecycle` telemetry is attached to run entries:

```json
{
  "resource_lifecycle": {
    "before": {
      "rss_bytes": 123,
      "cuda_available": true,
      "cuda_memory_allocated_bytes": 123,
      "cuda_memory_reserved_bytes": 456
    },
    "after_cleanup": {
      "rss_bytes": 125,
      "cuda_memory_allocated_bytes": 0,
      "cuda_memory_reserved_bytes": 400,
      "cuda_max_memory_allocated_bytes": 789
    },
    "cleanup_status": "ok|warning|failed"
  }
}
```

When using subprocess isolation, the `resource_lifecycle` field is emitted by the subprocess before exit.

## No Metric/Claim Semantics Changed

This fix is runtime orchestration/resource lifecycle only. It does not change:
- Planner semantics
- Metrics calculation
- Row classification (fail-closed policy unchanged)
- Campaign configs
- Evidence interpretation

## Verification Commands

```bash
# Run regression tests
uv run pytest tests/benchmark/test_camera_ready_gpu_cleanup.py -q

# Ruff checks
uv run ruff check \
  robot_sf/benchmark/camera_ready/campaign.py \
  scripts/tools/run_camera_ready_benchmark.py \
  tests/benchmark/test_camera_ready_gpu_cleanup.py
```

## Claim Boundary

This fix addresses cross-arm VRAM leaks in the camera-ready campaign runner. It does not claim that the original S30 campaign now succeeds—the actual S30 rerun remains out of scope for the code-only fix PRs. Empirical campaign success evidence requires a separate GPU smoke run (Phase 5) before S30 attempt 5.

## Related Issues

- Refs #4365: S30 attempt 5 blocked on cross-arm VRAM leak fix
- Related to #4520: Serial map-runner GPU VRAM leak fix (different code path)
