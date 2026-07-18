# Issue #4981 VecEnv Throughput Adjudication — Current-Head Verdict

This record freezes the complete standard-configuration adjudication of issue #4981's strict
`>3x` in-process rollout-throughput criterion. It exercises the vectorized environment (VecEnv),
LiDAR-batch, comparator, and fail-closed adjudication tooling already present on the accepted
current `main` commit; it does not make a new throughput claim beyond this host-specific
engineering measurement.

## Reproduction

```bash
uv run python scripts/validation/run_issue_4981_vecenv_throughput_acceptance.py \
  --run --output-dir output/issue_4981_vecenv_throughput_acceptance
```

The runner requires a clean tracked tree, reruns the four fallback/equivalence preflight tests,
then invokes the frozen four-mode comparator with five repetitions of 1,000 warmup and 10,000
measured steps per mode.

## Verdict

| Field | Value |
| --- | --- |
| status | `not_met` |
| acceptance_met | `false` |
| source_commit | `af54719e0619b81f35e81bf6f80bf72c593e3c2b` |
| source_host | `imech156-u` |
| source_platform | `Linux-6.8.0-87-generic-x86_64-with-glibc2.39` |
| profile_sha256 | `6eacd537b2c695d112359476eccef290bbc9cdb37e7b25b13ba437b0dbcd31c0` |
| preflight | `passed` (4 required pytest nodes) |
| baseline dummy (1 env) | `2543.46` transitions/sec |
| dummy (4 env) | `2809.80` transitions/sec — `1.105x` |
| subproc (4 env) | `5687.06` transitions/sec — `2.236x` (comparison-only) |
| threaded (4 env) | `2581.85` transitions/sec — `1.015x` |
| threaded_lidar_batch (4 env) | `2185.98` transitions/sec — `0.859x` |
| best in-process candidate | `threaded` at `1.015x` |
| strict `>3.0x` threshold | **not exceeded** |

## Interpretation and remaining work

The strict `>3x` in-process acceptance criterion is not met on this host at the accepted current
commit. The threaded path is effectively throughput-neutral, while coordinated LiDAR batching is
slower for this modest-obstacle workload. The process-isolated `subproc` row is retained for
comparison only and cannot satisfy the in-process acceptance criterion.

The issue remains open for a maintainer decision on whether this host-specific `not_met` result is
the engineering outcome or whether further in-process hot-path work is funded. No campaign,
GPU/Slurm run, training-quality result, navigation-benchmark result, or paper/dissertation claim
is implied.
