# Issue #4981 VecEnv Throughput Adjudication — Resolved Verdict

This record freezes the **first complete, current-HEAD adjudication** of the issue #4981
strict `>3x` in-process rollout-throughput criterion. It resolves the open item left by the
prior slices ("run the current standard-config throughput adjudication and resolve the parent's
strict `>3x` acceptance criterion"). It does **not** duplicate the toolkit built by PRs
#5017, #5123, #5183, #5619, #5639, #5704, #5668, and #5726; it exercises that toolkit end to end.

## Reproduction

```bash
uv run python scripts/validation/run_issue_4981_vecenv_throughput_acceptance.py \
  --run --output-dir output/issue_4981_vecenv_throughput_acceptance
```

The runner requires a clean tracked tree, re-runs the four required fallback/equivalence
preflight tests, then invokes the four-mode comparator with the frozen standard workload and
classifies the result as `met`, `not_met`, or `blocked`.

## Verdict (resolved)

| Field | Value |
| --- | --- |
| status | `not_met` |
| acceptance_met | `false` |
| source_commit | `be888a2beac66c986b5f5ef8dc0ed0884da4a223` |
| source_host | `imech156-u` |
| source_platform | `Linux-6.8.0-87-generic-x86_64-with-glibc2.39` |
| profile_sha256 | `6eacd537b2c695d112359476eccef290bbc9cdb37e7b25b13ba437b0dbcd31c0` |
| preflight | `passed` (4 required pytest nodes) |
| baseline dummy (1 env) | `2623.56` transitions/sec |
| dummy (4 env) | `2840.03` tps — `1.083x` |
| subproc (4 env) | `construction_failed` (tolerated, comparison-only mode) |
| threaded (4 env) | `2597.17` tps — `0.99x` |
| threaded_lidar_batch (4 env) | `2199.79` tps — `0.84x` |
| best in-process candidate | `threaded` at `0.99x` |
| strict `>3.0x` threshold | **not exceeded** |

## Interpretation

The strict `>3x` in-process acceptance criterion is **resolved as not met** on this host at the
current commit. The threaded modes do not exceed the single-environment dummy baseline; the
threaded path is effectively throughput-neutral (`0.99x`) and the coordinated LiDAR-batch path
is slightly slower (`0.84x`) because the barrier synchronization and padded-batch dispatch
dominate the homogeneous LiDAR raycast for this scenario's modest obstacle count.

## Claim boundary

Host-specific CPU implementation-performance acceptance only. This does **not** establish
training quality, navigation-benchmark performance, cross-host or GPU speedup, or any
paper/dissertation claim. The decision is specific to the recorded host and commit; a different
uncontended host may differ.

## What remains

- The strict `>3x` criterion is resolved (not met) but the issue stays open until a maintainer
  decides to either accept the host-specific `not_met` as the engineering outcome or fund a
  further in-process hot-path rework.
- The `subproc` mode consistently fails to construct under this comparator's direct-exec `spawn`
  context; it is correctly tolerated as a comparison-only mode and is not an in-process candidate.
