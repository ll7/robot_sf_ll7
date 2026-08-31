# Risk-tiered stale-base merge policy

## Decision

The repository keeps exact-head CI, exact-head review evidence, and the final
compare-and-swap (CAS) head check mandatory. Until GitHub's merge queue is
enabled for the protected `main` branch, ordinary pull requests may avoid a
full branch refresh after unrelated `main` movement. Changes that intersect
the explicit `base_sensitive` test selector still require a current base and
the focused `base_sensitive` test subset.

This is an integration-policy change only. It does not relax branch
protection, required checks, benchmark proof, domain approval, or any
compute/evidence gate.

## Selector contract

`scripts/dev/base_sensitive_selector.py` defines selector
`pytest-marker-files.v2`: a pull request is `base_sensitive` when its changed
file inventory intersects a test file containing the repository's registered
`@pytest.mark.base_sensitive` contract. A complete inventory with no
intersection is `ordinary`. Missing or malformed inventory is `unknown` and
fails closed.

The selector version is shared by the base-sensitive gate and ordinary-CAS
receipt validator. The v2 selector binds discovery to Git-tracked repository
files, excluding ignored nested worktrees and caches.

The selector is evaluated by
`scripts/dev/check_base_sensitive_gates.py --pr <number> --json`. A trusted
exact-head review records the result with one of these trailers:

```text
base-policy: ordinary-cas @ <head-sha>
base-policy: current-base @ <head-sha>
```

The first trailer authorizes only the ordinary CAS route; it is not a merge
authorization and does not waive exact-head review or CI. The second records
that the current-base route was selected and cannot make a stale base pass.

## Pre-merge contracts

For ordinary PRs, immediately before the guarded squash merge,
`scripts/dev/check_pr_current_base_cas.py` must observe the same expected head
SHA and current `main` SHA that were captured for the operation. The merge
still uses GitHub's `--match-head-commit` guard. Head movement, main movement,
unknown PR state, missing provenance, or a non-`main` target fails closed.

For base-sensitive PRs, the guarded merger additionally requires the existing
workflow-run/base freshness check and a passing `--run-subset` invocation of
`check_base_sensitive_gates.py`. A stale base cannot receive `merge-ready`
without the current-base proof; the ordinary trailer is the only bounded
exception and remains subject to the immediate CAS check.

Native merge-queue admission remains the stronger path when configured. The
in-repository queue gate continues to require its current synthetic queue
head and `ALLGREEN` strategy; this policy does not change repository settings.

## Boundary cases

- A changed PR head after CI or review is `stale_worktree` and must be
  re-reviewed.
- A base-sensitive PR with an old CI base is `stale_merge_base` and must be
  refreshed and rerun.
- An ordinary PR with a current exact-head policy trailer can proceed to the
  final CAS preflight even when its declared base is older than `main`.
- Missing current-main, changed-file, review-thread, metadata, or CAS
  provenance remains a fail-closed stop.

## Measurement boundary

The first live queue snapshot used for implementation on 2026-08-17 reported
15 stale and 3 blocked active PR lanes, with no healthy lanes. The compact
historical data available to the workflow does not expose attributable
stale-base hold duration or stale-base-caused red-main incidents, so P50/P95
hold latency and incident deltas are not inferred from this snapshot. A
normal-throughput observation window must record those values before the
policy is treated as empirically validated; this document records the
measurement boundary rather than claiming a throughput result. The bounded
observation task is tracked in
[#7261](https://github.com/ll7/robot_sf_ll7/issues/7261).
