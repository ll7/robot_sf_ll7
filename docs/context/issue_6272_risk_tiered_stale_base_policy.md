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

## Issue #9893 Observation and Retry Boundary (2026-09-28)

A four-PR event sample, captured through 11:55:14 UTC before later queue
updates to these PRs, was inspected to decide whether base-only movement
should invalidate exact-head review or trigger repeated refreshes:
[#9830](https://github.com/ll7/robot_sf_ll7/pull/9830),
[#9827](https://github.com/ll7/robot_sf_ll7/pull/9827),
[#9804](https://github.com/ll7/robot_sf_ll7/pull/9804), and
[#9817](https://github.com/ll7/robot_sf_ll7/pull/9817). All four used base
`17bd03e09ad01529d974d457075339760b6f1e21` (commit time 10:33:04 UTC). Their
latest exact-head reviews were submitted from 11:09:02 to 11:09:12 UTC. The
observed `main` advance to `4665cd13fc205761a4edb256a04d23d37ffbc235` was at
11:20:55 UTC, 11m43s–11m53s after those reviews.

The captured timing and rollup counts were:

| PR | Latest exact-head review (UTC) | First check (UTC) | Last completed check (UTC) | Rollup at capture: success / failed / skipped / unfinished |
| --- | --- | --- | --- | --- |
| [#9830](https://github.com/ll7/robot_sf_ll7/pull/9830) | 11:09:02 | 11:04:24 | 11:51:37 | 30 / 1 / 2 / 1 |
| [#9827](https://github.com/ll7/robot_sf_ll7/pull/9827) | 11:09:06 | 11:04:28 | 11:54:23 | 27 / 0 / 0 / 5 |
| [#9804](https://github.com/ll7/robot_sf_ll7/pull/9804) | 11:09:09 | 11:04:34 | 11:54:12 | 27 / 1 / 0 / 3 |
| [#9817](https://github.com/ll7/robot_sf_ll7/pull/9817) | 11:09:12 | 11:04:40 | 11:55:14 | 22 / 0 / 0 / 10 |

Measured from each latest review to its last completed check, the interval was
42m35s–46m02s (median 45m10s). This is a completion-tail diagnostic only: some
check rows were still unfinished and some had failed. It is not a green CI
duration, a typical CI-wait estimate, a normal-throughput sample, or evidence
that the policy caused a throughput or incident-rate change. The sample does
not satisfy [#7261](https://github.com/ll7/robot_sf_ll7/issues/7261), which
still requires its representative 7–14 day observation window.

The policy decision is to preserve accepted review evidence while the PR head
and final metadata remain unchanged, even if `main` advances. Base-sensitive
validation must still pass on the current base; if refreshing the branch
changes the PR head, the new head requires review. Bound each merge attempt to
one base refresh. A second authoritative stale-base observation is reported as
`stale_base_churn` with both SHAs and the refresh count, then the attempt stops.
This does not alter merge preflight or branch protection. The existing
ordinary-CAS exception remains bounded by exact-head policy evidence and the
immediate current-main compare-and-swap check.
