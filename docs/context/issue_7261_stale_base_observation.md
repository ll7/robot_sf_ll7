# Issue #7261 stale-base observation contract

Issue #7261 owns the post-rollout observation follow-up for the risk-tiered stale-base policy
selected by #6272. The repository must not infer a latency improvement from a live queue count or
from a PR being merged. This note defines the small, deterministic input contract used to measure
one named normal-throughput window after policy rollout.

## Contract and command

The input is `stale_base_observation_window.v1`, consumed by:

```bash
uv run python scripts/dev/measure_stale_base_policy.py <window.json> \
  --output output/stale_base_measurement.json \
  --markdown-output output/stale_base_measurement.md
```

The input must name:

- the policy rollout timestamp and a SHA-256-pinned policy decision snapshot;
- one `normal_throughput` window with start/end timestamps and SHA-256-pinned source snapshots;
- normalized PR records with `ordinary` or `base_sensitive` risk tier, wait type, hold timestamps,
  attribution, and exact-head/base evidence;
- red-main events whose classification is recomputed from exact head/base evidence; and
- an optional compatible pre-rollout baseline.

The command is input-backed and does not call GitHub, launch compute, change branch protection, or
modify the selected policy. Percentiles use deterministic nearest-rank selection. P50/P95 include
only stale-base-attributed waits with complete hold timestamps; every excluded or unknown record is
reported in missingness fields.

## Fail-closed states

- `available`: a source-backed current window has records; the pre-rollout baseline may still be
  `not_available` and is reported as such.
- `fixture_only`: the contract and metrics run on a committed synthetic fixture. This is not a
  normal-throughput observation.
- `not_available`: the input, source snapshot, or current window records are missing.
- `invalid_contract`: schema, timestamp, hash, or record validation failed.

An unknown stale-base attribution is never included in P50/P95. A red-main event is
`stale_base_attributable` only when the PR head is identical across the recorded CI, review, and
merge evidence and the CI base differs from the merge base. Otherwise it is `not_attributable` or
`unknown`; unknown incidents remain excluded from causal counts.

## Evidence boundary

The report is workflow evidence only. It does not establish benchmark performance, planner quality,
scientific causality, a paper-facing claim, or a rollback decision by itself. The real observation
remains unavailable until #6272's policy is deployed and a representative source-backed window is
captured. The committed fixture at
`tests/fixtures/stale_base_observation_window.v1.json` exists only for contract proof.
