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

- the policy rollout timestamp, rollout commit, and a SHA-256-pinned policy decision snapshot;
- the fixed `risk-tiered-stale-base.v1` semantics identifier;
- one named `normal_throughput` window with start/end/capture timestamps and SHA-256-pinned source
  snapshots;
- normalized PR records with `ordinary` or `base_sensitive` risk tier, wait type, hold timestamps,
  attribution, and exact-head/base evidence;
- a `red_main_coverage` object that declares complete, unavailable, or unknown coverage, plus
  red-main events whose classification is recomputed from exact head/base evidence; and
- an optional independent compatible pre-rollout baseline with its own evidence status and
  `pre_rollout` window.

Every source snapshot declares `source_kind` as `fixture`, `repository_snapshot`, or
`external_snapshot`. Fixture locators are structurally synthetic and cannot be accepted when the
top-level evidence status says `workflow_observation`. Repository snapshots are locally hashed;
external snapshots must use a content-addressed locator such as `sha256://<digest>`. Unavailable or
unverifiable sources remain unavailable, and digest mismatches invalidate the contract.

The command is input-backed and does not call GitHub, launch compute, change branch protection, or
modify the selected policy. Percentiles use deterministic nearest-rank selection. P50/P95 include
only hold records with a complete timestamp pair and exact evidence supporting stale-base
attribution; every excluded or unknown record is reported in missingness fields. The report includes
the exact input SHA-256 and a deterministic `observations` audit list preserving wait type,
timestamps, duration missingness, attribution reason, and exact evidence.

## Fail-closed states

- `available`: a source-backed current window has records; the pre-rollout baseline may still be
  `not_available` and is reported as such.
- `fixture_only`: the contract and metrics run on a committed synthetic fixture. This is not a
  normal-throughput observation.
- `not_available`: the input, source snapshot, or current window records are missing.
- `invalid_contract`: schema, timestamp, hash, or record validation failed.

An unknown or non-hold stale-base attribution is never included in P50/P95; a non-hold record that
declares stale-base attribution invalidates the contract. Hold timestamps must fall inside the
named window. Missing red-main coverage is reported as `coverage_status=not_available` with an
`unknown` rollback condition, not as an audited zero. Complete empty coverage is an explicit zero;
an unknown event keeps the rollback condition unknown. A red-main event is
`stale_base_attributable` only when the PR head is identical across the recorded CI, review, and
merge evidence and the CI base differs from the merge base. Otherwise it is `not_attributable` or
`unknown`; unknown incidents remain excluded from causal counts. A baseline is `available` only
when its own evidence is observation-grade, its semantics and repository match, and its window ends
no later than rollout. Fixture baselines remain `fixture_only`; post-rollout or incomplete
baselines are explicitly incompatible or unavailable.

## Evidence boundary

The report is workflow evidence only. It does not establish benchmark performance, planner quality,
scientific causality, a paper-facing claim, or a rollback decision by itself. The real observation
remains unavailable until #6272's policy is deployed and a representative source-backed window is
captured. The committed fixture at
`tests/fixtures/stale_base_observation_window.v1.json` exists only for contract proof. Changing its
top-level evidence status cannot promote it: the source-kind boundary and fixed
`evidence_class=workflow_only` report field keep synthetic data out of production or research
evidence. Landing this helper or generating `fixture_only` output does not complete #7261; closure
still requires a representative post-rollout window, durable inputs, a deterministic report, and
maintainer review.
