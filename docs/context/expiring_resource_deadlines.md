# Expiring-Resource Deadline Feasibility

**Status**: Operational tooling contract for issue #8905 (child of #8819); not benchmark or
paper-facing evidence, and it changes no scheduler, campaign, or evidence state.

A job can finish inside an expiring host, account, quota, licence, or data-route window and still
lose its outputs when retrieval, checksum verification, and preservation are not budgeted. The
checker `scripts/validation/check_expiring_resource_feasibility.py` reads one campaign manifest
(JSON or YAML) and evaluates its **optional** `expiring_resource` block. Manifests without the
block report `contract_absent`, are not applicable, and stay non-blocking, so historical
manifests keep their previous behavior.

## Optional manifest extension

```yaml
expiring_resource:
  schema: expiring_resource_contract.v1
  resource_class: slurm_gpu  # local|slurm_cpu|slurm_gpu|carla_host|multi_host|host|account|cache|licence|data_route
  admission_policy: block            # block (default) | report
  retention_class: durable_required  # durable_required|release_facing|historical|diagnostic|superseded|disposable
  deadline: {kind: known, timestamp: "2026-09-15T06:00:00+02:00", source: ops_window_notice,
             freshness: fresh, evidence_as_of: "2026-09-10T06:00:00Z"}
             # kind known|estimated|unknown|none; timezone-aware timestamp; sanitized slug
             # source; evidence older than 7 days is expired
  runtime: {expected_seconds: 14400, conservative_seconds: 21600, basis: campaign_history}
  queue_start: {status: not_queued}  # not_queued|queued|running|unknown; see start_timestamp fields
  output: {estimate_bytes: 1073741824, basis: episode_rate_measurement}
  reserves: {retrieval_seconds: 3600, verification_seconds: 1800, preservation_seconds: 1800}
  retrieval_throughput_bytes_per_second: 1048576  # optional cross-check
  latest_safe_submission: "..."      # optional declared value, contradiction-checked
```

`deadline.kind` distinguishes known, estimated, unknown, and no declared expiry (`none`, reported
as not applicable). A `queued` job without `expected_start_timestamp` is `unknown`; the checker
never guesses a scheduler start. Durable-required and release-facing retention need non-zero
retrieval and preservation reserves.

## Verdicts, codes, and integration

Exactly one verdict is computed: `fits_conservative` (fits with the conservative runtime and all
reserves), `fits_expected` (fits only with the expected runtime), `too_late`, or `unknown`.
`latest_safe_submission` = deadline - conservative runtime - retrieval - verification -
preservation reserves; the expected variant uses the expected runtime. Unknown, stale, or
contradictory evidence can never produce a positive fit verdict. Stable fail-closed reason codes
cover absent/no-expiry contracts, timestamp and timezone errors, deadline provenance, runtime and
reserve problems, output-size basis, queue-start contradictions, and declared latest-safe
conflicts; see the checker for the full list.

CLI: `uv run python scripts/validation/check_expiring_resource_feasibility.py --manifest <path>
[--case <name>] --check [--json] [--as-of ISO]`. Exit codes: `0` fit or not applicable, `1`
`too_late`, `2` `unknown`/unreadable. JSON is byte-stable and sanitized. The pre-sbatch campaign
gate `scripts/benchmark/preflight_campaign_checkpoints.py` runs this check before checkpoint
staging when the campaign manifest declares the block: `too_late` or `unknown` returns exit `3`
under `admission_policy: block` (the fail-closed default), while `report` records without
blocking.

## Fixtures and validation

`tests/validation/fixtures/expiring_resource_feasibility/cases.json` holds six sanitized cases:
`known_fit`, `conservative_failure` (conservative margin fails, expected fits), `unknown_start`,
`stale_deadline_source`, `zero_reserve`, and `timezone_boundaries` (absolute `+14:00` compare).

```bash
uv run python scripts/validation/check_expiring_resource_feasibility.py \
    --manifest tests/validation/fixtures/expiring_resource_feasibility/cases.json \
    --case known_fit --check --json
uv run pytest -q tests/validation/test_expiring_resource_feasibility.py
```

Non-goals: no scheduler/cancellation change, no fabricated deadline, no campaign scope or
wall-time rewrite, no historical manifest rewrite, no private route or locator disclosure.
