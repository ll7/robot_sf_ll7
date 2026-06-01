# Issue #2012 Failure Mechanism Classifier

Related issue: <https://github.com/ll7/robot_sf_ll7/issues/2012>

## Scope

`failure_mechanism_classification.v1` is a diagnostic classifier for paired fixed-horizon and
long-horizon benchmark episode records. It turns explicit episode metrics, outcomes, and optional
`scenario_cert.v1` records into reviewable mechanism labels without treating aggregate success
changes as causal evidence.

## Inputs

- Paired episode JSONL rows grouped by `scenario_id`, planner id, and seed.
- Default horizons are fixed `100` and long `500`; the CLI exposes overrides.
- Optional scenario certificates block planner attribution when classification is invalid,
  infeasible, dynamically overconstrained, or benchmark eligibility is excluded.

The CLI writes JSON and CSV:

```bash
uv run robot_sf_bench classify-failure-mechanisms \
  --episodes-jsonl output/example/episodes.jsonl \
  --out-json output/example/failure_mechanisms.json \
  --out-csv output/example/failure_mechanisms.csv
```

## Labels

The classifier emits the issue #1056 vocabulary: `time_budget_clean_relief`,
`exposure_enabled_completion`, `safety_regressed_long_horizon`,
`persistent_low_progress_timeout`, `scenario_contract_blocker`,
`unsupported_wait_then_go_hypothesis`, plus the stress-coverage-compatible `collision`,
`near_miss`, `timeout_without_progress`, and `unavailable`.

Rows become `unavailable` when paired records, required metrics, or execution evidence are missing,
or when records are fallback, degraded, failed, partial, or not available. Unavailable rows are not
counted as observed failure-mode coverage.

## Interpretation Boundary

The output is rule-based diagnostic evidence, not causal proof. In particular, the classifier does
not infer wait-then-go behavior from aggregate success alone; trace or video evidence is still
required before making that claim.
