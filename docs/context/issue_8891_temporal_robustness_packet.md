# Issue #8891 matched-budget temporal-robustness packet

Status: diagnostic, outcome-free, source-bound planning only. The [YAML manifest](../../configs/adversarial/issue_8891_temporal_robustness_packet.yaml) and [read-only validator](../../scripts/validation/check_issue_8891_temporal_robustness_packet.py) add no runtime or campaign path.
The packet does not run a simulator, sampler, search, replay, SLURM job,
registered campaign, or result admission. It makes no benchmark, publication,
empirical-evidence, or scientific-result claim.
## Frozen contract

Inherited #5326/#5304 values are objectives `worst_case_snqi` and
`temporal_robustness`; families `random`, `optuna`, and `cmaes`; budgets `16`,
`32`, and `64`; and search seeds `1101`, `2202`, and `3303`. Coordinate search
is excluded. Cells are cold-started with one sampler instance and ascending
attempt indices. The template retains 120 maximum steps; effective comparison
is 100 steps at `dt_s=0.1`.

The manifest has 54 run identities and 2,016 matched candidate slots. Identity
is lineage, not an outcome. Search, certification, deterministic replay, and
independent confirmation use separate ledgers and technical seed namespaces.
The inherited diagnostic threshold is three native confirmations of five.

The common counter is explicit: search evaluation/failure consumes one slot and
one simulator call; invalid proposals consume one slot and zero calls;
certification consumes zero; replay consumes one; each confirmation consumes
one. Hidden retries, padding, duplicate calls, and post-outcome changes fail
closed.

Temporal sidecars require the five frozen properties (`clearance`, `ttc`,
`goal`, `progress`, `collision`), signed margins, activation times, three gate
states, execution mode, packet/source provenance, and monitor metadata. Monitor
`dt_s` must match evaluation `dt_s`; missing, fallback, degraded, unavailable,
non-native, or monitor-only artifacts are excluded.

The in-memory canary has six disjoint synthetic candidates and 42 planned
simulator invocations. It creates no output and is diagnostic-only.
## Validation

```text
uv run python scripts/validation/check_issue_8891_temporal_robustness_packet.py --check
uv run python scripts/validation/check_issue_8891_temporal_robustness_packet.py --identities
uv run python scripts/validation/check_issue_8891_temporal_robustness_packet.py --canary --check-only
```

Private planning metadata bounds 14,112 maximum simulator invocations and keeps
raw traces out of Git. No scientific conclusion is asserted until a separately
authorized campaign and reproducible evidence pass repository benchmark gates.
