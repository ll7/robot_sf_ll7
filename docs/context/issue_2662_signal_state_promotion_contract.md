# Issue #2662 Signal-State Promotion Contract

Issue: #2662

Status: current schema/trace promotion contract with #2799 runtime denominator smoke; not a
traffic-light compliance or planner-reasoning claim.

Related surfaces:

- Signalized crossing scope: [issue_2474_signalized_crossing_benchmark.md](issue_2474_signalized_crossing_benchmark.md)
- Signal-state proxy smoke: [issue_2564_signal_state_proxy_smoke.md](issue_2564_signal_state_proxy_smoke.md)
- Runtime denominator smoke: [evidence/issue_2799_signalized_runtime/README.md](evidence/issue_2799_signalized_runtime/README.md)
- Waiting/crossing fixture: [issue_2527_waiting_crossing_fixture.md](issue_2527_waiting_crossing_fixture.md)
- Benchmark manifest: `configs/benchmarks/signalized_pedestrian_crossing_issue_2474.yaml`
- Trace implementation: `robot_sf/benchmark/map_runner.py`
- Contract tests:
  `tests/benchmark/test_signalized_crossing_benchmark_manifest.py` and
  `tests/benchmark/test_issue_2527_waiting_crossing_fixture.py`

## Contract States

`proxy_diagnostic` means signal metadata is recorded for trace/report inspection only. These fields
may include signal phase labels, right-of-way flags, intent phase, observation mode, and claim
boundary, but `planner_consumed_fields` must remain empty. Proxy rows cannot enter
signalized-crossing benchmark denominators and must not be described as traffic-light semantics.

`planner_observable` is reserved for an explicit runtime contract. A row must declare
`signal-state-observable.v1`, `planner_observable_signal_state`, `planner_observable` observation
mode, and `benchmark_evidence: true` before signal fields can be treated as consumed by a planner.
Issue #2799 provides the first simulator-backed smoke for this denominator boundary: two observable
rows enter the denominator and the `unavailable`/`proxy_diagnostic` rows remain excluded.

`unavailable` means signal-state metadata is absent. It fails closed with
`signal_state_metadata_absent` and contributes no planner-consumed or benchmark-evidence signal
fields.

## Promotion Requirements

A future signalized-crossing benchmark row must provide explicit signal runtime fields for:

- signal and conflict-zone identity;
- phase, elapsed time, and remaining time;
- robot and pedestrian right-of-way;
- legality state;
- planner observation mode;
- denominator policy for observable, hidden, or motion-only signal tracks.

Until those fields are present through the observable contract, proxy rows remain diagnostic-only
evidence for scenario/trace plumbing. The #2799 runtime smoke exercises the observable contract for
denominator/exclusion semantics, but it still does not prove traffic-light realism, legality
reasoning, or planner performance.

## Validation

Targeted validation:

```bash
uv run pytest tests/benchmark/test_issue_2527_waiting_crossing_fixture.py tests/benchmark/test_signalized_crossing_benchmark_manifest.py -q
uv run ruff check robot_sf/benchmark/map_runner.py tests/benchmark/test_issue_2527_waiting_crossing_fixture.py tests/benchmark/test_signalized_crossing_benchmark_manifest.py
uv run ruff format --check robot_sf/benchmark/map_runner.py tests/benchmark/test_issue_2527_waiting_crossing_fixture.py tests/benchmark/test_signalized_crossing_benchmark_manifest.py
uv run python scripts/validation/check_docs_proof_consistency.py --path docs/context/catalog.yaml
git diff --check
```

## Boundary

This contract is schema and reportability work only. It does not prove signal-legality behavior,
planner forced-waiting reasoning, traffic-signal realism, or planner performance.
