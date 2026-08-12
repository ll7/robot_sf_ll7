# Issue #6970 paired-effect retained-row metric contract

> Status: implemented instrumentation and preflight gate; no campaign was submitted and no
> benchmark, safety, or paper-facing result is claimed.

Issue [#6970](https://github.com/ll7/robot_sf_ll7/issues/6970) closes the identifiability gap
between the paired safety-wrapper design and the #4598 report builder. The versioned contract is
[`paired_effect_metric_contract_v1.yaml`](../../configs/benchmarks/paired_effect_metric_contract_v1.yaml)
(`paired_effect_metric_contract.v1`). It fixes the eight retained fields, their exact
`metric_values.<name>` paths, units, raw-versus-normalized representation, value type, emitting
component, and source definition.

## Gate boundary

Camera-ready configs may reference the contract with `retained_metric_contract`. Config loading
validates the referenced file before preflight dispatch. The runner then validates every written
or resumed episode row after the arm finishes. Missing `metric_values`, non-finite values,
boolean masquerades, out-of-range probabilities/rates, and empty output all fail closed. Similar
legacy fields such as `clearing_distance_min`, realized distance, success, or wrapper-on proxy
diagnostics are not aliases for the declared outcomes.

The existing #4830 factorial campaign config and the #3501 research design reference the
contract. The exposure audit command also lists other configs that declare `metric_values` or the
#4598 report builder but do not yet reference this paired contract; those follow-ups are not
silently changed by #6970.

## Validation

```text
scripts/dev/run_worktree_shared_venv.sh -- uv run pytest \
  tests/benchmark/test_paired_effect_metric_contract.py \
  tests/benchmark/camera_ready/test_safety_wrapper_factorial_preflight.py
scripts/dev/run_worktree_shared_venv.sh -- python scripts/benchmark/check_paired_effect_metric_contract.py \
  --contract configs/benchmarks/paired_effect_metric_contract_v1.yaml --audit-configs --json
```

The contract check is instrumentation evidence only. A passing check establishes retained-field
identity; it does not establish a wrapper effect or authorize issue #6971's campaign packet.
Issue #6971 remains downstream and blocked until its own design/cost decision is handled.
