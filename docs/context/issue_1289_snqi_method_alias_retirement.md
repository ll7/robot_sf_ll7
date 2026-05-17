# Issue #1289 SNQI Method Alias Retirement

Related issue: <https://github.com/ll7/robot_sf_ll7/issues/1289>

## Goal

The standalone `robot_sf_snqi recompute` CLI carried deprecated `--method` aliases:
`pareto_optimization`, `equal_weights`, and `safety_focused`. These aliases made the CLI surface
larger than the implementation contract and kept old names visible in help output.

## Decision

Retire the deprecated standalone CLI method aliases. `robot_sf_snqi recompute --method` now accepts
only canonical method names:

- `canonical`,
- `balanced`,
- `optimized`.

Deprecated names fail during argparse choice validation with the supported canonical choices in the
error message. The `weights_validation.py` weight-key alias path is intentionally unchanged because
it covers external weight-file keys, not recompute method names.

## Validation

Focused red/green regression test:

```bash
./.venv/bin/python -m pytest tests/test_snqi_cli_method_aliases.py -q
```

Result after implementation:

```text
4 passed
```

The test checks that recompute help lists only canonical method names and that each retired alias
raises argparse exit code `2` with an `invalid choice` message.

## Documentation

- `docs/snqi-weight-tools/README.md` now lists only canonical standalone CLI methods and records
  the retired aliases as a breaking change.
- `CHANGELOG.md` records the Unreleased breaking CLI cleanup.

## Follow-Up Boundary

This issue only retires aliases for the standalone `robot_sf_snqi` CLI. The older
`scripts/recompute_snqi_weights.py --strategy` values are a separate script surface and were not
changed here.
