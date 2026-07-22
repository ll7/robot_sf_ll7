# Mutation-testing triage guide (issue #5508)

This document explains how to triage surviving mutants from the bounded
mutation-testing lane and what to do when the CI gate fails.

## How the ratchet works

The script `scripts/dev/mutation_ratchet.py` runs `mutmut` on the configured
source paths (`robot_sf/research/aggregation.py`), collects all surviving
mutants, and compares them against the committed baseline at
`scripts/validation/mutation_baseline.json`.

- **No new survivors**: the job passes.
- **New survivor not in the baseline**: the job **fails** with a list of the
  regressing mutant IDs.
- **Fewer survivors than baseline**: advisory notice only; baseline can be
  refreshed with `--write-baseline` to lock in the improvement.

## When the gate fails

If the workflow reports **new un-baselined survivors**, you have two options:

### Option A — Kill the new mutants (preferred)

1. Inspect each new mutant with:
   ```bash
   uv run mutmut show <mutant-id>
   ```
2. Read the diff to understand which code transformation survived.
3. Add or update tests in `tests/research/test_aggregation.py` that cover the
   mutated logic.
4. Re-run the ratchet locally:
   ```bash
   uv run python scripts/dev/mutation_ratchet.py --check
   ```
5. Iterate until all new mutants are killed.

### Option B — Refresh the baseline (intentional increase)

If the new survivors are acceptable (e.g., they correspond to refactored code
that is intentionally not covered by unit tests), refresh the baseline:

```bash
uv run python scripts/dev/mutation_ratchet.py --write-baseline
```

Commit the updated `scripts/validation/mutation_baseline.json` and document
why the new survivors are tolerated in the commit / PR description.

## Interpreting mutants

Use `mutmut show` to inspect any surviving mutant:

```bash
uv run mutmut show robot_sf.research.aggregation.x_aggregate_metrics__mutmut_18
```

This prints a diff of the mutation applied to the source code. Common patterns:

| Mutation pattern | What it tests | Why it might survive |
|---|---|---|
| `==` → `!=`, `>` → `>=` | Boundary conditions | Missing edge-case tests |
| `and` → `or`, `not` removal | Boolean logic | Missing combinatorial coverage |
| `None` → `not None` | Optional handling | Missing None-path tests |
| Number literal change | Arithmetic | Missing accuracy/fixture tests |
| String mutation | String comparison | Missing exact-match assertions |

## Local workflow

```bash
# Full run + check against baseline
uv run python scripts/dev/mutation_ratchet.py --check

# Aggregate-only (print survivors without checking baseline)
uv run python scripts/dev/mutation_ratchet.py --aggregate-only

# Refresh baseline after reducing survivors
uv run python scripts/dev/mutation_ratchet.py --write-baseline

# Inspect a specific mutant
uv run mutmut show robot_sf.research.aggregation.x_aggregate_metrics__mutmut_18
```

## Avoiding flaky results

- Always run mutmut with the same seed if using random-dependent tests.
- Run from a clean worktree (no uncommitted changes) to avoid skewing results.
- The CI workflow uses `--frozen` sync and the pinned setup-ci-python action.
