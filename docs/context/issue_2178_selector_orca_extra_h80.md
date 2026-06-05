# Issue #2178 Selector ORCA-Extra h80 Rerun

Status: current, diagnostic-only; closes the Issue #2176 selector denominator caveat.

Issue #2178 reruns the `selector_only_minus_grouped_static` comparison from the Issue #2170
one-factor manifest after syncing the local worktree with the `orca` extra. The rerun proves
`rvo2` is importable, then executes the same h80/2-worker diagnostic comparison.

## Evidence

- Compact evidence bundle:
  `docs/context/evidence/issue_2178_selector_orca_extra_h80_2026-06-03/`
- Predecessor note:
  [issue_2176_remaining_one_factor_h80.md](issue_2176_remaining_one_factor_h80.md)
- Manifest:
  `configs/policy_search/ablation_manifests/issue_2170_one_factor_hybrid_component_manifest.yaml`

Commands:

```bash
CMAKE_BUILD_PARALLEL_LEVEL=8 uv sync --extra orca
uv run python -c 'import rvo2; print(rvo2.__name__)'
LOGURU_LEVEL=WARNING TF_CPP_MIN_LOG_LEVEL=2 uv run python scripts/tools/run_one_factor_ablation_pilot.py \
  --comparison-id selector_only_minus_grouped_static --horizon 80 --workers 2 \
  --output-dir output/issue_2178/selector_h80_w2
```

## Result

The selector-only rerun completed both candidates with 18/18 rows and zero failed jobs:

| Comparison | Status | Success delta | Collision delta | Near-miss delta | Avg-speed delta | Runtime delta |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| selector_only_minus_grouped_static | ok | 0.000 | 0.000 | 0.000 | -0.096 | -9.865s |

The previous Issue #2176 partial selector row was a local dependency/setup gap, not evidence of an
intrinsic selector failure. With the denominator corrected, the selector-only h80 comparison is
flat on success, collision, and near-miss rate. It reduces average speed by about `-0.096` and runs
about `9.865s` faster than the grouped static comparator in this local run.

## Interpretation

Confidence is about 0.75 that the selector-only component is not independently improving headline
h80 outcome rates on this short-horizon slice. This only fixes the h80 denominator; it does not
settle the h500 component question.

Next research direction: escalate the clean one-factor comparisons to the manifest h500 horizon if
runtime budget allows, because h80 still has a zero-success floor across these rows.

## Claim Boundary

- Diagnostic-only h80 local evidence.
- Not h500 benchmark evidence.
- No planner-promotion claim.
- No selector causality claim beyond the corrected h80 denominator.
