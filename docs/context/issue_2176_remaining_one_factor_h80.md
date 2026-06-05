# Issue #2176 Remaining One-Factor Hybrid Component h80 Comparisons

Status: current, diagnostic-only; selector-only row is partial.

Issue #2176 continues the Issue #2170 one-factor manifest for parent Issue #2104 after the first
Issue #2174 static-escape pilot. It runs the remaining selected h80 comparison rows with the
existing `scripts/tools/run_one_factor_ablation_pilot.py` wrapper, then promotes only the compact
summary under `docs/context/evidence/`.

## Evidence

- Compact evidence bundle:
  `docs/context/evidence/issue_2176_remaining_one_factor_h80_2026-06-03/`
- Manifest:
  `configs/policy_search/ablation_manifests/issue_2170_one_factor_hybrid_component_manifest.yaml`
- Predecessor pilot:
  [issue_2174_one_factor_ablation_pilot.md](issue_2174_one_factor_ablation_pilot.md)

Command:

```bash
LOGURU_LEVEL=WARNING TF_CPP_MIN_LOG_LEVEL=2 uv run python scripts/tools/run_one_factor_ablation_pilot.py \
  --comparison-id static_recenter_only_minus_base \
  --comparison-id escape_recenter_pair_minus_static_escape_only \
  --comparison-id grouped_transit_minus_escape_recenter_pair \
  --comparison-id continuous_checks_minus_grouped_static \
  --comparison-id selector_only_minus_grouped_static \
  --comparison-id speed_progress_2p4_minus_base \
  --horizon 80 --workers 2 \
  --output-dir output/issue_2176/remaining_h80_w2
```

## Result

The clean h80 rows did not move success rate, collision rate, or near-miss rate. Recenter-related
rows increased average speed by about `+0.090` on this slice, while grouped corridor-transit terms,
continuous static checks, and the grouped speed/progress-pressure sensitivity did not produce a
material clean h80 outcome change.

| Comparison | Status | Success delta | Collision delta | Near-miss delta | Avg-speed delta | Runtime delta |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| static_recenter_only_minus_base | ok | 0.000 | 0.000 | 0.000 | +0.090 | +16.263s |
| escape_recenter_pair_minus_static_escape_only | ok | 0.000 | 0.000 | 0.000 | +0.090 | +0.989s |
| grouped_transit_minus_escape_recenter_pair | ok | 0.000 | 0.000 | 0.000 | 0.000 | -0.120s |
| continuous_checks_minus_grouped_static | ok | 0.000 | 0.000 | 0.000 | -0.001 | +1.660s |
| selector_only_minus_grouped_static | partial | 0.000 | +0.022 | -0.144 | -0.041 | -12.594s |
| speed_progress_2p4_minus_base | ok | 0.000 | 0.000 | 0.000 | +0.011 | -0.099s |

Seven candidate rows wrote 18/18 jobs with zero failed jobs. The selector-only candidate wrote
15/18 jobs with 3 failed jobs despite process exit code 0; therefore its apparent near-miss
improvement and runtime reduction are incomplete-row signals, not a valid clean component effect.
The local check `uv run python -c 'import rvo2'` failed with `ModuleNotFoundError`, matching the
selector row's dependency-sensitive failure mode and making the selector rerun a setup task before
it is a research-result task.

## Interpretation

Confidence is about 0.70 that, on this short-horizon local slice, the remaining one-factor static
components are not independently moving headline outcome rates. The evidence is weaker for the
selector-only row because failed jobs change the denominator and may hide harder cases.

This does not close the Issue #2104/Issue #2170 research question. It narrows the next research
direction: either run the manifest horizon h500 rows for clean candidates, or first debug the
selector-only failed jobs so a complete denominator can be compared before any h500 escalation.

## Claim Boundary

- Diagnostic-only h80 local evidence.
- Not h500 benchmark evidence.
- No planner-promotion claim.
- No one-factor causality claim for the partial selector row.
- Raw `output/` files are disposable; this note and the compact evidence bundle are the durable
  review surfaces.

Update 2026-06-03: [issue_2178_selector_orca_extra_h80.md](issue_2178_selector_orca_extra_h80.md)
reruns the selector comparison after syncing the `orca` extra and proving `import rvo2`. The rerun
wrote 18/18 selector rows with zero failed jobs, removing this note's selector denominator caveat.
The corrected h80 selector row remains flat on success, collision, and near-miss rates.
