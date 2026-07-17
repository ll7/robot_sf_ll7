<!-- AI-GENERATED (robot_sf#5034, 2026-07-16) - NEEDS-REVIEW -->
# Issue #5034 Control-action-latency sweep evidence 2026-07-16

Plain-language summary: job 13516 completed the declared 7,344-episode fidelity campaign, and
this bundle registers the exact-scope latency subset plus a Social Navigation Quality Index (SNQI)
robustness analysis. The result is internal and diagnostic-only, not paper-facing evidence.

- Schema: `control-action-latency-sweep-evidence-promotion.v2`
- Job: `13516` (`5034c-issue5034-latency-sweep`)
- Execution commit: `c153848d7be2851b5c5e89c11055bf96ea778a84`
- Bundle-generator Git head: `484d3fd05a0e29da9e267fa18f817a1fe101de70` (provenance
  mismatch caveated below)
- Raw rows: `ignored_output/slurm/5034c-issue5034-latency-sweep-job-13516/campaign/raw/episode_rows.jsonl`
- Raw-row SHA-256: `6b34e690dfe6cc1ccccd9cd19bde8b3f6a3501bbc1b0a0b44639e151557b4134`
- Preflight decision: `ready`
- Evidence tier: `targeted smoke`
- Result classification: `diagnostic-only`
- Distance convention: `surface_clearance`
- Evidence boundary: **internal fixed-scope latency-sensitivity diagnostic, not paper-facing;
  native claims apply only to native rows, adapter rows remain explicitly labeled diagnostics, and
  fallback/degraded cells are caveats, not successes.**
- Promotion boundary: reads the completed fidelity-campaign rows, isolates the
  `control_action_latency` axis, and reports latency metadata plus metrics. It makes no
  simulator-realism or sim-to-real claim.

## Scope

- Full campaign: `7344/7344` unique episode rows across `153` run cells and `48` scenarios.
- Latency rows: `1296/1296` unique expected rows (missing `0`, extra `0`, duplicate `0`).
- Planners: `baseline_social_force, hybrid_rule_v0_minimal, orca`
- Execution modes: `baseline_social_force` native; `hybrid_rule_v0_minimal` and `orca` adapter.
- Seeds: `111, 112, 113`
- Latency-step coverage: required `[0, 1, 3]`, observed `[0, 1, 3]`, missing `none`
- Fixed-scope coverage: `verified` (1296/1296 expected rows)
- Fallback/degraded/unavailable rows: `0/0/0`

The strict native-only boundary is therefore met only by `baseline_social_force`. ORCA and hybrid
rows remain in the point-estimate ranking as explicitly labeled internal adapter diagnostics; they
are not relabeled as native success evidence.

## SNQI robustness ranking

SNQI means use the repository's canonical `SNQI-v0` implementation with the camera-ready v3
weights and normalization baseline. The slope is an ordinary least-squares fit over 0, 1, and 3
latency steps, expressed per 100 ms-equivalent step. Intervals are paired 95% cluster-bootstrap
intervals over 144 scenario-seed units per planner (10,000 resamples, seed 5892).

| Point rank | Planner group | Mode | SNQI at 0 ms | SNQI at 100 ms | SNQI at 300 ms | Slope / 100 ms | 95% interval |
|---:|---|---|---:|---:|---:|---:|---:|
| 1 | `default_social_force` | native | -0.179167 | -0.181555 | -0.187970 | -0.002974 | [-0.009792, 0.003717] |
| 2 | `hybrid_rule_v0_minimal` | adapter | -0.140104 | -0.155184 | -0.157923 | -0.005287 | [-0.011916, 0.001002] |
| 3 | `orca` | adapter | -0.088809 | -0.095167 | -0.128615 | -0.013762 | [-0.019933, -0.007928] |

The point-estimate order is social force, hybrid, then ORCA for resistance to added latency. Only
the social-force versus ORCA slope difference clears the paired 95% interval. Social force versus
hybrid is not separated, and hybrid versus ORCA narrowly remains uncertain. ORCA is the only
planner whose individual degradation interval excludes zero.

SNQI input caveat: the campaign did not emit `force_exceed_events` or `jerk_mean`, so the canonical
SNQI-v0 neutral defaults apply to those terms. Near-miss event counts were exactly recovered from
`near_miss_rate * steps` up to floating-point error.

## Aggregate metrics per latency cell

| Planner | Latency steps | Latency ms | Cells | Success | Collision | Min clearance |
|---|---:|---:|---:|---:|---:|---:|
| `baseline_social_force` | 0 | 0.0 | 144 | 0.0347222 | 0.472222 | 1.5732588488908965 |
| `baseline_social_force` | 1 | 100.0 | 144 | 0.0277778 | 0.465278 | 1.7255341168433949 |
| `baseline_social_force` | 3 | 300.0 | 144 | 0.0138889 | 0.451389 | 2.255336107663661 |
| `hybrid_rule_v0_minimal` | 0 | 0.0 | 144 | 0.305556 | 0.354167 | 2.031266687156269 |
| `hybrid_rule_v0_minimal` | 1 | 100.0 | 144 | 0.263889 | 0.388889 | 1.543325596837809 |
| `hybrid_rule_v0_minimal` | 3 | 300.0 | 144 | 0.25 | 0.402778 | 1.5002385047372302 |
| `orca` | 0 | 0.0 | 144 | 0.430556 | 0.375 | 1.7715113285911053 |
| `orca` | 1 | 100.0 | 144 | 0.402778 | 0.409722 | 1.7827716018461235 |
| `orca` | 3 | 300.0 | 144 | 0.277778 | 0.513889 | 1.8605650872433952 |

## Exclusions and caveats

- Excluded fallback/degraded/unavailable rows: `0`
- Adapter rows: `864` within the latency subset; retained only as labeled internal diagnostics.
- Provenance: the generated promotion packet recorded a different Git head from the job execution
  context. The execution commit above is anchored by the checksummed fixed-scope plan; the mismatch
  prevents stronger provenance claims.

Per the issue #691 benchmark fallback policy, excluded rows never contribute to the result metrics above.

## Reproducible SNQI derivation (issue #5912)

The canonical analyzer command
`scripts/benchmark/analyze_control_action_latency_snqi.py` derives `snqi_analysis.json` and
`snqi_by_latency.csv` from a **durable sufficient input** rather than the private raw JSONL. The
input `snqi_latency_inputs.csv` carries exactly the per-episode SNQI-v0 terms (success,
time-to-goal, collisions, near-miss rate x steps, comfort exposure) for the 1,296 latency cells;
its provenance sidecar `snqi_latency_inputs.csv.provenance.json` anchors it to the job 13516 raw
rows by SHA-256 `6b34e690...`. A fresh checkout can rerun:

```bash
uv run python scripts/benchmark/analyze_control_action_latency_snqi.py \
  --verify-against docs/context/evidence/issue_5034_control_action_latency_sweep/snqi_analysis.json
```

The analyzer validates the input checksum, the complete fixed-scope cross-product (3 planner
groups x 3 latency steps x 3 seeds x 48 scenarios = 1,296 cells), and that no fallback / degraded /
unavailable / non-native row enters the result set before computing SNQI-v0 per episode, the
per-unit ordinary-least-squares latency slope, and the paired cluster-bootstrap uncertainty.

Reproducibility contract: SNQI-v0 point estimates (per-planner means, deltas, slopes) are
deterministic and reproduce the registered packet to within `1e-9` (observed ~2e-16). The
pairwise slope differences, bootstrap percentile endpoints, and posterior probabilities are
Monte-Carlo / second-code-path quantities that reproduce within their documented tolerances but
are not byte-identical, because the registered packet's generating code was never committed and
its uncertainty block is internally inconsistent (the registered pairwise `slope_difference` does
not equal the difference of its own per-planner slopes, and the probabilities include half-integer
counts such as `0.68635 = 6863.5/10000`). This committed analyzer is the canonical deterministic
generator going forward.

## Files

- `summary.json`: full promotion packet (aggregate + per-cell + exclusions).
- `per_cell_metrics.csv`: compact per-cell latency metrics table.
- `snqi_analysis.json`: exact-scope verification, SNQI method, slopes, uncertainty, and caveats.
- `snqi_by_latency.csv`: compact planner-by-latency SNQI table.
- `snqi_latency_inputs.csv`: durable sufficient input for the SNQI analyzer (issue #5912).
- `snqi_latency_inputs.csv.provenance.json`: provenance anchoring the input to the raw rows.
- `manifest.sha256`: checksums for promoted compact artifacts.
- `README.md`: this human-readable summary.
