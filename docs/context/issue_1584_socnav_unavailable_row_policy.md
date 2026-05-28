# Issue #1584 SocNavBench Unavailable Row Policy

Date: 2026-05-28

Related issues:

- <https://github.com/ll7/robot_sf_ll7/issues/1584>
- <https://github.com/ll7/robot_sf_ll7/issues/1456>
- <https://github.com/ll7/robot_sf_ll7/issues/562>
- <https://github.com/ll7/robot_sf_ll7/issues/1134>
- <https://github.com/ll7/robot_sf_ll7/issues/1498>

## Decision

When required SocNavBench control-pipeline assets are unavailable, downstream benchmark reports
should represent the SocNavBench-family row as **unavailable/excluded**:

- `row_status`: `unavailable/excluded`
- `availability_status`: `not_available`
- `benchmark_success`: `false`
- `availability_reason` or `exclusion_reason`:
  `missing_socnavbench_control_pipeline_assets`
- required local asset gate: `third_party/socnavbench/wayptnav_data` plus the schematic assets
  listed in [issue_562_socnav_bench_reentry.md](issue_562_socnav_bench_reentry.md)

This is not a fallback row, not a degraded-success row, and not successful benchmark evidence. It is
an accepted exclusion that lets unrelated benchmark rows proceed while Issue #1456 remains the
restoration gate for SocNavBench assets.

## Interpretation Rules

Use this policy only when the report explicitly accepts missing SocNavBench assets as an exclusion.
Do not use it for unexpected planner crashes, malformed reports, or partial episode execution.

The row may appear in summary tables, release metadata, or issue notes so readers can see that the
planner family was considered. It must not contribute to success, collision, near-miss, SNQI,
runtime, ranking, or paper-facing aggregate evidence. If a tool computes campaign-level evidence
status, accepted unavailable rows keep `benchmark_success=false`; they may support
`evidence_status=partial` only when other executed rows are benchmark-valid.

If an older report schema requires `readiness_status`, keep the primary reader-facing label as
`unavailable/excluded` and preserve `availability_status=not_available`. Do not interpret any
fallback or degraded metadata as successful execution.

## Boundary With Issue #1456

Issue #1456 stays focused on asset restoration:

1. Stage or hydrate the missing SocNavBench assets with provenance.
2. Run `uv run python scripts/tools/prepare_socnav_assets.py`.
3. Re-run the focused probe documented in
   [issue_562_socnav_bench_reentry.md](issue_562_socnav_bench_reentry.md).
4. Re-enter broader benchmarks only after the probe completes without fallback, degraded, failed, or
   not-available status.

Downstream benchmark issues do not need to wait for Issue #1456 when they explicitly exclude
SocNavBench rows under this policy.

## Evidence Checked

- [issue_562_socnav_bench_reentry.md](issue_562_socnav_bench_reentry.md) records the focused
  fail-fast probe and missing local assets.
- [artifact_evidence_vocabulary.md](artifact_evidence_vocabulary.md) says fallback, degraded,
  adapter, and not-available execution modes must be named explicitly, and fallback is not
  benchmark success.
- [issue_691_benchmark_fallback_policy.md](issue_691_benchmark_fallback_policy.md) defines
  `availability_status=not_available` as a non-success benchmark contract result.
- [docs/benchmark_release_protocol.md](../benchmark_release_protocol.md) already distinguishes
  accepted unavailable/excluded rows from benchmark-valid evidence.
- Issue #1456 comments on 2026-05-23 through 2026-05-27 record the accepted split: asset
  restoration remains blocked, while downstream rows can proceed when SocNavBench is marked
  `unavailable/excluded`.

## Validation

Reference checks performed for this note:

```bash
rg -n "availability_status|readiness_status|benchmark_success|availability_reason|scenario_family_breakdown" robot_sf scripts tests docs --glob '!docs/context/evidence/**' --glob '!output/**'
rg -n "row|unavailable|excluded|fallback|degraded|not-available|status" docs/context/issue_*.md docs/context/*policy*.md --glob '!docs/context/evidence/**'
```

No benchmark execution was run because this issue defines reporting policy only. No `output/`
artifacts are required or promoted.
