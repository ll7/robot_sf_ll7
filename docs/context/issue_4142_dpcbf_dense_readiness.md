# Issue #4142 dense DPCBF comparison readiness surface

**Status:** diagnostic-only readiness/preflight surface. No campaign run, no Slurm/GPU
submission, no safety-performance or collision-reduction claim.

## What this adds

Issue #4142's 2026-07-02 campaign gate note requires the canonical dense-comparison inputs
to be predeclared and reviewable *before any campaign can be authorized*, with
fallback/degraded rows treated as caveats rather than success evidence. The comparison
packet `configs/research/issue_4142_dpcbf_dense_comparison_v1.yaml` predeclared those
inputs, but no code validated it — nothing guaranteed the three CBF arms stayed
predeclared, distinct, and fail-closed, that the referenced adapter/scenario configs
existed, or that the fallback exclusion stayed in force.

This slice closes that gap:

- Tightened the packet with a structured `canonical_command` field and an explicit
  `summary_contract.excluded_row_statuses` fail-closed exclusion list
  (`fallback`, `degraded`, `failed`, `ineligible`).
- `robot_sf/benchmark/issue_4142_dpcbf_dense_readiness.py` — a read-only readiness surface
  that reuses the canonical CBF runtime validator
  (`robot_sf.benchmark.cbf_safety_filter_runtime.runtime_config_from_mapping`) as the
  single source of truth for arm semantics, cross-checks each arm's adapter config against
  its runtime variant, confirms the scenario manifest exists, and enforces the fallback
  exclusion.
- `scripts/tools/check_issue_4142_dpcbf_dense_readiness.py` — a thin CLI (`--format
  markdown|json`, `--fail-on-blocked`) over that surface.
- `tests/benchmark/test_issue_4142_dpcbf_dense_readiness.py` — pins the fail-closed
  contract.

## Status semantics (fail-closed)

- `prerequisites_incomplete` — a required input is missing or invalid (packet/config
  absent or unparseable, an arm rejected by the canonical validator, the three required
  arms not all predeclared and distinct, an adapter config inconsistent with its arm, or
  the fallback exclusion not in force). The campaign must not be authorized.
- `inputs_ready_campaign_gated` — every packet input is present, valid, consistent, and
  fail-closed. The only remaining blockers are the declared downstream gates: no
  packet-consuming runner is wired to schema
  `robot_sf.issue_4142_dpcbf_dense_comparison.v1` yet, and running requires explicit
  human/Slurm authorization. This is the expected healthy state — it confirms the inputs
  are reviewable, **not** that the comparison may run. There is intentionally no
  `ready-to-run` state.

## Reproduce

```bash
# Markdown report against the current checkout.
uv run python scripts/tools/check_issue_4142_dpcbf_dense_readiness.py

# JSON, non-zero exit unless inputs_ready_campaign_gated (CI/preflight gate).
uv run python scripts/tools/check_issue_4142_dpcbf_dense_readiness.py --format json --fail-on-blocked

# Focused tests.
uv run pytest tests/benchmark/test_issue_4142_dpcbf_dense_readiness.py -q
```

At the commit that introduced this surface the tracked packet evaluates to
`inputs_ready_campaign_gated`: all three arms (`cbf_off`, `cbf_collision_cone_on`,
`cbf_dynamic_parabolic_v1_on`) are predeclared, distinct, runtime-valid, config-consistent,
and fallback/degraded rows are excluded — while the dense comparison campaign itself stays
gated.

## Artifact disposition

The CLI writes nothing to disk by default (report to stdout). No `output/` artifacts are
produced or promoted by this slice.

## Related

- Packet: `configs/research/issue_4142_dpcbf_dense_comparison_v1.yaml`
- Runtime arm contract: `robot_sf/benchmark/cbf_safety_filter_runtime.py`
- Prior slices: DPCBF arm (PR #4168), passthrough gate hardening (PR #4231)
- Parent: issue #3948; first CBF slice PR #4139
