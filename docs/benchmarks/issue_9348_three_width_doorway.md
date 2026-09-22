# Three-width doorway comparison (issue #9348)

Application note: SREV-adjacent benchmark slice; a three-width doorway comparison is a controlled
geometry experiment that varies only the free passage width. Terms: a *variant* is one generated
map/scenario pair at a fixed gap width; a *pair* is one shared seed executed across all three
widths; the *oracle* is the planner-free feasibility rollout that must run before any planner.

## What it does

`scripts/validation/run_issue_9348_three_width_doorway_preflight.py` pins the reusable
three-level configuration on top of the issue #6644 geometry family (no second generator):

- **Widths**: 0.8 m (narrower than the 2.0 m collision diameter,
  `infeasible_by_construction`), 2.0 m (equal, `boundary_tangent`), 2.2 m (wider,
  `geometrically_feasible_candidate`) at fixed 1.0 m depth.
- **Radius binding**: nominal 1.0 m from
  `robot_sf.common.robot_defaults.DEFAULT_ROBOT_RADIUS`, audited by #6645 with runtime
  binding by #6641; never inferred from a proxy.
- **Oracle-first**: the planner-free feasibility sweep runs per variant before any planner;
  planner rows stay `not_run` until a separately authorized campaign packet executes them.
- **Pairing**: the pair manifest shares one seed across all three widths per `pair_id` and
  carries configuration hashes now; realization hashes stay `pending_campaign`.

## Command

```bash
uv run python scripts/validation/run_issue_9348_three_width_doorway_preflight.py \
  --out-json output/benchmarks/issue_9348_preflight.json \
  --variants-dir output/benchmarks/issue_9348_variants
```

## Output

- Preflight report (`issue_9348_three_width_doorway_preflight.v1` JSON): baseline checks,
  three variant records with geometry, asset hashes, oracle verdicts, and explicit `not_run`
  planner rows.
- Variant assets (retained only with `--variants-dir`): per-width `variant.svg` and
  `scenario.yaml` with content hashes. Raw runs and videos stay in `output/`.

## Unavailable reasons and limits

- A missing, changed, or hash-mismatched baseline map fails the preflight before any variant
  is interpreted; the historical map is never overwritten.
- Variant scenarios that differ from baseline outside the explained allowlist
  (name/map/seeds/episode-horizon/radius/geometry metadata) fail the automated diff check.
- Widths outside the pinned three require a manifest amendment, not an ad-hoc flag.
- Evidence boundary: diagnostic within-simulator geometry evidence only. No real doorway
  standard, accessibility requirement, general safety claim, or planner ranking follows.
- The full 30-realization campaign and uncertainty-quantified comparison report belong to the
  successor issue; this slice delivers configuration, manifest, checks, and oracle preflight.
