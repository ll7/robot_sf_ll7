# Issue 659 gym-collision-avoidance Headless Reproduction

Date: 2026-03-20
Related issues:
- `robot_sf_ll7#639` fail-fast main-runtime source-harness probe
- `robot_sf_ll7#641` isolated side-environment reproduction
- `robot_sf_ll7#659` headless upstream reproduction

## Goal

Determine whether the upstream `gym-collision-avoidance` example can complete in the isolated side
environment once only non-planner headless blockers are neutralized explicitly.

This issue does **not** rewrite planner logic. It keeps the CADRL / GA3C-CADRL policy path intact and
makes every workaround explicit.

## Canonical probe artifacts

- JSON report:
  `output/benchmarks/external/gym_collision_avoidance_headless_probe/report.json`
- Markdown report:
  `output/benchmarks/external/gym_collision_avoidance_headless_probe/report.md`

Generated with:

```bash
uv run python scripts/tools/probe_gym_collision_avoidance_headless_reproduction.py \
  --repo-root output/repos/gym-collision-avoidance \
  --side-env-python output/benchmarks/external/gym_collision_avoidance_side_env/.venv/bin/python \
  --output-json output/benchmarks/external/gym_collision_avoidance_headless_probe/report.json \
  --output-md output/benchmarks/external/gym_collision_avoidance_headless_probe/report.md
```

## Staged result

Verdict: `headless source harness reproducible`

The staged probe exposed and then neutralized three blockers in order:

1. `TkAgg` on macOS
- upstream `visualize.py` forces `matplotlib.use('TkAgg')` on Darwin
- explicit headless shim: redirect that call to `Agg`
- next blocker exposed: `gym` 0.26 passive checker expects `numpy.bool8`

2. NumPy 2 alias mismatch
- explicit compatibility shim: restore `np.bool8 = np.bool_` only for the headless launcher
- next blocker exposed: legacy `moviepy/imageio` reset animation tries to download `ffmpeg-osx-v3.2.4`

3. Final reset animation export
- explicit headless shim: disable `gym_collision_avoidance.envs.visualize.animate_episode`
- result: upstream example completes and prints:
  - `All agents finished!`
  - `Experiment over.`

## Why this matters

This is the first result in this planner family that is stronger than a blocked import/runtime note.

What is now true:
- the CADRL / GA3C-CADRL policy path can execute through the upstream example loop in an isolated side environment,
- the remaining compatibility surface is explicit and mostly non-planner,
- a wrapper/parity issue is now justified if these shims stay benchmark-visible and are not hidden.

What is still **not** true:
- this is not main-runtime compatibility,
- this is not Robot SF benchmark integration,
- this is not benchmark-quality evidence,
- this is not full zero-shim source parity.

## Compatibility boundary

The headless reproduction uses exactly these explicit shims:

- redirect upstream `matplotlib.use('TkAgg')` to `Agg`
- restore `numpy.bool8` alias for the legacy `gym` passive checker
- disable final `animate_episode(...)` export for headless execution

Interpretation:
- planner logic and checkpoint loading stay upstream,
- the workaround surface is reviewable and narrow,
- but any future wrapper/parity claim must keep these compatibility assumptions explicit.

## Recommendation

Recommendation: `wrapper/parity issue now justified`

Reason:
- the upstream example path is no longer blocked at planner/runtime import,
- the example reaches completion under explicit headless compatibility shims,
- so the next credible step is no longer generic reproduction work.

Recommended next step:
1. open a wrapper/parity issue that treats this side-environment recipe as the source-faithfulness anchor,
2. keep the headless compatibility shims explicit in docs and metadata,
3. only then assess whether the CADRL-family policy is benchmark-strong enough to keep.
