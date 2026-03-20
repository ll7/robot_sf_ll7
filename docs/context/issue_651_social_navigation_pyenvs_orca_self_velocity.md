# Issue 651 Social-Navigation-PyEnvs ORCA Self-Velocity Contract Note

Date: 2026-03-20
Related issues:
- `robot_sf_ll7#644` Prototype benchmark-facing Social-Navigation-PyEnvs ORCA integration
- `robot_sf_ll7#649` Validate source parity for Social-Navigation-PyEnvs ORCA integration
- `robot_sf_ll7#651` Prototype corrected self-velocity contract for Social-Navigation-PyEnvs ORCA

## Goal

Correct the self-velocity contract used by `social_navigation_pyenvs_orca` so raw upstream
`ActionXY` parity is not broken by reconstructing self velocity from `heading + scalar speed` when
an explicit planar velocity is available.

## Implementation decision

Robot SF now exposes an explicit world-frame planar self velocity in SocNav structured observations:

- nested field: `robot.velocity_xy`
- flattened field: `robot_velocity_xy`

The `SocialNavigationPyEnvsORCAAdapter` now prefers that explicit planar velocity when present and
falls back to `linear_speed + heading` reconstruction only when no planar velocity is available.

This is a real contract correction, not a hidden heuristic:

- the self-state mapping is now explicit and benchmark-visible
- the projection boundary remains explicit:
  - upstream command space: `ActionXY`
  - benchmark command space: `unicycle_vw`

## Canonical validation commands

Parity probe:

```bash
uv run python scripts/tools/probe_social_navigation_pyenvs_orca_parity.py \
  --repo-root output/repos/Social-Navigation-PyEnvs \
  --output-json output/benchmarks/external/social_navigation_pyenvs_orca_parity_probe/report.json \
  --output-md output/benchmarks/external/social_navigation_pyenvs_orca_parity_probe/report.md
```

Paper-surface rerun:

```bash
uv run python scripts/tools/run_camera_ready_benchmark.py \
  --config configs/benchmarks/paper_experiment_matrix_v1_social_navigation_pyenvs_orca_only.yaml \
  --label issue651_social_navigation_pyenvs_orca_paper_surface \
  --log-level WARNING
```

## Parity result

Observed artifact:

- `output/benchmarks/external/social_navigation_pyenvs_orca_parity_probe/report.md`

Observed verdict:

- `adapter appears source-faithful but benchmark-misaligned`

Key scenario evidence:

- `circular_crossing_hsfm_new_guo`
  - wrapper mean/max `ActionXY` error: `0.0000 / 0.0000`
- `parallel_traffic_orca`
  - wrapper mean/max `ActionXY` error: `0.0000 / 0.0000`
  - oracle-heading mean/max `ActionXY` error: `0.0000 / 0.0000`
  - heading/velocity mismatch steps: `6`

Interpretation:

- the previous `#649` mismatch was a real adapter bug
- after consuming explicit planar self velocity, the adapter matches upstream raw `ActionXY`
  selection on the tested upstream scenarios
- the remaining fidelity gap is no longer in raw upstream policy inference

## Paper-surface result after the fix

Observed campaign:

- `output/benchmarks/camera_ready/paper_experiment_matrix_v1_social_navigation_pyenvs_orca_only_issue651_social_navigation_pyenvs_orca_paper_surface_20260320_133324`

Key `social_navigation_pyenvs_orca` metrics after the fix:

- episodes: `141`
- success: `0.0213`
- collisions: `0.0638`
- runtime: `37.3879s`
- SNQI: `-0.2974`
- projection rate: `0.0000`
- infeasible rate: `0.0000`

Comparison vs pre-fix `#644` paper-surface run:

- success: unchanged at `0.0213`
- collisions: improved from `0.0922` to `0.0638`
- runtime: slower from `31.0010s` to `37.3879s`
- SNQI: worsened from `-0.2907` to `-0.2974`

Comparison vs frozen canonical `orca` row:

- frozen `orca` success: `0.2340`
- frozen `orca` collisions: `0.0426`
- frozen `orca` runtime: `98.2864s`
- frozen `orca` SNQI: `-0.2325`

Interpretation:

- the fix repairs source parity
- it does not turn this planner into a strong benchmark baseline
- the current upstream-backed ORCA variant remains much weaker than the frozen canonical `orca`
  row on success and SNQI, even though it avoids the old projection/infeasibility pattern

## Final verdict

- `corrected but still weak`

What this means:

- the current `social_navigation_pyenvs_orca` entry is now a cleaner planner-zoo prototype
- it is better evidence for source-faithful upstream ORCA inference than the `#644` version
- it still should not be promoted beyond `experimental`

## Recommendation

- keep `social_navigation_pyenvs_orca` as an experimental upstream-backed planner-zoo entry
- treat `#644` paper-surface numbers as superseded by this corrected contract
- do not replace the frozen paper `orca` row with this variant
