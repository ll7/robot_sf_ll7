# Predictive braking development pair

[Back to Documentation Index](README.md)

Issue #10235 compares hybrid v4 with its existing present-position braking and
the opt-in predictive stopping tube introduced in #10228. The hybrid is a
rule-based planner, not a learned actor. Predictive braking is not implemented
for PPO or guarded PPO. The separately named **0.1.0 companion campaign** is
`configs/benchmarks/paper_experiment_matrix_v2_h600_s30_benchmark_data_v0_1_0_predictive_braking_v1.yaml`.
It keeps the original candidate as `hybrid_v4_default` and uses
`hybrid_v4_predictive_braking` for the opt-in copy. All parameters and scenario
overrides match except `v4_predictive_braking_enabled: true` and the explicit
experimental `v4_prediction_speed_error: 0.2` m/s allowance.

The original 0.1.0 learned-policy campaign and both default hybrid config files
remain byte-identical. Its pinned numerical-mode allowlist excludes hybrid;
this companion does not relax that policy or claim pinned learned inference.
The companion uses the same authored scenario matrix and horizon schedule,
differential drive and development seeds 1001–1030. Frozen files are read as
inputs and remain unchanged. This extends development comparisons, without
promoting an arm or changing released scenario behaviour.

## Run and analyze

Run large comparisons on allocated compute. Set `OMP_NUM_THREADS=1`,
`MKL_NUM_THREADS=1` and `OPENBLAS_NUM_THREADS=1` before every command. Use at most
four workers. The normal campaign runner can load the companion roster; the
native diagnostics runner additionally records the full-window prediction audit
and its sampled trajectories. Standard campaign episode metrics alone are
insufficient for this analysis, which fails on missing diagnostics.

```bash
scripts/dev/run_worktree_shared_venv.sh -- uv run python \
  scripts/validation/run_predictive_braking_diagnostics.py \
  --campaign configs/benchmarks/paper_experiment_matrix_v2_h600_s30_benchmark_data_v0_1_0_predictive_braking_v1.yaml \
  --seeds $(seq 1001 1030) --scenarios classic_station_platform_medium \
  --horizon 600 --workers 4 --output output/predictive_pair/station

scripts/dev/run_worktree_shared_venv.sh -- uv run python \
  scripts/analysis/analyze_predictive_braking_pair.py \
  --episodes output/predictive_pair/station/episodes.csv \
  --output output/predictive_pair/station/analysis

# Behaviour gate: all authored scenarios, unchanged authored budgets, paired dev seeds.
scripts/dev/run_worktree_shared_venv.sh -- uv run python \
  scripts/validation/run_predictive_braking_diagnostics.py \
  --campaign configs/benchmarks/paper_experiment_matrix_v2_h600_s30_benchmark_data_v0_1_0_predictive_braking_v1.yaml \
  --seeds 1001 1002 --empty --workers 4 --output output/predictive_pair/empty
```

The station 600-step diagnostic override matches the predecessor comparison;
the companion's normal campaign retains authored horizons (station: 650 steps).
Do not mix these budgets in one comparison. For a smaller smoke, supply explicit
dev seeds to both the runner and analyzer. Every failed gate episode must be
classified separately; identical actor-free timeouts in both arms establish
an inherited failure, not physical infeasibility.

The runner emits episode outcomes, a manifest with source HEAD and configuration
hashes, and compressed arrays in `traces/`. Each episode records its trace name
and SHA-256. Arrays `robot`, `positions`, `velocities`, `dt_s` and
`speed_error_m_s` permit recomputation with `audit_prediction_windows`.
Preserve the complete bundle outside disposable worktree output before delivery.
The analyzer requires each input's adjacent `manifest.json`, complete named
pairs for every declared scenario/seed and a matching producer/run contract.

## Metric contract and interpretation

`paired.csv` and `paired.md` show, per scenario, both success and contact
proportions, near-miss onsets, near-miss exposure in seconds and the 2 s
prediction-bound violation rate. Success gain and added near-miss cost appear
side by side. Contacts count episodes with any observed robot, pedestrian or
obstacle contact; success requires route completion. Near misses count
false-to-true transitions of the native near-miss state, with exposure equal
to the number of exposed steps times the simulator step duration. Native
near-miss state uses non-contact surface clearance below 0.50 m.

The versioned audit is `2s_all_sampled_offsets_euclidean_nearby_2m_v1`. A window
starts at every recorded frame for each pedestrian initially within 2 m centre
distance of the robot. It must have a full 2 s future. At **every sampled offset**
the audit compares actual pedestrian position against start position plus start
observed velocity times elapsed time. A residual norm greater than the configured
speed-error allowance times elapsed time (plus 1e-9 m roundoff tolerance) marks
that window violated. The rate is violated pedestrian-windows divided by all
eligible pedestrian-windows, pooled across episodes. The CSV retains both counts.
No eligible windows means an undefined rate (blank CSV / `n/a` Markdown), never
an observed zero rate. Incomplete terminal windows are excluded; no extrapolation
is performed. Changing actor-row counts or nonfinite traces fail rather than
guess identities. Respawns in stable simulator rows remain included as prediction
discontinuities.

This full Euclidean tube audit includes violations in any direction. It is
stricter than a toward-robot-only projection, and its rates must not be equated
with that earlier refute diagnostic. Frames use simulator-observed velocities,
not finite-difference replacements. Overlapping windows and nearby actors are
correlated; they are not independent trials. The 2 s horizon is a requested
diagnostic and can be shorter than a complete maximum-speed reaction and stop.

**Predictive braking is not a safety guarantee.** Sampled tube coverage cannot
prove continuous-time prediction coverage, zero observed contacts cannot prove
safety, and progress gain cannot erase higher near-miss cost. Wilson 95%
intervals in the CSV describe observed success/contact proportions, not paired
statistical significance or population safety. Fallback/degraded rows, mixed
settings, missing fields, duplicate/unmatched pairs and non-development seeds
are rejected. These outputs are diagnostic-only development evidence; release,
paper admission, allowance calibration and a near-miss budget remain separate.

The [development diagnostic record](validation/predictive_braking_pair/README.md)
preserves the paired station slice, actor-free gate, producer manifests and
individual failure classifications. Its full Euclidean bound rates retain their
arm-specific nearby-window denominators and must not be treated as population
safety probabilities.
