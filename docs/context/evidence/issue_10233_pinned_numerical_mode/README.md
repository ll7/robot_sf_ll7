# Prospective 0.1.0 numerical policy

AI-GENERATED — NEEDS-REVIEW

Diagnostic-only implementation and host cost evidence for the ruling in #10233,
using the CNN/MLP float64 policy measured in #10211. No release admission,
calibration, dissertation claim admission, or universal cross-CPU portability
claim follows from these development runs. No fallback or degraded rows were
included: the empty-world gate found zero such rows.

The new campaign configuration is
`configs/benchmarks/paper_experiment_matrix_v2_h600_s30_benchmark_data_v0_1_0.yaml`.
It runs three learned arms (release-robot PPO, expert PPO, guarded expert PPO)
and goal, social-force and ORCA controls, using dev seeds 1001–1030 and authored
scenario budgets. It is a prospective development campaign; publication export
and SNQI scoring are disabled pending numerical-context calibration. Existing
campaigns, frozen snapshots, release artifacts and the 0.0.8 CPU gate are intact.

`pinned_float64_v1` pins ATen DEFAULT, compatible MKL, Haswell OpenBLAS,
AVX2 oneDNN, one numerical thread, disabled MKLDNN and deterministic Torch
algorithms. CNN, social MLP, policy MLP and action-head inference use cached
NumPy float64 weights. The original observation adapter and Box action clipping
are retained. Attention, unknown layers, non-CPU and stochastic policies fail
closed. The observed BLAS architecture/thread count and Torch flags are checked;
late process initialization is refused. Kernel pinning is process-wide and can
also change control numerical semantics.

The canonical launcher initializes the mode before importing numerical stacks:

```sh
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
uv run python scripts/tools/run_camera_ready_benchmark.py \
  --config configs/benchmarks/paper_experiment_matrix_v2_h600_s30_benchmark_data_v0_1_0.yaml \
  --mode preflight --checkpoint-preflight-mode enforced_staged
```

For direct Python API use, call `bootstrap_numerical_mode("pinned_float64_v1")`
from `robot_sf._numerical_mode` before importing NumPy, Torch or campaign modules.
Campaign/run manifests record the mode and dtype. Learned-arm provenance retains
actual actor and kernel evidence; validation rejects missing evidence, float32,
dispatch drift and false pinned claims. Completed campaigns cross-check every
learned arm before recording run-level success.

## Inference cost

Host CPU: Intel Core i7-12700KF; Python 3.13.14, NumPy 2.4.6,
Torch distribution 2.13.0, Stable-Baselines3 2.9.0.

`inference_cost.json` retains all 27 paired coordinates and five timing blocks per
mode. Three scenarios: classic_cross_trap_medium, francis2023_narrow_doorway,
francis2023_circular_crossing; seeds 1001–1003. Each mode ran in a fresh interpreter
on identical checkpoint-adapted reset observations, 20 warmups then 5×100 calls.
Means below give each of the nine scenario/seed cases equal weight. Simulation,
checkpoint setup, adaptation and guard decisions are excluded. Shared host load
limits extrapolation; these are descriptive costs, not cluster budget estimates.

| Learned arm | Default ms/call | Pinned float64 ms/call | Ratio |
| --- | ---: | ---: | ---: |
| PPO, release robot | 2.5371 | 6.3200 | 2.491× |
| PPO, expert | 2.5332 | 6.2540 | 2.469× |
| Guarded PPO, expert inference | 2.6006 | 6.3993 | 2.461× |

The full NumPy actor matched an independent Torch float64 reference on all
27 cases with maximum absolute error 3.552713678800501e-15. Checkpoint, input
and measurement-source SHA256 digests are retained in the JSON. No cluster
budget is supplied, so campaign feasibility is not inferred from this probe.
The author's reopen condition remains excessive campaign inference cost.

Reproduce capture and run `--mode default` and `--mode pinned` sequentially in
fresh interpreters with `scripts/benchmark/measure_pinned_numerical_mode.py`.
Use the same `--inputs` NPZ file for both and separate `--output` JSON files.
The capture uses the real scenario reset observations and existing adapters.

## Empty-world gate

`empty_world_gate.json` retains every failure coordinate and observed class.
Both modes ran ten scenarios from #10211 × seeds 1001–1003 × six arms:
180 episodes each, 360 total, with at most four episode workers. All expected
coordinates were present; no unavailable arms or degraded rows occurred.
Pinned learned manifests passed validation. Observed failure labels are
termination classifications, not proven root causes.

| Arm | Success | Static geometry contact | Route timeout |
| --- | ---: | ---: | ---: |
| Goal | 14 | 13 | 3 |
| Social force | 24 | 0 | 6 |
| ORCA | 27 | 0 | 3 |
| PPO, release robot | 17 | 13 | 0 |
| PPO, expert | 10 | 20 | 0 |
| Guarded PPO | 24 | 0 | 6 |

Counts agree in both modes. Outcome changes, introduced failures and resolved
failures: zero across all 180 paired coordinates. This three-seed gate does not
cover the transitions at seeds 1006–1030 reported in #10211.
Reproduce with `scripts/benchmark/run_pinned_numerical_gate.py --mode default`
and `--mode pinned` in fresh interpreters, each with a separate `--output`.

## Compatibility and launcher proof

`byte_audit.json`: all 4,720 pre-existing configs, models, frozen snapshots and
release-related inputs audited against base
`77f61d9aaef935f39e848963d36ae25587041a5e`; zero changed bytes. The protected
inventory digest and unchanged legacy resolved-config hash are recorded.
Freeze commit `66f402ba176b13e45210d0da0b2cf20fcdc0cc02` was read only.
No release branch or artifact was written.

`launcher_smoke.json` records successful canonical-launcher execution of all
three learned arms on francis2023_frontal_approach / seed 1001, including final
run-manifest mode and effective context. This checks subprocess propagation,
retained arm validation and final manifest writing, with no publication export.
