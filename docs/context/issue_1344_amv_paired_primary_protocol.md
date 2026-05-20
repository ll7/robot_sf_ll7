# Issue #1344 AMV Paired Primary Protocol

Issue: [#1344](https://github.com/ll7/robot_sf_ll7/issues/1344)

Status date: 2026-05-20

## Goal

Define the first rerunnable paired nominal/stress AMV protocol as two camera-ready benchmark
campaign configs that share planner rows, seed policy, metrics, horizon, and kinematics while
changing only the scenario surface.

## Added Surface

- Nominal side: `configs/benchmarks/amv_paired_nominal_primary_v1.yaml`
- Stress side: `configs/benchmarks/amv_paired_stress_primary_v1.yaml`
- Nominal scenarios: `configs/scenarios/nominal_v1.yaml`
- Stress scenarios: `configs/scenarios/classic_interactions_francis2023.yaml`
- Planner rows: `goal`, `social_force`, `orca`
- Seed policy: `seed_set: eval` from `configs/benchmarks/seed_sets_v1.yaml`
- Kinematics: `differential_drive`

This is the primary-row first pass requested by the issue. It does not retune planners or replace
the existing camera-ready evidence bundle.

## SLURM Commands

Preflight each side through the generic launcher:

```bash
CAMERA_READY_BENCHMARK_CONFIG=configs/benchmarks/amv_paired_nominal_primary_v1.yaml \
CAMERA_READY_BENCHMARK_MODE=preflight \
CAMERA_READY_BENCHMARK_LABEL=issue1344-nominal-primary-preflight \
CAMERA_READY_BENCHMARK_OUTPUT_ROOT=output/benchmarks/issue_1344 \
scripts/dev/sbatch_use_max_time.sh --partition a30 --qos a30-gpu \
  --sbatch-arg=--partition=a30 --sbatch-arg=--qos=a30-gpu \
  SLURM/Auxme/camera_ready_benchmark.sl

CAMERA_READY_BENCHMARK_CONFIG=configs/benchmarks/amv_paired_stress_primary_v1.yaml \
CAMERA_READY_BENCHMARK_MODE=preflight \
CAMERA_READY_BENCHMARK_LABEL=issue1344-stress-primary-preflight \
CAMERA_READY_BENCHMARK_OUTPUT_ROOT=output/benchmarks/issue_1344 \
scripts/dev/sbatch_use_max_time.sh --partition l40s --qos l40s-gpu \
  --sbatch-arg=--partition=l40s --sbatch-arg=--qos=l40s-gpu \
  SLURM/Auxme/camera_ready_benchmark.sl
```

Run each side after preflight is clean:

```bash
CAMERA_READY_BENCHMARK_CONFIG=configs/benchmarks/amv_paired_nominal_primary_v1.yaml \
CAMERA_READY_BENCHMARK_MODE=run \
CAMERA_READY_BENCHMARK_LABEL=issue1344-nominal-primary \
CAMERA_READY_BENCHMARK_OUTPUT_ROOT=output/benchmarks/issue_1344 \
scripts/dev/sbatch_use_max_time.sh --partition a30 --qos a30-gpu \
  --sbatch-arg=--partition=a30 --sbatch-arg=--qos=a30-gpu \
  SLURM/Auxme/camera_ready_benchmark.sl

CAMERA_READY_BENCHMARK_CONFIG=configs/benchmarks/amv_paired_stress_primary_v1.yaml \
CAMERA_READY_BENCHMARK_MODE=run \
CAMERA_READY_BENCHMARK_LABEL=issue1344-stress-primary \
CAMERA_READY_BENCHMARK_OUTPUT_ROOT=output/benchmarks/issue_1344 \
scripts/dev/sbatch_use_max_time.sh --partition l40s --qos l40s-gpu \
  --sbatch-arg=--partition=l40s --sbatch-arg=--qos=l40s-gpu \
  SLURM/Auxme/camera_ready_benchmark.sl
```

## Interpretation Boundary

Report nominal competence and stress robustness separately. Nominal success is not safety evidence,
and fallback/degraded rows remain caveats rather than successful benchmark outcomes.
