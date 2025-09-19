# Random Baseline

The Random baseline produces simple random actions with optional safety clamps. It’s a lightweight
reference useful for sanity checks and lower-bound comparison.

## Modes

- velocity: emits {"vx", "vy"} in m/s, uniformly sampled and clamped to v_max
- unicycle: emits {"v", "omega"} with v ∈ [0, v_max] and omega ∈ [−omega_max, +omega_max]

Both modes are deterministic when a seed is provided.

## Quick usage

Programmatic (runner API):

```python
from robot_sf.benchmark.runner import run_batch

summary = run_batch(
    scenarios_or_path="configs/baselines/example_matrix.yaml",
    out_path="results/episodes_random.jsonl",
    algo="random",
    algo_config_path="configs/baselines/random.yaml",
    workers=2,
)
print(summary)
```

CLI:

```bash
robot_sf_bench run \
  --matrix configs/baselines/example_matrix.yaml \
  --out results/episodes_random.jsonl \
  --algo random \
  --algo-config configs/baselines/random.yaml
```

List algorithms:

```bash
robot_sf_bench list-algorithms
```

## Configuration

Use `configs/baselines/random.yaml` as a starting point:

```yaml
algorithm: random
params:
  mode: velocity   # velocity | unicycle
  v_max: 1.5
  omega_max: 2.0
  dt: 0.1
  safety_clamp: true
  noise_std: 0.0
```

Notes:
- mode: Selects action space. velocity → {vx, vy}, unicycle → {v, omega}
- v_max/omega_max: action range clamps; safety_clamp enforces the limits
- noise_std: optional Gaussian jitter on sampled actions
- Determinism is controlled by the `seed` passed to the runner (propagates to the planner)

## Behavior and safety

- Deterministic when seeded via the runner or planner constructor
- The benchmark runner applies a small per-step timeout and a final velocity clamp for robustness
- This ensures a random/bad baseline cannot stall multiprocess batches or produce unbounded velocities

## When to use

- Establish a naive lower bound for metrics and SNQI
- Validate scenario setups and pipeline correctness before benchmarking stronger baselines

