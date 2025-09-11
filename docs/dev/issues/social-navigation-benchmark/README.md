# Social Navigation Benchmark Initiative

This directory tracks the development of a reproducible, force‑field–aware social navigation benchmark built on top of `robot_sf` + the `FastPysfWrapper`.

> Goal: Provide a standardized suite of crowd navigation scenarios, metrics (including force/comfort measures), a composite Social Navigation Quality Index (SNQI), reference baselines, and scripts to reproduce results for an academic submission.

## 1. Scope (Living Statement)
The benchmark focuses on evaluating **robot navigation policies in dynamic pedestrian crowds** under varying densities, flow patterns, obstacle layouts, and group behaviors. Emphasis is placed on force-derived comfort & safety metrics (beyond simple collision counts) and reproducibility (seeded scenarios + locked dependencies).

### In Scope
- Deterministic synthetic scenario generation (maps + crowd params)
- Force / comfort / proximity / efficiency metrics
- Composite index (SNQI) with ablations & sensitivity
- Baseline planners (social-force planner, RL policy, random, optional ORCA)
- Reproducibility tooling (CLI, config schema, aggregation)

### Out of Scope (initial phase)
- Large-scale user studies
- Full probabilistic Bayesian calibration (kept as stretch)
- Complex multi-modal dataset ingestion beyond simple stats

## 2. Current Status
See `todo.md` for granular task tracking. This README gives conceptual overview; `todo.md` contains the actionable checklist.

### Running episodes, batches, and CLI (new)
You can now generate episode records and run batches directly from Python.

Single episode:

```python
from robot_sf.benchmark.runner import run_episode, validate_and_write
from robot_sf.benchmark.schema_validator import load_schema, validate_episode

scenario = {
	"id": "smoke-uni-low-open",
	"density": "low",
	"flow": "uni",
	"obstacle": "open",
	"groups": 0.0,
	"speed_var": "low",
	"goal_topology": "point",
	"robot_context": "embedded",
	"repeats": 1,
}
record = run_episode(scenario, seed=123, horizon=15, dt=0.1, record_forces=False)

# Validate against the JSON schema and append to JSONL
schema_path = "docs/dev/issues/social-navigation-benchmark/episode_schema.json"
schema = load_schema(schema_path)
validate_episode(record, schema)
validate_and_write(record, schema_path, "results/episodes.jsonl")
```

Batch run with repeats expanded and JSONL output:

```python
from robot_sf.benchmark.runner import run_batch

scenarios = [
	{
		"id": "batch-uni-low-open",
		"density": "low",
		"flow": "uni",
		"obstacle": "open",
		"groups": 0.0,
		"speed_var": "low",
		"goal_topology": "point",
		"robot_context": "embedded",
		"repeats": 5,
	}
]
summary = run_batch(
	scenarios,
	out_path="results/episodes.jsonl",
	schema_path="docs/dev/issues/social-navigation-benchmark/episode_schema.json",
	base_seed=42,
	horizon=50,
	dt=0.1,
	record_forces=False,
	append=False,
)
print(summary)
```

You can also pass a YAML scenario matrix instead of a list:

```python
from robot_sf.benchmark.runner import run_batch
summary = run_batch(
	"docs/dev/issues/social-navigation-benchmark/scenario_matrix.yaml",
	out_path="results/matrix_episodes.jsonl",
	schema_path="docs/dev/issues/social-navigation-benchmark/episode_schema.json",
	base_seed=0,
)
```

Notes
- Each JSONL line is schema-validated and includes `episode_id`, `scenario_id`, `seed`, `scenario_params`, `metrics`, config/git hashes, and timestamps.
- New field `algorithm_metadata` now captures the algorithm name plus its (possibly hashed) configuration for reproducibility. Always emitted by the runner; schema treats it as an object with `algorithm`, `config`, and `config_hash` keys (additional keys allowed for future expansion).
- SNQI weights/baselines can be provided via `snqi_weights` and `snqi_baseline` in both `run_episode` and `run_batch`.
- The current runner models the robot independently of pedestrian dynamics (one-way coupling). Two-way coupling can be integrated later.

CLI usage:

```bash
# Run a batch from a scenario matrix YAML and write JSONL
robot_sf_bench run \
	--matrix docs/dev/issues/social-navigation-benchmark/scenario_matrix.yaml \
	--out results/episodes.jsonl \
	--schema docs/dev/issues/social-navigation-benchmark/episode_schema.json \
	--base-seed 0 \
	--horizon 50 \
	--dt 0.1

# Compute baseline med/p95 statistics for SNQI normalization
robot_sf_bench baseline \
	--matrix docs/dev/issues/social-navigation-benchmark/scenario_matrix.yaml \
	--out results/baseline_stats.json \
	--jsonl results/baseline_episodes.jsonl \
	--schema docs/dev/issues/social-navigation-benchmark/episode_schema.json

# Quick scenario diversity summary (histograms)
robot_sf_bench summary \
	--in results/episodes.jsonl \
	--out-dir results/figs
```

Tip: During development, you can call `cli_main([...])` directly from `robot_sf.benchmark.cli` to avoid installing the console script.

## 3. Directory Layout (Planned / Evolving)
```
./README.md                                # High-level overview (this file)
./todo.md                                  # Master checklist
./scenarios/                               # Scenario config templates (YAML/JSON)
./schemas/                                 # JSON/YAML schemas for validation
./metrics/                                 # Metric calculation modules & tests
./cli/                                     # CLI entrypoint + helpers
./baselines/                               # Baseline adapter implementations
./results/                                 # (Git-ignored) raw JSONL episode outputs
./figures/                                 # Scripts to reproduce paper figures
./analysis/                                # Notebooks / summary scripts (optional)
```

## 4. Key Concepts
- **Scenario Spec:** Parameterized definition (density, flow_type, map_id, group_prob, seeds)
- **Episode Output:** JSON object with metadata + raw + derived metrics
- **Metric Suite:** Core (success, time, collisions) + Comfort (force quantiles, exceedance rate) + Smoothness + Efficiency
- **SNQI:** Weighted sum of normalized metrics; design emphasizes discriminative power & interpretability

## 5. Metrics (Draft)
| Category      | Metric                              | Notes |
|---------------|-------------------------------------|-------|
| Success       | success_rate                        | Binary per episode |
| Efficiency    | normalized_time, path_efficiency    | Normalize by shortest path |
| Safety        | collisions, near_misses             | Distance thresholds |
| Proximity     | min_distance, avg_speed             | Distribution summaries |
| Force/Comfort | force_q50/force_q90/force_max       | Force magnitudes |
| Comfort       | comfort_exposure (time > threshold) | Threshold from nominal baseline |
| Smoothness    | jerk_mean, curvature_mean           | Derived from trajectory |
| Stability     | seed_variance                       | Variation across seeds |

(See `todo.md` for implementation tasks.)

## 6. Composite Index (SNQI) – Draft Idea
```
SNQI = w1*Success - w2*NormTime - w3*CollisionRate - w4*ComfortExposure - w5*ForceExceedRate
```
- All components normalized to [0,1] (or inverted where necessary)
- Weight selection via sensitivity analysis (grid + Pareto filtering)

## 7. Baselines (Initial Set)
| Name        | Type        | Description |
|-------------|-------------|-------------|
| SFPlanner   | Analytical  | Default social-force driven navigation |
| PPOPolicy   | RL          | Pre-trained PPO checkpoint |
| Random      | Heuristic   | Uniform action sampling (sanity lower bound) |
| ORCA (opt.) | Reciprocal  | Classic ORCA/RVO if license/effort allows |

## 8. CLI Vision
```
robot_sf_bench run --suite core --algo SFPlanner --episodes 100 --out results/core_sf.jsonl
robot_sf_bench aggregate --in results/core_*.jsonl --out summary/core.csv
robot_sf_bench figures --config figures/config.yaml
```

## 9. Reproducibility Pillars
- Locked dependencies (`uv.lock`)
- Embedded git hash + config hash in each episode record
- Controlled random seeds (numpy, torch, python `random`)
- Scenario generation idempotent under same seed

## 10. Validation Strategy
1. Unit tests for each metric edge case
2. Sanity episodes: empty crowd, single pedestrian, dense corridor
3. Stability: multi-seed variance < chosen tolerance for stable metrics
4. Discriminative: each baseline differs on ≥2 core metrics in ≥60% of scenarios

## 11. Paper Skeleton (Draft)
1. Introduction (motivation & gap)  
2. Related Work (benchmarks & metrics)  
3. Benchmark Design (scenarios + methodology)  
4. Metrics & SNQI (definitions + rationale)  
5. Experimental Setup (baselines, configs)  
6. Results (tables, Pareto plots, sensitivity)  
7. Ablations & Discussion (metric impact)  
8. Reproducibility & Limitations  
9. Conclusion & Future Work  

## 12. Immediate Next Steps (Echo of `todo.md` Section 13)
- Draft scenario dimension list
- Formalize metric definitions doc stub
- JSON schema prototype for episode output

## 13. Contribution Workflow
1. Pick or add a task in `todo.md` (avoid scope creep in core phases)  
2. Implement / add tests  
3. Update README or schemas if public interface changes  
4. Link PR to `todo.md` entry and mark after merge  

## 14. Stretch Goals (Summary)
- Force divergence / curl measures
- Bayesian calibration of SF parameters
- Risk-aware planner & residual hybrid baseline
- Lightweight dashboard for interactive exploration

## 15. Changelog
- 2025-09-02: Initial README scaffold added.
- 2025-09-08: Added runner usage and batch API docs.
- 2025-09-09: Added CLI summary subcommand and avg_speed metric.

## 16. Maintainers / Contacts
(Add names/emails once ownership is defined.)

---
This README is a living document; keep it concise, push deep detail into `todo.md`, schemas, or inline docstrings.
