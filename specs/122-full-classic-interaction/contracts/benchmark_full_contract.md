# Contracts: Full Classic Interaction Benchmark

Purpose: Define programmatic interfaces prior to implementation. These function signatures will guide TDD.

## Module Namespace
`robot_sf.benchmark.full_classic`

## Public Dataclasses (subset)
- `BenchmarkConfig` (see data-model)

## Functions

### 1. load_scenario_matrix
```python
def load_scenario_matrix(path: str) -> list[dict]:
    """Load YAML scenario matrix.
    Returns list of raw scenario dicts.
    Raises FileNotFoundError or ValueError on parse/validation failure."""
```

### 2. plan_scenarios
```python
def plan_scenarios(raw: list[dict], cfg: BenchmarkConfig, *, rng: random.Random) -> list[ScenarioDescriptor]:
    """Expand raw scenarios with planned seeds and computed hash fragments.
    Respects cfg.initial_episodes for seed planning; seeds deterministic via rng.
    Validation: map existence, required keys. Raises ValueError on issues."""
```

### 3. expand_episode_jobs
```python
def expand_episode_jobs(scenarios: list[ScenarioDescriptor], cfg: BenchmarkConfig) -> list[EpisodeJob]:
    """Create EpisodeJob list (initial plan). Horizon override applied if set."""
```

### 4. run_episode_jobs
```python
def run_episode_jobs(jobs: list[EpisodeJob], cfg: BenchmarkConfig, manifest: BenchmarkManifest) -> Iterator[EpisodeRecord]:
    """Execute jobs (possibly in parallel). Yields EpisodeRecord as they complete.
    Resume: skip jobs whose episode_id already present in existing episodes file."""
```

### 5. append_episode_record
```python
def append_episode_record(path: str, record: EpisodeRecord) -> None:
    """Append JSON line atomically (fsync optional)."""
```

### 6. aggregate_metrics
```python
def aggregate_metrics(records: Iterable[EpisodeRecord], cfg: BenchmarkConfig) -> list[AggregateMetricsGroup]:
    """Compute grouped metrics + bootstrap CIs (group by archetype,density)."""
```

### 7. compute_effect_sizes
```python
def compute_effect_sizes(groups: list[AggregateMetricsGroup], cfg: BenchmarkConfig) -> list[EffectSizeReport]:
    """Produce effect size comparisons within each archetype across densities."""
```

### 8. evaluate_precision
```python
def evaluate_precision(groups: list[AggregateMetricsGroup], cfg: BenchmarkConfig) -> StatisticalSufficiencyReport:
    """Check CI half-width thresholds; include scaling efficiency if measured."""
```

### 9. generate_plots
```python
def generate_plots(groups: list[AggregateMetricsGroup], records: list[EpisodeRecord], out_dir: str, cfg: BenchmarkConfig) -> list[PlotArtifact]:
    """Produce standard plot set; skip advanced plots in smoke mode."""
```

### 10. generate_videos
```python
def generate_videos(records: list[EpisodeRecord], out_dir: str, cfg: BenchmarkConfig) -> list[VideoArtifact]:
    """Create annotated representative videos per archetype unless smoke or missing deps."""
```

### 11. write_manifest
```python
def write_manifest(manifest: BenchmarkManifest, path: str) -> None:
    """Serialize manifest (JSON)."""
```

### 12. run_full_benchmark (Primary Entry)
```python
def run_full_benchmark(cfg: BenchmarkConfig) -> BenchmarkManifest:
    """End-to-end orchestration. Creates directories, loads matrix, planning, resume scan,
    executes episodes (adaptive sampling loop), aggregates, effect sizes, precision check,
    plots, videos, manifest finalize. Returns manifest."""
```

### 13. adaptive_sampling_iteration
```python
def adaptive_sampling_iteration(current_records: list[EpisodeRecord], cfg: BenchmarkConfig, scenarios: list[ScenarioDescriptor], manifest: BenchmarkManifest) -> tuple[bool, list[EpisodeJob]]:
    """Decide whether additional episodes required. Returns (done_flag, new_jobs)."""
```

## Error Handling Contracts
- All public functions raise ValueError for validation issues (not silent). File I/O issues bubble as OSError.
- `run_full_benchmark` catches non-critical plot/video exceptions, recording status in artifacts list instead of raising.

## Logging
- Use `loguru` with INFO for high-level progress (planning, batches, aggregation) and DEBUG for details (per-batch CI stats).

## Determinism
- Accept external master seed; internal RNG objects created locally; no reliance on global random state.

## Test Surfaces (Planned)
- Unit: plan_scenarios validation, aggregate_metrics bootstrap consistency (seeded), effect size formulas, precision evaluation logic (synthetic data).
- Integration: smoke run creates all structural files; resume run skips existing; adaptive loop stops early when thresholds met.

