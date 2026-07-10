# Benchmark Schema Management

[← Back to Documentation Index](./README.md)

## Schema Consolidation

The RobotSF benchmark system uses a consolidated schema management approach to ensure consistency and prevent duplication across the codebase.

> **Looking for runnable examples?** See `examples/benchmarks/demo_full_classic_benchmark.py`
> for a programmatic walkthrough and consult [`examples/README.md`](../examples/README.md)
> for the full benchmarks catalog.

**See also**: [SNQI Weight Tools](./snqi-weight-tools/README.md) for weight recomputation and optimization, and [Distribution Plots](./distribution_plots.md) for visualization guidance.

## Pedestrian Walking-Speed Calibration

**Caveat for interpreting benchmark results.** The default simulated pedestrian
walks at approximately **0.65 m/s** for the whole episode. This is a side effect
of coupling the desired (preferred) walking speed to the spawn speed: the
spawn velocity defaults to 0.5 m/s (`PedSpawnConfig.initial_speed`) and the
goal-driving speed is derived as `peds_speed_mult * initial_speed` (1.3 × 0.5).
This is roughly **half** the ~1.3 m/s preferred walking speed reported for
unimpeded adults (Moussaïd et al. 2010, "The walking behaviour of pedestrian
social groups", doi:10.1371/journal.pone.0010047). Two consequences:

1. The default interaction regime is systematically gentler/slower than typical
   sidewalk traffic, which flatters reactive planners (more time to react).
2. Coupling desired speed to *spawn* speed is surprising: making pedestrians
   walk faster required spawning them faster.

Until a major campaign re-base, the **legacy default is intentionally preserved**
so existing benchmark numbers stay reproducible. Issue #4972 adds a decoupled
*desired-speed* axis and a **speed-tier** selector (`slow` / `typical` / `brisk`)
on `SimulationSettings` (`ped_speed_tier`) and `SceneConfig`
(`desired_speed_mean` / `desired_speed_std`):

- `slow` (~0.65 m/s) reproduces the legacy default as an explicit tier value;
- `typical` (~1.3 m/s) matches the literature preferred walking speed;
- `brisk` (~1.6 m/s) stress-tests reactive planners.

Benchmark reports and comparisons that vary pedestrian pace **must** record the
speed tier or desired-speed distribution alongside results, and should treat any
ranking computed only at the slow default as conditional on that regime. See
`robot_sf/sim/pedestrian_speed_tiers.py` for the tier mapping.

## Local Smoke Benchmark Demo

For a quick local sanity check that the benchmark runner can execute a small map scenario with two
low-cost planners, run:

```bash
uv run python scripts/demo/run_robot_sf_smoke.py
```

The command runs `configs/scenarios/single/planner_sanity_simple.yaml` with `simple_policy` and
`social_force`, then writes disposable local artifacts under
`output/demo/smoke_benchmark/`:

- `summary.json`: machine-readable planner status and aggregate metrics.
- `report.md`: short human-readable result table and artifact pointers.
- `episodes/*.jsonl`: per-planner episode records used to build the summary.

Pass `--verbose` when debugging simulator or map-runner logs.

These outputs are a local demo only. They are not durable benchmark evidence and should not be used
for paper-facing or promotion claims unless promoted separately with the repository's normal
artifact provenance and validation process.

## Tracking-Uncertainty Metrics

Robot SF has a bounded CLEAR Multi-Object Tracking (CLEAR MOT) diagnostic layer for
perception-gap evaluation. ScenarioBelief adapters can compare an oracle belief with a
visibility-limited or tracking-noise belief and emit Multiple Object Tracking Accuracy (MOTA) and
Multiple Object Tracking Precision (MOTP). MOTA counts missed detections, false positives, and ID
switches against the ground-truth detection count. MOTP reports mean matched localization error in
meters.

Tracking uncertainty remains disabled by default. Scenarios opt into synthetic planner-facing
visibility and centroid-offset stress through `observation_visibility.tracking_noise_std_m`; use
`0.0` for exact zero-noise reproduction. Benchmark planner-input vector noise continues to use the
existing `observation_noise` profile path in `robot_sf.benchmark.observation_noise`.

When a noise or perception-limited evaluation records CLEAR metrics, store them under the existing
`metrics.clear_tracking_uncertainty` block with `enabled`, `mota`, `motp_m`, and `counts` fields.
The aggregate/reporting path flattens that block into columns such as `clear_mota`,
`clear_motp_m`, `clear_missed_detection_count`, and `clear_false_positive_count`. These values are
diagnostic unless backed by an explicit perception-gap campaign and provenance.

Validation protocol:

```bash
uv run pytest tests/representation/test_scenario_belief.py \
  tests/test_metrics.py tests/test_aggregate.py
```

## Mechanism-Aware Diagnostic Reproduction

For a bounded one-command reproduction of a mechanism-aware diagnostic case, run:

```bash
uv run python scripts/demo/reproduce_mechanism_report.py --case topology-primary-route
```

The command wraps the topology-hypothesis diagnostic for the `topology_guided_hybrid_rule_v0`
candidate, whose registry `claim_scope` is `diagnostic_only`, on the double-bottleneck route case.
It writes disposable local artifacts under `output/demo/mechanism_report/topology_primary_route/`.
Its claim boundary is `diagnostic_only_not_benchmark_success`: a successful run shows that the
local diagnostic path can expose topology hypotheses for the selected case, not that the planner is
better, benchmark-successful, or paper-grade.

Benchmark outcomes are separate from dense training rewards. Benchmark claims must rely on
schema-checked episode records, deterministic metrics, termination/outcome fields, and explicit
runtime/readiness metadata; training reward totals are not benchmark-success evidence. See
[Robot SF Environment Contract And Training Provenance](./training/environment_contract.md) and the
[benchmark fallback policy](./context/issue_691_benchmark_fallback_policy.md).

For routine, low-stress planner calibration, use `configs/scenarios/nominal_v1.yaml`. This matrix
is intentionally separate from stress, adversarial, and camera-ready surfaces; nominal success is a
sanity check for basic shared-space competence, not safety or robustness evidence.

### Canonical Schema Location

**Canonical schema locations**:
- Episode schema: `robot_sf/benchmark/schemas/episode.schema.v1.json`
- Scenario schema: `robot_sf/benchmark/schema/scenarios.schema.json` (note: `schema/`, singular)

### Runtime Schema Resolution

Use the schema loader for runtime resolution:

```python
from robot_sf.benchmark.schema_loader import load_schema, get_schema_version

# Load episode schema
schema = load_schema("episode.schema.v1.json")

# Get schema version
version = get_schema_version("episode.schema.v1.json")
print(f"Schema version: {version}")  # SchemaVersion(major=1, minor=0, patch=0)
```

### Schema Validation

Schemas are automatically validated against JSON Schema draft 2020-12:

```python
from robot_sf.benchmark.validation_utils import validate_schema_integrity

errors = validate_schema_integrity(schema_data)
if errors:
    print(f"Schema validation errors: {errors}")
```

### Version Management

Schema evolution follows semantic versioning:

```python
from robot_sf.benchmark.version_utils import detect_breaking_changes, determine_version_bump

# Detect breaking changes between schema versions
breaking_changes = detect_breaking_changes(old_schema, new_schema)

# Determine appropriate version bump
bump_type = determine_version_bump(breaking_changes)  # 'major', 'minor', or 'patch'
```

### Git Hook Prevention

Git hooks prevent duplicate schema files from being committed:

```bash
# Pre-commit hook automatically blocks duplicate schemas
git add duplicate_episode_schema.json
git commit -m "Add schema"
# ERROR: Duplicate schema detected: duplicate_episode_schema.json
#        Canonical location: robot_sf/benchmark/schemas/episode.schema.v1.json
```

### Performance Characteristics

Schema loading is optimized with caching:
- First load: <50ms (typical)
- Cached loads: <1ms
- Performance budget: <100ms hard limit

### Migration Notes


## Algorithm Grouping & Aggregation Diagnostics

The classic benchmark aggregates metrics **per algorithm**. To guarantee separation:

  - emits a Loguru warning with `event="aggregation_missing_algorithms"`, and
  - annotates the JSON summary with `_meta.missing_algorithms`, `_meta.group_by`, and `_meta.effective_group_key` (`"scenario_params.algo | algo | scenario_id"`).
  - `event="episode_metadata_injection"` (nested value added) and
  - `event="episode_metadata_mismatch"` (nested value corrected to match top-level `algo`).

## Map Verification (CI Quality Gate)

The benchmark pipeline includes a Map Verification step that validates SVG assets before metrics aggregation. It guards against malformed or poorly organized maps that could skew navigation performance results.

### Why It Matters
Maps encode obstacles, corridors, and spawn semantics. Structural issues (invalid XML, unreadable files, oversized geometry, missing labeled layer groups) silently degrade benchmark comparability. Early detection preserves data integrity.

### Running Verification
CI invocation (excerpt):
```yaml
  - name: Map verification (CI mode)
    run: uv run python scripts/validation/verify_maps.py --scope ci --mode ci --output output/benchmarks/map_verification_manifest.json
```

Local smoke test:
```bash
uv run python scripts/validation/verify_maps.py --scope ci --mode ci --output output/tmp/verify_manifest.json
```

### Rule Set
| Rule | Severity | Description | Remediation |
|------|----------|-------------|-------------|
| R001 | ERROR | File must exist & be readable | Fix path/permissions |
| R002 | ERROR | Must parse as valid XML/SVG | Correct XML syntax, encoding |
| R003 | WARNING | File size > 5 MB | Simplify geometry, remove unused defs |
| R004 | WARNING | No Inkscape-labeled groups found | Add `inkscape:label` to semantic `<g>` groups |
| R005 | INFO | Layer stats (labeled vs total) | Ensure critical semantics have labels |

### Manifest Structure (excerpt)
```jsonc
{
  "run_id": "map_verification_20251120_220354_bb7cc5f6",
  "mode": "ci",
  "scope": "ci",
  "results": [
    {"map_id": "classic_corridor", "status": "warn", "rule_ids": ["R004"], "message": "No labeled layers found"},
    {"map_id": "classic_overtaking", "status": "warn", "rule_ids": ["R004"], "message": "No labeled layers found"}
  ],
  "summary": {"total": 25, "passed": 0, "failed": 0, "warned": 25}
}
```

### Usage Guidance
1. ERROR: Block merge; fix immediately.
2. WARNING: Schedule asset hygiene improvement; does not block benchmarks.
3. INFO: Iterative refinement hints; label more semantic groups over time.

### Extending Rules
Add new checks in `robot_sf/maps/verification/rules.py` (follow existing pattern). Prefer INFO or WARNING unless correctness is compromised.

### Validation Checklist

- Spot-check the first line of `episodes.jsonl`: `record["algo"] == record["scenario_params"]["algo"]`.
- Confirm aggregate outputs include `_meta.effective_group_key` and, when applicable, warnings describing any missing algorithms.
- Treat `AggregationMetadataError` as a signal to regenerate the episode data—legacy files lacking mirrored metadata are no longer accepted silently.

## Parquet Analytics Export

`episodes.jsonl` remains the source of truth for benchmark runs. For larger campaign analysis,
convert it into Parquet tables with the optional analytics extra:

```bash
uv sync --extra analytics
uv run robot_sf_bench export-parquet \
  --in output/benchmarks/classic_interactions/episodes.jsonl \
  --out-dir output/benchmarks/classic_interactions/parquet
```

The export writes:

- `episodes.parquet`: one fixed top-level row per episode.
- `metrics.parquet`: long-form typed metric rows keyed by `episode_id` and dotted `metric_path`.
- `scenario_params.parquet`: long-form scenario parameter rows keyed by dotted `param_path`.
- `algorithm_metadata.parquet`: long-form planner metadata rows keyed by dotted `metadata_path`.
- `metadata.json`: source JSONL hashes, row counts, table files, and export schema version.
- `duckdb_examples.sql`: copy-paste SQL examples for grouped safety metrics and failure mining.

Example DuckDB query:

```sql
SELECT
    e.algo,
    e.scenario_family,
    AVG(CASE WHEN m.metric_path = 'min_ttc' THEN m.value_number END) AS avg_min_ttc,
    AVG(CASE WHEN m.metric_path = 'clearance' THEN m.value_number END) AS avg_clearance
FROM read_parquet('episodes.parquet') AS e
JOIN read_parquet('metrics.parquet') AS m USING (episode_id)
GROUP BY e.algo, e.scenario_family
ORDER BY e.algo, e.scenario_family;
```

Use `--overwrite` only when replacing a known derived export. The metadata file records that the
Parquet files are derived views so downstream reports can trace back to the canonical JSONL input.

### Curated DuckDB Recipes

For repeatable analysis without notebooks, run a curated SQL recipe against the Parquet export:

```bash
uv run python scripts/tools/run_benchmark_sql_recipe.py \
  --recipe planner_outcome_summary \
  --export-dir output/benchmarks/classic_interactions/parquet \
  --output-csv output/benchmarks/classic_interactions/tables/planner_outcome_summary.csv \
  --output-markdown output/benchmarks/classic_interactions/tables/planner_outcome_summary.md
```

Recipes live under `scripts/tools/benchmark_sql_recipes/` with a manifest that records stable
recipe IDs, required tables and columns, output columns, and caveats. The runner validates required
Parquet files and columns before executing SQL so schema drift fails closed instead of producing
misleading zeros.

Initial recipes:

- `planner_outcome_summary`: planner-by-scenario-family success, collision, and safety summary.
- `failure_near_miss_mining`: rows for failures, collisions, and low-minimum-TTC episodes.
- `seed_variability_by_planner`: seed-level success variability by planner and scenario family.

Recipe outputs are derived analysis artifacts. Use them as inputs to table or figure generation
only together with the source Parquet export metadata and the original JSONL provenance.
