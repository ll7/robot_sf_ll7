# Phase 1 Data Model: Full Classic Interaction Benchmark

Purpose: Define entities, fields, relationships, and validation rules driving implementation & tests.

## 1. Entities Overview

| Entity | Purpose |
|--------|---------|
| BenchmarkConfig | User / CLI provided knobs controlling sampling, bootstrap, paths, workers, smoke mode. |
| ScenarioDescriptor | Static scenario metadata (archetype, density, map, seeds planned, parameters). |
| EpisodeJob | Concrete unit of work (scenario + seed + planned horizon). |
| EpisodeRecord | One executed episode (schema-conform JSON line). |
| AggregateMetricsGroup | Aggregated metrics for (archetype, density) with CIs. |
| EffectSizeReport | Cross-density comparative metrics per archetype. |
| StatisticalSufficiencyReport | Precision evaluation vs thresholds, adaptive sampling log. |
| PlotArtifact | Metadata for each generated plot (path, type, groups, status). |
| VideoArtifact | Metadata for each generated video (path, scenario, selection rationale, status). |
| BenchmarkManifest | Master manifest describing run parameters, hashes, counts, durations, scaling efficiency. |

## 2. Detailed Schemas (Proposed Typing)

```python
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Literal, Sequence, Optional, Dict, List, Tuple

@dataclass
class BenchmarkConfig:
    output_root: str
    scenario_matrix_path: str
    bootstrap_samples: int = 1000
    bootstrap_confidence: float = 0.95
    master_seed: int = 123
    initial_episodes: int = 150
    batch_size: int = 30
    max_episodes: int = 250
    workers: int = 4
    smoke: bool = False
    force_continue: bool = False  # allow reuse after matrix hash change
    snqi_weights_path: Optional[str] = None
    algo: str = "ppo"
    horizon_override: Optional[int] = None
    effect_size_reference_density: str = "low"  # baseline for Glass Δ

@dataclass
class ScenarioDescriptor:
    scenario_id: str
    archetype: str
    density: str
    map_path: str
    params: Dict[str, object]
    planned_seeds: List[int]
    max_episode_steps: int
    hash_fragment: str  # deterministic hash of defining fields

@dataclass
class EpisodeJob:
    job_id: str  # hash of scenario_id + seed + algo + horizon
    scenario_id: str
    seed: int
    archetype: str
    density: str
    horizon: int

@dataclass
class EpisodeRecord:
    episode_id: str
    scenario_id: str
    seed: int
    archetype: str
    density: str
    status: Literal["success","collision","timeout","error"]
    metrics: Dict[str, float]
    steps: int
    wall_time_sec: float
    algo: str
    created_at: float  # epoch

@dataclass
class AggregateMetric:
    name: str
    mean: float
    median: float
    p95: float
    mean_ci: Tuple[float,float] | None
    median_ci: Tuple[float,float] | None

@dataclass
class AggregateMetricsGroup:
    archetype: str
    density: str
    count: int
    metrics: Dict[str, AggregateMetric]

@dataclass
class EffectSizeEntry:
    metric: str
    density_low: str
    density_high: str
    diff: float  # absolute difference (high - low)
    standardized: float  # Cohen h or Glass Δ

@dataclass
class EffectSizeReport:
    archetype: str
    comparisons: List[EffectSizeEntry]

@dataclass
class PrecisionEntry:
    metric: str
    half_width: float
    target: float
    passed: bool

@dataclass
class ScenarioPrecisionStatus:
    scenario_id: str
    archetype: str
    density: str
    episodes: int
    metric_status: List[PrecisionEntry]
    all_pass: bool

@dataclass
class StatisticalSufficiencyReport:
    evaluations: List[ScenarioPrecisionStatus]
    final_pass: bool
    scaling_efficiency: Dict[int, float]  # worker_count -> efficiency

@dataclass
class PlotArtifact:
    artifact_id: str
    kind: str  # distribution|trajectory|kde|pareto|force_heatmap
    path_pdf: str
    path_png: Optional[str]
    groups: Dict[str,str]
    status: Literal["generated","skipped"]
    note: Optional[str] = None

@dataclass
class VideoArtifact:
    artifact_id: str
    archetype: str
    scenario_id: str
    episode_id: str
    selection_reason: str
    path_mp4: str
    status: Literal["generated","skipped","error"]
    note: Optional[str] = None

@dataclass
class BenchmarkManifest:
    git_hash: str
    scenario_matrix_hash: str
    config: BenchmarkConfig
    start_time: float
    end_time: float | None = None
    total_planned_jobs: int = 0
    executed_jobs: int = 0
    skipped_jobs: int = 0
    failed_jobs: int = 0
    episodes_path: str = ""
    aggregates_path: str = ""
    reports_path: str = ""
    plots_path: str = ""
    videos_path: str = ""
```

## 3. Relationships
- BenchmarkConfig → drives ScenarioDescriptor expansion.
- ScenarioDescriptor (1) → (many) EpisodeJob.
- EpisodeRecord corresponds 1:1 with EpisodeJob (successful or failed status recorded).
- AggregateMetricsGroup built from EpisodeRecord filtered by (archetype,density).
- EffectSizeReport derived from pairs of AggregateMetricsGroup within same archetype.
- StatisticalSufficiencyReport references per-scenario counts & precision entries.
- PlotArtifact / VideoArtifact reference underlying groups or representative episodes.
- BenchmarkManifest indexes paths and counts to assure reproducibility.

## 4. Validation Rules
- scenario_matrix_hash mismatch + !force_continue → abort before running.
- bootstrap_confidence in (0,1).
- initial_episodes <= max_episodes.
- batch_size divides evaluation cadence (no remainder logic needed).
- For precision: require at least min(30, initial_episodes) episodes before first evaluation to avoid unstable estimates.
- EpisodeRecord.status belongs to allowed set; metrics dict MUST include keys used by aggregates else raise.

## 5. Derived / Computed Fields
- job_id / episode_id: stable hash (e.g., sha1) of key fields; truncated for readability.
- hash_fragment: hash of scenario core fields (map, params, archetype, density).
- scaling_efficiency: computed post-run from wall-clock durations captured per worker counts (optional instrumentation hook capturing single-worker baseline time segment).

## 6. Edge Case Handling
- Zero variance metrics: standardized effect size set to 0.0 with note.
- All successes/no successes: Wilson interval ensures non-zero upper bound; mark in report.
- Missing force data: force_heatmap artifacts marked skipped with explanatory note.

## 7. Open Issues (None)
All entity design unknowns resolved in Phase 0 research.

