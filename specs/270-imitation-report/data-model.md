# Data Model: Automated Research Reporting

**Feature**: 270-imitation-report  
**Phase**: 1 (Design)  
**Date**: 2025-11-21

## Overview

This document defines the data entities, relationships, validation rules, and state transitions for the research reporting system. Entities map to the "Key Entities" section of the feature specification.

---

## Core Entities

### 1. ExperimentRun

**Purpose**: Logical grouping of baseline and pre-trained multi-seed executions for a single research experiment.

**Attributes**:
- `run_id` (str): Unique identifier `<timestamp>_<experiment_name>`
- `created_at` (datetime): UTC timestamp of run initiation
- `experiment_name` (str): Human-readable experiment label
- `seeds` (list[int]): Random seeds used across conditions
- `baseline_run_ids` (list[str]): Training run identifiers for baseline policies
- `pretrained_run_ids` (list[str]): Training run identifiers for pre-trained policies
- `hardware_profiles` (list[HardwareProfile]): Hardware configs per run
- `configs` (dict[str, Path]): Paths to captured config files (expert, BC, PPO)

**Relationships**:
- 1:N with `MetricRecord` (one run produces many per-seed metrics)
- 1:N with `AggregatedMetrics` (one run yields aggregated stats per condition)
- 1:1 with `ReproducibilityMetadata`

**Validation Rules**:
- `run_id` must match pattern `^\d{8}_\d{6}_[a-z0-9_-]+$`
- `seeds` must be non-empty list
- `created_at` must be valid ISO 8601 UTC timestamp

**State Transitions**:
```
PENDING → RUNNING → COMPLETED
              ↓
            FAILED
```

---

### 2. MetricRecord

**Purpose**: Per-seed, per-variant result capturing individual training outcomes.

**Attributes**:
- `seed` (int): Random seed for this record
- `policy_type` (str): `"baseline"` | `"pretrained"`
- `variant_id` (str | None): Ablation variant identifier (e.g., `"bc10_ds200"`) or None
- `success_rate` (float): [0.0, 1.0] - episode success fraction
- `collision_rate` (float): [0.0, 1.0] - episode collision fraction
- `timesteps_to_convergence` (int | None): PPO timesteps until threshold met, or None if unconverged
- `final_reward_mean` (float): Average episode reward post-convergence
- `run_duration_seconds` (float): Wall-clock time for this seed's training

**Relationships**:
- N:1 with `ExperimentRun`
- Feeds into `AggregatedMetrics` computation

**Validation Rules**:
- `success_rate`, `collision_rate` ∈ [0.0, 1.0]
- `timesteps_to_convergence` > 0 if present
- `policy_type` must be `"baseline"` or `"pretrained"`
- `variant_id` format: `^[a-z]+\d+(_[a-z]+\d+)*$` if present

**Derived Metrics**:
- Sample efficiency improvement computed from baseline vs pretrained `timesteps_to_convergence`

---

### 3. AggregatedMetrics

**Purpose**: Derived statistics per condition (baseline/pretrained/variant) with confidence intervals.

**Attributes**:
- `metric_name` (str): e.g., `"success_rate"`, `"timesteps_to_convergence"`
- `condition` (str): `"baseline"` | `"pretrained"` | variant_id
- `mean` (float): Arithmetic mean across seeds
- `median` (float): 50th percentile
- `p95` (float): 95th percentile
- `std` (float): Standard deviation
- `ci_low` (float | None): Lower bound of bootstrap CI
- `ci_high` (float | None): Upper bound of bootstrap CI
- `ci_confidence` (float): Confidence level (e.g., 0.95)
- `sample_size` (int): Number of seeds contributing to aggregation
- `effect_size` (float | None): Cohen's d when comparing two conditions

**Relationships**:
- N:1 with `ExperimentRun`
- Computed from multiple `MetricRecord` instances

**Validation Rules**:
- `ci_confidence` ∈ [0.0, 1.0]
- `sample_size` >= 1
- If `ci_low`/`ci_high` present, must satisfy `ci_low <= mean <= ci_high`
- `effect_size` only present when comparing paired conditions

**Computation Logic**:
```python
# Bootstrap resampling for CIs
samples = resample(metric_values, n_iterations=1000, seed=seed)
ci_low, ci_high = np.percentile(samples, [2.5, 97.5])

# Effect size (Cohen's d)
pooled_std = sqrt((std1**2 + std2**2) / 2)
effect_size = (mean1 - mean2) / pooled_std
```

---

### 4. AblationConfig

**Purpose**: Parameter slice definition for ablation studies.

**Attributes**:
- `variant_id` (str): Unique identifier (e.g., `"bc10_ds200"`)
- `bc_epochs` (int): Behavioral cloning epochs
- `dataset_size` (int): Number of expert trajectory episodes
- `other_params` (dict[str, Any]): Additional ablation parameters

**Relationships**:
- N:1 with `ExperimentRun` (one experiment may test multiple configs)
- 1:N with `MetricRecord` (one config produces records per seed)

**Validation Rules**:
- `bc_epochs` > 0
- `dataset_size` >= 100 (minimum per FR-013)
- `variant_id` must be unique within experiment

**Cartesian Product Generation**:
```python
from itertools import product

params = {
    "bc_epochs": [5, 10, 20],
    "dataset_size": [100, 200, 300]
}

variants = [
    AblationConfig(
        variant_id=f"bc{bc}_ds{ds}",
        bc_epochs=bc,
        dataset_size=ds
    )
    for bc, ds in product(params["bc_epochs"], params["dataset_size"])
]
```

---

### 5. HypothesisDefinition

**Purpose**: Statement and evaluation result for research hypothesis.

**Attributes**:
- `description` (str): Full hypothesis statement
- `metric` (str): Target metric name (e.g., `"timesteps_to_convergence"`)
- `threshold_value` (float): Required improvement percentage (40.0 for "≥40%")
- `threshold_type` (str): `"min"` | `"max"` (direction of improvement)
- `decision` (str): `"PASS"` | `"FAIL"` | `"INCOMPLETE"`
- `measured_value` (float | None): Actual improvement observed
- `ci_low` (float | None): CI lower bound for measured value
- `ci_high` (float | None): CI upper bound for measured value
- `note` (str): Explanatory text (e.g., "Based on 3/3 successful seeds")

**Relationships**:
- 1:1 with `ExperimentRun` (one primary hypothesis per experiment)
- May have N additional hypotheses for ablation variants

**Validation Rules**:
- `threshold_value` > 0
- `decision` ∈ {"PASS", "FAIL", "INCOMPLETE"}
- If `measured_value` present, `decision` derivable deterministically
- `note` required when `decision == "INCOMPLETE"`

**Evaluation Logic**:
```python
def evaluate_hypothesis(
    baseline_metric: float,
    pretrained_metric: float,
    threshold: float
) -> HypothesisDefinition:
    improvement_pct = 100 * (baseline_metric - pretrained_metric) / baseline_metric
    
    if improvement_pct >= threshold:
        decision = "PASS"
    elif improvement_pct < 0:
        decision = "FAIL"
        note = "Pre-training degraded performance"
    else:
        decision = "FAIL"
        note = f"Improvement {improvement_pct:.1f}% < threshold {threshold}%"
    
    return HypothesisDefinition(
        description=f"Pre-training reduces timesteps by ≥{threshold}%",
        metric="timesteps_to_convergence",
        threshold_value=threshold,
        threshold_type="min",
        decision=decision,
        measured_value=improvement_pct,
        note=note
    )
```

---

### 6. ReportArtifact

**Purpose**: Generated asset metadata for reproducibility and asset management.

**Attributes**:
- `path` (Path): Absolute or relative path to artifact
- `artifact_type` (str): `"figure"` | `"markdown"` | `"latex"` | `"json"` | `"csv"`
- `generated_at` (datetime): UTC timestamp of creation
- `sha256` (str | None): Hash for integrity verification
- `size_bytes` (int): File size

**Relationships**:
- N:1 with `ExperimentRun` (one run produces many artifacts)

**Validation Rules**:
- `path` must exist and be readable
- `artifact_type` must match file extension
- `generated_at` must be <= current time

**Lifecycle**:
```
GENERATING → COMPLETED → ARCHIVED (optional)
       ↓
    FAILED
```

---

### 7. ReproducibilityMetadata

**Purpose**: Provenance information ensuring experiment reproducibility.

**Attributes**:
- `git_commit` (str): Full commit hash (40 chars)
- `git_branch` (str): Branch name
- `git_dirty` (bool): Whether uncommitted changes present
- `python_version` (str): e.g., `"3.11.5"`
- `key_packages` (dict[str, str]): Package name → version
- `hardware` (HardwareProfile): CPU, memory, GPU details
- `seeds` (list[int]): All seeds used
- `configs` (dict[str, dict]): Captured config objects
- `timestamp` (datetime): UTC time of metadata capture

**Relationships**:
- 1:1 with `ExperimentRun`

**Validation Rules**:
- `git_commit` must match `^[0-9a-f]{40}$`
- `python_version` must be valid semver
- `timestamp` must precede or equal `ExperimentRun.created_at`

**Collection Implementation**:
```python
def collect_reproducibility_metadata(
    seeds: list[int],
    configs: dict[str, dict]
) -> ReproducibilityMetadata:
    return ReproducibilityMetadata(
        git_commit=subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip(),
        git_branch=subprocess.check_output(["git", "branch", "--show-current"]).decode().strip(),
        git_dirty=bool(subprocess.check_output(["git", "status", "--porcelain"]).decode().strip()),
        python_version=platform.python_version(),
        key_packages=get_package_versions(["stable-baselines3", "imitation", "robot-sf"]),
        hardware=collect_hardware_profile(),
        seeds=seeds,
        configs=configs,
        timestamp=datetime.now(UTC)
    )
```

---

### 8. HardwareProfile

**Purpose**: Hardware specifications for performance context.

**Attributes**:
- `cpu_model` (str): Processor name
- `cpu_cores` (int): Physical + logical core count
- `memory_gb` (int): RAM in gigabytes
- `gpu_model` (str | None): GPU identifier if present
- `gpu_memory_gb` (int | None): VRAM if GPU present

**Relationships**:
- Embedded in `ReproducibilityMetadata`
- N:1 with `ExperimentRun` (list of profiles if mixed hardware)

**Validation Rules**:
- `cpu_cores` >= 1
- `memory_gb` >= 1
- If `gpu_model` present, `gpu_memory_gb` must be present

**Collection**:
```python
import psutil
import platform

def collect_hardware_profile() -> HardwareProfile:
    return HardwareProfile(
        cpu_model=platform.processor(),
        cpu_cores=psutil.cpu_count(logical=True),
        memory_gb=round(psutil.virtual_memory().total / (1024**3)),
        gpu_model=get_gpu_model(),  # via nvidia-smi or None
        gpu_memory_gb=get_gpu_memory() if get_gpu_model() else None
    )
```

---

## Entity Relationships Diagram

```
ExperimentRun (1) ──< (N) MetricRecord
      │
      ├──< (N) AggregatedMetrics
      │
      ├──< (N) AblationConfig
      │
      ├──< (N) ReportArtifact
      │
      └──< (1) ReproducibilityMetadata
                    │
                    └──< (N) HardwareProfile

HypothesisDefinition (1) ──< (1) ExperimentRun
```

---

## Data Flow

```
1. Training Runs → MetricRecord (per seed)
2. MetricRecord → AggregatedMetrics (bootstrap aggregation)
3. AggregatedMetrics → HypothesisDefinition (threshold evaluation)
4. ExperimentRun + AggregatedMetrics → ReportArtifact (figure generation)
5. All entities → ReportArtifact (Markdown/LaTeX rendering)
6. System state → ReproducibilityMetadata (capture provenance)
```

---

## Persistence Strategy

**Primary Format**: JSON for structured data (metadata, aggregated metrics, hypothesis results)

**Secondary Format**: CSV for tabular data (flattened metrics for external tools)

**File Locations**:
```
output/research_reports/<timestamp>_<name>/
├── metadata.json              # ReproducibilityMetadata + ExperimentRun
├── aggregated_metrics.json    # List[AggregatedMetrics]
├── hypothesis_results.json    # List[HypothesisDefinition]
├── artifacts_manifest.json    # List[ReportArtifact]
└── data/
    ├── metrics.csv            # Flattened MetricRecord
    └── aggregated.csv         # Flattened AggregatedMetrics
```

**Schema Versioning**: All JSON files include `"schema_version": "1.0.0"` field for future evolution.

---

## Summary

Entities defined:
- ✅ ExperimentRun (experiment coordination)
- ✅ MetricRecord (per-seed results)
- ✅ AggregatedMetrics (statistical summaries)
- ✅ AblationConfig (parameter variants)
- ✅ HypothesisDefinition (evaluation outcomes)
- ✅ ReportArtifact (generated files)
- ✅ ReproducibilityMetadata (provenance)
- ✅ HardwareProfile (system specs)

All entities include validation rules, relationships, and state transitions where applicable.

**Status**: Ready for contract generation (JSON schemas)
