# Data Model: Accelerate PPO Training with Expert Trajectories

## Overview
This feature manages reproducible artefacts for expert PPO workflows, expert trajectory datasets, and comparative training runs. Data remains file-based but adheres to structured manifests so artefacts are traceable and resumable.

## Entities

### Expert Policy Artefact
- **Identifier**: `policy_id` (string, e.g., `ppo_expert_v1_seed42`)
- **Version**: Semantic component or timestamp appended to identifier
- **Seeds**: List of integer seeds used during training and evaluation
- **Scenario Profile**: Name of scenario configuration file(s) utilised
- **Metrics Snapshot**: Success rate, collision rate, path efficiency, comfort exposure, SNQI with 95% CI bounds
- **Checkpoint Path**: Relative path under `output/models/`
- **Config Manifest**: Reference to configuration file and git commit hash
- **Validation State**: Enum {`draft`, `approved`, `superseded`}
- **Relationships**: Generates zero or more `Trajectory Dataset` entities when `approved`

**Validation Rules**
- Metrics must include all required fields with numeric values and CI ranges where applicable
- Seeds list must contain ≥3 distinct integers
- Validation state transitions allowed only `draft → approved → superseded`

### Trajectory Dataset
- **Dataset ID**: String, e.g., `expert_traj_v1_seedblend`
- **Source Policy ID**: Foreign key to `Expert Policy Artefact`
- **Episode Count**: Integer ≥200 for production datasets (lower counts flagged as `draft`)
- **Storage Path**: Relative path under `output/benchmarks/expert_trajectories/`
- **Format**: One of {`npz`, `jsonl_frames`} (initial release targets `npz`)
- **Scenario Coverage Summary**: Mapping from scenario identifier to episode count
- **Integrity Report**: JSON object capturing validation outcomes (array alignment, missing values, etc.)
- **Metadata**: Timestamp, git hash, trajectory recorder version, collection seeds
- **Quality Status**: Enum {`draft`, `validated`, `quarantined`}
- **Relationships**: Feeds into one or more `Training Run Record` entities

**Validation Rules**
- Dataset size must not exceed 25 GB
- Integrity report must show zero blocking errors before status can be `validated`
- Scenario coverage must include at least the benchmark default scenario set
- Metadata must record the same git hash as the originating expert policy artefact

**State Transitions**
- `draft → validated` once integrity checks pass and QA approves
- `draft → quarantined` if any blocking validation error occurs
- `validated → quarantined` permitted if post-hoc review finds corruption

### Training Run Record
- **Run ID**: String, e.g., `ppo_pretrain_run_2025-11-20_seed42`
- **Run Type**: Enum {`expert_training`, `trajectory_collection`, `bc_pretrain`, `ppo_finetune`, `baseline_ppo`}
- **Input Artefacts**: List referencing policy IDs, dataset IDs, and configuration files used
- **Seeds**: List of seeds executed for the run (inherits from spec requirement of ≥3)
- **Metrics Summary**: Same metric suite as expert artefact plus training loss curves (where applicable)
- **Episode Log Path**: JSONL file path under `output/benchmarks/`
- **Wall-Clock Duration**: Hours/minutes recorded for reproducibility
- **Status**: Enum {`completed`, `failed`, `partial`}
- **Relationships**: Produces or updates `Expert Policy Artefact` (for expert runs), consumes `Trajectory Dataset` (for pretraining)

**Validation Rules**
- Must reference artefacts that exist and are in valid states (e.g., `Trajectory Dataset` must be `validated` before `bc_pretrain`)
- Episode log path must pass schema validation before status can be `completed`
- Failed runs must include diagnostic notes for future triage

## Relationships Diagram (Textual)
```
Expert Policy Artefact (approved)
    └── produces ──▶ Trajectory Dataset (validated)
            └── consumed by ──▶ Training Run Record (bc_pretrain / ppo_finetune)
Training Run Record (expert_training)
    └── outputs ──▶ Expert Policy Artefact
Training Run Record (baseline_ppo)
    └── provides baseline metrics ──▶ Comparative analysis reports
```

## Derived Data & Reports
- **Comparative Summary Report**: Aggregates metrics from paired `Training Run Record` instances (pre-trained vs baseline) and stores side-by-side statistics.
- **Integrity Dashboard**: Aggregates integrity reports across trajectory datasets to monitor recurring anomalies.
