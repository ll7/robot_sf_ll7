# Data Model — Verify Feature Extractor Training Flow

## Entities

### ExtractorConfigurationProfile
- **Purpose**: Captures metadata and hyperparameters for each feature extractor evaluated by the training comparison script.
- **Fields**:
  - `name` *(str, required)* — Stable identifier matching the registry entry in the script.
  - `description` *(str, optional)* — Human-readable summary of the extractor’s intent (e.g., "CNN visual encoder").
  - `parameters` *(dict[str, Any], optional)* — Parameter overrides supplied to the extractor factory.
  - `expected_resources` *(enum: "cpu", "gpu", "hybrid", default "cpu")* — Guides worker selection and logging.
  - `priority` *(int, optional)* — Relative ordering when the script sequences extractors.
- **Relationships**: Referenced by `ExtractorRunRecord.config_name`.
- **Validation Rules**:
  - `name` must be unique within a run configuration.
  - When `expected_resources="gpu"`, the script must warn if CUDA is unavailable.

### ExtractorRunRecord
- **Purpose**: Stores the lifecycle and outcomes for a single extractor during a comparison run.
- **Fields**:
  - `config_name` *(str, required)* — Foreign key to `ExtractorConfigurationProfile.name`.
  - `status` *(enum: "success", "failed", "skipped")* — Execution result; "success" corresponds to FR-008.
  - `start_time` *(datetime, ISO 8601)* and `end_time` *(datetime, ISO 8601)* — Wall-clock timestamps.
  - `duration_seconds` *(float)* — Convenience metric computed from timestamps.
  - `hardware_profile` *(HardwareProfile)* — Snapshot of host details captured at runtime.
  - `worker_mode` *(enum: "single-thread", "vectorized")* — Aligns with FR-002 and NFR-002.
  - `training_steps` *(int)* — Total environment steps executed.
  - `metrics` *(dict[str, float])* — Final metric values (e.g., mean episode reward, collisions).
  - `artifacts` *(dict[str, str])* — Paths relative to the run directory (e.g., checkpoints, TensorBoard logs).
- **Relationships**: Nested inside `TrainingRunSummary.extractor_results`.
- **Validation Rules**:
  - `status="success"` requires non-null `metrics` and `artifacts` entries.
  - `status="skipped"` requires a reason logged in the summary report.

### HardwareProfile
- **Purpose**: Captures host-specific details for auditability and cross-platform comparisons.
- **Fields**:
  - `platform` *(str)* — e.g., "macOS 15" or "Ubuntu 22.04".
  - `arch` *(str)* — CPU architecture (`arm64`, `x86_64`).
  - `gpu_model` *(str, optional)* — Populated on GPU-capable hosts.
  - `cuda_version` *(str, optional)* — Present when CUDA detected.
  - `python_version` *(str)* — Effective runtime version.
  - `workers` *(int)* — Number of parallel environments active.
- **Relationships**: Referenced by `ExtractorRunRecord.hardware_profile` and summarized at the `TrainingRunSummary` level.
- **Validation Rules**:
  - `gpu_model` and `cuda_version` must either both be present or both absent.

### TrainingRunSummary
- **Purpose**: Aggregates metadata and results for the entire comparison run.
- **Fields**:
  - `run_id` *(str)* — Unique identifier, typically `<timestamp>-<short-hash>`.
  - `created_at` *(datetime, ISO 8601)* — Timestamp when the run started.
  - `output_root` *(str)* — Absolute path to the timestamped directory.
  - `hardware_overview` *(list[HardwareProfile])* — Aggregated view for hosts used (single-entry for default workflow).
  - `extractor_results` *(list[ExtractorRunRecord])* — Detailed per-extractor outcomes.
  - `aggregate_metrics` *(dict[str, float])* — Top-level metrics (e.g., best reward, average training time) used in the consolidated summary.
  - `notes` *(list[str], optional)* — Human-readable warnings or skip reasons.
- **Relationships**: Owns `ExtractorRunRecord` instances and references hardware information for FR-004 compliance.
- **Validation Rules**:
  - Must contain at least one `ExtractorRunRecord`.
  - `aggregate_metrics` should include explicit keys outlined in the contract (e.g., `best_mean_reward`, `total_wall_time`).

## State Transitions
- `ExtractorRunRecord.status` transitions: `pending` (implicit) → `success` or `failed`; optional transition to `skipped` when validation prevents execution.
- `TrainingRunSummary` lifecycle: `initialized` (directory created) → `running` (extractors executing) → `completed` (summary artifacts written). Logs must reflect transitions for observability (Principle XII).
