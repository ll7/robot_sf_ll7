# Research: Performance Tracking & Telemetry

## Decision 1: Canonical telemetry artifact format
- **Decision**: Keep JSON/Markdown manifests as the primary telemetry artifacts, then optionally mirror metrics to TensorBoard when users explicitly enable the adapter.
- **Rationale**: Structured files satisfy artifact-policy, diff nicely in git, and integrate with downstream aggregation scripts. Optional TensorBoard mirroring lets teams reuse existing dashboards without breaking headless CI or requiring network credentials.
- **Alternatives considered**:
  - *TensorBoard-only*: rejected because it requires users to launch TensorBoard for even basic ETA checks and complicates artifact retention.
  - *Weights & Biases default*: rejected because many developers run offline/air-gapped and W&B adds network/API-key requirements that conflict with Principle I.

## Decision 2: Resource telemetry stack
- **Decision**: Use `psutil` for CPU/memory sampling, fallback to Python's `resource` module when psutil is unavailable, and add an optional NVML-backed helper (via `pynvml`) for GPU utilization when CUDA hardware is present.
- **Rationale**: `psutil` is already a repo dependency, works cross-platform, and exposes per-process metrics with low overhead. NVML is the standard for NVIDIA telemetry and can be guarded behind try/except to avoid import errors.
- **Alternatives considered**:
  - *gpustat/GPUtil*: convenient but adds heavier dependencies and has weaker error handling on macOS/CPU-only hosts.
  - *shelling out to `nvidia-smi`*: brittle and too slow for 1s sampling cadence.

## Decision 3: Recommendation engine design
- **Decision**: Implement rule-based recommendations driven by thresholds defined in config (e.g., throughput < 75% of baseline, CPU > 90% for 30s, GPU idle while env backlog > 0). Each rule emits a structured `PerformanceRecommendation` entry with remediation text and severity.
- **Rationale**: Deterministic rules are transparent, testable, and align with Principle VI (Metrics Transparency). They can run synchronously after each step or at run-end without ML dependencies.
- **Alternatives considered**:
  - *Machine-learned advisor*: overkill for MVP, harder to test, and would require new datasets.
  - *Manual text-only notes*: lacks structure for downstream tooling and can’t be programmatically checked.

## Decision 4: Performance smoke tests
- **Decision**: Extend `scripts/validation/performance_smoke_test.py` to accept telemetry flags and reuse the existing minimal scenario matrix, emitting telemetry JSON plus pass/fail gating. Provide a shortcut CLI wrapper (`scripts/telemetry/run_perf_tests.py`) that orchestrates the smoke test and writes comparison results next to run manifests.
- **Rationale**: Reusing the validated script honors the unified test suite, leverages the current artifact guard, and keeps performance enforcement in one place.
- **Alternatives considered**:
  - *New benchmark harness*: redundant and risks diverging from established thresholds.
  - *Ad-hoc manual perf notes*: not enforceable and fails Constitution Principle IX (tests for public behavior).
