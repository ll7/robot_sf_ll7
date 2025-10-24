# Phase 0 Research — Verify Feature Extractor Training Flow

## Output Directory Strategy
- **Decision**: Organize each training comparison run under `./tmp/multi_extractor_training/<timestamp>-<run-id>/` with per-extractor subfolders.
- **Rationale**: Timestamped folders preserve historical artifacts for reproducibility, satisfy the specification’s requirement to keep prior outputs, and align with Constitution Principle I on deterministic reruns.
- **Alternatives Considered**:
  - Auto-clean `./tmp` before each run — rejected because it destroys evidence needed for cross-run comparisons.
  - Single shared folder per extractor — rejected; mixing artifacts from multiple runs complicates debugging and violates reproducibility expectations.

## Cross-Platform Worker Configuration
- **Decision**: Default to a single synchronous environment on macOS (spawn start method, OBJC fork safety) and expose a configuration block that enables multi-process vectorization on Ubuntu RTX hardware.
- **Rationale**: Honors NFR-001 by keeping the macOS path reliable while allowing GPU-equipped systems to scale workers explicitly, meeting NFR-002.
- **Alternatives Considered**:
  - Automatically detect GPU and adjust workers — rejected to avoid hidden behavior and to respect Principle VII (no silent contract changes).
  - Forcing multi-process everywhere — rejected because macOS spawn overhead and objective-c fork rules cause instability.

## Aggregated Summary Format
- **Decision**: Emit both a machine-readable JSON summary (`summary.json`) and a human-readable Markdown table (`summary.md`) capturing run metadata, per-extractor status, and aggregated metrics.
- **Rationale**: Fulfills FR-009, simplifies downstream automation, and keeps engineers informed without loading custom tooling.
- **Alternatives Considered**:
  - JSON only — rejected; stakeholders requested a human-readable artifact.
  - Markdown only — rejected; automation and regression tooling need structured data.
