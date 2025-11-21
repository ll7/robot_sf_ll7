# Research Findings — Map Verification Workflow

## Observability Signals Beyond JSON Manifest
- **Decision**: Emit Loguru-structured INFO logs per map plus an aggregated JSON summary containing counts, durations, and slow-map warnings.
- **Rationale**: Aligns with Principle XII (single logging facade), keeps console output human-readable, and surfaces the same data for CI triage without additional scraping.
- **Alternatives Considered**:
  - *Pure JSON output only*: rejected because CI users still need readable streaming diagnostics.
  - *Custom stdout formatter*: redundant with Loguru features and harder to control globally.

## Factory Instantiation Strategy During Verification
- **Decision**: Use `make_robot_env` for default maps and `make_pedestrian_env` when manifest tags indicate pedestrian-only content, always passing a resolved `RobotSimulationConfig` and deterministic seed.
- **Rationale**: Ensures the same backend/environment glue used in production benchmarks, catching runtime incompatibilities early; also satisfies Principle II.
- **Alternatives Considered**:
  - *Direct class instantiation*: violates factory abstraction principle and risks drift.
  - *Skipping runtime instantiation*: would miss backend mismatches and dynamic asset issues.

## CI Gate & Performance Budget Enforcement
- **Decision**: Provide a `scripts/validation/verify_maps.py --ci` mode that respects `ROBOT_SF_PERF_*` thresholds, writes manifests under `output/validation/`, and returns non-zero exit codes on failures or hard timeouts.
- **Rationale**: Reuses existing performance guardrails, keeps artifact routing consistent, and offers a deterministic hook for GitHub Actions + VS Code tasks.
- **Alternatives Considered**:
  - *Ad-hoc CI script*: would duplicate logic and drift from local command.
  - *Slow-map warnings only*: insufficient guard; regressions could merge unnoticed.
