# Helper Catalog Contract

## Overview
Defines the required interfaces and documentation standards for reusable helpers extracted from examples and scripts into the `robot_sf` library.

## Modules & Responsibilities

### `robot_sf.benchmark.helper_catalog`
- **prepare_classic_env(config_override: RobotSimulationConfig | None) -> tuple[Env, list[int]]**
  - Initializes a classic interaction environment using factory functions.
  - Returns a configured environment and deterministic seed list.
  - MUST not perform file I/O beyond reading scenario configs.
  - Logs via Loguru; no print statements.
- **load_trained_policy(path: str) -> Policy**
  - Wraps `_load_policy` with caching and error handling.
  - Raises `FileNotFoundError` with actionable guidance.
- **run_episodes_with_recording(env, policy, seeds, record: bool, output_dir: Path) -> list[EpisodeSummary]**
  - Executes episodes, manages rendering/recording, and returns structured summaries.
  - Responsible for invoking `_warn_if_no_frames` equivalent helper.

### `robot_sf.render.helper_catalog`
- **ensure_output_dir(path: Path) -> Path**
  - Creates directories with `exist_ok=True`.
  - Returns normalized path for video outputs.
- **capture_frames(env, stride: int) -> list[np.ndarray]**
  - Provides reusable frame sampling logic for recording helpers.

### `robot_sf.docs.helper_catalog`
- **register_helper(name: str, summary: str, doc_path: str) -> None**
  - Updates central docs index programmatically.
  - Ensures helper entries are deduplicated.

## Documentation Requirements
- Every public helper MUST include a doctring with purpose, parameters, return schema, and side effects.
- Helper modules MUST appear in `docs/README.md` under an appropriate section with a short description.
- Example README files referencing helpers MUST link back to the catalog entries.

## Testing Requirements
- Unit tests MUST cover helper happy paths and failure modes (e.g., missing model file).
- Integration/validation scripts listed in `RegressionCheck` MUST run green after helper extraction.
- New helpers MUST include smoke tests proving they honor environment factory contracts.

## Logging & Error Handling
- Helpers MUST use Loguru and respect existing log level conventions.
- Warnings should surface actionable remediation steps (e.g., when recording is skipped).
- Errors MUST propagate exceptions rather than silently failing to maintain reproducibility guarantees.
