"""Standardized exit codes for SNQI tooling.

These codes are shared across weight recomputation and optimization scripts
to provide a stable contract for downstream automation (CI pipelines,
notebooks, wrappers). Keep this list in sync with the design doc section
"Exit Code Taxonomy" when expanded.

Current codes
-------------
0  SUCCESS                - Execution completed without detected errors.
1  INPUT_ERROR            - File I/O, JSON parse, or structural pre-validation failure.
2  VALIDATION_ERROR       - Schema or finiteness validation failure after result assembly.
3  RUNTIME_ERROR          - Unexpected runtime exception during processing/optimization.
4  MISSING_METRIC_ERROR   - (Reserved) Future: triggered when --fail-on-missing-metric is set.
5  OPTIONAL_DEPS_MISSING  - Optional dependency (e.g., matplotlib) missing when explicitly required.

Notes
-----
- Code 4 is defined early to avoid a future breaking bump when the flag is
  introduced (task: --fail-on-missing-metric).
- Add new codes only if they provide actionable differentiation for callers.
"""

EXIT_SUCCESS = 0
EXIT_INPUT_ERROR = 1
EXIT_VALIDATION_ERROR = 2
EXIT_RUNTIME_ERROR = 3
EXIT_MISSING_METRIC_ERROR = 4  # reserved for forthcoming feature
EXIT_OPTIONAL_DEPS_MISSING = 5

__all__ = [
    "EXIT_SUCCESS",
    "EXIT_INPUT_ERROR",
    "EXIT_VALIDATION_ERROR",
    "EXIT_RUNTIME_ERROR",
    "EXIT_MISSING_METRIC_ERROR",
    "EXIT_OPTIONAL_DEPS_MISSING",
]
