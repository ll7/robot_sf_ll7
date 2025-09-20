"""Helper to write a minimal scenario matrix for benchmark tests.

The goal is to keep semantic coverage (resume, reproducibility) while
reducing execution time to a few episodes.
"""

from __future__ import annotations

from pathlib import Path

MINIMAL_MATRIX_YAML = """# Minimal scenario matrix for performance-sensitive tests
scenarios:
  - name: mini-fast
    map_file: dummy_map.svg
    simulation_config:
      horizon: 12
      max_episodes: 2
    metadata:
      purpose: minimal-performance-smoke
    seeds:
      - 123
"""


def write_minimal_matrix(directory: Path) -> Path:
    """Write the minimal matrix YAML into directory and return its path."""
    path = directory / "mini_matrix.yaml"
    path.write_text(MINIMAL_MATRIX_YAML)
    return path
