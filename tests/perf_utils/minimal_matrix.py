"""Helper to write a minimal scenario matrix for benchmark tests.

The goal is to keep semantic coverage (resume, reproducibility) while
reducing execution time to a few episodes.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

MINIMAL_MATRIX_YAML = """# Minimal scenario matrix for performance-sensitive tests
# Satisfies planning validation requirements while keeping workload tiny.
scenarios:
  - name: mini-fast
    map_file: dummy_map.svg
    simulation_config:
      horizon: 12
      max_episodes: 2
      max_episode_steps: 12
    metadata:
      purpose: minimal-performance-smoke
      archetype: test
      density: low
    seeds:
      - 123
"""


def write_minimal_matrix(directory: Path) -> Path:
    """Write the minimal matrix YAML into directory and return its path."""
    path = directory / "mini_matrix.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(MINIMAL_MATRIX_YAML)
    return path
