"""Contract test T014 for `write_manifest`.

Expectation (final):
  - Writes JSON file with required top-level keys including 'config'.
  - Operation is atomic (not tested here) and overwrites existing if necessary.

Current state: NotImplementedError expected.
"""

from __future__ import annotations

from pathlib import Path

from robot_sf.benchmark.full_classic.io_utils import write_manifest


def test_write_manifest_creates_file(temp_results_dir):
    """TODO docstring. Document this function.

    Args:
        temp_results_dir: TODO docstring.
    """
    manifest_path = Path(temp_results_dir) / "manifest.json"

    class _Config:
        """TODO docstring. Document this class."""

        algo = "ppo"

    class _Manifest:
        """TODO docstring. Document this class."""

        git_hash = "deadbeef"
        scenario_matrix_hash = "cafebabe"
        config = _Config()
        start_time = 0.0
        end_time = None

    write_manifest(_Manifest(), str(manifest_path))
    assert manifest_path.exists()
