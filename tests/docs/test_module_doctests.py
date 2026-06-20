"""Execute docstring doctests for a curated allowlist of pure ``robot_sf`` modules.

``robot_sf/`` ships ~179 ``>>>`` doctest example lines across 23 modules, but nothing
executes them: ``pyproject.toml`` does not enable ``--doctest-modules`` and no test
collects them, so the examples can silently drift from the real API with no CI signal
(issue #3210).

A blanket ``--doctest-modules`` is intentionally *not* used: it would import heavy,
side-effecting modules (``robot_sf/sim/simulator.py``, ``robot_sf/benchmark/metrics.py``,
GUI/training entry points) and flake. Instead this test runs doctests only on an explicit
allowlist of lightweight, pure modules whose examples are deterministic, side-effect-free,
and currently passing.

Doctests run with ``ELLIPSIS`` (for timestamp/path-variable output) and
``NORMALIZE_WHITESPACE``.

Adding a module to the allowlist
--------------------------------
Run its doctests once (``python -c "import doctest, importlib;
doctest.testmod(importlib.import_module('<mod>'))"``); include it only if it imports
without heavy side effects and its examples pass or can be made to pass with a small,
faithful correction (no library behavior change). Genuinely illustrative pseudocode
should be converted to a fenced ``python`` block in the source docstring instead.
"""

from __future__ import annotations

import doctest
import importlib
import io

import pytest

# Doctest comparison flags applied to every allowlisted module.
DOCTEST_OPTIONFLAGS = doctest.ELLIPSIS | doctest.NORMALIZE_WHITESPACE

# Curated allowlist: pure, lightweight modules with deterministic, passing doctests.
DOCTEST_MODULES = [
    "robot_sf.common.logging",
    "robot_sf.research.logging_config",
    "robot_sf.research.artifact_paths",
    "robot_sf.maps.verification.logging",
    "robot_sf.sensor.registry",
]

# Modules that contain ``>>>`` examples but are deliberately EXCLUDED, with the reason.
# Excluding (rather than executing) keeps the runner fast and non-flaky; these remain
# candidates for a future faithful cleanup pass that converts illustrative examples to
# fenced ``python`` blocks or makes them deterministic.
#
#   robot_sf.benchmark.metrics           - heavy import (tensorflow/torch), ~8s; no real doctests
#   robot_sf.benchmark.utils             - file IO (ensure_directory) + env-dependent (fast-demo cap)
#   robot_sf.common.matplotlib_utils     - matplotlib figure side effects
#   robot_sf.gym_env.robot_env           - constructs a Gym env (sim startup)
#   robot_sf.maps.osm_zones_config       - illustrative pseudocode + YAML file IO
#   robot_sf.maps.verification           - package __init__ illustrative example
#   robot_sf.nav.obstacle                - illustrative pseudocode (undefined rect/obstacle)
#   robot_sf.nav.occupancy_grid          - illustrative array/grid pseudocode
#   robot_sf.nav.occupancy_grid_rasterization - illustrative array pseudocode
#   robot_sf.nav.occupancy_grid_utils    - illustrative array pseudocode
#   robot_sf.nav.svg_map_parser          - illustrative pseudocode + map file IO
#   robot_sf.ped_npc.ped_population       - illustrative pseudocode
#   robot_sf.planner.classic_global_planner - illustrative pseudocode
#   robot_sf.research                    - package __init__ tutorial (ReportOrchestrator signature)
#   robot_sf.research.metadata           - runs git/subprocess; illustrative subscript example
#   robot_sf.research.schema_loader      - illustrative example (undefined metadata_dict) + file IO
#   robot_sf.sensor.fusion_adapter       - illustrative pseudocode (undefined config)
#   robot_sf.sim.simulator               - constructs a simulator (sim startup)
EXCLUDED_MODULES = {
    "robot_sf.benchmark.metrics",
    "robot_sf.benchmark.utils",
    "robot_sf.common.matplotlib_utils",
    "robot_sf.gym_env.robot_env",
    "robot_sf.maps.osm_zones_config",
    "robot_sf.maps.verification",
    "robot_sf.nav.obstacle",
    "robot_sf.nav.occupancy_grid",
    "robot_sf.nav.occupancy_grid_rasterization",
    "robot_sf.nav.occupancy_grid_utils",
    "robot_sf.nav.svg_map_parser",
    "robot_sf.ped_npc.ped_population",
    "robot_sf.planner.classic_global_planner",
    "robot_sf.research",
    "robot_sf.research.metadata",
    "robot_sf.research.schema_loader",
    "robot_sf.sensor.fusion_adapter",
    "robot_sf.sim.simulator",
}


@pytest.mark.parametrize("module_name", DOCTEST_MODULES)
def test_module_doctests(module_name: str) -> None:
    """Allowlisted module doctests must execute and pass under the shared optionflags."""
    module = importlib.import_module(module_name)
    runner = doctest.DocTestRunner(optionflags=DOCTEST_OPTIONFLAGS)
    output = io.StringIO()
    for test in sorted(doctest.DocTestFinder().find(module), key=lambda t: t.name):
        runner.run(test, out=output.write)

    assert runner.tries > 0, (
        f"{module_name}: no executable doctests found — remove it from the allowlist or "
        "restore its examples (a passing allowlist entry must actually run something)."
    )
    assert runner.failures == 0, (
        f"{module_name}: {runner.failures}/{runner.tries} doctest example(s) failed\n"
        f"{output.getvalue()}"
    )


def test_allowlist_and_exclusions_are_disjoint() -> None:
    """Curated doctest entries are either executed or explicitly excluded, never both."""
    overlap = set(DOCTEST_MODULES) & EXCLUDED_MODULES
    assert not overlap, f"modules both allowlisted and excluded: {sorted(overlap)}"
