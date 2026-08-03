"""Simulator backend adapters for the robot_sf.sim facade.

This package documents its public backend-factory surface via ``__all__``: the
``dummy_backend`` and ``fast_pysf_backend`` modules that current consumers
import. ``fast_pysf_backend`` depends on the optional fast-pysf subtree, so it
is intentionally not imported here; the backend registry resolves it on demand
through ``importlib`` and its own try/except skip path. Keeping this package
free of eager backend imports lets ``import robot_sf.sim.backends`` succeed even
when fast-pysf is unavailable.
"""

from __future__ import annotations

_BACKEND_MODULES = ("dummy_backend", "fast_pysf_backend")

__all__ = list(_BACKEND_MODULES)
