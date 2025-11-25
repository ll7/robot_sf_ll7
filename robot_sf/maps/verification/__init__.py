"""Map verification workflow module.

This module provides tools for validating SVG maps in the robot_sf repository.
It includes:
- Rule-based geometry and metadata validation
- Environment instantiation testing
- CI integration support
- Structured JSON/JSONL output for tooling

Primary Entry Points
--------------------
- :mod:`robot_sf.maps.verification.runner` - Main verification orchestrator
- :mod:`robot_sf.maps.verification.rules` - Validation rule definitions
- :mod:`robot_sf.maps.verification.manifest` - Structured output writer

Usage Example
-------------
>>> from robot_sf.maps.verification.runner import verify_maps
>>> results = verify_maps(scope="all", mode="local")
>>> print(f"Passed: {results.passed}, Failed: {results.failed}")

See Also
--------
- scripts/validation/verify_maps.py : CLI entry point
- specs/001-map-verification : Feature specification
"""

# Names in __all__ are provided via lazy imports in __getattr__ below
# This avoids heavy module loading at import time
__all__ = [
    "MapRecord",
    "VerificationResult",
    "VerificationRunSummary",
    "verify_maps",
]

from robot_sf.maps.verification.context import VerificationResult, VerificationRunSummary
from robot_sf.maps.verification.map_inventory import MapRecord
from robot_sf.maps.verification.runner import verify_maps
