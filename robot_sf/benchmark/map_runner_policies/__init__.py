"""Per-algorithm policy builders for the map-runner benchmark.

This package decomposes the historically monolithic ``_build_policy`` dispatcher in
``robot_sf.benchmark.map_runner`` into one builder module per algorithm family
(see #3384). Each builder module exposes a ``build(...)`` function returning the
``(policy_fn, meta)`` pair ``_build_policy`` expects, and ``map_runner`` consults a
registry of these builders before falling through to its remaining inline branches.

The first migrated family is the built-in goal/simple policy (#3400).
"""

from __future__ import annotations
