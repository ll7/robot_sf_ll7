"""Compatibility shim for the multi-pedestrian example relocated to ``examples/advanced``.

Re-exports ``create_multi_pedestrian_map`` so legacy imports keep working.
"""

from importlib import import_module as _import_module

_ADVANCED_MODULE = _import_module("examples.advanced.08_multi_pedestrian")

create_multi_pedestrian_map = _ADVANCED_MODULE.create_multi_pedestrian_map

__all__ = ("create_multi_pedestrian_map",)

del _ADVANCED_MODULE
del _import_module
