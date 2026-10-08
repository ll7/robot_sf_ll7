"""Register built-ins on objective import without editing the source-pinned registry."""

from __future__ import annotations

import sys
from importlib import import_module
from importlib.abc import MetaPathFinder
from importlib.machinery import PathFinder, SourceFileLoader
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from importlib.machinery import ModuleSpec
    from types import ModuleType

_REGISTRY = "robot_sf.adversarial.objectives"
_SCORER = "robot_sf.adversarial.objectives_v2"


class _BuiltinObjectiveLoader(SourceFileLoader):
    """Complete registration after either supported import order finishes."""

    def exec_module(self, module: ModuleType) -> None:
        super().exec_module(module)
        if self.name == _REGISTRY:
            # If the scorer imported the registry, this returns its partial module;
            # its own loader completes registration after defining the scorer.
            import_module(_SCORER)
        else:
            module.register_constraints_first_lexicographic_v2()
            sys.meta_path[:] = [
                finder
                for finder in sys.meta_path
                if not isinstance(finder, _BuiltinObjectiveFinder)
            ]


class _BuiltinObjectiveFinder(MetaPathFinder):
    """One-shot hook restricted to the two built-in objective source modules."""

    def find_spec(
        self, fullname: str, path: list[str] | None, target: ModuleType | None = None
    ) -> ModuleSpec | None:
        if fullname not in {_REGISTRY, _SCORER}:
            return None
        spec = PathFinder.find_spec(fullname, path, target)
        if spec is not None and isinstance(spec.loader, SourceFileLoader):
            spec.loader = _BuiltinObjectiveLoader(fullname, spec.loader.path)
        return spec


def install_builtin_objective_registration() -> None:
    """Keep report-only imports lightweight; register v2 when objective code loads."""
    if _SCORER in sys.modules:
        sys.modules[_SCORER].register_constraints_first_lexicographic_v2()
    elif not any(isinstance(finder, _BuiltinObjectiveFinder) for finder in sys.meta_path):
        sys.meta_path.insert(0, _BuiltinObjectiveFinder())
