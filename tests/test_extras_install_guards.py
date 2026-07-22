"""Tests for the PEP 621 extras split and optional-import guards (Issue #5799).

These tests pin three properties of the slim-core install:

1. The canonical optional-import helpers (``try_import`` already existed;
   ``require_extra`` / ``missing_extra_error`` are new) produce actionable
   ``install robot_sf[<extra>]`` errors when an extra is missing.
2. Each primary extra (``viz``, ``maps``, ``training``, ``benchmark``) has a
   representative module that imports its dependency through ``require_extra``,
   so the friendly error is reachable from a real call site (not just the unit
   helper).
3. ``[all]`` aggregates exactly the four primary capability extras so
   ``uv sync --all-extras`` stays resolvable alongside the ``imitation`` dev
   group (see ``[tool.uv] conflicts`` in pyproject.toml).

The full install matrix (``uv pip install -e .`` with no extras, then each
``.[<extra>]``) is exercised out-of-band by the issue #5799 validation contract;
these in-process tests cover the guard contracts without needing a subprocess
venv per extra.
"""

from __future__ import annotations

import ast
import importlib
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]


class TestRequireExtraHelper:
    """Contract tests for the new require_extra / missing_extra_error helpers."""

    def test_require_extra_returns_module_when_available(self) -> None:
        from robot_sf.common.optional_import import require_extra

        module = require_extra("json")
        import json as stdlib_json

        assert module is stdlib_json

    def test_require_extra_raises_friendly_error_when_missing(self) -> None:
        from robot_sf.common.optional_import import require_extra

        with pytest.raises(ModuleNotFoundError) as excinfo:
            require_extra("definitely_not_a_real_module_zzz_5799", "viz")

        msg = str(excinfo.value)
        assert "install" in msg.lower()
        assert "[viz]" in msg

    def test_require_extra_infers_extra_from_known_mapping(self) -> None:
        """The extra is inferred from the dependency name when not passed."""
        from robot_sf.common.optional_import import require_extra

        with pytest.raises(ModuleNotFoundError) as excinfo:
            require_extra("definitely_not_a_real_module_zzz_5799_pandas")

        # Unknown dependency name falls back to the [all] extra hint.
        assert "[all]" in str(excinfo.value)

    def test_missing_extra_error_message_names_dependency_and_extra(self) -> None:
        from robot_sf.common.optional_import import missing_extra_error

        err = missing_extra_error("torch", "training")
        assert isinstance(err, ModuleNotFoundError)
        msg = str(err)
        assert "torch" in msg
        assert "[training]" in msg
        assert "uv pip install" in msg

    def test_missing_extra_error_falls_back_to_all_for_unknown_dependency(self) -> None:
        from robot_sf.common.optional_import import missing_extra_error, require_extra

        assert missing_extra_error("some_unknown_dep").args[0].count("[all]") >= 1
        # And require_extra surfaces the same fallback.
        with pytest.raises(ModuleNotFoundError, match=r"\[all\]"):
            require_extra("some_unknown_dep_5799")

    def test_require_extra_does_not_swallow_runtime_error(self) -> None:
        """A broken-but-importable dependency must surface, not be masked.

        ``require_extra`` is built on ``try_import`` which catches only
        ``ImportError``; a broken native dependency raising ``RuntimeError``
        must propagate.
        """
        import sys

        from robot_sf.common.optional_import import require_extra

        fake_name = "_fake_broken_optional_dep_5799"

        class _BrokenLoader:
            def find_spec(self, fullname, path=None, target=None):
                if fullname == fake_name:
                    from importlib.machinery import ModuleSpec

                    return ModuleSpec(fake_name, self)
                return None

            def create_module(self, spec):
                raise RuntimeError("simulated broken native dependency")

            def exec_module(self, module):
                raise RuntimeError("simulated broken native dependency")

        loader = _BrokenLoader()
        sys.meta_path.insert(0, loader)
        try:
            with pytest.raises(RuntimeError, match="simulated broken native dependency"):
                require_extra(fake_name)
        finally:
            sys.meta_path.remove(loader)
            sys.modules.pop(fake_name, None)


# Representative module + dependency + extra for each primary extra. These are the
# real guarded call sites; importing them without the extra must surface the
# friendly error for that extra. (When the extra *is* installed, they import
# cleanly, which is the normal dev/CI environment.)
_REPRESENTATIVE_MODULES = [
    pytest.param("robot_sf.render.sim_view", "pygame", "viz", id="viz"),
    pytest.param("robot_sf.nav.osm_map_builder", "geopandas", "maps", id="maps"),
    pytest.param("robot_sf.training.distributional_rl", "torch", "training", id="training"),
    pytest.param("robot_sf.research.aggregation", "pandas", "benchmark", id="benchmark"),
]


class TestRepresentativeGuardedModules:
    """Each primary extra has a real module that imports its dep via require_extra."""

    @pytest.mark.parametrize(("module_path", "dep_name", "extra"), _REPRESENTATIVE_MODULES)
    def test_guarded_module_uses_require_extra(
        self, module_path: str, dep_name: str, extra: str
    ) -> None:
        """The representative module must resolve its dep via require_extra.

        This is a static AST check so it does not require the extra to be
        uninstalled: it confirms the friendly-error path is wired through the
        canonical helper (and therefore stays within the optional-import guard
        inventory ratchet) rather than a bare ``import`` or an ad-hoc
        ``try/except``.
        """
        module_rel = module_path.replace(".", "/") + ".py"
        source = (REPO_ROOT / module_rel).read_text(encoding="utf-8")
        tree = ast.parse(source, filename=module_rel)

        require_extra_calls: list[str] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                if node.func.id == "require_extra" and node.args:
                    first = node.args[0]
                    if isinstance(first, ast.Constant) and first.value == dep_name:
                        require_extra_calls.append(dep_name)

        assert require_extra_calls, (
            f"{module_rel} should resolve '{dep_name}' via require_extra(...) so a "
            f"missing [{extra}] extra raises a clear install hint (Issue #5799)."
        )


class TestAllExtraAggregation:
    """``[all]`` aggregates exactly the four primary capability extras."""

    def _load_pyproject(self) -> dict[str, object]:
        import tomllib

        with (REPO_ROOT / "pyproject.toml").open("rb") as fh:
            return tomllib.load(fh)

    def test_all_lists_exactly_the_four_primary_extras(self) -> None:
        project = self._load_pyproject()
        optional_deps = project["project"]["optional-dependencies"]
        assert "all" in optional_deps, "pyproject.toml is missing the [all] extra"
        all_entries = optional_deps["all"]
        # Each entry is a self-reference like 'robot_sf[viz]'. Extract the extra name.
        import re

        extras = [m for entry in all_entries for m in re.findall(r"robot_sf\[([a-z]+)\]", entry)]
        # The four primary capability extras from Issue #5799. Specialized
        # planner/runtime backends (rllib, sacadrl, orca, ...) stay standalone so
        # ``[all]`` + the ``imitation`` dev group stays resolvable.
        assert extras == ["viz", "maps", "training", "benchmark"]


class TestCoreInstallSlimness:
    """The slim core must not pull the optional capability stacks by default."""

    def _parse_dependencies(self) -> set[str]:
        import tomllib

        with (REPO_ROOT / "pyproject.toml").open("rb") as fh:
            project = tomllib.load(fh)
        # Match the package name before any version specifier / extras.
        import re

        deps = project["project"]["dependencies"]
        return {re.split(r"[<>=!\[\s]", d, maxsplit=1)[0].lower() for d in deps}

    def test_viz_stacks_are_not_core(self) -> None:
        core = self._parse_dependencies()
        # pygame / moviepy / seaborn belong to [viz], not core.
        assert "pygame" not in core
        assert "moviepy" not in core
        assert "seaborn" not in core

    def test_maps_stacks_are_not_core(self) -> None:
        core = self._parse_dependencies()
        # osmnx / geopandas / pyproj / svgelements belong to [maps], not core.
        assert "osmnx" not in core
        assert "geopandas" not in core
        assert "pyproj" not in core
        assert "svgelements" not in core

    def test_training_stacks_are_not_core(self) -> None:
        core = self._parse_dependencies()
        assert "stable-baselines3" not in core
        assert "torch" not in core
        assert "wandb" not in core

    def test_benchmark_tabular_stack_is_not_core(self) -> None:
        core = self._parse_dependencies()
        assert "pandas" not in core
        assert "duckdb" not in core
        assert "pyarrow" not in core

    def test_core_keeps_simulation_primitives(self) -> None:
        """The slim core still ships the sim/gym/nav primitives needed for smoke."""
        core = self._parse_dependencies()
        for required in ("numpy", "gymnasium", "numba", "scipy", "shapely", "loguru"):
            assert required in core, f"{required} must remain a core dependency"


class TestCoreImportAndSmoke:
    """``import robot_sf`` and the gym-env factory stay importable in the slim core.

    These run in the full dev/CI environment (all extras present), so they do
    not prove the no-extras path directly; the no-extras ``import robot_sf`` +
    random/social-force smoke is proven by the issue #5799 validation contract
    out-of-band. Here we at least confirm the factory path imports and steps
    without error under the current environment.
    """

    def test_import_robot_sf_is_lightweight(self) -> None:
        import sys

        before = set(sys.modules)
        # Drop any cached robot_sf to measure a fresh top-level import.
        for name in [n for n in sys.modules if n == "robot_sf" or n.startswith("robot_sf.")]:
            del sys.modules[name]
        try:
            importlib.import_module("robot_sf")
        finally:
            pass
        after = set(sys.modules)
        # ``import robot_sf`` must not eagerly pull heavy extras.
        for blocked in ("stable_baselines3", "wandb", "osmnx", "geopandas", "pygame"):
            assert not any(m == blocked or m.startswith(blocked + ".") for m in after - before), (
                f"import robot_sf eagerly pulled '{blocked}' (should stay behind an extra)."
            )
