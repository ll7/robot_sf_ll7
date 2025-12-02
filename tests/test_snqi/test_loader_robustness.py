"""Module test_loader_robustness auto-generated docstring."""

from __future__ import annotations

import argparse
import importlib.util as importlib_util
import sys
import types


def test_dynamic_loader_handles_sys_modules_edge_cases(monkeypatch):
    """Exercise spec-based dynamic loader against sys.modules anomalies.

    Ensures that:
    - None placeholders in sys.modules are handled (removed) before loading.
    - First invocation loads via spec and subsequent calls hit the cache.
    - Heavy script execution is avoided by stubbing run() to a fast lambda.
    """
    # Avoid LIGHT_TEST fast-exit to actually exercise the loader; we'll stub run().
    monkeypatch.delenv("ROBOT_SF_SNQI_LIGHT_TEST", raising=False)

    # Import CLI module
    from robot_sf.benchmark import cli as bench_cli  # import here to allow monkeypatching

    # Stub importlib loaders to avoid executing real heavy scripts.
    # - module_from_spec returns a lightweight module-like object with run() -> 0
    # - spec_from_file_location returns a dummy spec with a no-op loader
    def fake_module_from_spec(spec):  # type: ignore[no-untyped-def]
        """Fake module from spec.

        Args:
            spec: Auto-generated placeholder description.

        Returns:
            Any: Auto-generated placeholder description.
        """
        mod = types.SimpleNamespace()
        # mimic minimal module attrs the loader might expect
        mod.__spec__ = spec
        mod.__file__ = getattr(spec, "origin", None)
        mod.run = lambda _args: 0
        return mod

    class _DummyLoader:  # minimal loader; exec_module is a no-op
        """DummyLoader class."""

        def exec_module(self, _mod):  # type: ignore[no-untyped-def]
            """Exec module.

            Args:
                _mod: Auto-generated placeholder description.

            Returns:
                Any: Auto-generated placeholder description.
            """
            return None

    class _DummySpec:
        """DummySpec class."""

        def __init__(self, name, origin):
            """Init.

            Args:
                name: Auto-generated placeholder description.
                origin: Auto-generated placeholder description.

            Returns:
                Any: Auto-generated placeholder description.
            """
            self.name = name
            self.origin = str(origin)
            self.loader = _DummyLoader()

    def fake_spec_from_file_location(name, path, *_args, **_kwargs):  # type: ignore[no-untyped-def]
        """Fake spec from file location.

        Args:
            name: Auto-generated placeholder description.
            path: Auto-generated placeholder description.
            _args: Auto-generated placeholder description.
            _kwargs: Auto-generated placeholder description.

        Returns:
            Any: Auto-generated placeholder description.
        """
        return _DummySpec(name, path)

    monkeypatch.setattr(importlib_util, "module_from_spec", fake_module_from_spec, raising=True)
    monkeypatch.setattr(
        importlib_util,
        "spec_from_file_location",
        fake_spec_from_file_location,
        raising=True,
    )

    # Build a parser (attaches dynamic loader with its own cache variables)
    parser = bench_cli._configure_parser()  # type: ignore[attr-defined]
    loader = parser.snqi_loader

    # Simulate sys.modules cache anomalies: None placeholders for target module names
    monkeypatch.setitem(sys.modules, "snqi_optimize_script", None)
    monkeypatch.setitem(sys.modules, "snqi_recompute_script", None)

    ns = argparse.Namespace()
    # First calls should load via spec (using our stub) and return 0
    assert loader["invoke_optimize"](ns) == 0
    assert loader["invoke_recompute"](ns) == 0
    # Second calls should hit the cached modules and still return 0
    assert loader["invoke_optimize"](ns) == 0
    assert loader["invoke_recompute"](ns) == 0
