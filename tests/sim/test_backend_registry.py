"""TODO docstring. Document this module."""

import importlib

import pytest

from robot_sf.sim import registry as loaded_registry


@pytest.fixture(autouse=True)
def _restore_loaded_registry_after_test():
    """Restore registry module state after tests that reload it with mocked imports."""
    yield
    importlib.reload(loaded_registry)


def test_default_backend_registered():
    """TODO docstring. Document this function."""
    backends = loaded_registry.list_backends()
    assert "dummy" in backends
    assert callable(loaded_registry.get_backend("dummy"))
    if "fast-pysf" in backends:
        assert callable(loaded_registry.get_backend("fast-pysf"))


def test_fast_pysf_backend_missing_dependency_is_skipped(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify missing fast-pysf dependencies do not fail registry import."""
    warnings: list[tuple[object, tuple[object, ...]]] = []

    def _capture_warning(message: object, *args: object, **_: object) -> None:
        warnings.append((message, args))

    def _missing_dependency_import(
        name: str,
        package: str | None = None,
    ) -> object:
        if name == "robot_sf.sim.backends.fast_pysf_backend":
            raise ModuleNotFoundError("No module named 'pysocialforce'", name="pysocialforce")
        return original_import_module(name, package=package)

    original_import_module = loaded_registry.importlib.import_module
    monkeypatch.setattr(loaded_registry.importlib, "import_module", _missing_dependency_import)
    monkeypatch.setattr(loaded_registry.logger, "warning", _capture_warning)

    importlib.reload(loaded_registry)

    assert "fast-pysf" not in loaded_registry.list_backends()
    assert "dummy" in loaded_registry.list_backends()
    assert loaded_registry.select_best_backend() == "dummy"
    assert any(
        "fast-pysf backend is unavailable/skipped" in str(message) for message, _args in warnings
    )


def test_unexpected_fast_pysf_import_error_fails_fast(monkeypatch: pytest.MonkeyPatch) -> None:
    """Unexpected fast-pysf import errors should still fail module import."""

    def _unexpected_import_error(
        name: str,
        package: str | None = None,
    ) -> object:
        if name == "robot_sf.sim.backends.fast_pysf_backend":
            raise ImportError("Unexpected import failure", name="unexpected_module")
        return original_import_module(name, package=package)

    original_import_module = loaded_registry.importlib.import_module
    monkeypatch.setattr(loaded_registry.importlib, "import_module", _unexpected_import_error)

    with pytest.raises(ImportError):
        importlib.reload(loaded_registry)
