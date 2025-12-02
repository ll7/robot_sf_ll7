"""Module test_backend_registry auto-generated docstring."""

from robot_sf.sim.registry import get_backend, list_backends


def test_default_backend_registered():
    """Test default backend registered.

    Returns:
        Any: Auto-generated placeholder description.
    """
    backends = list_backends()
    assert "fast-pysf" in backends
    factory = get_backend("fast-pysf")
    assert callable(factory)
