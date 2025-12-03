"""TODO docstring. Document this module."""

from robot_sf.sim.registry import get_backend, list_backends


def test_default_backend_registered():
    """TODO docstring. Document this function."""
    backends = list_backends()
    assert "fast-pysf" in backends
    factory = get_backend("fast-pysf")
    assert callable(factory)
