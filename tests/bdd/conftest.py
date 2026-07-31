"""pytest-bdd acceptance test configuration."""

import pytest


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers",
        "bdd: mark test as a pytest-bdd acceptance scenario",
    )
