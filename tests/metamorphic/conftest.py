"""Shared pytest fixtures for the environment-level metamorphic suite."""

from __future__ import annotations

import pytest

from tests.metamorphic.support import BASE_MAP, EPISODE_STEPS


@pytest.fixture
def base_map():
    """Return the deterministic explicit-pedestrian map."""
    return BASE_MAP


@pytest.fixture
def episode_steps():
    """Return the bounded step count shared by metamorphic tests."""
    return EPISODE_STEPS
