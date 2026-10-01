"""Verify the D-062 release policy using real public checkpoint bytes."""

import pytest

from robot_sf.models import resolve_model_path, sha256_of_file

MODEL_ID = "ppo_release_robot_b1002_last_20261001"
SHA256 = "764a7d88f5b608237641d973634899e05a67b25e65f8b1607cfca025459824bc"


@pytest.mark.slow
def test_release_robot_asset_resolves_and_verifies_sha256():
    """Hydration must resolve the pinned public asset and verify its complete ZIP bytes."""
    path = resolve_model_path(MODEL_ID)
    assert path.is_file()
    assert sha256_of_file(path) == SHA256
