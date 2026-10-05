"""Required pin witnesses must execute even from an unrelated working directory."""

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_release_pins_execute_outside_repository(tmp_path):
    """Catch the vacuous skip caused by cwd-relative release-manifest discovery."""
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            str(ROOT / "tests/benchmark/test_release_protocol.py"),
            "-k",
            "every_release_manifest_campaign_digest_matches_disk",
            "-q",
            "-o",
            "addopts=",
        ],
        cwd=tmp_path,
        env={**os.environ, "PYTHONPATH": str(ROOT)},
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "passed" in result.stdout and "skipped" not in result.stdout, result.stdout


def test_required_inventory_and_corrupted_campaign_fail(tmp_path):
    """Reject empty required inputs and run the real digest check on fake bytes."""
    import hashlib

    import pytest
    import yaml

    from tests.benchmark.test_release_protocol import (
        test_every_release_manifest_campaign_digest_matches_disk,
    )
    from tests.support.pin_inventory import required_pin_inventory

    with pytest.raises(ValueError, match="Required pin inventory is empty"):
        required_pin_inventory([], name="release manifests")
    campaign = tmp_path / "campaign.yaml"
    campaign.write_text("name: synthetic\n")
    manifest = tmp_path / "manifest.yaml"
    manifest.write_text(
        yaml.safe_dump(
            {
                "canonical_campaign_config": campaign.name,
                "campaign_config_sha256": hashlib.sha256(b"different").hexdigest(),
            }
        )
    )
    with pytest.raises(AssertionError, match="does not match"):
        test_every_release_manifest_campaign_digest_matches_disk(manifest)
