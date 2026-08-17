"""Tests for the issue #7340 real-manifest diagnostic contract."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from robot_sf.adversarial.feasibility_first_real import RealManifestError, _load_config


def test_real_manifest_config_rejects_claim_boundary_drift(tmp_path: Path) -> None:
    """The real-manifest runner must reject a config that broadens its claim boundary."""
    source = Path("configs/benchmarks/issue_7340_feasibility_first_real_manifest_v1.yaml")
    payload = yaml.safe_load(source.read_text(encoding="utf-8"))
    payload["claim_boundary"] = "benchmark comparison"
    config_path = tmp_path / source.name
    config_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")

    with pytest.raises(RealManifestError, match="claim_boundary"):
        _load_config(config_path)
