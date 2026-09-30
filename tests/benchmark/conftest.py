"""Explicit override layer for immutable historical H600 protocol fixtures."""

from pathlib import Path

import pytest

# These inputs are pinned release artifacts. Their original YAML and manifest
# hashes stay untouched; only fixtures requesting this layer opt into extension.
HISTORICAL_H600_CONFIGS = frozenset(
    {
        "paper_experiment_matrix_v1_h600_hybrid_roster.yaml",
        "paper_experiment_matrix_v1_h600_hybrid_vs_orca_s30.yaml",
        "paper_experiment_matrix_v1_h600_trace_capable_rerun.yaml",
        "paper_experiment_matrix_v2_h600_s30_extended_post1.yaml",
        "paper_experiment_matrix_v2_h600_s30_benchmark_data_2026_08.yaml",
        "paper_experiment_matrix_v2_h600_s30_runtime_smoke.yaml",
        "paper_experiment_matrix_v2_h600_s30_runtime_smoke_v0_3.yaml",
        "paper_experiment_matrix_v2_h600_s30_runtime_smoke_v0_4.yaml",
        "paper_experiment_matrix_v2_h600_hybrid_stress_smoke.yaml",
    }
)


@pytest.fixture
def historical_horizon_policy(monkeypatch):
    """Feed the opt-in through real parsing/admission, preserving source-byte pins."""
    from robot_sf.benchmark.camera_ready import _config

    assemble = _config._assemble_campaign_config

    def historical_override(parsed, *, payload, config_path, **kwargs):
        if Path(config_path).name in HISTORICAL_H600_CONFIGS:
            payload = {
                "protocol_version": "0.0.7",
                "horizon_policy": "legacy_fixed_extends_authored",
                **payload,
            }
        return assemble(parsed, payload=payload, config_path=config_path, **kwargs)

    monkeypatch.setattr(_config, "_assemble_campaign_config", historical_override)
