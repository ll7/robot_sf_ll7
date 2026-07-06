"""Tests for issue #4014 PPO sequence-model comparison artifacts."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest

from scripts.training import compare_issue_4014_ppo_sequence_models as compare

if TYPE_CHECKING:
    from pathlib import Path


def _write_perf(path: Path, *, available: bool = True, throughput: bool = True) -> None:
    payload = {
        "run_id": path.stem,
        "total_wall_clock_sec": 10.0 if throughput else None,
        "train_env_steps_per_sec_mean": 204.8 if throughput else None,
        "parameter_summary": {
            "available": available,
            "policy_parameter_count": 100,
            "policy_trainable_parameter_count": 90,
            "model_parameter_count": 120,
            "model_trainable_parameter_count": 110,
        },
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_inputs(tmp_path: Path) -> tuple[dict[str, Path], dict[str, Path]]:
    summaries: dict[str, Path] = {}
    configs: dict[str, Path] = {}
    for label in compare.REQUIRED_MODELS:
        summaries[label] = tmp_path / f"{label}.json"
        configs[label] = tmp_path / f"{label}.yaml"
        _write_perf(summaries[label])
        configs[label].write_text(f"policy_id: {label}\n", encoding="utf-8")
    return summaries, configs


def test_build_summary_rejects_dry_run_parameter_placeholder(tmp_path: Path) -> None:
    """Dry-run summaries cannot satisfy the issue #4014 evidence contract."""
    summaries, configs = _write_inputs(tmp_path)
    _write_perf(summaries["ppo_mamba"], available=False)

    with pytest.raises(ValueError, match="parameter_summary.available must be true"):
        compare.build_summary(perf_summaries=summaries, config_paths=configs, output_dir=tmp_path)


def test_build_summary_rejects_missing_throughput(tmp_path: Path) -> None:
    """Throughput metadata is mandatory for every comparison row."""
    summaries, configs = _write_inputs(tmp_path)
    _write_perf(summaries["recurrent_ppo_lstm"], throughput=False)

    with pytest.raises(ValueError, match="required numeric field missing"):
        compare.build_summary(perf_summaries=summaries, config_paths=configs, output_dir=tmp_path)


def test_comparison_artifact_emits_diagnostic_boundary(tmp_path: Path) -> None:
    """Complete rows produce summary, CSV, README, and checksum artifacts."""
    summaries, configs = _write_inputs(tmp_path)
    output_dir = tmp_path / "artifact"

    summary = compare.build_summary(
        perf_summaries=summaries,
        config_paths=configs,
        output_dir=output_dir,
    )
    paths = compare._write_outputs(summary, output_dir)

    assert {path.name for path in paths} == {
        "summary.json",
        "throughput.csv",
        "README.md",
        "SHA256SUMS",
    }
    payload = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
    assert payload["evidence_tier"] == "diagnostic-only"
    assert payload["smoke_not_campaign_evidence"] is True
    assert payload["closure_eligible"] is True
    readme = (output_dir / "README.md").read_text(encoding="utf-8")
    assert readme.startswith("# Issue #4014 PPO sequence encoder smoke comparison")
    assert "Claim boundary:" in readme
