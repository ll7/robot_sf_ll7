"""Tests for map-runner algorithm readiness profiles and SocNav preflight policy."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
import yaml

from robot_sf.benchmark import map_runner

if TYPE_CHECKING:
    from pathlib import Path


SCHEMA_PATH = "robot_sf/benchmark/schemas/episode.schema.v1.json"


def _scenario() -> dict[str, object]:
    """Return a minimal map scenario for run_map_batch tests."""
    return {
        "name": "profile-preflight-smoke",
        "metadata": {"supported": True},
        "map_file": "maps/svg_maps/classic_crossing.svg",
        "simulation_config": {"max_episode_steps": 5},
        "seeds": [1],
    }


def _patch_lightweight_batch(monkeypatch) -> None:
    """Patch expensive batch plumbing so tests exercise only selection/preflight logic."""
    monkeypatch.setattr(map_runner, "validate_scenario_list", lambda _scenarios: [])
    monkeypatch.setattr(map_runner, "load_schema", lambda _path: {})
    monkeypatch.setattr(
        map_runner,
        "_run_map_job_worker",
        lambda _job: {"episode_id": "ep-smoke"},
    )
    monkeypatch.setattr(map_runner, "_write_validated", lambda *_args, **_kwargs: None)


def test_baseline_safe_blocks_experimental_algo(tmp_path: Path, monkeypatch) -> None:
    """`baseline-safe` profile must reject experimental algorithm selections."""
    _patch_lightweight_batch(monkeypatch)
    out_path = tmp_path / "episodes.jsonl"

    with pytest.raises(ValueError, match="blocked by profile 'baseline-safe'"):
        map_runner.run_map_batch(
            [_scenario()],
            out_path,
            schema_path=SCHEMA_PATH,
            algo="ppo",
            benchmark_profile="baseline-safe",
            resume=False,
        )


def test_paper_baseline_requires_ppo_paper_gate(tmp_path: Path, monkeypatch) -> None:
    """Paper-baseline profile should fail when PPO provenance/quality gate is missing."""
    _patch_lightweight_batch(monkeypatch)
    out_path = tmp_path / "episodes.jsonl"

    with pytest.raises(ValueError, match="paper-grade gate failed"):
        map_runner.run_map_batch(
            [_scenario()],
            out_path,
            schema_path=SCHEMA_PATH,
            algo="ppo",
            benchmark_profile="paper-baseline",
            resume=False,
        )


def test_paper_baseline_allows_ppo_when_gate_is_met(tmp_path: Path, monkeypatch) -> None:
    """Paper-baseline should allow PPO when provenance + quality gate is satisfied."""
    _patch_lightweight_batch(monkeypatch)

    monkeypatch.setattr(
        map_runner,
        "_build_policy",
        lambda _algo, _cfg: (lambda _obs: (0.0, 0.0), {"status": "ok"}),
    )

    algo_cfg_path = tmp_path / "ppo_paper.yaml"
    algo_cfg_path.write_text(
        yaml.safe_dump(
            {
                "profile": "paper",
                "provenance": {
                    "training_config": "configs/training/ppo_imitation/expert_ppo_issue_403_grid.yaml",
                    "training_commit": "abc123def",
                    "dataset_version": "v1",
                    "checkpoint_id": "ppo-paper-001",
                    "normalization_id": "norm-v1",
                    "deterministic_seed_set": "eval",
                },
                "quality_gate": {
                    "min_success_rate": 0.60,
                    "measured_success_rate": 0.75,
                },
            }
        ),
        encoding="utf-8",
    )

    out_path = tmp_path / "episodes.jsonl"
    summary = map_runner.run_map_batch(
        [_scenario()],
        out_path,
        schema_path=SCHEMA_PATH,
        algo="ppo",
        algo_config_path=str(algo_cfg_path),
        benchmark_profile="paper-baseline",
        resume=False,
    )

    assert summary["written"] == 1
    assert summary["algorithm_readiness"]["profile"] == "paper-baseline"
    assert summary["algorithm_metadata_contract"]["baseline_category"] == "learning"


def test_socnav_fail_fast_policy_raises(tmp_path: Path, monkeypatch) -> None:
    """SocNav preflight policy `fail-fast` should propagate prereq failures."""
    _patch_lightweight_batch(monkeypatch)
    monkeypatch.setattr(
        map_runner,
        "_build_policy",
        lambda _algo, _cfg: (_ for _ in ()).throw(RuntimeError("missing socnav deps")),
    )

    with pytest.raises(RuntimeError, match="SocNav preflight failed"):
        map_runner.run_map_batch(
            [_scenario()],
            tmp_path / "episodes.jsonl",
            schema_path=SCHEMA_PATH,
            algo="socnav_sampling",
            benchmark_profile="experimental",
            socnav_missing_prereq_policy="fail-fast",
            resume=False,
        )


def test_socnav_skip_with_warning_policy_returns_skipped_summary(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """SocNav preflight `skip-with-warning` should skip the batch deterministically."""
    _patch_lightweight_batch(monkeypatch)
    monkeypatch.setattr(
        map_runner,
        "_build_policy",
        lambda _algo, _cfg: (_ for _ in ()).throw(RuntimeError("missing socnav deps")),
    )

    summary = map_runner.run_map_batch(
        [_scenario()],
        tmp_path / "episodes.jsonl",
        schema_path=SCHEMA_PATH,
        algo="socnav_sampling",
        benchmark_profile="experimental",
        socnav_missing_prereq_policy="skip-with-warning",
        resume=False,
    )

    assert summary["written"] == 0
    assert summary["total_jobs"] == 0
    assert summary["skipped_jobs"] == 1
    assert summary["preflight"]["status"] == "skipped"
    assert summary["algorithm_metadata_contract"]["canonical_algorithm"] == "socnav_sampling"


def test_socnav_fallback_policy_forces_allow_fallback(tmp_path: Path, monkeypatch) -> None:
    """SocNav `fallback` policy should retry preflight with `allow_fallback=True`."""
    _patch_lightweight_batch(monkeypatch)

    def _mock_build_policy(_algo: str, cfg: dict[str, object]):
        if not cfg.get("allow_fallback", False):
            raise RuntimeError("missing socnav deps")
        return (lambda _obs: (0.0, 0.0), {"status": "ok"})

    monkeypatch.setattr(map_runner, "_build_policy", _mock_build_policy)

    summary = map_runner.run_map_batch(
        [_scenario()],
        tmp_path / "episodes.jsonl",
        schema_path=SCHEMA_PATH,
        algo="socnav_sampling",
        benchmark_profile="experimental",
        socnav_missing_prereq_policy="fallback",
        resume=False,
    )

    assert summary["written"] == 1
    assert summary["preflight"]["status"] == "fallback"


def test_adapter_impact_eval_flag_surfaces_in_summary(tmp_path: Path, monkeypatch) -> None:
    """Adapter-impact mode should be represented in summary metadata contract."""
    _patch_lightweight_batch(monkeypatch)
    summary = map_runner.run_map_batch(
        [_scenario()],
        tmp_path / "episodes.jsonl",
        schema_path=SCHEMA_PATH,
        algo="goal",
        benchmark_profile="baseline-safe",
        adapter_impact_eval=True,
        resume=False,
    )
    impact = summary["algorithm_metadata_contract"].get("adapter_impact")
    assert isinstance(impact, dict)
    assert impact.get("requested") is True


def test_adapter_impact_summary_finalizes_from_worker_records(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """Summary adapter-impact status should finalize when worker records include counters."""
    monkeypatch.setattr(map_runner, "validate_scenario_list", lambda _scenarios: [])
    monkeypatch.setattr(map_runner, "load_schema", lambda _path: {})
    monkeypatch.setattr(map_runner, "_write_validated", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        map_runner,
        "_run_map_job_worker",
        lambda _job: {
            "episode_id": "ep-smoke",
            "algorithm_metadata": {
                "adapter_impact": {
                    "requested": True,
                    "native_steps": 3,
                    "adapted_steps": 2,
                }
            },
        },
    )

    summary = map_runner.run_map_batch(
        [_scenario()],
        tmp_path / "episodes.jsonl",
        schema_path=SCHEMA_PATH,
        algo="ppo",
        benchmark_profile="experimental",
        adapter_impact_eval=True,
        resume=False,
    )

    impact = summary["algorithm_metadata_contract"].get("adapter_impact")
    assert isinstance(impact, dict)
    assert impact.get("requested") is True
    assert impact.get("status") == "complete"
    assert impact.get("native_steps") == 3
    assert impact.get("adapted_steps") == 2
    assert impact.get("adapter_fraction") == pytest.approx(0.4)
