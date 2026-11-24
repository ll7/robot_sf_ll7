"""Map dispatch, resume, and integrity must count the same executable jobs."""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import TYPE_CHECKING

import pytest
import yaml

from robot_sf.benchmark.camera_ready._config import _load_campaign_scenarios
from robot_sf.benchmark.camera_ready._resume_plan import _expected_job_identities
from robot_sf.benchmark.camera_ready._run_state import _expected_episode_identities
from robot_sf.benchmark.camera_ready.campaign import _post_run_integrity_and_fairness
from robot_sf.benchmark.camera_ready_campaign import load_campaign_config, run_campaign
from robot_sf.benchmark.map_runner import map_runner_batch_plan, map_runner_identity
from robot_sf.benchmark.map_runner.map_runner_batch_plan import build_seed_jobs
from robot_sf.benchmark.seed_bands import DEV_SEEDS
from robot_sf.common.artifact_paths import get_repository_root

if TYPE_CHECKING:
    from pathlib import Path


@pytest.mark.parametrize("builder", ["dispatch", "resume", "integrity"])
@pytest.mark.parametrize(
    "excluded_metadata",
    [
        {"supported": False},
        {"metadata": {"supported": False}},
        {"metadata": {"behavior": "wait"}},
    ],
)
def test_map_job_denominator_excludes_nonexecuting_scenarios(
    builder: str, excluded_metadata: dict[str, object]
) -> None:
    """Skipped map scenarios cannot keep a complete arm permanently incomplete."""
    scenarios = [
        {"name": "kept", "map_file": "unused.svg", "seeds": [1001]},
        {
            "name": "excluded",
            "map_file": "unused.svg",
            "seeds": [1002],
            **excluded_metadata,
        },
    ]
    if builder == "dispatch":
        jobs = build_seed_jobs(scenarios, suite_seeds={}, suite_key="default")
        actual = {(scenario["name"], seed) for scenario, seed in jobs}
    elif builder == "resume":
        actual = _expected_job_identities(scenarios)
    else:
        actual = _expected_episode_identities(scenarios, [1001, 1002])
    assert actual == {("kept", 1001)}


def test_run_campaign_rejects_empty_executable_job_set(tmp_path: Path) -> None:
    """An enabled arm and an entirely filtered matrix are insufficient for evidence."""
    matrix = tmp_path / "matrix.yaml"
    source_matrix = (
        get_repository_root()
        / "configs/scenarios/classic_interactions_francis2023_release_0_0_8_v1.yaml"
    )
    matrix.write_text(
        yaml.safe_dump(
            {
                "includes": [str(source_matrix)],
                "select_scenarios": ["francis2023_blind_corner"],
                "scenario_overrides": {"supported": False},
            }
        ),
        encoding="utf-8",
    )
    config = tmp_path / "campaign.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "name": "empty-executable-job-set",
                "paper_facing": False,
                "scenario_matrix": str(matrix),
                "seed_policy": {"mode": "fixed-list", "seeds": [DEV_SEEDS[0]]},
                "snqi_weights": "configs/benchmarks/snqi_weights_camera_ready_v3.json",
                "snqi_baseline": "tests/data/snqi/resume_metric_v2_baseline.json",
                "snqi_contract": {"enabled": False},
                "horizon": 1,
                "dt": 0.1,
                "workers": 1,
                "resume": True,
                "bootstrap_samples": 20,
                "bootstrap_seed": DEV_SEEDS[0],
                "kinematics_matrix": ["differential_drive"],
                "export_publication_bundle": False,
                "planners": [
                    {
                        "key": "social_force",
                        "algo": "social_force",
                        "benchmark_profile": "baseline-safe",
                        "algo_config": (
                            "configs/algos/social_force_resolution_independent_v2_"
                            "kernel_wrapped_v2.yaml"
                        ),
                    },
                ],
            }
        ),
        encoding="utf-8",
    )

    result = run_campaign(
        load_campaign_config(config),
        output_root=tmp_path,
        campaign_id="empty",
        skip_publication_bundle=True,
    )
    summary = json.loads(
        (tmp_path / "empty" / "reports" / "campaign_summary.json").read_text(encoding="utf-8")
    )
    assert result["benchmark_success"] is False
    assert summary["campaign"]["status"] == "integrity_failed"
    assert summary["campaign"]["evidence_status"] == "invalid"
    assert summary["campaign"]["total_episodes"] == 0
    integrity = summary["campaign_integrity"]
    assert integrity["expected_identity_count"] == 0
    assert [blocker["invariant"] for blocker in integrity["blockers"]] == ["no_executable_jobs"]


def test_map_integrity_uses_runtime_name_precedence() -> None:
    """Map rows use their name even when legacy id aliases are also present."""
    scenario = {
        "name": "runtime-name",
        "id": "legacy-id",
        "map_file": "unused.svg",
        "seeds": [1001],
    }
    assert _expected_episode_identities([scenario], []) == {("runtime-name", 1001)}


def test_fresh_campaign_rejects_duplicate_scenario_names(tmp_path: Path) -> None:
    """Fresh config admission must reject names that make resume lookup ambiguous."""
    matrix = tmp_path / "matrix.yaml"
    matrix.write_text(
        "- name: duplicate\n  map_file: unused.svg\n  seeds: [1001]\n"
        "- name: duplicate\n  map_file: unused.svg\n  seeds: [1002]\n",
        encoding="utf-8",
    )
    config = tmp_path / "campaign.yaml"
    config.write_text(
        f"name: duplicate-names\nscenario_matrix: {matrix}\n"
        "planners:\n  - key: goal\n    algo: goal\n",
        encoding="utf-8",
    )
    cfg = load_campaign_config(config)
    with pytest.raises(ValueError, match="duplicate scenario"):
        _load_campaign_scenarios(cfg)


def test_campaign_integrity_uses_matrix_seed_suite(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Integrity must select the same suite fallback as runtime provenance wiring."""
    monkeypatch.setattr(
        map_runner_identity,
        "_resolve_seed_list",
        lambda _path: {
            "default": [1001],
            "classic_interactions": [1002, 1003],
        },
    )
    cfg = SimpleNamespace(planners=[], scenario_matrix_path=tmp_path / "classic_matrix.yaml")
    integrity, _, _ = _post_run_integrity_and_fairness(
        cfg,
        manifest_payload={},
        run_entries=[],
        planner_rows=[],
        warnings=[],
        scenarios=[{"name": "kept", "map_file": "unused.svg"}],
        resolved_seeds=[],
        campaign_root=tmp_path,
    )
    assert integrity["expected_identity_count"] == 2


def test_metadata_job_readers_do_not_admit_seed_execution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Enumerating stored identities must preserve the dispatch-only seed guard."""
    scenarios = [{"name": "kept", "map_file": "unused.svg", "seeds": [1001]}]

    def execution_guard(_seed: int, *, boundary: str) -> None:
        assert boundary == "map seed dispatch"
        raise RuntimeError("execution admission reached")

    monkeypatch.setattr(map_runner_batch_plan, "check_simulation_seed", execution_guard)
    assert _expected_job_identities(scenarios) == {("kept", 1001)}
    assert _expected_episode_identities(scenarios, []) == {("kept", 1001)}
    with pytest.raises(RuntimeError, match="execution admission reached"):
        build_seed_jobs(scenarios, suite_seeds={}, suite_key="default")


def test_fresh_dispatch_rejects_duplicate_seed_jobs() -> None:
    """Fresh dispatch cannot create rows that identity-based resume must reject."""
    scenarios = [{"name": "kept", "map_file": "unused.svg", "seeds": [1001, 1001]}]
    with pytest.raises(ValueError, match="duplicate job identities"):
        build_seed_jobs(scenarios, suite_seeds={}, suite_key="default")
