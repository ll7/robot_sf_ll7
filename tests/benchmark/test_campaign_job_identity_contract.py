"""Map dispatch, resume, and integrity must count the same executable jobs."""

from __future__ import annotations

from types import SimpleNamespace
from typing import TYPE_CHECKING

import pytest

from robot_sf.benchmark.camera_ready._config import _load_campaign_scenarios
from robot_sf.benchmark.camera_ready._resume_plan import _expected_job_identities
from robot_sf.benchmark.camera_ready._run_state import _expected_episode_identities
from robot_sf.benchmark.camera_ready.campaign import _post_run_integrity_and_fairness
from robot_sf.benchmark.camera_ready_campaign import load_campaign_config
from robot_sf.benchmark.map_runner import map_runner_batch_plan, map_runner_identity
from robot_sf.benchmark.map_runner.map_runner_batch_plan import build_seed_jobs

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
        {"name": "kept", "map_file": "unused.svg", "seeds": [100000]},
        {
            "name": "excluded",
            "map_file": "unused.svg",
            "seeds": [100001],
            **excluded_metadata,
        },
    ]
    if builder == "dispatch":
        jobs = build_seed_jobs(scenarios, suite_seeds={}, suite_key="default")
        actual = {(scenario["name"], seed) for scenario, seed in jobs}
    elif builder == "resume":
        actual = _expected_job_identities(scenarios)
    else:
        actual = _expected_episode_identities(scenarios, [100000, 100001])
    assert actual == {("kept", 100000)}


def test_map_integrity_uses_runtime_name_precedence() -> None:
    """Map rows use their name even when legacy id aliases are also present."""
    scenario = {
        "name": "runtime-name",
        "id": "legacy-id",
        "map_file": "unused.svg",
        "seeds": [100000],
    }
    assert _expected_episode_identities([scenario], []) == {("runtime-name", 100000)}


def test_fresh_campaign_rejects_duplicate_scenario_names(tmp_path: Path) -> None:
    """Fresh config admission must reject names that make resume lookup ambiguous."""
    matrix = tmp_path / "matrix.yaml"
    matrix.write_text(
        "- name: duplicate\n  map_file: unused.svg\n  seeds: [100000]\n"
        "- name: duplicate\n  map_file: unused.svg\n  seeds: [100001]\n",
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
            "default": [100000],
            "classic_interactions": [100001, 100002],
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
    scenarios = [{"name": "kept", "map_file": "unused.svg", "seeds": [100000]}]

    def execution_guard(_seed: int, *, boundary: str) -> None:
        assert boundary == "map seed dispatch"
        raise RuntimeError("execution admission reached")

    monkeypatch.setattr(map_runner_batch_plan, "check_simulation_seed", execution_guard)
    assert _expected_job_identities(scenarios) == {("kept", 100000)}
    assert _expected_episode_identities(scenarios, []) == {("kept", 100000)}
    with pytest.raises(RuntimeError, match="execution admission reached"):
        build_seed_jobs(scenarios, suite_seeds={}, suite_key="default")


def test_fresh_dispatch_rejects_duplicate_seed_jobs() -> None:
    """Fresh dispatch cannot create rows that identity-based resume must reject."""
    scenarios = [{"name": "kept", "map_file": "unused.svg", "seeds": [100000, 100000]}]
    with pytest.raises(ValueError, match="duplicate job identities"):
        build_seed_jobs(scenarios, suite_seeds={}, suite_key="default")
