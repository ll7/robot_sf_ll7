"""Regression coverage for fixed-ID camera-ready campaign resume scheduling."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from robot_sf.benchmark.camera_ready_campaign import load_campaign_config, run_campaign


def _fake_aggregates(*args: Any, **kwargs: Any) -> dict[str, Any]:
    """Return the smallest aggregate payload accepted by campaign reporting."""
    del args, kwargs
    metrics = {
        "success": {"mean": 1.0, "mean_ci": [1.0, 1.0]},
        "collisions": {"mean": 0.0, "mean_ci": [0.0, 0.0]},
        "near_misses": {"mean": 0.0, "mean_ci": [0.0, 0.0]},
        "time_to_goal_norm": {"mean": 0.5, "mean_ci": [0.5, 0.5]},
        "path_efficiency": {"mean": 1.0, "mean_ci": [1.0, 1.0]},
        "comfort_exposure": {"mean": 0.0, "mean_ci": [0.0, 0.0]},
        "jerk_mean": {"mean": 0.0, "mean_ci": [0.0, 0.0]},
        "snqi": {"mean": 1.0, "mean_ci": [1.0, 1.0]},
    }
    return {"synthetic": metrics, "_meta": {"warnings": [], "missing_algorithms": []}}


def _episode_rows(path: Path) -> list[dict[str, Any]]:
    """Load non-empty JSONL rows from *path*."""
    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]


def _write_resume_config(tmp_path: Path):
    """Create a two-arm fixed-ID campaign configuration for resume tests."""
    scenario_path = tmp_path / "scenarios.yaml"
    scenario_path.write_text(
        "\n".join(
            [
                "- id: unit-a",
                "  name: unit-a",
                "  map_file: maps/svg_maps/classic_crossing.svg",
                "  seeds: [101]",
                "- id: unit-b",
                "  name: unit-b",
                "  map_file: maps/svg_maps/classic_crossing.svg",
                "  seeds: [101]",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    config_path = tmp_path / "campaign.yaml"
    config_path.write_text(
        "\n".join(
            [
                "name: resume_scheduler_regression",
                f"scenario_matrix: {scenario_path.as_posix()}",
                "paper_facing: false",
                "preview_scenario_limit: 0",
                "resume: true",
                "export_publication_bundle: false",
                "seed_policy:",
                "  mode: fixed-list",
                "  seeds: [101]",
                "planners:",
                "  - key: complete",
                "    algo: complete",
                "    planner_group: core",
                "  - key: partial",
                "    algo: partial",
                "    planner_group: core",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return load_campaign_config(config_path)


def _install_campaign_stubs(monkeypatch, dispatched_arms, executed_units) -> None:
    """Install deterministic runner and aggregation stubs for campaign tests."""

    def _fake_run_batch(
        scenarios_or_path: list[dict[str, Any]],
        out_path: str | Path,
        schema_path: str | Path,
        *,
        algo: str,
        benchmark_profile: str,
        resume: bool,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Emulate runner-level unit filtering while recording arm dispatch."""
        del schema_path, kwargs
        dispatched_arms.append(algo)
        output_path = Path(out_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        prior_rows = _episode_rows(output_path) if resume and output_path.is_file() else []
        prior_ids = {str(row["episode_id"]) for row in prior_rows}

        missing_rows: list[dict[str, Any]] = []
        for scenario in scenarios_or_path:
            scenario_id = str(scenario.get("id") or scenario.get("name"))
            seed = int((scenario.get("seeds") or [101])[0])
            episode_id = f"{scenario_id}:{seed}"
            if episode_id in prior_ids:
                continue
            executed_units.append((algo, episode_id))
            missing_rows.append(
                {
                    "episode_id": episode_id,
                    "scenario_id": scenario_id,
                    "seed": seed,
                    "scenario_params": {
                        "algo": algo,
                        "metadata": {"archetype": "crossing"},
                    },
                    "metrics": {
                        "success": 1.0,
                        "collisions": 0.0,
                        "near_misses": 0.0,
                    },
                    "algorithm_metadata": {"algorithm": algo, "status": "ok"},
                }
            )

        with output_path.open("a", encoding="utf-8") as handle:
            for row in missing_rows:
                handle.write(json.dumps(row) + "\n")
        return {
            "total_jobs": len(missing_rows),
            "written": len(missing_rows),
            "failed_jobs": 0,
            "failures": [],
            "out_path": str(output_path),
            "algorithm_readiness": {
                "name": algo,
                "tier": "baseline-ready",
                "profile": benchmark_profile,
            },
            "preflight": {
                "status": "ok",
                "learned_policy_contract": {"status": "not_applicable"},
            },
        }

    monkeypatch.setattr(
        "robot_sf.benchmark.camera_ready_campaign.run_batch",
        _fake_run_batch,
    )
    monkeypatch.setattr(
        "robot_sf.benchmark.camera_ready_campaign.compute_aggregates_with_ci",
        _fake_aggregates,
    )


def _initial_campaign(tmp_path: Path, monkeypatch):
    """Run the fixture once and return its paths plus dispatch observability."""
    cfg = _write_resume_config(tmp_path)
    dispatched_arms: list[str] = []
    executed_units: list[tuple[str, str]] = []
    _install_campaign_stubs(monkeypatch, dispatched_arms, executed_units)
    first = run_campaign(cfg, output_root=tmp_path / "campaigns", label="initial")
    campaign_root = Path(first["campaign_root"])
    complete_path = campaign_root / "runs" / "complete__differential_drive" / "episodes.jsonl"
    partial_path = campaign_root / "runs" / "partial__differential_drive" / "episodes.jsonl"
    assert len(_episode_rows(complete_path)) == len(_episode_rows(partial_path)) == 2
    return cfg, first, complete_path, partial_path, dispatched_arms, executed_units


def test_fixed_campaign_id_resume_schedules_only_missing_units(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """A half-complete resume must not dispatch or duplicate completed units."""
    cfg, first, complete_path, partial_path, dispatched_arms, executed_units = _initial_campaign(
        tmp_path, monkeypatch
    )
    complete_rows = _episode_rows(complete_path)
    partial_rows = _episode_rows(partial_path)
    missing_row = partial_rows.pop()
    partial_path.write_text(
        "".join(json.dumps(row) + "\n" for row in partial_rows),
        encoding="utf-8",
    )
    completed_unit_keys = {
        (planner, str(row["episode_id"]))
        for planner, rows in (("complete", complete_rows), ("partial", partial_rows))
        for row in rows
    }

    dispatched_arms.clear()
    executed_units.clear()
    run_campaign(
        cfg,
        output_root=tmp_path / "campaigns",
        campaign_id=str(first["campaign_id"]),
    )

    assert dispatched_arms == ["partial"]
    assert executed_units == [("partial", str(missing_row["episode_id"]))]
    assert completed_unit_keys.isdisjoint(executed_units)
    all_episode_keys = [
        (path.parent.name, str(row["episode_id"]))
        for path in (complete_path, partial_path)
        for row in _episode_rows(path)
    ]
    assert len(all_episode_keys) == len(set(all_episode_keys))


def test_fully_complete_fixed_campaign_resume_dispatches_no_arms(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """A fully complete fixed-ID resume must reuse both arms without dispatch."""
    cfg, first, complete_path, partial_path, dispatched_arms, executed_units = _initial_campaign(
        tmp_path, monkeypatch
    )
    before = {str(path): _episode_rows(path) for path in (complete_path, partial_path)}
    dispatched_arms.clear()
    executed_units.clear()

    run_campaign(
        cfg,
        output_root=tmp_path / "campaigns",
        campaign_id=str(first["campaign_id"]),
    )

    assert dispatched_arms == []
    assert executed_units == []
    assert {str(path): _episode_rows(path) for path in (complete_path, partial_path)} == before


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("missing_summary", "requires a JSON-object summary"),
        ("malformed_episode", "Malformed JSON"),
        ("incompatible_context", "config-hash mismatch"),
    ],
)
def test_resume_fails_closed_before_dispatch_for_invalid_state(
    tmp_path: Path,
    monkeypatch,
    mutation: str,
    message: str,
) -> None:
    """Missing summaries, malformed rows, and context drift never dispatch an arm."""
    cfg, first, complete_path, _, dispatched_arms, executed_units = _initial_campaign(
        tmp_path, monkeypatch
    )
    campaign_root = Path(first["campaign_root"])
    if mutation == "missing_summary":
        (complete_path.parent / "summary.json").unlink()
    elif mutation == "malformed_episode":
        with complete_path.open("a", encoding="utf-8") as handle:
            handle.write("{malformed json\n")
    else:
        manifest_path = campaign_root / "campaign_manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["config_hash"] = "incompatible-config"
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    dispatched_arms.clear()
    executed_units.clear()
    with pytest.raises(ValueError, match=message):
        run_campaign(
            cfg,
            output_root=tmp_path / "campaigns",
            campaign_id=str(first["campaign_id"]),
        )

    assert dispatched_arms == []
    assert executed_units == []
