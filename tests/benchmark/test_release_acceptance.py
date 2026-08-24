"""Tests for full benchmark-data release acceptance."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from robot_sf.benchmark import release_acceptance
from robot_sf.benchmark.release_acceptance import (
    _episode_horizon,
    _read_campaign_summary,
    _read_episode_rows,
    _resolve_expected_matrix_axes,
    _scenario_id,
    _source_commit,
    _status_markers,
    _strict_int,
    validate_full_benchmark_release_acceptance,
)

_PLANNER_KEYS = tuple(f"planner_{index:02d}" for index in range(14))
_PLANNER_ALGORITHMS = {
    planner_key: ("guarded_ppo" if planner_key == "planner_11" else planner_key)
    for planner_key in _PLANNER_KEYS
}
_SCENARIO_IDS = tuple(f"scenario_{index:02d}" for index in range(48))
_SEEDS = tuple(range(111, 141))
_SOURCE_SHA = "a" * 40


def _full_manifest() -> SimpleNamespace:
    """Return the fixed S30/H600 acceptance contract."""
    return SimpleNamespace(
        schema_version="benchmark-release-manifest.v0.2",
        expected_episode_cells=20_160,
        expected_horizon_steps=600,
        planner_keys=_PLANNER_KEYS,
        expected_kinematics_matrix=("differential_drive",),
        resolved_scenario_ids=_SCENARIO_IDS,
        resolved_seeds=_SEEDS,
        planner_algorithms=_PLANNER_ALGORITHMS,
    )


def _write_full_campaign(tmp_path: Path) -> Path:
    """Write a complete 14-arm fixture with 48 scenarios and 30 seeds."""
    campaign_root = tmp_path / "campaign"
    runs: list[dict[str, Any]] = []
    planner_rows: list[dict[str, Any]] = []
    for planner_key in _PLANNER_KEYS:
        expected_algo = _PLANNER_ALGORITHMS[planner_key]
        metadata_algorithm = "ppo" if expected_algo == "guarded_ppo" else expected_algo
        relative_path = Path("runs") / planner_key / "episodes.jsonl"
        episode_path = campaign_root / relative_path
        episode_path.parent.mkdir(parents=True, exist_ok=True)
        lines = []
        for scenario_index, scenario_id in enumerate(_SCENARIO_IDS):
            for seed in _SEEDS:
                lines.append(
                    json.dumps(
                        {
                            "episode_id": f"{planner_key}-{scenario_id}-{seed}",
                            "scenario_id": scenario_id,
                            "seed": seed,
                            "horizon": 600,
                            "status": "success",
                            "algo": expected_algo,
                            "git_hash": _SOURCE_SHA,
                            "result_provenance": {
                                "repo_commit": _SOURCE_SHA,
                                "config_hash": f"{scenario_index:016x}",
                                "scenario_id": scenario_id,
                                "seed": seed,
                                "simulator_settings": {"horizon": 600},
                            },
                            "algorithm_metadata": {
                                "algorithm": metadata_algorithm,
                                "canonical_algorithm": expected_algo,
                                "planner_contract": {"planner_id": expected_algo},
                                "status": "ok",
                            },
                        }
                    )
                )
        episode_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        runs.append(
            {
                "planner": {
                    "key": planner_key,
                    "kinematics": "differential_drive",
                    "horizon": 600,
                },
                "status": "ok",
                "episodes_path": relative_path.as_posix(),
                "summary": {"episodes_total": 1440, "written": 1440},
            }
        )
        planner_rows.append(
            {
                "planner_key": planner_key,
                "kinematics": "differential_drive",
                "status": "ok",
                "readiness_status": "available",
                "availability_status": "available",
                "benchmark_success": "true",
                "episodes": 1440,
            }
        )
    (campaign_root / "reports").mkdir(parents=True, exist_ok=True)
    (campaign_root / "reports" / "campaign_summary.json").write_text(
        json.dumps(
            {
                "campaign": {
                    "status": "benchmark_success",
                    "benchmark_success": True,
                    "evidence_status": "valid",
                    "campaign_execution_status": "completed",
                    "git_hash": _SOURCE_SHA,
                    "row_status_summary": {
                        "successful_evidence_rows": 14,
                        "accepted_unavailable_rows": 0,
                        "unexpected_failed_rows": 0,
                        "fallback_or_degraded_rows": 0,
                    },
                },
                "runs": runs,
                "planner_rows": planner_rows,
                "campaign_integrity": {
                    "status": "valid",
                    "benchmark_success_allowed": True,
                },
            }
        ),
        encoding="utf-8",
    )
    return campaign_root


def test_full_release_acceptance_requires_all_arms_and_episode_cells(tmp_path: Path) -> None:
    """A complete S30/H600 fixture is accepted as publication-grade evidence."""
    campaign_root = _write_full_campaign(tmp_path)

    result = validate_full_benchmark_release_acceptance(
        campaign_root,
        manifest=_full_manifest(),
    )

    assert result["status"] == "valid"
    assert result["benchmark_success"] is True
    assert result["successful_planner_arms"] == 14
    assert result["observed_episode_rows"] == 20_160
    assert result["unique_episode_identities"] == 20_160
    assert result["source_commits"] == [_SOURCE_SHA]
    assert result["blockers"] == []


def test_full_release_rejects_cross_arm_episode_algorithm_identity(tmp_path: Path) -> None:
    """A row copied between planner arms cannot retain a valid full-release receipt."""
    campaign_root = _write_full_campaign(tmp_path)
    episode_path = campaign_root / "runs" / "planner_00" / "episodes.jsonl"
    rows = [json.loads(line) for line in episode_path.read_text(encoding="utf-8").splitlines()]
    rows[0]["algo"] = "planner_01"
    rows[0]["algorithm_metadata"].update(
        {
            "algorithm": "planner_01",
            "canonical_algorithm": "planner_01",
            "planner_contract": {"planner_id": "planner_01"},
        }
    )
    episode_path.write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n",
        encoding="utf-8",
    )

    result = validate_full_benchmark_release_acceptance(campaign_root, manifest=_full_manifest())

    assert result["status"] == "invalid"
    assert any(
        "planner algorithm aliases do not match" in blocker for blocker in result["blockers"]
    )


def test_full_release_binds_same_algorithm_artifacts_to_exact_arms(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Arm-bound provenance rejects swaps between planners sharing one base algorithm."""
    campaign_root = _write_full_campaign(tmp_path)
    summary_path = campaign_root / "reports" / "campaign_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    planner_specs: list[SimpleNamespace] = []
    for run in summary["runs"]:
        planner_key = run["planner"]["key"]
        expected_algo = _PLANNER_ALGORITHMS[planner_key]
        if planner_key in {"planner_00", "planner_01"}:
            expected_algo = "hybrid_rule_local_planner"
        planner_specs.append(
            SimpleNamespace(key=planner_key, algo=expected_algo, algo_config_path=None)
        )
        old_path = campaign_root / run["episodes_path"]
        new_path = campaign_root / "runs" / f"{planner_key}__differential_drive" / "episodes.jsonl"
        new_path.parent.mkdir(parents=True, exist_ok=True)
        old_path.replace(new_path)
        old_path.parent.rmdir()
        run["episodes_path"] = new_path.relative_to(campaign_root).as_posix()
        if planner_key in {"planner_00", "planner_01"}:
            rows = [json.loads(line) for line in new_path.read_text(encoding="utf-8").splitlines()]
            for row in rows:
                row["algo"] = expected_algo
                row["algorithm_metadata"].update(
                    {
                        "algorithm": expected_algo,
                        "canonical_algorithm": expected_algo,
                        "planner_contract": {"planner_id": expected_algo},
                    }
                )
            new_path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    summary_path.write_text(json.dumps(summary), encoding="utf-8")

    scenario_path = tmp_path / "scenarios.yaml"
    scenario_path.write_text("scenarios: []\n", encoding="utf-8")
    resolved_scenarios = [{"id": scenario_id} for scenario_id in _SCENARIO_IDS]
    monkeypatch.setattr(
        release_acceptance, "_load_campaign_scenarios", lambda _cfg: resolved_scenarios
    )
    monkeypatch.setattr(
        release_acceptance, "_resolved_seed_inventory", lambda _scenarios: list(_SEEDS)
    )
    expected_hashes = {
        spec.key: release_acceptance.sha256_file(
            campaign_root / "runs" / f"{spec.key}__differential_drive" / "episodes.jsonl"
        )
        for spec in planner_specs
    }

    def _arm_bound_sidecar(episodes_path: Path, *, planner_key: str, **_kwargs: Any) -> list[str]:
        if release_acceptance.sha256_file(episodes_path) == expected_hashes[planner_key]:
            return []
        return [f"planner {planner_key} sidecar raw artifact hash is stale"]

    monkeypatch.setattr(
        release_acceptance, "_stress_episode_provenance_blockers", _arm_bound_sidecar
    )
    config = SimpleNamespace(planners=tuple(planner_specs), scenario_matrix_path=scenario_path)
    manifest = _full_manifest()
    baseline = validate_full_benchmark_release_acceptance(
        campaign_root, manifest=manifest, campaign_config=config
    )
    assert baseline["status"] == "valid"

    first = campaign_root / "runs" / "planner_00__differential_drive" / "episodes.jsonl"
    second = campaign_root / "runs" / "planner_01__differential_drive" / "episodes.jsonl"
    first_payload = first.read_bytes()
    second_payload = second.read_bytes()
    first.write_bytes(second_payload)
    second.write_bytes(first_payload)

    result = validate_full_benchmark_release_acceptance(
        campaign_root, manifest=manifest, campaign_config=config
    )

    assert result["status"] == "invalid"
    assert sum("sidecar raw artifact hash is stale" in item for item in result["blockers"]) == 2


def test_full_release_rejects_forged_guarded_ppo_metadata_on_other_arm(tmp_path: Path) -> None:
    """Guarded PPO telemetry cannot authorize an exception on a different arm."""
    campaign_root = _write_full_campaign(tmp_path)
    episode_path = campaign_root / "runs" / "planner_00" / "episodes.jsonl"
    rows = [json.loads(line) for line in episode_path.read_text(encoding="utf-8").splitlines()]
    rows[0]["algorithm_metadata"] = {
        "algorithm": "ppo",
        "canonical_algorithm": "guarded_ppo",
        "planner_contract": {"planner_id": "guarded_ppo"},
        "guard_stats": {"fallback_safe": 1},
    }
    episode_path.write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n",
        encoding="utf-8",
    )

    result = validate_full_benchmark_release_acceptance(campaign_root, manifest=_full_manifest())

    assert result["status"] == "invalid"
    assert result["forbidden_status_counts"]["1"] == 1
    assert any("fallback_safe" in blocker for blocker in result["blockers"])


def test_full_release_rejects_malformed_guarded_fallback_controller_state(tmp_path: Path) -> None:
    """The Guarded PPO exception requires a structured fallback-controller state."""
    campaign_root = _write_full_campaign(tmp_path)
    episode_path = campaign_root / "runs" / "planner_11" / "episodes.jsonl"
    rows = [json.loads(line) for line in episode_path.read_text(encoding="utf-8").splitlines()]
    rows[0]["algorithm_metadata"].update(
        {
            "guard_stats": {"fallback_safe": 1},
            "shield_stats": {"last_decision": {"fallback_controller_state": "not-a-mapping"}},
        }
    )
    episode_path.write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n",
        encoding="utf-8",
    )

    result = validate_full_benchmark_release_acceptance(campaign_root, manifest=_full_manifest())

    assert result["status"] == "invalid"
    assert any("fallback_controller_state" in blocker for blocker in result["blockers"])


def test_full_release_rejects_fallback_even_when_campaign_reports_success(tmp_path: Path) -> None:
    """A campaign's permissive core-success status cannot authorize publication."""
    campaign_root = _write_full_campaign(tmp_path)
    summary_path = campaign_root / "reports" / "campaign_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["runs"][0]["summary"]["benchmark_availability"] = {"readiness_status": "fallback"}
    summary_path.write_text(json.dumps(summary), encoding="utf-8")

    result = validate_full_benchmark_release_acceptance(campaign_root, manifest=_full_manifest())

    assert result["status"] == "invalid"
    assert result["benchmark_success"] is False
    assert result["forbidden_status_counts"]["fallback"] == 1
    assert any("fallback" in blocker for blocker in result["blockers"])


def test_full_release_rejects_episode_fallback_markers(tmp_path: Path) -> None:
    """Episode-level fallback markers cannot hide behind successful arm summaries."""
    campaign_root = _write_full_campaign(tmp_path)
    episode_path = campaign_root / "runs" / _PLANNER_KEYS[0] / "episodes.jsonl"
    rows = [json.loads(line) for line in episode_path.read_text(encoding="utf-8").splitlines()]
    rows[0]["fallback_triggered"] = True
    rows[1]["algorithm_metadata"] = {"planner_kinematics": {"execution_mode": "fallback"}}
    episode_path.write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n",
        encoding="utf-8",
    )

    result = validate_full_benchmark_release_acceptance(campaign_root, manifest=_full_manifest())

    assert result["status"] == "invalid"
    assert result["forbidden_status_counts"]["true"] == 1
    assert result["forbidden_status_counts"]["fallback"] == 1
    assert any("fallback_triggered" in blocker for blocker in result["blockers"])
    assert any("planner_kinematics.execution_mode" in blocker for blocker in result["blockers"])


@pytest.mark.parametrize(
    ("algorithm_metadata", "forbidden_count_key", "path_fragment"),
    [
        (
            {"status": "predictive_foresight_model_fallback"},
            "predictive_foresight_model_fallback",
            "algorithm_metadata.status",
        ),
        (
            {
                "status": "ok",
                "foresight_prediction": {"fallback_used": True},
            },
            "true",
            "algorithm_metadata.foresight_prediction.fallback_used",
        ),
    ],
)
def test_full_release_rejects_and_counts_foresight_fallback_metadata(
    tmp_path: Path,
    algorithm_metadata: dict[str, Any],
    forbidden_count_key: str,
    path_fragment: str,
) -> None:
    """Foresight fallback status and provenance cannot hide in a successful row."""
    campaign_root = _write_full_campaign(tmp_path)
    episode_path = campaign_root / "runs" / _PLANNER_KEYS[0] / "episodes.jsonl"
    rows = [json.loads(line) for line in episode_path.read_text(encoding="utf-8").splitlines()]
    rows[0]["algorithm_metadata"] = algorithm_metadata
    episode_path.write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n",
        encoding="utf-8",
    )

    result = validate_full_benchmark_release_acceptance(campaign_root, manifest=_full_manifest())

    assert result["status"] == "invalid"
    assert result["benchmark_success"] is False
    assert result["forbidden_status_counts"][forbidden_count_key] == 1
    assert any(path_fragment in blocker for blocker in result["blockers"])


def test_full_release_rejects_duplicate_planner_aggregate_roster(tmp_path: Path) -> None:
    """Aggregate rows must cover the exact unique manifest roster."""
    campaign_root = _write_full_campaign(tmp_path)
    summary_path = campaign_root / "reports" / "campaign_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["planner_rows"][0]["planner_key"] = _PLANNER_KEYS[1]
    summary_path.write_text(json.dumps(summary), encoding="utf-8")

    result = validate_full_benchmark_release_acceptance(campaign_root, manifest=_full_manifest())

    assert result["status"] == "invalid"
    assert any("planner aggregate rows do not match" in blocker for blocker in result["blockers"])


def test_full_release_requires_exact_campaign_source_sha(tmp_path: Path) -> None:
    """The campaign source SHA must be valid and equal to episode provenance."""
    campaign_root = _write_full_campaign(tmp_path)
    summary_path = campaign_root / "reports" / "campaign_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["campaign"]["git_hash"] = "b" * 40
    summary_path.write_text(json.dumps(summary), encoding="utf-8")

    result = validate_full_benchmark_release_acceptance(campaign_root, manifest=_full_manifest())

    assert result["status"] == "invalid"
    assert any("do not match campaign.git_hash" in blocker for blocker in result["blockers"])


def test_full_release_rejects_arbitrary_same_count_identity_product(tmp_path: Path) -> None:
    """Exact row count cannot replace the manifest-resolved scenario/seed product."""
    campaign_root = _write_full_campaign(tmp_path)
    episode_path = campaign_root / "runs" / _PLANNER_KEYS[0] / "episodes.jsonl"
    rows = [json.loads(line) for line in episode_path.read_text(encoding="utf-8").splitlines()]
    rows[-1]["scenario_id"] = "unregistered_scenario"
    episode_path.write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n",
        encoding="utf-8",
    )

    result = validate_full_benchmark_release_acceptance(campaign_root, manifest=_full_manifest())

    assert result["status"] == "invalid"
    assert result["observed_episode_rows"] == 20_160
    assert result["unique_episode_identities"] == 20_160
    assert result["missing_episode_identities"] == 1
    assert result["unexpected_episode_identities"] == 1
    assert any("exact manifest-resolved" in blocker for blocker in result["blockers"])


def test_full_release_rejects_duplicate_or_missing_episode_identity(tmp_path: Path) -> None:
    """A 20,160-row count is insufficient when logical episode coverage is duplicated."""
    campaign_root = _write_full_campaign(tmp_path)
    episode_path = campaign_root / "runs" / _PLANNER_KEYS[0] / "episodes.jsonl"
    lines = episode_path.read_text(encoding="utf-8").splitlines()
    lines[-1] = lines[-2]
    episode_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    result = validate_full_benchmark_release_acceptance(campaign_root, manifest=_full_manifest())

    assert result["status"] == "invalid"
    assert result["unique_episode_identities"] == 20_159
    assert any("duplicate episode identity" in blocker for blocker in result["blockers"])


def test_release_acceptance_helpers_are_strict_about_shapes_and_provenance(tmp_path: Path) -> None:
    """Low-level readers and coercions reject malformed release evidence deterministically."""
    assert _strict_int(True) is None
    assert _strict_int(" 12 ") == 12
    assert _strict_int("not-an-int") is None
    assert _source_commit({"git_hash": " ABC "}) == "abc"
    assert _source_commit({"result_provenance": {"repo_commit": "DEF"}, "git_hash": "ABC"}) == "def"
    assert _episode_horizon({"horizon": "600"}) == (600, True)
    assert _episode_horizon({"result_provenance": {"simulator_settings": {"horizon": 600}}}) == (
        600,
        True,
    )
    assert _episode_horizon({"result_provenance": {"simulator_settings": {}}}) == (None, False)
    assert _scenario_id({"id": "primary", "scenario_id": "secondary"}) == "primary"
    assert _scenario_id({"scenario_id": "secondary"}) == "secondary"
    assert _scenario_id({"name": "named"}) == "named"
    assert _scenario_id({}) == ""

    markers = _status_markers(
        {
            "row_status": "degraded",
            "readiness_status": "failed",
            "availability_status": "unavailable",
            "evidence_status": "excluded",
            "execution_status": "not-available",
            "benchmark_success": "no",
            "degraded": True,
            "algorithm_metadata": {
                "status": "error",
                "fallback_or_degraded": True,
                "planner_kinematics": {"execution_mode": "fallback"},
                "adapter_impact": {"execution_mode": "degraded"},
            },
            "algorithm_metadata_contract": {"status": "fallback"},
            "benchmark_availability": {
                "status": "failed",
                "readiness_status": "unavailable",
                "availability_status": "excluded",
                "execution_mode": "fallback",
            },
        },
        "row",
    )
    marker_values = {value for _, value in markers}
    assert {
        "degraded",
        "failed",
        "unavailable",
        "excluded",
        "not-available",
        "false",
        "true",
        "error",
        "fallback",
    } <= marker_values

    missing_summary, missing_error = _read_campaign_summary(tmp_path / "missing")
    assert missing_summary is None
    assert missing_error and "cannot be read" in missing_error
    report_dir = tmp_path / "reports"
    report_dir.mkdir()
    (report_dir / "campaign_summary.json").write_text("[]", encoding="utf-8")
    object_summary, object_error = _read_campaign_summary(tmp_path)
    assert object_summary is None
    assert object_error == "campaign summary must be a JSON object"

    episode_path = tmp_path / "episodes.jsonl"
    episode_path.write_text("\n[]\n", encoding="utf-8")
    rows, row_error = _read_episode_rows(episode_path)
    assert rows == []
    assert row_error and "episode row must be an object" in row_error
    episode_path.write_text("{malformed}\n", encoding="utf-8")
    rows, row_error = _read_episode_rows(episode_path)
    assert rows == []
    assert row_error and "invalid JSON" in row_error
    rows, row_error = _read_episode_rows(tmp_path / "missing-episodes.jsonl")
    assert rows == []
    assert row_error and "cannot read episode artifact" in row_error


def test_release_acceptance_resolves_config_axes_and_rejects_drift(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Resolved config axes are checked against the manifest instead of trusted blindly."""
    monkeypatch.setattr(
        release_acceptance,
        "_load_campaign_scenarios",
        lambda _config: [{"id": "first"}, {"scenario_id": "second"}, {"name": "third"}, {}],
    )
    monkeypatch.setattr(
        release_acceptance,
        "_resolved_seed_inventory",
        lambda _scenarios: (1, 2, 3),
    )
    manifest = SimpleNamespace(resolved_seeds=(1, "bad"))

    scenario_ids, seeds, blockers = _resolve_expected_matrix_axes(manifest, object())

    assert scenario_ids == ("first", "second", "third", "")
    assert seeds == (1, 2, 3)
    assert "empty scenario identifier" in " ".join(blockers)
    assert "resolved seeds do not match" in " ".join(blockers)


def test_release_acceptance_handles_unavailable_config_and_legacy_manifest(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Missing canonical inputs and legacy manifests remain explicit non-success states."""
    monkeypatch.setattr(
        release_acceptance,
        "load_campaign_config",
        lambda _path: (_ for _ in ()).throw(OSError("missing config")),
    )
    manifest = SimpleNamespace(
        canonical_campaign_config_path=tmp_path / "missing.yaml",
        resolved_scenario_ids=(),
        resolved_seeds=(),
    )
    scenario_ids, seeds, blockers = _resolve_expected_matrix_axes(manifest, None)
    assert scenario_ids == ()
    assert seeds == ()
    assert "cannot be resolved" in " ".join(blockers)
    assert "axes are unavailable" in " ".join(blockers)

    legacy = validate_full_benchmark_release_acceptance(
        tmp_path,
        manifest=SimpleNamespace(schema_version="benchmark-release-manifest.v0.1"),
    )
    assert legacy["status"] == "not_applicable"
    assert legacy["benchmark_success"] is False


def test_release_acceptance_rejects_malformed_run_and_aggregate_rows(tmp_path: Path) -> None:
    """Malformed run and aggregate rows cannot be promoted by matching top-level counts."""
    campaign_root = tmp_path / "campaign"
    (campaign_root / "reports").mkdir(parents=True)
    episode_path = campaign_root / "runs" / "planner_00" / "episodes.jsonl"
    episode_path.parent.mkdir(parents=True)
    episode_path.write_text(
        "\n".join(
            [
                json.dumps({"status": "failed", "seed": "bad"}),
                json.dumps({"scenario_id": "scenario_00", "seed": 1, "git_hash": "bad"}),
                json.dumps({"scenario_id": "scenario_00", "seed": 1, "git_hash": "bad"}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    summary = {
        "campaign": {
            "status": "not-ready",
            "benchmark_success": False,
            "evidence_status": "invalid",
            "campaign_execution_status": "failed",
            "git_hash": "bad",
            "row_status_summary": {"successful_evidence_rows": "bad"},
        },
        "campaign_integrity": {"status": "invalid"},
        "runs": [
            None,
            {
                "planner": {},
                "status": "failed",
                "summary": {
                    "benchmark_success": False,
                    "failed_jobs": 2,
                    "failures": ["boom"],
                },
            },
            {
                "planner": {"key": "planner_00", "kinematics": "differential_drive", "horizon": 0},
                "status": "ok",
                "episodes_path": "../outside.jsonl",
            },
            {
                "planner": {
                    "key": "planner_00",
                    "kinematics": "differential_drive",
                    "horizon": 600,
                },
                "status": "ok",
                "episodes_path": "runs/planner_00/episodes.jsonl",
                "summary": {"written": 0},
            },
        ],
        "planner_rows": [
            None,
            {
                "planner_key": "outside",
                "kinematics": "differential_drive",
                "status": "failed",
                "episodes": 0,
                "benchmark_success": False,
            },
        ],
    }
    (campaign_root / "reports" / "campaign_summary.json").write_text(
        json.dumps(summary),
        encoding="utf-8",
    )

    result = validate_full_benchmark_release_acceptance(campaign_root, manifest=_full_manifest())

    assert result["status"] == "invalid"
    assert result["observed_episode_rows"] == 3
    assert result["unique_episode_identities"] == 1
    assert any("runs[0] must be an object" in blocker for blocker in result["blockers"])
    assert any("episodes_path rejected" in blocker for blocker in result["blockers"])
    assert any("duplicate episode identity" in blocker for blocker in result["blockers"])
    assert any("planner_rows[0] must be an object" in blocker for blocker in result["blockers"])
    assert any("outside the manifest roster" in blocker for blocker in result["blockers"])


def test_release_acceptance_bounds_duplicate_blockers() -> None:
    """Repeated row errors remain bounded and deterministic."""
    blockers: list[str] = []
    for _ in range(150):
        release_acceptance._append_blocker(blockers, "same blocker")
    for index in range(150):
        release_acceptance._append_blocker(blockers, f"blocker-{index}")

    assert blockers[0] == "same blocker"
    assert len(blockers) == 100
