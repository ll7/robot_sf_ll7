"""Tests for full benchmark-data release acceptance."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from robot_sf.benchmark import release_acceptance
from robot_sf.benchmark.identity.hash_utils import sha256_file
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
from robot_sf.benchmark.result_provenance import (
    build_result_provenance_manifest,
    write_result_provenance_manifest,
)
from robot_sf.common.artifact_paths import get_repository_root

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
        canonical_campaign_config_path=Path("unavailable-test-campaign.yaml"),
    )


def test_full_release_roster_resolution_helpers_fail_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Canonical config and planner-roster resolution reject ambiguous inputs."""
    assert release_acceptance._full_release_planner_items(None) == ()
    assert release_acceptance._full_release_planner_items([{"key": "arm", "algo": "goal"}]) == (
        ("arm", "goal"),
    )

    resolved = SimpleNamespace(planners=())
    monkeypatch.setattr(release_acceptance, "load_campaign_config", lambda _path: resolved)
    manifest = SimpleNamespace(canonical_campaign_config_path=Path("campaign.yaml"))
    assert release_acceptance._full_release_campaign_config(manifest, None) == (resolved, [])
    monkeypatch.setattr(
        release_acceptance,
        "load_campaign_config",
        lambda _path: (_ for _ in ()).throw(ValueError("bad config")),
    )
    config, blockers = release_acceptance._full_release_campaign_config(manifest, None)
    assert config is None
    assert blockers == ["canonical campaign config cannot be resolved for provenance"]
    missing_config, missing_blockers = release_acceptance._full_release_campaign_config(
        SimpleNamespace(), None
    )
    assert missing_config is None
    assert missing_blockers == ["canonical campaign config is required for full-release provenance"]

    roster_manifest = SimpleNamespace(
        planner_algorithms=[
            {"key": "arm_a", "algo": "goal"},
            {"key": "arm_a", "algo": "orca"},
            {"key": "", "algo": "goal"},
            {"key": "unexpected", "algo": "social_force"},
        ]
    )
    algorithms, roster_blockers = release_acceptance._full_release_algorithm_roster(
        roster_manifest, None, ("arm_a", "missing")
    )
    assert algorithms == {"arm_a": "goal", "unexpected": "social_force"}
    assert any("empty key or algo" in blocker for blocker in roster_blockers)
    assert any("conflicts for 'arm_a'" in blocker for blocker in roster_blockers)
    assert any("is missing ['missing']" in blocker for blocker in roster_blockers)
    assert any("has unexpected ['unexpected']" in blocker for blocker in roster_blockers)


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


def _write_provenance_bound_full_campaign(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    shared_first_algorithm: bool = False,
    telemetry: dict[str, str] | None = None,
) -> tuple[Path, SimpleNamespace]:
    """Write a full fixture with the same sidecars and arm paths as production."""
    campaign_root = _write_full_campaign(tmp_path)
    summary_path = campaign_root / "reports" / "campaign_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    source_repository_root = tmp_path / "frozen-source"
    schema_path = source_repository_root / "robot_sf/benchmark/schemas/episode.schema.v1.json"
    schema_path.parent.mkdir(parents=True, exist_ok=True)
    schema_path.write_bytes(
        (get_repository_root() / "robot_sf/benchmark/schemas/episode.schema.v1.json").read_bytes()
    )
    scenario_path = (
        source_repository_root / "configs/scenarios/classic_interactions_francis2023.yaml"
    )
    scenario_path.parent.mkdir(parents=True, exist_ok=True)
    scenario_path.write_text("scenarios: []\n", encoding="utf-8")
    resolved_scenarios = [{"id": scenario_id} for scenario_id in _SCENARIO_IDS]
    monkeypatch.setattr(
        release_acceptance, "_load_campaign_scenarios", lambda _cfg: resolved_scenarios
    )
    monkeypatch.setattr(
        release_acceptance, "_resolved_seed_inventory", lambda _scenarios: list(_SEEDS)
    )
    effective_scenarios = [
        release_acceptance._scenario_with_kinematics(
            scenario,
            kinematics="differential_drive",
            holonomic_command_mode="vx_vy",
        )
        for scenario in resolved_scenarios
    ]
    if telemetry is not None:
        for scenario in effective_scenarios:
            scenario["telemetry"] = dict(telemetry)
            scenario["metadata"] = {"telemetry": dict(telemetry)}
    planner_specs: list[SimpleNamespace] = []
    for run in summary["runs"]:
        planner_key = run["planner"]["key"]
        expected_algo = _PLANNER_ALGORITHMS[planner_key]
        algo_config_path: Path | None = None
        if shared_first_algorithm and planner_key in {"planner_00", "planner_01"}:
            expected_algo = "hybrid_rule_local_planner"
            algo_config_path = source_repository_root / "configs/algos" / f"{planner_key}.yaml"
            algo_config_path.parent.mkdir(parents=True, exist_ok=True)
            algo_config_path.write_text(f"planner_key: {planner_key}\n", encoding="utf-8")
        planner_specs.append(
            SimpleNamespace(
                key=planner_key,
                algo=expected_algo,
                algo_config_path=algo_config_path,
            )
        )
        old_path = campaign_root / run["episodes_path"]
        episode_path = (
            campaign_root / "runs" / f"{planner_key}__differential_drive" / "episodes.jsonl"
        )
        episode_path.parent.mkdir(parents=True, exist_ok=True)
        old_path.replace(episode_path)
        old_path.parent.rmdir()
        rows = [json.loads(line) for line in episode_path.read_text(encoding="utf-8").splitlines()]
        for row in rows:
            row["config_hash"] = row["result_provenance"]["config_hash"]
            if shared_first_algorithm and planner_key in {"planner_00", "planner_01"}:
                row["algo"] = expected_algo
                row["algorithm_metadata"].update(
                    {
                        "algorithm": expected_algo,
                        "canonical_algorithm": expected_algo,
                        "planner_contract": {"planner_id": expected_algo},
                    }
                )
        episode_path.write_text(
            "\n".join(json.dumps(row) for row in rows) + "\n",
            encoding="utf-8",
        )
        sidecar = build_result_provenance_manifest(
            out_path=episode_path,
            episode_records=rows,
            schema_path=schema_path,
            scenario_path=scenario_path,
            scenarios=effective_scenarios,
            algo=expected_algo,
            algo_config_path=algo_config_path,
            benchmark_profile="release-acceptance-test",
            suite_key="classic_interactions",
            total_jobs=len(rows),
            written=len(rows),
            horizon=600,
            dt=0.1,
            record_forces=False,
            active_observation_mode=None,
            active_observation_level=None,
        )
        sidecar["run"]["repo_commit"] = _SOURCE_SHA
        write_result_provenance_manifest(
            episode_path.with_name(f"{episode_path.name}.provenance.json"), sidecar
        )
        run["episodes_path"] = episode_path.relative_to(campaign_root).as_posix()
    summary_path.write_text(json.dumps(summary), encoding="utf-8")
    config = SimpleNamespace(
        planners=tuple(planner_specs),
        scenario_matrix_path=scenario_path,
        holonomic_command_mode="vx_vy",
        telemetry=telemetry,
        source_repository_root=source_repository_root,
    )
    return campaign_root, config


def _write_effective_component_campaign(  # noqa: C901
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    override_mode: str = "valid",
) -> tuple[Path, SimpleNamespace]:
    """Build a full fixture whose first arm selects ORCA for one canonical scenario."""
    campaign_root, config = _write_provenance_bound_full_campaign(tmp_path, monkeypatch)
    candidate_path = (
        config.source_repository_root / "configs/policy_search/scenario_adaptive_candidate.yaml"
    )
    candidate_path.parent.mkdir(parents=True, exist_ok=True)
    base_config_path = config.source_repository_root / "configs/algos/hybrid.yaml"
    override_config_path = config.source_repository_root / "configs/algos/orca.yaml"
    base_config_path.parent.mkdir(parents=True, exist_ok=True)
    base_config_path.write_text("planner: hybrid\n", encoding="utf-8")
    override_config_path.write_text("planner: orca\n", encoding="utf-8")
    if override_mode == "valid":
        override_payload = (
            "scenario_algo_overrides:\n"
            "  scenario_00:\n"
            "    algo: orca\n"
            "    base_config_path: configs/algos/orca.yaml\n"
        )
    elif override_mode == "malformed":
        override_payload = "scenario_algo_overrides:\n  scenario_00: orca\n"
    elif override_mode == "malformed_base":
        override_payload = "scenario_algo_overrides:\n  scenario_00: orca\n"
    elif override_mode == "external":
        external_config_path = tmp_path / "external.yaml"
        external_config_path.write_text("planner: external\n", encoding="utf-8")
        override_payload = (
            "scenario_algo_overrides:\n"
            "  scenario_00:\n"
            f"    base_config_path: {external_config_path}\n"
        )
    elif override_mode == "symlink":
        symlink_config_path = config.source_repository_root / "configs/algos/orca-link.yaml"
        symlink_config_path.symlink_to(override_config_path)
        override_payload = (
            "scenario_algo_overrides:\n"
            "  scenario_00:\n"
            "    algo: orca\n"
            "    base_config_path: configs/algos/orca-link.yaml\n"
        )
    elif override_mode == "unknown":
        override_payload = "scenario_algo_overrides:\n  unknown_scenario:\n    algo: orca\n"
    else:
        override_payload = ""
    candidate_path.write_text(
        "algo: hybrid_rule_local_planner\n"
        "base_config_path: configs/algos/hybrid.yaml\n"
        "params: {}\n" + override_payload,
        encoding="utf-8",
    )
    planner_specs = []
    for planner in config.planners:
        if planner.key != "planner_00":
            planner_specs.append(planner)
            continue
        planner_specs.append(
            SimpleNamespace(
                key=planner.key,
                algo="hybrid_rule_local_planner",
                algo_config_path=candidate_path,
            )
        )
    config.planners = tuple(planner_specs)

    episode_path = campaign_root / "runs" / "planner_00__differential_drive" / "episodes.jsonl"
    rows = [json.loads(line) for line in episode_path.read_text(encoding="utf-8").splitlines()]
    for row in rows:
        row["algo"] = "hybrid_rule_local_planner"
        row["algorithm_metadata"].update(
            {
                "algorithm": "hybrid_rule_local_planner",
                "canonical_algorithm": "hybrid_rule_local_planner",
                "planner_contract": {"planner_id": "hybrid_rule_local_planner"},
            }
        )
        if row["scenario_id"] == "scenario_00" and override_mode not in {
            "malformed_base",
            "external",
            "unknown",
        }:
            row["algo"] = "orca"
            row["algorithm_metadata"].update(
                {
                    "algorithm": "orca",
                    "canonical_algorithm": "orca",
                    "planner_contract": {"planner_id": "orca"},
                }
            )
    episode_path.write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n",
        encoding="utf-8",
    )
    sidecar = build_result_provenance_manifest(
        out_path=episode_path,
        episode_records=rows,
        schema_path=(
            config.source_repository_root / "robot_sf/benchmark/schemas/episode.schema.v1.json"
        ),
        scenario_path=config.scenario_matrix_path,
        scenarios=release_acceptance._result_provenance_scenarios(
            config,
            [{"id": scenario_id} for scenario_id in _SCENARIO_IDS],
            kinematics="differential_drive",
        ),
        algo="hybrid_rule_local_planner",
        algo_config_path=candidate_path,
        benchmark_profile="release-acceptance-test",
        suite_key="classic_interactions",
        total_jobs=len(rows),
        written=len(rows),
        horizon=600,
        dt=0.1,
        record_forces=False,
        active_observation_mode=None,
        active_observation_level=None,
    )
    sidecar["run"]["repo_commit"] = _SOURCE_SHA
    write_result_provenance_manifest(
        episode_path.with_name(f"{episode_path.name}.provenance.json"), sidecar
    )
    return campaign_root, config


def test_full_release_acceptance_requires_all_arms_and_episode_cells(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A complete S30/H600 fixture is accepted as publication-grade evidence."""
    campaign_root, config = _write_provenance_bound_full_campaign(tmp_path, monkeypatch)

    result = validate_full_benchmark_release_acceptance(
        campaign_root,
        manifest=_full_manifest(),
        campaign_config=config,
        source_repository_root=config.source_repository_root,
    )

    assert result["status"] == "valid"
    assert result["benchmark_success"] is True
    assert result["successful_planner_arms"] == 14
    assert result["observed_episode_rows"] == 20_160
    assert result["unique_episode_identities"] == 20_160
    assert result["source_commits"] == [_SOURCE_SHA]
    assert result["blockers"] == []


def test_full_release_acceptance_uses_frozen_source_repository_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A validator checkout must resolve canonical sidecars against the execution checkout."""
    campaign_root, config = _write_provenance_bound_full_campaign(tmp_path, monkeypatch)

    default_result = validate_full_benchmark_release_acceptance(
        campaign_root,
        manifest=_full_manifest(),
        campaign_config=config,
    )
    assert default_result["status"] == "invalid"
    assert any("trusted" in blocker for blocker in default_result["blockers"])

    source_result = validate_full_benchmark_release_acceptance(
        campaign_root,
        manifest=_full_manifest(),
        campaign_config=config,
        source_repository_root=config.source_repository_root,
    )
    assert source_result["status"] == "valid"
    assert source_result["blockers"] == []


def test_full_release_acceptance_does_not_persist_trusted_source_paths(tmp_path: Path) -> None:
    """A rejected trusted source is reported without leaking its machine-local path."""
    private_source = tmp_path / "private-source-marker" / "missing"

    result = validate_full_benchmark_release_acceptance(
        tmp_path / "campaign",
        manifest=_full_manifest(),
        source_repository_root=private_source,
    )

    assert result["status"] == "invalid"
    assert any("trusted source repository root" in item for item in result["blockers"])
    assert str(private_source) not in json.dumps(result)


def test_source_repository_path_retargets_validator_paths_and_rejects_external_paths(
    tmp_path: Path,
) -> None:
    """Source binding maps validator-root absolutes but rejects unrelated absolute paths."""
    source_root = tmp_path / "frozen-source"
    validator_path = get_repository_root() / "configs/scenarios/example.yaml"
    assert release_acceptance._source_repository_path(validator_path, source_root) == (
        source_root / "configs/scenarios/example.yaml"
    )
    with pytest.raises(ValueError, match="outside trusted repositories"):
        release_acceptance._source_repository_path(tmp_path.parent / "external.yaml", source_root)


def test_full_release_accepts_effective_policy_search_component_algorithm(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A canonical scenario override may replace an arm's base algorithm in episode rows."""
    campaign_root, config = _write_effective_component_campaign(tmp_path, monkeypatch)

    result = validate_full_benchmark_release_acceptance(
        campaign_root,
        manifest=_full_manifest(),
        campaign_config=config,
        source_repository_root=config.source_repository_root,
    )

    assert result["status"] == "valid"
    assert result["blockers"] == []


def test_full_release_rejects_malformed_scenario_override_when_rows_use_base_algorithm(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Malformed overrides cannot be hidden by rows that happen to use the base algorithm."""
    campaign_root, config = _write_effective_component_campaign(
        tmp_path,
        monkeypatch,
        override_mode="malformed_base",
    )

    result = validate_full_benchmark_release_acceptance(
        campaign_root,
        manifest=_full_manifest(),
        campaign_config=config,
        source_repository_root=config.source_repository_root,
    )

    assert result["status"] == "invalid"
    assert any(
        "scenario_algo_overrides entries must be mappings" in item for item in result["blockers"]
    )


def test_full_release_rejects_external_nested_candidate_config(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Nested candidate configs must remain inside the explicit trusted source checkout."""
    campaign_root, config = _write_effective_component_campaign(
        tmp_path,
        monkeypatch,
        override_mode="external",
    )

    result = validate_full_benchmark_release_acceptance(
        campaign_root,
        manifest=_full_manifest(),
        campaign_config=config,
        source_repository_root=config.source_repository_root,
    )

    assert result["status"] == "invalid"
    assert any("outside trusted source repository root" in item for item in result["blockers"])


def test_full_release_rejects_symlinked_nested_candidate_config(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Nested candidate config symlinks are rejected before resolution can hide the link."""
    campaign_root, config = _write_effective_component_campaign(
        tmp_path,
        monkeypatch,
        override_mode="symlink",
    )

    result = validate_full_benchmark_release_acceptance(
        campaign_root,
        manifest=_full_manifest(),
        campaign_config=config,
        source_repository_root=config.source_repository_root,
    )

    assert result["status"] == "invalid"
    assert any("contains a symlink component" in item for item in result["blockers"])


def test_full_release_rejects_unknown_scenario_override_when_rows_use_base_algorithm(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Unused override keys cannot bypass the canonical scenario matrix binding."""
    campaign_root, config = _write_effective_component_campaign(
        tmp_path,
        monkeypatch,
        override_mode="unknown",
    )

    result = validate_full_benchmark_release_acceptance(
        campaign_root,
        manifest=_full_manifest(),
        campaign_config=config,
        source_repository_root=config.source_repository_root,
    )

    assert result["status"] == "invalid"
    assert any("not in the canonical campaign matrix" in item for item in result["blockers"])


@pytest.mark.parametrize("override_mode", ["missing", "malformed"])
def test_full_release_rejects_unbound_effective_policy_search_component(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    override_mode: str,
) -> None:
    """Rows cannot claim ORCA when the canonical candidate has no valid scenario override."""
    campaign_root, config = _write_effective_component_campaign(
        tmp_path,
        monkeypatch,
        override_mode=override_mode,
    )

    result = validate_full_benchmark_release_acceptance(
        campaign_root,
        manifest=_full_manifest(),
        campaign_config=config,
        source_repository_root=config.source_repository_root,
    )

    assert result["status"] == "invalid"
    assert any("planner algorithm aliases do not match" in item for item in result["blockers"])


def test_full_release_rejects_effective_component_on_wrong_scenario_or_arm(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The effective algorithm is bound to both its canonical scenario and containing arm."""
    campaign_root, config = _write_effective_component_campaign(tmp_path, monkeypatch)
    first_path = campaign_root / "runs" / "planner_00__differential_drive" / "episodes.jsonl"
    first_rows = [json.loads(line) for line in first_path.read_text(encoding="utf-8").splitlines()]
    first_rows[len(_SEEDS)]["algo"] = "orca"
    first_rows[len(_SEEDS)]["algorithm_metadata"].update(
        {
            "algorithm": "orca",
            "canonical_algorithm": "orca",
            "planner_contract": {"planner_id": "orca"},
        }
    )
    first_path.write_text(
        "\n".join(json.dumps(row) for row in first_rows) + "\n",
        encoding="utf-8",
    )
    first_sidecar_path = first_path.with_name(f"{first_path.name}.provenance.json")
    first_sidecar = json.loads(first_sidecar_path.read_text(encoding="utf-8"))
    first_sidecar["raw_artifacts"][0]["sha256"] = sha256_file(first_path)
    first_sidecar_path.write_text(json.dumps(first_sidecar), encoding="utf-8")

    second_path = campaign_root / "runs" / "planner_01__differential_drive" / "episodes.jsonl"
    second_rows = [
        json.loads(line) for line in second_path.read_text(encoding="utf-8").splitlines()
    ]
    second_rows[0]["algo"] = "orca"
    second_rows[0]["algorithm_metadata"].update(
        {
            "algorithm": "orca",
            "canonical_algorithm": "orca",
            "planner_contract": {"planner_id": "orca"},
        }
    )
    second_path.write_text(
        "\n".join(json.dumps(row) for row in second_rows) + "\n",
        encoding="utf-8",
    )
    second_sidecar_path = second_path.with_name(f"{second_path.name}.provenance.json")
    second_sidecar = json.loads(second_sidecar_path.read_text(encoding="utf-8"))
    second_sidecar["raw_artifacts"][0]["sha256"] = sha256_file(second_path)
    second_sidecar_path.write_text(json.dumps(second_sidecar), encoding="utf-8")

    result = validate_full_benchmark_release_acceptance(
        campaign_root,
        manifest=_full_manifest(),
        campaign_config=config,
        source_repository_root=config.source_repository_root,
    )

    assert result["status"] == "invalid"
    assert sum("planner algorithm aliases do not match" in item for item in result["blockers"]) >= 2


def test_full_release_rejects_unrecognized_episode_status(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Publication rows must use a documented scientific terminal outcome."""
    campaign_root, config = _write_provenance_bound_full_campaign(tmp_path, monkeypatch)
    episode_path = campaign_root / "runs/planner_00__differential_drive/episodes.jsonl"
    rows = [json.loads(line) for line in episode_path.read_text(encoding="utf-8").splitlines()]
    rows[0]["status"] = "garbage"
    episode_path.write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n",
        encoding="utf-8",
    )
    sidecar_path = episode_path.with_name(f"{episode_path.name}.provenance.json")
    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    sidecar["raw_artifacts"][0]["sha256"] = sha256_file(episode_path)
    sidecar_path.write_text(json.dumps(sidecar), encoding="utf-8")

    result = validate_full_benchmark_release_acceptance(
        campaign_root,
        manifest=_full_manifest(),
        campaign_config=config,
        source_repository_root=config.source_repository_root,
    )

    assert result["status"] == "invalid"
    assert any("recognized scientific terminal outcome" in item for item in result["blockers"])


@pytest.mark.parametrize("field", ["git_hash", "config_hash"])
def test_full_release_rejects_conflicting_present_provenance_alias(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    field: str,
) -> None:
    """A top-level alias cannot disagree with the producer's nested provenance."""
    campaign_root, config = _write_provenance_bound_full_campaign(tmp_path, monkeypatch)
    episode_path = campaign_root / "runs/planner_00__differential_drive/episodes.jsonl"
    rows = [json.loads(line) for line in episode_path.read_text(encoding="utf-8").splitlines()]
    rows[0][field] = "b" * 64 if field == "config_hash" else "b" * 40
    episode_path.write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n",
        encoding="utf-8",
    )
    sidecar_path = episode_path.with_name(f"{episode_path.name}.provenance.json")
    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    sidecar["raw_artifacts"][0]["sha256"] = sha256_file(episode_path)
    sidecar_path.write_text(json.dumps(sidecar), encoding="utf-8")

    result = validate_full_benchmark_release_acceptance(
        campaign_root,
        manifest=_full_manifest(),
        campaign_config=config,
        source_repository_root=config.source_repository_root,
    )

    assert result["status"] == "invalid"
    assert any("aliases conflict" in item for item in result["blockers"])


def test_full_release_sidecar_identity_matches_telemetry_enabled_producer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Telemetry-enabled admission hashes the exact map-runner scenario payload."""
    telemetry = {
        "schema_version": "analysis-telemetry-profile.v1",
        "analysis_trace": "all",
        "planner_debug_trace": "none",
    }
    campaign_root, config = _write_provenance_bound_full_campaign(
        tmp_path,
        monkeypatch,
        telemetry=telemetry,
    )

    result = validate_full_benchmark_release_acceptance(
        campaign_root,
        manifest=_full_manifest(),
        campaign_config=config,
        source_repository_root=config.source_repository_root,
    )

    assert result["status"] == "valid"
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
    campaign_root, config = _write_provenance_bound_full_campaign(
        tmp_path,
        monkeypatch,
        shared_first_algorithm=True,
    )
    manifest = _full_manifest()
    baseline = validate_full_benchmark_release_acceptance(
        campaign_root,
        manifest=manifest,
        campaign_config=config,
        source_repository_root=config.source_repository_root,
    )
    assert baseline["status"] == "valid"

    first = campaign_root / "runs" / "planner_00__differential_drive" / "episodes.jsonl"
    second = campaign_root / "runs" / "planner_01__differential_drive" / "episodes.jsonl"
    first_payload = first.read_bytes()
    second_payload = second.read_bytes()
    first_sidecar = first.with_name(f"{first.name}.provenance.json")
    second_sidecar = second.with_name(f"{second.name}.provenance.json")
    first_sidecar_payload = first_sidecar.read_bytes()
    second_sidecar_payload = second_sidecar.read_bytes()
    first.write_bytes(second_payload)
    second.write_bytes(first_payload)
    first_sidecar.write_bytes(second_sidecar_payload)
    second_sidecar.write_bytes(first_sidecar_payload)

    result = validate_full_benchmark_release_acceptance(
        campaign_root,
        manifest=manifest,
        campaign_config=config,
        source_repository_root=config.source_repository_root,
    )

    assert result["status"] == "invalid"
    assert not any("sidecar raw artifact hash is stale" in item for item in result["blockers"])
    assert any(
        "sidecar raw artifact is not the run artifact" in item
        or "sidecar config path is not bound to its arm" in item
        for item in result["blockers"]
    )


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


@pytest.mark.parametrize("summary_surface", ["run_summary", "planner_row"])
def test_full_release_binds_guarded_exception_on_arm_aggregate_surfaces(
    tmp_path: Path,
    summary_surface: str,
) -> None:
    """A non-Guarded arm cannot forge Guarded PPO aggregate telemetry."""
    campaign_root = _write_full_campaign(tmp_path)
    summary_path = campaign_root / "reports" / "campaign_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    target = (
        summary["runs"][0]["summary"]
        if summary_surface == "run_summary"
        else summary["planner_rows"][0]
    )
    target["algorithm_metadata"] = {
        "algorithm": "ppo",
        "canonical_algorithm": "guarded_ppo",
        "planner_contract": {"planner_id": "guarded_ppo"},
        "guard_stats": {"fallback_safe": 1},
    }
    summary_path.write_text(json.dumps(summary), encoding="utf-8")

    result = validate_full_benchmark_release_acceptance(
        campaign_root,
        manifest=_full_manifest(),
    )

    assert result["status"] == "invalid"
    assert any("fallback_safe" in blocker for blocker in result["blockers"])


def test_full_release_requires_canonical_config_for_v02_admission(tmp_path: Path) -> None:
    """Resolved axes alone cannot downgrade v0.2 to unbound compatibility admission."""
    campaign_root = _write_full_campaign(tmp_path)
    manifest = _full_manifest()
    del manifest.canonical_campaign_config_path

    result = validate_full_benchmark_release_acceptance(campaign_root, manifest=manifest)

    assert result["status"] == "invalid"
    assert "canonical campaign config is required for full-release provenance" in result["blockers"]


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


@pytest.mark.parametrize(
    ("metadata_container", "metadata"),
    tuple(
        (container, metadata)
        for container in ("algorithm_metadata", "algorithm_metadata_contract")
        for metadata in (
            {"fallback_reason": "alternate planner invoked"},
            {"planner_diagnostics": {"fallback_count": 1}},
            {"runtime": {"fallback_count": 1}},
        )
    ),
)
def test_status_markers_reject_runtime_markers_anywhere_in_canonical_metadata(
    metadata_container: str, metadata: dict[str, Any]
) -> None:
    """Both canonical metadata containers fail closed for nested runtime fallbacks."""
    payload = {"status": "ok", metadata_container: metadata}

    markers = _status_markers(payload, "row", expected_algorithm="goal")

    assert markers
    assert any("fallback" in value or "fallback" in path for path, value in markers)


def test_status_markers_ignores_declarative_algorithm_configuration() -> None:
    """Declarative config/contract text is not execution evidence."""
    payload = {
        "status": "ok",
        "algorithm_metadata": {
            "config": {"fallback_reason": "configured alternate policy"},
            "planner_contract": {"fallback_count": 1},
            "safety_shield_contract": {"fallback_used": True},
        },
        "algorithm_metadata_contract": {
            "config": {"runtime": {"fallback_count": 1}},
            "planner_contract": {"fallback_reason": "declared policy"},
            "safety_shield_contract": {"fallback": True},
        },
    }

    assert _status_markers(payload, "row", expected_algorithm="goal") == []


def test_native_protective_reorient_is_not_execution_fallback() -> None:
    """Native static protective reorientation remains admissible telemetry."""
    payload = {
        "status": "ok",
        "algorithm_metadata": {
            "planner_runtime": {
                "fallback_count": 0,
                "protective_stop_count": 2,
                "last_decision": {
                    "planner_mode": "PROTECTIVE_REORIENT",
                    "selected_source": "static_protective_reorient",
                },
            }
        },
    }

    assert _status_markers(payload, "row", expected_algorithm="hybrid_rule_v3") == []


def test_guarded_ppo_safe_exception_requires_an_explicit_expected_arm() -> None:
    """A safe-shield counter is allowed only when the caller binds the guarded arm."""
    payload = {
        "status": "ok",
        "algorithm_metadata": {
            "algorithm": "ppo",
            "canonical_algorithm": "guarded_ppo",
            "planner_contract": {"planner_id": "guarded_ppo"},
            "guard_stats": {"fallback_safe": 1},
        },
    }

    assert _status_markers(payload, "row")
    assert _status_markers(payload, "row", expected_algorithm="goal")
    assert _status_markers(payload, "row", expected_algorithm="guarded_ppo") == []


@pytest.mark.parametrize("value", [0.0, 1.5, "0", True, -1])
def test_guarded_ppo_safe_exception_rejects_non_integer_counters(value: object) -> None:
    """Even numerically zero Guarded PPO counters must retain integer JSON type."""
    payload = {
        "status": "success",
        "algorithm_metadata": {
            "algorithm": "ppo",
            "canonical_algorithm": "guarded_ppo",
            "planner_contract": {"planner_id": "guarded_ppo"},
            "guard_stats": {"fallback_safe": value},
        },
    }

    markers = _status_markers(payload, "row", expected_algorithm="guarded_ppo")

    assert markers
    assert any("fallback_safe" in path for path, _marker in markers)


def test_full_release_rejects_unbound_guarded_safe_aggregate_metadata(tmp_path: Path) -> None:
    """Aggregate metadata cannot smuggle a guarded exception onto another arm."""
    campaign_root = _write_full_campaign(tmp_path)
    summary_path = campaign_root / "reports" / "campaign_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["planner_rows"][0]["algorithm_metadata"] = {
        "algorithm": "ppo",
        "canonical_algorithm": "guarded_ppo",
        "planner_contract": {"planner_id": "guarded_ppo"},
        "guard_stats": {"fallback_safe": 1},
    }
    summary_path.write_text(json.dumps(summary), encoding="utf-8")

    result = validate_full_benchmark_release_acceptance(campaign_root, manifest=_full_manifest())

    assert result["status"] == "invalid"
    assert any("fallback_safe" in blocker for blocker in result["blockers"])


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
