"""Evidence checks for the frozen-release trace comparison."""

from __future__ import annotations

import copy
import hashlib
import io
import json
import sys
import tarfile
from typing import TYPE_CHECKING

import pytest
import yaml

from robot_sf.benchmark.utils import _config_hash
from scripts.validation import check_issue_9671_trace_reexport as checker
from scripts.validation.check_issue_9671_trace_reexport import SOURCE_SHA, check

if TYPE_CHECKING:
    from pathlib import Path


def _row(seed: int, status: str, *, trace: bool) -> dict:
    params = {
        "algo": "ppo",
        "run_horizon": 600,
        "run_dt": 0.1,
        "simulation_config": {
            "route_spawn_seed": seed,
            "goal_completion_policy": "goal_zone_entry_v1",
        },
    }
    if trace:
        params.update(
            record_forces=True,
            record_planner_decision_trace=True,
            record_simulation_step_trace=True,
        )
    config_hash = _config_hash(params)
    return {
        "episode_id": f"classic_doorway_medium--{seed}--{config_hash}",
        "config_hash": config_hash,
        "result_provenance": {"config_hash": config_hash},
        "algo": "ppo",
        "scenario_id": "classic_doorway_medium",
        "seed": seed,
        "status": status,
        "steps": 1,
        "interaction_exposure": {"interaction_exposure_steps": 0},
        "git_hash": SOURCE_SHA,
        "scenario_params": params,
        "algorithm_metadata": {
            "planner_kinematics": {"execution_mode": "native"},
            "simulation_step_trace": {
                "schema_version": "simulation-step-trace.v1",
                "dt": 0.1,
                "steps": [
                    {
                        "step": 0,
                        "time_s": 0.1,
                        "robot": {"position": [0.0, 0.0], "velocity": [0.0, 0.0], "heading": 0.0},
                        "pedestrians": [],
                    }
                ],
            },
        },
    }


def _archive(path: Path, rows: list[dict]) -> None:
    data = b"\n".join(json.dumps(row).encode() for row in rows) + b"\n"
    member = tarfile.TarInfo("bundle/payload/runs/ppo__differential_drive/episodes.jsonl")
    member.size = len(data)
    with tarfile.open(path, "w:gz") as bundle:
        bundle.addfile(member, io.BytesIO(data))


def _bindings(tmp_path: Path, seeds: list[int]) -> tuple[dict, dict, dict, dict]:
    configs = {}
    manifests = {}
    digests = {}
    effective = {"headon_group": "test-headon-hash", "doorway": "test-doorway-hash"}
    for name in effective:
        selected = seeds if name == "headon_group" else []
        config = {
            "horizon": 600,
            "dt": 0.1,
            "kinematics_matrix": ["differential_drive"],
            "record_forces": True,
            "record_planner_decision_trace": True,
            "record_simulation_step_trace": True,
            "scenario_matrix": "configs/scenarios/classic_interactions_francis2023_goal_zone_entry_v1.yaml",
            "scenario_candidates": ["classic_doorway_medium"] if selected else [],
            "seed_policy": {"seeds": selected},
            "planners": [{"key": "ppo"}] if selected else [],
        }
        config_path = tmp_path / f"{name}.yaml"
        config_path.write_text(yaml.safe_dump(config))
        configs[name] = config_path
        digests[name] = hashlib.sha256(config_path.read_bytes()).hexdigest()
        manifest_path = tmp_path / f"{name}-manifest.json"
        manifest_path.write_text(
            json.dumps(
                {
                    "git": {"commit": SOURCE_SHA},
                    "config_hash": effective[name],
                    "scenario_matrix": config["scenario_matrix"],
                    "campaign_id": checker.CAMPAIGN_ID[name],
                    "scenario_matrix_hash": "test-matrix-hash",
                    "invoked_command": (
                        "python scripts/tools/run_camera_ready_benchmark.py "
                        f"--config {checker.FROZEN_PUBLIC_ROOT / checker.DIAGNOSTIC_CONFIG[name]} "
                        f"--mode run --campaign-id {checker.CAMPAIGN_ID[name]}"
                    ),
                    "seed_policy": {"resolved_seeds": selected},
                }
            )
        )
        manifests[name] = manifest_path
    return configs, manifests, digests, effective


def _producer_trace(tmp_path: Path, rows: list[dict]) -> Path:
    path = tmp_path / "runs" / "ppo__differential_drive" / "episodes.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    for row in rows:
        config_hash = _config_hash(row["scenario_params"])
        row["config_hash"] = config_hash
        row["result_provenance"]["config_hash"] = config_hash
        row["episode_id"] = f"classic_doorway_medium--{row['seed']}--{config_hash}"
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    sidecar = {
        "run": {
            "repo_commit": SOURCE_SHA,
            "invocation": (
                "scripts/tools/run_camera_ready_benchmark.py "
                f"--config {checker.FROZEN_PUBLIC_ROOT / checker.DIAGNOSTIC_CONFIG['headon_group']} "
                f"--mode run --campaign-id {checker.CAMPAIGN_ID['headon_group']}"
            ),
        },
        "inputs": {
            "schema_path": {
                "path": "robot_sf/benchmark/schemas/episode.schema.v1.json",
                "sha256": checker.FROZEN_INPUT_SHA256[
                    "robot_sf/benchmark/schemas/episode.schema.v1.json"
                ],
                "artifact_status": "available",
            },
            "scenario_matrix": {
                "path": "configs/scenarios/classic_interactions_francis2023_goal_zone_entry_v1.yaml",
                "sha256": checker.FROZEN_INPUT_SHA256[
                    "configs/scenarios/classic_interactions_francis2023_goal_zone_entry_v1.yaml"
                ],
                "artifact_status": "available",
            },
            "algo_config": {"path": None, "sha256": None, "artifact_status": "not_provided"},
        },
        "campaign_identity": {
            "scenario_matrix_hash": "test-arm-hash",
            "algorithm": "ppo",
        },
        "raw_artifacts": [
            {
                "kind": "episodes_jsonl",
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        ],
        "rows": [
            {
                "jsonl_line": index,
                "episode_id": row["episode_id"],
                "scenario_id": row["scenario_id"],
                "seed": row["seed"],
                "repo_commit": SOURCE_SHA,
                "config_hash": row["config_hash"],
            }
            for index, row in enumerate(rows)
        ],
    }
    path.with_name(path.name + ".provenance.json").write_text(json.dumps(sidecar))
    return path


def _check(archive: Path, traces: Path, tmp_path: Path, seeds: list[int]) -> dict:
    configs, manifests, digests, effective = _bindings(tmp_path, seeds)
    traces = _producer_trace(
        tmp_path, [json.loads(line) for line in traces.read_text().splitlines()]
    )
    return check(
        archive,
        [traces],
        configs,
        manifests,
        expected={("ppo", "classic_doorway_medium", seed) for seed in seeds},
        expected_archive_sha256=None,
        expected_config_sha256=digests,
        expected_effective_hash=effective,
    )


def test_reports_mismatch_and_absent_release_row(tmp_path: Path) -> None:
    archive = tmp_path / "release.tar.gz"
    _archive(archive, [_row(113, "collision", trace=False)])
    traces = tmp_path / "episodes.jsonl"
    traces.write_text(
        "\n".join(
            json.dumps(row)
            for row in [_row(113, "success", trace=True), _row(22, "collision", trace=True)]
        )
        + "\n"
    )
    report = _check(archive, traces, tmp_path, [113, 22])
    assert [row["comparison"] for row in report["comparisons"]] == [
        "no_release_row",
        "mismatch",
    ]
    assert report["comparisons"][1]["release_interaction_exposure_steps"] == 0
    assert report["comparisons"][1]["trace_interaction_exposure_steps"] == 0


def test_rejects_parameter_drift(tmp_path: Path) -> None:
    archive = tmp_path / "release.tar.gz"
    _archive(archive, [_row(113, "collision", trace=False)])
    trace = _row(113, "collision", trace=True)
    trace["scenario_params"]["run_dt"] = 0.2
    traces = tmp_path / "episodes.jsonl"
    traces.write_text(json.dumps(trace) + "\n")
    with pytest.raises(ValueError, match="changes release scenario/planner parameters"):
        _check(archive, traces, tmp_path, [113])


def test_rejects_missing_per_pedestrian_force(tmp_path: Path) -> None:
    archive = tmp_path / "release.tar.gz"
    _archive(archive, [_row(113, "collision", trace=False)])
    trace = _row(22, "collision", trace=True)
    trace["algorithm_metadata"]["simulation_step_trace"]["steps"][0]["pedestrians"] = [
        {"position": [1.0, 1.0], "velocity": [0.0, 0.0]}
    ]
    trace["algorithm_metadata"]["simulation_step_trace"]["steps"][0]["planner"] = {
        "ammv": {"pedestrian_force_vectors": [None]}
    }
    traces = tmp_path / "episodes.jsonl"
    traces.write_text(json.dumps(trace) + "\n")
    with pytest.raises(ValueError, match="lacks pedestrian forces"):
        _check(archive, traces, tmp_path, [22])


@pytest.mark.parametrize(
    ("metadata_key", "metadata_value"),
    [
        ("execution_mode", "fallback"),
        ("planner_runtime", {"fallback_used": True}),
    ],
)
def test_rejects_fallback_runtime(
    tmp_path: Path, metadata_key: str, metadata_value: object
) -> None:
    archive = tmp_path / "release.tar.gz"
    _archive(archive, [_row(113, "collision", trace=False)])
    trace = _row(113, "collision", trace=True)
    trace["algorithm_metadata"][metadata_key] = metadata_value
    traces = tmp_path / "episodes.jsonl"
    traces.write_text(json.dumps(trace) + "\n")
    with pytest.raises(ValueError, match="inadmissible runtime mode"):
        _check(archive, traces, tmp_path, [113])


@pytest.mark.parametrize("defect", ["short", "noncontiguous"])
def test_rejects_incomplete_step_trace(tmp_path: Path, defect: str) -> None:
    archive = tmp_path / "release.tar.gz"
    _archive(archive, [_row(113, "collision", trace=False)])
    trace = _row(113, "collision", trace=True)
    if defect == "short":
        trace["steps"] = 600
        trace["truncated"] = True
        match = "step count does not match episode"
    else:
        trace["steps"] = 2
        second = copy.deepcopy(trace["algorithm_metadata"]["simulation_step_trace"]["steps"][0])
        second["step"] = 2
        second["time_s"] = 0.3
        trace["algorithm_metadata"]["simulation_step_trace"]["steps"].append(second)
        match = "noncontiguous index/time"
    traces = tmp_path / "episodes.jsonl"
    traces.write_text(json.dumps(trace) + "\n")
    with pytest.raises(ValueError, match=match):
        _check(archive, traces, tmp_path, [113])


def test_rejects_unpaired_scientific_parameter_drift(tmp_path: Path) -> None:
    archive = tmp_path / "release.tar.gz"
    _archive(archive, [_row(113, "collision", trace=False)])
    trace = _row(22, "collision", trace=True)
    trace["scenario_params"]["run_dt"] = 0.2
    traces = tmp_path / "episodes.jsonl"
    traces.write_text(json.dumps(trace) + "\n")
    with pytest.raises(ValueError, match="changes release scenario/planner parameters"):
        _check(archive, traces, tmp_path, [22])


def test_rejects_unbound_config_bytes(tmp_path: Path) -> None:
    archive = tmp_path / "release.tar.gz"
    _archive(archive, [_row(113, "collision", trace=False)])
    traces = tmp_path / "episodes.jsonl"
    traces.write_text(json.dumps(_row(113, "collision", trace=True)) + "\n")
    configs, manifests, digests, effective = _bindings(tmp_path, [113])
    traces = _producer_trace(tmp_path, [json.loads(traces.read_text())])
    configs["doorway"].write_text(configs["doorway"].read_text() + "# drift\n")
    with pytest.raises(ValueError, match="diagnostic config SHA-256 mismatch"):
        check(
            archive,
            [traces],
            configs,
            manifests,
            expected={("ppo", "classic_doorway_medium", 113)},
            expected_archive_sha256=None,
            expected_config_sha256=digests,
            expected_effective_hash=effective,
        )


def test_rejects_unstaged_config_invocation(tmp_path: Path) -> None:
    archive = tmp_path / "release.tar.gz"
    _archive(archive, [_row(113, "collision", trace=False)])
    configs, manifests, digests, effective = _bindings(tmp_path, [113])
    traces = _producer_trace(tmp_path, [_row(113, "collision", trace=True)])
    manifest = json.loads(manifests["headon_group"].read_text())
    manifest["invoked_command"] = manifest["invoked_command"].replace(
        str(checker.FROZEN_PUBLIC_ROOT / checker.DIAGNOSTIC_CONFIG["headon_group"]),
        "configs/benchmarks/issue_9671_trace_headon_group_v007.yaml",
    )
    manifests["headon_group"].write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="campaign manifest identity mismatch"):
        check(
            archive,
            [traces],
            configs,
            manifests,
            expected={("ppo", "classic_doorway_medium", 113)},
            expected_archive_sha256=None,
            expected_config_sha256=digests,
            expected_effective_hash=effective,
        )


def test_rejects_mixed_trace_bytes_without_matching_producer_manifest(tmp_path: Path) -> None:
    archive = tmp_path / "release.tar.gz"
    _archive(archive, [_row(113, "collision", trace=False)])
    configs, manifests, digests, effective = _bindings(tmp_path, [113, 22])
    traces = _producer_trace(tmp_path, [_row(113, "collision", trace=True)])
    with traces.open("a") as stream:
        stream.write(json.dumps(_row(22, "collision", trace=True)) + "\n")
    with pytest.raises(ValueError, match="does not bind JSONL bytes"):
        check(
            archive,
            [traces],
            configs,
            manifests,
            expected={("ppo", "classic_doorway_medium", seed) for seed in (113, 22)},
            expected_archive_sha256=None,
            expected_config_sha256=digests,
            expected_effective_hash=effective,
        )


def test_rejects_coherent_file_and_sidecar_from_another_campaign(tmp_path: Path) -> None:
    archive = tmp_path / "release.tar.gz"
    _archive(archive, [_row(113, "collision", trace=False)])
    configs, manifests, digests, effective = _bindings(tmp_path, [113])
    traces = _producer_trace(tmp_path, [_row(113, "collision", trace=True)])
    sidecar_path = traces.with_name(traces.name + ".provenance.json")
    sidecar = json.loads(sidecar_path.read_text())
    sidecar["run"]["invocation"] = (
        "scripts/tools/run_camera_ready_benchmark.py "
        "--config configs/benchmarks/other_campaign.yaml --mode run "
        "--campaign-id other_campaign"
    )
    sidecar_path.write_text(json.dumps(sidecar))
    with pytest.raises(ValueError, match="producer invocation does not match diagnostic campaign"):
        check(
            archive,
            [traces],
            configs,
            manifests,
            expected={("ppo", "classic_doorway_medium", 113)},
            expected_archive_sha256=None,
            expected_config_sha256=digests,
            expected_effective_hash=effective,
        )


def test_cli_keeps_report_and_exits_nonzero_on_outcome_mismatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    report = {"comparison_counts": {"mismatch": 1}, "comparisons": [{"comparison": "mismatch"}]}
    monkeypatch.setattr(checker, "check", lambda *_args, **_kwargs: report)
    output = tmp_path / "comparison.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "check_issue_9671_trace_reexport.py",
            "--release-archive",
            "unused.tar.gz",
            "--traces",
            "unused.jsonl",
            "--headon-config",
            "unused-headon.yaml",
            "--doorway-config",
            "unused-doorway.yaml",
            "--headon-manifest",
            "unused-headon.json",
            "--doorway-manifest",
            "unused-doorway.json",
            "--output",
            str(output),
        ],
    )
    assert checker.main() == 2
    assert json.loads(output.read_text()) == report
