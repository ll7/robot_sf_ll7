"""Fixture coverage for historical benchmark hard-case materialization."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import yaml

import scripts.tools.materialize_benchmark_hard_cases as materializer
from scripts.tools.materialize_benchmark_hard_cases import (
    REPLAY_CHECKOUT_SNAPSHOT_SCHEMA,
    MaterializationError,
    _compare,
    _materialize_replay_matrix,
    _replay_checkout_provenance,
    _replay_checkout_stability_status,
    _replay_identity,
    _replay_ineligibility,
    _run_replay,
    _sanitize,
    _selected_cases,
    _showcase_tool_snapshot,
    materialize,
)
from scripts.tools.materialize_benchmark_hard_cases import (
    _classify_replay_row as _classify_replay_row_impl,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
SOURCE_REVISION = subprocess.run(
    ["git", "rev-parse", "HEAD^"], cwd=REPO_ROOT, check=True, capture_output=True, text=True
).stdout.strip()
MATRIX_RELATIVE = "configs/scenarios/classic_interactions_francis2023.yaml"
_SOURCE_ENVIRONMENT = object()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fixture_execution_environment() -> dict[str, Any]:
    identity = {
        "python_version": "3.12.0-fixture",
        "python_implementation": "CPython-fixture",
        "platform_system": "Linux-fixture",
        "platform_release": "fixture",
        "machine": "x86_64-fixture",
        "uv_lock_sha256": "0" * 64,
    }
    return {
        "schema_version": materializer.SOURCE_ENVIRONMENT_PROVENANCE_SCHEMA,
        "complete": True,
        **identity,
        "identity_sha256": hashlib.sha256(
            json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
    }


def _different_fixture_execution_environment() -> dict[str, Any]:
    environment = _fixture_execution_environment()
    environment["platform_release"] = "different-runtime-fixture"
    identity_fields = {
        key: environment[key]
        for key in (
            "python_version",
            "python_implementation",
            "platform_system",
            "platform_release",
            "machine",
            "uv_lock_sha256",
        )
    }
    environment["identity_sha256"] = hashlib.sha256(
        json.dumps(identity_fields, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return environment


def _classify_replay_row(
    case: dict[str, Any],
    source_row: dict[str, Any],
    replay_row: dict[str, Any],
    replay_revision: Any,
    *,
    replay_environment_identity: Any = _SOURCE_ENVIRONMENT,
    **kwargs: Any,
) -> dict[str, Any]:
    """Use a same-as-source replay environment unless a test overrides it."""
    if replay_environment_identity is _SOURCE_ENVIRONMENT:
        provenance = source_row.get("runtime_input_provenance")
        replay_environment_identity = (
            provenance.get("source_environment") if isinstance(provenance, dict) else None
        )
    if "replay_checkout" not in kwargs:
        kwargs["replay_checkout"] = {
            "replay_checkout_clean": kwargs.pop("replay_checkout_clean", None),
            "replay_checkout_stability_status": kwargs.pop(
                "replay_checkout_stability_status", None
            ),
        }
    return _classify_replay_row_impl(
        case,
        source_row,
        replay_row,
        replay_revision,
        replay_environment_identity=replay_environment_identity,
        **kwargs,
    )


def _record_fixture_runtime_inputs(row: dict[str, Any]) -> None:
    """Attach synthetic historical hashes to classifier fixtures only."""
    params = row.get("scenario_params")
    metadata = row.get("algorithm_metadata")
    config = metadata.get("config") if isinstance(metadata, dict) else None
    if not isinstance(params, dict) or not isinstance(config, dict):
        return
    assets: list[dict[str, str]] = []
    refs: list[tuple[str, str, Path]] = []
    map_value = params.get("map_file")
    if isinstance(map_value, str) and map_value:
        map_path = Path(map_value)
        if not map_path.is_absolute():
            map_path = REPO_ROOT / map_path
        refs.append(("scenario_map", map_value, map_path))
    for key, value in materializer._runtime_model_paths(config):
        paths = materializer._runtime_model_path_components(key, value)
        if paths is not None:
            refs.extend((key, value, path) for path in paths)
    for kind, reference, path in refs:
        try:
            name = path.name
            repository_path = path.resolve().relative_to(REPO_ROOT.resolve()).as_posix()
        except (OSError, ValueError):
            repository_path = ""
            name = path.name
        if repository_path:
            source_bytes = subprocess.run(
                ["git", "show", f"{SOURCE_REVISION}:{repository_path}"],
                cwd=REPO_ROOT,
                check=True,
                capture_output=True,
            ).stdout
        else:
            source_bytes = path.read_bytes()
        assets.append(
            {
                "kind": kind,
                "name": name,
                "reference": reference,
                "sha256": hashlib.sha256(source_bytes).hexdigest(),
            }
        )
    if row.get("algo") == "crowdnav_height":
        assets.extend(_crowdnav_height_fixture_assets(config))
    environment = _fixture_execution_environment()
    row["runtime_input_provenance"] = {
        "schema_version": materializer.SOURCE_RUNTIME_INPUT_PROVENANCE_SCHEMA,
        "source_revision": row.get("git_hash"),
        "source_checkout_clean": True,
        "complete": True,
        "source_environment": {
            **environment,
        },
        "assets": assets,
    }


def _crowdnav_height_fixture_assets(config: dict[str, Any]) -> list[dict[str, str]]:
    assets = []
    for descriptor in materializer._crowdnav_height_input_specs(config):
        identity = materializer._crowdnav_height_runtime_asset_identity(descriptor)
        if identity.get("status") == "verified":
            assets.append(
                {
                    "kind": descriptor["kind"],
                    "name": identity["name"],
                    "reference": descriptor["reference"],
                    "sha256": identity["sha256"],
                }
            )
    return assets


def _create_crowdnav_height_assets(
    root: Path, checkpoint_name: str = "237800.pt", *, include_checkpoint: bool = True
) -> tuple[Path, Path]:
    repo_root = root / "CrowdNav_HEIGHT"
    repo_root.mkdir(parents=True)
    (repo_root / "training" / "networks").mkdir(parents=True)
    (repo_root / "training" / "networks" / "model.py").write_text(
        "# fixture upstream source\n", encoding="utf-8"
    )
    subprocess.run(["git", "init", str(repo_root)], check=True, capture_output=True)
    subprocess.run(
        ["git", "-C", str(repo_root), "config", "user.email", "fixture@example.invalid"],
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ["git", "-C", str(repo_root), "config", "user.name", "Fixture"],
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ["git", "-C", str(repo_root), "add", "training/networks/model.py"],
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ["git", "-C", str(repo_root), "commit", "-m", "fixture upstream source"],
        check=True,
        capture_output=True,
    )
    model_dir = root / "HEIGHT" / "HEIGHT"
    (model_dir / "configs").mkdir(parents=True)
    (model_dir / "checkpoints").mkdir(parents=True)
    (model_dir / "configs" / "config.py").write_text("class Config: pass\n", encoding="utf-8")
    if include_checkpoint:
        (model_dir / "checkpoints" / checkpoint_name).write_bytes(b"fixture HEIGHT checkpoint")
    return repo_root, model_dir


def _crowdnav_height_source_row(
    repo_root: Path,
    model_dir: Path,
    *,
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    row = _source_row(scenario_id="scenario_a", episode_id="episode-a")
    row["algo"] = "crowdnav_height"
    row["algorithm_metadata"]["algorithm"] = "crowdnav_height"
    row["algorithm_metadata"]["config"] = (
        config
        if config is not None
        else {
            "repo_root": str(repo_root),
            "model_dir": str(model_dir),
            "checkpoint_name": "237800.pt",
        }
    )
    _record_fixture_runtime_inputs(row)
    return row


def _clean_checkout_snapshot(revision: str) -> dict[str, Any]:
    return {
        "schema": REPLAY_CHECKOUT_SNAPSHOT_SCHEMA,
        "revision": revision,
        "clean": True,
        "status_sha256": hashlib.sha256(b"").hexdigest(),
        "status_entries": [],
    }


def _source_row(
    *, scenario_id: str, episode_id: str, include_params: bool = True
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "algo": "goal",
        "algorithm_metadata": {
            "algorithm": "goal",
            "config": {},
            "config_hash": "44136fa355b3678a",
            "planner_kinematics": {"execution_mode": "native"},
            "status": "ok",
        },
        "episode_id": episode_id,
        "git_hash": SOURCE_REVISION,
        "horizon": 10,
        "metrics": {
            "collisions": 0.0,
            "total_collision_count": 0.0,
            "near_misses": 1.0,
            "force_exceed_events": 0.0,
            "comfort_exposure": 0.1,
            "clearing_distance_min": 0.5,
            "time_to_goal_norm": 1.0,
            "path_efficiency": 0.7,
        },
        "outcome": {
            "collision_event": True,
            "route_complete": False,
            "timeout_event": False,
        },
        "scenario_id": scenario_id,
        "seed": 111,
        "status": "collision",
    }
    if include_params:
        row["scenario_params"] = {
            "id": scenario_id,
            "map_file": "maps/svg_maps/classic_bottleneck_high.svg",
            "record_forces": True,
            "robot_config": {"type": "differential_drive"},
            "run_dt": 0.1,
            "run_horizon": 10,
        }
        _record_fixture_runtime_inputs(row)
    return row


def _build_inputs(root: Path, *, include_unavailable: bool = True) -> tuple[Path, Path, Path]:
    campaign_root = root / "payload"
    (campaign_root / "release").mkdir(parents=True)
    (campaign_root / "runs").mkdir(parents=True)
    matrix = REPO_ROOT / MATRIX_RELATIVE
    release_manifest = {
        "scenario": {"matrix_path": MATRIX_RELATIVE, "matrix_sha256": _sha256(matrix)}
    }
    release_path = campaign_root / "release" / "release_manifest.resolved.json"
    release_path.write_text(json.dumps(release_manifest), encoding="utf-8")
    campaign_manifest = {
        "campaign_id": "fixture-campaign",
        "git": {"commit": SOURCE_REVISION},
        "planners": [{"key": "goal", "benchmark_profile": "baseline-safe"}],
    }
    campaign_path = campaign_root / "campaign_manifest.json"
    campaign_path.write_text(json.dumps(campaign_manifest), encoding="utf-8")

    rows = [_source_row(scenario_id="scenario_a", episode_id="episode-a")]
    cases = []
    for index, row in enumerate(rows):
        cases.append(_case_for_row(row, index, "collision_event"))
    if include_unavailable:
        missing = _source_row(
            scenario_id="scenario_b", episode_id="episode-b", include_params=False
        )
        rows.append(missing)
        cases.append(_case_for_row(missing, 1, "minimum_clearance"))
    episode_path = campaign_root / "runs" / "goal__differential_drive" / "episodes.jsonl"
    episode_path.parent.mkdir(parents=True)
    raw_lines = [json.dumps(row, sort_keys=True).encode() + b"\n" for row in rows]
    episode_path.write_bytes(b"".join(raw_lines))
    relative_episode = "runs/goal__differential_drive/episodes.jsonl"
    for index, case in enumerate(cases):
        case["source"] = {
            "episode_file": relative_episode,
            "episode_file_sha256": _sha256(episode_path),
            "line_number": index + 1,
            "record_sha256": hashlib.sha256(raw_lines[index]).hexdigest(),
        }
    checksums = {
        "campaign_manifest.json": _sha256(campaign_path),
        "release/release_manifest.resolved.json": _sha256(release_path),
        relative_episode: _sha256(episode_path),
    }
    summary = {
        "schema_version": "benchmark-showcase.v1",
        "selection": {"ordering": "named group then round robin", "top_k_per_group": 1},
        "source": {
            "bundle_sha256": None,
            "campaign_id": "fixture-campaign",
            "campaign_source_revision": SOURCE_REVISION,
            "embedded_checksums": {"status": "passed"},
            "scenario_matrix": MATRIX_RELATIVE,
            "source_files_sha256": checksums,
        },
        "cases": cases,
        "tool_provenance": {
            "git_revision": "showcase-revision",
            "files": {
                "scripts/replay_episode_figure.py": _sha256(
                    REPO_ROOT / "scripts/replay_episode_figure.py"
                )
            },
        },
    }
    summary_path = root / "showcase_summary.json"
    summary_path.write_text(json.dumps(summary, sort_keys=True), encoding="utf-8")
    return summary_path, campaign_root, matrix


def _case_for_row(row: dict[str, Any], index: int, group: str) -> dict[str, Any]:
    source_metrics = row["metrics"]
    showcase_metrics = {
        "minimum_clearance_m": source_metrics.get("clearing_distance_min"),
        "near_misses": source_metrics.get("near_misses"),
        "force_exceed_events": source_metrics.get("force_exceed_events"),
        "comfort_exposure": source_metrics.get("comfort_exposure"),
        "time_to_goal_norm": source_metrics.get("time_to_goal_norm"),
        "path_efficiency": source_metrics.get("path_efficiency"),
        "snqi": source_metrics.get("snqi"),
        "success_metric": (
            float(source_metrics["success"])
            if isinstance(source_metrics.get("success"), bool)
            else source_metrics.get("success")
        ),
        "collisions_metric": source_metrics.get("collisions"),
        "total_collision_count": source_metrics.get("total_collision_count"),
    }
    return {
        "benchmark_eligible": True,
        "case_id": f"case-{index + 1:016x}",
        "episode_id": row["episode_id"],
        "metrics": showcase_metrics,
        "outcome": {
            "route_complete": row["outcome"].get("route_complete", row["outcome"].get("success")),
            "collision_event": row["outcome"].get(
                "collision_event", row["outcome"].get("collision")
            ),
            "timeout_event": row["outcome"].get("timeout_event", row["outcome"].get("timeout")),
        },
        "planner_key": row["algo"],
        "replay": {"status": "unavailable", "renderer_called": False},
        "scenario_family": "fixture",
        "scenario_id": row["scenario_id"],
        "seed": row["seed"],
        "selected_groups": [group],
    }


def _args(summary: Path, campaign_root: Path, matrix: Path, out_dir: Path) -> argparse.Namespace:
    return argparse.Namespace(
        summary=summary,
        campaign_root=campaign_root,
        matrix=matrix,
        bundle=None,
        out_dir=out_dir,
        replay_limit=0,
        resume_from=None,
    )


def test_materializes_rows_deterministically_and_keeps_unavailable_rows_visible(
    tmp_path: Path,
) -> None:
    summary, campaign_root, matrix = _build_inputs(tmp_path)
    first = materialize(_args(summary, campaign_root, matrix, tmp_path / "first"))
    second = materialize(_args(summary, campaign_root, matrix, tmp_path / "second"))

    assert first == second
    assert first["selection"]["case_count"] == 2
    assert first["selection"]["selected_group_counts"] == {
        "collision_event": 1,
        "minimum_clearance": 1,
    }
    assert first["selection"]["planner_counts"] == {"goal": 2}
    assert first["selection"]["scenario_family_counts"] == {"fixture": 2}
    assert first["selection"]["scenario_id_count"] == 2
    assert first["source"]["source_campaign_id"] == "fixture-campaign"
    assert "campaign_id" not in first["source"]
    assert first["source"]["showcase_tool_revision"] == "showcase-revision"
    assert first["source"]["showcase_tool_source_snapshot_status"] == ("verified_file_hash_match")
    assert (
        first["source"]["showcase_tool_source_snapshot_revision"]
        == first["replay"]["materializer_revision"]
    )
    assert first["source"]["showcase_tool_source_files_sha256"] == {
        "scripts/replay_episode_figure.py": _sha256(REPO_ROOT / "scripts/replay_episode_figure.py")
    }
    assert first["criticality_anomaly_counts"] == {
        "collision_event_without_positive_collision_metric": 2
    }
    case_a = json.loads((tmp_path / "first/cases/case-0000000000000001/case.json").read_text())
    assert case_a["source"]["record_sha256"]
    assert case_a["source"]["campaign_source_revision"] == SOURCE_REVISION
    assert case_a["source_showcase_renderer"]["status"] == "unavailable"
    assert case_a["criticality"]["anomalies"] == [
        "collision_event_without_positive_collision_metric"
    ]
    assert case_a["replay_input"]["status"] == "materialized"
    replay_input_path = tmp_path / "first/cases/case-0000000000000001/replay_input"
    replay_input_matrix = yaml.safe_load((replay_input_path / "replay_matrix.yaml").read_text())
    assert replay_input_matrix["scenarios"][0]["seeds"] == [111]
    assert case_a["replay_input"]["scenario_matrix_sha256"] == _sha256(
        replay_input_path / "replay_matrix.yaml"
    )
    case_b = json.loads((tmp_path / "first/cases/case-0000000000000002/case.json").read_text())
    assert case_b["replay"]["status"] == "unavailable_scenario_parameters"
    assert case_b["replay_input"]["status"] == "unavailable"
    assert case_b["source_record"]["scenario_id"] == "scenario_b"
    assert _sha256(tmp_path / "first/manifest.json") == _sha256(tmp_path / "second/manifest.json")
    report = (tmp_path / "first/report.md").read_text(encoding="utf-8")
    assert "- Source campaign: `fixture-campaign`" in report
    assert "Showcase source snapshot: `verified_file_hash_match` at" in report
    assert "- Campaign:" not in report
    first_row = json.loads(
        (campaign_root / "runs/goal__differential_drive/episodes.jsonl").read_text().splitlines()[0]
    )
    assert _replay_ineligibility(first_row, matrix, True) is None
    replay_dir = tmp_path / "replay"
    replay_dir.mkdir()
    replay_matrix = _materialize_replay_matrix(first_row, matrix, replay_dir)
    replay_payload = yaml.safe_load(replay_matrix.read_text())
    assert replay_payload["scenarios"][0]["seeds"] == [111]
    assert (replay_matrix.parent / replay_payload["scenarios"][0]["map_file"]).resolve() == (
        REPO_ROOT / "maps/svg_maps/classic_bottleneck_high.svg"
    ).resolve()


def test_showcase_execution_revision_is_preserved_when_snapshot_does_not_match() -> None:
    """A non-matching current tree is never substituted for the executed revision."""
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        capture_output=True,
        check=True,
        text=True,
    ).stdout.strip()
    record = _showcase_tool_snapshot(
        {
            "tool_provenance": {
                "git_revision": "0" * 40,
                "files": {"scripts/replay_episode_figure.py": "0" * 64},
            }
        },
        revision,
    )

    assert record["showcase_tool_revision"] == "0" * 40
    assert record["showcase_tool_source_snapshot_revision"] is None
    assert record["showcase_tool_source_snapshot_status"] == "source_files_not_matched"


def test_showcase_snapshot_requires_a_file_hash_inventory() -> None:
    """A historical revision without file hashes stays explicitly unverified."""
    record = _showcase_tool_snapshot(
        {"tool_provenance": {"git_revision": "historical-revision"}}, "HEAD"
    )

    assert record["showcase_tool_revision"] == "historical-revision"
    assert record["showcase_tool_source_snapshot_revision"] is None
    assert record["showcase_tool_source_snapshot_status"] == "unavailable_file_inventory"


def test_rejects_duplicate_materialized_case_identity() -> None:
    cases = [{"case_id": "case-0000000000000001"}, {"case_id": "case-0000000000000001"}]
    with pytest.raises(MaterializationError, match="duplicate"):
        _selected_cases({"cases": cases})


def test_comparison_preserves_mismatch_and_unknown_metrics() -> None:
    expected = _source_row(scenario_id="scenario_a", episode_id="episode-a")
    assert _compare(expected, expected)["overall"] == "match"
    changed = json.loads(json.dumps(expected))
    changed["outcome"]["collision_event"] = False
    changed["metrics"]["near_misses"] = 0.0
    mismatch = _compare(expected, changed)
    assert mismatch["overall"] == "mismatch"
    assert mismatch["outcomes"]["collision_event"]["status"] == "mismatch"
    assert mismatch["metrics"]["near_misses"]["status"] == "mismatch"
    del changed["metrics"]["path_efficiency"]
    assert _compare(expected, changed)["metrics"]["path_efficiency"]["status"] == "unavailable"


def test_replay_identity_requires_selected_case_planner_scenario_and_seed() -> None:
    case = {"planner_key": "goal", "scenario_id": "scenario_a", "seed": 111}
    observed = {"algo": "goal", "scenario_id": "scenario_a", "seed": 111}
    assert _replay_identity(case, observed)["status"] == "match"
    observed["seed"] = 112
    assert _replay_identity(case, observed)["status"] == "mismatch"


@pytest.mark.parametrize(
    ("mutation", "expected_status"),
    [
        ("source_fallback", "fallback_or_degraded_not_evidence"),
        ("replay_fallback", "fallback_or_degraded_not_evidence"),
        ("source_failed", "unavailable_execution_evidence"),
        ("replay_failed", "unavailable_execution_evidence"),
        ("nested_failed", "unavailable_execution_evidence"),
        ("source_unavailable", "unavailable_execution_evidence"),
        ("replay_unavailable", "unavailable_execution_evidence"),
        ("source_unavailable_flag", "unavailable_execution_evidence"),
        ("ineligible", "unavailable_source_row_not_benchmark_eligible"),
    ],
)
def test_exact_replay_requires_successful_available_nondegraded_evidence(
    mutation: str, expected_status: str
) -> None:
    source = _source_row(scenario_id="scenario_a", episode_id="episode-a")
    observed = json.loads(json.dumps(source))
    case = {
        "planner_key": "goal",
        "scenario_id": "scenario_a",
        "seed": 111,
        "benchmark_eligible": True,
    }
    if mutation.endswith("fallback"):
        target = source if mutation.startswith("source") else observed
        target["algorithm_metadata"]["planner_runtime"] = {"fallback_used": True}
    elif mutation == "nested_failed":
        observed["algorithm_metadata"]["planner_runtime"] = {"status": "failed"}
    elif mutation.endswith("failed"):
        target = source if mutation.startswith("source") else observed
        target["algorithm_metadata"]["status"] = "failed"
    elif mutation.endswith("unavailable"):
        target = source if mutation.startswith("source") else observed
        target["availability_status"] = "unavailable"
    elif mutation == "source_unavailable_flag":
        source["availability"] = False
    else:
        case["benchmark_eligible"] = False

    classification = _classify_replay_row(
        case, source, observed, SOURCE_REVISION, replay_checkout_clean=True
    )
    assert classification["status"] == expected_status


@pytest.mark.parametrize("side", ["source", "replay"])
@pytest.mark.parametrize(
    ("marker", "expected_status"),
    [
        (False, "exact_match"),
        (True, "unavailable_execution_evidence"),
        ("true", "unavailable_execution_evidence"),
        (None, "unavailable_execution_evidence"),
        (0, "unavailable_execution_evidence"),
        ({}, "unavailable_execution_evidence"),
    ],
)
def test_explicit_unavailable_marker_must_be_a_boolean_false_to_pass(
    side: str, marker: Any, expected_status: str
) -> None:
    source = _source_row(scenario_id="scenario_a", episode_id="episode-a")
    observed = json.loads(json.dumps(source))
    target = source if side == "source" else observed
    target["unavailable"] = marker
    case = {
        "planner_key": "goal",
        "scenario_id": "scenario_a",
        "seed": 111,
        "benchmark_eligible": True,
    }

    classification = _classify_replay_row(
        case,
        source,
        observed,
        SOURCE_REVISION,
        replay_checkout_clean=True,
        replay_checkout_stability_status="clean_stable",
    )

    assert classification["status"] == expected_status
    assert classification[f"execution_evidence_{side}"] == (
        "available" if marker is False else "unavailable_execution_availability"
    )


@pytest.mark.parametrize(
    ("side", "diagnostics", "expected_status"),
    [
        ("source", {"fallback": True}, "fallback_or_degraded_not_evidence"),
        ("replay", {"fallback": True}, "fallback_or_degraded_not_evidence"),
        ("source", {"fallback_count": 1}, "fallback_or_degraded_not_evidence"),
        ("replay", {"controller_fallback_count": 2}, "fallback_or_degraded_not_evidence"),
        ("source", {"fallback": "true"}, "unavailable_execution_evidence"),
        ("replay", {"fallback": None}, "unavailable_execution_evidence"),
        ("replay", {"fallback_count": "2"}, "unavailable_execution_evidence"),
        ("source", {"fallback_count": True}, "unavailable_execution_evidence"),
        ("replay", {"fallback_count": float("nan")}, "unavailable_execution_evidence"),
        ("source", {"planner_fallback_count": -1}, "unavailable_execution_evidence"),
        ("replay", {"fallback_reason": ["not-a-string"]}, "unavailable_execution_evidence"),
        ("replay", {"fallback": False, "fallback_count": 0}, "exact_match"),
    ],
)
def test_nested_planner_diagnostics_fallback_markers_fail_closed(
    side: str, diagnostics: dict[str, Any], expected_status: str
) -> None:
    source = _source_row(scenario_id="scenario_a", episode_id="episode-a")
    observed = json.loads(json.dumps(source))
    target = source if side == "source" else observed
    target["algorithm_metadata"]["planner_diagnostics"] = diagnostics
    case = {
        "planner_key": "goal",
        "scenario_id": "scenario_a",
        "seed": 111,
        "benchmark_eligible": True,
    }

    classification = _classify_replay_row(
        case,
        source,
        observed,
        SOURCE_REVISION,
        replay_checkout_clean=True,
        replay_checkout_stability_status=(
            "clean_stable" if expected_status == "exact_match" else None
        ),
    )
    assert classification["status"] == expected_status


@pytest.mark.parametrize(
    ("side", "runtime_status", "expected_evidence"),
    [
        ("source", "mystery", "unavailable_execution_availability"),
        ("replay", "mystery", "unavailable_execution_availability"),
        ("source", "unknown", "unavailable_execution_availability"),
        ("replay", "unknown", "unavailable_execution_availability"),
        ("source", None, "unavailable_execution_availability"),
        ("replay", 7, "unavailable_execution_availability"),
        ("source", "ok", "available"),
        ("replay", "success", "available"),
    ],
)
def test_nested_runtime_status_requires_a_known_available_value(
    side: str, runtime_status: Any, expected_evidence: str
) -> None:
    source = _source_row(scenario_id="scenario_a", episode_id="episode-a")
    observed = json.loads(json.dumps(source))
    target = source if side == "source" else observed
    target["algorithm_metadata"]["planner_runtime"] = {"status": runtime_status}
    case = {
        "planner_key": "goal",
        "scenario_id": "scenario_a",
        "seed": 111,
        "benchmark_eligible": True,
    }

    classification = _classify_replay_row(
        case,
        source,
        observed,
        SOURCE_REVISION,
        replay_checkout_clean=True,
        replay_checkout_stability_status="clean_stable",
    )

    assert classification[f"execution_evidence_{side}"] == expected_evidence
    assert classification["status"] == (
        "exact_match" if expected_evidence == "available" else "unavailable_execution_evidence"
    )


@pytest.mark.parametrize(
    ("side", "mutation"),
    [
        ("source", ("status", "error")),
        ("source", ("status", "invalid")),
        ("source", ("termination_reason", "error")),
        ("source", ("invalid_run", True)),
        ("source", ("event_ledger", {"exact_events": {"invalid_run": True}})),
        ("replay", ("status", "error")),
        ("replay", ("status", "invalid")),
        ("replay", ("termination_reason", "error")),
        ("replay", ("invalid_run", True)),
        ("replay", ("event_ledger", {"exact_events": {"invalid_run": True}})),
    ],
)
def test_canonical_invalid_runs_are_never_exact_replay_evidence(
    side: str, mutation: tuple[str, Any]
) -> None:
    source = _source_row(scenario_id="scenario_a", episode_id="episode-a")
    observed = json.loads(json.dumps(source))
    target = source if side == "source" else observed
    target[mutation[0]] = mutation[1]
    case = {
        "planner_key": "goal",
        "scenario_id": "scenario_a",
        "seed": 111,
        "benchmark_eligible": True,
    }

    classification = _classify_replay_row(
        case,
        source,
        observed,
        SOURCE_REVISION,
        replay_checkout_clean=True,
        replay_checkout_stability_status="clean_stable",
    )

    assert classification["status"] == "invalid_run_not_evidence"
    assert classification[f"invalid_run_evidence_{side}"] == "invalid"


def test_malformed_explicit_invalid_run_marker_is_unavailable() -> None:
    source = _source_row(scenario_id="scenario_a", episode_id="episode-a")
    source["invalid_run"] = "false"
    case = {
        "planner_key": "goal",
        "scenario_id": "scenario_a",
        "seed": 111,
        "benchmark_eligible": True,
    }

    classification = _classify_replay_row(
        case,
        source,
        json.loads(json.dumps(source)),
        SOURCE_REVISION,
        replay_checkout_clean=True,
        replay_checkout_stability_status="clean_stable",
    )

    assert classification["status"] == "unavailable_invalid_run_evidence"
    assert classification["invalid_run_evidence_source"] == "unavailable"


@pytest.mark.parametrize("missing_side", ["source", "replay", "both"])
def test_exact_replay_requires_nonempty_planner_config_hashes(missing_side: str) -> None:
    source = _source_row(scenario_id="scenario_a", episode_id="episode-a")
    observed = json.loads(json.dumps(source))
    case = {
        "planner_key": "goal",
        "scenario_id": "scenario_a",
        "seed": 111,
        "benchmark_eligible": True,
    }
    if missing_side in {"source", "both"}:
        source["algorithm_metadata"].pop("config_hash")
    if missing_side in {"replay", "both"}:
        observed["algorithm_metadata"].pop("config_hash")

    classification = _classify_replay_row(
        case, source, observed, SOURCE_REVISION, replay_checkout_clean=True
    )
    assert classification["status"] == "unavailable_planner_config_identity"


@pytest.mark.parametrize(
    ("checkout_clean", "expected_status"),
    [
        (False, "replay_checkout_dirty"),
        (True, "replay_checkout_cleanliness_unavailable"),
        (None, "replay_checkout_cleanliness_unavailable"),
    ],
)
def test_exact_replay_requires_clean_replay_checkout(
    checkout_clean: bool | None, expected_status: str
) -> None:
    row = _source_row(scenario_id="scenario_a", episode_id="episode-a")
    case = {
        "planner_key": "goal",
        "scenario_id": "scenario_a",
        "seed": 111,
        "benchmark_eligible": True,
    }

    classification = _classify_replay_row(
        case, row, row, SOURCE_REVISION, replay_checkout_clean=checkout_clean
    )
    assert classification["comparison"]["overall"] == "match"
    assert classification["same_repository_revision"] is True
    assert classification["status"] == expected_status


@pytest.mark.parametrize("asset_kind", ["scenario_map", "checkpoint_path"])
def test_exact_replay_requires_runtime_inputs_bound_to_source_git_tree(
    tmp_path: Path, asset_kind: str
) -> None:
    source = _source_row(scenario_id="scenario_a", episode_id="episode-a")
    runtime_asset = tmp_path / f"{asset_kind}.bin"
    runtime_asset.write_bytes(b"source runtime bytes")
    source_asset_sha256 = _sha256(runtime_asset)
    if asset_kind == "scenario_map":
        source["scenario_params"]["map_file"] = str(runtime_asset)
    else:
        source["algorithm_metadata"]["config"]["checkpoint_path"] = str(runtime_asset)
    _record_fixture_runtime_inputs(source)
    observed = json.loads(json.dumps(source))
    runtime_asset.write_bytes(b"changed replay bytes")
    replay_asset_sha256 = _sha256(runtime_asset)
    case = {
        "planner_key": "goal",
        "scenario_id": "scenario_a",
        "seed": 111,
        "benchmark_eligible": True,
    }

    classification = _classify_replay_row(
        case,
        source,
        observed,
        SOURCE_REVISION,
        replay_checkout_clean=True,
        replay_checkout_stability_status="clean_stable",
    )

    assert source_asset_sha256 != replay_asset_sha256
    assert classification["comparison"]["overall"] == "match"
    assert classification["same_repository_revision"] is True
    assert classification["status"] == "unavailable_runtime_input_identity"
    identity = classification["runtime_input_identity"]
    assert identity["status"] == "unavailable"
    assert identity["replay"]["status"] == "unavailable"
    assert identity["replay"]["reason"] == "runtime_asset_not_source_bound"


def test_exact_replay_records_git_tree_identity_for_tracked_runtime_inputs() -> None:
    row = _source_row(scenario_id="scenario_a", episode_id="episode-a")
    row["algorithm_metadata"]["config"]["checkpoint_path"] = "pyproject.toml"
    _record_fixture_runtime_inputs(row)
    case = {
        "planner_key": "goal",
        "scenario_id": "scenario_a",
        "seed": 111,
        "benchmark_eligible": True,
    }

    classification = _classify_replay_row(
        case,
        row,
        json.loads(json.dumps(row)),
        SOURCE_REVISION,
        replay_checkout_clean=True,
        replay_checkout_stability_status="clean_stable",
    )

    assert classification["status"] == "exact_match"
    identity = classification["runtime_input_identity"]
    assert identity["status"] == "verified_git_tree_match"
    assert [asset["kind"] for asset in identity["source"]["assets"]] == [
        "checkpoint_path",
        "scenario_map",
    ]
    assert all(len(asset["sha256"]) == 64 for asset in identity["source"]["assets"])


@pytest.mark.parametrize(
    ("environment_mutation", "expected_status", "identity_status"),
    [
        ("different", "replay_environment_mismatch", "replay_environment_mismatch"),
        ("missing", "unavailable_replay_environment_identity", "replay_environment_unavailable"),
    ],
)
def test_exact_replay_requires_source_bound_matching_environment(
    environment_mutation: str, expected_status: str, identity_status: str
) -> None:
    source = _source_row(scenario_id="scenario_a", episode_id="episode-a")
    replay = json.loads(json.dumps(source))
    replay_environment = json.loads(
        json.dumps(source["runtime_input_provenance"]["source_environment"])
    )
    if environment_mutation == "different":
        replay_environment = _different_fixture_execution_environment()
    else:
        replay_environment = None
    case = {
        "planner_key": "goal",
        "scenario_id": "scenario_a",
        "seed": 111,
        "benchmark_eligible": True,
    }

    classification = _classify_replay_row(
        case,
        source,
        replay,
        SOURCE_REVISION,
        replay_checkout_clean=True,
        replay_checkout_stability_status="clean_stable",
        replay_environment_identity=replay_environment,
    )

    assert classification["status"] == expected_status
    assert classification["runtime_input_identity"]["status"] == identity_status


def test_missing_historical_runtime_input_provenance_is_not_reconstructed() -> None:
    source = _source_row(scenario_id="scenario_a", episode_id="episode-a")
    source.pop("runtime_input_provenance")
    case = {
        "planner_key": "goal",
        "scenario_id": "scenario_a",
        "seed": 111,
        "benchmark_eligible": True,
    }

    classification = _classify_replay_row(
        case,
        source,
        json.loads(json.dumps(source)),
        SOURCE_REVISION,
        replay_checkout_clean=True,
        replay_checkout_stability_status="clean_stable",
    )

    assert classification["comparison"]["overall"] == "match"
    assert classification["status"] == "unavailable_runtime_input_identity"
    identity = classification["runtime_input_identity"]
    assert identity["source"]["status"] == "unavailable"
    assert (
        identity["source"]["reason"]
        == "historical_source_environment_or_runtime_inputs_not_recorded"
    )


def test_missing_historical_source_environment_remains_unavailable() -> None:
    source = _source_row(scenario_id="scenario_a", episode_id="episode-a")
    source["runtime_input_provenance"].pop("source_environment")
    case = {
        "planner_key": "goal",
        "scenario_id": "scenario_a",
        "seed": 111,
        "benchmark_eligible": True,
    }

    classification = _classify_replay_row(
        case,
        source,
        json.loads(json.dumps(source)),
        SOURCE_REVISION,
        replay_checkout_clean=True,
        replay_checkout_stability_status="clean_stable",
    )

    assert classification["status"] == "unavailable_runtime_input_identity"
    identity = classification["runtime_input_identity"]
    assert identity["source"]["status"] == "unavailable"
    assert identity["source"]["reason"] == "historical_source_environment_not_recorded"


def test_malformed_source_checkpoint_asset_list_fails_closed(tmp_path: Path) -> None:
    source = _source_row(scenario_id="scenario_a", episode_id="episode-a")
    prefix = tmp_path / "sacadrl"
    for component in (
        prefix.with_name("sacadrl.meta"),
        prefix.with_name("sacadrl.index"),
        prefix.with_name("sacadrl.data-00000-of-00001"),
    ):
        component.write_bytes(b"fixture checkpoint")
    source["algo"] = "sacadrl"
    source["algorithm_metadata"]["config"] = {
        "sacadrl_model_id": "registered-sacadrl",
        "sacadrl_checkpoint_path": str(prefix),
    }
    _record_fixture_runtime_inputs(source)
    source["runtime_input_provenance"]["assets"] = None
    case = {
        "planner_key": "sacadrl",
        "scenario_id": "scenario_a",
        "seed": 111,
        "benchmark_eligible": True,
    }

    classification = _classify_replay_row(
        case,
        source,
        json.loads(json.dumps(source)),
        SOURCE_REVISION,
        replay_checkout_clean=True,
        replay_checkout_stability_status="clean_stable",
    )

    assert classification["status"] == "unavailable_runtime_input_identity"
    assert classification["runtime_input_identity"]["source"]["reason"] == (
        "source_runtime_input_reference_unresolved"
    )


def test_malformed_native_model_path_fails_closed() -> None:
    source = _source_row(scenario_id="scenario_a", episode_id="episode-a")
    source["algorithm_metadata"]["config"]["checkpoint_path"] = 123
    case = {
        "planner_key": "goal",
        "scenario_id": "scenario_a",
        "seed": 111,
        "benchmark_eligible": True,
    }

    classification = _classify_replay_row(
        case,
        source,
        json.loads(json.dumps(source)),
        SOURCE_REVISION,
        replay_checkout_clean=True,
        replay_checkout_stability_status="clean_stable",
    )

    assert classification["status"] == "unavailable_runtime_input_identity"
    assert classification["runtime_input_identity"]["source"]["reason"] == (
        "source_runtime_input_reference_unresolved"
    )


def test_predictive_checkpoint_bytes_are_part_of_runtime_identity(
    tmp_path: Path, monkeypatch
) -> None:
    source = _source_row(scenario_id="scenario_a", episode_id="episode-a")
    checkpoint = tmp_path / "predictive.pt"
    checkpoint.write_bytes(b"source predictive checkpoint")
    config = {
        "predictive_model_id": "registered-predictor",
        "predictive_checkpoint_path": str(checkpoint),
    }
    source["algo"] = "prediction_planner"
    source["algorithm_metadata"]["algorithm"] = "prediction_planner"
    source["algorithm_metadata"]["config"] = config
    _record_fixture_runtime_inputs(source)
    replay = json.loads(json.dumps(source))
    checkpoint.write_bytes(b"changed predictive checkpoint")
    real_identity = materializer._git_tree_file_identity

    def identity_with_runtime_bytes(revision: str, path: Path, *, kind: str) -> dict[str, Any]:
        if path.resolve() == checkpoint.resolve():
            return {"kind": kind, "status": "verified", "sha256": _sha256(path)}
        return real_identity(revision, path, kind=kind)

    monkeypatch.setattr(materializer, "_git_tree_file_identity", identity_with_runtime_bytes)
    case = {
        "planner_key": "prediction_planner",
        "scenario_id": "scenario_a",
        "seed": 111,
        "benchmark_eligible": True,
    }

    classification = _classify_replay_row(
        case,
        source,
        replay,
        SOURCE_REVISION,
        replay_checkout_clean=True,
        replay_checkout_stability_status="clean_stable",
    )

    assert classification["status"] == "runtime_input_identity_mismatch"
    source_assets = classification["runtime_input_identity"]["source"]["assets"]
    replay_assets = classification["runtime_input_identity"]["replay"]["assets"]
    assert any(asset["kind"] == "predictive_checkpoint_path" for asset in source_assets)
    assert next(
        asset["sha256"] for asset in source_assets if asset["kind"] == "predictive_checkpoint_path"
    ) != next(
        asset["sha256"] for asset in replay_assets if asset["kind"] == "predictive_checkpoint_path"
    )


def test_sacadrl_checkpoint_bundle_bytes_are_part_of_runtime_identity(
    tmp_path: Path, monkeypatch
) -> None:
    source = _source_row(scenario_id="scenario_a", episode_id="episode-a")
    prefix = tmp_path / "sacadrl"
    components = [
        prefix.with_name("sacadrl.meta"),
        prefix.with_name("sacadrl.index"),
        prefix.with_name("sacadrl.data-00000-of-00001"),
    ]
    for component in components:
        component.write_bytes(f"source {component.suffix}".encode())
    config = {
        "sacadrl_model_id": "registered-sacadrl",
        "sacadrl_checkpoint_path": str(prefix),
    }
    source["algo"] = "sacadrl"
    source["algorithm_metadata"]["algorithm"] = "sacadrl"
    source["algorithm_metadata"]["config"] = config
    _record_fixture_runtime_inputs(source)
    replay = json.loads(json.dumps(source))
    components[2].write_bytes(b"changed TensorFlow data shard")
    real_identity = materializer._git_tree_file_identity

    def identity_with_runtime_bytes(revision: str, path: Path, *, kind: str) -> dict[str, Any]:
        if path.resolve() in {component.resolve() for component in components}:
            return {"kind": kind, "status": "verified", "sha256": _sha256(path)}
        return real_identity(revision, path, kind=kind)

    monkeypatch.setattr(materializer, "_git_tree_file_identity", identity_with_runtime_bytes)
    case = {
        "planner_key": "sacadrl",
        "scenario_id": "scenario_a",
        "seed": 111,
        "benchmark_eligible": True,
    }

    classification = _classify_replay_row(
        case,
        source,
        replay,
        SOURCE_REVISION,
        replay_checkout_clean=True,
        replay_checkout_stability_status="clean_stable",
    )

    assert classification["status"] == "runtime_input_identity_mismatch"
    source_assets = classification["runtime_input_identity"]["source"]["assets"]
    replay_assets = classification["runtime_input_identity"]["replay"]["assets"]
    source_data_hash = next(
        asset["sha256"]
        for asset in source_assets
        if asset["kind"] == "sacadrl_checkpoint_path" and ".data-" in asset["name"]
    )
    replay_data_hash = next(
        asset["sha256"]
        for asset in replay_assets
        if asset["kind"] == "sacadrl_checkpoint_path" and ".data-" in asset["name"]
    )
    assert source_data_hash != replay_data_hash


def test_crowdnav_height_default_checkpoint_and_model_inputs_are_bound(
    tmp_path: Path, monkeypatch
) -> None:
    repo_root, model_dir = _create_crowdnav_height_assets(tmp_path)
    monkeypatch.setattr(materializer, "CROWDNAV_HEIGHT_DEFAULT_REPO_ROOT", str(repo_root))
    monkeypatch.setattr(materializer, "CROWDNAV_HEIGHT_DEFAULT_MODEL_DIR", str(model_dir))
    source = _crowdnav_height_source_row(repo_root, model_dir, config={})
    replay = json.loads(json.dumps(source))
    case = {
        "planner_key": "crowdnav_height",
        "scenario_id": "scenario_a",
        "seed": 111,
        "benchmark_eligible": True,
    }

    classification = _classify_replay_row(
        case,
        source,
        replay,
        SOURCE_REVISION,
        replay_checkout_clean=True,
        replay_checkout_stability_status="clean_stable",
    )

    assert classification["status"] == "exact_match"
    identity = classification["runtime_input_identity"]
    checkpoint = next(
        asset
        for asset in identity["replay"]["assets"]
        if asset["kind"] == materializer.CROWDNAV_HEIGHT_CHECKPOINT_ASSET_KIND
    )
    assert checkpoint["reference"].endswith("/checkpoints/237800.pt")
    assert checkpoint["sha256"] == _sha256(model_dir / "checkpoints" / "237800.pt")
    assert {
        asset["kind"] for asset in identity["source"]["assets"]
    } >= materializer.CROWDNAV_HEIGHT_ASSET_KINDS


def test_crowdnav_height_explicit_checkpoint_byte_change_is_a_mismatch(
    tmp_path: Path,
) -> None:
    checkpoint_name = "selected-model.pt"
    repo_root, model_dir = _create_crowdnav_height_assets(tmp_path, checkpoint_name=checkpoint_name)
    source = _crowdnav_height_source_row(
        repo_root,
        model_dir,
        config={
            "repo_root": str(repo_root),
            "model_dir": str(model_dir),
            "checkpoint_name": checkpoint_name,
        },
    )
    replay = json.loads(json.dumps(source))
    checkpoint_path = model_dir / "checkpoints" / checkpoint_name
    checkpoint_path.write_bytes(b"changed explicit HEIGHT checkpoint")
    case = {
        "planner_key": "crowdnav_height",
        "scenario_id": "scenario_a",
        "seed": 111,
        "benchmark_eligible": True,
    }

    classification = _classify_replay_row(
        case,
        source,
        replay,
        SOURCE_REVISION,
        replay_checkout_clean=True,
        replay_checkout_stability_status="clean_stable",
    )

    assert classification["status"] == "runtime_input_identity_mismatch"
    checkpoint = next(
        asset
        for asset in classification["runtime_input_identity"]["replay"]["assets"]
        if asset["kind"] == materializer.CROWDNAV_HEIGHT_CHECKPOINT_ASSET_KIND
    )
    assert checkpoint["sha256"] == _sha256(checkpoint_path)


def test_missing_crowdnav_height_default_checkpoint_is_unavailable(
    tmp_path: Path, monkeypatch
) -> None:
    repo_root, model_dir = _create_crowdnav_height_assets(tmp_path, include_checkpoint=False)
    monkeypatch.setattr(materializer, "CROWDNAV_HEIGHT_DEFAULT_REPO_ROOT", str(repo_root))
    monkeypatch.setattr(materializer, "CROWDNAV_HEIGHT_DEFAULT_MODEL_DIR", str(model_dir))
    source = _crowdnav_height_source_row(repo_root, model_dir, config={})
    case = {
        "planner_key": "crowdnav_height",
        "scenario_id": "scenario_a",
        "seed": 111,
        "benchmark_eligible": True,
    }

    classification = _classify_replay_row(
        case,
        source,
        json.loads(json.dumps(source)),
        SOURCE_REVISION,
        replay_checkout_clean=True,
        replay_checkout_stability_status="clean_stable",
    )

    assert classification["status"] == "unavailable_runtime_input_identity"
    identity = classification["runtime_input_identity"]
    assert identity["source"]["status"] == "unavailable"
    assert identity["replay"]["status"] == "unavailable"
    assert any(
        asset.get("reason") == "crowdnav_height_runtime_file_missing"
        for asset in identity["replay"]["assets"]
    )


def test_exact_replay_rejects_different_tracked_scenario_map() -> None:
    source = _source_row(scenario_id="scenario_a", episode_id="episode-a")
    replay = json.loads(json.dumps(source))
    replay["scenario_params"]["map_file"] = "maps/svg_maps/02_simple_maps.svg"
    case = {
        "planner_key": "goal",
        "scenario_id": "scenario_a",
        "seed": 111,
        "benchmark_eligible": True,
    }

    classification = _classify_replay_row(
        case,
        source,
        replay,
        SOURCE_REVISION,
        replay_checkout_clean=True,
        replay_checkout_stability_status="clean_stable",
    )

    assert classification["comparison"]["overall"] == "match"
    assert classification["status"] == "runtime_input_identity_mismatch"
    identity = classification["runtime_input_identity"]
    assert identity["source"]["status"] == "verified"
    assert identity["replay"]["status"] == "verified"


def test_episode_failure_outcome_does_not_mark_execution_as_failed() -> None:
    row = _source_row(scenario_id="scenario_a", episode_id="episode-a")
    row["status"] = "failure"
    case = {
        "planner_key": "goal",
        "scenario_id": "scenario_a",
        "seed": 111,
        "benchmark_eligible": True,
    }

    classification = _classify_replay_row(
        case,
        row,
        row,
        SOURCE_REVISION,
        replay_checkout_clean=True,
        replay_checkout_stability_status="clean_stable",
    )
    assert classification["episode_status"] == "failure"
    assert classification["status"] == "exact_match"


def test_replay_checkout_provenance_records_dirty_status(monkeypatch) -> None:
    calls = iter(
        [
            SimpleNamespace(stdout="revision-a\n"),
            SimpleNamespace(stdout=" M robot_sf/planner/example.py\n"),
        ]
    )

    def fake_run(command, **kwargs):
        del command, kwargs
        return next(calls)

    monkeypatch.setattr("scripts.tools.materialize_benchmark_hard_cases.subprocess.run", fake_run)
    identity = _replay_checkout_provenance()
    assert identity["revision"] == "revision-a"
    assert identity["clean"] is False
    assert identity["status_entries"] == [" M robot_sf/planner/example.py"]
    assert (
        identity["status_sha256"] == hashlib.sha256(b" M robot_sf/planner/example.py\n").hexdigest()
    )


def test_replay_checkout_stability_requires_both_revision_and_status_hashes() -> None:
    before = _clean_checkout_snapshot("revision-a")
    after = _clean_checkout_snapshot("revision-a")
    assert _replay_checkout_stability_status(before, after, "revision-a") == "clean_stable"

    after.pop("status_sha256")
    assert _replay_checkout_stability_status(before, after, "revision-a") == "unavailable"


@pytest.mark.parametrize(
    "mutation",
    [
        "missing_schema",
        "missing_status_entries",
        "clean_with_dirty_entries",
        "arbitrary_digest",
        "malformed_entry",
    ],
)
def test_checkout_snapshot_schema_and_status_digest_must_be_consistent(mutation: str) -> None:
    before = _clean_checkout_snapshot("revision-a")
    after = _clean_checkout_snapshot("revision-a")
    if mutation == "missing_schema":
        before.pop("schema")
    elif mutation == "missing_status_entries":
        before.pop("status_entries")
    elif mutation == "clean_with_dirty_entries":
        before["status_entries"] = [" M robot_sf/planner/example.py"]
        before["status_sha256"] = hashlib.sha256(b" M robot_sf/planner/example.py\n").hexdigest()
    elif mutation == "arbitrary_digest":
        before["status_sha256"] = "0" * 64
    else:
        before["status_entries"] = [" M robot_sf/planner/example.py\n"]
        before["status_sha256"] = hashlib.sha256(b" M robot_sf/planner/example.py\n\n").hexdigest()

    assert _replay_checkout_stability_status(before, after, "revision-a") == "unavailable"


def test_checkout_snapshot_consistent_dirty_entries_are_not_clean() -> None:
    before = _clean_checkout_snapshot("revision-a")
    after = _clean_checkout_snapshot("revision-a")
    for snapshot in (before, after):
        snapshot["clean"] = False
        snapshot["status_entries"] = [" M robot_sf/planner/example.py"]
        snapshot["status_sha256"] = hashlib.sha256(b" M robot_sf/planner/example.py\n").hexdigest()

    assert _replay_checkout_stability_status(before, after, "revision-a") == "dirty"


@pytest.mark.parametrize(
    ("source_mode", "observed_mode", "expected_status"),
    [
        ("native", "native", "exact_match"),
        ("native", "adapter", "execution_mode_identity_mismatch"),
        ("unknown", "unknown", "unavailable_execution_mode_identity"),
        ("native", "malformed-extra-line", "invalid_output"),
    ],
)
def test_replay_records_episode_checksum_and_row_count(
    tmp_path: Path,
    monkeypatch,
    source_mode: str,
    observed_mode: str,
    expected_status: str,
) -> None:
    row = _source_row(scenario_id="scenario_a", episode_id="episode-a")
    monkeypatch.setattr(
        materializer, "_execution_environment_identity", _fixture_execution_environment
    )
    row["algorithm_metadata"]["planner_kinematics"]["execution_mode"] = source_mode
    case = {
        "planner_key": "goal",
        "scenario_id": "scenario_a",
        "seed": 111,
        "benchmark_eligible": True,
    }
    case_dir = tmp_path / f"{source_mode}-{observed_mode}"
    observed = json.loads(json.dumps(row))
    observed["algorithm_metadata"]["planner_kinematics"]["execution_mode"] = observed_mode
    real_run = subprocess.run

    def fake_run(command, *, cwd, stdout=None, stderr=None, check, **kwargs):
        if command[:2] in (["git", "ls-tree"], ["git", "cat-file"]):
            return real_run(command, cwd=cwd, check=check, **kwargs)
        if command[:2] == ["git", "rev-parse"]:
            return SimpleNamespace(stdout=f"{SOURCE_REVISION}\n")
        if command[:2] == ["git", "status"]:
            return SimpleNamespace(stdout="")
        assert stdout is not None and stderr is not None
        episode_path = Path(cwd) / command[command.index("--out") + 1]
        episode_path.parent.mkdir(parents=True, exist_ok=True)
        output = json.dumps(observed) + "\n"
        if observed_mode == "malformed-extra-line":
            output += "not-json\n"
        episode_path.write_text(output, encoding="utf-8")
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr("scripts.tools.materialize_benchmark_hard_cases.subprocess.run", fake_run)
    replay = _run_replay(
        case,
        row,
        REPO_ROOT / MATRIX_RELATIVE,
        case_dir,
        {"planners": [{"key": "goal", "benchmark_profile": "baseline-safe"}]},
    )

    episode_path = case_dir / "replay" / "episodes.jsonl"
    assert replay["status"] == expected_status
    assert replay["replay_row_count"] == 1
    assert replay["replay_malformed_line_count"] == int(observed_mode == "malformed-extra-line")
    assert replay["episode_output_sha256"] == _sha256(episode_path)
    assert replay["episode_output_checksum_status"] == "captured_at_run"
    assert replay["replay_environment_identity"] == _fixture_execution_environment()
    assert replay["replay_checkout_stability_status"] == "clean_stable"
    assert replay["replay_checkout_before"]["revision"] == SOURCE_REVISION
    assert replay["replay_checkout_after"]["revision"] == SOURCE_REVISION
    if observed_mode != "malformed-extra-line":
        assert replay["execution_mode_source"] == source_mode
        assert replay["execution_mode"] == observed_mode


@pytest.mark.parametrize(
    ("post_revision", "post_status", "expected_status", "checkout_status"),
    [
        (
            SOURCE_REVISION,
            " M robot_sf/planner/example.py\n",
            "replay_checkout_dirty",
            "dirty",
        ),
        (
            "moved-revision",
            "",
            "replay_checkout_head_changed_during_run",
            "head_changed",
        ),
    ],
)
def test_checkout_mutation_during_replay_blocks_exact_match(
    tmp_path: Path,
    monkeypatch,
    post_revision: str,
    post_status: str,
    expected_status: str,
    checkout_status: str,
) -> None:
    row = _source_row(scenario_id="scenario_a", episode_id="episode-a")
    monkeypatch.setattr(
        materializer, "_execution_environment_identity", _fixture_execution_environment
    )
    case = {
        "planner_key": "goal",
        "scenario_id": "scenario_a",
        "seed": 111,
        "benchmark_eligible": True,
    }

    state = {"revision": SOURCE_REVISION, "status": ""}
    real_run = subprocess.run

    def fake_run(command, *, cwd, stdout=None, stderr=None, check, **kwargs):
        if command[:2] in (["git", "ls-tree"], ["git", "cat-file"]):
            return real_run(command, cwd=cwd, check=check, **kwargs)
        if command[:2] == ["git", "rev-parse"]:
            return SimpleNamespace(stdout=f"{state['revision']}\n")
        if command[:2] == ["git", "status"]:
            return SimpleNamespace(stdout=state["status"])
        assert stdout is not None and stderr is not None
        episode_path = Path(cwd) / command[command.index("--out") + 1]
        episode_path.parent.mkdir(parents=True, exist_ok=True)
        episode_path.write_text(json.dumps(row) + "\n", encoding="utf-8")
        state["revision"] = post_revision
        state["status"] = post_status
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr("scripts.tools.materialize_benchmark_hard_cases.subprocess.run", fake_run)
    replay = _run_replay(
        case,
        row,
        REPO_ROOT / MATRIX_RELATIVE,
        tmp_path / checkout_status,
        {"planners": [{"key": "goal", "benchmark_profile": "baseline-safe"}]},
    )

    assert replay["comparison"]["overall"] == "match"
    assert replay["replay_checkout_stability_status"] == checkout_status
    assert replay["replay_checkout_before"]["clean"] is True
    assert replay["replay_checkout_after"]["revision"] == post_revision
    assert replay["status"] == expected_status


def test_nonfinite_values_are_null_with_explicit_source_paths() -> None:
    cleaned, paths = _sanitize({"metrics": {"clearance": float("inf")}})
    assert cleaned == {"metrics": {"clearance": None}}
    assert paths == [{"field": "$.metrics.clearance", "source_value": "inf"}]


def test_source_checksum_mismatch_fails_closed(tmp_path: Path) -> None:
    summary, campaign_root, matrix = _build_inputs(tmp_path, include_unavailable=False)
    episode = campaign_root / "runs/goal__differential_drive/episodes.jsonl"
    episode.write_text(episode.read_text() + "{}\n", encoding="utf-8")
    with pytest.raises(MaterializationError, match="checksum mismatch"):
        materialize(_args(summary, campaign_root, matrix, tmp_path / "output"))


@pytest.mark.parametrize(
    ("measurement", "field", "replacement", "message"),
    [
        ("outcome", "collision_event", False, "showcase outcome collision_event differs"),
        ("metrics", "near_misses", 99.0, "showcase metric near_misses differs"),
        ("selected_groups", None, ["timeout_event"], "selected event group timeout_event"),
    ],
)
def test_summary_measurements_and_event_groups_are_bound_to_hashed_source_row(
    tmp_path: Path,
    measurement: str,
    field: str | None,
    replacement: Any,
    message: str,
) -> None:
    summary_path, campaign_root, matrix = _build_inputs(tmp_path, include_unavailable=False)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    case = summary["cases"][0]
    if measurement == "selected_groups":
        case[measurement] = replacement
    else:
        case[measurement][field] = replacement
    summary_path.write_text(json.dumps(summary, sort_keys=True), encoding="utf-8")

    with pytest.raises(MaterializationError, match=message):
        materialize(_args(summary_path, campaign_root, matrix, tmp_path / "output"))


def test_resume_reuses_matching_attempt_without_reexecution(tmp_path: Path) -> None:
    summary, campaign_root, matrix = _build_inputs(tmp_path, include_unavailable=False)
    previous_dir = tmp_path / "previous"
    previous = materialize(_args(summary, campaign_root, matrix, previous_dir))
    case_file = previous_dir / previous["cases"][0]["case_file"]
    case_record = json.loads(case_file.read_text())
    replay = {
        "attempted": True,
        "status": "mismatch",
        "returncode": 0,
        "replay_revision": SOURCE_REVISION,
        "replay_environment_identity": _fixture_execution_environment(),
        "replay_checkout_clean": True,
        "replay_checkout_status_sha256": hashlib.sha256(b"").hexdigest(),
        "episode_output": "replay/episodes.jsonl",
    }
    case_record["replay"] = replay
    case_file.write_text(json.dumps(case_record), encoding="utf-8")
    replay_dir = case_file.parent / "replay"
    replay_dir.mkdir()
    observed = _source_row(scenario_id="scenario_a", episode_id="episode-a")
    (replay_dir / "episodes.jsonl").write_text(json.dumps(observed) + "\n", encoding="utf-8")
    manifest_path = previous_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["cases"][0]["replay"] = replay
    manifest["replay"]["attempted"] = 1
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    args = _args(summary, campaign_root, matrix, tmp_path / "resumed")
    args.resume_from = previous_dir
    resumed = materialize(args)
    case_relative = resumed["cases"][0]["case_file"]
    resumed_case = json.loads((tmp_path / "resumed" / case_relative).read_text())
    assert resumed["replay"]["reused_attempts"] == 1
    assert resumed["replay"]["new_attempted"] == 0
    assert resumed["replay"]["replay_revision"] == SOURCE_REVISION
    assert resumed["replay"]["replay_revisions"] == [SOURCE_REVISION]
    assert resumed["replay"]["materializer_revision"] != SOURCE_REVISION
    assert resumed_case["replay"]["reused"] is True
    replay_artifact = (
        tmp_path / "resumed" / case_relative.replace("case.json", "replay/episodes.jsonl")
    )
    assert replay_artifact.is_file()
    assert resumed_case["replay"]["episode_output_sha256"] == _sha256(replay_artifact)
    assert (
        resumed_case["replay"]["episode_output_checksum_status"]
        == "calculated_on_resume_prior_receipt_unavailable"
    )
    assert (
        resumed_case["replay"]["episode_output_checksum_origin"]
        == "calculated_on_resume_prior_receipt_unavailable"
    )
    assert resumed_case["replay"]["execution_mode_source"] == "native"
    assert resumed_case["replay"]["execution_mode"] == "native"
    assert resumed_case["replay"]["replay_checkout_stability_status"] == "unavailable"
    assert resumed_case["replay"]["replay_checkout_clean"] is None
    assert resumed_case["replay"]["status"] == "replay_checkout_cleanliness_unavailable"


@pytest.mark.parametrize(
    ("mutation", "expected_status"),
    [
        ("metrics", "mismatch_same_revision"),
        ("identity", "replay_identity_mismatch"),
        ("unchanged", "replay_artifact_checksum_unverified"),
    ],
)
def test_resume_rederives_classification_when_prior_episode_checksum_is_absent(
    tmp_path: Path, mutation: str, expected_status: str
) -> None:
    summary, campaign_root, matrix = _build_inputs(tmp_path, include_unavailable=False)
    previous_dir = tmp_path / "previous"
    previous = materialize(_args(summary, campaign_root, matrix, previous_dir))
    case_file = previous_dir / previous["cases"][0]["case_file"]
    case_record = json.loads(case_file.read_text(encoding="utf-8"))
    source_episode = campaign_root / "runs/goal__differential_drive/episodes.jsonl"
    source_row = json.loads(source_episode.read_text(encoding="utf-8").splitlines()[0])
    observed = json.loads(json.dumps(source_row))
    if mutation == "metrics":
        observed["metrics"]["near_misses"] = 99.0
    elif mutation == "identity":
        observed["seed"] = 112

    replay_dir = case_file.parent / "replay"
    replay_dir.mkdir()
    (replay_dir / "episodes.jsonl").write_text(json.dumps(observed) + "\n", encoding="utf-8")
    replay = {
        "attempted": True,
        "status": "exact_match",
        "returncode": 0,
        "source_revision": SOURCE_REVISION,
        "replay_revision": SOURCE_REVISION,
        "replay_environment_identity": _fixture_execution_environment(),
        "replay_checkout_before": _clean_checkout_snapshot(SOURCE_REVISION),
        "replay_checkout_after": _clean_checkout_snapshot(SOURCE_REVISION),
        "replay_checkout_clean": True,
        "replay_checkout_status_sha256": hashlib.sha256(b"").hexdigest(),
        "same_repository_revision": True,
        "identity": {"status": "match"},
        "comparison": {"overall": "match"},
        "execution_mode": "native",
        "execution_mode_source": "native",
        "config_hash_source": source_row["algorithm_metadata"]["config_hash"],
        "config_hash_replay": observed["algorithm_metadata"]["config_hash"],
        "episode_output": "replay/episodes.jsonl",
    }
    # Deliberately omit episode_output_sha256 to model a prior receipt that did
    # not bind the preserved replay bytes.
    case_record["replay"] = replay
    case_file.write_text(json.dumps(case_record), encoding="utf-8")
    manifest_path = previous_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["cases"][0]["replay"] = replay
    manifest["replay"]["attempted"] = 1
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    args = _args(summary, campaign_root, matrix, tmp_path / "resumed")
    args.resume_from = previous_dir
    resumed = materialize(args)
    resumed_case = json.loads(
        (tmp_path / "resumed" / resumed["cases"][0]["case_file"]).read_text(encoding="utf-8")
    )

    assert resumed_case["replay"]["status"] == expected_status
    assert resumed_case["replay"]["status"] != "exact_match"
    assert (
        resumed_case["replay"]["episode_output_checksum_status"]
        == "calculated_on_resume_prior_receipt_unavailable"
    )
    if mutation == "metrics":
        assert resumed_case["replay"]["comparison"]["overall"] == "mismatch"
    elif mutation == "identity":
        assert resumed_case["replay"]["identity"]["status"] == "mismatch"
    else:
        assert resumed_case["replay"]["checksum_unverified_derived_status"] == "exact_match"
        repeated_args = _args(summary, campaign_root, matrix, tmp_path / "resumed-again")
        repeated_args.resume_from = tmp_path / "resumed"
        repeated = materialize(repeated_args)
        repeated_case = json.loads(
            (tmp_path / "resumed-again" / repeated["cases"][0]["case_file"]).read_text(
                encoding="utf-8"
            )
        )
        assert repeated_case["replay"]["status"] == "replay_artifact_checksum_unverified"


@pytest.mark.parametrize(
    ("mutation", "expected_stability", "expected_clean", "expected_status"),
    [
        ("valid", "clean_stable", True, "exact_match"),
        (
            "manifest_conflict",
            "clean_stable",
            True,
            "replay_receipt_manifest_mismatch",
        ),
        ("missing_status_entries", "unavailable", None, "replay_checkout_cleanliness_unavailable"),
        (
            "clean_with_dirty_entries",
            "unavailable",
            None,
            "replay_checkout_cleanliness_unavailable",
        ),
        ("arbitrary_digest", "unavailable", None, "replay_checkout_cleanliness_unavailable"),
        ("missing_schema", "unavailable", None, "replay_checkout_cleanliness_unavailable"),
    ],
)
def test_resume_recomputes_checkout_snapshot_consistency(
    tmp_path: Path,
    mutation: str,
    expected_stability: str,
    expected_clean: bool | None,
    expected_status: str,
) -> None:
    summary, campaign_root, matrix = _build_inputs(tmp_path, include_unavailable=False)
    previous_dir = tmp_path / "previous"
    previous = materialize(_args(summary, campaign_root, matrix, previous_dir))
    case_file = previous_dir / previous["cases"][0]["case_file"]
    case_record = json.loads(case_file.read_text(encoding="utf-8"))
    source_episode = campaign_root / "runs/goal__differential_drive/episodes.jsonl"
    source_row = json.loads(source_episode.read_text(encoding="utf-8").splitlines()[0])
    output_bytes = (json.dumps(source_row) + "\n").encode("utf-8")
    replay_dir = case_file.parent / "replay"
    replay_dir.mkdir()
    (replay_dir / "episodes.jsonl").write_bytes(output_bytes)

    before = _clean_checkout_snapshot(SOURCE_REVISION)
    after = _clean_checkout_snapshot(SOURCE_REVISION)
    if mutation == "missing_status_entries":
        before.pop("status_entries")
    elif mutation == "clean_with_dirty_entries":
        for snapshot in (before, after):
            snapshot["status_entries"] = [" M robot_sf/planner/example.py"]
            snapshot["status_sha256"] = hashlib.sha256(
                b" M robot_sf/planner/example.py\n"
            ).hexdigest()
    elif mutation == "arbitrary_digest":
        before["status_sha256"] = "0" * 64
    elif mutation == "missing_schema":
        before.pop("schema")

    replay = {
        "attempted": True,
        "status": "exact_match",
        "returncode": 0,
        "source_revision": SOURCE_REVISION,
        "replay_revision": SOURCE_REVISION,
        "replay_environment_identity": _fixture_execution_environment(),
        "replay_checkout_before": before,
        "replay_checkout_after": after,
        "replay_checkout_clean": True,
        "episode_output": "replay/episodes.jsonl",
        "episode_output_sha256": hashlib.sha256(output_bytes).hexdigest(),
        "episode_output_checksum_origin": "captured_at_run",
        "episode_output_checksum_status": "captured_at_run",
    }
    case_record["replay"] = replay
    case_file.write_text(json.dumps(case_record), encoding="utf-8")
    manifest_path = previous_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["cases"][0]["replay"] = replay
    if mutation == "manifest_conflict":
        manifest["cases"][0]["replay"] = {"attempted": False, "status": "not_attempted"}
    manifest["replay"]["attempted"] = 1
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    args = _args(summary, campaign_root, matrix, tmp_path / "resumed")
    args.resume_from = previous_dir
    resumed = materialize(args)
    resumed_case = json.loads(
        (tmp_path / "resumed" / resumed["cases"][0]["case_file"]).read_text(encoding="utf-8")
    )

    assert resumed_case["replay"]["episode_output_checksum_status"] == "verified"
    assert resumed_case["replay"]["replay_checkout_stability_status"] == expected_stability
    assert resumed_case["replay"]["replay_checkout_clean"] is expected_clean
    assert resumed_case["replay"]["status"] == expected_status
    if mutation == "manifest_conflict":
        assert (
            resumed_case["replay"]["resume_receipt_consistency_status"]
            == "manifest_case_replay_mismatch"
        )
    elif mutation == "valid":
        assert resumed_case["replay"]["resume_receipt_consistency_status"] == "match"


@pytest.mark.parametrize(
    ("prior_environment_matches_source", "expected_status"),
    [(True, "exact_match"), (False, "replay_environment_mismatch")],
)
def test_resume_preserves_prior_environment_across_materializer_environment_change(
    tmp_path: Path,
    monkeypatch,
    prior_environment_matches_source: bool,
    expected_status: str,
) -> None:
    summary, campaign_root, matrix = _build_inputs(tmp_path, include_unavailable=False)
    previous_dir = tmp_path / "previous"
    previous = materialize(_args(summary, campaign_root, matrix, previous_dir))
    case_file = previous_dir / previous["cases"][0]["case_file"]
    case_record = json.loads(case_file.read_text(encoding="utf-8"))
    source_episode = campaign_root / "runs/goal__differential_drive/episodes.jsonl"
    source_row = json.loads(source_episode.read_text(encoding="utf-8").splitlines()[0])
    output_bytes = (json.dumps(source_row) + "\n").encode("utf-8")
    replay_dir = case_file.parent / "replay"
    replay_dir.mkdir()
    (replay_dir / "episodes.jsonl").write_bytes(output_bytes)
    prior_environment = (
        _fixture_execution_environment()
        if prior_environment_matches_source
        else _different_fixture_execution_environment()
    )
    current_environment = (
        _different_fixture_execution_environment()
        if prior_environment_matches_source
        else _fixture_execution_environment()
    )
    replay = {
        "attempted": True,
        "status": "exact_match",
        "returncode": 0,
        "source_revision": SOURCE_REVISION,
        "replay_revision": SOURCE_REVISION,
        "replay_environment_identity": prior_environment,
        "replay_checkout_before": _clean_checkout_snapshot(SOURCE_REVISION),
        "replay_checkout_after": _clean_checkout_snapshot(SOURCE_REVISION),
        "replay_checkout_clean": True,
        "episode_output": "replay/episodes.jsonl",
        "episode_output_sha256": hashlib.sha256(output_bytes).hexdigest(),
        "episode_output_checksum_origin": "captured_at_run",
        "episode_output_checksum_status": "captured_at_run",
    }
    case_record["replay"] = replay
    case_file.write_text(json.dumps(case_record), encoding="utf-8")
    manifest_path = previous_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["cases"][0]["replay"] = replay
    manifest["replay"]["attempted"] = 1
    manifest["execution_environment"]["replay_attempt_environments"][case_record["case_id"]] = (
        prior_environment
    )
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    monkeypatch.setattr(
        materializer,
        "_execution_environment_identity",
        lambda: current_environment,
    )
    args = _args(summary, campaign_root, matrix, tmp_path / "resumed")
    args.resume_from = previous_dir
    resumed = materialize(args)
    resumed_case = json.loads(
        (tmp_path / "resumed" / resumed["cases"][0]["case_file"]).read_text(encoding="utf-8")
    )

    assert resumed_case["replay"]["status"] == expected_status
    assert resumed_case["replay"]["replay_environment_identity"] == prior_environment
    assert (
        resumed["execution_environment"]["materializer_environment_identity"] == current_environment
    )
    assert (
        resumed["execution_environment"]["replay_attempt_environments"][resumed_case["case_id"]]
        == prior_environment
    )


def test_resume_preserves_the_expected_hash_after_a_replay_artifact_mismatch(
    tmp_path: Path,
) -> None:
    summary, campaign_root, matrix = _build_inputs(tmp_path, include_unavailable=False)
    previous_dir = tmp_path / "previous"
    previous = materialize(_args(summary, campaign_root, matrix, previous_dir))
    case_file = previous_dir / previous["cases"][0]["case_file"]
    case_record = json.loads(case_file.read_text(encoding="utf-8"))
    source_episode = campaign_root / "runs/goal__differential_drive/episodes.jsonl"
    source_row = json.loads(source_episode.read_text(encoding="utf-8").splitlines()[0])
    original_bytes = (json.dumps(source_row) + "\n").encode()
    observed = json.loads(json.dumps(source_row))
    observed["metrics"]["near_misses"] = 99.0
    replay_dir = case_file.parent / "replay"
    replay_dir.mkdir()
    episode_path = replay_dir / "episodes.jsonl"
    episode_path.write_text(json.dumps(observed) + "\n", encoding="utf-8")
    replay = {
        "attempted": True,
        "status": "exact_match",
        "returncode": 0,
        "source_revision": SOURCE_REVISION,
        "replay_revision": SOURCE_REVISION,
        "replay_environment_identity": _fixture_execution_environment(),
        "replay_checkout_before": _clean_checkout_snapshot(SOURCE_REVISION),
        "replay_checkout_after": _clean_checkout_snapshot(SOURCE_REVISION),
        "replay_checkout_clean": True,
        "replay_checkout_status_sha256": hashlib.sha256(b"").hexdigest(),
        "same_repository_revision": True,
        "identity": {"status": "match"},
        "comparison": {"overall": "match"},
        "execution_mode": "native",
        "execution_mode_source": "native",
        "config_hash_source": source_row["algorithm_metadata"]["config_hash"],
        "config_hash_replay": observed["algorithm_metadata"]["config_hash"],
        "episode_output": "replay/episodes.jsonl",
        "episode_output_sha256": hashlib.sha256(original_bytes).hexdigest(),
        "episode_output_checksum_origin": "captured_at_run",
        "episode_output_checksum_status": "captured_at_run",
    }
    case_record["replay"] = replay
    case_file.write_text(json.dumps(case_record), encoding="utf-8")
    manifest_path = previous_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["cases"][0]["replay"] = replay
    manifest["replay"]["attempted"] = 1
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    args = _args(summary, campaign_root, matrix, tmp_path / "resumed")
    args.resume_from = previous_dir
    resumed = materialize(args)
    resumed_case_path = tmp_path / "resumed" / resumed["cases"][0]["case_file"]
    resumed_case = json.loads(resumed_case_path.read_text(encoding="utf-8"))
    observed_hash = _sha256(resumed_case_path.parent / resumed_case["replay"]["episode_output"])
    expected_hash = hashlib.sha256(original_bytes).hexdigest()
    assert resumed_case["replay"]["status"] == "replay_artifact_checksum_mismatch"
    assert resumed_case["replay"]["episode_output_sha256"] == expected_hash
    assert resumed_case["replay"]["episode_output_observed_sha256"] == observed_hash

    repeated_args = _args(summary, campaign_root, matrix, tmp_path / "resumed-again")
    repeated_args.resume_from = tmp_path / "resumed"
    repeated = materialize(repeated_args)
    repeated_case = json.loads(
        (tmp_path / "resumed-again" / repeated["cases"][0]["case_file"]).read_text(encoding="utf-8")
    )
    assert repeated_case["replay"]["status"] == "replay_artifact_checksum_mismatch"
    assert repeated_case["replay"]["episode_output_sha256"] == expected_hash


def test_resume_preserves_attempt_when_prior_replay_directory_is_missing(tmp_path: Path) -> None:
    summary, campaign_root, matrix = _build_inputs(tmp_path, include_unavailable=False)
    previous_dir = tmp_path / "previous"
    previous = materialize(_args(summary, campaign_root, matrix, previous_dir))
    case_file = previous_dir / previous["cases"][0]["case_file"]
    case_record = json.loads(case_file.read_text(encoding="utf-8"))
    prior_replay = {
        "attempted": True,
        "status": "runner_failed",
        "returncode": 1,
        "replay_revision": "prior-replay-revision",
        "replay_environment_identity": _fixture_execution_environment(),
    }
    case_record["replay"] = prior_replay
    case_file.write_text(json.dumps(case_record), encoding="utf-8")
    manifest_path = previous_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["cases"][0]["replay"] = prior_replay
    manifest["replay"]["attempted"] = 1
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    args = _args(summary, campaign_root, matrix, tmp_path / "resumed")
    args.replay_limit = 1
    args.resume_from = previous_dir
    resumed = materialize(args)
    resumed_case = json.loads(
        (tmp_path / "resumed" / resumed["cases"][0]["case_file"]).read_text(encoding="utf-8")
    )

    assert resumed["replay"]["attempted"] == 1
    assert resumed["replay"]["new_attempted"] == 0
    assert resumed["replay"]["preserved_attempts_missing_artifacts"] == 1
    assert resumed["replay"]["status_counts"] == {"replay_artifact_missing_on_resume": 1}
    assert resumed_case["replay"]["attempted"] is True
    assert resumed_case["replay"]["status"] == "replay_artifact_missing_on_resume"
    assert resumed_case["replay"]["resume_prior_status"] == "runner_failed"
    assert resumed_case["replay"]["resume_artifact_status"] == "missing"
