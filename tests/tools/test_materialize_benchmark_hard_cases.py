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

from scripts.tools.materialize_benchmark_hard_cases import (
    MaterializationError,
    _classify_replay_row,
    _compare,
    _materialize_replay_matrix,
    _replay_checkout_provenance,
    _replay_identity,
    _replay_ineligibility,
    _run_replay,
    _sanitize,
    _selected_cases,
    _showcase_tool_snapshot,
    materialize,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
MATRIX_RELATIVE = "configs/scenarios/classic_interactions_francis2023.yaml"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


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
        "git_hash": "source-revision",
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
        "git": {"commit": "source-revision"},
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
            "campaign_source_revision": "source-revision",
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
    assert case_a["source"]["campaign_source_revision"] == "source-revision"
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
        case, source, observed, "source-revision", replay_checkout_clean=True
    )
    assert classification["status"] == expected_status


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
        case, source, observed, "source-revision", replay_checkout_clean=True
    )
    assert classification["status"] == "unavailable_planner_config_identity"


@pytest.mark.parametrize(
    ("checkout_clean", "expected_status"),
    [
        (False, "replay_checkout_dirty"),
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
        case, row, row, "source-revision", replay_checkout_clean=checkout_clean
    )
    assert classification["comparison"]["overall"] == "match"
    assert classification["same_repository_revision"] is True
    assert classification["status"] == expected_status


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
        case, row, row, "source-revision", replay_checkout_clean=True
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

    def fake_run(command, *, cwd, stdout, stderr, check):
        del stdout, stderr, check
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
        "source-revision",
        replay_checkout_clean=True,
        replay_checkout_status_sha256=hashlib.sha256(b"").hexdigest(),
    )

    episode_path = case_dir / "replay" / "episodes.jsonl"
    assert replay["status"] == expected_status
    assert replay["replay_row_count"] == 1
    assert replay["replay_malformed_line_count"] == int(observed_mode == "malformed-extra-line")
    assert replay["episode_output_sha256"] == _sha256(episode_path)
    assert replay["episode_output_checksum_status"] == "captured_at_run"
    if observed_mode != "malformed-extra-line":
        assert replay["execution_mode_source"] == source_mode
        assert replay["execution_mode"] == observed_mode


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
        "replay_revision": "prior-replay-revision",
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
    assert resumed["replay"]["replay_revision"] == "prior-replay-revision"
    assert resumed["replay"]["replay_revisions"] == ["prior-replay-revision"]
    assert resumed["replay"]["materializer_revision"] != "prior-replay-revision"
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
        "source_revision": "source-revision",
        "replay_revision": "source-revision",
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
        "source_revision": "source-revision",
        "replay_revision": "source-revision",
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
