"""Focused contracts for adversarial replay-gallery selection and verification."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

import pytest
import yaml

from robot_sf.adversarial import replay_gallery
from robot_sf.benchmark.fallback_policy import availability_payload


def _candidate() -> dict[str, Any]:
    return {
        "start": {"x": 1.0, "y": 1.0, "theta": 0.0},
        "goal": {"x": 3.0, "y": 1.0, "theta": 0.0},
        "spawn_time_s": 0.0,
        "pedestrian_speed_mps": 1.0,
        "pedestrian_delay_s": 0.0,
        "scenario_seed": 7,
    }


def _episode(
    *, revision: str | None = "a" * 40, config_hash: str = "test-algo-config"
) -> dict[str, Any]:
    record: dict[str, Any] = {
        "episode_id": "test_episode",
        "scenario_id": "test_scenario",
        "seed": 7,
        "algo": "goal",
        "algorithm_metadata": {
            "status": "ok",
            "planner_kinematics": {"execution_mode": "native"},
            "config_hash": config_hash,
        },
        "status": "failure",
        "termination_reason": "collision",
        "outcome": {
            "route_complete": False,
            "collision_event": True,
            "timeout_event": False,
        },
        "metrics": {
            "success": False,
            "collisions": 1,
            "near_misses": 0,
            "snqi": 1.0,
            "path_efficiency": 0.5,
        },
    }
    if revision is not None:
        record["git_hash"] = revision
    return record


def _source_manifest(
    tmp_path: Path,
    *,
    source_revision: str | None = "a" * 40,
    manifest_revision: str | None = None,
) -> Path:
    bundle = tmp_path / "candidate_0000"
    bundle.mkdir()
    scenario_path = bundle / "scenario.yaml"
    candidate = _candidate()
    scenario = {
        "name": "test_scenario",
        "seeds": [7],
        "metadata": {"adversarial_candidate": candidate},
    }
    scenario_path.write_text(
        yaml.safe_dump({"scenarios": [scenario]}, sort_keys=False),
        encoding="utf-8",
    )
    episode_path = bundle / "episode_records.jsonl"
    episode_path.write_text(json.dumps(_episode(revision=source_revision)) + "\n", encoding="utf-8")
    payload = {
        "schema_version": "adversarial-search-manifest.v1",
        "config": {
            "policy": "goal",
            "objective": "constraints_first_lexicographic_v1",
            "search_method": "random",
            "seed": 101,
            "budget": 1,
            "record_forces": True,
        },
        "candidates": [
            {
                "candidate": candidate,
                "objective_value": 4.5,
                "analysis_eligibility": {
                    "eligible": True,
                    "certificate_ok": True,
                    "execution_mode": "native",
                },
                "certification_status": {
                    "schema_version": "scenario_cert.v1",
                    "status": "passed",
                    "reason": "valid route certificate",
                    "details": {"certificates": [{"classification": "valid"}]},
                },
                "failure_attribution": {
                    "status": "attributed",
                    "primary_failure": "collision",
                    "reasons": ["source episode records collision"],
                    "details": {
                        "termination_reason": "collision",
                        "execution_mode": "native",
                        "readiness_status": "native",
                        "availability_status": "available",
                    },
                },
                "effective_scenario_hash": replay_gallery.compute_effective_scenario_hash(
                    scenario, {}
                ),
                "scenario_yaml_path": str(scenario_path),
                "episode_record_path": str(episode_path),
                "bundle_path": str(bundle),
            }
        ],
    }
    if manifest_revision is not None:
        payload["source_revision"] = manifest_revision
    manifest_path = tmp_path / "search_manifest.json"
    manifest_path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")
    return manifest_path


def _replay_episode(
    *, revision: str | None = "a" * 40, config_hash: str = "test-algo-config"
) -> dict[str, Any]:
    record = _episode(revision=revision, config_hash=config_hash)
    record["algorithm_metadata"]["simulation_step_trace"] = {
        "schema_version": "simulation-step-trace.v1",
        "dt": 0.1,
        "steps": [
            {
                "time_s": 0.1,
                "robot": {"position": [1.0, 1.0], "heading": 0.0, "velocity": [1.0, 0.0]},
                "pedestrians": [{"position": [1.5, 1.0], "surface_clearance_m": 0.1}],
            },
            {
                "time_s": 0.2,
                "robot": {"position": [1.1, 1.0], "heading": 0.0, "velocity": [1.0, 0.0]},
                "pedestrians": [{"position": [1.6, 1.0], "surface_clearance_m": 0.1}],
            },
        ],
    }
    return record


def _runner_summary(
    *, preflight_status: str = "ok", failures: list[dict[str, str]] | None = None
) -> dict[str, Any]:
    """Build a canonical map-runner summary with its matching availability receipt."""
    failures = failures or []
    summary = {
        "total_jobs": 1,
        "written": 1,
        "failed_jobs": len(failures),
        "skipped_jobs": 0,
        "failures": failures,
        "preflight": {"status": preflight_status},
        "algorithm_metadata_contract": {
            "status": "ok",
            "planner_kinematics": {"execution_mode": "native"},
        },
    }
    summary["benchmark_availability"] = availability_payload(summary)
    return summary


def _install_fake_replay(
    monkeypatch: Any,
    *,
    revision: str | None = "a" * 40,
    replay_config_hash: str = "test-algo-config",
    mismatch: bool = False,
    runner_summary: dict[str, Any] | None = None,
    checkout_clean: bool = True,
    checkout_revision: str | None = None,
    rendered_map_paths: list[Path | None] | None = None,
) -> list[dict[str, Any]]:
    calls: list[dict[str, Any]] = []

    def fake_run_batch(scenario_path: Path, **kwargs: Any) -> dict[str, Any]:
        calls.append({"scenario_path": scenario_path, **kwargs})
        scenarios = yaml.safe_load(scenario_path.read_text(encoding="utf-8"))["scenarios"]
        record = _replay_episode(revision=revision, config_hash=replay_config_hash)
        record["scenario_id"] = scenarios[0]["name"]
        if mismatch:
            record["outcome"]["collision_event"] = False
            record["metrics"]["collisions"] = 0
        kwargs["out_path"].write_text(json.dumps(record) + "\n", encoding="utf-8")
        return runner_summary or _runner_summary()

    def fake_render(
        episode_row: Any, outputs: list[str], out_dir: Path, **kwargs: Any
    ) -> dict[str, Any]:
        assert episode_row.episode_id == "test_episode"
        assert outputs == ["still", "filmstrip", "trajectory"]
        if rendered_map_paths is not None:
            map_value = episode_row.raw.get("replay_map_path")
            rendered_map_paths.append(Path(map_value) if map_value else None)
        out_dir.mkdir(parents=True, exist_ok=True)
        artifact = out_dir / "trajectory.png"
        artifact.write_bytes(b"fixture png")
        return {
            "determinism_check_status": "pass",
            "artifact_paths": [str(artifact)],
            "provenance_sidecar": None,
            "caption_fragment": None,
        }

    monkeypatch.setattr(replay_gallery, "run_batch", fake_run_batch)
    monkeypatch.setattr(replay_gallery, "replay_episode_and_generate_figures", fake_render)
    monkeypatch.setattr(
        replay_gallery,
        "_git_checkout_state",
        lambda _root: {
            "revision": checkout_revision or revision,
            "clean": checkout_clean,
            "dirty_paths": [] if checkout_clean else ["fixture-dirty.py"],
        },
    )
    return calls


def test_gallery_materializes_replays_compares_and_renders(
    tmp_path: Path, monkeypatch: Any
) -> None:
    manifest = _source_manifest(tmp_path)
    calls = _install_fake_replay(monkeypatch)

    output_dir = tmp_path / "gallery"
    result = replay_gallery.build_replay_gallery(manifest, output_dir, video=False)

    assert result["summary"]["selected_case_count"] == 1
    assert result["summary"]["replay_match_count"] == 1
    assert result["source"]["search_method"] == "random"
    assert result["selection"]["mechanism_cluster_deduplication"]["status"] == "available"
    case = result["cases"][0]
    assert case["replay_match"] == "match"
    assert case["verification_status"] == "verified"
    assert case["feasibility_verdict"]["status"] == "admissible_by_source_certificate"
    assert case["rendering"]["status"] == "rendered"
    assert case["rendering"]["artifacts"] == [f"cases/{case['case_id']}/figures/trajectory.png"]
    assert case["video_status"] == {
        "requested": False,
        "renderer": "none",
        "status": "disabled",
        "reason": None,
        "artifacts": [],
    }
    assert calls[0]["record_simulation_step_trace"] is True
    assert calls[0]["video_enabled"] is False
    assert calls[0]["scenario_path"].is_file()
    assert (output_dir / case["case_manifest_path"]).is_file()
    assert (output_dir / "README.md").is_file()
    assert (output_dir / "gallery_manifest.json").is_file()


@pytest.mark.parametrize(
    ("field", "value", "expected_disposition"),
    [
        ("availability_status", "fallback", "source_availability_fallback"),
        ("availability_status", "degraded", "source_availability_degraded"),
        ("availability_status", None, "source_availability_missing_or_malformed"),
    ],
)
def test_gallery_excludes_source_rows_without_native_available_execution(
    tmp_path: Path,
    field: str,
    value: str | None,
    expected_disposition: str,
) -> None:
    manifest = _source_manifest(tmp_path)
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    details = payload["candidates"][0]["failure_attribution"]["details"]
    if value is None:
        details.pop(field)
    else:
        details[field] = value
    manifest.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")

    result = replay_gallery.build_replay_gallery(
        manifest, tmp_path / "gallery", render=False, video=False
    )

    assert result["summary"]["selected_case_count"] == 0
    assert result["summary"]["dispositions"][expected_disposition] == 1


def test_gallery_excludes_source_runtime_fallback_even_if_manifest_says_native(
    tmp_path: Path,
) -> None:
    manifest = _source_manifest(tmp_path)
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    episode_path = Path(payload["candidates"][0]["episode_record_path"])
    episode = _episode()
    episode["algorithm_metadata"]["planner_runtime"] = {"fallback_used": True}
    episode_path.write_text(json.dumps(episode) + "\n", encoding="utf-8")

    result = replay_gallery.build_replay_gallery(
        manifest, tmp_path / "gallery", render=False, video=False
    )

    assert result["summary"]["selected_case_count"] == 0
    assert result["summary"]["dispositions"]["source_runtime_fallback_or_degraded"] == 1


def test_gallery_does_not_treat_unavailable_diagnostic_metrics_as_runtime_fallback(
    tmp_path: Path, monkeypatch: Any
) -> None:
    manifest = _source_manifest(tmp_path)
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    source_episode_path = Path(payload["candidates"][0]["episode_record_path"])
    source_episode = _episode()
    source_episode["metrics"]["diagnostic_only"] = {"status": "unavailable"}
    source_episode["algorithm_metadata"]["paired_effect_metric_producer"] = {
        "status": "unavailable",
        "fields": {"optional_metric": {"status": "unavailable"}},
    }
    source_episode["algorithm_metadata"]["simulation_step_trace"] = {
        "reset": {"routes": {"status": "unavailable"}}
    }
    source_episode_path.write_text(json.dumps(source_episode) + "\n", encoding="utf-8")
    _install_fake_replay(monkeypatch)
    fake_run_batch = replay_gallery.run_batch

    def replay_with_unavailable_diagnostic_metric(
        scenario_path: Path, **kwargs: Any
    ) -> dict[str, Any]:
        summary = fake_run_batch(scenario_path, **kwargs)
        record_path = kwargs["out_path"]
        replay_episode = json.loads(record_path.read_text(encoding="utf-8"))
        replay_episode["metrics"]["diagnostic_only"] = {"status": "unavailable"}
        replay_episode["algorithm_metadata"]["paired_effect_metric_producer"] = {
            "status": "unavailable",
            "fields": {"optional_metric": {"status": "unavailable"}},
        }
        replay_episode["algorithm_metadata"]["simulation_step_trace"] = {
            "reset": {"routes": {"status": "unavailable"}}
        }
        record_path.write_text(json.dumps(replay_episode) + "\n", encoding="utf-8")
        return summary

    monkeypatch.setattr(replay_gallery, "run_batch", replay_with_unavailable_diagnostic_metric)

    result = replay_gallery.build_replay_gallery(
        manifest, tmp_path / "gallery", render=False, video=False
    )

    assert result["summary"]["selected_case_count"] == 1
    assert result["cases"][0]["replay_match"] == "match"


def test_gallery_still_rejects_explicit_fallback_inside_diagnostic_metadata(
    tmp_path: Path,
) -> None:
    manifest = _source_manifest(tmp_path)
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    episode_path = Path(payload["candidates"][0]["episode_record_path"])
    episode = _episode()
    episode["algorithm_metadata"]["simulation_step_trace"] = {
        "status": "unavailable",
        "fallback_used": True,
    }
    episode_path.write_text(json.dumps(episode) + "\n", encoding="utf-8")

    result = replay_gallery.build_replay_gallery(
        manifest, tmp_path / "gallery", render=False, video=False
    )

    assert result["summary"]["selected_case_count"] == 0
    assert result["summary"]["dispositions"]["source_runtime_fallback_or_degraded"] == 1


def test_gallery_does_not_verify_replay_from_a_dirty_checkout(
    tmp_path: Path, monkeypatch: Any
) -> None:
    manifest = _source_manifest(tmp_path)
    _install_fake_replay(monkeypatch, checkout_clean=False)

    result = replay_gallery.build_replay_gallery(manifest, tmp_path / "gallery", video=False)

    case = result["cases"][0]
    assert case["replay_match"] == "match"
    assert case["verification_status"] == "outcome_reproduced_checkout_dirty"
    assert case["source"]["gallery_checkout"]["dirty_paths"] == ["fixture-dirty.py"]


def test_gallery_does_not_verify_when_checkout_becomes_dirty_during_replay(
    tmp_path: Path, monkeypatch: Any
) -> None:
    manifest = _source_manifest(tmp_path)
    _install_fake_replay(monkeypatch)
    checkout_states = iter(
        [
            {"revision": "a" * 40, "clean": True, "dirty_paths": []},
            {
                "revision": "a" * 40,
                "clean": False,
                "dirty_paths": ["robot_sf/adversarial/replay_gallery.py"],
            },
        ]
    )
    monkeypatch.setattr(replay_gallery, "_git_checkout_state", lambda _root: next(checkout_states))

    result = replay_gallery.build_replay_gallery(manifest, tmp_path / "gallery", video=False)

    case = result["cases"][0]
    assert case["verification_status"] == "outcome_reproduced_checkout_dirty"
    assert case["source"]["gallery_checkout"]["clean"] is True
    assert case["source"]["gallery_checkout_after_replay"] == {
        "revision": "a" * 40,
        "clean": False,
        "dirty_paths": ["robot_sf/adversarial/replay_gallery.py"],
    }


def test_gallery_does_not_verify_replay_from_a_different_checkout_revision(
    tmp_path: Path, monkeypatch: Any
) -> None:
    manifest = _source_manifest(tmp_path)
    _install_fake_replay(monkeypatch, revision="a" * 40, checkout_revision="b" * 40)

    result = replay_gallery.build_replay_gallery(manifest, tmp_path / "gallery", video=False)

    case = result["cases"][0]
    assert case["replay_match"] == "match"
    assert case["verification_status"] == "outcome_reproduced_checkout_revision_changed"


def test_gallery_can_verify_episode_source_when_manifest_has_no_revision(
    tmp_path: Path, monkeypatch: Any
) -> None:
    manifest = _source_manifest(tmp_path)
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    assert "source_revision" not in payload
    _install_fake_replay(monkeypatch, revision="a" * 40)

    result = replay_gallery.build_replay_gallery(manifest, tmp_path / "gallery", video=False)

    case = result["cases"][0]
    assert case["source"]["manifest_revision"] is None
    assert case["source"]["episode_revision"] == "a" * 40
    assert case["verification_status"] == "verified"


def test_gallery_uses_manifest_revision_when_episode_revision_is_missing(
    tmp_path: Path, monkeypatch: Any
) -> None:
    manifest = _source_manifest(tmp_path, source_revision=None, manifest_revision="a" * 40)
    _install_fake_replay(monkeypatch, revision="a" * 40)

    result = replay_gallery.build_replay_gallery(manifest, tmp_path / "gallery", video=False)

    case = result["cases"][0]
    assert case["source"]["episode_revision"] is None
    assert case["source"]["manifest_revision"] == "a" * 40
    assert case["verification_status"] == "verified"


def test_gallery_rejects_replay_with_different_planner_config_hash(
    tmp_path: Path, monkeypatch: Any
) -> None:
    manifest = _source_manifest(tmp_path)
    _install_fake_replay(monkeypatch, replay_config_hash="different-config")

    result = replay_gallery.build_replay_gallery(manifest, tmp_path / "gallery", video=False)

    case = result["cases"][0]
    assert case["replay_match"] == "mismatch"
    assert case["verification_status"] == "replay_input_mismatch"
    assert case["replay"]["algorithm_config_binding"]["status"] == "mismatch"


def test_gallery_passes_materialized_map_to_renderer_and_marks_external_map_unbound(
    tmp_path: Path, monkeypatch: Any
) -> None:
    manifest = _source_manifest(tmp_path)
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    candidate_row = payload["candidates"][0]
    scenario_path = Path(candidate_row["scenario_yaml_path"])
    scenario_payload = yaml.safe_load(scenario_path.read_text(encoding="utf-8"))
    map_path = tmp_path / "fixture-map.svg"
    map_bytes = b"<svg xmlns='http://www.w3.org/2000/svg'></svg>\n"
    map_path.write_bytes(map_bytes)
    scenario = scenario_payload["scenarios"][0]
    scenario["map_file"] = str(map_path)
    scenario_path.write_text(yaml.safe_dump(scenario_payload, sort_keys=False), encoding="utf-8")
    candidate_row["effective_scenario_hash"] = replay_gallery.compute_effective_scenario_hash(
        scenario, {}
    )
    manifest.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")
    rendered_maps: list[Path | None] = []
    _install_fake_replay(monkeypatch, rendered_map_paths=rendered_maps)

    output_dir = tmp_path / "gallery"
    result = replay_gallery.build_replay_gallery(manifest, output_dir, video=False)

    case = result["cases"][0]
    rendered_map = rendered_maps[0]
    assert rendered_map is not None and rendered_map.is_file()
    assert rendered_map.read_bytes() == map_bytes
    assert case["replay_match"] == "match"
    assert case["verification_status"] == "outcome_reproduced_source_inputs_unbound"
    assert case["rendering"]["map_context"] == {
        "status": "provided",
        "path": rendered_map.relative_to(output_dir).as_posix(),
        "sha256": replay_gallery._sha256_file(rendered_map),
    }


def test_gallery_detects_materialized_map_bytes_changed_before_replay(
    tmp_path: Path, monkeypatch: Any
) -> None:
    manifest = _source_manifest(tmp_path)
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    candidate_row = payload["candidates"][0]
    scenario_path = Path(candidate_row["scenario_yaml_path"])
    scenario_payload = yaml.safe_load(scenario_path.read_text(encoding="utf-8"))
    map_path = tmp_path / "fixture-map.png"
    map_path.write_bytes(b"source map bytes")
    scenario = scenario_payload["scenarios"][0]
    scenario["map_file"] = str(map_path)
    scenario_path.write_text(yaml.safe_dump(scenario_payload, sort_keys=False), encoding="utf-8")
    candidate_row["effective_scenario_hash"] = replay_gallery.compute_effective_scenario_hash(
        scenario, {}
    )
    manifest.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")
    materialize_scenario = replay_gallery._materialize_scenario

    def tamper_materialized_map(source: Path, input_dir: Path, **kwargs: Any) -> dict[str, Any]:
        materialization = materialize_scenario(source, input_dir, **kwargs)
        for asset in materialization.get("assets", []):
            if asset.get("field") == "map_file":
                bundled_map = input_dir.parent / asset["bundle_path"]
                bundled_map.write_bytes(bundled_map.read_bytes() + b"tampered")
        return materialization

    monkeypatch.setattr(replay_gallery, "_materialize_scenario", tamper_materialized_map)
    _install_fake_replay(monkeypatch)

    result = replay_gallery.build_replay_gallery(manifest, tmp_path / "gallery", video=False)

    case = result["cases"][0]
    assert case["replay_match"] == "mismatch"
    assert case["verification_status"] == "replay_input_mismatch"
    assert any(
        check.get("input") == "map_file"
        and check.get("status") == "mismatch"
        and "bundle_sha256" in check
        for check in case["replay"]["source_input_binding"]["checks"]
    )


def test_gallery_detects_map_bytes_changed_after_candidate_selection(
    tmp_path: Path, monkeypatch: Any
) -> None:
    manifest = _source_manifest(tmp_path)
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    candidate_row = payload["candidates"][0]
    scenario_path = Path(candidate_row["scenario_yaml_path"])
    scenario_payload = yaml.safe_load(scenario_path.read_text(encoding="utf-8"))
    map_path = tmp_path / "fixture-map.svg"
    map_path.write_bytes(b"selected map bytes")
    scenario = scenario_payload["scenarios"][0]
    scenario["map_file"] = str(map_path)
    scenario_path.write_text(yaml.safe_dump(scenario_payload, sort_keys=False), encoding="utf-8")
    candidate_row["effective_scenario_hash"] = replay_gallery.compute_effective_scenario_hash(
        scenario, {}
    )
    manifest.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")
    materialize_scenario = replay_gallery._materialize_scenario

    def mutate_source_map(source: Path, input_dir: Path, **kwargs: Any) -> dict[str, Any]:
        map_path.write_bytes(b"changed after selection")
        return materialize_scenario(source, input_dir, **kwargs)

    monkeypatch.setattr(replay_gallery, "_materialize_scenario", mutate_source_map)
    calls = _install_fake_replay(monkeypatch)

    result = replay_gallery.build_replay_gallery(manifest, tmp_path / "gallery", video=False)

    case = result["cases"][0]
    assert case["verification_status"] == "not_replayed_map_changed_after_selection"
    assert case["materialization"]["selection_binding"] == {
        "status": "mismatch",
        "selected_source_sha256": hashlib.sha256(b"selected map bytes").hexdigest(),
        "materialized_source_sha256": hashlib.sha256(b"changed after selection").hexdigest(),
    }
    assert calls == []


def test_tracked_source_file_binding_requires_matching_tracked_bytes(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    config = repo / "planner.json"
    source_bytes = b'{"max_speed": 0.5}\n'
    config.write_bytes(source_bytes)
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    subprocess.run(["git", "add", "planner.json"], cwd=repo, check=True)

    bound = replay_gallery._tracked_source_file_binding(
        root=repo,
        field="algo_config_path",
        source_path="planner.json",
        repository_relative=True,
        expected_sha256=hashlib.sha256(source_bytes).hexdigest(),
    )

    assert bound["status"] == "bound"
    config.write_bytes(b'{"max_speed": 0.4}\n')
    mismatch = replay_gallery._tracked_source_file_binding(
        root=repo,
        field="algo_config_path",
        source_path="planner.json",
        repository_relative=True,
        expected_sha256=hashlib.sha256(source_bytes).hexdigest(),
    )
    assert mismatch["status"] == "mismatch"


def test_gallery_rejects_raw_category_difference_even_when_objective_projection_matches() -> None:
    source = _episode()
    replay = _episode()
    for episode in (source, replay):
        episode["outcome"].update(
            route_complete=False,
            collision_event=False,
            timeout_event=False,
        )
        episode["metrics"].update(success=False, collisions=0)
    replay["outcome"]["timeout_event"] = True
    replay["termination_reason"] = "timeout"

    matches, comparison = replay_gallery._outcomes_match(
        source, replay, "constraints_first_lexicographic_v1", tolerance=1e-6
    )

    assert comparison["projection_matches"] is True
    assert comparison["failure_attribution_matches"] is False
    assert "outcome.timeout_event" in comparison["differences"]
    assert "failure_attribution.primary_failure" in comparison["differences"]
    assert matches is False


def test_gallery_marks_requested_video_unavailable_when_runner_emits_none(
    tmp_path: Path, monkeypatch: Any
) -> None:
    manifest = _source_manifest(tmp_path)
    calls = _install_fake_replay(monkeypatch)

    result = replay_gallery.build_replay_gallery(
        manifest,
        tmp_path / "gallery",
        video=True,
        render=False,
    )

    case = result["cases"][0]
    assert calls[0]["video_enabled"] is True
    assert case["video_status"] == {
        "requested": True,
        "renderer": "synthetic",
        "status": "unavailable",
        "reason": "canonical_runner_did_not_emit_video_artifact",
        "artifacts": [],
    }
    assert case["replay"]["video_status"] == case["video_status"]


def test_gallery_preserves_non_finite_source_episode_bytes(
    tmp_path: Path, monkeypatch: Any
) -> None:
    manifest = _source_manifest(tmp_path)
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    episode_path = Path(payload["candidates"][0]["episode_record_path"])
    source_record = _episode()
    source_record["metrics"]["min_separation_corrupted_m"] = float("inf")
    source_bytes = (json.dumps(source_record) + "\n").encode("utf-8")
    episode_path.write_bytes(source_bytes)
    _install_fake_replay(monkeypatch)

    output_dir = tmp_path / "gallery"
    result = replay_gallery.build_replay_gallery(manifest, output_dir, video=False)

    case = result["cases"][0]
    case_dir = (output_dir / case["case_manifest_path"]).parent
    bundled_episode = case_dir / case["source"]["bundled_episode_record_path"]
    assert bundled_episode.read_bytes() == source_bytes


def test_gallery_copies_and_hashes_file_backed_runner_configuration(
    tmp_path: Path, monkeypatch: Any
) -> None:
    manifest = _source_manifest(tmp_path)
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    config_path = tmp_path / "planner.json"
    config_bytes = b'{"max_speed": 0.5}\n'
    config_path.write_bytes(config_bytes)
    payload["config"]["algo_config_path"] = str(config_path)
    manifest.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")
    calls = _install_fake_replay(monkeypatch)

    output_dir = tmp_path / "gallery"
    result = replay_gallery.build_replay_gallery(manifest, output_dir, video=False)

    case = result["cases"][0]
    case_dir = (output_dir / case["case_manifest_path"]).parent
    file_input = case["execution_config"]["file_inputs"][0]
    bundled_config = case_dir / file_input["bundle_path"]
    assert file_input["field"] == "algo_config_path"
    assert bundled_config.read_bytes() == config_bytes
    assert file_input["source_sha256"] == replay_gallery._sha256_file(config_path)
    assert calls[0]["algo_config_path"] == bundled_config.as_posix()
    assert case["execution_config"]["horizon"] == 100
    assert case["execution_config"]["dt"] == 0.1


@pytest.mark.parametrize(
    ("field", "value", "reason"),
    [
        ("horizon", 0, "horizon must be a positive integer"),
        ("workers", True, "workers must be a positive integer"),
        ("dt", float("inf"), "dt must be a finite positive number"),
        ("record_forces", "false", "record_forces must be a boolean"),
        ("benchmark_profile", "", "benchmark_profile must be a non-empty string"),
    ],
)
def test_gallery_preserves_invalid_runner_configuration_without_replaying(
    tmp_path: Path,
    monkeypatch: Any,
    field: str,
    value: Any,
    reason: str,
) -> None:
    manifest = _source_manifest(tmp_path)
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["config"][field] = value
    manifest.write_text(json.dumps(payload) + "\n", encoding="utf-8")
    calls = _install_fake_replay(monkeypatch)

    result = replay_gallery.build_replay_gallery(manifest, tmp_path / "gallery", video=False)

    case = result["cases"][0]
    assert case["verification_status"] == "not_replayed_config_unavailable"
    assert case["execution_config"]["status"] == "unavailable"
    assert reason in case["replay_error"]
    assert calls == []


def test_gallery_uses_canonical_defaults_for_null_runner_settings(
    tmp_path: Path, monkeypatch: Any
) -> None:
    manifest = _source_manifest(tmp_path)
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["config"].update(
        horizon=None,
        dt=None,
        workers=None,
        record_forces=None,
        benchmark_profile=None,
    )
    manifest.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")
    calls = _install_fake_replay(monkeypatch)

    result = replay_gallery.build_replay_gallery(manifest, tmp_path / "gallery", video=False)

    case = result["cases"][0]
    assert case["execution_config"]["status"] == "ready"
    assert case["execution_config"]["horizon"] == 100
    assert case["execution_config"]["dt"] == 0.1
    assert case["execution_config"]["workers"] == 1
    assert case["execution_config"]["record_forces"] is True
    assert case["execution_config"]["benchmark_profile"] == "baseline-safe"
    assert calls[0]["horizon"] == 100
    assert calls[0]["dt"] == 0.1


def test_gallery_preserves_unknown_source_revision_without_calling_it_verified(
    tmp_path: Path, monkeypatch: Any
) -> None:
    manifest = _source_manifest(tmp_path, source_revision=None)
    _install_fake_replay(monkeypatch, revision=None)

    result = replay_gallery.build_replay_gallery(manifest, tmp_path / "gallery", video=False)

    case = result["cases"][0]
    assert case["replay_match"] == "match"
    assert case["verification_status"] == "outcome_reproduced_source_revision_unknown"
    assert result["source"]["revision_status"] == "unknown"


def test_gallery_preserves_conflicting_manifest_and_episode_revisions(
    tmp_path: Path, monkeypatch: Any
) -> None:
    manifest = _source_manifest(tmp_path)
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["source_revision"] = "b" * 40
    manifest.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")
    _install_fake_replay(monkeypatch, revision="a" * 40)

    result = replay_gallery.build_replay_gallery(manifest, tmp_path / "gallery", video=False)

    case = result["cases"][0]
    assert case["replay_match"] == "match"
    assert case["verification_status"] == "source_revision_conflict"
    assert case["source"]["manifest_revision"] == "b" * 40
    assert case["source"]["episode_revision"] == "a" * 40
    assert case["source"]["revision_conflict"] is True


def test_gallery_uses_top_level_search_method_and_labels_changed_revision(
    tmp_path: Path, monkeypatch: Any
) -> None:
    manifest = _source_manifest(tmp_path)
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["config"].pop("search_method")
    payload["search_method"] = "tpe"
    manifest.write_text(json.dumps(payload) + "\n", encoding="utf-8")
    _install_fake_replay(monkeypatch, revision="b" * 40)

    result = replay_gallery.build_replay_gallery(manifest, tmp_path / "gallery", video=False)

    case = result["cases"][0]
    assert result["source"]["search_method"] == "tpe"
    assert case["source"]["search_method"] == "tpe"
    assert case["replay_match"] == "match"
    assert case["verification_status"] == "outcome_reproduced_revision_changed"


def test_gallery_reports_outcome_mismatch_without_verified_status(
    tmp_path: Path, monkeypatch: Any
) -> None:
    manifest = _source_manifest(tmp_path)
    _install_fake_replay(monkeypatch, mismatch=True)

    result = replay_gallery.build_replay_gallery(manifest, tmp_path / "gallery", video=False)

    case = result["cases"][0]
    assert case["replay_match"] == "mismatch"
    assert case["verification_status"] == "replay_mismatch"
    assert case["replay"]["outcome_matches"] is False


@pytest.mark.parametrize(
    ("preflight_status", "failures", "expected_error"),
    [
        ("fallback", [], "replay_preflight_not_successful"),
        ("skipped", [], "replay_preflight_not_successful"),
        ("ok", [{"error": "planner worker failed"}], "replay_runner_reported_failures"),
    ],
)
def test_gallery_preserves_comparison_but_rejects_unavailable_replay(
    tmp_path: Path,
    monkeypatch: Any,
    preflight_status: str,
    failures: list[dict[str, str]],
    expected_error: str,
) -> None:
    manifest = _source_manifest(tmp_path)
    summary = _runner_summary(preflight_status=preflight_status, failures=failures)
    _install_fake_replay(monkeypatch, runner_summary=summary)

    result = replay_gallery.build_replay_gallery(manifest, tmp_path / "gallery", video=False)

    case = result["cases"][0]
    assert case["replay_match"] == "unavailable"
    assert case["verification_status"] == "replay_execution_unavailable"
    assert case["replay"]["availability_error"] == expected_error
    assert case["replay"]["summary"] == summary
    assert case["replay"]["identity_matches"] is True
    assert case["replay"]["outcome_matches"] is True
    assert case["replay"]["objective_matches"] is True


def test_gallery_fails_closed_when_canonical_availability_is_missing(
    tmp_path: Path, monkeypatch: Any
) -> None:
    manifest = _source_manifest(tmp_path)
    summary = _runner_summary()
    summary.pop("benchmark_availability")
    _install_fake_replay(monkeypatch, runner_summary=summary)

    result = replay_gallery.build_replay_gallery(manifest, tmp_path / "gallery", video=False)

    case = result["cases"][0]
    assert case["replay_match"] == "unavailable"
    assert case["verification_status"] == "replay_execution_unavailable"
    assert case["replay"]["availability_error"] == (
        "replay_benchmark_availability_missing_or_malformed"
    )
    assert case["replay"]["outcome_matches"] is True


def test_gallery_rejects_runtime_fallback_marker_even_when_row_matches(
    tmp_path: Path, monkeypatch: Any
) -> None:
    manifest = _source_manifest(tmp_path)
    summary = _runner_summary()
    summary["algorithm_metadata_contract"]["planner_runtime"] = {"fallback_used": True}
    summary["benchmark_availability"] = availability_payload(summary)
    _install_fake_replay(monkeypatch, runner_summary=summary)

    result = replay_gallery.build_replay_gallery(manifest, tmp_path / "gallery", video=False)

    case = result["cases"][0]
    assert case["replay_match"] == "unavailable"
    assert case["verification_status"] == "replay_execution_unavailable"
    assert case["replay"]["availability_error"] == "replay_benchmark_unavailable"
    assert case["replay"]["summary"]["benchmark_availability"]["benchmark_success"] is False
    assert case["replay"]["identity_matches"] is True
    assert case["replay"]["outcome_matches"] is True


def test_gallery_rejects_adapter_mode_replay_even_when_source_was_native(
    tmp_path: Path, monkeypatch: Any
) -> None:
    manifest = _source_manifest(tmp_path)
    summary = _runner_summary()
    summary["algorithm_metadata_contract"]["planner_kinematics"]["execution_mode"] = "adapter"
    summary["benchmark_availability"] = availability_payload(summary)
    _install_fake_replay(monkeypatch, runner_summary=summary)

    result = replay_gallery.build_replay_gallery(manifest, tmp_path / "gallery", video=False)

    case = result["cases"][0]
    assert case["replay_match"] == "unavailable"
    assert case["verification_status"] == "replay_execution_unavailable"
    assert case["replay"]["availability_error"] == "replay_execution_mode_not_native"


def test_gallery_rejects_fallback_marker_in_replay_episode_row(
    tmp_path: Path, monkeypatch: Any
) -> None:
    manifest = _source_manifest(tmp_path)
    _install_fake_replay(monkeypatch)
    fake_run_batch = replay_gallery.run_batch

    def replay_with_row_fallback(scenario_path: Path, **kwargs: Any) -> dict[str, Any]:
        summary = fake_run_batch(scenario_path, **kwargs)
        record_path = kwargs["out_path"]
        record = json.loads(record_path.read_text(encoding="utf-8"))
        record["algorithm_metadata"]["planner_runtime"] = {"fallback_used": True}
        record_path.write_text(json.dumps(record) + "\n", encoding="utf-8")
        return summary

    monkeypatch.setattr(replay_gallery, "run_batch", replay_with_row_fallback)

    result = replay_gallery.build_replay_gallery(manifest, tmp_path / "gallery", video=False)

    case = result["cases"][0]
    assert case["replay_match"] == "unavailable"
    assert case["verification_status"] == "replay_execution_unavailable"
    assert case["replay"]["availability_error"] == ("replay_episode_runtime_fallback_or_degraded")
    assert case["replay"]["identity_matches"] is True
    assert case["replay"]["outcome_matches"] is True


def test_gallery_rejects_scenario_changed_after_selection_before_replay(
    tmp_path: Path, monkeypatch: Any
) -> None:
    manifest = _source_manifest(tmp_path)
    calls = _install_fake_replay(monkeypatch)
    materialize_scenario = replay_gallery._materialize_scenario

    def mutate_scenario_then_materialize(
        source: Path, input_dir: Path, **kwargs: Any
    ) -> dict[str, Any]:
        payload = yaml.safe_load(source.read_text(encoding="utf-8"))
        payload["scenarios"][0]["metadata"]["runtime_option"] = "changed-after-selection"
        source.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
        return materialize_scenario(source, input_dir, **kwargs)

    monkeypatch.setattr(replay_gallery, "_materialize_scenario", mutate_scenario_then_materialize)

    result = replay_gallery.build_replay_gallery(manifest, tmp_path / "gallery", video=False)

    case = result["cases"][0]
    assert case["replay_match"] == "unavailable"
    assert case["verification_status"] == "not_replayed_scenario_changed_after_selection"
    assert case["materialization"]["selection_binding"]["status"] == "mismatch"
    assert calls == []


def test_gallery_rejects_source_episode_changed_after_selection_before_replay(
    tmp_path: Path, monkeypatch: Any
) -> None:
    manifest = _source_manifest(tmp_path)
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    episode_path = Path(payload["candidates"][0]["episode_record_path"])
    calls = _install_fake_replay(monkeypatch)
    materialize_scenario = replay_gallery._materialize_scenario

    def mutate_episode_then_materialize(
        source: Path, input_dir: Path, **kwargs: Any
    ) -> dict[str, Any]:
        episode = _episode()
        episode["outcome"]["collision_event"] = False
        episode_path.write_text(json.dumps(episode) + "\n", encoding="utf-8")
        return materialize_scenario(source, input_dir, **kwargs)

    monkeypatch.setattr(replay_gallery, "_materialize_scenario", mutate_episode_then_materialize)

    result = replay_gallery.build_replay_gallery(manifest, tmp_path / "gallery", video=False)

    case = result["cases"][0]
    assert case["replay_match"] == "unavailable"
    assert case["verification_status"] == "not_replayed_source_episode_changed_after_selection"
    assert case["materialization"]["source_episode_selection_binding"]["status"] == "mismatch"
    assert calls == []


def test_gallery_rejects_route_overrides_changed_after_selection_before_replay(
    tmp_path: Path, monkeypatch: Any
) -> None:
    manifest = _source_manifest(tmp_path)
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    scenario_path = Path(payload["candidates"][0]["scenario_yaml_path"])
    scenario_payload = yaml.safe_load(scenario_path.read_text(encoding="utf-8"))
    scenario = scenario_payload["scenarios"][0]
    route_path = scenario_path.parent / "route-overrides.yaml"
    route_payload = {"route": {"waypoints": [[1.0, 1.0], [2.0, 2.0]]}}
    route_path.write_text(yaml.safe_dump(route_payload), encoding="utf-8")
    scenario["route_overrides_file"] = str(route_path)
    scenario_path.write_text(yaml.safe_dump(scenario_payload, sort_keys=False), encoding="utf-8")
    payload["candidates"][0]["effective_scenario_hash"] = (
        replay_gallery.compute_effective_scenario_hash(scenario, route_payload)
    )
    manifest.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")
    calls = _install_fake_replay(monkeypatch)
    materialize_scenario = replay_gallery._materialize_scenario

    def mutate_route_then_materialize(
        source: Path, input_dir: Path, **kwargs: Any
    ) -> dict[str, Any]:
        changed_route = {"route": {"waypoints": [[1.0, 1.0], [3.0, 3.0]]}}
        route_path.write_text(yaml.safe_dump(changed_route), encoding="utf-8")
        return materialize_scenario(source, input_dir, **kwargs)

    monkeypatch.setattr(replay_gallery, "_materialize_scenario", mutate_route_then_materialize)

    result = replay_gallery.build_replay_gallery(manifest, tmp_path / "gallery", video=False)

    case = result["cases"][0]
    assert case["replay_match"] == "unavailable"
    assert case["verification_status"] == (
        "not_replayed_effective_scenario_changed_after_selection"
    )
    assert case["materialization"]["selection_binding"]["status"] == "mismatch"
    assert calls == []


def test_gallery_deduplicates_identical_effective_scenarios(tmp_path: Path) -> None:
    manifest = _source_manifest(tmp_path)
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["candidates"].append(dict(payload["candidates"][0]))
    manifest.write_text(json.dumps(payload) + "\n", encoding="utf-8")

    result = replay_gallery.build_replay_gallery(
        manifest, tmp_path / "gallery", render=False, video=False
    )

    assert result["summary"]["selected_case_count"] == 1
    assert result["summary"]["dispositions"]["duplicate_effective_scenario"] == 1


def test_gallery_repeated_fixture_runs_have_byte_stable_manifests(
    tmp_path: Path, monkeypatch: Any
) -> None:
    manifest = _source_manifest(tmp_path)
    _install_fake_replay(monkeypatch)

    first = replay_gallery.build_replay_gallery(manifest, tmp_path / "gallery_one", video=False)
    second = replay_gallery.build_replay_gallery(manifest, tmp_path / "gallery_two", video=False)

    assert first["cases"][0]["case_id"] == second["cases"][0]["case_id"]
    assert (tmp_path / "gallery_one" / "gallery_manifest.json").read_bytes() == (
        tmp_path / "gallery_two" / "gallery_manifest.json"
    ).read_bytes()


def test_gallery_requires_candidate_parameters_to_match_generated_scenario(
    tmp_path: Path,
) -> None:
    manifest = _source_manifest(tmp_path)
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["candidates"][0]["candidate"]["start"]["x"] = 2.0
    manifest.write_text(json.dumps(payload) + "\n", encoding="utf-8")

    result = replay_gallery.build_replay_gallery(
        manifest, tmp_path / "gallery", render=False, video=False
    )

    assert result["summary"]["selected_case_count"] == 0
    assert result["summary"]["dispositions"]["source_candidate_identity_mismatch"] == 1


def test_gallery_rejects_candidate_fields_the_objective_adapter_cannot_reconstruct(
    tmp_path: Path,
) -> None:
    manifest = _source_manifest(tmp_path)
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    row = payload["candidates"][0]
    row["candidate"]["pedestrian_acceleration_mps2"] = "unknown"
    scenario_path = Path(row["scenario_yaml_path"])
    scenario_payload = yaml.safe_load(scenario_path.read_text(encoding="utf-8"))
    scenario_payload["scenarios"][0]["metadata"]["adversarial_candidate"] = row["candidate"]
    scenario_path.write_text(yaml.safe_dump(scenario_payload, sort_keys=False), encoding="utf-8")
    manifest.write_text(json.dumps(payload) + "\n", encoding="utf-8")

    result = replay_gallery.build_replay_gallery(
        manifest, tmp_path / "gallery", render=False, video=False
    )

    assert result["summary"]["selected_case_count"] == 0
    assert result["summary"]["dispositions"]["candidate_parameters_invalid"] == 1


def test_candidate_objective_adapter_preserves_supported_optional_dimensions() -> None:
    candidate = {
        **_candidate(),
        "pedestrian_acceleration_mps2": 0.2,
        "group_size": 3,
        "vru_profile": "adult_walking",
    }

    adapted = replay_gallery._candidate_spec(candidate)

    assert adapted.pedestrian_acceleration_mps2 == 0.2
    assert adapted.group_size == 3
    assert adapted.vru_profile == "adult_walking"


def test_gallery_requires_declared_effective_scenario_hash_to_match_inputs(
    tmp_path: Path,
) -> None:
    manifest = _source_manifest(tmp_path)
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["candidates"][0]["effective_scenario_hash"] = "0" * 64
    manifest.write_text(json.dumps(payload) + "\n", encoding="utf-8")

    result = replay_gallery.build_replay_gallery(
        manifest, tmp_path / "gallery", render=False, video=False
    )

    assert result["summary"]["selected_case_count"] == 0
    assert result["summary"]["dispositions"]["effective_scenario_hash_missing_or_mismatch"] == 1


def test_gallery_does_not_present_successful_episodes_as_falsification_cases(
    tmp_path: Path,
) -> None:
    manifest = _source_manifest(tmp_path)
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    row = payload["candidates"][0]
    row["failure_attribution"] = {
        "status": "attributed",
        "primary_failure": "success",
        "reasons": ["episode completed without an attributed failure"],
        "details": {
            "termination_reason": "success",
            "execution_mode": "native",
            "readiness_status": "native",
            "availability_status": "available",
        },
    }
    episode_path = Path(row["episode_record_path"])
    record = _episode()
    record["status"] = "success"
    record["termination_reason"] = "success"
    record["outcome"].update(route_complete=True, collision_event=False)
    record["metrics"].update(success=True, collisions=0)
    episode_path.write_text(json.dumps(record) + "\n", encoding="utf-8")
    manifest.write_text(json.dumps(payload) + "\n", encoding="utf-8")

    result = replay_gallery.build_replay_gallery(
        manifest, tmp_path / "gallery", render=False, video=False
    )

    assert result["summary"]["selected_case_count"] == 0
    assert result["summary"]["dispositions"]["source_episode_not_a_failure"] == 1


def test_gallery_accounts_invalid_and_failed_candidates_without_selecting_them(
    tmp_path: Path,
) -> None:
    manifest = _source_manifest(tmp_path)
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    invalid = dict(payload["candidates"][0])
    invalid["certification_status"] = {"details": {"certificates": [{"classification": "invalid"}]}}
    failed = dict(payload["candidates"][0])
    failed["error"] = "simulator initialization failed"
    payload["candidates"] = [invalid, failed]
    manifest.write_text(json.dumps(payload) + "\n", encoding="utf-8")

    result = replay_gallery.build_replay_gallery(
        manifest, tmp_path / "gallery", render=False, video=False
    )

    assert result["summary"]["selected_case_count"] == 0
    assert result["summary"]["dispositions"] == {
        "certificate_invalid": 1,
        "evaluation_failed": 1,
    }
    assert result["cases"] == []


@pytest.mark.parametrize(
    ("certificate", "expected"),
    [
        ({"classification": "VALID"}, "admissible_by_source_certificate"),
        ({"classification": "geometrically_infeasible"}, "invalid_or_infeasible"),
        ({"classification": "knife_edge"}, "stress_only"),
        ({}, "unknown"),
        (
            {
                "details": {
                    "certificates": [
                        {"classification": "valid"},
                        {"classification": "invalid"},
                    ]
                }
            },
            "unknown",
        ),
    ],
)
def test_feasibility_verdict_preserves_certificate_strength(
    certificate: dict[str, Any], expected: str
) -> None:
    verdict = replay_gallery._feasibility_verdict(certificate)

    assert verdict["status"] == expected
    assert verdict["source_certificate"] == certificate
    if expected == "unknown":
        assert verdict["reason"] == "certificate classification does not establish feasibility"


def test_certificate_classification_ignores_malformed_entries_without_inventing_a_verdict() -> None:
    assert replay_gallery._certification_classification(None) is None
    assert (
        replay_gallery._certification_classification(
            {"details": {"certificates": [{"classification": "valid"}, None]}}
        )
        == "valid"
    )
    assert (
        replay_gallery._certification_classification({"details": {"certificates": "malformed"}})
        is None
    )


def test_gallery_artifact_resolution_uses_source_bundle_before_checkout(
    tmp_path: Path,
) -> None:
    source_root = tmp_path / "source"
    current_root = tmp_path / "current"
    source_dir = source_root / "run"
    source_dir.mkdir(parents=True)
    current_root.mkdir()
    source_artifact = source_root / "run" / "artifact.json"
    checkout_artifact = current_root / "run" / "artifact.json"
    source_artifact.write_text("source", encoding="utf-8")
    checkout_artifact.parent.mkdir()
    checkout_artifact.write_text("checkout", encoding="utf-8")
    manifest = source_root / "search_manifest.json"

    assert (
        replay_gallery._resolve_artifact_path(
            "run/artifact.json",
            source_manifest=manifest,
            root=current_root,
            source_root=source_root,
        )
        == source_artifact.resolve()
    )
    assert (
        replay_gallery._resolve_artifact_path(
            " ", source_manifest=manifest, root=current_root, source_root=source_root
        )
        is None
    )
    assert (
        replay_gallery._resolve_referenced_file(
            "artifact.json", source_dir, source_root, current_root
        )
        == source_artifact.resolve()
    )
    assert (
        replay_gallery._resolve_referenced_file(
            "missing.json", source_dir, source_root, current_root
        )
        is None
    )
