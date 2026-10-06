"""Focused contracts for adversarial replay-gallery selection and verification."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
import yaml

from robot_sf.adversarial import replay_gallery
from robot_sf.benchmark.episode_replay_figure import (
    EpisodeRow,
    _pedestrian_trajectories,
    build_replay_from_episode_row,
)
from robot_sf.benchmark.fallback_policy import availability_payload
from robot_sf.scenario_certification import v1 as scenario_certification_v1


def _candidate() -> dict[str, Any]:
    return {
        "start": {"x": 1.0, "y": 1.0, "theta": 0.0},
        "goal": {"x": 3.0, "y": 1.0, "theta": 0.0},
        "spawn_time_s": 0.0,
        "pedestrian_speed_mps": 1.0,
        "pedestrian_delay_s": 0.0,
        "scenario_seed": 7,
    }


def _scenario_certificate(
    *, classification: str = "valid", scenario_id: str = "test_scenario"
) -> dict[str, Any]:
    """Build a complete canonical certificate fixture for one candidate scenario."""
    eligibility = {
        "valid": "eligible",
        "hard_but_solvable": "eligible",
        "knife_edge": "stress_only",
        "invalid": "excluded",
        "geometrically_infeasible": "excluded",
        "kinodynamically_infeasible": "excluded",
        "dynamically_overconstrained": "excluded",
    }[classification]
    route = {
        "route_id": "test-route",
        "spawn_id": 0,
        "goal_id": 0,
        "classification": classification,
        "benchmark_eligibility": eligibility,
        "reasons": [],
        "checks": {},
        "evidence": {},
    }
    return {
        "schema_version": "scenario_cert.v1",
        "scenario_id": scenario_id,
        "source": "unit-test fixture",
        "classification": classification,
        "benchmark_eligibility": eligibility,
        "reasons": [],
        "checks": {
            "route_count": 1,
            "all_routes_benchmark_eligible": eligibility == "eligible",
        },
        "route_certificates": [route],
        "evidence": {},
    }


def _certification_status(
    certificate: dict[str, Any], *, status: str = "passed", schema_version: str = "scenario_cert.v1"
) -> dict[str, Any]:
    """Wrap one scenario certificate in the adversarial certification receipt."""
    return {
        "schema_version": schema_version,
        "status": status,
        "reason": "unit-test fixture",
        "details": {"certificates": [certificate]},
    }


def _bind_certificate_to_scenario(status: dict[str, Any], scenario_path: Path) -> dict[str, Any]:
    """Mirror canonical source and loader-mapping evidence in a file-backed receipt."""
    certificates = status["details"]["certificates"]
    certificate = certificates[0]
    loaded = replay_gallery.scenario_loader.load_scenarios(scenario_path)
    certificate["source"] = str(scenario_path)
    certificate["evidence"]["scenario_fingerprint"] = replay_gallery._fingerprint_mapping(
        loaded[0],
        scenario_path=scenario_path,
        source_root=replay_gallery._repository_root(),
    )
    if loaded[0].get("map_id") is not None:
        map_binding, map_binding_error = scenario_certification_v1._capture_map_id_input_binding(
            loaded[0],
            scenario_path=scenario_path,
            source_root=replay_gallery._repository_root(),
        )
        assert map_binding_error is None and map_binding is not None
        certificate["evidence"]["map_id_input_binding"] = map_binding
    return status


@pytest.fixture(autouse=True)
def _isolated_map_registry_cache() -> Iterator[None]:
    """Keep environment-selected registry entries within their owning test."""
    registry_loader = replay_gallery.scenario_loader._load_map_registry
    registry_loader.cache_clear()
    try:
        yield
    finally:
        # Monkeypatch restores the environment, but the loader caches its
        # no-argument lookup independently of ROBOT_SF_MAP_REGISTRY.
        registry_loader.cache_clear()


@pytest.fixture(autouse=True)
def _temporary_checkout_root(tmp_path: Path, monkeypatch: Any) -> None:
    """Keep output-boundary tests and all generated bundles inside pytest's temp tree."""
    monkeypatch.setattr(replay_gallery, "_repository_root", lambda: tmp_path)
    monkeypatch.setattr(
        replay_gallery,
        "_git_checkout_state",
        lambda _root: {"revision": "a" * 40, "clean": True, "dirty_paths": []},
    )


def _episode(
    *,
    revision: str | None = "a" * 40,
    config_hash: str = "test-algo-config",
    execution_mode: str = "native",
) -> dict[str, Any]:
    record: dict[str, Any] = {
        "episode_id": "test_episode",
        "scenario_id": "test_scenario",
        "seed": 7,
        "algo": "goal",
        "algorithm_metadata": {
            "status": "ok",
            "planner_kinematics": {"execution_mode": execution_mode},
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
    execution_mode: str = "native",
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
    certification_status = _bind_certificate_to_scenario(
        _certification_status(_scenario_certificate()), scenario_path
    )
    episode_path = bundle / "episode_records.jsonl"
    episode_path.write_text(
        json.dumps(_episode(revision=source_revision, execution_mode=execution_mode)) + "\n",
        encoding="utf-8",
    )
    source_readiness = "native" if execution_mode == "native" else "adapter"
    analysis_eligible = execution_mode == "native"
    reason_codes = [] if analysis_eligible else ["execution_mode_not_native"]
    effective_scenario_hash = replay_gallery.compute_effective_scenario_hash(scenario, {})
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
                    "schema_version": "search_analysis_eligibility.v1",
                    "eligible": analysis_eligible,
                    "certificate_ok": True,
                    "reason_codes": reason_codes,
                    "trace_present": True,
                    "execution_mode": execution_mode,
                    "objective_scored": True,
                    "effective_scenario_hash": effective_scenario_hash,
                },
                "certification_status": certification_status,
                "failure_attribution": {
                    "status": "attributed",
                    "primary_failure": "collision",
                    "reasons": ["source episode records collision"],
                    "details": {
                        "termination_reason": "collision",
                        "execution_mode": execution_mode,
                        "readiness_status": source_readiness,
                        "availability_status": "available",
                    },
                },
                "effective_scenario_hash": effective_scenario_hash,
                "scenario_yaml_path": str(scenario_path),
                "episode_record_path": str(episode_path),
                "bundle_path": str(bundle),
            }
        ],
        "summary": {"num_candidates": 1},
    }
    if manifest_revision is not None:
        payload["source_revision"] = manifest_revision
    manifest_path = tmp_path / "search_manifest.json"
    manifest_path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")
    return manifest_path


def _bind_effective_scenario_hash(
    candidate_row: dict[str, Any],
    scenario: dict[str, Any],
    route_overrides: dict[str, Any] | None = None,
) -> str:
    effective_hash = replay_gallery.compute_effective_scenario_hash(scenario, route_overrides or {})
    candidate_row["effective_scenario_hash"] = effective_hash
    candidate_row["analysis_eligibility"]["effective_scenario_hash"] = effective_hash
    return effective_hash


def _replay_episode(
    *,
    revision: str | None = "a" * 40,
    config_hash: str = "test-algo-config",
    execution_mode: str = "native",
) -> dict[str, Any]:
    record = _episode(revision=revision, config_hash=config_hash, execution_mode=execution_mode)
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
    *,
    preflight_status: str = "ok",
    failures: list[dict[str, str]] | None = None,
    execution_mode: str = "native",
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
            "planner_kinematics": {"execution_mode": execution_mode},
        },
    }
    summary["benchmark_availability"] = availability_payload(summary)
    return summary


def _install_fake_replay(  # noqa: PLR0913 - fixture knobs keep synthetic replay cases explicit
    monkeypatch: Any,
    *,
    revision: str | None = "a" * 40,
    replay_config_hash: str = "test-algo-config",
    mismatch: bool = False,
    runner_summary: dict[str, Any] | None = None,
    checkout_clean: bool = True,
    checkout_revision: str | None = None,
    rendered_map_paths: list[Path | None] | None = None,
    vary_runtime_receipts: bool = False,
    replay_execution_mode: str = "native",
) -> list[dict[str, Any]]:
    calls: list[dict[str, Any]] = []

    def fake_run_batch(scenario_path: Path, **kwargs: Any) -> dict[str, Any]:
        calls.append(
            {
                "scenario_path": scenario_path,
                "map_registry_env": os.environ.get("ROBOT_SF_MAP_REGISTRY"),
                **kwargs,
            }
        )
        scenarios = yaml.safe_load(scenario_path.read_text(encoding="utf-8"))["scenarios"]
        record = _replay_episode(
            revision=revision,
            config_hash=replay_config_hash,
            execution_mode=replay_execution_mode,
        )
        record["scenario_id"] = scenarios[0]["name"]
        if vary_runtime_receipts:
            run_number = len(calls)
            record["provenance"] = {
                "run_id": f"episode-run-{run_number}",
                "runtime_sec": float(run_number),
                "invocation": f"runner --nonce {run_number}",
            }
        if mismatch:
            record["outcome"]["collision_event"] = False
            record["metrics"]["collisions"] = 0
        kwargs["out_path"].write_text(json.dumps(record) + "\n", encoding="utf-8")
        summary = runner_summary or _runner_summary(execution_mode=replay_execution_mode)
        if vary_runtime_receipts:
            run_number = len(calls)
            summary = dict(summary)
            summary["out_path"] = str(kwargs["out_path"])
            summary["batch_runtime_sec"] = float(run_number)
            summary["provenance"] = {
                "run_id": f"runner-run-{run_number}",
                "invocation": f"runner --out {kwargs['out_path']} --nonce {run_number}",
                "result_manifest_path": f"{kwargs['out_path']}.manifest.json",
                "config_identity": {
                    "scenario_path": str(scenario_path),
                    "schema_path": str(kwargs["schema_path"]),
                    "scenario_matrix_hash": f"matrix-{run_number}",
                },
            }
            summary["benchmark_availability"] = availability_payload(summary)
        return summary

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

    output_dir = tmp_path / "output" / "gallery"
    result = replay_gallery.build_replay_gallery(manifest, output_dir, video=False)

    assert result["summary"]["selected_case_count"] == 1
    assert result["summary"]["replay_match_count"] == 1
    assert result["source"]["search_method"] == "random"
    assert result["source"]["candidate_inventory"] == {
        "status": "complete_search_run",
        "complete": True,
        "represented_candidate_count": 1,
        "declared_search_budget": 1,
        "producer_candidate_count": 1,
    }
    assert result["selection"]["mechanism_cluster_deduplication"]["status"] == "available"
    case = result["cases"][0]
    assert case["replay_match"] == "match"
    assert case["verification_status"] == "outcome_reproduced_source_inputs_unbound"
    assert case["source_input_binding"]["status"] == "unknown"
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
    assert "accounts for every candidate" in (output_dir / "README.md").read_text(encoding="utf-8")
    assert (output_dir / "gallery_manifest.json").is_file()


def test_gallery_rejects_a_truncated_direct_search_manifest(tmp_path: Path) -> None:
    """A partial candidate list cannot be presented as a completed search run."""
    manifest = _source_manifest(tmp_path)
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["config"]["budget"] = 2
    manifest.write_text(json.dumps(payload) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="incomplete.*config.budget"):
        replay_gallery.build_replay_gallery(
            manifest, tmp_path / "output" / "gallery", render=False, video=False
        )


def test_gallery_rejects_candidate_list_disagreeing_with_producer_summary(
    tmp_path: Path,
) -> None:
    """The embedded producer count must match the rows read from its manifest."""
    manifest = _source_manifest(tmp_path)
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["summary"]["num_candidates"] = 2
    manifest.write_text(json.dumps(payload) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="list length does not match summary.num_candidates"):
        replay_gallery.build_replay_gallery(
            manifest, tmp_path / "output" / "gallery", render=False, video=False
        )


def test_gallery_rejects_unlabeled_manifest_without_candidate_summary(tmp_path: Path) -> None:
    """Missing completion accounting is allowed only for an explicit historical subset."""
    manifest = _source_manifest(tmp_path)
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload.pop("summary")
    manifest.write_text(json.dumps(payload) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="requires summary.num_candidates"):
        replay_gallery.build_replay_gallery(
            manifest, tmp_path / "output" / "gallery", render=False, video=False
        )


def test_historical_compatibility_subset_is_explicitly_partial_in_gallery_readme(
    tmp_path: Path,
) -> None:
    """The #1501 compatibility fixture cannot imply top-K over its original run."""
    repo_root = Path(__file__).resolve().parents[2]
    manifest_path = (
        repo_root / "tests/fixtures/adversarial_replay_gallery/issue_1501_compat/manifest.json"
    )
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    inventory = replay_gallery._candidate_inventory_status(
        payload=payload,
        config=payload["config"],
        candidate_count=len(payload["candidates"]),
    )
    output = tmp_path / "README.md"
    replay_gallery._write_gallery_readme(
        tmp_path,
        [],
        {
            "selected_case_count": 0,
            "replay_match_count": 0,
            "replay_mismatch_count": 0,
            "replay_unavailable_count": 0,
        },
        candidate_inventory=inventory,
    )

    readme = output.read_text(encoding="utf-8")
    assert inventory["status"] == "historical_candidate_subset"
    assert inventory["complete"] is False
    assert "not a complete" in readme
    assert "1 of 32 declared slots" in readme


def test_tracked_compatibility_fixture_without_registry_binding_remains_unknown() -> None:
    """A legacy map-id certificate cannot claim static admissibility without registry provenance."""
    repo_root = Path(__file__).resolve().parents[2]
    fixture_root = repo_root / "tests/fixtures/adversarial_replay_gallery/issue_1501_compat"
    scenario_path = fixture_root / "scenario.yaml"
    manifest = json.loads((fixture_root / "manifest.json").read_text(encoding="utf-8"))
    candidate = manifest["candidates"][0]
    certificate_status = candidate["certification_status"]
    loaded_scenario = replay_gallery.scenario_loader.load_scenarios(scenario_path)[0]
    current_map_binding, map_binding_error = (
        scenario_certification_v1._capture_map_id_input_binding(
            loaded_scenario, scenario_path=scenario_path, source_root=repo_root
        )
    )
    assert map_binding_error is None and current_map_binding is not None
    certificate, error = replay_gallery._validated_scenario_certificate(
        certificate_status,
        expected_scenario_id="crossing_ttc_template_adversarial_0008",
        loaded_scenario=loaded_scenario,
        scenario_path=scenario_path,
        root=repo_root,
        source_root=repo_root,
        map_id_input_binding=current_map_binding,
    )
    provenance = json.loads(
        (fixture_root / "scenario_certification_provenance.json").read_text(encoding="utf-8")
    )

    assert certificate is None
    assert error == "certificate_map_id_input_binding_unknown"
    raw_certificate = certificate_status["details"]["certificates"][0]
    assert raw_certificate["classification"] == "hard_but_solvable"
    assert raw_certificate["benchmark_eligibility"] == "eligible"
    assert provenance["certificate"]["scenario_id"] == raw_certificate["scenario_id"]
    assert (
        provenance["certificate"]["sha256_canonical_json"]
        == hashlib.sha256(
            json.dumps(raw_certificate, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
    )
    assert provenance["claim_boundary"]["dynamic_task_feasibility"] == "unknown"
    assert "post-hoc" in provenance["claim_boundary"]["note"]
    historical_certifier_path = "robot_sf/scenario_certification/v1.py"
    for relative_path, expected_digest in provenance["input_sha256"].items():
        if relative_path == historical_certifier_path:
            assert (
                expected_digest
                != hashlib.sha256((repo_root / relative_path).read_bytes()).hexdigest()
            )
            continue
        assert (
            hashlib.sha256((repo_root / relative_path).read_bytes()).hexdigest() == expected_digest
        )

    historical_manifest = json.loads(
        (fixture_root / "manifest.pre_strict_gate.json").read_text(encoding="utf-8")
    )
    historical_status = historical_manifest["candidates"][0]["certification_status"]
    historical_certificate, historical_error = replay_gallery._validated_scenario_certificate(
        historical_status,
        expected_scenario_id="crossing_ttc_template_adversarial_0008",
        loaded_scenario=loaded_scenario,
        scenario_path=scenario_path,
        root=repo_root,
        source_root=repo_root,
        map_id_input_binding=current_map_binding,
    )
    assert historical_certificate is None
    assert historical_error == "certificate_structurally_invalid"


def test_explicit_unknown_map_id_binding_remains_unknown() -> None:
    """An explicit producer-side unknown binding cannot certify a map-id scenario."""
    repo_root = Path(__file__).resolve().parents[2]
    fixture_root = repo_root / "tests/fixtures/adversarial_replay_gallery/issue_1501_compat"
    scenario_path = fixture_root / "scenario.yaml"
    manifest = json.loads((fixture_root / "manifest.json").read_text(encoding="utf-8"))
    certificate_status = manifest["candidates"][0]["certification_status"]
    loaded_scenario = replay_gallery.scenario_loader.load_scenarios(scenario_path)[0]
    current_binding, binding_error = scenario_certification_v1._capture_map_id_input_binding(
        loaded_scenario, scenario_path=scenario_path, source_root=repo_root
    )
    assert binding_error is None and current_binding is not None
    certificate_payload = certificate_status["details"]["certificates"][0]
    certificate_payload["evidence"]["map_id_input_binding"] = {
        "status": "unknown",
        "reason": "test_binding_unavailable",
    }

    certificate, error = replay_gallery._validated_scenario_certificate(
        certificate_status,
        expected_scenario_id="crossing_ttc_template_adversarial_0008",
        loaded_scenario=loaded_scenario,
        scenario_path=scenario_path,
        root=repo_root,
        source_root=repo_root,
        map_id_input_binding=current_binding,
    )

    assert certificate is None
    assert error == "certificate_map_id_input_binding_unknown"


def test_canonical_certifier_records_portable_map_id_registry_and_map_digests() -> None:
    """New map-id certificates bind the exact registry and selected map inputs."""
    repo_root = Path(__file__).resolve().parents[2]
    fixture_root = repo_root / "tests/fixtures/adversarial_replay_gallery/issue_1501_compat"
    scenario_path = fixture_root / "scenario.yaml"
    replay_gallery.scenario_loader._load_map_registry.cache_clear()
    try:
        certificate = scenario_certification_v1.certify_scenario_file(scenario_path)[0]
    finally:
        replay_gallery.scenario_loader._load_map_registry.cache_clear()

    binding = certificate.evidence["map_id_input_binding"]
    registry_path = replay_gallery.scenario_loader._resolve_map_registry_path()
    assert registry_path is not None
    loaded = replay_gallery.scenario_loader.load_scenarios(scenario_path)[0]
    map_path = (scenario_path.parent / loaded["map_file"]).resolve()
    assert binding["schema_version"] == "scenario_map_id_input_binding.v1"
    assert binding["map_id"] == "classic_cross_trap"
    assert binding["registry_locator"] == "repo:maps/registry.yaml"
    registry_payload = yaml.safe_load(registry_path.read_text(encoding="utf-8"))
    selected_row = next(
        row for row in registry_payload["maps"] if row.get("map_id") == "classic_cross_trap"
    )
    assert binding["selected_registry_entry"] == dict(selected_row)
    assert (
        binding["selected_registry_entry_sha256"]
        == hashlib.sha256(
            json.dumps(dict(selected_row), sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
    )
    assert binding["resolved_map_locator"] == "repo:maps/svg_maps/classic_crossing.svg"
    assert binding["resolved_map_sha256"] == replay_gallery._sha256_file(map_path)


def test_source_bound_certificate_is_portable_across_checkout_roots_and_binds_referenced_bytes(
    tmp_path: Path, monkeypatch: Any
) -> None:
    """Path relocation preserves a certificate while changing referenced bytes invalidates it."""
    repo_root = Path(__file__).resolve().parents[2]
    fixture_root = repo_root / "tests/fixtures/adversarial_replay_gallery/issue_1501_compat"
    manifest = json.loads((fixture_root / "manifest.json").read_text(encoding="utf-8"))
    certificate_status = manifest["candidates"][0]["certification_status"]

    other_root = tmp_path / "different-checkout-root"
    other_fixture_root = other_root / "tests/fixtures/adversarial_replay_gallery/issue_1501_compat"
    other_fixture_root.parent.mkdir(parents=True)
    shutil.copytree(fixture_root, other_fixture_root)
    other_maps = other_root / "maps"
    (other_maps / "svg_maps").mkdir(parents=True)
    shutil.copy2(repo_root / "maps/registry.yaml", other_maps / "registry.yaml")
    shutil.copy2(
        repo_root / "maps/svg_maps/classic_crossing.svg",
        other_maps / "svg_maps/classic_crossing.svg",
    )
    monkeypatch.setenv("ROBOT_SF_MAP_REGISTRY", str(other_maps / "registry.yaml"))

    other_scenario_path = other_fixture_root / "scenario.yaml"
    replay_gallery.scenario_loader._load_map_registry.cache_clear()
    try:
        loaded = replay_gallery.scenario_loader.load_scenarios(other_scenario_path)[0]
        copied_certificate_status = json.loads(json.dumps(certificate_status))
        copied_certificate = copied_certificate_status["details"]["certificates"][0]
        copied_map_binding, copied_map_binding_error = (
            scenario_certification_v1._capture_map_id_input_binding(
                loaded, scenario_path=other_scenario_path, source_root=other_root
            )
        )
        assert copied_map_binding_error is None and copied_map_binding is not None
        copied_certificate["evidence"]["map_id_input_binding"] = copied_map_binding
        certificate, error = replay_gallery._validated_scenario_certificate(
            copied_certificate_status,
            expected_scenario_id="crossing_ttc_template_adversarial_0008",
            loaded_scenario=dict(loaded),
            scenario_path=other_scenario_path,
            root=other_root,
            source_root=other_root,
            map_id_input_binding=copied_map_binding,
        )
        assert error is None
        assert certificate is not None

        route_overrides = other_fixture_root / "route_overrides.yaml"
        route_overrides.write_bytes(route_overrides.read_bytes() + b"\n# content changed\n")
        changed_loaded = replay_gallery.scenario_loader.load_scenarios(other_scenario_path)[0]
        changed_certificate, changed_error = replay_gallery._validated_scenario_certificate(
            copied_certificate_status,
            expected_scenario_id="crossing_ttc_template_adversarial_0008",
            loaded_scenario=dict(changed_loaded),
            scenario_path=other_scenario_path,
            root=other_root,
            source_root=other_root,
            map_id_input_binding=copied_map_binding,
        )
    finally:
        replay_gallery.scenario_loader._load_map_registry.cache_clear()

    assert changed_certificate is None
    assert changed_error == "certificate_scenario_fingerprint_mismatch"


@pytest.mark.parametrize(
    "change",
    ["unrelated_registry_row", "selected_registry_metadata", "registry_target", "map_bytes"],
)
def test_map_id_certificate_rejects_changed_registry_or_resolved_map_inputs(  # noqa: PLR0915 - explicit drift cases
    tmp_path: Path, monkeypatch: Any, change: str
) -> None:
    """Unchanged scenario YAML cannot retain a certificate after map-id inputs drift."""
    repo_root = Path(__file__).resolve().parents[2]
    fixture_root = repo_root / "tests/fixtures/adversarial_replay_gallery/issue_1501_compat"
    input_root = tmp_path / "map-id-binding"
    case_root = input_root / "case"
    shutil.copytree(fixture_root, case_root)
    scenario_path = case_root / "scenario.yaml"
    original_scenario_bytes = scenario_path.read_bytes()

    source_registry = replay_gallery.scenario_loader._resolve_map_registry_path()
    assert source_registry is not None
    registry_payload = yaml.safe_load(source_registry.read_text(encoding="utf-8"))
    selected_row = next(
        row for row in registry_payload["maps"] if row.get("map_id") == "classic_cross_trap"
    )
    source_map_ref = Path(selected_row["path"])
    source_map = (
        source_map_ref if source_map_ref.is_absolute() else source_registry.parent / source_map_ref
    )
    registry_path = input_root / "maps" / "registry.yaml"
    map_target = registry_path.parent / source_map_ref
    map_target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_map, map_target)

    # Removing this optional catalog attestation lets the map-bytes mutation
    # exercise the certificate binding instead of the registry's own stale-hash gate.
    selected_row.pop("source_sha256", None)
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    registry_path.write_text(yaml.safe_dump(registry_payload, sort_keys=False), encoding="utf-8")
    monkeypatch.setenv("ROBOT_SF_MAP_REGISTRY", str(registry_path))
    replay_gallery.scenario_loader._load_map_registry.cache_clear()
    try:
        loaded = dict(replay_gallery.scenario_loader.load_scenarios(scenario_path)[0])
        status = _bind_certificate_to_scenario(
            _certification_status(
                _scenario_certificate(
                    classification="hard_but_solvable",
                    scenario_id="crossing_ttc_template_adversarial_0008",
                )
            ),
            scenario_path,
        )
        scenario_identity = yaml.safe_load(original_scenario_bytes)["scenarios"][0]
        original_snapshot, original_snapshot_error = replay_gallery._snapshot_map_id_input(
            scenario_identity,
            scenario_path=scenario_path,
            episode_path=case_root / "unused-episode.jsonl",
            source_record={},
            root=tmp_path,
            source_root=tmp_path,
        )
        assert original_snapshot_error is None and original_snapshot is not None
        original_binding = status["details"]["certificates"][0]["evidence"]["map_id_input_binding"]
        certificate, error = replay_gallery._validated_scenario_certificate(
            status,
            expected_scenario_id="crossing_ttc_template_adversarial_0008",
            loaded_scenario=loaded,
            scenario_path=scenario_path,
            root=tmp_path,
            source_root=tmp_path,
            map_id_input_binding=original_binding,
        )
        assert error is None and certificate is not None

        if change == "unrelated_registry_row":
            changed_registry = yaml.safe_load(registry_path.read_text(encoding="utf-8"))
            unrelated_row = next(
                row for row in changed_registry["maps"] if row.get("map_id") != "classic_cross_trap"
            )
            unrelated_row["limitations"] = ["unrelated registry row changed"]
            registry_path.write_text(
                yaml.safe_dump(changed_registry, sort_keys=False), encoding="utf-8"
            )
        elif change == "selected_registry_metadata":
            changed_registry = yaml.safe_load(registry_path.read_text(encoding="utf-8"))
            changed_row = next(
                row for row in changed_registry["maps"] if row.get("map_id") == "classic_cross_trap"
            )
            existing_limitations = changed_row.get("limitations")
            changed_row["limitations"] = [
                *(existing_limitations if isinstance(existing_limitations, list) else []),
                "certificate-binding metadata drift",
            ]
            registry_path.write_text(
                yaml.safe_dump(changed_registry, sort_keys=False), encoding="utf-8"
            )
        elif change == "registry_target":
            alternate_ref = source_map_ref.with_name("certificate-binding-copy.svg")
            alternate_target = registry_path.parent / alternate_ref
            alternate_target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(map_target, alternate_target)
            changed_registry = yaml.safe_load(registry_path.read_text(encoding="utf-8"))
            changed_row = next(
                row for row in changed_registry["maps"] if row.get("map_id") == "classic_cross_trap"
            )
            changed_row["path"] = alternate_ref.as_posix()
            registry_path.write_text(
                yaml.safe_dump(changed_registry, sort_keys=False), encoding="utf-8"
            )
        else:
            map_target.write_bytes(map_target.read_bytes() + b"\n<!-- map bytes changed -->\n")

        assert scenario_path.read_bytes() == original_scenario_bytes
        current_snapshot, snapshot_error = replay_gallery._snapshot_map_id_input(
            scenario_identity,
            scenario_path=scenario_path,
            episode_path=case_root / "unused-episode.jsonl",
            source_record={},
            root=tmp_path,
            source_root=tmp_path,
        )
        assert snapshot_error is None and current_snapshot is not None
        current_binding = replay_gallery._map_id_input_binding_from_paths(
            map_id=current_snapshot["map_id"],
            required_profile=current_snapshot["required_profile"],
            registry_path=current_snapshot["registry_path"],
            registry_row=current_snapshot["registry_row"],
            map_path=current_snapshot["map_path"],
            map_sha256=current_snapshot["map_sha256"],
            scenario_path=scenario_path,
            source_root=tmp_path,
        )
        changed_certificate, changed_error = replay_gallery._validated_scenario_certificate(
            status,
            expected_scenario_id="crossing_ttc_template_adversarial_0008",
            loaded_scenario=loaded,
            scenario_path=scenario_path,
            root=tmp_path,
            source_root=tmp_path,
            map_id_input_binding=current_binding,
        )
    finally:
        replay_gallery.scenario_loader._load_map_registry.cache_clear()

    if change == "unrelated_registry_row":
        assert changed_certificate is not None
        assert changed_error is None
        assert current_binding == original_binding
        assert current_snapshot["registry_sha256"] != original_snapshot["registry_sha256"]
    else:
        assert changed_certificate is None
        assert changed_error == "certificate_map_id_input_binding_mismatch"
        if change == "selected_registry_metadata":
            assert current_snapshot["map_id"] == original_snapshot["map_id"]
            assert current_snapshot["map_path"] == original_snapshot["map_path"]
            assert current_snapshot["map_sha256"] == original_snapshot["map_sha256"]
            assert current_snapshot["registry_row"] != original_snapshot["registry_row"]
            assert current_binding != original_binding


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
        manifest, tmp_path / "output" / "gallery", render=False, video=False
    )

    assert result["summary"]["selected_case_count"] == 0
    assert result["summary"]["dispositions"][expected_disposition] == 1


@pytest.mark.parametrize("execution_mode", ["native", "mixed"])
def test_gallery_excludes_source_runtime_fallback_even_if_manifest_says_available(
    tmp_path: Path,
    execution_mode: str,
) -> None:
    manifest = _source_manifest(tmp_path, execution_mode=execution_mode)
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    episode_path = Path(payload["candidates"][0]["episode_record_path"])
    episode = _episode(execution_mode=execution_mode)
    episode["algorithm_metadata"]["planner_runtime"] = {"fallback_used": True}
    episode_path.write_text(json.dumps(episode) + "\n", encoding="utf-8")

    result = replay_gallery.build_replay_gallery(
        manifest, tmp_path / "output" / "gallery", render=False, video=False
    )

    assert result["summary"]["selected_case_count"] == 0
    assert result["summary"]["dispositions"]["source_runtime_fallback_or_degraded"] == 1


@pytest.mark.parametrize("decision_label", ["ppo_clear", "ppo_safe"])
def test_gallery_accepts_normal_guard_decision_with_fallback_controller_state(
    decision_label: str,
) -> None:
    record = {
        "algorithm_metadata": {
            "status": "ok",
            "planner_runtime": {
                "last_decision": {
                    "schema_version": "shield-decision.v1",
                    "decision_label": decision_label,
                    "fallback_controller_state": {
                        "policy": "RiskDWAPlannerAdapter",
                        "prior_available": False,
                        "action_adaptation": {
                            "mode": "direct_policy_command",
                            "residual_clipped": False,
                        },
                    },
                }
            },
        }
    }

    assert replay_gallery._runtime_algorithm_fallback_marker(record) is None


def test_gallery_rejects_active_guard_fallback_decision() -> None:
    record = {
        "algorithm_metadata": {
            "status": "ok",
            "planner_runtime": {
                "last_decision": {
                    "schema_version": "shield-decision.v1",
                    "decision_label": "fallback_safe",
                    "fallback_controller_state": {"policy": "RiskDWAPlannerAdapter"},
                }
            },
        }
    }

    assert replay_gallery._runtime_algorithm_fallback_marker(record) == (
        "planner_runtime.last_decision.fallback_used",
        "true",
    )


@pytest.mark.parametrize("marker", [("fallback", True), ("degraded", True)])
def test_gallery_preserves_explicit_markers_inside_fallback_controller_state(
    marker: tuple[str, bool],
) -> None:
    record = {
        "algorithm_metadata": {
            "status": "ok",
            "planner_runtime": {
                "last_decision": {
                    "schema_version": "shield-decision.v1",
                    "decision_label": "ppo_safe",
                    "fallback_controller_state": {
                        "policy": "RiskDWAPlannerAdapter",
                        marker[0]: marker[1],
                    },
                }
            },
        }
    }

    assert replay_gallery._runtime_algorithm_fallback_marker(record) == (
        f"planner_runtime.last_decision.controller_state.{marker[0]}",
        "true",
    )


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
        manifest, tmp_path / "output" / "gallery", render=False, video=False
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
        manifest, tmp_path / "output" / "gallery", render=False, video=False
    )

    assert result["summary"]["selected_case_count"] == 0
    assert result["summary"]["dispositions"]["source_runtime_fallback_or_degraded"] == 1


def test_gallery_does_not_verify_replay_from_a_dirty_checkout(
    tmp_path: Path, monkeypatch: Any
) -> None:
    manifest = _source_manifest(tmp_path)
    _install_fake_replay(monkeypatch, checkout_clean=False)

    result = replay_gallery.build_replay_gallery(
        manifest, tmp_path / "output" / "gallery", video=False
    )

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

    result = replay_gallery.build_replay_gallery(
        manifest, tmp_path / "output" / "gallery", video=False
    )

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

    result = replay_gallery.build_replay_gallery(
        manifest, tmp_path / "output" / "gallery", video=False
    )

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

    result = replay_gallery.build_replay_gallery(
        manifest, tmp_path / "output" / "gallery", video=False
    )

    case = result["cases"][0]
    assert case["source"]["manifest_revision"] is None
    assert case["source"]["episode_revision"] == "a" * 40
    assert case["verification_status"] == "outcome_reproduced_source_inputs_unbound"


def test_gallery_uses_manifest_revision_when_episode_revision_is_missing(
    tmp_path: Path, monkeypatch: Any
) -> None:
    manifest = _source_manifest(tmp_path, source_revision=None, manifest_revision="a" * 40)
    _install_fake_replay(monkeypatch, revision="a" * 40)

    result = replay_gallery.build_replay_gallery(
        manifest, tmp_path / "output" / "gallery", video=False
    )

    case = result["cases"][0]
    assert case["source"]["episode_revision"] is None
    assert case["source"]["manifest_revision"] == "a" * 40
    assert case["verification_status"] == "outcome_reproduced_source_inputs_unbound"


def test_gallery_rejects_replay_with_different_planner_config_hash(
    tmp_path: Path, monkeypatch: Any
) -> None:
    manifest = _source_manifest(tmp_path)
    _install_fake_replay(monkeypatch, replay_config_hash="different-config")

    result = replay_gallery.build_replay_gallery(
        manifest, tmp_path / "output" / "gallery", video=False
    )

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
    candidate_row["certification_status"] = _bind_certificate_to_scenario(
        candidate_row["certification_status"], scenario_path
    )
    _bind_effective_scenario_hash(candidate_row, scenario)
    manifest.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")
    rendered_maps: list[Path | None] = []
    _install_fake_replay(monkeypatch, rendered_map_paths=rendered_maps)

    output_dir = tmp_path / "output" / "gallery"
    result = replay_gallery.build_replay_gallery(manifest, output_dir, video=False)

    case = result["cases"][0]
    rendered_map = rendered_maps[0]
    assert rendered_map is None
    assert case["replay_match"] == "match"
    assert case["verification_status"] == "outcome_reproduced_source_inputs_unbound"
    map_context = case["rendering"]["map_context"]
    assert map_context["status"] == "unavailable"
    assert map_context["reason"] == "existing_renderer_cannot_decode_map_overlay"
    assert "inputs/assets/" in map_context["path"]
    assert map_context["path"].endswith(map_path.name)
    assert map_context["sha256"] == hashlib.sha256(map_bytes).hexdigest()
    assert map_context["renderer_error"].startswith("UnidentifiedImageError:")


def test_renderer_map_context_marks_missing_image_reader_unavailable(
    tmp_path: Path, monkeypatch: Any
) -> None:
    map_path = tmp_path / "map.png"
    map_bytes = b"map bytes"
    map_path.write_bytes(map_bytes)
    monkeypatch.setattr("robot_sf.common.optional_import.try_import", lambda _name: None)

    result = replay_gallery._renderer_map_context(map_path)

    assert result == {
        "status": "unavailable",
        "reason": "existing_renderer_cannot_decode_map_overlay",
        "renderer_error": "ImportError: matplotlib.image is unavailable",
        "sha256": hashlib.sha256(map_bytes).hexdigest(),
    }


def test_gallery_materializes_map_id_registry_and_pins_runner_resolution(
    tmp_path: Path, monkeypatch: Any
) -> None:
    manifest = _source_manifest(tmp_path)
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    candidate_row = payload["candidates"][0]
    scenario_path = Path(candidate_row["scenario_yaml_path"])
    scenario_payload = yaml.safe_load(scenario_path.read_text(encoding="utf-8"))
    scenario = scenario_payload["scenarios"][0]
    scenario["map_id"] = "classic_cross_trap"
    scenario_path.write_text(yaml.safe_dump(scenario_payload, sort_keys=False), encoding="utf-8")
    candidate_row["certification_status"] = _bind_certificate_to_scenario(
        candidate_row["certification_status"], scenario_path
    )
    _bind_effective_scenario_hash(candidate_row, scenario)
    manifest.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")
    rendered_maps: list[Path | None] = []
    calls = _install_fake_replay(monkeypatch, rendered_map_paths=rendered_maps)

    output_dir = tmp_path / "output" / "gallery"
    result = replay_gallery.build_replay_gallery(manifest, output_dir, video=False)

    case = result["cases"][0]
    resolution = case["materialization"]["map_resolution"]
    case_dir = output_dir / "cases" / case["case_id"]
    map_asset = next(
        asset for asset in case["materialization"]["assets"] if asset["field"] == "resolved_map_id"
    )
    registry_asset = next(
        asset
        for asset in case["materialization"]["assets"]
        if asset["field"] == "map_registry_source"
    )
    assert resolution["status"] == "materialized"
    assert resolution["map_id"] == "classic_cross_trap"
    assert (
        replay_gallery._sha256_file(case_dir / map_asset["bundle_path"])
        == resolution["resolved_map_sha256"]
    )
    assert (
        replay_gallery._sha256_file(case_dir / registry_asset["bundle_path"])
        == resolution["source_registry_sha256"]
    )
    runner_registry = case_dir / resolution["runner_registry_bundle_path"]
    assert runner_registry.is_file()
    assert replay_gallery._sha256_file(runner_registry) == resolution["runner_registry_sha256"]
    assert Path(calls[0]["map_registry_env"]) == runner_registry
    assert case["replay"]["source_input_binding"]["status"] == "unknown"
    source_registry_check = next(
        check
        for check in case["replay"]["source_input_binding"]["checks"]
        if check.get("input") == "source_episode_map_registry"
    )
    assert source_registry_check["status"] == "unknown"
    assert source_registry_check["reason"] == "source_episode_map_registry_digest_missing"
    assert case["verification_status"] == "outcome_reproduced_source_inputs_unbound"
    assert rendered_maps == [None]
    assert case["rendering"]["map_context"]["status"] == "unavailable"
    assert case["rendering"]["map_context"]["reason"] == (
        "existing_renderer_cannot_decode_map_overlay"
    )

    second_output = tmp_path / "output" / "gallery-second"
    second_result = replay_gallery.build_replay_gallery(
        manifest, second_output, render=False, video=False
    )
    second_case = second_result["cases"][0]
    second_resolution = second_case["materialization"]["map_resolution"]
    second_registry = (
        second_output
        / "cases"
        / second_case["case_id"]
        / second_resolution["runner_registry_bundle_path"]
    )
    assert case["case_id"] == second_case["case_id"]
    assert resolution["runner_registry_sha256"] == second_resolution["runner_registry_sha256"]
    assert runner_registry.read_bytes() == second_registry.read_bytes()

    previous_registry = os.environ.get("ROBOT_SF_MAP_REGISTRY")
    monkeypatch.setenv("ROBOT_SF_MAP_REGISTRY", str(runner_registry))
    replay_gallery.scenario_loader._load_map_registry.cache_clear()
    try:
        materialized_scenario = case_dir / "inputs" / "scenario.yaml"
        loaded = replay_gallery.scenario_loader.load_scenarios(
            materialized_scenario, base_dir=materialized_scenario
        )
        resolved = (materialized_scenario.parent / loaded[0]["map_file"]).resolve()
    finally:
        replay_gallery.scenario_loader._load_map_registry.cache_clear()
        if previous_registry is None:
            monkeypatch.delenv("ROBOT_SF_MAP_REGISTRY", raising=False)
        else:
            monkeypatch.setenv("ROBOT_SF_MAP_REGISTRY", previous_registry)
    assert replay_gallery._sha256_file(resolved) == resolution["resolved_map_sha256"]


def test_map_id_materialization_uses_registry_bytes_captured_at_selection(
    tmp_path: Path, monkeypatch: Any
) -> None:
    source_registry = replay_gallery.scenario_loader._resolve_map_registry_path()
    assert source_registry is not None
    source_payload = yaml.safe_load(source_registry.read_text(encoding="utf-8"))
    source_row = next(
        row for row in source_payload["maps"] if row.get("map_id") == "classic_cross_trap"
    )
    selected_map = replay_gallery.scenario_loader.resolve_map_id(
        "classic_cross_trap", source=source_registry
    )
    alternate_map = Path(__file__).resolve().parents[2] / "maps/svg_maps/02_simple_maps.svg"
    override_registry = tmp_path / "map_registry.yaml"
    selected_row = dict(source_row)
    selected_row["path"] = str(selected_map)
    payload = dict(source_payload)
    payload["maps"] = [selected_row]
    override_registry.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    monkeypatch.setenv("ROBOT_SF_MAP_REGISTRY", str(override_registry))
    replay_gallery.scenario_loader._load_map_registry.cache_clear()

    snapshot, error = replay_gallery._snapshot_map_id_input(
        {"map_id": "classic_cross_trap"},
        scenario_path=tmp_path / "scenario.yaml",
        episode_path=tmp_path / "episode.jsonl",
        source_record={"scenario_params": {"map_file": str(selected_map)}},
        root=tmp_path,
        source_root=tmp_path,
    )
    assert error is None and snapshot is not None
    assert snapshot["registry_override_used"] is True
    assert snapshot["source_registry_binding"]["status"] == "unknown"
    selected_registry_bytes = snapshot["registry_bytes"]
    selected_map_sha256 = snapshot["map_sha256"]

    changed_row = dict(selected_row)
    changed_row["path"] = str(alternate_map)
    changed_row["source_sha256"] = replay_gallery._sha256_file(alternate_map)
    changed_payload = dict(source_payload)
    changed_payload["maps"] = [changed_row]
    override_registry.write_text(yaml.safe_dump(changed_payload, sort_keys=False), encoding="utf-8")

    input_dir = tmp_path / "case" / "inputs"
    assets_dir = input_dir / "assets"
    assets_dir.mkdir(parents=True)
    materialized, materialization_error = replay_gallery._materialize_map_id_input(
        snapshot,
        assets_dir=assets_dir,
        input_dir=input_dir,
    )
    assert materialization_error is None and materialized is not None
    assert materialized["receipt"]["source_registry_override_used"] is True
    source_registry_asset = next(
        asset for asset in materialized["assets"] if asset["field"] == "map_registry_source"
    )
    bundled_source_registry = tmp_path / "case" / source_registry_asset["bundle_path"]
    assert bundled_source_registry.read_bytes() == selected_registry_bytes
    input_binding = replay_gallery._source_input_binding(
        {
            "assets": materialized["assets"],
            "map_resolution": materialized["receipt"],
        },
        {"file_inputs": []},
        effective_scenario_hash="a" * 64,
        root=tmp_path,
        case_dir=tmp_path / "case",
    )
    assert input_binding["status"] == "unknown"
    override_check = next(
        check
        for check in input_binding["checks"]
        if check.get("input") == "source_map_registry_environment"
    )
    assert override_check["status"] == "unknown"
    assert override_check["reason"] == "source_registry_override_not_attested_by_episode_provenance"
    monkeypatch.setenv("ROBOT_SF_MAP_REGISTRY", str(input_dir / "map_registry.yaml"))
    replay_gallery.scenario_loader._load_map_registry.cache_clear()
    resolved = replay_gallery.scenario_loader.resolve_map_id(
        "classic_cross_trap", source=input_dir / "scenario.yaml"
    )

    assert replay_gallery._sha256_file(resolved) == selected_map_sha256


def test_selection_binding_rejects_alternate_runner_map_registry_path(tmp_path: Path) -> None:
    map_id_snapshot = {
        "map_id": "fixture-map",
        "registry_sha256": "a" * 64,
        "map_sha256": "b" * 64,
    }
    runner_registry_path = "inputs/map_registry.yaml"
    runner_registry_sha256 = "c" * 64
    selected = {
        "scenario_sha256": "scenario-digest",
        "effective_scenario_hash": "effective-digest",
        "map_file_declared": False,
        "map_id_snapshot": map_id_snapshot,
    }
    materialization = {
        "source_scenario_sha256": "scenario-digest",
        "effective_scenario_hash": "effective-digest",
        "runner_map_registry_path": runner_registry_path,
        "map_resolution": {
            "status": "materialized",
            "map_id": "fixture-map",
            "source_registry_sha256": "a" * 64,
            "resolved_map_sha256": "b" * 64,
            "runner_registry_bundle_path": runner_registry_path,
            "runner_registry_sha256": runner_registry_sha256,
        },
        "assets": [
            {
                "field": "resolved_map_id",
                "source_sha256": "b" * 64,
                "bundle_path": "inputs/assets/map.svg",
            },
            {
                "field": "map_registry_source",
                "source_sha256": "a" * 64,
                "bundle_path": "inputs/assets/source-map-registry.yaml",
            },
            {
                "field": "runner_map_registry",
                "source_sha256": runner_registry_sha256,
                "bundle_path": runner_registry_path,
            },
        ],
    }
    case_dir = tmp_path / "case"
    case_dir.mkdir()
    (case_dir / "inputs" / "assets").mkdir(parents=True)
    for asset in materialization["assets"]:
        path = case_dir / asset["bundle_path"]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"fixture")
        asset["source_sha256"] = replay_gallery._sha256_file(path)
    # Keep the selected source digests aligned with the exact fixture bytes.
    materialization["map_resolution"]["source_registry_sha256"] = materialization["assets"][1][
        "source_sha256"
    ]
    materialization["map_resolution"]["resolved_map_sha256"] = materialization["assets"][0][
        "source_sha256"
    ]
    map_id_snapshot["registry_sha256"] = materialization["assets"][1]["source_sha256"]
    map_id_snapshot["map_sha256"] = materialization["assets"][0]["source_sha256"]
    materialization["map_resolution"]["runner_registry_sha256"] = materialization["assets"][2][
        "source_sha256"
    ]

    error, _ = replay_gallery._materialized_selection_binding(
        selected, materialization, case_dir=case_dir
    )
    assert error is None

    materialization["runner_map_registry_path"] = "inputs/alternate-map-registry.yaml"
    error, binding = replay_gallery._materialized_selection_binding(
        selected, materialization, case_dir=case_dir
    )
    assert error == "not_replayed_map_id_input_unavailable"
    assert binding["reason"] == "runner_map_registry_path_or_digest_mismatch"
    replay_gallery.scenario_loader._load_map_registry.cache_clear()


def test_selection_binding_rejects_route_asset_digest_change_with_same_semantics(
    tmp_path: Path,
) -> None:
    selected_route_bytes = b"routes:\n  - {route_id: north, points: [[0, 0], [1, 1]]}\n"
    materialized_route_bytes = b"routes: [{route_id: north, points: [[0, 0], [1, 1]]}]\n"
    assert yaml.safe_load(selected_route_bytes) == yaml.safe_load(materialized_route_bytes)
    selected_route_sha256 = hashlib.sha256(selected_route_bytes).hexdigest()
    materialized_route_sha256 = hashlib.sha256(materialized_route_bytes).hexdigest()
    assert selected_route_sha256 != materialized_route_sha256

    selected = {
        "scenario_sha256": "scenario-digest",
        "effective_scenario_hash": "effective-digest",
        "map_file_declared": False,
        "route_overrides_sha256": selected_route_sha256,
    }
    materialization = {
        "source_scenario_sha256": "scenario-digest",
        "effective_scenario_hash": "effective-digest",
        "assets": [
            {
                "field": "route_overrides_file",
                "source_sha256": materialized_route_sha256,
                "bundle_path": "inputs/assets/routes.yaml",
            }
        ],
    }
    case_dir = tmp_path / "case"
    assets_dir = case_dir / "inputs" / "assets"
    assets_dir.mkdir(parents=True)
    (assets_dir / "routes.yaml").write_bytes(materialized_route_bytes)

    error, binding = replay_gallery._materialized_selection_binding(
        selected, materialization, case_dir=case_dir
    )

    assert error == "not_replayed_route_overrides_changed_after_selection"
    assert binding == {
        "status": "mismatch",
        "selected_source_sha256": selected_route_sha256,
        "materialized_source_sha256": materialized_route_sha256,
    }


def test_selection_binding_rejects_changed_bundled_route_override_bytes(
    tmp_path: Path,
) -> None:
    selected_route_bytes = b"routes: [{route_id: north, points: [[0, 0], [1, 1]]}]\n"
    bundled_route_bytes = b"routes: [{route_id: south, points: [[0, 0], [1, 1]]}]\n"
    selected_route_sha256 = hashlib.sha256(selected_route_bytes).hexdigest()
    selected = {
        "scenario_sha256": "scenario-digest",
        "effective_scenario_hash": "effective-digest",
        "map_file_declared": False,
        "route_overrides_sha256": selected_route_sha256,
    }
    materialization = {
        "source_scenario_sha256": "scenario-digest",
        "effective_scenario_hash": "effective-digest",
        "assets": [
            {
                "field": "route_overrides_file",
                "source_sha256": selected_route_sha256,
                "bundle_path": "inputs/assets/routes.yaml",
            }
        ],
    }
    case_dir = tmp_path / "case"
    assets_dir = case_dir / "inputs" / "assets"
    assets_dir.mkdir(parents=True)
    (assets_dir / "routes.yaml").write_bytes(bundled_route_bytes)

    error, binding = replay_gallery._materialized_selection_binding(
        selected, materialization, case_dir=case_dir
    )

    assert error == "not_replayed_route_overrides_changed_after_selection"
    assert binding["status"] == "mismatch"
    assert binding["input"] == "route_overrides_file"


def test_gallery_output_must_resolve_inside_ignored_output_tree(
    tmp_path: Path, monkeypatch: Any
) -> None:
    manifest = _source_manifest(tmp_path)
    outside = tmp_path / "docs" / "gallery"

    with pytest.raises(ValueError, match="inside the repository ignored output/"):
        replay_gallery.build_replay_gallery(manifest, outside, video=False)

    output_root = tmp_path / "output"
    output_root.mkdir()
    external_root = tmp_path / "external"
    external_root.mkdir()
    (output_root / "escape").symlink_to(external_root, target_is_directory=True)
    with pytest.raises(ValueError, match="inside the repository ignored output/"):
        replay_gallery.build_replay_gallery(
            manifest, output_root / "escape" / "gallery", video=False
        )

    linked_repo = tmp_path / "linked_repo"
    linked_repo.mkdir()
    (linked_repo / "output").symlink_to(external_root, target_is_directory=True)
    monkeypatch.setattr(replay_gallery, "_repository_root", lambda: linked_repo)
    with pytest.raises(ValueError, match="output/ must not be a symlink"):
        replay_gallery.build_replay_gallery(
            manifest, linked_repo / "output" / "gallery", video=False
        )


def test_reused_pedestrian_slot_is_split_into_distinct_tracks() -> None:
    trace = {
        "schema_version": "simulation-step-trace.v1",
        "steps": [
            {
                "time_s": 0.1,
                "robot": {"position": [0.0, 0.0], "heading": 0.0},
                "pedestrians": [{"id": "simulator-slot-0", "position": [36.0, 36.0]}],
            },
            {
                "time_s": 0.2,
                "robot": {"position": [0.1, 0.0], "heading": 0.0},
                "pedestrians": [{"id": "simulator-slot-0", "position": [4.0, 2.0]}],
            },
        ],
    }

    steps, _critical_step, _clearance, continuity = replay_gallery._replay_steps_from_trace(trace)
    row = EpisodeRow.from_dict(
        {
            "episode_id": "reused-slot",
            "scenario_id": "fixture",
            "seed": 1,
            "replay_steps": steps,
        }
    )
    replay = build_replay_from_episode_row(row)
    assert replay is not None

    tracks = _pedestrian_trajectories(replay)

    assert list(tracks) == ["simulator-slot-0", "simulator-slot-0#segment-1"]
    assert [len(points) for points in tracks.values()] == [1, 1]
    assert continuity["discontinuity_split_count"] == 1
    assert continuity["maximum_assumed_track_speed_mps"] == 12.0


def test_trace_visual_annotations_keep_collision_and_clearance_sources_separate() -> None:
    trace = {
        "schema_version": "simulation-step-trace.v1",
        "steps": [
            {
                "time_s": 0.1,
                "robot": {"position": [0.0, 0.0], "heading": 0.0},
                "pedestrians": [{"position": [1.0, 0.0], "surface_clearance_m": 0.4}],
            },
            {
                "time_s": 0.2,
                "robot": {"position": [0.1, 0.0], "heading": 0.0},
                "pedestrians": [{"position": [0.2, 0.0], "surface_clearance_m": -0.01}],
            },
            {
                "time_s": 0.3,
                "robot": {"position": [0.2, 0.0], "heading": 0.0},
                "pedestrians": [{"position": [0.2, 0.0], "surface_clearance_m": 0.0}],
            },
        ],
    }
    replay_steps, _critical, _clearance, _continuity = replay_gallery._replay_steps_from_trace(
        trace
    )
    record = {
        "event_ledger": {
            "exact_events": {"collision": True},
            "collision_events": [
                {
                    "collision_time": 0.3,
                    "exact_event_source": "runtime.step.meta.is_pedestrian_collision",
                }
            ],
        }
    }

    annotations = replay_gallery._trace_visual_annotations(record, trace, replay_steps)

    minimum = annotations["minimum_clearance"]
    collision = annotations["collision_events"][0]
    assert minimum["source_trace_step_index"] == 1
    assert minimum["render_step_index"] == 1
    assert minimum["source_path"].endswith("steps[1].pedestrians[0].surface_clearance_m")
    assert collision["event_time_s"] == 0.3
    assert collision["render_step_index"] == 2
    assert collision["source_path"] == "event_ledger.collision_events[0].collision_time"
    assert minimum["render_step_index"] != collision["render_step_index"]


def test_trace_filmstrip_sampling_retains_distinct_source_events() -> None:
    assert replay_gallery._deterministic_trace_frame_steps(
        10, protected_indices=[8, 9, 8, "invalid", -1, 10]
    ) == [0, 2, 4, 7, 8, 9]


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
    candidate_row["certification_status"] = _bind_certificate_to_scenario(
        candidate_row["certification_status"], scenario_path
    )
    _bind_effective_scenario_hash(candidate_row, scenario)
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

    result = replay_gallery.build_replay_gallery(
        manifest, tmp_path / "output" / "gallery", video=False
    )

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
    candidate_row["certification_status"] = _bind_certificate_to_scenario(
        candidate_row["certification_status"], scenario_path
    )
    _bind_effective_scenario_hash(candidate_row, scenario)
    manifest.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")
    materialize_scenario = replay_gallery._materialize_scenario

    def mutate_source_map(source: Path, input_dir: Path, **kwargs: Any) -> dict[str, Any]:
        map_path.write_bytes(b"changed after selection")
        return materialize_scenario(source, input_dir, **kwargs)

    monkeypatch.setattr(replay_gallery, "_materialize_scenario", mutate_source_map)
    calls = _install_fake_replay(monkeypatch)

    result = replay_gallery.build_replay_gallery(
        manifest, tmp_path / "output" / "gallery", video=False
    )

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
        tmp_path / "output" / "gallery",
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

    output_dir = tmp_path / "output" / "gallery"
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

    output_dir = tmp_path / "output" / "gallery"
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

    result = replay_gallery.build_replay_gallery(
        manifest, tmp_path / "output" / "gallery", video=False
    )

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

    result = replay_gallery.build_replay_gallery(
        manifest, tmp_path / "output" / "gallery", video=False
    )

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

    result = replay_gallery.build_replay_gallery(
        manifest, tmp_path / "output" / "gallery", video=False
    )

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

    result = replay_gallery.build_replay_gallery(
        manifest, tmp_path / "output" / "gallery", video=False
    )

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

    result = replay_gallery.build_replay_gallery(
        manifest, tmp_path / "output" / "gallery", video=False
    )

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

    result = replay_gallery.build_replay_gallery(
        manifest, tmp_path / "output" / "gallery", video=False
    )

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

    result = replay_gallery.build_replay_gallery(
        manifest, tmp_path / "output" / "gallery", video=False
    )

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

    result = replay_gallery.build_replay_gallery(
        manifest, tmp_path / "output" / "gallery", video=False
    )

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

    result = replay_gallery.build_replay_gallery(
        manifest, tmp_path / "output" / "gallery", video=False
    )

    case = result["cases"][0]
    assert case["replay_match"] == "unavailable"
    assert case["verification_status"] == "replay_execution_unavailable"
    assert case["replay"]["availability_error"] == "replay_benchmark_unavailable"
    assert case["replay"]["summary"]["benchmark_availability"]["benchmark_success"] is False
    assert case["replay"]["identity_matches"] is True
    assert case["replay"]["outcome_matches"] is True


@pytest.mark.parametrize("execution_mode", ["adapter", "mixed"])
def test_gallery_keeps_adapter_and_mixed_runtime_fallback_unavailable(
    tmp_path: Path, monkeypatch: Any, execution_mode: str
) -> None:
    manifest = _source_manifest(tmp_path, execution_mode=execution_mode)
    summary = _runner_summary(execution_mode=execution_mode)
    summary["algorithm_metadata_contract"]["planner_runtime"] = {"fallback_used": True}
    summary["benchmark_availability"] = availability_payload(summary)
    _install_fake_replay(
        monkeypatch,
        runner_summary=summary,
        replay_execution_mode=execution_mode,
    )

    result = replay_gallery.build_replay_gallery(
        manifest, tmp_path / "output" / "gallery", video=False
    )

    case = result["cases"][0]
    assert case["replay_match"] == "unavailable"
    assert case["replay"]["availability_error"] == "replay_benchmark_unavailable"


@pytest.mark.parametrize(
    ("preflight_status", "expected_error"),
    [
        ("fallback", "replay_preflight_not_successful"),
        ("skipped", "replay_preflight_not_successful"),
    ],
)
@pytest.mark.parametrize("execution_mode", ["adapter", "mixed"])
def test_gallery_keeps_adapter_and_mixed_fallback_or_degraded_preflight_unavailable(
    tmp_path: Path,
    monkeypatch: Any,
    execution_mode: str,
    preflight_status: str,
    expected_error: str,
) -> None:
    manifest = _source_manifest(tmp_path, execution_mode=execution_mode)
    summary = _runner_summary(preflight_status=preflight_status, execution_mode=execution_mode)
    _install_fake_replay(
        monkeypatch,
        runner_summary=summary,
        replay_execution_mode=execution_mode,
    )

    result = replay_gallery.build_replay_gallery(
        manifest, tmp_path / "output" / "gallery", video=False
    )

    case = result["cases"][0]
    assert case["replay_match"] == "unavailable"
    assert case["replay"]["availability_error"] == expected_error


@pytest.mark.parametrize("execution_mode", ["adapter", "mixed"])
def test_gallery_replays_supported_adapter_and_mixed_modes(
    tmp_path: Path, monkeypatch: Any, execution_mode: str
) -> None:
    """Replay native-only-ineligible adapter and mixed rows for diagnostics only."""
    manifest = _source_manifest(tmp_path, execution_mode=execution_mode)
    source_payload = json.loads(manifest.read_text(encoding="utf-8"))
    source_eligibility = source_payload["candidates"][0]["analysis_eligibility"]
    assert source_eligibility["eligible"] is False
    assert source_eligibility["reason_codes"] == ["execution_mode_not_native"]
    summary = _runner_summary(execution_mode=execution_mode)
    _install_fake_replay(
        monkeypatch,
        runner_summary=summary,
        replay_execution_mode=execution_mode,
    )

    gallery_dir = tmp_path / "output" / "gallery"
    result = replay_gallery.build_replay_gallery(manifest, gallery_dir, video=False)

    case = result["cases"][0]
    assert result["summary"]["selected_case_count"] == 1
    assert case["replay_match"] == "match"
    assert case["replay"]["benchmark_availability"]["execution_mode"] == execution_mode
    assert case["replay"]["benchmark_availability"]["readiness_status"] == "adapter"
    assert case["execution_mode_claim_boundary"] == "diagnostic_only"
    readme = (gallery_dir / "README.md").read_text(encoding="utf-8")
    assert f"- Execution mode: source `{execution_mode}`; replay `{execution_mode}`" in readme
    assert "- Claim boundary: `diagnostic_only`" in readme


@pytest.mark.parametrize(
    ("source_mode", "replay_mode"),
    [("native", "adapter"), ("adapter", "mixed"), ("mixed", "adapter")],
)
def test_gallery_rejects_replay_summary_mode_mismatch(
    tmp_path: Path,
    monkeypatch: Any,
    source_mode: str,
    replay_mode: str,
) -> None:
    manifest = _source_manifest(tmp_path, execution_mode=source_mode)
    summary = _runner_summary(execution_mode=replay_mode)
    _install_fake_replay(
        monkeypatch,
        runner_summary=summary,
        replay_execution_mode=replay_mode,
    )

    result = replay_gallery.build_replay_gallery(
        manifest, tmp_path / "output" / "gallery", video=False
    )

    case = result["cases"][0]
    assert case["replay_match"] == "unavailable"
    assert case["verification_status"] == "replay_execution_unavailable"
    assert case["replay"]["availability_error"] == "replay_execution_mode_mismatch"


def test_gallery_rejects_replay_episode_mode_mismatch(tmp_path: Path, monkeypatch: Any) -> None:
    manifest = _source_manifest(tmp_path, execution_mode="adapter")
    summary = _runner_summary(execution_mode="adapter")
    _install_fake_replay(
        monkeypatch,
        runner_summary=summary,
        replay_execution_mode="mixed",
    )

    result = replay_gallery.build_replay_gallery(
        manifest, tmp_path / "output" / "gallery", video=False
    )

    case = result["cases"][0]
    assert case["replay_match"] == "unavailable"
    assert case["verification_status"] == "replay_execution_unavailable"
    assert case["replay"]["availability_error"] == "replay_episode_execution_mode_mismatch"


@pytest.mark.parametrize(
    ("execution_mode", "expected_error"),
    [
        ("unknown", "replay_execution_mode_unknown"),
        ("experimental", "replay_execution_mode_unsupported"),
    ],
)
def test_replay_availability_rejects_unknown_or_unsupported_mode(
    execution_mode: str,
    expected_error: str,
) -> None:
    summary = _runner_summary(execution_mode=execution_mode)

    _availability, error = replay_gallery._replay_availability(
        summary, expected_execution_mode="adapter"
    )

    assert error == expected_error


def test_gallery_rejects_misreported_adapter_readiness(
    tmp_path: Path,
) -> None:
    manifest = _source_manifest(tmp_path, execution_mode="adapter")
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["candidates"][0]["failure_attribution"]["details"]["readiness_status"] = "native"
    manifest.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")

    result = replay_gallery.build_replay_gallery(
        manifest, tmp_path / "output" / "gallery", render=False, video=False
    )

    assert result["summary"]["selected_case_count"] == 0
    assert result["summary"]["dispositions"]["source_readiness_execution_mode_mismatch"] == 1


def test_gallery_rejects_adapter_source_episode_with_native_metadata(
    tmp_path: Path,
) -> None:
    manifest = _source_manifest(tmp_path, execution_mode="adapter")
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    episode_path = Path(payload["candidates"][0]["episode_record_path"])
    episode_path.write_text(json.dumps(_episode()) + "\n", encoding="utf-8")

    result = replay_gallery.build_replay_gallery(
        manifest, tmp_path / "output" / "gallery", render=False, video=False
    )

    assert result["summary"]["selected_case_count"] == 0
    assert result["summary"]["dispositions"]["source_execution_mode_mismatch"] == 1


def test_gallery_requires_analysis_eligibility_mode_to_match_source_mode(
    tmp_path: Path,
) -> None:
    manifest = _source_manifest(tmp_path, execution_mode="adapter")
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["candidates"][0]["analysis_eligibility"]["execution_mode"] = "mixed"
    manifest.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")

    result = replay_gallery.build_replay_gallery(
        manifest, tmp_path / "output" / "gallery", render=False, video=False
    )

    assert result["summary"]["selected_case_count"] == 0
    assert result["summary"]["dispositions"]["source_execution_mode_mismatch"] == 1


def test_gallery_keeps_adapter_native_only_analysis_exclusions_fail_closed(
    tmp_path: Path,
) -> None:
    manifest = _source_manifest(tmp_path, execution_mode="adapter")
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    eligibility = payload["candidates"][0]["analysis_eligibility"]
    eligibility["reason_codes"] = ["execution_mode_not_native", "objective_unscored"]
    manifest.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")

    result = replay_gallery.build_replay_gallery(
        manifest, tmp_path / "output" / "gallery", render=False, video=False
    )

    assert result["summary"]["selected_case_count"] == 0
    assert result["summary"]["dispositions"]["source_analysis_eligibility_reasons_mismatch"] == 1


@pytest.mark.parametrize("execution_mode", ["adapter", "mixed"])
def test_gallery_rejects_non_native_analysis_receipt_without_effective_hash(
    tmp_path: Path, execution_mode: str
) -> None:
    manifest = _source_manifest(tmp_path, execution_mode=execution_mode)
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["candidates"][0]["analysis_eligibility"].pop("effective_scenario_hash")
    manifest.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")

    result = replay_gallery.build_replay_gallery(
        manifest, tmp_path / "output" / "gallery", render=False, video=False
    )

    assert result["summary"]["selected_case_count"] == 0
    assert (
        result["summary"]["dispositions"]["source_analysis_eligibility_effective_hash_mismatch"]
        == 1
    )


@pytest.mark.parametrize("execution_mode", ["adapter", "mixed"])
def test_gallery_rejects_non_native_analysis_receipt_with_mismatched_effective_hash(
    tmp_path: Path, execution_mode: str
) -> None:
    manifest = _source_manifest(tmp_path, execution_mode=execution_mode)
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    eligibility = payload["candidates"][0]["analysis_eligibility"]
    expected_hash = eligibility["effective_scenario_hash"]
    eligibility["effective_scenario_hash"] = "0" * 64 if expected_hash != "0" * 64 else "1" * 64
    manifest.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")

    result = replay_gallery.build_replay_gallery(
        manifest, tmp_path / "output" / "gallery", render=False, video=False
    )

    assert result["summary"]["selected_case_count"] == 0
    assert (
        result["summary"]["dispositions"]["source_analysis_eligibility_effective_hash_mismatch"]
        == 1
    )


def test_gallery_rejects_adapter_receipt_marked_eligible_with_native_only_reason(
    tmp_path: Path,
) -> None:
    manifest = _source_manifest(tmp_path, execution_mode="adapter")
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["candidates"][0]["analysis_eligibility"]["eligible"] = True
    manifest.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")

    result = replay_gallery.build_replay_gallery(
        manifest, tmp_path / "output" / "gallery", render=False, video=False
    )

    assert result["summary"]["selected_case_count"] == 0
    assert result["summary"]["dispositions"]["source_analysis_eligibility_status_mismatch"] == 1


def test_gallery_rejects_unsupported_analysis_eligibility_schema(
    tmp_path: Path,
) -> None:
    manifest = _source_manifest(tmp_path, execution_mode="adapter")
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["candidates"][0]["analysis_eligibility"]["schema_version"] = (
        "search_analysis_eligibility.v2"
    )
    manifest.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")

    result = replay_gallery.build_replay_gallery(
        manifest, tmp_path / "output" / "gallery", render=False, video=False
    )

    assert result["summary"]["selected_case_count"] == 0
    assert result["summary"]["dispositions"]["source_analysis_eligibility_schema_unsupported"] == 1


def test_gallery_rejects_malformed_analysis_eligibility_reason_list(
    tmp_path: Path,
) -> None:
    manifest = _source_manifest(tmp_path, execution_mode="adapter")
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["candidates"][0]["analysis_eligibility"]["reason_codes"] = "execution_mode_not_native"
    manifest.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")

    result = replay_gallery.build_replay_gallery(
        manifest, tmp_path / "output" / "gallery", render=False, video=False
    )

    assert result["summary"]["selected_case_count"] == 0
    assert result["summary"]["dispositions"]["source_analysis_eligibility_reasons_malformed"] == 1


def test_gallery_rejects_mixed_analysis_eligibility_reasons(
    tmp_path: Path,
) -> None:
    manifest = _source_manifest(tmp_path, execution_mode="mixed")
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["candidates"][0]["analysis_eligibility"]["reason_codes"] = [
        "execution_mode_not_native",
        "objective_unscored",
    ]
    manifest.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")

    result = replay_gallery.build_replay_gallery(
        manifest, tmp_path / "output" / "gallery", render=False, video=False
    )

    assert result["summary"]["selected_case_count"] == 0
    assert result["summary"]["dispositions"]["source_analysis_eligibility_reasons_mismatch"] == 1


@pytest.mark.parametrize(
    ("execution_mode", "expected_disposition"),
    [
        ("unknown", "source_execution_mode_unknown"),
        ("experimental", "source_execution_mode_unsupported"),
    ],
)
def test_gallery_rejects_unknown_or_unsupported_source_execution_mode(
    tmp_path: Path,
    execution_mode: str,
    expected_disposition: str,
) -> None:
    manifest = _source_manifest(tmp_path, execution_mode=execution_mode)

    result = replay_gallery.build_replay_gallery(
        manifest, tmp_path / "output" / "gallery", render=False, video=False
    )

    assert result["summary"]["selected_case_count"] == 0
    assert result["summary"]["dispositions"][expected_disposition] == 1


@pytest.mark.parametrize("execution_mode", ["native", "adapter", "mixed"])
def test_gallery_rejects_fallback_marker_in_replay_episode_row(
    tmp_path: Path, monkeypatch: Any, execution_mode: str
) -> None:
    manifest = _source_manifest(tmp_path, execution_mode=execution_mode)
    _install_fake_replay(monkeypatch, replay_execution_mode=execution_mode)
    fake_run_batch = replay_gallery.run_batch

    def replay_with_row_fallback(scenario_path: Path, **kwargs: Any) -> dict[str, Any]:
        summary = fake_run_batch(scenario_path, **kwargs)
        record_path = kwargs["out_path"]
        record = json.loads(record_path.read_text(encoding="utf-8"))
        record["algorithm_metadata"]["planner_runtime"] = {"fallback_used": True}
        record_path.write_text(json.dumps(record) + "\n", encoding="utf-8")
        return summary

    monkeypatch.setattr(replay_gallery, "run_batch", replay_with_row_fallback)

    result = replay_gallery.build_replay_gallery(
        manifest, tmp_path / "output" / "gallery", video=False
    )

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

    result = replay_gallery.build_replay_gallery(
        manifest, tmp_path / "output" / "gallery", video=False
    )

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

    result = replay_gallery.build_replay_gallery(
        manifest, tmp_path / "output" / "gallery", video=False
    )

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
    payload["candidates"][0]["certification_status"] = _bind_certificate_to_scenario(
        payload["candidates"][0]["certification_status"], scenario_path
    )
    _bind_effective_scenario_hash(payload["candidates"][0], scenario, route_payload)
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

    result = replay_gallery.build_replay_gallery(
        manifest, tmp_path / "output" / "gallery", video=False
    )

    case = result["cases"][0]
    assert case["replay_match"] == "unavailable"
    assert case["verification_status"] == (
        "not_replayed_effective_scenario_changed_after_selection"
    )
    assert case["materialization"]["selection_binding"]["status"] == "mismatch"
    assert calls == []


def test_gallery_rejects_route_input_drift_before_certificate_selection(tmp_path: Path) -> None:
    """A certificate cannot survive changed referenced bytes before candidate selection."""
    manifest = _source_manifest(tmp_path)
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    candidate_row = payload["candidates"][0]
    scenario_path = Path(candidate_row["scenario_yaml_path"])
    scenario_payload = yaml.safe_load(scenario_path.read_text(encoding="utf-8"))
    scenario = scenario_payload["scenarios"][0]
    route_path = scenario_path.parent / "route-overrides.yaml"
    original_route = {"route": {"waypoints": [[1.0, 1.0], [2.0, 2.0]]}}
    route_path.write_text(yaml.safe_dump(original_route), encoding="utf-8")
    scenario["route_overrides_file"] = str(route_path)
    scenario_path.write_text(yaml.safe_dump(scenario_payload, sort_keys=False), encoding="utf-8")
    candidate_row["certification_status"] = _bind_certificate_to_scenario(
        candidate_row["certification_status"], scenario_path
    )
    _bind_effective_scenario_hash(candidate_row, scenario, original_route)
    route_path.write_text(
        yaml.safe_dump({"route": {"waypoints": [[1.0, 1.0], [3.0, 3.0]]}}),
        encoding="utf-8",
    )
    manifest.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")

    result = replay_gallery.build_replay_gallery(
        manifest, tmp_path / "output" / "gallery", render=False, video=False
    )

    assert result["summary"]["selected_case_count"] == 0
    assert result["summary"]["dispositions"] == {"certificate_scenario_fingerprint_mismatch": 1}


def test_gallery_rechecks_route_bytes_against_selection_snapshot(
    tmp_path: Path, monkeypatch: Any
) -> None:
    """Route bytes changing after hashing cannot retain the bound static certificate."""
    manifest = _source_manifest(tmp_path)
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    candidate_row = payload["candidates"][0]
    scenario_path = Path(candidate_row["scenario_yaml_path"])
    scenario_payload = yaml.safe_load(scenario_path.read_text(encoding="utf-8"))
    scenario = scenario_payload["scenarios"][0]
    route_path = scenario_path.parent / "route-overrides.yaml"
    original_route = {"route": {"waypoints": [[1.0, 1.0], [2.0, 2.0]]}}
    changed_route = {"route": {"waypoints": [[1.0, 1.0], [3.0, 3.0]]}}
    route_path.write_text(yaml.safe_dump(original_route), encoding="utf-8")
    scenario["route_overrides_file"] = str(route_path)
    scenario_path.write_text(yaml.safe_dump(scenario_payload, sort_keys=False), encoding="utf-8")
    candidate_row["certification_status"] = _bind_certificate_to_scenario(
        candidate_row["certification_status"], scenario_path
    )
    _bind_effective_scenario_hash(candidate_row, scenario, original_route)
    manifest.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")

    original_hash = replay_gallery._source_effective_scenario_hash

    def mutate_route_after_snapshot(*args: Any, **kwargs: Any) -> Any:
        snapshot = original_hash(*args, **kwargs)
        route_path.write_text(yaml.safe_dump(changed_route), encoding="utf-8")
        return snapshot

    monkeypatch.setattr(
        replay_gallery, "_source_effective_scenario_hash", mutate_route_after_snapshot
    )
    result = replay_gallery.build_replay_gallery(
        manifest, tmp_path / "output" / "gallery", render=False, video=False
    )

    assert result["summary"]["selected_case_count"] == 0
    assert result["summary"]["dispositions"] == {
        "scenario_route_overrides_changed_during_selection": 1
    }


def test_gallery_deduplicates_identical_effective_scenarios(tmp_path: Path) -> None:
    manifest = _source_manifest(tmp_path)
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["candidates"].append(dict(payload["candidates"][0]))
    payload["config"]["budget"] = 2
    payload["summary"]["num_candidates"] = 2
    manifest.write_text(json.dumps(payload) + "\n", encoding="utf-8")

    result = replay_gallery.build_replay_gallery(
        manifest, tmp_path / "output" / "gallery", render=False, video=False
    )

    assert result["summary"]["selected_case_count"] == 1
    assert result["summary"]["dispositions"]["duplicate_effective_scenario"] == 1


def test_gallery_binds_parse_digest_clustering_and_bundle_to_one_manifest_snapshot(
    tmp_path: Path, monkeypatch: Any
) -> None:
    manifest = _source_manifest(tmp_path, manifest_revision="a" * 40)
    snapshot = manifest.read_bytes()
    replacement = json.loads(snapshot)
    replacement["source_revision"] = "b" * 40
    replacement["candidates"] = []
    replacement_bytes = json.dumps(replacement, sort_keys=True).encode("utf-8") + b"\n"

    original_cluster = replay_gallery._failure_cluster_by_candidate

    def replace_source_after_snapshot(source_snapshot: bytes) -> Any:
        assert source_snapshot == snapshot
        manifest.write_bytes(replacement_bytes)
        return original_cluster(source_snapshot)

    monkeypatch.setattr(
        replay_gallery, "_failure_cluster_by_candidate", replace_source_after_snapshot
    )
    _install_fake_replay(monkeypatch)

    result = replay_gallery.build_replay_gallery(
        manifest, tmp_path / "output" / "gallery", render=False, video=False
    )

    bundled_snapshot = (
        tmp_path / "output" / "gallery" / "source_search_manifest.json"
    ).read_bytes()
    assert manifest.read_bytes() == replacement_bytes
    assert bundled_snapshot == snapshot
    assert result["source"]["manifest_sha256"] == hashlib.sha256(snapshot).hexdigest()
    assert result["source"]["source_revision"] == "a" * 40
    assert result["summary"]["selected_case_count"] == 1


def test_gallery_repeated_fixture_runs_have_byte_stable_manifests(
    tmp_path: Path, monkeypatch: Any
) -> None:
    manifest = _source_manifest(tmp_path)
    _install_fake_replay(monkeypatch)

    first = replay_gallery.build_replay_gallery(
        manifest, tmp_path / "output" / "gallery_one", video=False
    )
    second = replay_gallery.build_replay_gallery(
        manifest, tmp_path / "output" / "gallery_two", video=False
    )

    assert first["cases"][0]["case_id"] == second["cases"][0]["case_id"]
    assert (tmp_path / "output" / "gallery_one" / "gallery_manifest.json").read_bytes() == (
        tmp_path / "output" / "gallery_two" / "gallery_manifest.json"
    ).read_bytes()


def test_gallery_manifest_limits_repeat_run_variation_to_declared_receipt_fields(
    tmp_path: Path, monkeypatch: Any
) -> None:
    """Run receipts may vary while case identity and relative paths remain stable."""
    manifest = _source_manifest(tmp_path)
    _install_fake_replay(monkeypatch, vary_runtime_receipts=True)

    first_dir = tmp_path / "output" / "gallery_one"
    second_dir = tmp_path / "output" / "gallery_two"
    replay_gallery.build_replay_gallery(manifest, first_dir, video=False)
    replay_gallery.build_replay_gallery(manifest, second_dir, video=False)
    first_path = first_dir / "gallery_manifest.json"
    second_path = second_dir / "gallery_manifest.json"
    first = json.loads(first_path.read_text(encoding="utf-8"))
    second = json.loads(second_path.read_text(encoding="utf-8"))

    variable_paths = first["reproducibility"]["variable_json_paths"]
    assert variable_paths == second["reproducibility"]["variable_json_paths"]
    assert first["cases"][0]["case_id"] == second["cases"][0]["case_id"]
    assert first["cases"][0]["replay"]["summary"]["out_path"] == ("replay/episode_records.jsonl")
    assert second["cases"][0]["replay"]["summary"]["out_path"] == ("replay/episode_records.jsonl")
    assert (
        first["cases"][0]["replay"]["summary"]["provenance"]["config_identity"]["scenario_path"]
        == "inputs/scenario.yaml"
    )
    assert (
        first["cases"][0]["replay"]["summary"]["provenance"]["result_manifest_path"]
        == "replay/episode_records.jsonl.manifest.json"
    )
    assert str(first_dir) not in first["cases"][0]["replay"]["summary"]["provenance"]["invocation"]
    assert (
        first["cases"][0]["replay"]["episode_record_sha256"]
        != second["cases"][0]["replay"]["episode_record_sha256"]
    )
    assert (
        first["cases"][0]["replay"]["summary"]["batch_runtime_sec"]
        != second["cases"][0]["replay"]["summary"]["batch_runtime_sec"]
    )
    assert (
        first["cases"][0]["replay"]["summary"]["provenance"]["run_id"]
        != second["cases"][0]["replay"]["summary"]["provenance"]["run_id"]
    )

    def without_documented_variable_paths(payload: dict[str, Any]) -> dict[str, Any]:
        normalized = json.loads(json.dumps(payload))
        for path in variable_paths:
            parts = path.split(".")

            def remove(node: Any, index: int) -> None:
                token = parts[index]
                wildcard = token.endswith("[*]")
                key = token[:-3] if wildcard else token
                if not isinstance(node, dict) or key not in node:
                    return
                if index == len(parts) - 1:
                    node.pop(key, None)
                    return
                child = node[key]
                if wildcard:
                    if isinstance(child, list):
                        for item in child:
                            remove(item, index + 1)
                else:
                    remove(child, index + 1)

            remove(normalized, 0)
        return normalized

    assert without_documented_variable_paths(first) == without_documented_variable_paths(second)


def test_gallery_requires_candidate_parameters_to_match_generated_scenario(
    tmp_path: Path,
) -> None:
    manifest = _source_manifest(tmp_path)
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["candidates"][0]["candidate"]["start"]["x"] = 2.0
    manifest.write_text(json.dumps(payload) + "\n", encoding="utf-8")

    result = replay_gallery.build_replay_gallery(
        manifest, tmp_path / "output" / "gallery", render=False, video=False
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
    row["certification_status"] = _bind_certificate_to_scenario(
        row["certification_status"], scenario_path
    )
    manifest.write_text(json.dumps(payload) + "\n", encoding="utf-8")

    result = replay_gallery.build_replay_gallery(
        manifest, tmp_path / "output" / "gallery", render=False, video=False
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
        manifest, tmp_path / "output" / "gallery", render=False, video=False
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
        manifest, tmp_path / "output" / "gallery", render=False, video=False
    )

    assert result["summary"]["selected_case_count"] == 0
    assert result["summary"]["dispositions"]["source_episode_not_a_failure"] == 1


def test_gallery_accounts_invalid_and_failed_candidates_without_selecting_them(
    tmp_path: Path,
) -> None:
    manifest = _source_manifest(tmp_path)
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    invalid = dict(payload["candidates"][0])
    invalid["certification_status"] = _certification_status(
        _scenario_certificate(classification="invalid"), status="failed"
    )
    failed = dict(payload["candidates"][0])
    failed["error"] = "simulator initialization failed"
    payload["candidates"] = [invalid, failed]
    payload["config"]["budget"] = 2
    payload["summary"]["num_candidates"] = 2
    manifest.write_text(json.dumps(payload) + "\n", encoding="utf-8")

    result = replay_gallery.build_replay_gallery(
        manifest, tmp_path / "output" / "gallery", render=False, video=False
    )

    assert result["summary"]["selected_case_count"] == 0
    assert result["summary"]["dispositions"] == {
        "certificate_status_not_passed": 1,
        "evaluation_failed": 1,
    }
    assert result["cases"] == []


@pytest.mark.parametrize(
    ("receipt_mutation", "expected_disposition"),
    [
        (lambda receipt: receipt.update(status="failed"), "certificate_status_not_passed"),
        (lambda receipt: receipt.pop("schema_version"), "certificate_status_schema_invalid"),
        (
            lambda receipt: receipt.update(schema_version="scenario_cert.v0"),
            "certificate_status_schema_invalid",
        ),
        (
            lambda receipt: receipt["details"]["certificates"][0].update(scenario_id="other"),
            "certificate_scenario_id_mismatch",
        ),
        (
            lambda receipt: receipt["details"]["certificates"][0].update(
                source="some/other/scenario.yaml"
            ),
            "certificate_source_mismatch",
        ),
        (
            lambda receipt: receipt["details"]["certificates"][0]["evidence"].update(
                scenario_fingerprint="0" * 16
            ),
            "certificate_scenario_fingerprint_mismatch",
        ),
        (
            lambda receipt: receipt["details"]["certificates"][0]["checks"].update(route_count=2),
            "certificate_incomplete",
        ),
        (
            lambda receipt: receipt["details"]["certificates"][0].update(
                benchmark_eligibility="excluded"
            ),
            "certificate_eligibility_inconsistent",
        ),
        (
            lambda receipt: receipt["details"]["certificates"][0]["route_certificates"][0].update(
                classification="hard_but_solvable"
            ),
            "certificate_eligibility_inconsistent",
        ),
        (
            lambda receipt: receipt["details"]["certificates"][0].pop("evidence"),
            "certificate_structurally_invalid",
        ),
        (
            lambda receipt: receipt["details"]["certificates"].clear(),
            "certificate_list_incomplete_or_ambiguous",
        ),
    ],
    ids=(
        "failed-status",
        "missing-status-schema",
        "invalid-status-schema",
        "wrong-id",
        "wrong-source",
        "wrong-scenario-fingerprint",
        "incomplete",
        "top-level-eligibility-mismatch",
        "top-level-classification-not-worst-route",
        "invalid-certificate-schema",
        "missing-certificate",
    ),
)
def test_gallery_keeps_unbound_or_incomplete_certificates_out_of_selection(
    tmp_path: Path,
    receipt_mutation: Any,
    expected_disposition: str,
) -> None:
    manifest = _source_manifest(tmp_path)
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    receipt = payload["candidates"][0]["certification_status"]
    receipt_mutation(receipt)
    manifest.write_text(json.dumps(payload) + "\n", encoding="utf-8")

    result = replay_gallery.build_replay_gallery(
        manifest, tmp_path / "output" / "gallery", render=False, video=False
    )

    assert result["summary"]["selected_case_count"] == 0
    assert result["summary"]["dispositions"] == {expected_disposition: 1}
    assert result["cases"] == []


@pytest.mark.parametrize(
    ("certificate", "expected"),
    [
        (_certification_status(_scenario_certificate()), "admissible_by_source_certificate"),
        (
            _certification_status(_scenario_certificate(classification="knife_edge")),
            "stress_only",
        ),
        (_certification_status(_scenario_certificate(), status="failed"), "unknown"),
        ({}, "unknown"),
        (
            _certification_status(_scenario_certificate())
            | {"details": {"certificates": [_scenario_certificate(), _scenario_certificate()]}},
            "unknown",
        ),
    ],
)
def test_feasibility_verdict_does_not_promote_an_unbound_certificate(
    certificate: dict[str, Any], expected: str
) -> None:
    verdict = replay_gallery._feasibility_verdict(certificate)

    assert expected in {"admissible_by_source_certificate", "stress_only", "unknown"}
    assert verdict["status"] == "unknown"
    assert verdict["source_certificate"] == certificate
    assert verdict["reason"] == "certificate_not_bound_to_loaded_scenario"


def test_certificate_classification_rejects_malformed_entries_without_inventing_a_verdict() -> None:
    assert replay_gallery._certification_classification(None) is None
    assert (
        replay_gallery._certification_classification(_certification_status(_scenario_certificate()))
        is None
    )
    malformed_entries = _certification_status(_scenario_certificate())
    malformed_entries["details"]["certificates"].append(None)
    assert replay_gallery._certification_classification(malformed_entries) is None


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
