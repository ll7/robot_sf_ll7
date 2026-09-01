"""Contract tests for the deterministic no-submit Slurm launch manifest."""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
from typing import Any

import jsonschema
import pytest

from robot_sf.benchmark import release_protocol
from robot_sf.benchmark import slurm_launch_manifest as launch_manifest
from robot_sf.benchmark.slurm_launch_manifest import (
    EXPECTED_EPISODE_CELLS,
    SCHEMA_VERSION,
    _canonical_json_bytes,
    generate_slurm_launch_manifest,
    sha256_file,
    validate_launch_manifest,
)

_ROOT = Path(__file__).resolve().parents[2]
_SLURM_SCRIPT = _ROOT / "scripts/tools/slurm_campaign_preflight.py"
_SLURM_SPEC = importlib.util.spec_from_file_location("slurm_campaign_preflight", _SLURM_SCRIPT)
assert _SLURM_SPEC and _SLURM_SPEC.loader
slurm_campaign_preflight = importlib.util.module_from_spec(_SLURM_SPEC)
_SLURM_SPEC.loader.exec_module(slurm_campaign_preflight)
_GENERATOR_SCRIPT = _ROOT / "scripts/tools/generate_slurm_launch_manifest.py"
_GENERATOR_SPEC = importlib.util.spec_from_file_location(
    "generate_slurm_launch_manifest", _GENERATOR_SCRIPT
)
assert _GENERATOR_SPEC and _GENERATOR_SPEC.loader
generate_launch_cli = importlib.util.module_from_spec(_GENERATOR_SPEC)
_GENERATOR_SPEC.loader.exec_module(generate_launch_cli)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _fixture(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, Path, Path, str]:
    source_commit = "a" * 40
    planner_keys = [f"planner_{index:02d}" for index in range(14)]
    resolved_seeds = list(range(111, 141))
    config = tmp_path / "configs" / "release.json"
    scenario_matrix = tmp_path / "inputs" / "scenario_matrix.json"
    seed_sets = tmp_path / "inputs" / "seed_sets.json"
    campaign_root = tmp_path / "campaign"
    config.parent.mkdir(parents=True)
    scenario_matrix.parent.mkdir(parents=True, exist_ok=True)
    campaign_root.mkdir()
    config.write_text('{"release": true}\n', encoding="utf-8")
    scenario_matrix.write_text('{"scenarios": 48}\n', encoding="utf-8")
    seed_sets.write_text(json.dumps(resolved_seeds) + "\n", encoding="utf-8")
    resolved_manifest = {
        "release_id": "release-s30-h600",
        "release_tag": "release-s30-h600-test",
        "canonical_campaign_config": "configs/release.json",
        "canonical_campaign_config_sha256": sha256_file(config),
        "scenario": {
            "matrix_path": "inputs/scenario_matrix.json",
            "matrix_sha256": sha256_file(scenario_matrix),
        },
        "seed_policy": {"seed_sets_path": "inputs/seed_sets.json"},
        "planners": {
            "keys": planner_keys,
            "groups": {},
            "config_identities": [
                {"key": key, "algo": key, "path": None, "sha256": None} for key in planner_keys
            ],
        },
        "kinematics": {"matrix": ["differential_drive"]},
        "matrix": {"expected_episode_cells": EXPECTED_EPISODE_CELLS, "horizon_steps": 600},
        "release_contract": {
            "resolved_seeds": resolved_seeds,
            "seed_sets_sha256": sha256_file(seed_sets),
        },
    }
    identity = {
        "schema_version": "benchmark-release-resolved-identity.v1",
        "template": "release-template.yaml",
        "source_commit": source_commit,
        "release_tag": resolved_manifest["release_tag"],
        "publication": {},
        "resolved_manifest": resolved_manifest,
        "resolved_manifest_sha256": hashlib.sha256(
            _canonical_json_bytes(resolved_manifest)
        ).hexdigest(),
    }
    identity_path = tmp_path / "release_identity.resolved.json"
    _write_json(identity_path, identity)

    validate_config = tmp_path / "preflight" / "validate_config.json"
    preview = tmp_path / "preflight" / "preview_scenarios.json"
    matrix_summary = tmp_path / "reports" / "matrix_summary.json"
    _write_json(
        validate_config,
        {
            "campaign_id": "release-test",
            "scenario_count": 48,
            "planner_count": 14,
            "horizon": 600,
            "seed_policy": {"resolved_seeds": resolved_seeds},
        },
    )
    _write_json(preview, {"campaign_id": "release-test", "scenario_count": 48})
    _write_json(
        matrix_summary,
        {
            "campaign_id": "release-test",
            "rows": [
                {
                    "planner_key": key,
                    "scenario_count": 48,
                    "resolved_seeds": resolved_seeds,
                    "repeats": 30,
                    "horizon": 600,
                    "kinematics": "differential_drive",
                }
                for key in planner_keys
            ],
        },
    )
    runner_path = tmp_path / "runner_preflight.json"
    _write_json(
        runner_path,
        {
            "mode": "preflight",
            "manifest_validation": {"status": "valid", "problems": []},
            "resolved_manifest": resolved_manifest,
            "campaign_id": "release-test",
            "campaign_root": str(campaign_root),
            "validate_config_path": str(validate_config),
            "preview_scenarios_path": str(preview),
            "matrix_summary_json": str(matrix_summary),
        },
    )
    monkeypatch.setattr(
        release_protocol, "verify_resolved_release_identity", lambda *args, **kwargs: None
    )
    return identity_path, runner_path, config, source_commit


def _schema() -> dict[str, Any]:
    path = _ROOT / "robot_sf/benchmark/schemas/robot_sf_slurm_launch_manifest.v1.json"
    return json.loads(path.read_text(encoding="utf-8"))


def test_generation_is_deterministic_and_matches_schema(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    identity_path, runner_path, _config, source_commit = _fixture(tmp_path, monkeypatch)
    output_path = tmp_path / "output" / "slurm_launch_manifest.json"

    payload = generate_slurm_launch_manifest(
        resolved_identity_path=identity_path,
        runner_preflight_path=runner_path,
        output_path=output_path,
        repository_root=tmp_path,
    )
    first_bytes = output_path.read_bytes()
    generate_slurm_launch_manifest(
        resolved_identity_path=identity_path,
        runner_preflight_path=runner_path,
        output_path=output_path,
        repository_root=tmp_path,
    )

    assert output_path.read_bytes() == first_bytes
    assert payload["schema_version"] == SCHEMA_VERSION
    assert payload["no_submit"] is True
    assert payload["matrix"]["expected_episode_cells"] == EXPECTED_EPISODE_CELLS
    assert len(payload["cells"]) == 14
    assert {cell["declared_rows"] for cell in payload["cells"]} == {1440}
    jsonschema.Draft202012Validator.check_schema(_schema())
    jsonschema.Draft202012Validator(_schema()).validate(payload)
    assert (
        validate_launch_manifest(
            payload,
            manifest_path=output_path,
            repository_root=tmp_path,
            actual_public_commit=source_commit,
        )
        == []
    )
    report = slurm_campaign_preflight.preflight(
        payload,
        manifest_path=output_path,
        actual_public_commit=source_commit,
        repository_root=tmp_path,
    )
    assert report["submit_safe"] is True
    assert report["no_submit"] is True


def test_validator_rejects_identity_and_cell_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    identity_path, runner_path, _config, source_commit = _fixture(tmp_path, monkeypatch)
    output_path = tmp_path / "slurm_launch_manifest.json"
    payload = generate_slurm_launch_manifest(
        resolved_identity_path=identity_path,
        runner_preflight_path=runner_path,
        output_path=output_path,
        repository_root=tmp_path,
    )

    payload["matrix"]["planner_keys"][0] = "planner_drift"
    payload["cells"][0]["declared_rows"] = 1439
    blockers = validate_launch_manifest(
        payload,
        manifest_path=output_path,
        repository_root=tmp_path,
        actual_public_commit=source_commit,
    )
    assert any("planner_keys do not match resolved identity" in blocker for blocker in blockers)
    assert any("declared_rows" in blocker for blocker in blockers)


def test_validator_rejects_hash_drift_and_traversal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    identity_path, runner_path, _config, source_commit = _fixture(tmp_path, monkeypatch)
    output_path = tmp_path / "slurm_launch_manifest.json"
    payload = generate_slurm_launch_manifest(
        resolved_identity_path=identity_path,
        runner_preflight_path=runner_path,
        output_path=output_path,
        repository_root=tmp_path,
    )

    payload["inputs"][0]["sha256"] = "b" * 64
    payload["cells"][0]["output_root"] = "slurm_cells/../escape"
    blockers = validate_launch_manifest(
        payload,
        manifest_path=output_path,
        repository_root=tmp_path,
        actual_public_commit=source_commit,
    )
    assert any("does not match file bytes" in blocker for blocker in blockers)
    assert any("escapes slurm_cells" in blocker for blocker in blockers)


def test_runner_preflight_mismatch_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    identity_path, runner_path, _config, _source_commit = _fixture(tmp_path, monkeypatch)
    runner = json.loads(runner_path.read_text(encoding="utf-8"))
    runner["manifest_validation"]["status"] = "invalid"
    runner_path.write_text(json.dumps(runner) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="manifest_validation.status"):
        generate_slurm_launch_manifest(
            resolved_identity_path=identity_path,
            runner_preflight_path=runner_path,
            output_path=tmp_path / "slurm_launch_manifest.json",
            repository_root=tmp_path,
        )


def test_consumer_rechecks_bound_runner_semantics(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    identity_path, runner_path, _config, source_commit = _fixture(tmp_path, monkeypatch)
    output_path = tmp_path / "slurm_launch_manifest.json"
    payload = generate_slurm_launch_manifest(
        resolved_identity_path=identity_path,
        runner_preflight_path=runner_path,
        output_path=output_path,
        repository_root=tmp_path,
    )

    runner = json.loads(runner_path.read_text(encoding="utf-8"))
    runner["resolved_manifest"]["matrix"]["horizon_steps"] = 599
    runner_path.write_text(json.dumps(runner) + "\n", encoding="utf-8")
    payload["preflight"]["runner_report"]["sha256"] = sha256_file(runner_path)
    blockers = validate_launch_manifest(
        payload,
        manifest_path=output_path,
        repository_root=tmp_path,
        actual_public_commit=source_commit,
    )
    assert any("bound runner preflight is invalid" in blocker for blocker in blockers)


def test_cli_emits_generation_receipt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    identity_path, runner_path, _config, _source_commit = _fixture(tmp_path, monkeypatch)
    output_path = tmp_path / "slurm_launch_manifest.json"

    assert (
        generate_launch_cli.main(
            [
                "--resolved-identity",
                str(identity_path),
                "--runner-preflight",
                str(runner_path),
                "--output",
                str(output_path),
                "--repository-root",
                str(tmp_path),
            ]
        )
        == 0
    )
    receipt = json.loads(capsys.readouterr().out)
    assert receipt["status"] == "generated"
    assert receipt["sha256"] == sha256_file(output_path)
    assert receipt["expected_episode_cells"] == EXPECTED_EPISODE_CELLS


def test_validator_reports_malformed_top_level_contract(tmp_path: Path) -> None:
    blockers = validate_launch_manifest(
        {
            "schema_version": SCHEMA_VERSION,
            "matrix": {},
            "source": "invalid",
            "release": [],
            "aggregate": [],
        },
        manifest_path=tmp_path / "manifest.json",
        repository_root=tmp_path,
    )
    assert any("manifest_kind" in blocker for blocker in blockers)
    assert any("matrix.planner_arms" in blocker for blocker in blockers)
    assert any("source identity block" in blocker for blocker in blockers)
    assert any("release identity block" in blocker for blocker in blockers)
    assert any("aggregate artifact contract" in blocker for blocker in blockers)


def test_validator_reports_malformed_cells_and_records(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    identity_path, runner_path, _config, source_commit = _fixture(tmp_path, monkeypatch)
    output_path = tmp_path / "slurm_launch_manifest.json"
    payload = generate_slurm_launch_manifest(
        resolved_identity_path=identity_path,
        runner_preflight_path=runner_path,
        output_path=output_path,
        repository_root=tmp_path,
    )
    payload["cells"][0] = {
        "key": "",
        "planner_key": "different",
        "scenario_count": "48",
        "seed_count": 30,
        "kinematics": "unicycle",
        "declared_rows": 1439,
        "instantiated_rows": 1440,
        "output_root": "/absolute",
        "artifact_contract": "",
        "status": "ready",
        "execution_status": "started",
    }
    payload["cells"][1]["output_root"] = "not_slurm_cells/arm"
    payload["inputs"][0] = "not-a-record"
    payload["preflight"]["artifacts"] = []
    payload["aggregate"]["status"] = "success"
    payload["benchmark_result"] = {"success": True}
    blockers = validate_launch_manifest(
        payload,
        manifest_path=output_path,
        repository_root=tmp_path,
        actual_public_commit=source_commit,
    )
    assert any("future outcome field" in blocker for blocker in blockers)
    assert any("cells[0].key is missing" in blocker for blocker in blockers)
    assert any("cells[0].output_root must be relative" in blocker for blocker in blockers)
    assert any("inputs[0] must be an object" in blocker for blocker in blockers)
    assert any("preflight artifact records are missing" in blocker for blocker in blockers)


@pytest.mark.parametrize(
    ("campaign_root", "message"),
    [
        ("", "runner campaign_root is missing"),
        ("not-a-directory", "runner campaign_root is not a directory"),
    ],
)
def test_generator_rejects_invalid_campaign_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    campaign_root: str,
    message: str,
) -> None:
    identity_path, runner_path, _config, _source_commit = _fixture(tmp_path, monkeypatch)
    runner = json.loads(runner_path.read_text(encoding="utf-8"))
    if campaign_root == "not-a-directory":
        (tmp_path / campaign_root).write_text("file\n", encoding="utf-8")
    runner["campaign_root"] = campaign_root
    runner_path.write_text(json.dumps(runner) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        generate_slurm_launch_manifest(
            resolved_identity_path=identity_path,
            runner_preflight_path=runner_path,
            output_path=tmp_path / "slurm_launch_manifest.json",
            repository_root=tmp_path,
        )


def test_generator_rejects_outside_output_and_input_hash_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    identity_path, runner_path, _config, _source_commit = _fixture(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match="output escapes repository"):
        generate_slurm_launch_manifest(
            resolved_identity_path=identity_path,
            runner_preflight_path=runner_path,
            output_path=tmp_path.parent / "slurm_launch_manifest.json",
            repository_root=tmp_path,
        )

    identity = json.loads(identity_path.read_text(encoding="utf-8"))
    identity["resolved_manifest"]["scenario"]["matrix_sha256"] = "b" * 64
    identity["resolved_manifest_sha256"] = hashlib.sha256(
        _canonical_json_bytes(identity["resolved_manifest"])
    ).hexdigest()
    identity_path.write_text(json.dumps(identity) + "\n", encoding="utf-8")
    runner = json.loads(runner_path.read_text(encoding="utf-8"))
    runner["resolved_manifest"] = identity["resolved_manifest"]
    runner_path.write_text(json.dumps(runner) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="input hash mismatch: scenario_matrix"):
        generate_slurm_launch_manifest(
            resolved_identity_path=identity_path,
            runner_preflight_path=runner_path,
            output_path=tmp_path / "slurm_launch_manifest.json",
            repository_root=tmp_path,
        )


def test_generator_rejects_non_object_runner_report(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    identity_path, runner_path, _config, _source_commit = _fixture(tmp_path, monkeypatch)
    runner_path.write_text("[]\n", encoding="utf-8")
    with pytest.raises(ValueError, match="JSON value must be an object"):
        generate_slurm_launch_manifest(
            resolved_identity_path=identity_path,
            runner_preflight_path=runner_path,
            output_path=tmp_path / "slurm_launch_manifest.json",
            repository_root=tmp_path,
        )


def test_scalar_and_path_guards_fail_closed(tmp_path: Path) -> None:
    regular = tmp_path / "regular.json"
    regular.write_text("{}\n", encoding="utf-8")
    directory = tmp_path / "directory"
    directory.mkdir()
    invalid_json = tmp_path / "invalid.json"
    invalid_json.write_text("not-json\n", encoding="utf-8")
    non_object_json = tmp_path / "list.json"
    non_object_json.write_text("[]\n", encoding="utf-8")

    with pytest.raises(ValueError, match="cannot read JSON object"):
        launch_manifest._load_json_object(invalid_json)
    with pytest.raises(ValueError, match="must be an object"):
        launch_manifest._load_json_object(non_object_json)
    with pytest.raises(ValueError, match="file is missing"):
        launch_manifest._resolve_file("", anchor=tmp_path, repository_root=tmp_path, label="file")
    with pytest.raises(ValueError, match="escapes repository"):
        launch_manifest._resolve_file(
            tmp_path.parent / "outside.json",
            anchor=tmp_path,
            repository_root=tmp_path,
            label="file",
        )
    with pytest.raises(ValueError, match="not a regular file"):
        launch_manifest._resolve_file(
            directory, anchor=tmp_path, repository_root=tmp_path, label="file"
        )
    with pytest.raises(ValueError, match="directory is missing"):
        launch_manifest._resolve_directory(
            "", anchor=tmp_path, repository_root=tmp_path, label="directory"
        )
    with pytest.raises(ValueError, match="not a directory"):
        launch_manifest._resolve_directory(
            regular, anchor=tmp_path, repository_root=tmp_path, label="directory"
        )
    with pytest.raises(ValueError, match="must be an object"):
        launch_manifest._require_mapping(None, "mapping")
    with pytest.raises(ValueError, match="string is missing"):
        launch_manifest._require_string(None, "string")
    with pytest.raises(ValueError, match="not a SHA-256"):
        launch_manifest._require_sha("bad", "sha")
    with pytest.raises(ValueError, match="not an exact Git SHA"):
        launch_manifest._require_commit("bad", "commit")
    with pytest.raises(ValueError, match="must be an integer"):
        launch_manifest._require_int(True, "integer")
    with pytest.raises(ValueError, match="must equal 30"):
        launch_manifest._require_int(29, "integer", 30)
    with pytest.raises(ValueError, match="input is missing"):
        launch_manifest._identity_input_records(
            {"scenario": {"matrix_path": None, "matrix_sha256": None}},
            output_parent=tmp_path,
            repository_root=tmp_path,
        )
    assert (
        launch_manifest._manifest_record(
            role="regular", path=regular, output_parent=tmp_path, extra={"kind": "fixture"}
        )["kind"]
        == "fixture"
    )

    symlink = tmp_path / "linked.json"
    try:
        symlink.symlink_to(regular)
    except OSError:
        return
    with pytest.raises(ValueError, match="contains a symlink"):
        launch_manifest._resolve_file(
            symlink, anchor=tmp_path, repository_root=tmp_path, label="file"
        )
