"""Tests for the local model-artifact promotion planner."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from scripts.tools import plan_model_artifact_promotion

if TYPE_CHECKING:
    from pathlib import Path


def _write(path: Path, payload: bytes | str) -> None:
    """Write a small fixture file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(payload, bytes):
        path.write_bytes(payload)
    else:
        path.write_text(payload, encoding="utf-8")


def _repo_fixture(tmp_path: Path) -> dict[str, Path]:
    """Create a tiny repo-like tree with local and durable model configs."""
    present_model = tmp_path / "output" / "model_cache" / "present" / "model.zip"
    _write(present_model, b"checkpoint")
    present_config = tmp_path / "configs" / "baselines" / "present.yaml"
    _write(
        present_config,
        """
model_path: output/model_cache/present/model.zip
profile: experimental
provenance:
  training_config: configs/training/present.yaml
""".strip()
        + "\n",
    )
    missing_config = tmp_path / "configs" / "baselines" / "missing.yaml"
    _write(
        missing_config,
        """
model_path: output/models/missing/model.zip
profile: experimental
""".strip()
        + "\n",
    )
    registered_config = tmp_path / "configs" / "baselines" / "registered.yaml"
    _write(registered_config, "model_id: durable_model\n")
    local_only_config = tmp_path / "configs" / "baselines" / "local_only.yaml"
    _write(local_only_config, "model_id: local_only_model\n")
    source_only_config = tmp_path / "configs" / "baselines" / "source_only.yaml"
    _write(source_only_config, "model_id: source_only_model\n")
    run_id_only_config = tmp_path / "configs" / "baselines" / "run_id_only.yaml"
    _write(run_id_only_config, "model_id: run_id_only_model\n")
    promoted_config = tmp_path / "configs" / "benchmarks" / "promoted.yaml"
    _write(promoted_config, "model_path: output/model_cache/promoted/model.zip\n")
    matrix_config = tmp_path / "configs" / "baselines" / "example_matrix.yaml"
    _write(
        matrix_config,
        """
- id: demo
  repeats: 1
""".strip()
        + "\n",
    )
    blocklist = tmp_path / "configs" / "baselines" / "local_model_artifact_blocklist.yaml"
    _write(
        blocklist,
        """
version: 1
follow_up_issue: https://github.com/ll7/robot_sf_ll7/issues/1764
blocked_references:
  - path: configs/baselines/missing.yaml
    field: model_path
    value: output/models/missing/model.zip
    reason: Synthetic missing checkpoint is local-only.
    availability: unavailable
    decision: recover_or_retire
    next_action: Recover the checkpoint before use, or retire this config.
""".strip()
        + "\n",
    )
    registry = tmp_path / "model" / "registry.yaml"
    _write(
        registry,
        """
version: 1
models:
  - model_id: durable_model
    local_path: output/model_cache/durable_model/model.zip
    public_artifact_source: github_release
    github_release:
      url: https://github.com/ll7/robot_sf_ll7/releases/download/artifact/models/durable.zip
      sha256: abc123
      size_bytes: 123
  - model_id: source_only_model
    public_artifact_source: github_release
  - model_id: run_id_only_model
    wandb_run_id: abc123
  - model_id: local_only_model
    local_path: output/model_cache/local_only/model.zip
    local_only: true
    replacement_model_id: durable_model
""".strip()
        + "\n",
    )
    promoted_surfaces = tmp_path / "configs" / "benchmarks" / "promoted_config_surfaces.yaml"
    _write(
        promoted_surfaces,
        f"""
version: 1
promoted_configs:
  - path: {promoted_config.relative_to(tmp_path).as_posix()}
    reason: Synthetic promoted benchmark config.
""".strip()
        + "\n",
    )
    return {
        "present_model": present_model,
        "present_config": present_config,
        "missing_config": missing_config,
        "registered_config": registered_config,
        "local_only_config": local_only_config,
        "source_only_config": source_only_config,
        "run_id_only_config": run_id_only_config,
        "promoted_config": promoted_config,
        "matrix_config": matrix_config,
        "registry": registry,
        "blocklist": blocklist,
        "promoted_surfaces": promoted_surfaces,
    }


def test_plan_classifies_present_missing_registered_and_promoted_local_configs(
    tmp_path: Path,
) -> None:
    """Planner rows should distinguish local artifacts from durable registry ids."""
    paths = _repo_fixture(tmp_path)

    report = plan_model_artifact_promotion.build_promotion_report(
        [
            paths["present_config"],
            paths["missing_config"],
            paths["registered_config"],
            paths["local_only_config"],
            paths["source_only_config"],
            paths["run_id_only_config"],
            paths["promoted_config"],
        ],
        repo_root=tmp_path,
        registry_path=paths["registry"],
        promoted_surfaces_path=paths["promoted_surfaces"],
    )

    rows = {row["config_path"]: row for row in report["rows"]}
    present = rows["configs/baselines/present.yaml"]
    assert present["classification"] == "promotable"
    assert present["artifact"]["source_path"] == "output/model_cache/present/model.zip"
    assert present["artifact"]["size_bytes"] == len(b"checkpoint")
    assert present["artifact"]["sha256"]
    assert present["promotion_plan"]["target_registry_id"] == "present_local_artifact_candidate"
    assert present["claim_boundary"] == "not benchmark evidence until durable artifact exists"

    missing = rows["configs/baselines/missing.yaml"]
    assert missing["classification"] == "unavailable"
    assert missing["availability"] == "unavailable"
    assert missing["decision"] == "recover_or_retire"
    assert missing["artifact"]["exists"] is False
    assert missing["blocker_reason"] == "Synthetic missing checkpoint is local-only."
    assert "Recover the checkpoint before use" in missing["action"]

    registered = rows["configs/baselines/registered.yaml"]
    assert registered["classification"] == "already_registered"
    assert registered["model_id"] == "durable_model"
    assert registered["registry_entry"]["public_artifact_source"] == "github_release"

    local_only = rows["configs/baselines/local_only.yaml"]
    assert local_only["classification"] == "retired_local_only"
    assert local_only["decision"] == "retired_until_durable_artifact_recovered"
    assert local_only["registry_entry"]["replacement_model_id"] == "durable_model"

    source_only = rows["configs/baselines/source_only.yaml"]
    assert source_only["classification"] == "manual_decision_required"
    assert source_only["decision"] == "resolve_model_id_alias"

    run_id_only = rows["configs/baselines/run_id_only.yaml"]
    assert run_id_only["classification"] == "manual_decision_required"
    assert run_id_only["decision"] == "resolve_model_id_alias"

    promoted = rows["configs/benchmarks/promoted.yaml"]
    assert promoted["classification"] == "manual_decision_required"
    assert promoted["surface"] == "benchmark_promoted"
    assert "must not be treated as benchmark evidence" in promoted["action"]


def test_write_report_emits_json_and_issue_table(tmp_path: Path, capsys) -> None:
    """The CLI should write a reviewable JSON plan plus issue-ready Markdown."""
    paths = _repo_fixture(tmp_path)
    output = tmp_path / "docs" / "context" / "evidence" / "plan.json"

    exit_code = plan_model_artifact_promotion.main(
        [
            "write-report",
            "--repo-root",
            str(tmp_path),
            "--registry-path",
            str(paths["registry"]),
            "--promoted-surfaces",
            str(paths["promoted_surfaces"]),
            "--output",
            str(output),
            str(paths["present_config"]),
            str(paths["missing_config"]),
            str(paths["registered_config"]),
        ]
    )

    assert exit_code == 0
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["schema_version"] == "robot-sf-model-artifact-promotion-plan.v1"
    assert [row["classification"] for row in payload["rows"]] == [
        "promotable",
        "unavailable",
        "already_registered",
    ]
    stdout = capsys.readouterr().out
    assert "| Config | Classification | Decision | Action |" in stdout
    assert "configs/baselines/present.yaml" in stdout


def test_directory_scan_handles_non_mapping_yaml_and_skips_manifests(tmp_path: Path) -> None:
    """Directory scans should report unsupported YAML shapes without scanning manifests as configs."""
    paths = _repo_fixture(tmp_path)

    report = plan_model_artifact_promotion.build_promotion_report(
        [tmp_path / "configs" / "baselines"],
        repo_root=tmp_path,
        registry_path=paths["registry"],
        promoted_surfaces_path=paths["promoted_surfaces"],
    )

    rows = {row["config_path"]: row for row in report["rows"]}
    assert "configs/baselines/local_model_artifact_blocklist.yaml" not in rows
    assert rows["configs/baselines/example_matrix.yaml"]["decision"] == "unsupported_yaml_shape"


def test_initial_issue_config_list_gets_one_row_per_config() -> None:
    """The issue #1764 target list should remain the default planning surface."""
    report = plan_model_artifact_promotion.build_promotion_report(
        plan_model_artifact_promotion.INITIAL_TARGET_CONFIGS
    )

    assert [row["config_path"] for row in report["rows"]] == [
        path.as_posix() for path in plan_model_artifact_promotion.INITIAL_TARGET_CONFIGS
    ]
    assert all(row["claim_boundary"] for row in report["rows"])
    assert {row["classification"] for row in report["rows"]} == {"retired_local_only"}
