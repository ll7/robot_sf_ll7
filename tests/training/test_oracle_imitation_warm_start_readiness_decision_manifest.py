"""Focused checks for oracle-imitation warm-start readiness decision manifests."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import yaml

from robot_sf.training.oracle_imitation_warm_start_readiness import check_warm_start_readiness
from scripts.validation.check_oracle_imitation_warm_start_readiness import main as check_cli_main
from tests.training.test_oracle_imitation_warm_start_readiness import (
    _ready_manifest,
    _write_manifest,
    _write_training_ready_packet,
)

if TYPE_CHECKING:
    from pathlib import Path


def _write_ready_manifest(tmp_path: Path, *, missing_file: bool = False) -> Path:
    manifest = _ready_manifest(tmp_path)
    if missing_file:
        manifest["baseline_config"] = "configs/does/not/exist.yaml"
    return _write_manifest(tmp_path, manifest)


def test_missing_trace_manifest_provenance_blocks_readiness(tmp_path: Path) -> None:
    """A packet without the durable source-manifest URI remains blocked."""
    manifest_dict = _ready_manifest(tmp_path)
    packet = _write_training_ready_packet(tmp_path)
    payload = yaml.safe_load(packet.read_text(encoding="utf-8"))
    payload["artifact_paths"].pop("trace_source_manifest_uri")
    packet.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    manifest_dict["dataset_launch_packet"] = str(packet)
    manifest = _write_manifest(tmp_path, manifest_dict)

    report = check_warm_start_readiness(manifest)

    assert report["status"] == "blocked"
    assert any(
        blocker.startswith("dataset_launch_packet not training-ready")
        for blocker in report["blockers"]
    )


def test_missing_dataset_provenance_field_is_actionable_blocker(tmp_path: Path) -> None:
    """Missing packet provenance reports the field-level blocker."""

    manifest_dict = _ready_manifest(tmp_path)
    packet = _write_training_ready_packet(tmp_path)
    payload = yaml.safe_load(packet.read_text(encoding="utf-8"))
    payload.pop("generating_commit")
    packet.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    manifest_dict["dataset_launch_packet"] = str(packet)
    manifest = _write_manifest(tmp_path, manifest_dict)

    report = check_warm_start_readiness(manifest)

    assert report["status"] == "blocked"
    assert any(
        blocker
        == (
            "dataset_launch_packet not training-ready: "
            "generating_commit must be a 40-character git SHA"
        )
        for blocker in report["blockers"]
    )


def test_missing_collection_manifest_destination_is_actionable_blocker(tmp_path: Path) -> None:
    """Missing collection output-manifest destination reports the field-level blocker."""

    manifest_dict = _ready_manifest(tmp_path)
    packet = _write_training_ready_packet(tmp_path)
    payload = yaml.safe_load(packet.read_text(encoding="utf-8"))
    payload["collection_roots"].pop("manifest_destination")
    packet.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    manifest_dict["dataset_launch_packet"] = str(packet)
    manifest = _write_manifest(tmp_path, manifest_dict)

    report = check_warm_start_readiness(manifest)

    assert report["status"] == "blocked"
    assert any(
        blocker
        == (
            "dataset_launch_packet not training-ready: "
            "collection_roots.manifest_destination must be a non-empty string"
        )
        for blocker in report["blockers"]
    )


def test_cli_writes_blocked_decision_manifest_when_file_missing(tmp_path: Path) -> None:
    """The CLI writes a stable blocked decision manifest for unmet prerequisites."""
    manifest = _write_ready_manifest(tmp_path, missing_file=True)
    output = tmp_path / "decision.json"

    exit_code = check_cli_main(["--manifest", str(manifest), "--output", str(output)])

    assert exit_code == 1
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["issue"] == 1496
    assert payload["schema"] == "oracle-imitation-warm-start-readiness-decision.v1"
    assert payload["report"]["status"] == "blocked"
    assert payload["report"]["blockers"]


def test_cli_require_ready_keeps_structured_blockers_in_output(tmp_path: Path) -> None:
    """The fail-closed gate must not replace structured blockers with unknown fallback data."""
    manifest = _write_ready_manifest(tmp_path, missing_file=True)
    output = tmp_path / "decision.json"

    exit_code = check_cli_main(
        ["--manifest", str(manifest), "--output", str(output), "--require-ready"]
    )

    assert exit_code == 1
    report = json.loads(output.read_text(encoding="utf-8"))["report"]
    assert report["experiment_id"] == "unit_test_warm_start"
    assert report["prerequisites"]["baseline_config"]["ready"] is False
    assert any(
        blocker.startswith("baseline_config is not an existing file")
        for blocker in report["blockers"]
    )


def test_cli_writes_ready_decision_manifest(tmp_path: Path) -> None:
    """The CLI writes a ready decision manifest when every prerequisite is satisfied."""
    manifest = _write_ready_manifest(tmp_path)
    output = tmp_path / "decision.json"

    exit_code = check_cli_main(["--manifest", str(manifest), "--output", str(output), "--json"])

    assert exit_code == 0
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["schema"] == "oracle-imitation-warm-start-readiness-decision.v1"
    assert payload["report"]["status"] == "ready"
    assert payload["report"]["blockers"] == []


def test_cli_writes_invalid_decision_manifest_for_malformed_input(tmp_path: Path) -> None:
    """Malformed input still writes a schema-compatible invalid decision manifest."""
    manifest_dict = _ready_manifest(tmp_path)
    manifest_dict["schema_version"] = "wrong.v0"
    manifest = _write_manifest(tmp_path, manifest_dict)
    output = tmp_path / "decision.json"

    exit_code = check_cli_main(["--manifest", str(manifest), "--output", str(output)])

    assert exit_code == 2
    report = json.loads(output.read_text(encoding="utf-8"))["report"]
    assert report["status"] == "invalid"
    assert report["error"].startswith("schema_version must be")
