"""Focused tests for the SREV-06 diagnostic review-context component."""

from __future__ import annotations

import ast
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import time
from dataclasses import replace
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator

from robot_sf.analysis_workbench import review_context
from robot_sf.analysis_workbench.review_context import (
    COMPONENT_ID,
    OUTPUT_CAPABILITY_FILENAME,
    OUTPUT_REPORT_FILENAME,
    _percentile,
    _result_document,
    descriptor,
    output_schemas,
    run,
)
from robot_sf.analysis_workbench.review_contracts import (
    ComponentResult,
    ReviewContractsValidationError,
    component_descriptor_from_dict,
    component_result_from_dict,
)

FIXTURE_DIR = "tests/fixtures/scenario_review/review_context"
SOURCE_COMMIT = "a" * 40
CONFIG_IDENTITY = "srev06-test-config"

CAMPAIGN = {
    "schema_version": "campaign-result.v1",
    "campaign_id": "camp-t",
    "source_commit": SOURCE_COMMIT,
    "config_identity": CONFIG_IDENTITY,
    "execution_status": "native",
    "episodes": [
        {
            "episode_id": "ep-1",
            "planner": "orca",
            "scenario_id": "crossing",
            "seed": 11,
            "config": {"config_id": "cfg-a"},
            "outcome": "success",
            "metrics": {"clearance_m": 1.5, "speed_m_s": 0.8},
        },
        {
            "episode_id": "ep-2",
            "planner": "orca",
            "scenario_id": "crossing",
            "seed": 12,
            "config": {"config_id": "cfg-a"},
            "outcome": "success",
            "metrics": {"clearance_m": 2.5, "speed_m_s": 1.0},
        },
        {
            "episode_id": "ep-3",
            "planner": "social-force",
            "scenario_id": "doorway",
            "seed": 13,
            "config": {"config_id": "cfg-b"},
            "outcome": "collision",
            "metrics": {"clearance_m": -0.2},
        },
        {
            "episode_id": "ep-4",
            "planner": "social-force",
            "scenario_id": "doorway",
            "seed": 14,
            "config": {"config_id": "cfg-b"},
            "outcome": "success",
            "metrics": {"clearance_m": 2.5, "speed_m_s": 1.2},
        },
    ],
}

SELECTION = {
    "schema_version": "episode-selection.v1",
    "campaign_id": "camp-t",
    "source_commit": SOURCE_COMMIT,
    "config_identity": CONFIG_IDENTITY,
    "execution_status": "native",
    "selected_episode_ids": ["ep-1", "ep-3"],
}


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")


def _stage(tmp_path: Path, campaign: dict | None = None, selection: dict | None = None) -> None:
    _write_json(tmp_path / "campaign.json", campaign or CAMPAIGN)
    _write_json(tmp_path / "selection.json", selection or SELECTION)


def _directory_digest(path: Path) -> str:
    digest = hashlib.sha256()
    for child in sorted(item for item in path.rglob("*") if item.is_file()):
        digest.update(str(child.relative_to(path)).encode("utf-8"))
        digest.update(b"\0")
        digest.update(child.read_bytes())
    return digest.hexdigest()


def _canonical_source_ref(
    store: Path, *, config_identity: str = CONFIG_IDENTITY, artifact_id: str = "campaign-store"
) -> dict[str, str]:
    return {
        "artifact_id": artifact_id,
        "uri": store.name,
        "format": "campaign-result-store",
        "schema": "campaign-result-store.v2",
        "sha256": _directory_digest(store),
        "source_commit": SOURCE_COMMIT,
        "config_identity": config_identity,
    }


def _write_canonical_store(
    tmp_path: Path, rows: list[dict], manifest: dict, *, name: str = "campaign-store"
) -> Path:
    store = tmp_path / name
    store.mkdir()
    rows_text = "\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n"
    (store / "episodes.jsonl").write_text(rows_text, encoding="utf-8")
    _write_json(store / "manifest.json", manifest)
    return store


def _source_ref(
    path: Path,
    *,
    artifact_id: str,
    uri: str | None = None,
    source_format: str = "campaign-result",
    schema: str | None = None,
    sha256: str | None = None,
) -> dict[str, str]:
    schema = schema or f"{source_format}.v1"
    return {
        "artifact_id": artifact_id,
        "uri": uri or path.name,
        "format": source_format,
        "schema": schema,
        "sha256": sha256 or hashlib.sha256(path.read_bytes()).hexdigest(),
        "source_commit": SOURCE_COMMIT,
        "config_identity": CONFIG_IDENTITY,
    }


def _request(
    tmp_path: Path,
    *,
    request_id: str = "t",
    component_id: str = COMPONENT_ID,
    required: tuple[str, ...] = (),
    config_extra: dict | None = None,
    output: str = "out",
    sources: list[dict] | None = None,
) -> object:
    config: dict = {"campaign_id": "camp-t"}
    config.update(config_extra or {})
    payload = {
        "schema_version": "component-request.v1",
        "request_id": request_id,
        "component_id": component_id,
        "sources": sources or [_source_ref(tmp_path / "campaign.json", artifact_id="campaign")],
        "output_directory": output,
        "config": config,
        "required_capabilities": list(required),
    }
    from robot_sf.analysis_workbench.review_context import component_request_from_dict

    return component_request_from_dict(payload)


def _report(tmp_path: Path, output: str = "out") -> dict:
    return json.loads((tmp_path / output / OUTPUT_REPORT_FILENAME).read_text())


def test_success_reports_admitted_grain_and_tie_percentiles(tmp_path: Path) -> None:
    _stage(tmp_path)
    source_before = (tmp_path / "campaign.json").read_bytes()
    result = run(_request(tmp_path), base=tmp_path)
    assert result.status == "complete"
    assert result.reason == ""
    assert (tmp_path / "campaign.json").read_bytes() == source_before
    report = _report(tmp_path)
    assert report["denominator"] == 4
    assert (
        report["source_provenance"][0]["sha256_observed"]
        == hashlib.sha256(source_before).hexdigest()
    )
    assert report["grain"] == {
        "planner_ids": ["orca", "social-force"],
        "scenario_ids": ["crossing", "doorway"],
        "seeds": [11, 12, 13, 14],
        "config_ids": ["cfg-a", "cfg-b"],
        "episodes": 4,
    }
    assert report["outcomes"] == {"collision": 1, "success": 3}
    clearance = report["metrics"]["clearance_m"]
    assert clearance["count"] == 4 and clearance["missing"] == 0
    assert clearance["min"] == -0.2 and clearance["max"] == 2.5
    assert clearance["p50"] == 2.0
    assert clearance["percentile_method"] == "linear-interpolation-on-sorted-values"
    speed = report["metrics"]["speed_m_s"]
    assert speed["count"] == 3 and speed["missing"] == 1 and speed["denominator"] == 4
    assert len(result.artifacts) == 3
    assert {item["artifact_id"] for item in result.artifacts} == {
        OUTPUT_REPORT_FILENAME,
        "context-report.html",
        OUTPUT_CAPABILITY_FILENAME,
    }
    assert result.provenance["source_integrity"] == "digest_and_schema_verified"
    assert result.provenance["sources"][0]["source_commit"] == SOURCE_COMMIT
    assert result.provenance["sources"][0]["config_identity"] == CONFIG_IDENTITY
    rendered = (tmp_path / "out" / "context-report.html").read_text(encoding="utf-8")
    assert "Selection status" in rendered
    assert "Source provenance" in rendered
    assert SOURCE_COMMIT in rendered and CONFIG_IDENTITY in rendered


def test_result_and_descriptor_are_shared_schema_valid(tmp_path: Path) -> None:
    _stage(tmp_path)
    info = descriptor()
    assert component_descriptor_from_dict(info).component_id == COMPONENT_ID
    assert isinstance(info["supported_input_versions"], list)
    assert isinstance(info["output_types"], list)
    result = run(_request(tmp_path), base=tmp_path)
    parsed = component_result_from_dict(_result_document(result))
    assert parsed.status == "complete"
    assert set(output_schemas()) == {"review-context.v1", "missing-capability-report.v1"}
    for document in (
        _report(tmp_path),
        json.loads((tmp_path / "out" / OUTPUT_CAPABILITY_FILENAME).read_text()),
    ):
        schema = output_schemas()[document["schema_version"]]
        assert list(Draft202012Validator(schema).iter_errors(document)) == []


def test_leaf_schema_rejects_untyped_metric_and_provenance_fields(tmp_path: Path) -> None:
    """Leaf payloads reject arbitrary metric/provenance objects."""
    _stage(tmp_path)
    run(_request(tmp_path), base=tmp_path)
    report = _report(tmp_path)
    report["metrics"]["clearance_m"]["unexpected"] = {"not": "a summary field"}
    report["source_provenance"][0]["unexpected"] = "not allowed"
    errors = list(Draft202012Validator(output_schemas()["review-context.v1"]).iter_errors(report))
    assert errors


def test_repeated_excerpts_never_inflate_counts(tmp_path: Path) -> None:
    duplicate = dict(CAMPAIGN)
    duplicate["episodes"] = list(CAMPAIGN["episodes"]) + [dict(CAMPAIGN["episodes"][0])]
    _stage(tmp_path, campaign=duplicate)
    result = run(_request(tmp_path), base=tmp_path)
    assert result.status == "complete"
    assert _report(tmp_path)["denominator"] == 4
    assert any(item["code"] == "duplicate_episode_excerpt" for item in result.diagnostics)


def test_selection_coverage_with_unknown_ids_is_partial(tmp_path: Path) -> None:
    _stage(tmp_path)
    selection = {**SELECTION, "selected_episode_ids": ["ep-1", "ghost"]}
    _write_json(tmp_path / "selection.json", selection)
    sources = [
        _source_ref(tmp_path / "campaign.json", artifact_id="campaign"),
        _source_ref(
            tmp_path / "selection.json",
            artifact_id="selection",
            source_format="episode-selection",
        ),
    ]
    result = run(
        _request(tmp_path, required=("episode-selection",), sources=sources),
        base=tmp_path,
    )
    assert result.status == "partial"
    assert "unknown_selected_ids" in result.reason
    assert _report(tmp_path)["selection_coverage"] == {
        "status": "used",
        "selected": 1,
        "denominator": 4,
        "unknown_selected_ids": ["ghost"],
        "source_artifact_id": "selection",
    }


def test_required_capability_requires_an_actual_source(tmp_path: Path) -> None:
    _stage(tmp_path)
    result = run(_request(tmp_path, required=("episode-selection",)), base=tmp_path)
    assert result.status == "unavailable"
    assert "required_source_family_missing" in result.reason
    assert not (tmp_path / "out").exists()


def test_supplied_optional_selection_is_used_and_reported(tmp_path: Path) -> None:
    """An optional selection ref must affect coverage when it is supplied."""
    _stage(tmp_path)
    sources = [
        _source_ref(tmp_path / "campaign.json", artifact_id="campaign"),
        _source_ref(
            tmp_path / "selection.json",
            artifact_id="selection",
            source_format="episode-selection",
        ),
    ]
    result = run(_request(tmp_path, sources=sources), base=tmp_path)
    assert result.status == "complete"
    assert _report(tmp_path)["selection_coverage"] == {
        "status": "used",
        "selected": 2,
        "denominator": 4,
        "unknown_selected_ids": [],
        "source_artifact_id": "selection",
    }
    capability = json.loads(
        (tmp_path / "out" / OUTPUT_CAPABILITY_FILENAME).read_text(encoding="utf-8")
    )
    assert capability["missing_capabilities"] == []
    assert capability["skipped_optional_streams"] == []


def test_explicit_optional_selection_skip_is_partial_and_reported(tmp_path: Path) -> None:
    """An explicit optional skip cannot silently produce complete selection coverage."""
    _stage(tmp_path)
    sources = [
        _source_ref(tmp_path / "campaign.json", artifact_id="campaign"),
        _source_ref(
            tmp_path / "selection.json",
            artifact_id="selection",
            source_format="episode-selection",
        ),
    ]
    result = run(
        _request(
            tmp_path,
            sources=sources,
            config_extra={"skip_optional_capabilities": ["episode-selection"]},
        ),
        base=tmp_path,
    )
    assert result.status == "partial"
    assert _report(tmp_path)["selection_coverage"] == {
        "status": "skipped",
        "selected": 0,
        "denominator": 4,
        "unknown_selected_ids": [],
        "source_artifact_id": None,
    }
    capability = json.loads(
        (tmp_path / "out" / OUTPUT_CAPABILITY_FILENAME).read_text(encoding="utf-8")
    )
    assert capability["missing_capabilities"] == []
    assert capability["skipped_optional_streams"] == ["episode-selection"]


def test_canonical_result_store_directory_uses_owner_loader(tmp_path: Path) -> None:
    """A canonical JSONL result-store directory is adapted without flattening its contract."""
    store = tmp_path / "campaign-store"
    store.mkdir()
    rows = "\n".join(json.dumps(row, sort_keys=True) for row in CAMPAIGN["episodes"]) + "\n"
    (store / "episodes.jsonl").write_text(rows, encoding="utf-8")
    _write_json(
        store / "manifest.json",
        {
            "schema_version": "campaign-result-store.v2",
            "study_id": "camp-t",
            "source_commit": SOURCE_COMMIT,
            "config_hash": CONFIG_IDENTITY,
        },
    )
    source = _canonical_source_ref(store)
    result = run(_request(tmp_path, sources=[source]), base=tmp_path)
    assert result.status == "complete"
    report = _report(tmp_path)
    assert report["denominator"] == 4
    assert report["source_provenance"][0]["canonical_source"] is True
    assert report["source_provenance"][0]["schema_declared"] == "campaign-result-store.v2"


def test_canonical_owner_config_hash_and_digest_are_independent(tmp_path: Path) -> None:
    """The owner config hash binds identity while config digest remains provenance."""
    config_digest = "resolved-config-digest"
    rows = [
        {
            "episode_id": "owner-success",
            "planner": "orca",
            "scenario_id": "crossing",
            "seed": 11,
            "row_status": "native",
            "execution_status": "success",
            "status": "success",
            "config_hash": CONFIG_IDENTITY,
            "config_digest": config_digest,
            "outcome": {"label": "success"},
            "metrics": {"clearance_m": 1.5},
            "provenance": {
                "git_hash": SOURCE_COMMIT,
                "config_hash": CONFIG_IDENTITY,
                "config_digest": config_digest,
            },
        }
    ]
    store = _write_canonical_store(
        tmp_path,
        rows,
        {"schema_version": "campaign-result-store.v2", "study_id": "camp-t"},
    )

    result = run(_request(tmp_path, sources=[_canonical_source_ref(store)]), base=tmp_path)

    assert result.status == "complete"
    report = _report(tmp_path)
    assert report["grain"]["config_ids"] == [CONFIG_IDENTITY]
    assert report["source_provenance"][0]["config_identity"] == CONFIG_IDENTITY
    assert report["source_provenance"][0]["config_digest"] == config_digest
    assert result.provenance["source_identity"]["config_digests"] == [config_digest]


def test_canonical_owner_preserves_multiple_config_digests_without_rebinding_identity(
    tmp_path: Path,
) -> None:
    """Independent per-row digests remain provenance, not canonical identity."""
    rows = []
    for index, config_digest in enumerate(("planner-a-digest", "planner-b-digest"), start=1):
        rows.append(
            {
                "episode_id": f"owner-{index}",
                "planner": f"planner-{index}",
                "scenario_id": "crossing",
                "seed": index,
                "row_status": "native",
                "config_hash": CONFIG_IDENTITY,
                "config_digest": config_digest,
                "outcome": {"label": "success"},
                "metrics": {"clearance_m": 1.5},
                "provenance": {
                    "git_hash": SOURCE_COMMIT,
                    "config_hash": CONFIG_IDENTITY,
                    "config_digest": config_digest,
                },
            }
        )
    store = _write_canonical_store(
        tmp_path,
        rows,
        {"schema_version": "campaign-result-store.v2", "study_id": "camp-t"},
    )

    result = run(_request(tmp_path, sources=[_canonical_source_ref(store)]), base=tmp_path)

    assert result.status == "complete"
    report_source = _report(tmp_path)["source_provenance"][0]
    assert report_source["config_digest"] is None
    assert report_source["config_digests"] == ["planner-a-digest", "planner-b-digest"]
    assert result.provenance["source_identity"]["config_digests"] == [
        "planner-a-digest",
        "planner-b-digest",
    ]


def test_canonical_owner_export_rows_accept_descriptive_execution_status(
    tmp_path: Path,
) -> None:
    """Owner v2 rows use row_status for execution mode and retain outcome labels."""
    rows = [
        {
            "episode_id": "owner-success",
            "planner": "orca",
            "scenario_id": "crossing",
            "seed": 11,
            "row_status": "native",
            "execution_status": "success",
            "status": "success",
            "config_hash": CONFIG_IDENTITY,
            "outcome": {"label": "success"},
            "metrics": {"clearance_m": 1.5},
            "provenance": {"git_hash": SOURCE_COMMIT, "config_hash": CONFIG_IDENTITY},
        },
        {
            "episode_id": "owner-collision",
            "planner": "orca",
            "scenario_id": "crossing",
            "seed": 12,
            "row_status": "native",
            "execution_status": "collision",
            "status": "collision",
            "config_hash": CONFIG_IDENTITY,
            "outcome": {"label": "collision"},
            "metrics": {"clearance_m": -0.2},
            "provenance": {"git_hash": SOURCE_COMMIT, "config_hash": CONFIG_IDENTITY},
        },
    ]
    store = _write_canonical_store(
        tmp_path,
        rows,
        {"schema_version": "campaign-result-store.v2", "study_id": "camp-t"},
    )
    request = _request(
        tmp_path,
        config_extra={"config_identity": CONFIG_IDENTITY},
        sources=[_canonical_source_ref(store)],
    )

    result = run(request, base=tmp_path)

    assert result.status == "complete"
    assert _report(tmp_path)["outcomes"] == {"collision": 1, "success": 1}


def test_canonical_owner_export_fallback_row_is_not_native(tmp_path: Path) -> None:
    """A descriptive outcome never upgrades an explicitly fallback row."""
    rows = [
        {
            "episode_id": "owner-native",
            "planner": "orca",
            "scenario_id": "crossing",
            "seed": 11,
            "row_status": "native",
            "execution_status": "success",
            "status": "success",
            "config_hash": CONFIG_IDENTITY,
            "outcome": {"label": "success"},
            "provenance": {"git_hash": SOURCE_COMMIT, "config_hash": CONFIG_IDENTITY},
        },
        {
            "episode_id": "owner-fallback",
            "planner": "orca",
            "scenario_id": "crossing",
            "seed": 12,
            "row_status": "fallback",
            "execution_status": "success",
            "status": "success",
            "config_hash": CONFIG_IDENTITY,
            "outcome": {"label": "success"},
            "provenance": {"git_hash": SOURCE_COMMIT, "config_hash": CONFIG_IDENTITY},
        },
    ]
    store = _write_canonical_store(
        tmp_path,
        rows,
        {"schema_version": "campaign-result-store.v2", "study_id": "camp-t"},
    )
    result = run(
        _request(tmp_path, sources=[_canonical_source_ref(store)]),
        base=tmp_path,
    )

    assert result.status == "partial"
    report = _report(tmp_path)
    assert report["denominator"] == 1
    assert report["exclusions"]["by_status"] == {"fallback": 1}


def test_canonical_identity_cannot_be_synthesized_from_request(tmp_path: Path) -> None:
    """A canonical store without owner identity is unavailable, not complete."""
    store = _write_canonical_store(
        tmp_path,
        [dict(CAMPAIGN["episodes"][0])],
        {"schema_version": "campaign-result-store.v2", "study_id": "camp-t"},
    )
    caller_identity = "caller-only-config"
    source = _canonical_source_ref(store, config_identity=caller_identity)
    request = _request(
        tmp_path,
        config_extra={"config_identity": caller_identity},
        sources=[source],
    )

    result = run(request, base=tmp_path)

    assert result.status == "unavailable"
    assert "canonical_identity_unbound" in result.reason
    assert not (tmp_path / "out" / OUTPUT_REPORT_FILENAME).exists()


@pytest.mark.parametrize(
    ("field", "value", "reason_fragment"),
    [
        ("study_id", "camp-mutated", "campaign_context_unavailable"),
        ("config_hash", "mutated-config", "canonical_identity_mismatch"),
    ],
)
def test_canonical_identity_mutation_is_unavailable(
    tmp_path: Path, field: str, value: str, reason_fragment: str
) -> None:
    """Mutated owner identity cannot be hidden by matching caller values."""
    row = {
        **CAMPAIGN["episodes"][0],
        "config_hash": CONFIG_IDENTITY,
        "provenance": {"git_hash": SOURCE_COMMIT, "config_hash": CONFIG_IDENTITY},
    }
    manifest = {"schema_version": "campaign-result-store.v2", "study_id": "camp-t"}
    if field == "study_id":
        manifest[field] = value
    else:
        row["config_hash"] = value
        row["provenance"] = {"git_hash": SOURCE_COMMIT, "config_hash": value}
    store = _write_canonical_store(tmp_path, [row], manifest)

    result = run(
        _request(
            tmp_path,
            sources=[_canonical_source_ref(store)],
            config_extra={"config_identity": CONFIG_IDENTITY},
        ),
        base=tmp_path,
    )

    assert result.status == "unavailable"
    assert reason_fragment in result.reason
    assert not (tmp_path / "out" / OUTPUT_REPORT_FILENAME).exists()


def test_canonical_result_store_rejects_symlink_entries(tmp_path: Path) -> None:
    """Canonical directory adaptation keeps the leaf no-follow boundary."""
    store = tmp_path / "campaign-store"
    store.mkdir()
    rows = "\n".join(json.dumps(row, sort_keys=True) for row in CAMPAIGN["episodes"]) + "\n"
    (store / "episodes.jsonl").write_text(rows, encoding="utf-8")
    outside = tmp_path / "outside.jsonl"
    outside.write_text(rows, encoding="utf-8")
    (store / "outside-link.jsonl").symlink_to(outside)
    source = {
        "artifact_id": "campaign-store",
        "uri": store.name,
        "format": "campaign-result-store",
        "schema": "campaign-result-store.v2",
        "sha256": "0" * 64,
        "source_commit": SOURCE_COMMIT,
        "config_identity": CONFIG_IDENTITY,
    }
    result = run(_request(tmp_path, sources=[source]), base=tmp_path)
    assert result.status == "failed"
    assert "source_unreadable_or_unsafe" in result.reason


def test_canonical_owner_loader_uses_one_snapshot_after_source_replacement(
    tmp_path: Path, monkeypatch
) -> None:
    """Owner path reads use the bounded snapshot, not a replaced source path."""
    rows = [dict(row) for row in CAMPAIGN["episodes"]]
    store = _write_canonical_store(
        tmp_path,
        rows,
        {
            "schema_version": "campaign-result-store.v2",
            "study_id": "camp-t",
            "source_commit": SOURCE_COMMIT,
            "config_hash": CONFIG_IDENTITY,
        },
    )
    source = _canonical_source_ref(store)
    outside = tmp_path / "outside.jsonl"
    outside.write_text(
        json.dumps({**rows[0], "episode_id": "outside"}, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    original_snapshot = review_context._snapshot_directory

    def replace_after_snapshot(directory: Path, files: list[Path], snapshot_root: Path):
        snapshot = original_snapshot(directory, files, snapshot_root)
        source_path = directory / "episodes.jsonl"
        source_path.unlink()
        source_path.symlink_to(outside)
        return snapshot

    monkeypatch.setattr(review_context, "_snapshot_directory", replace_after_snapshot)

    result = run(_request(tmp_path, sources=[source]), base=tmp_path)

    assert result.status == "complete"
    assert _report(tmp_path)["denominator"] == len(rows)


@pytest.mark.skipif(not hasattr(os, "mkfifo"), reason="FIFO support is platform-specific")
def test_canonical_result_store_rejects_fifo_entry_without_blocking(tmp_path: Path) -> None:
    """A canonical FIFO is rejected during bounded inventory without opening it."""
    store = tmp_path / "campaign-store"
    store.mkdir()
    _write_json(
        store / "manifest.json",
        {"schema_version": "campaign-result-store.v2", "study_id": "camp-t"},
    )
    os.mkfifo(store / "episodes.jsonl")
    source = {
        "artifact_id": "campaign-store",
        "uri": store.name,
        "format": "campaign-result-store",
        "schema": "campaign-result-store.v2",
        "sha256": "0" * 64,
        "source_commit": SOURCE_COMMIT,
        "config_identity": CONFIG_IDENTITY,
    }

    result = run(_request(tmp_path, sources=[source]), base=tmp_path)

    assert result.status == "failed"
    assert "source_unreadable_or_unsafe" in result.reason


@pytest.mark.skipif(not hasattr(os, "mkfifo"), reason="FIFO support is platform-specific")
def test_canonical_snapshot_rejects_fifo_replacement_without_blocking(
    tmp_path: Path, monkeypatch
) -> None:
    """A FIFO inserted after inventory cannot block the descriptor-backed snapshot."""
    store = _write_canonical_store(
        tmp_path,
        [dict(CAMPAIGN["episodes"][0])],
        {"schema_version": "campaign-result-store.v2", "study_id": "camp-t"},
    )
    source = _canonical_source_ref(store)
    original_inventory = review_context._directory_files

    def replace_after_inventory(directory: Path) -> list[Path]:
        files = original_inventory(directory)
        episodes_path = directory / "episodes.jsonl"
        episodes_path.unlink()
        os.mkfifo(episodes_path)
        return files

    monkeypatch.setattr(review_context, "_directory_files", replace_after_inventory)
    started = time.monotonic()

    result = run(_request(tmp_path, sources=[source]), base=tmp_path)

    assert time.monotonic() - started < 1.0
    assert result.status == "failed"
    assert "source_unreadable_or_unsafe" in result.reason


def test_cross_source_identity_mismatch_fails_closed(tmp_path: Path) -> None:
    """Campaign and selection sources must share the bound commit/config identity."""
    _stage(tmp_path)
    wrong_commit = "b" * 40
    selection = {**SELECTION, "source_commit": wrong_commit}
    _write_json(tmp_path / "selection.json", selection)
    sources = [
        _source_ref(tmp_path / "campaign.json", artifact_id="campaign"),
        _source_ref(
            tmp_path / "selection.json",
            artifact_id="selection",
            source_format="episode-selection",
        ),
    ]
    request = _request(tmp_path, required=("episode-selection",), sources=sources)
    request = replace(
        request,
        sources=(request.sources[0], replace(request.sources[1], source_commit=wrong_commit)),
    )
    result = run(request, base=tmp_path)
    assert result.status == "failed"
    assert "cross_source_identity_mismatch" in result.reason
    assert not (tmp_path / "out" / OUTPUT_REPORT_FILENAME).exists()


def test_missing_payload_identity_is_rejected(tmp_path: Path) -> None:
    """A source cannot pass by omitting its payload commit/config identity."""
    campaign = {key: value for key, value in CAMPAIGN.items() if key != "source_commit"}
    _stage(tmp_path, campaign=campaign)
    result = run(_request(tmp_path), base=tmp_path)
    assert result.status == "failed"
    assert "source_payload_source_commit_missing" in result.reason


@pytest.mark.parametrize(
    ("alias", "value", "diagnostic"),
    [
        ("git_hash", "b" * 40, "source_commit_mismatch"),
        ("commit_sha", "b" * 40, "source_commit_mismatch"),
        ("commit", "b" * 40, "source_commit_mismatch"),
        ("config_hash", "mutated-config", "config_identity_mismatch"),
        ("config_identity", "mutated-config", "config_identity_mismatch"),
    ],
)
def test_nested_identity_alias_contradictions_are_rejected(
    tmp_path: Path, alias: str, value: str, diagnostic: str
) -> None:
    """Nested owner identity aliases cannot contradict the source declaration."""
    campaign = {
        **CAMPAIGN,
        "provenance": {"git_hash": SOURCE_COMMIT, "config_hash": CONFIG_IDENTITY},
    }
    campaign["provenance"][alias] = value
    _stage(tmp_path, campaign=campaign)

    result = run(_request(tmp_path), base=tmp_path)

    assert result.status == "failed"
    assert diagnostic in result.reason
    assert not (tmp_path / "out" / OUTPUT_REPORT_FILENAME).exists()


@pytest.mark.parametrize(
    ("container", "alias", "value", "diagnostic"),
    [
        ("provenance", "git_hash", "b" * 40, "source_commit_mismatch"),
        ("result_provenance", "config_hash", "foreign-row-config", "config_identity_mismatch"),
        ("cell_context", "git_hash", "b" * 40, "source_commit_mismatch"),
        ("algorithm_metadata", "config_hash", "foreign-row-config", "config_identity_mismatch"),
    ],
)
def test_regular_campaign_rows_bind_nested_identity_aliases_before_parsing(
    tmp_path: Path, container: str, alias: str, value: str, diagnostic: str
) -> None:
    """Nested row identity aliases cannot bypass regular campaign admission."""
    campaign = json.loads(json.dumps(CAMPAIGN))
    row = campaign["episodes"][0]
    if container == "algorithm_metadata":
        row[container] = {"analysis_trace": {alias: value}}
    else:
        row[container] = {alias: value}
    _stage(tmp_path, campaign=campaign)

    result = run(_request(tmp_path), base=tmp_path)

    assert result.status == "failed"
    assert diagnostic in result.reason
    assert result.artifacts == ()
    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize("campaign_alias", ["campaign_id", "study_id"])
def test_regular_campaign_rows_bind_nested_campaign_aliases_before_parsing(
    tmp_path: Path, campaign_alias: str
) -> None:
    """Nested campaign/study aliases cannot smuggle a foreign campaign row."""
    campaign = json.loads(json.dumps(CAMPAIGN))
    campaign["episodes"][0]["provenance"] = {campaign_alias: "camp-foreign"}
    _stage(tmp_path, campaign=campaign)

    result = run(_request(tmp_path), base=tmp_path)

    assert result.status == "failed"
    assert "mixed_campaign_row" in result.reason
    assert not (tmp_path / "out").exists()


def test_output_materialization_never_publishes_partial_final(tmp_path: Path, monkeypatch) -> None:
    """An interrupted atomic write leaves neither a partial final nor temp residue."""
    target = tmp_path / "context-report.json"

    def fail_fsync(_file_descriptor: int) -> None:
        raise OSError("simulated interruption")

    monkeypatch.setattr(review_context.os, "fsync", fail_fsync)
    with pytest.raises(OSError, match="simulated interruption"):
        review_context._write_json(target, {"schema_version": "test"})
    assert not target.exists()
    assert list(tmp_path.glob(".context-report.json.*.partial")) == []


def test_malformed_selection_is_not_an_empty_selection(tmp_path: Path) -> None:
    _stage(
        tmp_path,
        selection={
            "schema_version": "episode-selection.v1",
            "source_commit": SOURCE_COMMIT,
            "config_identity": CONFIG_IDENTITY,
            "execution_status": "native",
        },
    )
    sources = [
        _source_ref(tmp_path / "campaign.json", artifact_id="campaign"),
        _source_ref(
            tmp_path / "selection.json",
            artifact_id="selection",
            source_format="episode-selection",
        ),
    ]
    result = run(
        _request(tmp_path, required=("episode-selection",), sources=sources),
        base=tmp_path,
    )
    assert result.status == "failed"
    assert "invalid_episode_selection" in result.reason
    assert not (tmp_path / "out" / OUTPUT_REPORT_FILENAME).exists()


@pytest.mark.parametrize(
    ("campaign_id", "diagnostic"),
    [(None, "selection_campaign_unbound"), ("camp-other", "selection_campaign_mismatch")],
)
def test_selection_campaign_id_is_required_before_ids_are_interpreted(
    tmp_path: Path, campaign_id: str | None, diagnostic: str
) -> None:
    """Supplied selection IDs are never interpreted without a matching campaign."""
    _stage(tmp_path)
    selection = dict(SELECTION)
    if campaign_id is None:
        selection.pop("campaign_id")
    else:
        selection["campaign_id"] = campaign_id
    _write_json(tmp_path / "selection.json", selection)
    sources = [
        _source_ref(tmp_path / "campaign.json", artifact_id="campaign"),
        _source_ref(
            tmp_path / "selection.json",
            artifact_id="selection",
            source_format="episode-selection",
        ),
    ]

    result = run(
        _request(tmp_path, required=("episode-selection",), sources=sources),
        base=tmp_path,
    )

    assert result.status == "failed"
    assert diagnostic in result.reason
    assert not (tmp_path / "out" / OUTPUT_REPORT_FILENAME).exists()


@pytest.mark.parametrize(
    ("campaign_id", "diagnostic"),
    [(None, "selection_campaign_unbound"), ("camp-other", "selection_campaign_mismatch")],
)
def test_optional_selection_campaign_identity_failure_publishes_no_report(
    tmp_path: Path, campaign_id: str | None, diagnostic: str
) -> None:
    """An invalid supplied optional selection is terminal before report publication."""
    _stage(tmp_path)
    selection = dict(SELECTION)
    if campaign_id is None:
        selection.pop("campaign_id")
    else:
        selection["campaign_id"] = campaign_id
    _write_json(tmp_path / "selection.json", selection)
    sources = [
        _source_ref(tmp_path / "campaign.json", artifact_id="campaign"),
        _source_ref(
            tmp_path / "selection.json",
            artifact_id="selection",
            source_format="episode-selection",
        ),
    ]

    result = run(_request(tmp_path, sources=sources), base=tmp_path)

    assert result.status == "failed"
    assert diagnostic in result.reason
    assert result.artifacts == ()
    assert not (tmp_path / "out").exists()


def test_missing_campaign_reference_is_unavailable_context(tmp_path: Path) -> None:
    _stage(tmp_path)
    result = run(_request(tmp_path, config_extra={"campaign_id": "camp-missing"}), base=tmp_path)
    assert result.status == "unavailable"
    assert "campaign_context_unavailable" in result.reason
    assert not (tmp_path / "out").exists()


def test_no_campaign_reference_is_unavailable_context(tmp_path: Path) -> None:
    _stage(tmp_path)
    result = run(_request(tmp_path, config_extra={"campaign_id": ""}), base=tmp_path)
    assert result.status == "unavailable"
    assert "campaign_context_unavailable" in result.reason


def test_corrupt_source_is_failed_without_report(tmp_path: Path) -> None:
    (tmp_path / "campaign.json").write_text("{not json", encoding="utf-8")
    result = run(_request(tmp_path), base=tmp_path)
    assert result.status == "failed"
    assert result.artifacts == ()
    assert not (tmp_path / "out").exists()


def test_declared_digest_and_provenance_are_admission_requirements(tmp_path: Path) -> None:
    _stage(tmp_path)
    wrong = _source_ref(tmp_path / "campaign.json", artifact_id="campaign", sha256="b" * 64)
    result = run(_request(tmp_path, sources=[wrong]), base=tmp_path)
    assert result.status == "failed"
    assert "source_digest_mismatch" in result.reason
    valid_request = _request(tmp_path, output="out-2")
    missing_provenance = replace(
        valid_request,
        sources=(replace(valid_request.sources[0], source_commit=""),),
    )
    result = run(missing_provenance, base=tmp_path)
    assert result.status == "failed"
    assert "source_commit_missing" in result.reason


def test_source_symlink_escape_is_rejected(tmp_path: Path) -> None:
    _stage(tmp_path)
    outside = tmp_path.parent / f"{tmp_path.name}-outside.json"
    _write_json(outside, CAMPAIGN)
    link = tmp_path / "campaign-link.json"
    link.symlink_to(outside)
    try:
        result = run(
            _request(
                tmp_path,
                sources=[_source_ref(link, artifact_id="campaign", uri=link.name)],
            ),
            base=tmp_path,
        )
        assert result.status == "failed"
        assert "source_unreadable_or_unsafe" in result.reason
    finally:
        outside.unlink(missing_ok=True)


def test_output_parent_symlink_escape_is_rejected(tmp_path: Path) -> None:
    _stage(tmp_path)
    outside = tmp_path.parent / f"{tmp_path.name}-output"
    outside.mkdir()
    link = tmp_path / "output-link"
    link.symlink_to(outside, target_is_directory=True)
    result = run(_request(tmp_path, output="output-link/out"), base=tmp_path)
    assert result.status == "failed"
    assert not (outside / OUTPUT_REPORT_FILENAME).exists()
    outside.rmdir()


def test_output_parent_replacement_during_publication_is_fail_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A reserved output directory replaced by a symlink cannot receive reports."""
    _stage(tmp_path)
    outside = tmp_path.parent / f"{tmp_path.name}-publication-outside"
    outside.mkdir()
    original_write_json = review_context._write_json
    swapped = False

    def replace_parent_then_write(path: Path, payload: dict, **kwargs: object) -> str:
        nonlocal swapped
        if not swapped:
            path.parent.rmdir()
            path.parent.symlink_to(outside, target_is_directory=True)
            swapped = True
        return original_write_json(path, payload, **kwargs)

    monkeypatch.setattr(review_context, "_write_json", replace_parent_then_write)
    try:
        result = run(_request(tmp_path), base=tmp_path)
        assert result.status == "failed"
        assert "output directory replaced during publication" in result.reason
        assert result.artifacts == ()
        assert not any(outside.iterdir())
    finally:
        output_link = tmp_path / "out"
        if output_link.is_symlink():
            output_link.unlink()
        outside.rmdir()


def test_canonical_source_selection_never_silently_merges(tmp_path: Path) -> None:
    _stage(tmp_path)
    other = {**CAMPAIGN, "campaign_id": "camp-other"}
    _write_json(tmp_path / "campaign-other.json", other)
    sources = [
        _source_ref(tmp_path / "campaign.json", artifact_id="campaign-a"),
        _source_ref(tmp_path / "campaign-other.json", artifact_id="campaign-b"),
    ]
    result = run(_request(tmp_path, sources=sources), base=tmp_path)
    assert result.status == "failed"
    assert "canonical_campaign_source_unavailable" in result.reason
    selected = run(
        _request(
            tmp_path,
            output="selected",
            sources=sources,
            config_extra={"canonical_campaign_artifact_id": "campaign-a"},
        ),
        base=tmp_path,
    )
    assert selected.status == "complete"
    assert _report(tmp_path, "selected")["denominator"] == 4


def test_conflicting_duplicate_episode_fails_closed(tmp_path: Path) -> None:
    conflict = dict(CAMPAIGN)
    conflict["episodes"] = list(CAMPAIGN["episodes"]) + [
        {**CAMPAIGN["episodes"][0], "metrics": {"clearance_m": 99.0}}
    ]
    _stage(tmp_path, campaign=conflict)
    result = run(_request(tmp_path), base=tmp_path)
    assert result.status == "failed"
    assert "conflicting_duplicate_episode" in result.reason
    assert not (tmp_path / "out" / OUTPUT_REPORT_FILENAME).exists()


def test_mixed_campaign_row_fails_closed(tmp_path: Path) -> None:
    mixed = dict(CAMPAIGN)
    mixed["episodes"] = list(CAMPAIGN["episodes"]) + [
        {**CAMPAIGN["episodes"][0], "episode_id": "ep-other", "campaign_id": "camp-other"}
    ]
    _stage(tmp_path, campaign=mixed)
    result = run(_request(tmp_path), base=tmp_path)
    assert result.status == "failed"
    assert "mixed_campaign_row" in result.reason


def test_malformed_campaign_row_is_failed_not_empty_unavailable(tmp_path: Path) -> None:
    """Malformed source rows are distinct from a valid empty campaign."""
    malformed = {**CAMPAIGN, "episodes": [None]}
    _stage(tmp_path, campaign=malformed)

    result = run(_request(tmp_path), base=tmp_path)

    assert result.status == "failed"
    assert "malformed_campaign_source" in result.reason
    assert "episode_row_malformed" in result.reason
    assert not (tmp_path / "out" / OUTPUT_REPORT_FILENAME).exists()


def test_fallback_and_degraded_rows_are_excluded_from_denominator(tmp_path: Path) -> None:
    fallback = {
        **CAMPAIGN,
        "episodes": list(CAMPAIGN["episodes"])
        + [
            {
                **CAMPAIGN["episodes"][0],
                "episode_id": "ep-fallback",
                "outcome": "success",
                "row_status": "fallback",
            }
        ],
    }
    _stage(tmp_path, campaign=fallback)
    result = run(_request(tmp_path), base=tmp_path)
    assert result.status == "partial"
    report = _report(tmp_path)
    assert report["denominator"] == 4
    assert report["outcomes"] == {"collision": 1, "success": 3}
    assert report["exclusions"] == {
        "count": 1,
        "by_status": {"fallback": 1},
        "rows": [
            {
                "episode_id": "ep-fallback",
                "status": "fallback",
                "reason": "non_admissible_execution_status",
            }
        ],
    }
    rendered = (tmp_path / "out" / "context-report.html").read_text()
    assert "fallback" in rendered
    assert "non_admissible_execution_status" in rendered


def test_huge_numeric_value_is_a_stable_partial_result(tmp_path: Path) -> None:
    huge = {
        **CAMPAIGN,
        "episodes": [{**CAMPAIGN["episodes"][0], "metrics": {"clearance_m": 10**400}}],
    }
    _stage(tmp_path, campaign=huge)
    result = run(_request(tmp_path), base=tmp_path)
    assert result.status == "partial"
    assert "metric_value_invalid" in result.reason
    assert _report(tmp_path)["denominator"] == 1


def test_in_memory_config_rejects_non_finite_numbers(tmp_path: Path) -> None:
    """The API gate rejects NaN before a config can reach output or source logic."""
    _stage(tmp_path)
    request = _request(tmp_path)
    request = replace(request, config={**request.config, "non_finite": math.nan})

    result = run(request, base=tmp_path)

    assert result.status == "failed"
    assert "request config_non_finite_number" in result.reason
    assert not (tmp_path / "out").exists()


def test_canonical_jsonl_rejects_non_finite_numbers_before_owner_loader(
    tmp_path: Path,
) -> None:
    """Canonical JSONL NaN is rejected before the owner loader's permissive parser."""
    store = tmp_path / "campaign-store"
    store.mkdir()
    row = {
        "episode_id": "owner-nan",
        "planner": "orca",
        "scenario_id": "crossing",
        "seed": 11,
        "row_status": "native",
        "config_hash": CONFIG_IDENTITY,
        "outcome": {"label": "success"},
        "metrics": {"clearance_m": math.nan},
        "provenance": {"git_hash": SOURCE_COMMIT, "config_hash": CONFIG_IDENTITY},
    }
    (store / "episodes.jsonl").write_text(
        json.dumps(row, allow_nan=True, sort_keys=True) + "\n", encoding="utf-8"
    )
    _write_json(
        store / "manifest.json",
        {
            "schema_version": "campaign-result-store.v2",
            "study_id": "camp-t",
            "source_commit": SOURCE_COMMIT,
            "config_hash": CONFIG_IDENTITY,
        },
    )

    result = run(
        _request(tmp_path, sources=[_canonical_source_ref(store)]),
        base=tmp_path,
    )

    assert result.status == "failed"
    assert any("non_finite_number" in item.get("detail", "") for item in result.diagnostics)
    assert not (tmp_path / "out" / OUTPUT_REPORT_FILENAME).exists()


def test_result_document_rejects_non_finite_provenance() -> None:
    """The shared result serialization boundary never returns NaN/Infinity."""
    result = ComponentResult(
        request_id="t",
        component_id=COMPONENT_ID,
        status="failed",
        provenance={"non_finite": math.inf},
    )

    with pytest.raises(ReviewContractsValidationError) as error:
        _result_document(result)

    assert any("result_non_finite_number" in item for item in error.value.errors)


def test_cli_result_serializer_falls_back_to_strict_failure() -> None:
    """An invalid result is replaced before the CLI can emit non-finite JSON."""
    invalid = ComponentResult(
        request_id="t",
        component_id=COMPONENT_ID,
        status="failed",
        provenance={"non_finite": math.nan},
    )

    serialized_result, serialized = review_context._serialized_cli_result(invalid, {})

    assert serialized_result.status == "failed"
    assert "result_serialization_error" in serialized_result.reason
    assert "NaN" not in serialized and "Infinity" not in serialized
    assert component_result_from_dict(json.loads(serialized)).status == "failed"


def test_output_parent_file_is_a_stable_failure(tmp_path: Path) -> None:
    _stage(tmp_path)
    (tmp_path / "parent").write_text("not a directory", encoding="utf-8")
    result = run(_request(tmp_path, output="parent/out"), base=tmp_path)
    assert result.status == "failed"
    assert "output directory" in result.reason


def test_direct_malformed_api_call_returns_result_envelope() -> None:
    result = run({})
    assert result.status == "failed"
    assert "invalid_request" in result.reason


def test_missing_required_capability_is_unavailable(tmp_path: Path) -> None:
    _stage(tmp_path)
    result = run(_request(tmp_path, required=("rvo2-binary",)), base=tmp_path)
    assert result.status == "unavailable"
    assert "missing_required_capabilities" in result.reason


def test_incompatible_component_version_fails(tmp_path: Path) -> None:
    _stage(tmp_path)
    result = run(_request(tmp_path, config_extra={"min_component_version": "2.0.0"}), base=tmp_path)
    assert result.status == "failed"
    assert "incompatible_component_version" in result.reason


def test_output_collision_fails(tmp_path: Path) -> None:
    _stage(tmp_path)
    (tmp_path / "out").mkdir()
    result = run(_request(tmp_path), base=tmp_path)
    assert result.status == "failed"
    assert "output_collision" in result.reason


def test_unsupported_component_is_unavailable(tmp_path: Path) -> None:
    _stage(tmp_path)
    result = run(_request(tmp_path, component_id="srev99-nope"), base=tmp_path)
    assert result.status == "unavailable"
    assert "unsupported_component" in result.reason


def test_percentile_ties_are_deterministic() -> None:
    assert _percentile([2.5, 2.5], 0.5) == 2.5
    assert _percentile([1.0, 2.0, 3.0, 4.0], 0.5) == 2.5
    assert _percentile([5.0], 0.9) == 5.0


def test_module_touches_no_simulator_paths() -> None:
    """The adapter must stay observational: no sim/planner imports."""
    tree = ast.parse(
        (
            Path(__file__).resolve().parents[2] / "robot_sf/analysis_workbench/review_context.py"
        ).read_bytes()
    )
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and not node.level:
            imported.add(node.module.split(".")[0])
    imported -= {"robot_sf", "__future__", "typing"}
    assert not (imported & {"sim", "planner", "training", "torch"}), imported


def test_cli_produces_schema_valid_report_from_fixture_request() -> None:
    repo = Path(__file__).resolve().parents[2]
    fixture = repo / FIXTURE_DIR
    output_rel = "output/scenario_review/srev-06-cli-test"
    shutil.rmtree(repo / output_rel, ignore_errors=True)
    try:
        completed = subprocess.run(
            [
                sys.executable,
                "-m",
                "robot_sf.analysis_workbench.review_context",
                "--input",
                str(fixture / "request.json"),
                "--config",
                str(fixture / "config.json"),
                "--output",
                output_rel,
            ],
            capture_output=True,
            text=True,
            cwd=repo,
            check=False,
        )
        assert completed.returncode == 0, completed.stderr[-2000:]
        envelope = json.loads(completed.stdout)
        assert component_result_from_dict(envelope).status == "complete"
        report = json.loads((repo / output_rel / OUTPUT_REPORT_FILENAME).read_text())
        assert report["denominator"] == 4
        assert (repo / output_rel / "context-report.html").exists()
        assert (repo / output_rel / OUTPUT_CAPABILITY_FILENAME).exists()
    finally:
        shutil.rmtree(repo / output_rel, ignore_errors=True)


def test_cli_malformed_request_prints_failed_result_envelope(tmp_path: Path) -> None:
    request_path = tmp_path / "request.json"
    request_path.write_text("{not json", encoding="utf-8")
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "robot_sf.analysis_workbench.review_context",
            "--input",
            str(request_path),
            "--output",
            "out",
        ],
        capture_output=True,
        text=True,
        cwd=Path(__file__).resolve().parents[2],
        check=False,
    )
    assert completed.returncode == 1
    parsed = component_result_from_dict(json.loads(completed.stdout))
    assert parsed.status == "failed"
    assert "invalid_request" in parsed.reason


def test_cli_rejects_oversized_control_document(tmp_path: Path) -> None:
    """The CLI bounds request bytes before parsing or invoking the component."""
    request_path = tmp_path / "request.json"
    request_path.write_bytes(b"{" + b'"padding":"' + b"a" * (1024 * 1024) + b'"}')
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "robot_sf.analysis_workbench.review_context",
            "--input",
            str(request_path),
            "--output",
            "out",
        ],
        capture_output=True,
        text=True,
        cwd=Path(__file__).resolve().parents[2],
        check=False,
    )
    assert completed.returncode == 1
    parsed = component_result_from_dict(json.loads(completed.stdout))
    assert parsed.status == "failed"
    assert "control document too large" in parsed.reason


@pytest.mark.skipif(not hasattr(os, "mkfifo"), reason="FIFO support is platform-specific")
@pytest.mark.parametrize("fifo_option", ["--input", "--config"])
def test_cli_rejects_fifo_control_document_without_blocking(
    tmp_path: Path, fifo_option: str
) -> None:
    """FIFO and special control inputs fail promptly with a result envelope."""
    repo = Path(__file__).resolve().parents[2]
    fifo_path = tmp_path / "control.fifo"
    os.mkfifo(fifo_path)
    command = [
        sys.executable,
        "-m",
        "robot_sf.analysis_workbench.review_context",
        "--input",
        str(fifo_path) if fifo_option == "--input" else str(repo / FIXTURE_DIR / "request.json"),
    ]
    if fifo_option == "--config":
        command.extend(["--config", str(fifo_path)])
    command.extend(["--output", "fifo-output"])

    completed = subprocess.run(
        command,
        capture_output=True,
        text=True,
        cwd=repo,
        check=False,
        timeout=5,
    )

    assert completed.returncode == 1
    parsed = component_result_from_dict(json.loads(completed.stdout))
    assert parsed.status == "failed"
    assert "not a regular file" in parsed.reason
