"""Focused tests for the SREV-06 diagnostic review-context component."""

from __future__ import annotations

import ast
import hashlib
import json
import shutil
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

from jsonschema import Draft202012Validator

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
    "execution_status": "native",
    "selected_episode_ids": ["ep-1", "ep-3"],
}


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")


def _stage(tmp_path: Path, campaign: dict | None = None, selection: dict | None = None) -> None:
    _write_json(tmp_path / "campaign.json", campaign or CAMPAIGN)
    _write_json(tmp_path / "selection.json", selection or SELECTION)


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
        "selected": 1,
        "denominator": 4,
        "unknown_selected_ids": ["ghost"],
    }


def test_required_capability_requires_an_actual_source(tmp_path: Path) -> None:
    _stage(tmp_path)
    result = run(_request(tmp_path, required=("episode-selection",)), base=tmp_path)
    assert result.status == "unavailable"
    assert "required_source_family_missing" in result.reason
    assert not (tmp_path / "out").exists()


def test_malformed_selection_is_not_an_empty_selection(tmp_path: Path) -> None:
    _stage(
        tmp_path, selection={"schema_version": "episode-selection.v1", "execution_status": "native"}
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
