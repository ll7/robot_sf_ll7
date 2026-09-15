"""Focused tests for the SREV-06 review-context component (issue #9275)."""

from __future__ import annotations

import ast
import json
import subprocess
import sys
from pathlib import Path

from robot_sf.analysis_workbench.review_context import (
    COMPONENT_ID,
    _percentile,
    descriptor,
    run,
)

FIXTURE_DIR = "tests/fixtures/scenario_review/review_context"

CAMPAIGN = {
    "campaign_id": "camp-t",
    "episodes": [
        {
            "episode_id": "ep-1",
            "seed": 11,
            "config": {"config_id": "cfg-a"},
            "outcome": "success",
            "metrics": {"clearance_m": 1.5, "speed_m_s": 0.8},
        },
        {
            "episode_id": "ep-2",
            "seed": 12,
            "config": {"config_id": "cfg-a"},
            "outcome": "success",
            "metrics": {"clearance_m": 2.5, "speed_m_s": 1.0},
        },
        {
            "episode_id": "ep-3",
            "seed": 13,
            "config": {"config_id": "cfg-b"},
            "outcome": "collision",
            "metrics": {"clearance_m": -0.2},
        },
        {
            "episode_id": "ep-4",
            "seed": 14,
            "config": {"config_id": "cfg-b"},
            "outcome": "success",
            "metrics": {"clearance_m": 2.5, "speed_m_s": 1.2},
        },
    ],
}

SELECTION = {"selected_episode_ids": ["ep-1", "ep-3"]}


def _stage(tmp_path: Path) -> None:
    (tmp_path / "campaign.json").write_text(json.dumps(CAMPAIGN), encoding="utf-8")
    (tmp_path / "selection.json").write_text(json.dumps(SELECTION), encoding="utf-8")


def _request(
    *,
    request_id: str = "t",
    component_id: str = COMPONENT_ID,
    required: tuple[str, ...] = (),
    config_extra: dict | None = None,
    output: str = "out",
    sources: list[dict] | None = None,
) -> dict:
    from robot_sf.analysis_workbench.review_context import (
        component_request_from_dict,
    )

    config: dict = {"campaign_id": "camp-t"}
    config.update(config_extra or {})
    payload = {
        "schema_version": "component-request.v1",
        "request_id": request_id,
        "component_id": component_id,
        "sources": (
            sources
            if sources is not None
            else [
                {"artifact_id": "campaign", "uri": "campaign.json", "format": "campaign-result"},
            ]
        ),
        "output_directory": output,
        "config": config,
        "required_capabilities": list(required),
    }
    return component_request_from_dict(payload)


def _report(tmp_path: Path, output: str = "out") -> dict:
    return json.loads((tmp_path / output / "context-report.json").read_text())


def test_success_reports_denominators_and_tie_percentiles(tmp_path: Path) -> None:
    _stage(tmp_path)
    source_before = (tmp_path / "campaign.json").read_bytes()
    result = run(_request(), base=tmp_path)
    assert result.status == "complete"
    assert result.reason == ""
    assert (tmp_path / "campaign.json").read_bytes() == source_before
    report = _report(tmp_path)
    assert report["denominator"] == 4
    assert report["grain"] == {
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
    assert len(result.artifacts) == 2


def test_repeated_excerpts_never_inflate_counts(tmp_path: Path) -> None:
    _stage(tmp_path)
    first = run(_request(output="out-a"), base=tmp_path)
    payload = dict(CAMPAIGN)
    payload["episodes"] = list(CAMPAIGN["episodes"]) + [dict(CAMPAIGN["episodes"][0])]
    (tmp_path / "campaign.json").write_text(json.dumps(payload), encoding="utf-8")
    second = run(_request(output="out-b"), base=tmp_path)
    assert first.status == second.status == "complete"
    assert _report(tmp_path, "out-a")["denominator"] == 4
    assert _report(tmp_path, "out-b")["denominator"] == 4
    assert "duplicate_episode_excerpt:ep-1" in "; ".join(d["code"] for d in second.diagnostics)


def test_selection_coverage_with_unknown_ids_is_partial(tmp_path: Path) -> None:
    _stage(tmp_path)
    (tmp_path / "selection.json").write_text(
        json.dumps({"selected_episode_ids": ["ep-1", "ghost"]}), encoding="utf-8"
    )
    from robot_sf.analysis_workbench.review_context import (
        component_request_from_dict,
    )

    payload = {
        "schema_version": "component-request.v1",
        "request_id": "t",
        "component_id": COMPONENT_ID,
        "sources": [
            {"artifact_id": "campaign", "uri": "campaign.json", "format": "campaign-result"},
            {"artifact_id": "sel", "uri": "selection.json", "format": "episode-selection"},
        ],
        "output_directory": "out",
        "config": {"campaign_id": "camp-t"},
        "required_capabilities": ["episode-selection"],
    }
    result = run(component_request_from_dict(payload), base=tmp_path)
    assert result.status == "partial"
    assert "unknown_selected_ids:ghost" in result.reason
    report = _report(tmp_path)
    assert report["selection_coverage"] == {
        "selected": 1,
        "denominator": 4,
        "unknown_selected_ids": ["ghost"],
    }


def test_missing_campaign_reference_is_unavailable_context(tmp_path: Path) -> None:
    _stage(tmp_path)
    result = run(_request(config_extra={"campaign_id": "camp-missing"}), base=tmp_path)
    assert result.status == "partial"
    assert "campaign_context_unavailable" in result.reason


def test_corrupt_source_is_failed_or_partial(tmp_path: Path) -> None:
    (tmp_path / "campaign.json").write_text("{not json", encoding="utf-8")
    result = run(_request(), base=tmp_path)
    assert result.status in ("partial", "failed")
    assert result.artifacts == ()


def test_missing_required_capability_is_unavailable(tmp_path: Path) -> None:
    _stage(tmp_path)
    result = run(_request(required=("rvo2-binary",)), base=tmp_path)
    assert result.status == "unavailable"
    assert "missing_required_capabilities" in result.reason


def test_incompatible_component_version_fails(tmp_path: Path) -> None:
    _stage(tmp_path)
    result = run(_request(config_extra={"min_component_version": "2.0.0"}), base=tmp_path)
    assert result.status == "failed"
    assert "incompatible_component_version" in result.reason


def test_output_collision_fails(tmp_path: Path) -> None:
    _stage(tmp_path)
    (tmp_path / "out").mkdir()
    result = run(_request(), base=tmp_path)
    assert result.status == "failed"
    assert "output_collision" in result.reason


def test_unsupported_component_is_unavailable(tmp_path: Path) -> None:
    _stage(tmp_path)
    result = run(_request(component_id="srev99-nope"), base=tmp_path)
    assert result.status == "unavailable"
    assert "unsupported_component" in result.reason


def test_percentile_ties_are_deterministic() -> None:
    assert _percentile([2.5, 2.5], 0.5) == 2.5
    assert _percentile([1.0, 2.0, 3.0, 4.0], 0.5) == 2.5
    assert _percentile([5.0], 0.9) == 5.0


def test_descriptor_declares_capabilities() -> None:
    info = descriptor()
    assert info["component_id"] == COMPONENT_ID
    assert list(info["required_capabilities"]) == ["campaign-result"]
    assert "episode-selection" in info["optional_capabilities"]


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
            imported.update(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and not node.level:
            imported.add(node.module.split(".")[0])
    imported -= {"robot_sf", "__future__", "typing"}
    assert not (imported & {"sim", "planner", "training", "torch"}), imported


def test_cli_produces_report_from_fixture_request() -> None:
    import shutil

    repo = Path(__file__).resolve().parents[2]
    fixture = repo / "tests/fixtures/scenario_review/review_context"
    assert (fixture / "request.json").exists()
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
        payload = json.loads(completed.stdout)
        assert payload["status"] == "complete"
        report = json.loads((repo / output_rel / "context-report.json").read_text())
        assert report["denominator"] == 4
        assert (repo / output_rel / "context-report.html").exists()
    finally:
        shutil.rmtree(repo / output_rel, ignore_errors=True)
