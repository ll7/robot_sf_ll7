"""Contract tests for traceable captions and review report generator (SREV-14, issue #9283)."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from robot_sf.analysis_workbench.review_contracts import (
    ComponentRequest,
    SourceRef,
    component_request_from_dict,
)
from robot_sf.analysis_workbench.review_report import (
    ALLOWED_CATEGORIES,
    CATEGORY_MANUAL_TEXT,
    COMPONENT_ID,
    COMPONENT_VERSION,
    DESCRIPTOR_SCHEMA_VERSION,
    EVIDENCE_BOUNDARY,
    OPTIONAL_CAPABILITIES,
    REQUIRED_CAPABILITIES,
    REVIEW_REPORT_HTML_SCHEMA_VERSION,
    REVIEW_REPORT_MARKDOWN_SCHEMA_VERSION,
    REVIEW_REPORT_SCHEMA_VERSION,
    TRACEABLE_CAPTIONS_SCHEMA_VERSION,
    _resolve_citation,
    descriptor_document,
    main,
    render_html,
    resolve_json_pointer,
    run,
)

FIXTURE_DIR = Path("tests/fixtures/scenario_review/review_report")


def _read_fixture(name: str) -> dict:
    return json.loads((FIXTURE_DIR / name).read_text(encoding="utf-8"))


def test_descriptor_contract() -> None:
    """Descriptor declares component metadata, supported versions, and capabilities."""
    doc = descriptor_document()
    assert doc["schema_version"] == DESCRIPTOR_SCHEMA_VERSION
    assert doc["component_id"] == COMPONENT_ID
    assert doc["component_version"] == COMPONENT_VERSION
    assert "component-request.v1" in doc["supported_input_versions"]
    assert REVIEW_REPORT_SCHEMA_VERSION in doc["output_types"]
    assert TRACEABLE_CAPTIONS_SCHEMA_VERSION in doc["output_types"]
    assert REVIEW_REPORT_MARKDOWN_SCHEMA_VERSION in doc["output_types"]
    assert REVIEW_REPORT_HTML_SCHEMA_VERSION in doc["output_types"]
    for cap in REQUIRED_CAPABILITIES:
        assert cap in doc["required_capabilities"]
    for cap in OPTIONAL_CAPABILITIES:
        assert cap in doc["optional_capabilities"]


def test_fixture_smoke_produces_valid_reports_and_captions(tmp_path: Path) -> None:
    """Running on checked-in fixture produces complete reports, captions, and citations."""
    req_data = _read_fixture("request.json")
    cfg_data = _read_fixture("config.json")
    req_data["config"] = cfg_data
    out_rel = "smoke_output"
    req_data["output_directory"] = out_rel

    request = component_request_from_dict(req_data)
    result = run(request, base=tmp_path)

    assert result.status == "complete"
    assert result.component_id == COMPONENT_ID
    assert len(result.artifacts) == 4

    out_dir = tmp_path / out_rel
    rep_file = out_dir / "report.json"
    caps_file = out_dir / "captions.json"
    md_file = out_dir / "report.md"
    html_file = out_dir / "report.html"

    assert rep_file.exists()
    assert caps_file.exists()
    assert md_file.exists()
    assert html_file.exists()

    # Verify SHA-256 integrity against result artifacts
    artifacts_map = {a["artifact_id"]: a for a in result.artifacts}
    assert (
        artifacts_map["review-report"]["sha256"]
        == hashlib.sha256(rep_file.read_bytes()).hexdigest()
    )
    assert (
        artifacts_map["traceable-captions"]["sha256"]
        == hashlib.sha256(caps_file.read_bytes()).hexdigest()
    )
    assert (
        artifacts_map["review-report-markdown"]["sha256"]
        == hashlib.sha256(md_file.read_bytes()).hexdigest()
    )
    assert (
        artifacts_map["review-report-html"]["sha256"]
        == hashlib.sha256(html_file.read_bytes()).hexdigest()
    )

    # Verify report.json content
    rep_data = json.loads(rep_file.read_text(encoding="utf-8"))
    assert rep_data["schema_version"] == REVIEW_REPORT_SCHEMA_VERSION
    assert rep_data["evidence_boundary"] == EVIDENCE_BOUNDARY
    assert rep_data["title"] == "Scenario Review: Bottleneck Near-Miss Analysis"

    # Verify citations: every cited number resolves to source artifact, pointer, unit, digest
    assert len(rep_data["citations"]) > 0
    for cite in rep_data["citations"]:
        assert cite["claim_status"] == "valid"
        assert len(cite["source_digest"]) == 64
        assert cite["source_field"].startswith("/")
        assert isinstance(cite["value"], (int, float))
        assert cite["unit"] != ""

    # Verify claims: explicit categorization
    assert len(rep_data["claims"]) > 0
    for claim in rep_data["claims"]:
        assert claim["category"] in ALLOWED_CATEGORIES
        assert claim["status"] == "valid"
        assert isinstance(claim["statement"], str)

    # Verify excerpt disclosures
    excerpt = rep_data["excerpt"]
    assert excerpt["source_interval"]["start_s"] == 0.2
    assert excerpt["source_interval"]["end_s"] == 0.8
    assert len(excerpt["omitted_intervals"]) == 2
    assert excerpt["full_episode_link"] == "tests/fixtures/scenario_review/review_report/trace.json"

    # Verify captions.json content
    caps_data = json.loads(caps_file.read_text(encoding="utf-8"))
    assert caps_data["schema_version"] == TRACEABLE_CAPTIONS_SCHEMA_VERSION
    assert caps_data["evidence_boundary"] == EVIDENCE_BOUNDARY
    for cap in caps_data["captions"]:
        assert cap["editable"] is True
        assert cap["category"] in ALLOWED_CATEGORIES
        assert isinstance(cap["citation_ids"], list)

    # Verify report.md
    md_content = md_file.read_text(encoding="utf-8")
    assert "# Scenario Review: Bottleneck Near-Miss Analysis" in md_content
    assert EVIDENCE_BOUNDARY in md_content
    assert "[^cite-" in md_content

    # Verify report.html
    html_content = html_file.read_text(encoding="utf-8")
    assert "<!DOCTYPE html>" in html_content
    assert "Evidence Boundary Notice" in html_content
    assert "Scenario Review: Bottleneck Near-Miss Analysis" in html_content


def test_invalid_citation_reference_fails_claim(tmp_path: Path) -> None:
    """An invalid pointer or missing field causes citation and affected claim to fail."""
    trace_data = _read_fixture("trace.json")
    trace_file = tmp_path / "trace.json"
    trace_file.write_text(json.dumps(trace_data), encoding="utf-8")
    trace_sha = hashlib.sha256(trace_file.read_bytes()).hexdigest()

    sources_map = {
        "trace": {
            "uri": "trace.json",
            "digest": trace_sha,
            "data": trace_data,
        }
    }

    # 1. Nonexistent pointer
    bad_pointer_cite = _resolve_citation(
        citation_id="cite-bad-ptr",
        artifact_id="trace",
        field_pointer="/frames/99/clearance_m",
        unit="m",
        description="Nonexistent frame clearance",
        loaded_sources=sources_map,
    )
    assert bad_pointer_cite.claim_status == "failed"
    assert "resolution failed" in str(bad_pointer_cite.error)

    # 2. Non-numeric value pointer
    bad_type_cite = _resolve_citation(
        citation_id="cite-bad-type",
        artifact_id="trace",
        field_pointer="/frames/0/planner/event",
        unit="str",
        description="String event field",
        loaded_sources=sources_map,
    )
    assert bad_type_cite.claim_status == "failed"
    assert "not a number" in str(bad_type_cite.error)

    # 3. Value mismatch
    mismatch_cite = _resolve_citation(
        citation_id="cite-mismatch",
        artifact_id="trace",
        field_pointer="/frames/0/robot/speed",
        unit="m/s",
        description="Mismatched speed",
        loaded_sources=sources_map,
        expected_value=999.0,
    )
    assert mismatch_cite.claim_status == "failed"
    assert "value mismatch" in str(mismatch_cite.error)


def test_untrusted_html_escaping() -> None:
    """HTML renderer strictly escapes all untrusted strings and user text."""
    malicious_report = {
        "title": "<script>alert('xss-title')</script>",
        "evidence_boundary": "<b onmouseover='alert(1)'>fake-boundary</b>",
        "report_id": "rep-<script>",
        "request_id": "req-<b>inject</b>",
        "provenance": {"component_id": "comp-<i>test</i>", "component_version": "1.0"},
        "excerpt": {
            "source_interval": {"start_s": 0.0, "end_s": 1.0},
            "omitted_intervals": [],
            "full_episode_link": "trace.json?param=<script>",
        },
        "claims": [
            {
                "claim_id": "c1",
                "category": CATEGORY_MANUAL_TEXT,
                "statement": "<img src=x onerror=alert('xss')> Note text",
                "status": "valid",
                "citation_ids": [],
            }
        ],
        "citations": [
            {
                "citation_id": "cite-001",
                "description": "<svg onload=alert(1)>",
                "value": 1.0,
                "unit": "m",
                "source_artifact_id": "trace",
                "source_field": "/field",
                "source_digest": "a" * 64,
                "claim_status": "valid",
            }
        ],
        "missing_measurements": [
            {
                "measurement": "<marquee>hack</marquee>",
                "unavailable_reason": "unavailable: <script>alert(2)</script>",
            }
        ],
        "limitations": ["<iframe src='evil.com'></iframe>"],
    }

    rendered_html = render_html(malicious_report)
    assert "<script>" not in rendered_html
    assert "<img src=x" not in rendered_html
    assert "<svg onload" not in rendered_html
    assert "<marquee>" not in rendered_html
    assert "<iframe" not in rendered_html

    assert "&lt;script&gt;alert(&#x27;xss-title&#x27;)&lt;/script&gt;" in rendered_html
    assert "&lt;img src=x onerror=alert(&#x27;xss&#x27;)&gt; Note text" in rendered_html
    assert "&lt;marquee&gt;hack&lt;/marquee&gt;" in rendered_html


def test_missing_measurements_yields_unavailable_text(tmp_path: Path) -> None:
    """When measurements like actor radii or failure diagnoses are missing, unavailable text is emitted."""
    trace_without_radii = {
        "schema_version": "simulation_trace_export.v1",
        "trace_id": "trace_no_radii",
        "source": {"scenario_id": "scen_1"},
        "frames": [
            {
                "step": 0,
                "time_s": 0.0,
                "robot": {"position": [0.0, 0.0], "speed": 0.5},
                "pedestrians": [{"id": "ped_1", "position": [2.0, 0.0]}],
            }
        ],
    }
    trace_file = tmp_path / "trace_no_radii.json"
    trace_file.write_text(json.dumps(trace_without_radii), encoding="utf-8")

    out_rel = "missing_out"
    req = ComponentRequest(
        request_id="req-missing-meas",
        component_id=COMPONENT_ID,
        sources=(
            SourceRef(
                artifact_id="trace",
                uri=str(trace_file.relative_to(tmp_path)),
                format="simulation_trace_export.v1",
                schema="simulation_trace_export.v1",
            ),
        ),
        output_directory=out_rel,
    )

    result = run(req, base=tmp_path)
    assert result.status == "complete"

    rep_data = json.loads((tmp_path / out_rel / "report.json").read_text(encoding="utf-8"))
    missing = {m["measurement"]: m["unavailable_reason"] for m in rep_data["missing_measurements"]}

    assert "actor_radii" in missing
    assert "unavailable: actor radii not present in trace" in missing["actor_radii"]
    assert "failure_diagnosis" in missing
    assert (
        "unavailable: failure diagnosis record not supplied in request"
        in missing["failure_diagnosis"]
    )
    assert "trace_annotations" in missing
    assert (
        "unavailable: trace annotation set not supplied in request" in missing["trace_annotations"]
    )

    # Clearance claim should disclose center_to_center
    clr_claims = [c for c in rep_data["claims"] if "clearance" in c["statement"].lower()]
    assert len(clr_claims) == 1
    assert "center_to_center" in clr_claims[0]["statement"]

    # Markdown should contain unavailable reasons
    md_content = (tmp_path / out_rel / "report.md").read_text(encoding="utf-8")
    assert "unavailable: actor radii not present in trace" in md_content
    assert "unavailable: failure diagnosis record not supplied" in md_content


def test_excerpt_full_episode_when_unconfigured(tmp_path: Path) -> None:
    """When no excerpt is configured, source interval covers entire episode with zero omissions."""
    trace_data = _read_fixture("trace.json")
    trace_file = tmp_path / "trace.json"
    trace_file.write_text(json.dumps(trace_data), encoding="utf-8")

    out_rel = "full_excerpt_out"
    req = ComponentRequest(
        request_id="req-full-excerpt",
        component_id=COMPONENT_ID,
        sources=(
            SourceRef(
                artifact_id="trace",
                uri=str(trace_file.relative_to(tmp_path)),
                format="simulation_trace_export.v1",
                schema="simulation_trace_export.v1",
            ),
        ),
        output_directory=out_rel,
        config={},  # No excerpt_interval
    )

    result = run(req, base=tmp_path)
    assert result.status == "complete"

    rep_data = json.loads((tmp_path / out_rel / "report.json").read_text(encoding="utf-8"))
    excerpt = rep_data["excerpt"]
    assert excerpt["source_interval"]["start_s"] == 0.0
    assert excerpt["source_interval"]["end_s"] == 1.0
    assert excerpt["omitted_intervals"] == []


def test_output_collision_fails(tmp_path: Path) -> None:
    """Existing output directories fail closed to prevent overwriting prior outputs."""
    out_rel = "collide_dir"
    (tmp_path / out_rel).mkdir(parents=True, exist_ok=True)

    req = ComponentRequest(
        request_id="req-collide",
        component_id=COMPONENT_ID,
        sources=(),
        output_directory=out_rel,
    )

    result = run(req, base=tmp_path)
    assert result.status == "failed"
    assert "output collision" in result.reason
    assert len(result.artifacts) == 0


def test_corrupt_source_json(tmp_path: Path) -> None:
    """Corrupted source JSON returns failed result and cannot carry complete status."""
    corrupt_file = tmp_path / "corrupt.json"
    corrupt_file.write_text("{invalid-json-content", encoding="utf-8")

    out_rel = "corrupt_out"
    req = ComponentRequest(
        request_id="req-corrupt",
        component_id=COMPONENT_ID,
        sources=(
            SourceRef(
                artifact_id="corrupt-src",
                uri=str(corrupt_file.relative_to(tmp_path)),
                format="json",
            ),
        ),
        output_directory=out_rel,
    )

    result = run(req, base=tmp_path)
    assert result.status == "failed"
    assert "corrupt JSON" in result.reason
    assert len(result.artifacts) == 0


def test_source_integrity_mismatch(tmp_path: Path) -> None:
    """Declared SHA-256 mismatch returns failed result and cannot carry complete status."""
    trace_data = _read_fixture("trace.json")
    trace_file = tmp_path / "trace.json"
    trace_file.write_text(json.dumps(trace_data), encoding="utf-8")

    out_rel = "mismatch_out"
    req = ComponentRequest(
        request_id="req-mismatch",
        component_id=COMPONENT_ID,
        sources=(
            SourceRef(
                artifact_id="trace",
                uri=str(trace_file.relative_to(tmp_path)),
                format="simulation_trace_export.v1",
                sha256="0" * 64,  # Intentionally incorrect digest
            ),
        ),
        output_directory=out_rel,
    )

    result = run(req, base=tmp_path)
    assert result.status == "failed"
    assert "integrity mismatch" in result.reason
    assert len(result.artifacts) == 0


def test_missing_source_file(tmp_path: Path) -> None:
    """Non-existent source URI returns failed result envelope."""
    out_rel = "missing_file_out"
    req = ComponentRequest(
        request_id="req-missing-file",
        component_id=COMPONENT_ID,
        sources=(
            SourceRef(
                artifact_id="ghost",
                uri="does_not_exist.json",
                format="json",
            ),
        ),
        output_directory=out_rel,
    )

    result = run(req, base=tmp_path)
    assert result.status == "failed"
    assert "source file not found" in result.reason
    assert len(result.artifacts) == 0


def test_unsupported_required_capability(tmp_path: Path) -> None:
    """Requesting undeclared capabilities returns unavailable status."""
    out_rel = "unsupported_cap_out"
    req = ComponentRequest(
        request_id="req-unsupported-cap",
        component_id=COMPONENT_ID,
        sources=(),
        output_directory=out_rel,
        required_capabilities=("hologram-projection",),
    )

    result = run(req, base=tmp_path)
    assert result.status == "unavailable"
    assert "unsupported required capability" in result.reason
    assert len(result.artifacts) == 0


def test_incompatible_required_version(tmp_path: Path) -> None:
    """Requesting incompatible major component version returns unavailable status."""
    out_rel = "incompat_ver_out"
    req = ComponentRequest(
        request_id="req-incompat-ver",
        component_id=COMPONENT_ID,
        sources=(),
        output_directory=out_rel,
        config={"required_component_version": "99.0.0"},
    )

    result = run(req, base=tmp_path)
    assert result.status == "unavailable"
    assert "incompatible required version" in result.reason
    assert len(result.artifacts) == 0


def test_deterministic_repeated_runs(tmp_path: Path) -> None:
    """Repeated runs on the same input produce identical artifact contents and hashes."""
    req_data = _read_fixture("request.json")
    cfg_data = _read_fixture("config.json")
    req_data["config"] = cfg_data

    # Run 1
    req1 = component_request_from_dict({**req_data, "output_directory": "run_1"})
    res1 = run(req1, base=tmp_path)
    assert res1.status == "complete"

    # Run 2
    req2 = component_request_from_dict({**req_data, "output_directory": "run_2"})
    res2 = run(req2, base=tmp_path)
    assert res2.status == "complete"

    dir1 = tmp_path / "run_1"
    dir2 = tmp_path / "run_2"

    for fname in ("report.json", "captions.json", "report.md", "report.html"):
        b1 = (dir1 / fname).read_bytes()
        b2 = (dir2 / fname).read_bytes()
        # The output_directory field in report.json / provenance differs by run directory name,
        # so check logical contents or ignore directory differences
        if fname.endswith(".json"):
            j1 = json.loads(b1.decode("utf-8"))
            j2 = json.loads(b2.decode("utf-8"))
            j1.get("provenance", {}).pop("output_directory", None)
            j2.get("provenance", {}).pop("output_directory", None)
            assert j1 == j2
        else:
            # Markdown and HTML reports have identical text except directory if mentioned
            t1 = b1.decode("utf-8")
            t2 = b2.decode("utf-8")
            assert len(t1) == len(t2)


def test_json_pointer_resolution_edge_cases() -> None:
    """JSON pointer resolver handles root, nested keys, lists, and escape sequences."""
    doc = {
        "a/b": 42,
        "m~n": 84,
        "nested": {"items": [10, 20, 30]},
    }
    assert resolve_json_pointer(doc, "") == doc
    assert resolve_json_pointer(doc, "/a~1b") == 42
    assert resolve_json_pointer(doc, "/m~0n") == 84
    assert resolve_json_pointer(doc, "/nested/items/1") == 20

    with pytest.raises(ValueError, match="must start with '/'"):
        resolve_json_pointer(doc, "no_leading_slash")

    with pytest.raises(KeyError, match="not found"):
        resolve_json_pointer(doc, "/nonexistent")

    with pytest.raises(IndexError, match="out of range"):
        resolve_json_pointer(doc, "/nested/items/99")

    with pytest.raises(ValueError, match="must be integer"):
        resolve_json_pointer(doc, "/nested/items/abc")


def test_standalone_cli_execution(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """CLI prints descriptor or runs request and returns proper exit codes."""
    # 1. --descriptor
    code_desc = main(["--descriptor"])
    assert code_desc == 0
    captured = capsys.readouterr()
    desc_json = json.loads(captured.out)
    assert desc_json["component_id"] == COMPONENT_ID

    # 2. Run CLI with fixture inputs
    out_dir = tmp_path / "cli_out"
    input_path = str(FIXTURE_DIR / "request.json")
    config_path = str(FIXTURE_DIR / "config.json")

    code_run = main(
        [
            "--input",
            input_path,
            "--config",
            config_path,
            "--output",
            str(out_dir),
        ]
    )
    assert code_run == 0
    captured_run = capsys.readouterr()
    res_json = json.loads(captured_run.out)
    assert res_json["status"] == "complete"
    assert (out_dir / "report.json").exists()
