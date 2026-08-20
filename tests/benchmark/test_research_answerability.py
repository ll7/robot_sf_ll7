"""Tests for fail-closed research answerability and yield reporting."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest
import yaml

from robot_sf.benchmark.research_answerability import (
    DECISION_REQUIRED_PROOF_SURFACES,
    PROOF_BINDING_SCHEMA,
    PROOF_SURFACES,
    answerability_from_manifest,
    evaluate_answerability,
)
from scripts.analysis.report_research_yield import (
    ResearchYieldError,
    build_research_yield_report,
    load_snapshot,
    render_markdown,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
EXAMPLE_MANIFEST = REPO_ROOT / "configs/benchmarks/research_campaign_manifest.example.yaml"
ISSUE_6474_FIXTURE = REPO_ROOT / "tests/fixtures/research_answerability/issue_6474_bounded.json"
YIELD_FIXTURE = REPO_ROOT / "tests/fixtures/research_yield_snapshot.v1.json"


def _example_contract() -> dict[str, object]:
    payload = yaml.safe_load(EXAMPLE_MANIFEST.read_text(encoding="utf-8"))
    return copy.deepcopy(payload["answerability"])


def _proof_contract() -> dict[str, object]:
    contract = json.loads(ISSUE_6474_FIXTURE.read_text(encoding="utf-8"))
    contract["proof_surfaces"] = {
        name: {"status": "passed", "required": True} for name in PROOF_SURFACES
    }
    return contract


def _proof_binding() -> dict[str, str]:
    """Return a deterministic synthetic identity for strict evaluator tests."""
    return {
        "schema_version": PROOF_BINDING_SCHEMA,
        "campaign_id": "issue_6474_fixture",
        "source_manifest": "tests/fixtures/research_answerability/issue_6474_bounded.json",
        "campaign_config": "configs/benchmarks/issue_3425_empirical_vertical_slice_smoke.yaml",
        "manifest_sha256": "a" * 64,
        "config_sha256": "b" * 64,
        "proof_digest": "c" * 64,
    }


def test_example_contract_is_diagnostic_only() -> None:
    """The canonical example is executable only as a bounded diagnostic packet."""
    result = evaluate_answerability(_example_contract())

    assert result.state == "diagnostic_only"
    assert result.as_dict()["decision_capable"] is False


def test_optional_unavailable_metric_is_preserved_without_blocking() -> None:
    """A bounded fixture may keep optional unavailable metrics explicit."""
    contract = json.loads(ISSUE_6474_FIXTURE.read_text(encoding="utf-8"))

    result = evaluate_answerability(contract)

    assert result.state == "answerable"
    assert result.warnings
    assert "secondary_realism_metric" in result.warnings[0]


def test_missing_proof_surface_is_invalid() -> None:
    """A declared proof set must name every admission surface explicitly."""
    contract = _proof_contract()
    del contract["proof_surfaces"]["result_packet"]

    result = evaluate_answerability(contract)

    assert result.state == "invalid_contract"
    assert "result_packet" in result.reasons[0]


@pytest.mark.parametrize("status", ["not_run", "unavailable", "failed"])
def test_required_proof_status_blocks_decision_capable_answerability(status: str) -> None:
    """Required proof cannot be silently promoted to a decision-capable result."""
    contract = _proof_contract()
    contract["proof_surfaces"]["analysis"] = {
        "status": status,
        "required": True,
        **(
            {"unavailable_reason": "analysis proof was not produced"}
            if status == "unavailable"
            else {}
        ),
    }

    result = evaluate_answerability(contract)

    assert result.state == "blocked_missing_proof"
    assert "analysis" in result.reasons[0]


@pytest.mark.parametrize("status", ["unavailable", "failed", "not_run"])
def test_optional_nonpassed_proof_is_a_warning(status: str) -> None:
    """Optional non-passed proof remains visible without blocking admission."""
    contract = _proof_contract()
    contract["proof_surfaces"]["result_packet"] = {
        "status": status,
        "required": False,
        **(
            {"unavailable_reason": "packet export is not available for this local comparison"}
            if status == "unavailable"
            else {}
        ),
    }

    result = evaluate_answerability(contract)

    assert result.state == "answerable"
    assert any("result_packet" in warning and status in warning for warning in result.warnings)


def test_all_required_proof_surfaces_pass() -> None:
    """A complete passed proof set preserves decision-capable answerability."""
    result = evaluate_answerability(_proof_contract())

    assert result.state == "answerable"
    assert not any("proof surfaces" in warning for warning in result.warnings)


def test_strict_admission_requires_claim_specific_proof_floor() -> None:
    """Production admission cannot make every claim-critical surface optional."""
    contract = _proof_contract()
    contract["proof_binding"] = _proof_binding()
    for surface in DECISION_REQUIRED_PROOF_SURFACES:
        contract["proof_surfaces"][surface] = {
            "status": "unavailable",
            "required": False,
            "unavailable_reason": "declared optional in a malicious manifest",
        }

    result = evaluate_answerability(contract, enforce_admission_proof=True)

    assert result.state == "blocked_missing_proof"
    assert any(surface in result.reasons[0] for surface in DECISION_REQUIRED_PROOF_SURFACES)


def test_strict_admission_requires_verified_proof_binding() -> None:
    """A passed declarative proof set cannot authorize without exact input identity."""
    contract = _proof_contract()

    result = evaluate_answerability(contract, enforce_admission_proof=True)

    assert result.state == "blocked_missing_proof"
    assert "proof_binding" in result.reasons[0]


@pytest.mark.parametrize(
    ("section", "field", "value", "expected_state", "reason_fragment"),
    [
        (
            "analysis",
            "dry_run_status",
            "not_required",
            "blocked_analysis_contract",
            "requires analysis dry-run status 'passed'",
        ),
        (
            "design",
            "power_status",
            "not_required",
            "blocked_underpowered",
            "requires power status 'adequate'",
        ),
    ],
)
def test_strict_decision_capable_admission_rejects_waived_dry_run_or_power(
    section: str,
    field: str,
    value: str,
    expected_state: str,
    reason_fragment: str,
) -> None:
    """Decision-capable admission cannot waive the dry-run or power proof floor."""
    contract = _proof_contract()
    contract["proof_binding"] = _proof_binding()
    contract["design"]["mode"] = "decision_capable"
    contract["artifacts"]["durability_status"] = "ready"
    contract["analysis"]["dry_run_status"] = "passed"
    contract["design"]["power_status"] = "adequate"
    contract[section][field] = value

    result = evaluate_answerability(contract, enforce_admission_proof=True)

    assert result.state == expected_state
    assert reason_fragment in result.reasons[0]


def test_optional_fallback_producer_remains_visible_as_warning() -> None:
    """Optional fallback/degraded producers cannot disappear from the answerability report."""
    contract = json.loads(ISSUE_6474_FIXTURE.read_text(encoding="utf-8"))
    contract["producers"][1].update({"status": "blocked", "execution_mode": "fallback"})

    result = evaluate_answerability(contract)

    assert result.state == "answerable"
    assert any("secondary_realism_metric" in warning for warning in result.warnings)


@pytest.mark.parametrize(
    ("section", "field", "value", "expected"),
    [
        ("producers", "status", "missing", "blocked_missing_producer"),
        ("producers", "execution_mode", "fallback", "blocked_missing_producer"),
        ("design", "power_status", "underpowered", "blocked_underpowered"),
        ("analysis", "dry_run_status", "failed", "blocked_analysis_contract"),
        ("analysis", "comparability_status", "mismatched", "blocked_noncomparable_rows"),
        ("artifacts", "durability_status", "blocked", "blocked_artifact_plan"),
    ],
)
def test_known_answerability_blockers_are_fail_closed(
    section: str, field: str, value: str, expected: str
) -> None:
    """Known campaign failure classes map to explicit non-answerable states."""
    contract = _example_contract()
    if section == "producers":
        contract[section][0][field] = value
    else:
        contract[section][field] = value

    assert evaluate_answerability(contract).state == expected


def test_malformed_contract_is_invalid() -> None:
    """Missing schema fields cannot be mistaken for an underpowered campaign."""
    contract = _example_contract()
    del contract["estimand"]["primary"]

    result = evaluate_answerability(contract)

    assert result.state == "invalid_contract"
    assert "primary" in result.reasons[0]


@pytest.mark.parametrize("checksums", [None, [], ["  "], ["summary.json", 1]])
def test_invalid_checksum_declarations_are_fail_closed(checksums: object) -> None:
    """Artifact provenance requires a non-empty list of non-empty checksum names."""
    contract = _example_contract()
    if checksums is None:
        del contract["artifacts"]["checksums"]
    else:
        contract["artifacts"]["checksums"] = checksums

    result = evaluate_answerability(contract)

    assert result.state == "invalid_contract"
    assert "checksums" in result.reasons[0]


@pytest.mark.parametrize(
    ("case_id", "mutator", "expected"),
    [
        (
            "6970_missing_normalized_producer",
            lambda contract: contract["producers"][0].update(
                {"status": "missing", "field": "normalized_reference_value"}
            ),
            "blocked_missing_producer",
        ),
        (
            "6849_underpowered_held_out_design",
            lambda contract: contract["design"].update({"power_status": "underpowered"}),
            "blocked_underpowered",
        ),
        (
            "6980_missing_reference_exposure",
            lambda contract: contract["analysis"].update({"comparability_status": "mismatched"}),
            "blocked_noncomparable_rows",
        ),
        (
            "6814_missing_durable_provenance",
            lambda contract: contract["artifacts"].update({"durability_status": "blocked"}),
            "blocked_artifact_plan",
        ),
    ],
)
def test_known_failure_cases_have_explicit_states(case_id: str, mutator, expected: str) -> None:
    """Known issue failure classes cannot be silently promoted to answerable."""
    contract = _example_contract()
    mutator(contract)

    result = evaluate_answerability(contract)

    assert case_id
    assert result.state == expected


def test_manifest_without_answerability_is_not_declared() -> None:
    """Existing manifests remain loadable but can be gated explicitly."""
    manifest = {"campaign": {}}

    result = answerability_from_manifest(manifest)

    assert result["state"] == "not_declared"
    assert result["decision_capable"] is False


def test_research_yield_report_separates_empirical_and_infrastructure() -> None:
    """Yield dimensions remain separate and carry the frozen source digest."""
    snapshot = load_snapshot(YIELD_FIXTURE)
    report = build_research_yield_report(snapshot, source_path=YIELD_FIXTURE)

    assert report["records_total"] == 5
    assert report["empirical_answers"] == {
        "records": 3,
        "statuses": {"completed": 1, "inconclusive": 2},
    }
    assert report["infrastructure_throughput"]["records"] == 2
    assert report["lag_days"]["approval_to_first_result"]["median_days"] == 2.0
    assert report["source_snapshot"]["sha256"]
    assert "closure" in report["definitions"]["empirical_answers"]
    assert "## Empirical Answers" in render_markdown(report)


def test_research_yield_report_renders_query_defined_dimensions() -> None:
    """Issue #7090 dimensions are copied from explicit snapshot queries, not inferred."""
    snapshot = load_snapshot(YIELD_FIXTURE)
    report = build_research_yield_report(snapshot, source_path=YIELD_FIXTURE)
    markdown = render_markdown(report)

    duplicate_dimension = report["dimensions"]["duplicate_competing_prs"]
    assert duplicate_dimension["denominator"] == 5
    assert duplicate_dimension["buckets"] == {
        "competing_pr": 1,
        "duplicate_and_competing": 0,
        "duplicate_pr": 1,
        "no_duplicate_or_competing": 3,
    }
    assert report["dimensions"]["post_merge_repairs"]["buckets"]["post_merge_repair"] == 1
    assert report["dimensions"]["admitted_result_packets"]["buckets"]["admitted_packet"] == 1
    assert report["dimensions"]["blocked_age_categories"]["buckets"] == {
        "blocked_0_7_days": 1,
        "blocked_8_30_days": 1,
        "blocked_over_30_days": 0,
        "not_blocked": 3,
    }
    assert "## Query-Defined Dimensions" in markdown
    assert "duplicate_competing_prs" in markdown
    assert "duplicate_or_competing_pr classification" in duplicate_dimension["query"]


def test_research_yield_report_rejects_unknown_kind(tmp_path: Path) -> None:
    """The report must not silently classify an unknown workflow record."""
    payload = json.loads(YIELD_FIXTURE.read_text(encoding="utf-8"))
    payload["records"][0]["kind"] = "merged_issue"
    path = tmp_path / "invalid.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ResearchYieldError, match="kind is unsupported"):
        load_snapshot(path)


def test_research_yield_report_rejects_duplicate_record_ids(tmp_path: Path) -> None:
    """A frozen snapshot cannot count one work item twice under different rows."""
    payload = json.loads(YIELD_FIXTURE.read_text(encoding="utf-8"))
    payload["records"][1]["id"] = payload["records"][0]["id"]
    path = tmp_path / "duplicate_record.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ResearchYieldError, match="id is duplicated"):
        load_snapshot(path)


def test_research_yield_report_rejects_non_finite_lag(tmp_path: Path) -> None:
    """NaN lag values cannot enter a reproducible JSON report."""
    payload = json.loads(YIELD_FIXTURE.read_text(encoding="utf-8"))
    payload["records"][0]["approval_to_first_result_days"] = float("nan")
    path = tmp_path / "non_finite_lag.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ResearchYieldError, match="non-negative numeric values"):
        load_snapshot(path)


def test_research_yield_report_validates_in_memory_snapshots() -> None:
    """The public report builder must retain fail-closed validation without file loading."""
    snapshot = load_snapshot(YIELD_FIXTURE)
    del snapshot["dimensions"]["blocked_age_categories"]

    with pytest.raises(ResearchYieldError, match="missing required names"):
        build_research_yield_report(snapshot)


def test_research_yield_report_rejects_unknown_dimension(tmp_path: Path) -> None:
    """Snapshot dimensions are explicit reporting queries, not an open-ended tag bag."""
    payload = json.loads(YIELD_FIXTURE.read_text(encoding="utf-8"))
    payload["dimensions"]["merged_without_review"] = {
        "query": "unsupported query",
        "denominator": 0,
        "buckets": {},
    }
    path = tmp_path / "invalid_dimension.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ResearchYieldError, match="unsupported names"):
        load_snapshot(path)


def test_research_yield_report_rejects_missing_required_dimension(tmp_path: Path) -> None:
    """Every supported dimension must remain explicit in the frozen snapshot."""
    payload = json.loads(YIELD_FIXTURE.read_text(encoding="utf-8"))
    del payload["dimensions"]["blocked_age_categories"]
    path = tmp_path / "missing_dimension.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ResearchYieldError, match="missing required names"):
        load_snapshot(path)


def test_research_yield_report_rejects_unknown_dimension_bucket(tmp_path: Path) -> None:
    """Known dimensions cannot accept inferred or ad-hoc bucket names."""
    payload = json.loads(YIELD_FIXTURE.read_text(encoding="utf-8"))
    payload["dimensions"]["post_merge_repairs"]["buckets"]["repair_inferred_from_ci"] = 1
    path = tmp_path / "unknown_bucket.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ResearchYieldError, match="buckets contain unsupported names"):
        load_snapshot(path)


def test_research_yield_report_rejects_unknown_dimension_field(tmp_path: Path) -> None:
    """A dimension cannot silently preserve fields outside its versioned contract."""
    payload = json.loads(YIELD_FIXTURE.read_text(encoding="utf-8"))
    payload["dimensions"]["post_merge_repairs"]["review_source"] = "live-state"
    path = tmp_path / "unknown_dimension_field.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ResearchYieldError, match="contains unsupported fields"):
        load_snapshot(path)


def test_research_yield_report_rejects_dimension_denominator_mismatch(tmp_path: Path) -> None:
    """Dimension denominators must match the explicit bucket total."""
    payload = json.loads(YIELD_FIXTURE.read_text(encoding="utf-8"))
    payload["dimensions"]["admitted_result_packets"]["denominator"] = 6
    path = tmp_path / "denominator_mismatch.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ResearchYieldError, match="buckets sum to 5, expected denominator 6"):
        load_snapshot(path)


@pytest.mark.parametrize("bad_count", [-1, 1.5, True])
def test_research_yield_report_rejects_non_integer_dimension_counts(
    tmp_path: Path, bad_count: object
) -> None:
    """Dimension counts are non-negative integers; bool is rejected despite being an int subtype."""
    payload = json.loads(YIELD_FIXTURE.read_text(encoding="utf-8"))
    payload["dimensions"]["duplicate_competing_prs"]["buckets"]["duplicate_pr"] = bad_count
    path = tmp_path / "bad_bucket_count.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ResearchYieldError, match="must be a non-negative integer"):
        load_snapshot(path)


@pytest.mark.parametrize("bad_denominator", [-1, 5.0, False])
def test_research_yield_report_rejects_non_integer_dimension_denominator(
    tmp_path: Path, bad_denominator: object
) -> None:
    """Dimension denominators use the same non-negative integer contract as bucket counts."""
    payload = json.loads(YIELD_FIXTURE.read_text(encoding="utf-8"))
    payload["dimensions"]["blocked_age_categories"]["denominator"] = bad_denominator
    path = tmp_path / "bad_denominator.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ResearchYieldError, match="must be a non-negative integer"):
        load_snapshot(path)
