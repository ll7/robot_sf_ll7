"""Focused tests for the issue #8566 source-bound fixture packet."""

from __future__ import annotations

import copy
import json
import subprocess
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable

import pytest

from robot_sf.benchmark.source_bound_component_contrasts import (
    SourceBoundComponentContrastError,
    _component_contrast,
    _validate_fixture_rows,
    build_source_bound_report,
    load_fixture_rows,
    load_source_bound_config,
    validate_source_bound_config,
    validate_source_bound_report,
    write_source_bound_receipt,
    write_source_bound_report,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG = (
    REPO_ROOT
    / "configs"
    / "analysis"
    / "issue_8566_source_bound_uncertainty_component_contrasts.yaml"
)
FIXTURE = (
    REPO_ROOT
    / "tests"
    / "fixtures"
    / "issue_8566_source_bound_component_contrasts"
    / "paired_rows.json"
)
REPORT = (
    REPO_ROOT
    / "docs"
    / "context"
    / "evidence"
    / "issue_8566_source_bound_component_contrasts_2026-09-10"
    / "report.json"
)
RECEIPT = REPORT.with_name("SHA256SUMS")


def test_durable_report_is_reproducible_and_source_complete_only_for_fixture() -> None:
    """The checked-in report is deterministic and labels external rows unavailable."""

    config = load_source_bound_config(CONFIG)
    validate_source_bound_config(config, repo_root=REPO_ROOT)
    report = build_source_bound_report(CONFIG, repo_root=REPO_ROOT)
    expected = json.loads(REPORT.read_text(encoding="utf-8"))

    assert report == expected
    validate_source_bound_report(report)
    assert report["status"] == "diagnostic_only"
    assert report["evidence_status"] == "not_benchmark_evidence"
    assert report["review_marker"] == "AI-GENERATED NEEDS-REVIEW"
    assert report["semantics"] == {
        "benchmark_evidence": False,
        "campaign_executed": False,
        "claim_promotion": "none",
        "degraded_rows": 0,
        "external_artifact_hydrated": False,
        "fallback_rows": 0,
        "new_episode_created": False,
    }
    assert all(
        component["status"] == "unavailable"
        for target in report["source_coverage"]
        for component in target["component_coverage"]
    )

    checked = subprocess.run(
        ["sha256sum", "-c", str(RECEIPT)],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert checked.returncode == 0, checked.stdout + checked.stderr
    receipt = RECEIPT.read_text(encoding="utf-8")
    assert "configs/publication/dissertation_coverage_v1.yaml" in receipt
    assert (
        "configs/benchmarks/releases/benchmark_data_release_s30_h600_2026_09_erratum_1.json"
        in receipt
    )
    assert "configs/benchmarks/releases/recovery/job_15180_59577bad_recovery_v1.json" in receipt


def test_public_writers_preserve_report_and_receipt_bytes(tmp_path: Path) -> None:
    """Public writers reproduce the validated report and its source receipt."""

    report = build_source_bound_report(CONFIG, repo_root=REPO_ROOT)
    report_path = tmp_path / "report.json"
    receipt_path = tmp_path / "SHA256SUMS"

    write_source_bound_report(report, report_path)
    write_source_bound_receipt(
        report,
        config_path=CONFIG,
        repo_root=REPO_ROOT,
        report_path=REPORT,
        receipt_path=receipt_path,
    )

    assert json.loads(report_path.read_text(encoding="utf-8")) == report
    assert receipt_path.read_text(encoding="utf-8").startswith("# AI-GENERATED NEEDS-REVIEW\n")
    assert receipt_path.read_text(encoding="utf-8") == RECEIPT.read_text(encoding="utf-8")


@pytest.mark.parametrize(
    ("mutate", "match"),
    [
        (lambda config: config.__setitem__("schema_version", "wrong"), "schema_version"),
        (lambda config: config.__setitem__("issue", 1), "issue"),
        (lambda config: config.__setitem__("status", "ready"), "status"),
        (lambda config: config.__setitem__("base_commit", "bad"), "base_commit"),
        (
            lambda config: config["execution"].__setitem__("campaign_execution", True),
            "campaign_execution",
        ),
        (lambda config: config["analysis"].__setitem__("pairing_key", []), "pairing_key"),
        (lambda config: config["analysis"].__setitem__("clustering_key", "seed"), "clustering_key"),
        (
            lambda config: config["analysis"]["uncertainty"].__setitem__("method", "wrong"),
            "uncertainty.method",
        ),
        (
            lambda config: config["analysis"]["uncertainty"].__setitem__("confidence", 1.0),
            "confidence",
        ),
        (
            lambda config: config["analysis"]["uncertainty"].__setitem__("seed", True),
            "uncertainty.seed",
        ),
        (
            lambda config: config["analysis"]["multiplicity"].__setitem__("method", "wrong"),
            "multiplicity.method",
        ),
        (
            lambda config: config["analysis"]["multiplicity"].__setitem__("alpha", 1.0),
            "alpha",
        ),
        (lambda config: config.__setitem__("components", []), "components"),
        (
            lambda config: (
                config["components"].__getitem__(1).__setitem__("id", config["components"][0]["id"])
            ),
            "duplicate component",
        ),
        (
            lambda config: config["generator"].__setitem__("path", "docs/context/INDEX.md"),
            "generator.path",
        ),
        (lambda config: config.__setitem__("external_targets", []), "external_targets"),
        (
            lambda config: config["external_targets"][0].__setitem__("status", "available"),
            "external_targets",
        ),
        (
            lambda config: config["external_targets"][0].__setitem__(
                "component_ids", ["terminal_outcome"]
            ),
            "component_ids",
        ),
        (lambda config: config.__setitem__("source_bindings", []), "source_bindings"),
        (
            lambda config: config["source_bindings"][1].__setitem__(
                "source_id", config["source_bindings"][0]["source_id"]
            ),
            "duplicate source id",
        ),
        (
            lambda config: config["fixture"].__setitem__("source_id", "unbound"),
            "fixture.source_id",
        ),
        (
            lambda config: config["fixture"].__setitem__(
                "scenarios", ["fixture_corridor", "fixture_corridor"]
            ),
            "scenarios",
        ),
        (
            lambda config: config["fixture"].__setitem__("seeds", [111, 111]),
            "seeds",
        ),
        (
            lambda config: config["fixture"].__setitem__("seeds", [True]),
            "fixture.seeds",
        ),
        (
            lambda config: config["source_bindings"][0].__setitem__("kind", "unknown"),
            "kind is unsupported",
        ),
        (
            lambda config: config["source_bindings"][1].__setitem__("source_commit", "bad"),
            "source_commit",
        ),
        (
            lambda config: config["source_bindings"][0].__setitem__("tracked_commit", "0" * 40),
            "cannot claim",
        ),
    ],
)
def test_config_contract_rejects_unsafe_mutations(
    mutate: Callable[[dict[str, object]], None], match: str
) -> None:
    """Unsafe identity, analysis, source, and availability mutations fail closed."""

    config = copy.deepcopy(load_source_bound_config(CONFIG))
    mutate(config)

    with pytest.raises(SourceBoundComponentContrastError, match=match):
        validate_source_bound_config(config, repo_root=REPO_ROOT)


def test_fixture_components_have_explicit_paired_uncertainty_and_multiplicity() -> None:
    """Each fixture component has the frozen denominator, cluster, and Holm family."""

    report = build_source_bound_report(CONFIG, repo_root=REPO_ROOT)

    assert report["analysis"]["pairing_key"] == [
        "corpus_id",
        "release_id",
        "contrast_id",
        "scenario_id",
        "seed",
    ]
    assert report["analysis"]["clustering_key"] == "scenario_id"
    assert report["analysis"]["multiplicity"]["n_comparisons"] == 3
    assert len(report["component_contrasts"]) == 3
    for component in report["component_contrasts"]:
        assert component["status"] == "diagnostic_only"
        assert component["source_refs"] == [
            {
                "path": "tests/fixtures/issue_8566_source_bound_component_contrasts/paired_rows.json",
                "sha256": "49824d8a10e4865917de6ca501be8b00d1fe09bb34ce71ec3865b6bed797434b",
                "source_id": "issue_8566_fixture_rows",
            }
        ]
        assert component["support"] == component["denominator"] == 6
        assert component["uncertainty"]["method"] == ("paired_scenario_block_percentile_bootstrap")
        assert component["uncertainty"]["ci_low"] is not None
        assert component["uncertainty"]["ci_high"] is not None
        assert component["multiplicity"]["status"] == "applied"
        assert component["multiplicity"]["n_comparisons"] == 3


def test_source_digest_mismatch_fails_closed() -> None:
    """Changing a declared source digest rejects the packet before computation."""

    config = copy.deepcopy(load_source_bound_config(CONFIG))
    config["source_bindings"][0]["sha256"] = "0" * 64

    with pytest.raises(SourceBoundComponentContrastError, match="digest mismatch"):
        validate_source_bound_config(config, repo_root=REPO_ROOT)


def test_fixture_path_cannot_drift_from_its_source_binding() -> None:
    """The fixture path and the digest-bound source id must identify the same bytes."""

    config = copy.deepcopy(load_source_bound_config(CONFIG))
    config["fixture"]["path"] = "configs/publication/dissertation_coverage_v1.yaml"

    with pytest.raises(SourceBoundComponentContrastError, match="fixture.path"):
        validate_source_bound_config(config, repo_root=REPO_ROOT)


def test_release_crossing_fixture_row_fails_closed() -> None:
    """Rows cannot silently cross the fixture release identity."""

    config = validate_source_bound_config(load_source_bound_config(CONFIG), repo_root=REPO_ROOT)
    payload = load_fixture_rows(FIXTURE)
    payload["rows"][0]["release_id"] = "other_release"

    with pytest.raises(SourceBoundComponentContrastError, match="crosses"):
        _validate_fixture_rows(config, payload)


def test_missing_pair_is_blocked_without_zero_imputation() -> None:
    """A missing pair emits no effect or uncertainty estimate."""

    config = validate_source_bound_config(load_source_bound_config(CONFIG), repo_root=REPO_ROOT)
    payload = load_fixture_rows(FIXTURE)
    payload["rows"] = payload["rows"][:-1]
    rows_by_key, missing_keys = _validate_fixture_rows(config, payload)
    component = _component_contrast(
        config["components"][0],
        config=config,
        rows_by_key=rows_by_key,
        missing_keys=missing_keys,
    )

    assert missing_keys == [("fixture_crossing", 112)]
    assert component["status"] == "blocked"
    assert component["effect"] is None
    assert component["support"] == 5
    assert component["denominator"] == 6
    assert component["uncertainty"]["ci_low"] is None
    assert component["uncertainty"]["p_value_raw"] is None


def test_unavailable_cell_cannot_carry_numeric_value() -> None:
    """Unavailable component cells require a reason and cannot be treated as zero."""

    config = validate_source_bound_config(load_source_bound_config(CONFIG), repo_root=REPO_ROOT)
    payload = load_fixture_rows(FIXTURE)
    cell = payload["rows"][0]["components"]["snqi"]
    cell["status"] = "unavailable"
    cell["reason"] = "source cell absent"

    with pytest.raises(SourceBoundComponentContrastError, match="must not carry numeric"):
        _validate_fixture_rows(config, payload)


def test_release_lineages_remain_separate() -> None:
    """August and September metadata are retained as distinct source lineages."""

    report = build_source_bound_report(CONFIG, repo_root=REPO_ROOT)
    lineages = {item["id"]: item for item in report["lineage_guard"]["retained_lineages"]}

    assert set(lineages) == {"august_b1d5", "september_59577"}
    assert lineages["august_b1d5"]["source_commit"] != lineages["september_59577"]["source_commit"]
    assert report["lineage_guard"]["prohibited_pooling"] == ["august_b1d5", "september_59577"]
