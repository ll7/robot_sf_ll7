"""Determinism, reproduction, and fail-closed tests for issue #6592 precision derivation.

These tests prove:
- Byte-for-byte / SHA-256 reproducibility of every produced artifact from frozen rows.
- Fail-closed behavior on SHA-256 mismatch, wrong row count, and wrong family count.
- No produced output uses the token 'power' to describe an observed-data computation.
- No output claims 30 seeds were prospectively sized to detect the derived effect.
- The machine-readable report carries a schema_version and blocked_review_pending.
"""

# evidence-writer-exempt: these tests exercise the precision derivation script
# against the frozen #5351 successor rows; they do not generate new evidence artifacts.

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from unittest.mock import patch

import pytest

from tests.support.script_loader import load_script_module

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SCRIPT_PATH = _REPO_ROOT / "scripts" / "analysis" / "derive_retrospective_precision_issue_6592.py"
_ROWS_PATH = (
    _REPO_ROOT
    / "docs"
    / "context"
    / "evidence"
    / "issue_5351_hierarchical_paired_release_analysis"
    / "successor_rows.jsonl"
)
_FAMILY_MAPPING_PATH = (
    _REPO_ROOT
    / "docs"
    / "context"
    / "evidence"
    / "issue_6592_retrospective_precision"
    / "scenario_family_mapping.json"
)
_EXPECTED_ROWS_SHA256 = "c45c2ed8defdadaf47c001277e6bf9ca0c2238c101570d1d64be8015060febea"
_EXPECTED_FAMILY_MAPPING_SHA256 = "edd5dbed94bc4795255e7728e627fe8fb3282ab5efde8f64dfb92181758ef510"


@pytest.fixture(scope="module")
def precision_module():
    """Load the issue #6592 precision derivation script once per test module."""
    return load_script_module(
        _SCRIPT_PATH,
        name="derive_retrospective_precision_issue_6592",
    )


@pytest.fixture(scope="module")
def precision_run(precision_module, tmp_path_factory):
    """Run the full precision derivation once and return (report, evidence_dir).

    Patches the evidence catalog registration so tmp paths do not trigger
    relative_to errors.
    """
    evidence_dir = tmp_path_factory.mktemp("precision_6592")
    with patch("robot_sf.evidence.writers._maybe_register"):
        exit_code = precision_module.main(
            [
                "--repo-root",
                str(_REPO_ROOT),
                "--evidence-dir",
                str(evidence_dir),
            ]
        )
    assert exit_code == 0
    report_path = evidence_dir / "retrospective_precision_report.json"
    with report_path.open(encoding="utf-8") as fh:
        report = json.load(fh)
    return report, evidence_dir


def _sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as fh:
        while chunk := fh.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


# ---------------------------------------------------------------------------
# Determinism and reproduction
# ---------------------------------------------------------------------------


class TestDeterminism:
    """Prove byte-for-byte reproducibility of every produced artifact."""

    def test_report_is_deterministic(self, precision_module, tmp_path_factory):
        """Two independent runs produce identical report bytes."""
        dirs = []
        with patch("robot_sf.evidence.writers._maybe_register"):
            for i in range(2):
                d = tmp_path_factory.mktemp(f"det_run_{i}")
                rc = precision_module.main(
                    [
                        "--repo-root",
                        str(_REPO_ROOT),
                        "--evidence-dir",
                        str(d),
                    ]
                )
                assert rc == 0
                dirs.append(d)

        report_a = (dirs[0] / "retrospective_precision_report.json").read_bytes()
        report_b = (dirs[1] / "retrospective_precision_report.json").read_bytes()
        assert report_a == report_b, "report JSON is not byte-for-byte deterministic"

    def test_readme_is_deterministic(self, precision_module, tmp_path_factory):
        """Two independent runs produce identical README bytes."""
        dirs = []
        with patch("robot_sf.evidence.writers._maybe_register"):
            for i in range(2):
                d = tmp_path_factory.mktemp(f"det_readme_{i}")
                rc = precision_module.main(
                    [
                        "--repo-root",
                        str(_REPO_ROOT),
                        "--evidence-dir",
                        str(d),
                    ]
                )
                assert rc == 0
                dirs.append(d)

        readme_a = (dirs[0] / "README.md").read_bytes()
        readme_b = (dirs[1] / "README.md").read_bytes()
        assert readme_a == readme_b, "README is not byte-for-byte deterministic"

    def test_sha256sums_cover_all_artifacts(self, precision_run):
        """SHA256SUMS covers every produced artifact and checksums match."""
        _report, evidence_dir = precision_run
        sums_path = evidence_dir / "SHA256SUMS"
        assert sums_path.is_file()

        recorded: dict[str, str] = {}
        for line in sums_path.read_text(encoding="utf-8").strip().splitlines():
            digest, name = line.split("  ", 1)
            recorded[name] = digest

        for artifact in sorted(evidence_dir.iterdir()):
            if artifact.name == "SHA256SUMS" or not artifact.is_file():
                continue
            assert artifact.name in recorded, f"{artifact.name} missing from SHA256SUMS"
            actual = _sha256(artifact)
            assert actual == recorded[artifact.name], f"SHA256SUMS mismatch for {artifact.name}"

    def test_committed_sha256sums_cover_family_mapping(self):
        """The committed evidence checksum list covers the pinned mapping, not its sidecar."""
        sums_path = _FAMILY_MAPPING_PATH.with_name("SHA256SUMS")
        recorded = {
            line.split("  ", 1)[1]: line.split("  ", 1)[0]
            for line in sums_path.read_text(encoding="utf-8").strip().splitlines()
        }
        assert recorded["scenario_family_mapping.json"] == _sha256(_FAMILY_MAPPING_PATH)
        assert "SHA256SUMS.review.json" not in recorded


# ---------------------------------------------------------------------------
# Fail-closed behavior
# ---------------------------------------------------------------------------


class TestFailClosed:
    """Prove the analysis fails closed on input/unit/provenance disagreement."""

    def test_sha256_mismatch_fails_closed(self, precision_module, tmp_path):
        """Modified rows file triggers SHA-256 mismatch failure."""
        tampered = tmp_path / "successor_rows.jsonl"
        tampered.write_text('{"tampered": true}\n', encoding="utf-8")
        with pytest.raises(precision_module.RetrospectivePrecisionError, match="SHA-256 mismatch"):
            precision_module.load_and_verify_frozen_rows(tampered)

    def test_wrong_row_count_fails_closed(self, precision_module, tmp_path):
        """Truncated rows file triggers row count mismatch failure.

        Monkeypatches the expected SHA-256 to match the truncated file so the
        SHA gate passes and the row-count gate is exercised.
        """
        truncated = tmp_path / "successor_rows.jsonl"
        rows = []
        for i in range(100):
            rows.append(
                json.dumps(
                    {
                        "schema_version": "EpisodeEventLedger.v2",
                        "scenario_id": f"scenario_{i}",
                        "seed": i,
                        "planner": "orca",
                        "exact_events": {
                            "collision": False,
                            "goal_reached": True,
                            "timeout": False,
                            "invalid_run": False,
                        },
                        "surrogate_events": {"near_miss": False},
                        "provenance": {
                            "completion_time": 10.0,
                            "near_miss_count": 0,
                            "exposure": {"time": 10.0, "distance": 5.0, "opportunity": 1.0},
                            "interaction_exposure": {
                                "schema_version": "interaction_exposure.v1",
                                "status": "computed",
                                "interaction_exposure_steps": 100,
                                "interaction_exposure_denominator_steps": 100,
                            },
                        },
                    }
                )
            )
        content = "\n".join(rows) + "\n"
        truncated.write_text(content, encoding="utf-8")
        actual_sha = hashlib.sha256(content.encode()).hexdigest()
        with (
            patch.object(precision_module, "EXPECTED_ROWS_SHA256", actual_sha),
            pytest.raises(precision_module.RetrospectivePrecisionError, match="row count mismatch"),
        ):
            precision_module.load_and_verify_frozen_rows(truncated)

    def test_missing_rows_file_fails_closed(self, precision_module, tmp_path):
        """Missing rows file triggers fail-closed error."""
        missing = tmp_path / "nonexistent.jsonl"
        with pytest.raises(precision_module.RetrospectivePrecisionError, match="not found"):
            precision_module.load_and_verify_frozen_rows(missing)

    def test_wrong_family_count_fails_closed(self, precision_module):
        """Family mapping producing wrong count triggers fail-closed error."""
        rows = [
            {"scenario_id": "scenario_a", "planner": "orca", "seed": 1},
            {"scenario_id": "scenario_b", "planner": "orca", "seed": 1},
        ]
        family_of = {"scenario_a": "fam_a", "scenario_b": "fam_b"}
        with pytest.raises(
            precision_module.RetrospectivePrecisionError, match="family count mismatch"
        ):
            precision_module.verify_family_count(rows, family_of)

    def test_uncovered_scenario_fails_closed(self, precision_module):
        """Missing scenario in family_of triggers fail-closed error."""
        rows = [
            {"scenario_id": "scenario_a", "planner": "orca", "seed": 1},
            {"scenario_id": "scenario_b", "planner": "orca", "seed": 1},
        ]
        family_of = {"scenario_a": "fam_a"}
        with pytest.raises(precision_module.RetrospectivePrecisionError, match="missing"):
            precision_module.verify_family_count(rows, family_of)


# ---------------------------------------------------------------------------
# Report schema and claim boundary
# ---------------------------------------------------------------------------


class TestReportSchema:
    """Prove the report carries required schema and claim boundary fields."""

    def test_schema_version(self, precision_run):
        """Report carries the expected schema_version."""
        report, _ = precision_run
        assert report["schema_version"] == "retrospective_precision_report.v1"

    def test_claim_gate_blocked_review_pending(self, precision_run):
        """Report claim gate is blocked_review_pending."""
        report, _ = precision_run
        assert report["claim_gate"]["status"] == "blocked_review_pending"

    def test_evidence_status(self, precision_run):
        """Report evidence status is not_benchmark_evidence."""
        report, _ = precision_run
        assert report["evidence_status"] == "not_benchmark_evidence"

    def test_estimand_fields(self, precision_run):
        """Report contains estimand, comparison unit, and outer resampling unit."""
        report, _ = precision_run
        estimand = report["estimand"]
        assert estimand["outer_resampling_unit"] == ("one-stage scenario-family cluster bootstrap")
        assert estimand["n_families"] == 35
        assert estimand["n_cells_per_pair"] == 1440
        assert estimand["comparison_unit"] == "matched planner-scenario-seed cell"

    def test_interval_construction(self, precision_run):
        """Report contains confidence level and interval construction method."""
        report, _ = precision_run
        interval = report["interval_construction"]
        assert interval["confidence"] == 0.95
        assert interval["method"] == "equal-tailed percentile bootstrap"
        assert interval["bootstrap_samples"] == 2000

    def test_frozen_input_provenance(self, precision_run):
        """Report contains frozen input provenance with correct checksums."""
        report, _ = precision_run
        prov = report["frozen_input_provenance"]
        assert prov["successor_rows_sha256"] == _EXPECTED_ROWS_SHA256
        assert prov["total_rows"] == 20160
        assert prov["arms_count"] == 14
        assert prov["rows_per_arm"] == 1440
        assert prov["family_count"] == 35
        assert prov["scenario_family_mapping_sha256"] == _EXPECTED_FAMILY_MAPPING_SHA256

    def test_exclusions_present(self, precision_run):
        """Report contains material exclusions for rare-event, family, and independence."""
        report, _ = precision_run
        exclusion_types = {e["type"] for e in report["exclusions"]}
        assert "rare_event" in exclusion_types
        assert "family_generalization" in exclusion_types
        assert "non_independent_interpretation" in exclusion_types
        assert "prospective_sizing" in exclusion_types

    def test_multiplicity_handling(self, precision_run):
        """Report contains multiplicity handling with Holm step-down."""
        report, _ = precision_run
        mult = report["multiplicity"]
        assert mult["method"] == "holm_step_down"
        assert mult["n_exposed_contrasts"] == 39

    def test_headline_contrasts_have_precision(self, precision_run):
        """Each headline collision contrast has CI width and MRRD."""
        report, _ = precision_run
        headline = report["headline_collision_precisions"]
        assert len(headline) == 13
        for entry in headline:
            assert entry["outcome"] == "collision"
            assert entry["ci_width"] > 0
            assert entry["mrrd_practical_simulated"] > 0
            assert entry["mrrd_statistical"] > 0
            assert entry["n_families"] == 35
            assert entry["n_cells"] == 1440

    def test_sensitivity_analyses_present(self, precision_run):
        """Report contains sensitivity analyses across event rate grid."""
        report, _ = precision_run
        sensitivity = report["sensitivity_analyses"]
        assert len(sensitivity) > 0
        for entry in sensitivity:
            assert entry["outcome"] == "collision"
            grid = entry["grid"]
            assert len(grid) == 9  # SENSITIVITY_EVENT_RATES has 9 entries
            for point in grid:
                assert point["n_families"] == 35
                assert point["bootstrap_se"] >= 0
                assert point["outcome"] == "collision"
                assert point["outcome_model"] == "independent Bernoulli null arms"
        assert any(point["bootstrap_se"] > 0 for entry in sensitivity for point in entry["grid"])


# ---------------------------------------------------------------------------
# Forbidden language checks
# ---------------------------------------------------------------------------


class TestForbiddenLanguage:
    """Prove no output uses forbidden language."""

    def test_no_power_token_in_report(self, precision_run):
        """Report JSON does not use 'power' to describe any computation."""
        report, _ = precision_run
        report_str = json.dumps(report).lower()
        assert "power" not in report_str, "report JSON contains the token 'power'"

    def test_no_power_token_in_readme(self, precision_run):
        """README does not use 'power' to describe any computation."""
        _, evidence_dir = precision_run
        readme = (evidence_dir / "README.md").read_text(encoding="utf-8").lower()
        assert "power" not in readme, "README contains the token 'power'"

    def test_no_prospective_sizing_claim_in_report(self, precision_run):
        """Report does not claim 30 seeds were prospectively sized."""
        report, _ = precision_run
        report_str = json.dumps(report).lower()
        assert "prospectively sized to detect" not in report_str
        assert "30 seeds were prospectively sized" not in report_str

    def test_no_prospective_sizing_claim_in_readme(self, precision_run):
        """README does not claim 30 seeds were prospectively sized."""
        _, evidence_dir = precision_run
        readme = (evidence_dir / "README.md").read_text(encoding="utf-8").lower()
        assert "prospectively sized to detect" not in readme
        assert "30 seeds were prospectively sized" not in readme


# ---------------------------------------------------------------------------
# Frozen input verification
# ---------------------------------------------------------------------------


class TestFrozenInputVerification:
    """Prove the frozen input is correctly verified."""

    def test_frozen_rows_sha256(self):
        """The frozen successor_rows.jsonl matches the expected SHA-256."""
        assert _ROWS_PATH.is_file(), f"frozen rows not found: {_ROWS_PATH}"
        actual = _sha256(_ROWS_PATH)
        assert actual == _EXPECTED_ROWS_SHA256

    def test_frozen_rows_load_and_verify(self, precision_module):
        """load_and_verify_frozen_rows succeeds on the real frozen rows."""
        rows = precision_module.load_and_verify_frozen_rows(_ROWS_PATH)
        assert len(rows) == 20160

    def test_family_mapping_produces_35(self, precision_module):
        """The pinned family mapping produces the admitted 35 cluster units."""
        rows = precision_module.load_and_verify_frozen_rows(_ROWS_PATH)
        family_of = precision_module.load_and_verify_family_mapping(_FAMILY_MAPPING_PATH)
        n_families = precision_module.verify_family_mapping_against_configs(
            rows, family_of, repo_root=_REPO_ROOT
        )
        assert n_families == 35

    def test_family_mapping_sha256(self):
        """The durable family partition is checksum-pinned."""
        assert _sha256(_FAMILY_MAPPING_PATH) == _EXPECTED_FAMILY_MAPPING_SHA256

    def test_family_mapping_tamper_fails_closed(self, precision_module, tmp_path):
        """A modified family partition is rejected before analysis."""
        tampered = tmp_path / "scenario_family_mapping.json"
        tampered.write_bytes(_FAMILY_MAPPING_PATH.read_bytes() + b"\n")
        with pytest.raises(
            precision_module.RetrospectivePrecisionError,
            match="scenario-family mapping SHA-256 mismatch",
        ):
            precision_module.load_and_verify_family_mapping(tampered)

    def test_observed_risk_difference_uses_frozen_cells(self, precision_module, precision_run):
        """Observed risk difference is not replaced by the bootstrap mean."""
        report, _ = precision_run
        rows = precision_module.load_and_verify_frozen_rows(_ROWS_PATH)
        family_of = precision_module.load_and_verify_family_mapping(_FAMILY_MAPPING_PATH)
        cells = precision_module.build_matched_cells_from_ledger_rows(
            rows, planner_pair=("goal", "orca"), family_of=family_of
        )
        expected = sum(cell.collision_a - cell.collision_b for cell in cells) / len(cells)
        entry = next(
            item
            for item in report["contrast_precisions"]
            if item["planner_pair"] == ["goal", "orca"] and item["outcome"] == "collision"
        )
        assert entry["observed_risk_difference"] == pytest.approx(expected)
