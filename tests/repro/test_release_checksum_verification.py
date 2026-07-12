"""Contract tests for release checksum manifest, verification, and cold-start reproduction.

Tests verify:
- Checksum manifest YAML schema and content
- Verification script logic
- Cold-start reproduction report generator
- Documentation and discoverability
"""

from __future__ import annotations

import hashlib
import json
import shutil
import tarfile
from io import BytesIO
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]
MANIFEST_PATH = ROOT / "configs" / "releases" / "release_0_0_2_checksum_manifest.yaml"
VERIFY_SCRIPT = ROOT / "scripts" / "repro" / "verify_release_checksums.py"
REPORT_SCRIPT = ROOT / "scripts" / "repro" / "cold_start_reproduction_report.py"
RELEASE_REPRO_DOC = ROOT / "docs" / "benchmark_release_reproducibility.md"
REPRO_002_DOC = ROOT / "docs" / "benchmark_release_0_0_2_reproduction.md"
RELEASE_METADATA = (
    ROOT
    / "docs"
    / "experiments"
    / "publication"
    / "20260414_benchmark_release_0_0_2"
    / "release_metadata.json"
)
BUNDLE_SMOKE_SCRIPT = ROOT / "scripts" / "repro" / "benchmark_bundle_smoke.sh"
DURABLE_EVIDENCE_REPORT = (
    ROOT
    / "docs"
    / "context"
    / "evidence"
    / "issue_5366_cold_start_reproduction_2026-07-12"
    / "cold_start_reproduction_report.json"
)


def _read_text_file(path: Path) -> str:
    if not path.is_file():
        raise FileNotFoundError(f"Expected file does not exist: {path}")
    return path.read_text(encoding="utf-8")


def _read_yaml(path: Path) -> dict[str, Any]:
    return yaml.safe_load(_read_text_file(path))


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(_read_text_file(path))


class TestChecksumManifestSchema:
    """Tests for the checksum manifest YAML file."""

    def test_manifest_file_exists(self) -> None:
        assert MANIFEST_PATH.is_file(), f"Manifest not found: {MANIFEST_PATH}"

    def test_manifest_has_required_schema_version(self) -> None:
        manifest = _read_yaml(MANIFEST_PATH)
        assert manifest["schema_version"] == "release-checksum-manifest.v1"

    def test_manifest_has_release_metadata(self) -> None:
        manifest = _read_yaml(MANIFEST_PATH)
        assert manifest["release_tag"] == "0.0.2"
        assert manifest["release_id"] == "benchmark_release_0_0_2"
        assert "release_url" in manifest
        assert "doi_url" in manifest
        assert "target_commit" in manifest
        assert manifest["release_url"] == "https://github.com/ll7/robot_sf_ll7/releases/tag/0.0.2"
        assert manifest["doi_url"] == "https://doi.org/10.5281/zenodo.19563812"

    def test_manifest_bundle_archive_has_sha256(self) -> None:
        manifest = _read_yaml(MANIFEST_PATH)
        bundle = manifest["artifact_set"]["bundle_archive"]
        assert "sha256" in bundle
        assert len(bundle["sha256"]) == 64
        assert (
            bundle["name"]
            == "paper_experiment_matrix_7planners_v1_release_v0_0_2_20260414_134316_publication_bundle.tar.gz"
        )

    def test_manifest_embedded_artifacts_have_checksums(self) -> None:
        manifest = _read_yaml(MANIFEST_PATH)
        embedded = manifest.get("embedded_artifacts", {})
        assert len(embedded) >= 3
        for name, info in embedded.items():
            assert "path_in_archive" in info, f"{name} missing path_in_archive"
            assert "sha256" in info, f"{name} missing sha256"
            assert len(info["sha256"]) == 64, f"{name} sha256 wrong length"

    def test_manifest_campaign_metadata(self) -> None:
        manifest = _read_yaml(MANIFEST_PATH)
        campaign = manifest["campaign"]
        assert (
            campaign["campaign_id"]
            == "paper_experiment_matrix_7planners_v1_release_v0_0_2_20260414_134316"
        )
        assert campaign["total_episodes"] == 987
        assert campaign["successful_runs"] == 7

    def test_manifest_seed_policy(self) -> None:
        manifest = _read_yaml(MANIFEST_PATH)
        seed_policy = manifest["seed_policy"]
        assert seed_policy["mode"] == "seed-set"
        assert seed_policy["seed_set"] == "eval"
        assert set(seed_policy["resolved_seeds"]) == {111, 112, 113}
        assert seed_policy["repeat_count_per_planner_seed"] == 47

    def test_manifest_planners(self) -> None:
        manifest = _read_yaml(MANIFEST_PATH)
        planners = set(manifest["planners"])
        assert planners == {
            "goal",
            "orca",
            "ppo",
            "prediction_planner",
            "sacadrl",
            "social_force",
            "socnav_sampling",
        }

    def test_manifest_reproducibility_contract(self) -> None:
        manifest = _read_yaml(MANIFEST_PATH)
        contract = manifest["reproducibility_contract"]
        exact = set(contract["exact_match_planners"])
        assert "goal" in exact
        assert "ppo" in exact
        borderline = {p["name"] for p in contract["borderline_planners"]}
        assert "orca" in borderline
        assert "prediction_planner" in borderline

    def test_manifest_references_existing_docs(self) -> None:
        manifest = _read_yaml(MANIFEST_PATH)
        for doc_ref in manifest.get("reference_documentation", []):
            doc_path = ROOT / doc_ref
            assert doc_path.is_file(), f"Referenced doc not found: {doc_ref}"

    def test_manifest_checksums_match_release_metadata(self) -> None:
        manifest = _read_yaml(MANIFEST_PATH)
        metadata = _read_json(RELEASE_METADATA)

        assert (
            manifest["artifact_set"]["bundle_archive"]["sha256"]
            == metadata["assets"]["bundle_archive"]["sha256"]
        )

        key_map = {
            "publication_manifest": "embedded_manifest",
            "checksums": "embedded_checksums",
        }
        for manifest_key, metadata_key in key_map.items():
            manifest_sha = manifest["embedded_artifacts"][manifest_key]["sha256"]
            metadata_sha = metadata["assets"][metadata_key]["sha256"]
            assert manifest_sha == metadata_sha, (
                f"SHA mismatch for {manifest_key}: manifest={manifest_sha} vs metadata={metadata_sha}"
            )


class TestVerificationScript:
    """Tests for the verification script logic."""

    def test_script_exists_and_is_executable(self) -> None:
        assert VERIFY_SCRIPT.is_file()
        assert VERIFY_SCRIPT.stat().st_mode & 0o111

    def test_script_imports(self) -> None:
        import importlib.util

        spec = importlib.util.spec_from_file_location("verify_release_checksums", VERIFY_SCRIPT)
        assert spec is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        assert hasattr(module, "verify_release")
        assert hasattr(module, "main")

    def test_bundle_checksum_verification_pass(self, tmp_path: Path) -> None:
        from scripts.repro.verify_release_checksums import _verify_bundle_checksum

        bundle_content = b"test bundle content"
        expected_sha = hashlib.sha256(bundle_content).hexdigest()
        bundle_path = tmp_path / "test_bundle.tar.gz"
        bundle_path.write_bytes(bundle_content)

        result = _verify_bundle_checksum(bundle_path, expected_sha)
        assert result["match"] is True
        assert result["actual_sha256"] == expected_sha

    def test_bundle_checksum_verification_fail(self, tmp_path: Path) -> None:
        from scripts.repro.verify_release_checksums import _verify_bundle_checksum

        bundle_content = b"test bundle content"
        bundle_path = tmp_path / "test_bundle.tar.gz"
        bundle_path.write_bytes(bundle_content)

        result = _verify_bundle_checksum(bundle_path, "0" * 64)
        assert result["match"] is False

    def test_embedded_artifact_verification(self, tmp_path: Path) -> None:
        from scripts.repro.verify_release_checksums import _verify_embedded_artifacts

        inner_content = b"inner file content"
        inner_sha = hashlib.sha256(inner_content).hexdigest()

        tar_path = tmp_path / "test.tar.gz"
        with tarfile.open(tar_path, "w:gz") as tar:
            info = tarfile.TarInfo(name="test/file.txt")
            info.size = len(inner_content)
            tar.addfile(info, BytesIO(inner_content))

        embedded = {
            "test_file": {
                "path_in_archive": "test/file.txt",
                "sha256": inner_sha,
            },
            "missing_file": {
                "path_in_archive": "nonexistent/file.txt",
                "sha256": "0" * 64,
            },
        }

        results = _verify_embedded_artifacts(tar_path, embedded)
        assert len(results) == 2

        test_result = next(r for r in results if r["name"] == "test_file")
        assert test_result["match"] is True
        assert test_result["found"] is True

        missing_result = next(r for r in results if r["name"] == "missing_file")
        assert missing_result["match"] is False
        assert missing_result["found"] is False

    def test_full_verification_with_mock_bundle(self, tmp_path: Path) -> None:
        from scripts.repro.verify_release_checksums import verify_release

        inner_content = b"inner content"
        inner_sha = hashlib.sha256(inner_content).hexdigest()

        tar_path = tmp_path / "bundle.tar.gz"
        with tarfile.open(tar_path, "w:gz") as tar:
            info = tarfile.TarInfo(name="inner/manifest.json")
            info.size = len(inner_content)
            tar.addfile(info, BytesIO(inner_content))

        bundle_sha = hashlib.sha256(tar_path.read_bytes()).hexdigest()

        manifest = {
            "release_tag": "test",
            "release_id": "test_release",
            "artifact_set": {
                "bundle_archive": {
                    "name": "bundle.tar.gz",
                    "url": "https://example.com/bundle.tar.gz",
                    "sha256": bundle_sha,
                    "size_bytes": tar_path.stat().st_size,
                },
            },
            "embedded_artifacts": {
                "manifest": {
                    "path_in_archive": "inner/manifest.json",
                    "sha256": inner_sha,
                },
            },
        }

        manifest_path = tmp_path / "manifest.yaml"
        manifest_path.write_text(yaml.dump(manifest))

        report = verify_release(
            manifest_path=manifest_path,
            bundle_path=tar_path,
            output_dir=tmp_path / "output",
            download=False,
        )

        assert report["overall_verdict"] == "pass"
        assert report["verdicts"]["bundle_checksum"]["match"] is True
        assert len(report["verdicts"]["embedded_artifacts"]) == 1
        assert report["verdicts"]["embedded_artifacts"][0]["match"] is True

    def test_full_verification_bundle_mismatch(self, tmp_path: Path) -> None:
        from scripts.repro.verify_release_checksums import verify_release

        tar_path = tmp_path / "bundle.tar.gz"
        with tarfile.open(tar_path, "w:gz"):
            pass

        manifest = {
            "release_tag": "test",
            "release_id": "test_release",
            "artifact_set": {
                "bundle_archive": {
                    "name": "bundle.tar.gz",
                    "url": "https://example.com/bundle.tar.gz",
                    "sha256": "0" * 64,
                },
            },
        }

        manifest_path = tmp_path / "manifest.yaml"
        manifest_path.write_text(yaml.dump(manifest))

        report = verify_release(
            manifest_path=manifest_path,
            bundle_path=tar_path,
            output_dir=tmp_path / "output",
            download=False,
        )

        assert report["overall_verdict"] == "fail"
        assert "Bundle checksum mismatch" in report["errors"][0]

    def test_verification_fails_closed_for_non_mapping_manifest(self, tmp_path: Path) -> None:
        from scripts.repro.verify_release_checksums import verify_release

        manifest_path = tmp_path / "manifest.yaml"
        manifest_path.write_text("- not\\n- a mapping\\n")

        report = verify_release(
            manifest_path=manifest_path,
            bundle_path=None,
            output_dir=tmp_path / "output",
            download=False,
        )

        assert report["overall_verdict"] == "error"
        assert report["errors"] == ["Checksum manifest root must be a mapping."]

    def test_download_uses_manifest_release_tag(self, tmp_path: Path) -> None:
        from scripts.repro.verify_release_checksums import _download_bundle

        with patch("scripts.repro.verify_release_checksums.subprocess.check_call") as check_call:
            _download_bundle("https://example.com/bundle.tar.gz", tmp_path, "1.2.3")

        assert check_call.call_args.args[0][3] == "1.2.3"


class TestReproductionReport:
    """Tests for the cold-start reproduction report generator."""

    def test_script_exists(self) -> None:
        assert REPORT_SCRIPT.is_file()
        assert REPORT_SCRIPT.stat().st_mode & 0o111

    def test_script_imports(self) -> None:
        import importlib.util

        spec = importlib.util.spec_from_file_location(
            "cold_start_reproduction_report",
            REPORT_SCRIPT,
        )
        assert spec is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        assert hasattr(module, "generate_reproduction_report")

    def test_report_schema_version(self, tmp_path: Path) -> None:
        from scripts.repro.cold_start_reproduction_report import generate_reproduction_report

        mock_manifest = {
            "release_tag": "0.0.2",
            "release_id": "test",
            "artifact_set": {
                "bundle_archive": {
                    "name": "test.tar.gz",
                    "url": "https://example.com/test.tar.gz",
                    "sha256": "0" * 64,
                },
            },
            "embedded_artifacts": {},
        }
        with (
            patch("scripts.repro.cold_start_reproduction_report._load_manifest") as mock_load,
            patch(
                "scripts.repro.cold_start_reproduction_report._step_verify_checksums"
            ) as mock_verify,
        ):
            mock_load.return_value = mock_manifest
            mock_verify.return_value = {
                "step": "verify_checksums",
                "status": "skip",
                "reason": "mocked",
            }
            report = generate_reproduction_report(
                tag="0.0.2",
                output_dir=tmp_path,
                local_repo=tmp_path / "repo",
                checksums_only=True,
            )
            assert report["schema"] == "cold-start-reproduction-report.v1"
            assert report["release_tag"] == "0.0.2"

    def test_report_records_environment(self, tmp_path: Path) -> None:
        from scripts.repro.cold_start_reproduction_report import generate_reproduction_report

        mock_manifest = {
            "release_tag": "0.0.2",
            "release_id": "test",
            "artifact_set": {
                "bundle_archive": {
                    "name": "test.tar.gz",
                    "url": "https://example.com/test.tar.gz",
                    "sha256": "0" * 64,
                },
            },
            "embedded_artifacts": {},
        }
        with (
            patch("scripts.repro.cold_start_reproduction_report._load_manifest") as mock_load,
            patch(
                "scripts.repro.cold_start_reproduction_report._step_verify_checksums"
            ) as mock_verify,
        ):
            mock_load.return_value = mock_manifest
            mock_verify.return_value = {
                "step": "verify_checksums",
                "status": "skip",
                "reason": "mocked",
            }
            report = generate_reproduction_report(
                tag="0.0.2",
                output_dir=tmp_path,
                local_repo=tmp_path / "repo",
                checksums_only=True,
            )
            env = report["environment"]
            assert "platform" in env
            assert "python_version" in env
            assert "architecture" in env

    def test_load_manifest_rejects_non_mapping_root(self, tmp_path: Path, monkeypatch: Any) -> None:
        from scripts.repro.cold_start_reproduction_report import _load_manifest

        manifest_path = tmp_path / "configs" / "releases" / "release_1_2_3_checksum_manifest.yaml"
        manifest_path.parent.mkdir(parents=True)
        manifest_path.write_text("- not\\n- a mapping\\n")
        monkeypatch.chdir(tmp_path)

        with pytest.raises(ValueError, match="root must be a mapping"):
            _load_manifest("1.2.3")

    def test_subset_config_path_uses_manifest_release_tag(self, tmp_path: Path) -> None:
        from scripts.repro.cold_start_reproduction_report import _step_run_subset

        result = _step_run_subset(tmp_path, {"release_tag": "1.2.3"})

        assert result["status"] == "skip"
        assert "v1_2_3_scoped.yaml" in result["reason"]


@pytest.fixture(scope="class")
def actual_execution_results(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    """Run each real release-verification flow once for the execution-test class."""
    from scripts.repro.verify_release_checksums import verify_release

    execution_dir = tmp_path_factory.mktemp("actual_execution")
    verify_report = verify_release(
        manifest_path=MANIFEST_PATH,
        bundle_path=None,
        output_dir=execution_dir / "verification",
        download=True,
    )

    from scripts.repro.cold_start_reproduction_report import generate_reproduction_report

    reproduction_dir = execution_dir / "reproduction"
    reproduction_report = generate_reproduction_report(
        tag="0.0.2",
        output_dir=reproduction_dir,
        local_repo=ROOT,
        checksums_only=True,
    )

    return {
        "verification_report": verify_report,
        "reproduction_report": reproduction_report,
        "reproduction_dir": reproduction_dir,
    }


@pytest.mark.skipif(
    not shutil.which("gh"),
    reason="GitHub CLI 'gh' is required for actual execution tests",
)
class TestActualExecution:
    """Tests that validate actual execution of verification and report generation.

    These tests run the real scripts and verify the output artifacts exist
    with correct structure. They require network access to download the bundle.
    """

    def test_checksum_verification_report_structure(
        self, actual_execution_results: dict[str, Any]
    ) -> None:
        report = actual_execution_results["verification_report"]

        assert report["schema"] == "release-checksum-verification.v1"
        assert report["release_tag"] == "0.0.2"
        assert report["overall_verdict"] == "pass"
        assert "environment" in report
        assert "verdicts" in report
        assert "bundle_checksum" in report["verdicts"]
        assert report["verdicts"]["bundle_checksum"]["match"] is True
        assert len(report["errors"]) == 0

    def test_checksum_verification_report_json_serializable(
        self, actual_execution_results: dict[str, Any]
    ) -> None:
        report = actual_execution_results["verification_report"]

        json_str = json.dumps(report, indent=2, sort_keys=True)
        assert len(json_str) > 100
        parsed = json.loads(json_str)
        assert parsed["overall_verdict"] == "pass"

    def test_cold_start_report_generates_valid_report(
        self, actual_execution_results: dict[str, Any]
    ) -> None:
        report = actual_execution_results["reproduction_report"]

        assert report["schema"] == "cold-start-reproduction-report.v1"
        assert report["release_tag"] == "0.0.2"
        assert report["overall_verdict"] == "partial"
        assert report["steps"]["clone"]["status"] == "skip"
        assert report["instruction_gaps"]
        assert report["deviations"]
        assert "environment" in report
        assert "steps" in report
        assert "verify_checksums" in report["steps"]
        assert report["steps"]["verify_checksums"]["status"] == "pass"

    def test_cold_start_report_embedded_artifacts_verified(
        self, actual_execution_results: dict[str, Any]
    ) -> None:
        report = actual_execution_results["reproduction_report"]

        embedded = report["steps"]["verify_checksums"]["embedded_artifacts"]
        assert len(embedded) >= 3
        for artifact in embedded:
            assert artifact["match"] is True
            assert artifact["actual_sha256"] == artifact["expected_sha256"]

    def test_report_file_written_to_disk(self, actual_execution_results: dict[str, Any]) -> None:
        report = actual_execution_results["reproduction_report"]
        output_dir = actual_execution_results["reproduction_dir"]

        report_path = output_dir / "reproduction_report.json"
        report_path.write_text(json.dumps(report, indent=2, sort_keys=True))
        assert report_path.is_file()
        loaded = json.loads(report_path.read_text())
        assert loaded["overall_verdict"] == "partial"


class TestDocumentation:
    """Tests for documentation completeness."""

    def test_release_reproducibility_doc_references_manifest(self) -> None:
        doc = _read_text_file(RELEASE_REPRO_DOC)
        assert "checksum" in doc.lower() or "sha256" in doc.lower()

    def test_release_002_reproduction_doc_exists(self) -> None:
        assert REPRO_002_DOC.is_file()

    def test_release_002_reproduction_doc_has_sha256(self) -> None:
        doc = _read_text_file(REPRO_002_DOC)
        assert "64e8510ab7ba934103c709907f66a783c7b3dd2dd58aa4bd725e762da2734d90" in doc

    def test_release_metadata_file_exists(self) -> None:
        assert RELEASE_METADATA.is_file()

    def test_release_metadata_has_checksums(self) -> None:
        metadata = _read_json(RELEASE_METADATA)
        assert "assets" in metadata
        assert "bundle_archive" in metadata["assets"]
        assert "sha256" in metadata["assets"]["bundle_archive"]

    def test_benchmark_bundle_smoke_script_exists(self) -> None:
        assert BUNDLE_SMOKE_SCRIPT.is_file()

    def test_benchmark_bundle_smoke_script_is_executable(self) -> None:
        assert BUNDLE_SMOKE_SCRIPT.stat().st_mode & 0o111


class TestDurableEvidenceReport:
    """Tests for the durable cold-start reproduction evidence report.

    This report is the actual output of running the verification and
    cold-start reproduction scripts, promoted as trackable evidence
    per issue #5366 criterion 3 (durable reproduction report).
    """

    def test_evidence_report_exists(self) -> None:
        assert DURABLE_EVIDENCE_REPORT.is_file(), (
            f"Durable evidence report not found: {DURABLE_EVIDENCE_REPORT}"
        )

    def test_evidence_report_has_schema(self) -> None:
        report = _read_json(DURABLE_EVIDENCE_REPORT)
        assert report["schema"] == "cold-start-reproduction-report.v1"

    def test_evidence_report_pins_the_checksum_manifest(self) -> None:
        report = _read_json(DURABLE_EVIDENCE_REPORT)
        manifest_path = ROOT / report["config_path"]
        assert manifest_path.is_file()
        assert report["config_sha256"] == hashlib.sha256(manifest_path.read_bytes()).hexdigest()
        assert len(report["config_commit"]) == 40

    def test_evidence_report_is_partial_when_the_clone_was_skipped(self) -> None:
        report = _read_json(DURABLE_EVIDENCE_REPORT)
        assert report["overall_verdict"] == "partial"
        assert report["steps"]["clone"]["status"] == "skip"
        assert any("clean release-tag clone was skipped" in item for item in report["deviations"])
        assert any(
            "clean non-development machine/person" in item for item in report["instruction_gaps"]
        )

    def test_evidence_report_has_environment(self) -> None:
        report = _read_json(DURABLE_EVIDENCE_REPORT)
        env = report["environment"]
        assert "platform" in env
        assert "python_version" in env
        assert "architecture" in env

    def test_evidence_report_verify_checksums_pass(self) -> None:
        report = _read_json(DURABLE_EVIDENCE_REPORT)
        step = report["steps"]["verify_checksums"]
        assert step["status"] == "pass"
        assert step["bundle_checksum_match"] is True

    def test_evidence_report_all_embedded_artifacts_match(self) -> None:
        report = _read_json(DURABLE_EVIDENCE_REPORT)
        artifacts = report["steps"]["verify_checksums"]["embedded_artifacts"]
        assert len(artifacts) >= 3
        for art in artifacts:
            assert art["match"] is True
            assert art["actual_sha256"] == art["expected_sha256"]

    def test_evidence_report_build_pass(self) -> None:
        report = _read_json(DURABLE_EVIDENCE_REPORT)
        step = report["steps"]["build"]
        assert step["status"] == "pass"

    def test_evidence_report_preflight_pass(self) -> None:
        report = _read_json(DURABLE_EVIDENCE_REPORT)
        step = report["steps"]["run_subset"]
        assert step["status"] == "pass"
        assert step["preflight_status"] == "pass"
