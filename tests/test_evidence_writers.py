"""Tests for the shared evidence writers module."""

# evidence-writer-exempt: fixtures write into throwaway tmp_path directories that
# merely mention "docs/context/evidence" in their string paths to exercise
# register_evidence()/write_*(catalog_area=...); no real evidence tree is touched.

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
import yaml

from robot_sf.evidence.writers import (
    REVIEW_SIDECAR_SCHEMA_VERSION,
    extract_marker_date,
    register_evidence,
    review_marker,
    review_marker_comment,
    review_marker_json,
    sha256_file,
    write_csv,
    write_json,
    write_review_sidecar,
    write_sha256sums,
    write_text,
)

if TYPE_CHECKING:
    from pathlib import Path

    from pytest import MonkeyPatch


_MINIMAL_CATALOG = (
    "version: 1\n"
    "status_values:\n"
    "  evidence: Evidence pointer or manifest.\n"
    "freshness_values:\n"
    "  evidence: Evidence pointer.\n"
    "entries:\n"
    "- path: docs/context/issue_0000_placeholder.md\n"
    "  status: current\n"
    "  freshness: maintained\n"
    "  area: workflow_evidence\n"
)


def _init_fake_repo(tmp_path: Path) -> Path:
    """Write a minimal ``docs/context/catalog.yaml`` under ``tmp_path``."""
    catalog = tmp_path / "docs" / "context" / "catalog.yaml"
    catalog.parent.mkdir(parents=True, exist_ok=True)
    catalog.write_text(_MINIMAL_CATALOG, encoding="utf-8")
    return catalog


class TestReviewMarker:
    """Test the review marker helpers."""

    def test_review_marker_html_comment(self) -> None:
        marker = review_marker("robot_sf#4891")
        assert marker == "<!-- AI-GENERATED (robot_sf#4891) - NEEDS-REVIEW -->"

    def test_review_marker_with_date(self) -> None:
        marker = review_marker("robot_sf#4891", marker_date="2026-07-09")
        assert marker == "<!-- AI-GENERATED (robot_sf#4891, 2026-07-09) - NEEDS-REVIEW -->"

    def test_review_marker_none_date_omits_date(self) -> None:
        marker = review_marker("robot_sf#4891", marker_date=None)
        assert marker == "<!-- AI-GENERATED (robot_sf#4891) - NEEDS-REVIEW -->"

    def test_review_marker_json(self) -> None:
        assert review_marker_json() == "AI-GENERATED NEEDS-REVIEW"

    def test_review_marker_comment(self) -> None:
        assert review_marker_comment() == "# AI-GENERATED NEEDS-REVIEW"


class TestExtractMarkerDate:
    """Test the shared extract_marker_date provenance helper."""

    def test_extracts_date_from_iso_timestamp(self) -> None:
        assert (
            extract_marker_date({"generated_at_utc": "2026-07-08T12:34:56+00:00"}) == "2026-07-08"
        )

    def test_missing_generated_at_returns_none(self) -> None:
        assert extract_marker_date({}) is None

    def test_empty_generated_at_returns_none(self) -> None:
        assert extract_marker_date({"generated_at_utc": ""}) is None


class TestSha256File:
    """Test the sha256_file helper."""

    def test_known_content(self, tmp_path: Path) -> None:
        test_file = tmp_path / "test.txt"
        test_file.write_text("hello\n", encoding="utf-8")
        digest = sha256_file(test_file)
        assert len(digest) == 64
        assert digest == "5891b5b522d5df086d0ff0b110fbd9d21bb4fc7163af34d08286a2e846f6be03"


class TestWriteJson:
    """Test the write_json helper."""

    def test_adds_review_marker(self, tmp_path: Path) -> None:
        path = tmp_path / "test.json"
        payload = {"key": "value"}
        write_json(path, payload)
        content = path.read_text(encoding="utf-8")
        assert "AI-GENERATED NEEDS-REVIEW" in content
        assert '"review_marker"' in content

    def test_preserves_original_payload(self, tmp_path: Path) -> None:
        path = tmp_path / "test.json"
        payload = {"key": "value", "nested": {"a": 1}}
        write_json(path, payload)
        import json

        data = json.loads(path.read_text(encoding="utf-8"))
        assert data["key"] == "value"
        assert data["nested"]["a"] == 1
        assert data["review_marker"] == "AI-GENERATED NEEDS-REVIEW"


class TestWriteCsv:
    """Test the write_csv helper."""

    def test_prepends_marker(self, tmp_path: Path) -> None:
        path = tmp_path / "test.csv"
        rows = [{"col1": "val1", "col2": "val2"}]
        write_csv(path, rows)
        content = path.read_text(encoding="utf-8")
        assert content.startswith("# AI-GENERATED NEEDS-REVIEW\n")

    def test_writes_header_and_rows(self, tmp_path: Path) -> None:
        path = tmp_path / "test.csv"
        rows = [
            {"a": 1, "b": 2},
            {"a": 3, "b": 4},
        ]
        write_csv(path, rows)
        lines = path.read_text(encoding="utf-8").strip().split("\n")
        assert lines[0] == "# AI-GENERATED NEEDS-REVIEW"
        assert "a,b" in lines[1]
        assert "1,2" in lines[2]


class TestWriteSha256sums:
    """Test the write_sha256sums helper."""

    def test_prepends_marker(self, tmp_path: Path) -> None:
        test_file = tmp_path / "test.txt"
        test_file.write_text("data", encoding="utf-8")
        write_sha256sums(tmp_path)
        sha_file = tmp_path / "SHA256SUMS"
        content = sha_file.read_text(encoding="utf-8")
        assert content.startswith("# AI-GENERATED NEEDS-REVIEW\n")

    def test_excludes_self_from_hash(self, tmp_path: Path) -> None:
        test_file = tmp_path / "test.txt"
        test_file.write_text("data", encoding="utf-8")
        write_sha256sums(tmp_path)
        sha_file = tmp_path / "SHA256SUMS"
        content = sha_file.read_text(encoding="utf-8")
        assert "SHA256SUMS" not in content.split("\n", 1)[1]


class TestWriteReviewSidecar:
    """Test the write_review_sidecar helper (issue #5911)."""

    def test_writes_marker_sidecar_with_schema_and_digest(self, tmp_path: Path) -> None:
        artifact = tmp_path / "archive.json"
        artifact.write_text('{"a": 1}\n', encoding="utf-8")
        sidecar = write_review_sidecar(artifact, repo_root=tmp_path)

        import json

        assert sidecar == artifact.with_name("archive.json.review.json")
        payload = json.loads(sidecar.read_text(encoding="utf-8"))
        assert payload["schema_version"] == REVIEW_SIDECAR_SCHEMA_VERSION
        assert payload["artifact_path"] == "archive.json"
        assert payload["artifact_sha256"] == sha256_file(artifact)
        assert payload["review_marker"] == review_marker_json()
        assert payload["preserved_exact_bytes"] is True

    def test_artifact_bytes_are_not_mutated(self, tmp_path: Path) -> None:
        """The sidecar must not change the artifact bytes or its digest."""
        artifact = tmp_path / "archive.json"
        original_bytes = b'{"schema_version": "x.v1", "entries": []}\n'
        artifact.write_bytes(original_bytes)
        expected_digest = sha256_file(artifact)

        write_review_sidecar(artifact, repo_root=tmp_path)

        assert artifact.read_bytes() == original_bytes
        assert sha256_file(artifact) == expected_digest

    def test_registers_evidence_bundle_for_sidecar_only_artifacts(self, tmp_path: Path) -> None:
        """A byte-preserving artifact and its sidecar share one catalog bundle row."""
        from scripts.dev.check_docs_evidence_integrity import check_files

        catalog = _init_fake_repo(tmp_path)
        evidence_dir = tmp_path / "docs/context/evidence/issue_5911_archive"
        evidence_dir.mkdir(parents=True)
        artifact = evidence_dir / "archive.json"
        artifact.write_text('{"a": 1}\n', encoding="utf-8")

        sidecar = write_review_sidecar(artifact, repo_root=tmp_path)

        payload = yaml.safe_load(catalog.read_text(encoding="utf-8"))
        registered_paths = {entry["path"] for entry in payload["entries"]}
        assert "docs/context/evidence/issue_5911_archive" in registered_paths
        paths = [
            "docs/context/evidence/issue_5911_archive/archive.json",
            sidecar.relative_to(tmp_path).as_posix(),
        ]
        assert check_files(paths, root=tmp_path) == []


class TestWriteText:
    """Test marker enforcement for Markdown/text evidence."""

    def test_prepends_issue_marker(self, tmp_path: Path) -> None:
        path = tmp_path / "README.md"
        write_text(path, "# Report\n", issue_ref="robot_sf#4921", marker_date="2026-07-14")
        assert path.read_text(encoding="utf-8") == (
            "<!-- AI-GENERATED (robot_sf#4921, 2026-07-14) - NEEDS-REVIEW -->\n# Report\n"
        )

    def test_rejects_unmarked_text_without_issue(self, tmp_path: Path) -> None:
        path = tmp_path / "README.md"
        try:
            write_text(path, "# Report\n")
        except ValueError as exc:
            assert "AI-GENERATED" in str(exc)
        else:
            raise AssertionError("unmarked evidence text should be rejected")

    def test_rejects_empty_unmarked_text_with_value_error(self, tmp_path: Path) -> None:
        path = tmp_path / "README.md"
        try:
            write_text(path, "")
        except ValueError as exc:
            assert "AI-GENERATED" in str(exc)
        else:
            raise AssertionError("empty unmarked evidence text should be rejected")

    def test_preserves_existing_pinned_marker(self, tmp_path: Path) -> None:
        path = tmp_path / "README.md"
        content = "<!-- AI-GENERATED (robot_sf#4921, 2026-07-14) - NEEDS-REVIEW -->\n# Report\n"
        write_text(path, content)
        assert path.read_text(encoding="utf-8") == content


class TestRegisterEvidence:
    """Test the docs/context/catalog.yaml registration helper (issue #6116)."""

    def test_appends_additive_entry(self, tmp_path: Path) -> None:
        catalog = _init_fake_repo(tmp_path)
        original_text = catalog.read_text(encoding="utf-8")
        evidence = tmp_path / "docs/context/evidence/issue_6116_probe/README.md"
        evidence.parent.mkdir(parents=True, exist_ok=True)
        evidence.write_text("<!-- AI-GENERATED (robot_sf#6116) - NEEDS-REVIEW -->\n", "utf-8")

        appended = register_evidence(evidence, area="benchmark_evidence", repo_root=tmp_path)

        assert appended is True
        new_text = catalog.read_text(encoding="utf-8")
        # Additive: existing bytes are an unmodified prefix, only a new row is appended.
        assert new_text.startswith(original_text)
        payload = yaml.safe_load(new_text)
        registered_paths = {entry["path"] for entry in payload["entries"]}
        assert "docs/context/evidence/issue_6116_probe/README.md" in registered_paths
        new_entry = next(
            e
            for e in payload["entries"]
            if e["path"] == "docs/context/evidence/issue_6116_probe/README.md"
        )
        assert new_entry == {
            "path": "docs/context/evidence/issue_6116_probe/README.md",
            "status": "evidence",
            "freshness": "evidence",
            "area": "benchmark_evidence",
        }

    def test_idempotent_second_call_is_noop(self, tmp_path: Path) -> None:
        _init_fake_repo(tmp_path)
        evidence = tmp_path / "docs/context/evidence/issue_6116_probe/README.md"
        evidence.parent.mkdir(parents=True, exist_ok=True)
        evidence.write_text("<!-- AI-GENERATED (robot_sf#6116) - NEEDS-REVIEW -->\n", "utf-8")

        first = register_evidence(evidence, area="benchmark_evidence", repo_root=tmp_path)
        second = register_evidence(evidence, area="benchmark_evidence", repo_root=tmp_path)

        assert first is True
        assert second is False

    def test_covered_by_registered_ancestor_directory_is_noop(self, tmp_path: Path) -> None:
        catalog = _init_fake_repo(tmp_path)
        bundle_dir = tmp_path / "docs/context/evidence/issue_6116_probe"
        bundle_dir.mkdir(parents=True, exist_ok=True)
        (bundle_dir / "README.md").write_text("x", "utf-8")
        with catalog.open("a", encoding="utf-8") as handle:
            handle.write(
                "- path: docs/context/evidence/issue_6116_probe\n"
                "  status: evidence\n"
                "  freshness: evidence\n"
                "  area: benchmark_evidence\n"
            )

        appended = register_evidence(
            bundle_dir / "README.md", area="benchmark_evidence", repo_root=tmp_path
        )

        assert appended is False

    def test_rejects_path_outside_evidence_dir(self, tmp_path: Path) -> None:
        _init_fake_repo(tmp_path)
        outside = tmp_path / "docs" / "context" / "issue_6116_note.md"
        outside.parent.mkdir(parents=True, exist_ok=True)
        outside.write_text("x", "utf-8")

        with pytest.raises(ValueError, match="outside"):
            register_evidence(outside, area="benchmark_evidence", repo_root=tmp_path)

    def test_rejects_noncanonical_status(self, tmp_path: Path) -> None:
        _init_fake_repo(tmp_path)
        evidence = tmp_path / "docs/context/evidence/issue_6116_probe/README.md"
        evidence.parent.mkdir(parents=True, exist_ok=True)
        evidence.write_text("x", "utf-8")

        with pytest.raises(ValueError, match="status"):
            register_evidence(
                evidence, area="benchmark_evidence", status="bogus", repo_root=tmp_path
            )

    def test_missing_catalog_raises(self, tmp_path: Path) -> None:
        evidence = tmp_path / "docs/context/evidence/issue_6116_probe/README.md"
        evidence.parent.mkdir(parents=True, exist_ok=True)
        evidence.write_text("x", "utf-8")

        with pytest.raises(FileNotFoundError):
            register_evidence(evidence, area="benchmark_evidence", repo_root=tmp_path)


class TestWriterCatalogRegistration:
    """Test shared writers register evidence-tree output (issue #6116)."""

    def test_write_json_registers_when_catalog_area_set(
        self, tmp_path: Path, monkeypatch: MonkeyPatch
    ) -> None:
        monkeypatch.setattr("robot_sf.evidence.writers._repo_root", lambda: tmp_path)
        catalog = _init_fake_repo(tmp_path)
        path = tmp_path / "docs/context/evidence/issue_6116_probe/summary.json"
        path.parent.mkdir(parents=True, exist_ok=True)

        write_json(path, {"ok": True}, catalog_area="benchmark_evidence")

        payload = yaml.safe_load(catalog.read_text(encoding="utf-8"))
        registered_paths = {entry["path"] for entry in payload["entries"]}
        assert "docs/context/evidence/issue_6116_probe/summary.json" in registered_paths

    def test_write_json_registers_evidence_path_by_default(
        self, tmp_path: Path, monkeypatch: MonkeyPatch
    ) -> None:
        monkeypatch.setattr("robot_sf.evidence.writers._repo_root", lambda: tmp_path)
        catalog = _init_fake_repo(tmp_path)
        path = tmp_path / "docs/context/evidence/issue_6116_probe/summary.json"
        path.parent.mkdir(parents=True, exist_ok=True)

        write_json(path, {"ok": True})

        payload = yaml.safe_load(catalog.read_text(encoding="utf-8"))
        registered_paths = {entry["path"] for entry in payload["entries"]}
        assert "docs/context/evidence/issue_6116_probe" in registered_paths

    def test_write_json_omits_registration_outside_evidence_tree(
        self, tmp_path: Path, monkeypatch: MonkeyPatch
    ) -> None:
        monkeypatch.setattr("robot_sf.evidence.writers._repo_root", lambda: tmp_path)
        catalog = _init_fake_repo(tmp_path)
        original_text = catalog.read_text(encoding="utf-8")
        output_dir = tmp_path / "output"
        output_dir.mkdir()

        write_json(output_dir / "summary.json", {"ok": True})

        assert catalog.read_text(encoding="utf-8") == original_text

    def test_write_sha256sums_registers_whole_bundle_directory(
        self, tmp_path: Path, monkeypatch: MonkeyPatch
    ) -> None:
        monkeypatch.setattr("robot_sf.evidence.writers._repo_root", lambda: tmp_path)
        catalog = _init_fake_repo(tmp_path)
        bundle_dir = tmp_path / "docs/context/evidence/issue_6116_bundle"
        bundle_dir.mkdir(parents=True, exist_ok=True)
        write_json(bundle_dir / "summary.json", {"ok": True})
        write_text(
            bundle_dir / "README.md",
            "body\n",
            issue_ref="robot_sf#6116",
        )

        write_sha256sums(bundle_dir, catalog_area="benchmark_evidence")

        payload = yaml.safe_load(catalog.read_text(encoding="utf-8"))
        registered_paths = {entry["path"] for entry in payload["entries"]}
        assert "docs/context/evidence/issue_6116_bundle" in registered_paths
        # A single directory-level entry covers every file already in the bundle,
        # including the ones written before the catalog-registering final step.
        summary_rel = "docs/context/evidence/issue_6116_bundle/summary.json"
        assert summary_rel not in registered_paths  # covered by the ancestor dir, not duplicated


class TestRegisterEvidenceIntegrationWithCheck:
    """Prove the registration helper satisfies the real docs-evidence-integrity check."""

    def test_unregistered_evidence_fails_then_passes_after_registration(
        self, tmp_path: Path
    ) -> None:
        from scripts.dev.check_docs_evidence_integrity import check_files

        _init_fake_repo(tmp_path)
        evidence = tmp_path / "docs/context/evidence/issue_6116_probe/README.md"
        evidence.parent.mkdir(parents=True, exist_ok=True)
        evidence.write_text("<!-- AI-GENERATED (robot_sf#6116) - NEEDS-REVIEW -->\nbody\n", "utf-8")
        rel = evidence.relative_to(tmp_path).as_posix()

        before = check_files([rel], root=tmp_path)
        assert any("is not registered in" in problem for problem in before)

        register_evidence(evidence, area="benchmark_evidence", repo_root=tmp_path)

        after = check_files([rel], root=tmp_path)
        assert after == []


class TestProductionWriterCatalogRegistration:
    """Exercise a production caller, not only the shared helper (issue #6116)."""

    def test_acceptance_audit_registers_emitted_evidence_paths(
        self, tmp_path: Path, monkeypatch: MonkeyPatch
    ) -> None:
        """The production audit writer leaves its changed evidence files integrity-clean."""
        from scripts.analysis import audit_issue_4013_acceptance as acceptance_audit
        from scripts.dev.check_docs_evidence_integrity import check_files

        monkeypatch.setattr("robot_sf.evidence.writers._repo_root", lambda: tmp_path)
        production_audit = {
            "schema_version": "test.v1",
            "closure_status": "blocked",
            "claim_boundary": "test-only production writer probe",
            "next_empirical_action": "none",
            "criteria": [
                {
                    "criterion": "test criterion",
                    "status": "met",
                    "evidence": ["test evidence"],
                    "remaining_work": None,
                }
            ],
            "merged_pr_evidence": [{"pr": 1, "evidence": "test evidence"}],
            "blockers_remaining": [],
            "intentional_non_actions": ["test non-action"],
        }
        monkeypatch.setattr(
            acceptance_audit,
            "build_acceptance_audit",
            lambda *, evidence_dir: production_audit,
        )
        _init_fake_repo(tmp_path)
        evidence_dir = tmp_path / "docs/context/evidence/issue_4013_production_probe"

        acceptance_audit.write_acceptance_audit(evidence_dir=evidence_dir)

        emitted_paths = [
            "docs/context/evidence/issue_4013_production_probe/acceptance_audit.v1.json",
            "docs/context/evidence/issue_4013_production_probe/acceptance_audit.v1.md",
        ]
        assert check_files(emitted_paths, root=tmp_path) == []
